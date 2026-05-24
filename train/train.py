"""
SFT 训练脚本 —— Qwen3-8B 工具调用蒸馏（LoRA）

运行方式（单卡 GPU）:
    python train/train.py --model_path /path/to/Qwen3-8B --data train/sft_data.jsonl

迁移到 Ascend 910B NPU（使用 accelerate）:
    accelerate launch --config_file train/accelerate_npu.yaml train/train.py \
        --model_path /path/to/Qwen3-8B --data train/sft_data.jsonl

NPU 注意事项:
    1. 需要安装 torch_npu（华为提供），脚本已在 import 处做兼容
    2. 910B 支持 bf16，训练参数已默认 bf16=True
    3. Flash Attention 2 在 NPU 上支持不稳定，默认关闭（attn_implementation="eager"）
    4. 如需 DeepSpeed ZeRO，910B 支持 ZeRO-1/2/3，配置文件自行提供
    5. PEFT LoRA 通过 peft 库实现，与 torch_npu 兼容
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from multiprocessing.pool import ThreadPool

from tqdm import tqdm

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ── NPU 兼容 ──────────────────────────────────────────────────────────────────
# 在 Ascend 910B 上，import torch_npu 会注册 'npu' 设备
# GPU 环境下此 import 不存在，安全跳过
try:
    import torch_npu  # noqa: F401
    _HAS_NPU = torch.npu.is_available()
except (ImportError, AttributeError):
    _HAS_NPU = False

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Label masking: 仅在 assistant 轮次上计算 loss
# ──────────────────────────────────────────────────────────────────────────────
IGNORE_INDEX = -100


def _encode_message(msg: dict, tokenizer) -> tuple[list[int], list[int]]:
    """将单条消息编码为 (input_ids, labels)，assistant 轮次 labels = token ids，其余 = IGNORE_INDEX。"""
    role    = msg["role"]
    content = msg["content"]
    if role == "tool":
        content = f"<tool_response>\n{content}\n</tool_response>"

    prefix_ids  = tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
    content_ids = tokenizer.encode(content,                  add_special_tokens=False)
    suffix_ids  = tokenizer.encode("<|im_end|>\n",           add_special_tokens=False)

    ids = prefix_ids + content_ids + suffix_ids
    if role == "assistant":
        lbls = [IGNORE_INDEX] * len(prefix_ids) + content_ids + suffix_ids
    else:
        lbls = [IGNORE_INDEX] * len(ids)
    return ids, lbls


def build_input_and_labels(
    messages: list[dict],
    tokenizer,
    max_length: int,
    drop_long: bool = False,
) -> tuple[dict | None, bool]:
    """
    将 Qwen3 格式的 messages 逐段 tokenize 并构建 labels。

    返回 (item, is_long)：
      item    —— 训练样本字典，None 表示丢弃
      is_long —— 原始序列是否超过 max_length

    drop_long=True : 超过 max_length 的样本直接丢弃
    drop_long=False: 从尾部截断，优先保留最后的 answer 轮次（默认）
    """
    # 一次性 tokenize 所有消息段
    segments: list[tuple[list[int], list[int]]] = [
        _encode_message(msg, tokenizer) for msg in messages
    ]
    total = sum(len(ids) for ids, _ in segments)
    is_long = total > max_length

    if is_long:
        if drop_long:
            return None, True

        # 从尾部保留：先固定 system / user 轮次作为 head
        head_ids, head_lbls = [], []
        tail_segments = []
        for msg, seg in zip(messages, segments):
            if msg["role"] in ("system", "user"):
                head_ids.extend(seg[0])
                head_lbls.extend(seg[1])
            else:
                tail_segments.append(seg)

        budget = max_length - len(head_ids)
        tail_ids, tail_lbls = [], []
        for ids, lbls in reversed(tail_segments):
            if len(ids) <= budget:
                tail_ids  = ids  + tail_ids
                tail_lbls = lbls + tail_lbls
                budget -= len(ids)
            else:
                break  # 整段塞不下则跳过，保证不截断单轮

        all_input_ids = head_ids + tail_ids
        all_labels    = head_lbls + tail_lbls
    else:
        all_input_ids = [id_ for ids, _ in segments for id_ in ids]
        all_labels    = [lbl  for _, lbls in segments for lbl in lbls]

    # 安全截断（边界保护）
    all_input_ids = all_input_ids[:max_length]
    all_labels    = all_labels[:max_length]

    # 整条样本没有任何可学习 token，丢弃
    if all(lbl == IGNORE_INDEX for lbl in all_labels):
        return None, is_long

    return {
        "input_ids":      all_input_ids,
        "labels":         all_labels,
        "attention_mask": [1] * len(all_input_ids),
    }, is_long


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int, drop_long: bool = False):
        self.samples: list[dict] = []

        logger.info(f"Loading data from {data_path} ...")
        logger.info(f"max_length={max_length}, drop_long={drop_long}")

        raw: list[list[dict]] = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                raw.append(json.loads(line)["messages"])
        logger.info(f"Raw samples: {len(raw)}")

        n_truncated    = 0
        n_dropped_long = 0
        n_no_label     = 0

        num_threads = min(os.cpu_count() or 1, 16)
        logger.info(f"Tokenizing with {num_threads} threads ...")
        process_fn = partial(
            build_input_and_labels,
            tokenizer=tokenizer,
            max_length=max_length,
            drop_long=drop_long,
        )
        with ThreadPool(num_threads) as pool:
            results = list(tqdm(
                pool.imap(process_fn, raw),
                total=len(raw),
                desc="Tokenizing",
                unit="sample",
                dynamic_ncols=True,
            ))

        for (item, is_long) in results:
            if item is None:
                if is_long and drop_long:
                    n_dropped_long += 1
                else:
                    n_no_label += 1
            else:
                if is_long:
                    n_truncated += 1
                self.samples.append(item)

        logger.info(
            f"Dataset stats:\n"
            f"  Total raw       : {len(raw)}\n"
            f"  Truncated (kept): {n_truncated}\n"
            f"  Dropped (long)  : {n_dropped_long}\n"
            f"  Dropped (no lbl): {n_no_label}\n"
            f"  Final samples   : {len(self.samples)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Data Collator（padding）
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SFTDataCollator:
    tokenizer: object
    pad_to_multiple_of: int = 8  # 910B 上对齐 8 有性能收益

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        # 对齐到 pad_to_multiple_of
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id or 0

        input_ids      = []
        labels         = []
        attention_masks = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            labels.append(f["labels"] + [IGNORE_INDEX] * pad_len)
            attention_masks.append(f["attention_mask"] + [0] * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids,       dtype=torch.long),
            "labels":         torch.tensor(labels,          dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Trainer：分块 cross-entropy，避免 logits.float() OOM
# ──────────────────────────────────────────────────────────────────────────────
class SFTTrainer(Trainer):
    """
    覆盖 compute_loss，解决两个 OOM 来源：
    1. accelerate 的 ConvertOutputsToFp32 包装器：会在 model(**inputs) 返回时
       把整个 logits 张量转为 fp32，导致 OOM。通过 unwrap_model + autocast 绕过。
    2. transformers ForCausalLMLoss 内部 logits.float()：通过分块（chunked）
       cross-entropy 只在小 chunk 上升精度，峰值显存从 ~20 GiB 降到 ~几百 MB。
    """

    # 每次转为 fp32 计算的 token 数量；越小越省显存，但略慢
    _LOSS_CHUNK = 512

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")   # [B, T]

        # 模型权重已是 bf16（torch_dtype=bfloat16），不开 accelerate mixed precision。
        # 手动加 autocast 确保 forward 内所有激活保持 bf16，节省约一半激活显存。
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            outputs = model(**inputs)
        logits  = outputs.logits        # [B, T, V]  bf16

        # ── 因果 LM shift ────────────────────────────────────────────────────
        shift_logits = logits[..., :-1, :].contiguous()    # [B, T-1, V]
        shift_labels = labels[..., 1:].contiguous()         # [B, T-1]

        B, T, V     = shift_logits.shape
        flat_logits = shift_logits.view(-1, V)              # [B*(T-1), V]  bf16
        flat_labels = shift_labels.view(-1)                 # [B*(T-1)]

        # ── 分块 cross-entropy：每次只升精度 _LOSS_CHUNK 个 token ───────────
        total_loss = flat_logits.new_zeros((), dtype=torch.float32)
        n_valid    = 0
        for start in range(0, flat_logits.size(0), self._LOSS_CHUNK):
            end         = min(start + self._LOSS_CHUNK, flat_logits.size(0))
            chunk_logit = flat_logits[start:end].float()    # 仅此 chunk 升精度
            chunk_label = flat_labels[start:end]
            n_valid    += (chunk_label != IGNORE_INDEX).sum().item()
            total_loss += torch.nn.functional.cross_entropy(
                chunk_logit, chunk_label,
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )

        loss = total_loss / max(n_valid, 1)
        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ScriptArguments:
    model_path:  str  = field(metadata={"help": "Qwen3-8B 模型路径"})
    data:        str  = field(default="train/sft_data.jsonl", metadata={"help": "训练数据 jsonl 路径"})
    max_length:  int  = field(default=16384, metadata={"help": "最大序列长度（需要 SDPA 或 flash_attn）"})
    drop_long:   bool = field(default=False, metadata={"help": "True=直接丢弃超长序列；False=从尾部截断保留 answer"})
    # ── LoRA 参数 ──────────────────────────────────────────────────────────────
    lora_r:       int   = field(default=64,    metadata={"help": "LoRA rank"})
    lora_alpha:   int   = field(default=128,   metadata={"help": "LoRA alpha（通常为 2*r）"})
    lora_dropout: float = field(default=0.05,  metadata={"help": "LoRA dropout"})
    # Qwen3-8B 的 attention + MLP 投影层，覆盖全部线性层以获得最佳效果
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "逗号分隔的 LoRA 目标模块名"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))

    # 默认训练参数（可通过命令行覆盖）
    default_training_args = [
        "--output_dir",             "train/output",
        "--num_train_epochs",       "3",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--learning_rate",          "1e-5",
        "--lr_scheduler_type",      "cosine",
        "--warmup_ratio",           "0.05",
        "--weight_decay",           "0.01",
        # bf16 由模型权重本身保证（torch_dtype=bfloat16），不走 accelerate mixed precision，
        # 避免 ConvertOutputsToFp32 把整个 logits 张量转 fp32 导致 OOM。
        "--bf16",                   "False",
        "--logging_steps",          "10",
        "--save_strategy",          "steps",
        "--save_steps",             "200",
        "--save_total_limit",       "3",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "2",
        "--report_to",              "none",
        # 关闭 CUDA 专有优化，确保 NPU 兼容
        "--optim",                  "adamw_torch",
    ]

    # 合并命令行参数
    all_args = sys.argv[1:] + default_training_args
    script_args, training_args = parser.parse_args_into_dataclasses(
        args=all_args, look_for_args_file=False
    )

    set_seed(training_args.seed)

    logger.info(f"Model: {script_args.model_path}")
    logger.info(f"Data:  {script_args.data}")
    logger.info(f"NPU available: {_HAS_NPU}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    _is_local = os.path.isdir(script_args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_path,
        trust_remote_code=True,
        padding_side="right",   # SFT 右 padding
        local_files_only=_is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Attention 实现选择 ─────────────────────────────────────────────────────
    # GPU:  优先 flash_attention_2（需要 pip install flash-attn），否则 sdpa
    # NPU:  sdpa（torch_npu ≥2.1 将 SDPA 路由到 npu_fusion_attention，O(n) 内存）
    # 两者内存复杂度均为 O(n)，支持 32K+ 序列长度
    if _HAS_NPU:
        attn_impl = "sdpa"
    else:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
    logger.info(f"attn_implementation: {attn_impl}")

    # ── 模型 ───────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,   # 910B 原生支持 bf16
        attn_implementation=attn_impl,
        local_files_only=_is_local,
    )
    model.enable_input_require_grads()  # LoRA + gradient_checkpointing 必须

    # ── LoRA ───────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=script_args.lora_target_modules.split(","),
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ── Dataset & Collator ────────────────────────────────────────────────────
    train_dataset = SFTDataset(
        data_path=script_args.data,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        drop_long=script_args.drop_long,
    )
    collator = SFTDataCollator(tokenizer=tokenizer)

    # ── Trainer ────────────────────────────────────────────────────────────────
    # HuggingFace Trainer 通过 accelerate 管理设备，torch_npu 注册 'npu'
    # 后 Trainer 会自动使用 NPU；无需修改任何训练逻辑。
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting training ...")
    trainer.train()

    logger.info(f"Saving LoRA adapter to {training_args.output_dir}")
    # 只保存 LoRA adapter 权重（小很多），完整模型在推理时 merge 或直接 load_peft_model
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Done. To merge LoRA weights:")
    logger.info("  from peft import PeftModel")
    logger.info(f"  model = PeftModel.from_pretrained(base_model, '{training_args.output_dir}')")
    logger.info("  model = model.merge_and_unload()")


if __name__ == "__main__":
    main()
