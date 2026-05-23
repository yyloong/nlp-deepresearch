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
import sys
from dataclasses import dataclass, field

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


def build_input_and_labels(
    messages: list[dict],
    tokenizer,
    max_length: int,
) -> dict | None:
    """
    将 Qwen3 格式的 messages 逐段 tokenize 并构建 labels。

    assistant 轮次（content + <|im_end|>）计算 loss，
    其余轮次（system / user / tool）的 label 设为 IGNORE_INDEX。

    手动分段 tokenize 而非对整体字符串做 offset mapping，
    是为了在 NPU 上避免 offset_mapping 可能带来的兼容性问题。

    注意: BPE 在段落边界处可能与整体编码略有不同，
    但实测影响极小，是 SFT 训练的通行做法。
    """
    all_input_ids: list[int] = []
    all_labels: list[int] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # tool role 内容需要加 <tool_response> 包裹（Qwen3 推理时 template 会加，训练时手动加）
        if role == "tool":
            content = f"<tool_response>\n{content}\n</tool_response>"

        prefix = f"<|im_start|>{role}\n"
        suffix = "<|im_end|>\n"

        prefix_ids  = tokenizer.encode(prefix,  add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        suffix_ids  = tokenizer.encode(suffix,  add_special_tokens=False)

        all_input_ids.extend(prefix_ids + content_ids + suffix_ids)

        if role == "assistant":
            # 对 content + <|im_end|> 计算 loss，prefix 忽略
            all_labels.extend(
                [IGNORE_INDEX] * len(prefix_ids)
                + content_ids
                + suffix_ids
            )
        else:
            all_labels.extend([IGNORE_INDEX] * (len(prefix_ids) + len(content_ids) + len(suffix_ids)))

    # 截断
    all_input_ids = all_input_ids[:max_length]
    all_labels    = all_labels[:max_length]

    # 如果整条样本没有任何可学习 token，丢弃
    if all(lbl == IGNORE_INDEX for lbl in all_labels):
        return None

    return {
        "input_ids":      all_input_ids,
        "labels":         all_labels,
        "attention_mask": [1] * len(all_input_ids),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples: list[dict] = []

        logger.info(f"Loading data from {data_path} ...")
        raw: list[list[dict]] = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                raw.append(json.loads(line)["messages"])

        skipped = 0
        for messages in raw:
            item = build_input_and_labels(messages, tokenizer, max_length)
            if item is None:
                skipped += 1
                continue
            self.samples.append(item)

        logger.info(f"Loaded {len(self.samples)} samples (skipped {skipped} empty)")

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
# 参数
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "Qwen3-8B 模型路径"})
    data:       str = field(default="train/sft_data.jsonl", metadata={"help": "训练数据 jsonl 路径"})
    max_length: int = field(default=8192,  metadata={"help": "最大序列长度"})
    seed:       int = field(default=42,    metadata={"help": "随机种子"})
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
        "--bf16",                   "True",
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

    set_seed(script_args.seed)

    logger.info(f"Model: {script_args.model_path}")
    logger.info(f"Data:  {script_args.data}")
    logger.info(f"NPU available: {_HAS_NPU}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_path,
        trust_remote_code=True,
        padding_side="right",   # SFT 右 padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 模型 ───────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,   # 910B 原生支持 bf16
        # attn_implementation="flash_attention_2"  # NPU 上暂不启用，需要验证支持情况
        attn_implementation="eager",   # NPU/GPU 均兼容
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
    )
    collator = SFTDataCollator(tokenizer=tokenizer)

    # ── Trainer ────────────────────────────────────────────────────────────────
    # HuggingFace Trainer 通过 accelerate 管理设备，torch_npu 注册 'npu'
    # 后 Trainer 会自动使用 NPU；无需修改任何训练逻辑。
    trainer = Trainer(
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
