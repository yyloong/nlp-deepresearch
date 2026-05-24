#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SFT 训练启动脚本 —— 支持单卡 GPU / Ascend 910B NPU
#
# 用法:
#   bash train/run_train.sh gpu   # 单卡 GPU
#   bash train/run_train.sh npu   # 单卡 910B NPU
#   bash train/run_train.sh       # 自动检测（有 NPU 用 NPU，否则 GPU）
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 基础配置（按需修改）────────────────────────────────────────────────────
MODEL_PATH="/path/to/Qwen3-8B"          # 模型路径
DATA_PATH="${SCRIPT_DIR}/sft_data.jsonl" # 转换后的训练数据
OUTPUT_DIR="${SCRIPT_DIR}/output"

# ── 训练超参 ────────────────────────────────────────────────────────────────
MAX_LENGTH=32768
EPOCHS=3
BATCH_SIZE=1                            # 单卡 per-device batch size
GRAD_ACC=16                             # 等效 global batch = 16
LR=1e-5
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=3
LOGGING_STEPS=10

# ── LoRA 超参 ────────────────────────────────────────────────────────────────
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# ─────────────────────────────────────────────────────────────────────────────
# 设备检测
# ─────────────────────────────────────────────────────────────────────────────
DEVICE="${1:-auto}"

if [ "$DEVICE" = "auto" ]; then
    if python -c "import torch_npu; assert torch_npu.npu.is_available()" 2>/dev/null; then
        DEVICE="npu"
    else
        DEVICE="gpu"
    fi
fi

echo "=================================================="
echo "  Device : ${DEVICE}"
echo "  Model  : ${MODEL_PATH}"
echo "  Data   : ${DATA_PATH}"
echo "  Output : ${OUTPUT_DIR}"
echo "  seq_len: ${MAX_LENGTH}"
echo "=================================================="

# ─────────────────────────────────────────────────────────────────────────────
# 数据预处理（如果 sft_data.jsonl 不存在则先转换）
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "$DATA_PATH" ]; then
    echo "[INFO] sft_data.jsonl 不存在，运行数据转换..."
    python "${SCRIPT_DIR}/convert_data.py"
fi

TOTAL=$(wc -l < "$DATA_PATH")
echo "[INFO] 训练集样本数: ${TOTAL}"

# ─────────────────────────────────────────────────────────────────────────────
# GPU 模式（单卡，直接 python）
# ─────────────────────────────────────────────────────────────────────────────
if [ "$DEVICE" = "gpu" ]; then
    # 指定使用哪张 GPU（默认第 0 张，多卡改为 0,1,2,3 等）
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

    python "${SCRIPT_DIR}/train.py" \
        --model_path        "$MODEL_PATH"         \
        --data              "$DATA_PATH"           \
        --max_length        "$MAX_LENGTH"          \
        --lora_r            "$LORA_R"              \
        --lora_alpha        "$LORA_ALPHA"          \
        --lora_dropout      "$LORA_DROPOUT"        \
        --lora_target_modules "$LORA_TARGETS"      \
        --output_dir        "$OUTPUT_DIR"          \
        --num_train_epochs  "$EPOCHS"              \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACC"  \
        --learning_rate     "$LR"                  \
        --lr_scheduler_type "$LR_SCHEDULER"        \
        --warmup_ratio      "$WARMUP_RATIO"        \
        --weight_decay      "$WEIGHT_DECAY"        \
        --bf16              True                   \
        --gradient_checkpointing True              \
        --save_strategy     steps                  \
        --save_steps        "$SAVE_STEPS"          \
        --save_total_limit  "$SAVE_TOTAL_LIMIT"    \
        --logging_steps     "$LOGGING_STEPS"       \
        --optim             adamw_torch            \
        --dataloader_num_workers 2                 \
        --report_to         none

# ─────────────────────────────────────────────────────────────────────────────
# NPU 模式（通过 accelerate 启动，使用 accelerate_npu.yaml 配置）
# ─────────────────────────────────────────────────────────────────────────────
elif [ "$DEVICE" = "npu" ]; then
    # 指定使用哪张 NPU（默认第 0 张）
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
    echo "[INFO] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"

    # torch_npu 推荐的环境变量
    export HCCL_CONNECT_TIMEOUT=600
    export TASK_QUEUE_ENABLE=1           # 异步算子队列，提升吞吐
    export COMBINED_ENABLE=1             # 算子合并优化

    accelerate launch \
        --config_file "${SCRIPT_DIR}/accelerate_npu.yaml" \
        "${SCRIPT_DIR}/train.py" \
        --model_path        "$MODEL_PATH"         \
        --data              "$DATA_PATH"           \
        --max_length        "$MAX_LENGTH"          \
        --lora_r            "$LORA_R"              \
        --lora_alpha        "$LORA_ALPHA"          \
        --lora_dropout      "$LORA_DROPOUT"        \
        --lora_target_modules "$LORA_TARGETS"      \
        --output_dir        "$OUTPUT_DIR"          \
        --num_train_epochs  "$EPOCHS"              \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACC"  \
        --learning_rate     "$LR"                  \
        --lr_scheduler_type "$LR_SCHEDULER"        \
        --warmup_ratio      "$WARMUP_RATIO"        \
        --weight_decay      "$WEIGHT_DECAY"        \
        --bf16              True                   \
        --gradient_checkpointing True              \
        --save_strategy     steps                  \
        --save_steps        "$SAVE_STEPS"          \
        --save_total_limit  "$SAVE_TOTAL_LIMIT"    \
        --logging_steps     "$LOGGING_STEPS"       \
        --optim             adamw_torch            \
        --dataloader_num_workers 2                 \
        --report_to         none

else
    echo "[ERROR] 未知设备类型: ${DEVICE}，请传入 gpu 或 npu"
    exit 1
fi

echo ""
echo "[INFO] 训练完成！checkpoint 保存在: ${OUTPUT_DIR}"
echo ""
echo "[INFO] 合并 LoRA 权重到完整模型（可选）:"
echo "  python - <<'EOF'"
echo "  from transformers import AutoModelForCausalLM"
echo "  from peft import PeftModel"
echo "  import torch"
echo "  base = AutoModelForCausalLM.from_pretrained('${MODEL_PATH}', torch_dtype=torch.bfloat16)"
echo "  model = PeftModel.from_pretrained(base, '${OUTPUT_DIR}')"
echo "  model = model.merge_and_unload()"
echo "  model.save_pretrained('${OUTPUT_DIR}/merged')"
echo "EOF"
