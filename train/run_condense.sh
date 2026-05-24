#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_condense.sh — 对超长 SFT 样本做离线 self-condense 预处理
#
# 用法:
#   bash train/run_condense.sh            # 使用默认配置
#   bash train/run_condense.sh --force    # 强制重新处理（忽略已有输出）
#
# 前提:
#   vLLM 服务已在 BASE_URL 启动并加载好模型
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# vLLM 监听本地端口，代理会导致连接失败
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 配置（按需修改）──────────────────────────────────────────────────────────
INPUT="${SCRIPT_DIR}/sft_data.jsonl"          # 原始 SFT 数据（convert_data.py 的输出）
OUTPUT="${SCRIPT_DIR}/sft_data_condensed.jsonl" # 压缩后输出
TOKENIZER="/home/u-longyy/Qwen3-8B"          # tokenizer 本地路径（与训练保持一致）
TARGET_LENGTH=12288                           # back 的 token 上限（训练窗口16384 - 摘要4096）

# vLLM API 配置
BASE_URL="http://127.0.0.1:7999/v1"
API_KEY="dummy"
MODEL="qwen_auto"                             # 与 vLLM --served-model-name 一致

# 并发与输出长度
CONCURRENCY=16                                # 并发请求数
# 对齐 main_agent_smart.yaml：max_tokens=4096
# target_length = 训练窗口(16384) - 摘要最大长度(4096) = 12288
# 保证 [system, summary_user(≤4096 tok), back(≤12288 tok)] 总计 ≤ 16384
CONDENSE_MAX_TOKENS=4096
TARGET_LENGTH=12288

# 本批次只处理原始 token 数 ≤ MAX_INPUT_TOKENS 的样本（0=不过滤）
# 先跑较短的样本可以快速产出训练数据，超长样本留待后续批次
MAX_INPUT_TOKENS=100000

# 是否强制重新处理
FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

# ─────────────────────────────────────────────────────────────────────────────
# 检查输入
# ─────────────────────────────────────────────────────────────────────────────
if [[ ! -f "$INPUT" ]]; then
    echo "[ERROR] 输入文件不存在: $INPUT"
    echo "        请先运行 convert_data.py 生成 sft_data.jsonl"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 跳过已处理
# ─────────────────────────────────────────────────────────────────────────────
if [[ -f "$OUTPUT" && "$FORCE" == false ]]; then
    echo "[INFO] 输出文件已存在: $OUTPUT"
    echo "       若需重新处理，请运行: bash train/run_condense.sh --force"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# 检查 vLLM 服务
# ─────────────────────────────────────────────────────────────────────────────
echo "[INFO] 检查 vLLM 服务 ${BASE_URL} ..."
if ! curl -sf "${BASE_URL}/models" -o /dev/null; then
    echo "[ERROR] vLLM 服务未响应: ${BASE_URL}"
    echo "        请先启动 vLLM，例如:"
    echo "          python -m vllm.entrypoints.openai.api_server \\"
    echo "            --model ${TOKENIZER} \\"
    echo "            --served-model-name ${MODEL} \\"
    echo "            --port 8000"
    exit 1
fi
echo "[INFO] vLLM 服务正常"

# ─────────────────────────────────────────────────────────────────────────────
# 打印配置
# ─────────────────────────────────────────────────────────────────────────────
echo "=================================================="
echo "  Input         : ${INPUT}"
echo "  Output        : ${OUTPUT}"
echo "  Tokenizer     : ${TOKENIZER}"
echo "  Target length : ${TARGET_LENGTH} tokens"
echo "  API base_url  : ${BASE_URL}"
echo "  Model         : ${MODEL}"
echo "  Concurrency   : ${CONCURRENCY}"
echo "  Max tokens    : ${CONDENSE_MAX_TOKENS}"
echo "  Max input tok : ${MAX_INPUT_TOKENS} (0=no filter)"
echo "=================================================="

# ─────────────────────────────────────────────────────────────────────────────
# 运行压缩
# ─────────────────────────────────────────────────────────────────────────────
cd "$ROOT_DIR"

conda run -n server python train/condense_long.py \
    --input               "$INPUT"               \
    --output              "$OUTPUT"              \
    --tokenizer           "$TOKENIZER"           \
    --target_length       "$TARGET_LENGTH"       \
    --base_url            "$BASE_URL"            \
    --api_key             "$API_KEY"             \
    --model               "$MODEL"               \
    --concurrency         "$CONCURRENCY"         \
    --condense_max_tokens "$CONDENSE_MAX_TOKENS" \
    --max_input_tokens    "$MAX_INPUT_TOKENS"

echo ""
echo "=================================================="
echo "  压缩完成: ${OUTPUT}"
echo "  接下来可以运行训练:"
echo "    bash train/run_train.sh gpu"
echo "  注意在 run_train.sh 里把 DATA_PATH 改为:"
echo "    DATA_PATH=\"\${SCRIPT_DIR}/sft_data_condensed.jsonl\""
echo "=================================================="
