#!/usr/bin/env bash
#
# Deep Research Agent — 串行轨迹生成 + 评估
#
# Usage:
#   ./run_serial.sh                           # 使用默认参数
#   ./run_serial.sh --limit 10 --no-eval      # 只跑 10 条，跳过评估
#   ./run_serial.sh --max-tokens 8192         # 自定义参数
#   ./run_serial.sh --help                    # 查看所有参数
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── 固定默认值 ──
DATASET="${ROOT}/browsecomp_plus_hard50.jsonl"
INDEX_PATH="${ROOT}/indexes/browsecomp_plus_bm25.sqlite"
MODEL="${MODEL:-qwen_auto}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
OUTPUT_DIR="${ROOT}/runs"
MAX_TURNS="${MAX_TURNS:-30}"
MAX_TOKENS="${MAX_TOKENS:-8912}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-1}"
SEARCH_K="${SEARCH_K:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

# ── 清理代理变量（vLLM 在 localhost，不能走 proxy）──
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY 2>/dev/null || true

# ── 激活 conda 环境 ──
CONDA_ENV="${CONDA_ENV:-server}"
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV}" 2>/dev/null || {
        echo "WARNING: conda env '${CONDA_ENV}' not found, using current python"
    }
fi

# ── 检查前置条件 ──
if [ ! -f "${INDEX_PATH}" ]; then
    echo "ERROR: BM25 index not found at ${INDEX_PATH}"
    exit 1
fi
if [ ! -f "${DATASET}" ]; then
    echo "ERROR: Dataset not found at ${DATASET}"
    exit 1
fi

# ── 运行 ──
echo "=== Deep Research Agent (Serial) ==="
echo "Dataset:    ${DATASET}"
echo "Index:      ${INDEX_PATH}"
echo "Model:      ${MODEL}"
echo "Base URL:   ${BASE_URL}"
echo "max_turns:  ${MAX_TURNS}"
echo "max_tokens: ${MAX_TOKENS}"
echo "Output:     ${OUTPUT_DIR}/"
echo "===================================="
echo

cd "${ROOT}"

exec python run_serial.py \
    --dataset "${DATASET}" \
    --index-path "${INDEX_PATH}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-turns "${MAX_TURNS}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-tool-calls-per-turn "${MAX_TOOL_CALLS}" \
    --no-verify \
    --search-k "${SEARCH_K}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --think-trunc-no-think \
    "$@"
