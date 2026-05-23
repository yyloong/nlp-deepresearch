#!/usr/bin/env bash
#
# Deep Research Agent — Basic (search + get_document + submit_answer, no verify, no subagent)
#
# Usage:
#   ./run_basic.sh                           # 使用默认参数
#   ./run_basic.sh --limit 10 --no-eval      # 只跑 10 条，跳过评估
#   ./run_basic.sh --max-tokens 8192         # 自定义参数
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

DATASET="${ROOT}/browsecomp_plus_hard50.jsonl"
INDEX_PATH="${ROOT}/indexes/browsecomp_plus_bm25.sqlite"
#MODEL="${MODEL:-qwen_auto}"
MODEL="${MODEL:-DeepSeek-V4-Flash}"
#BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
BASE_URL="${BASE_URL:-https://api.zinyy.tech/v1}"
OUTPUT_DIR="${ROOT}/runs"
MAX_TURNS="${MAX_TURNS:-30}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
SEARCH_K="${SEARCH_K:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY 2>/dev/null || true
export http_proxy=http://pc.zinyy.tech:7899 
export https_proxy=http://pc.zinyy.tech:7899 



CONDA_ENV="${CONDA_ENV:-server}"
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV}" 2>/dev/null || {
        echo "WARNING: conda env '${CONDA_ENV}' not found, using current python"
    }
fi

if [ ! -f "${INDEX_PATH}" ]; then
    echo "ERROR: BM25 index not found at ${INDEX_PATH}"
    exit 1
fi
if [ ! -f "${DATASET}" ]; then
    echo "ERROR: Dataset not found at ${DATASET}"
    exit 1
fi

echo "=== Deep Research Agent (Basic) ==="
echo "Dataset:    ${DATASET}"
echo "Index:      ${INDEX_PATH}"
echo "Model:      ${MODEL}"
echo "Base URL:   ${BASE_URL}"
echo "max_turns:  ${MAX_TURNS}"
echo "max_tokens: ${MAX_TOKENS}"
echo "search_k:   ${SEARCH_K}"
echo "Output:     ${OUTPUT_DIR}/"
echo "===================================="
echo

cd "${ROOT}"

exec python run_serial.py \
    --agent-config configs/main_agent_basic.yaml \
    --dataset "${DATASET}" \
    --index-path "${INDEX_PATH}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-turns "${MAX_TURNS}" \
    --max-tokens "${MAX_TOKENS}" \
    --search-k "${SEARCH_K}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --no-verify \
    "$@"
