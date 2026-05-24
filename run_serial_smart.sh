#!/usr/bin/env bash
#
# Smart Agent — smart_search + get_document + submit_answer.
# All agent params from YAML. Only MODEL/BASE_URL from env.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

DATASET="${ROOT}/browsecomp_plus_hard50.jsonl"
INDEX_PATH="${ROOT}/indexes/browsecomp_plus_bm25.sqlite"
MODEL="${MODEL:-qwen_auto}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
OUTPUT_DIR="${ROOT}/runs"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY 2>/dev/null || true

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

echo "=== Smart Agent ==="
echo "Dataset:  ${DATASET}"
echo "Model:    ${MODEL}"
echo "Base URL: ${BASE_URL}"
echo "Output:   ${OUTPUT_DIR}/"
echo "===================="
echo

cd "${ROOT}"

exec python run_serial.py \
    --agent-config configs/main_agent_smart.yaml \
    --dataset "${DATASET}" \
    --index-path "${INDEX_PATH}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    "$@"
