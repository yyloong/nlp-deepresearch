#!/usr/bin/env bash
# 用 DeepSeek main agent + 本地 Qwen3-8B summary 跑 hard50 eval
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

LOCAL_VLLM_URL="${LOCAL_VLLM_URL:-http://127.0.0.1:8000/v1}"
DATASET="${ROOT}/browsecomp_plus_hard50.jsonl"
INDEX_PATH="${ROOT}/indexes/browsecomp_plus_bm25.sqlite"
OUTPUT_DIR="${ROOT}/runs"
CONCURRENCY="${CONCURRENCY:-50}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

[[ -f "${DATASET}"    ]] || { echo "[ERROR] 数据集不存在: ${DATASET}";    exit 1; }
[[ -f "${INDEX_PATH}" ]] || { echo "[ERROR] BM25 索引不存在: ${INDEX_PATH}"; exit 1; }
[[ -f "${ROOT}/secrets.json" ]] || { echo "[ERROR] secrets.json 不存在"; exit 1; }

echo "[INFO] 检查本地 vLLM (${LOCAL_VLLM_URL}) ..."
curl -sf "${LOCAL_VLLM_URL}/models" -o /dev/null || { echo "[ERROR] 本地 vLLM 未响应"; exit 1; }
echo "[INFO] 本地 vLLM 正常"

CONDA_ENV="${CONDA_ENV:-server}"
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV}" 2>/dev/null || true
fi

exec python -u run_serial.py \
    --agent-config configs/main_agent_deepseek.yaml \
    --dataset     "${DATASET}"     \
    --index-path  "${INDEX_PATH}"  \
    --output-dir  "${OUTPUT_DIR}"  \
    --concurrency "${CONCURRENCY}" \
    "$@"
