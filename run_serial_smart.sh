#!/usr/bin/env bash
#
# Smart Agent — smart_search + get_document + submit_answer.
# All agent params from YAML. Only MODEL/BASE_URL from env.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

DATASET="${ROOT}/browsecomp_plus_hard50.jsonl"
INDEX_PATH="${ROOT}/indexes/browsecomp_plus_bm25.sqlite"
OUTPUT_DIR="${ROOT}/runs"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

# ── 全局 fallback（当 YAML 里未配置 per-agent API 时使用）──
MODEL="${MODEL:-qwen_auto}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"

# ── Main Agent（DeepSeek API 蒸馏模式）──────────────────────────────────────
# 设置后 main agent 使用 DeepSeek，summary subagent 仍用本地 vLLM
# export MAIN_AGENT_BASE_URL="https://api.deepseek.com/v1"
# export MAIN_AGENT_API_KEY="sk-xxxx"
# export MAIN_AGENT_MODEL="deepseek-chat"

# ── Summary Subagent（本地 Qwen3-8B）────────────────────────────────────────
# export SUMMARY_AGENT_BASE_URL="http://127.0.0.1:8000/v1"
# export SUMMARY_AGENT_API_KEY="dummy"
# export SUMMARY_AGENT_MODEL="qwen_auto"

# ── Relevance Judge / Verify Agent（默认本地）───────────────────────────────
# export JUDGE_AGENT_BASE_URL=...  (relevance_judge_agent.yaml 如需覆盖)

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
