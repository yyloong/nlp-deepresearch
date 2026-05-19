#!/usr/bin/env bash
#
# Deep Research Agent — 一键轨迹生成 + 评估
#
# Usage:
#   ./run_agent.sh                           # 使用默认参数
#   ./run_agent.sh --limit 10 --no-eval      # 只跑 10 条，跳过评估
#   ./run_agent.sh --n-envs 8 --max-tokens 8192  # 自定义参数
#   ./run_agent.sh --help                    # 查看所有参数
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── 固定默认值 ──
DATASET="${ROOT}/browsecomp_plus_hard50.jsonl"
INDEX_PATH="${ROOT}/indexes/browsecomp_plus_bm25.sqlite"
MODEL="${MODEL:-qwen_auto}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
OUTPUT_DIR="${ROOT}/runs"
N_ENVS="${N_ENVS:-10}"
MAX_TURNS="${MAX_TURNS:-30}"
MAX_TOKENS="${MAX_TOKENS:-8912}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-1}"
SEARCH_K="${SEARCH_K:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NO_THINK="${NO_THINK:-0}"

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
    echo "Build it first: python -m agent.build_bm25_index --corpus-path ${ROOT}/browsecomp-plus-corpus --index-path ${INDEX_PATH}"
    exit 1
fi

if [ ! -f "${DATASET}" ]; then
    echo "ERROR: Dataset not found at ${DATASET}"
    exit 1
fi

# ── 运行 ──
echo "=== Deep Research Agent ==="
echo "Dataset:    ${DATASET}"
echo "Index:      ${INDEX_PATH}"
echo "Model:      ${MODEL}"
echo "Base URL:   ${BASE_URL}"
echo "n_envs:     ${N_ENVS}"
echo "max_tokens: ${MAX_TOKENS}"
echo "max_tool_calls: ${MAX_TOOL_CALLS}"
echo "Output:     ${OUTPUT_DIR}/"
echo "==========================="
echo

cd "${ROOT}"

NO_THINK_FLAG=()
if [[ -n "${NO_THINK}" && "${NO_THINK}" != "0" && "${NO_THINK}" != "false" ]]; then
    NO_THINK_FLAG=(--no-think)
fi

exec python -m agent.agent_loop \
    --dataset "${DATASET}" \
    --index-path "${INDEX_PATH}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-envs "${N_ENVS}" \
    --max-turns "${MAX_TURNS}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-tool-calls-per-turn "${MAX_TOOL_CALLS}" \
    --search-k "${SEARCH_K}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    "${NO_THINK_FLAG[@]}" \
    "$@"
