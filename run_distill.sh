#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_distill.sh — DeepSeek 蒸馏数据收集
#
# Main agent:      DeepSeek API（教师模型，高质量轨迹）
# Summary agent:   本地 Qwen3-8B vLLM（文档摘要）
# Judge agent:     本地 Qwen3-8B vLLM（文档相关性过滤）
#
# 用法:
#   export DEEPSEEK_API_KEY="sk-xxxx"
#   bash run_distill.sh                    # 跑全部 780 条训练题
#   bash run_distill.sh --limit 50         # 调试：只跑前 50 条
#   bash run_distill.sh --query-ids 1,2,3  # 指定 query_id
#
# 前提:
#   本地 vLLM 已启动（Qwen3-8B），供 summary/judge subagent 使用
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── DeepSeek API 配置 ─────────────────────────────────────────────────────────
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?请设置 DEEPSEEK_API_KEY 环境变量}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"

# ── 本地 vLLM 配置（summary / judge subagent）────────────────────────────────
LOCAL_VLLM_URL="${LOCAL_VLLM_URL:-http://127.0.0.1:8000/v1}"
LOCAL_VLLM_MODEL="${LOCAL_VLLM_MODEL:-qwen_auto}"

export SUMMARY_AGENT_BASE_URL="${LOCAL_VLLM_URL}"
export SUMMARY_AGENT_API_KEY="dummy"
export SUMMARY_AGENT_MODEL="${LOCAL_VLLM_MODEL}"

# judge/verify agent 也用本地（yaml 里不配置 base_url 则自动 fallback）

# ── 数据集与索引 ──────────────────────────────────────────────────────────────
DATASET="${ROOT}/distill/browsecomp_plus_train780.jsonl"
INDEX_PATH="${ROOT}/indexes/browsecomp_plus_bm25.sqlite"
OUTPUT_DIR="${ROOT}/runs_distill"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

# ── 代理设置（DeepSeek 需要，本地 vLLM 不需要）──────────────────────────────
# 注意：DeepSeek API 走公网，本地 vLLM 走 127.0.0.1
# run_serial.py 里 per-agent client 已分离，代理设置对本地 client 无效
# 如果 DeepSeek 需要代理，在这里配置：
# export https_proxy="http://proxy:port"
# export no_proxy="127.0.0.1,localhost"
# 如果不需要代理（直连），清空：
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

# ── 检查输入文件 ──────────────────────────────────────────────────────────────
if [[ ! -f "${DATASET}" ]]; then
    echo "[ERROR] 训练数据集不存在: ${DATASET}"
    echo "        请先生成: python3 -c \"...\" (见注释)"
    exit 1
fi
if [[ ! -f "${INDEX_PATH}" ]]; then
    echo "[ERROR] BM25 索引不存在: ${INDEX_PATH}"
    exit 1
fi

# ── 检查本地 vLLM ─────────────────────────────────────────────────────────────
echo "[INFO] 检查本地 vLLM (${LOCAL_VLLM_URL}) ..."
if ! curl -sf "${LOCAL_VLLM_URL}/models" -o /dev/null; then
    echo "[ERROR] 本地 vLLM 未响应: ${LOCAL_VLLM_URL}"
    echo "        请先启动 vLLM (Qwen3-8B)"
    exit 1
fi
echo "[INFO] 本地 vLLM 正常"

# ── 打印配置 ──────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  模式:           蒸馏数据收集"
echo "  Main Agent:     DeepSeek (${DEEPSEEK_MODEL})"
echo "  Summary/Judge:  本地 vLLM (${LOCAL_VLLM_MODEL} @ ${LOCAL_VLLM_URL})"
echo "  数据集:         ${DATASET}"
echo "  索引:           ${INDEX_PATH}"
echo "  输出目录:       ${OUTPUT_DIR}/"
echo "=================================================="
echo ""

mkdir -p "${OUTPUT_DIR}"

CONDA_ENV="${CONDA_ENV:-server}"
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV}" 2>/dev/null || echo "WARNING: conda env '${CONDA_ENV}' not found"
fi

cd "${ROOT}"

exec python run_serial.py \
    --agent-config configs/main_agent_deepseek.yaml \
    --dataset      "${DATASET}"    \
    --index-path   "${INDEX_PATH}" \
    --base-url     "${LOCAL_VLLM_URL}"   \
    --model        "${LOCAL_VLLM_MODEL}" \
    --output-dir   "${OUTPUT_DIR}"       \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --no-eval \
    "$@"
