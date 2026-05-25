#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_distill.sh — DeepSeek 蒸馏数据收集
#
# Main agent:    DeepSeek API（教师模型，只用 smart_search）
# Summary agent: 本地 Qwen3-8B vLLM（文档摘要 subagent）
#
# API Key 配置：在 secrets.json 中填写（参考 secrets.json.example）
#   {
#     "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
#     "DEEPSEEK_API_KEY":  "sk-xxxx",
#     "DEEPSEEK_MODEL":    "deepseek-chat"
#   }
#
# 用法:
#   bash run_distill.sh                    # 全部 780 条训练题
#   bash run_distill.sh --limit 5          # 调试：只跑前 5 条
#   bash run_distill.sh --query-ids 1,2,3  # 指定 query_id
#
# 前提:
#   本地 vLLM 已启动（Qwen3-8B），供 summary subagent 使用
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

# ── 本地 vLLM 地址（仅用于启动前健康检查，实际配置从 secrets.json 读取）────────
LOCAL_VLLM_URL="${LOCAL_VLLM_URL:-http://127.0.0.1:8000/v1}"

# ── 数据集与索引 ──────────────────────────────────────────────────────────────
DATASET="${ROOT}/distill/browsecomp_plus_train780.jsonl"
INDEX_PATH="${ROOT}/indexes/browsecomp_plus_bm25.sqlite"
OUTPUT_DIR="${ROOT}/runs_distill"
CONCURRENCY="${CONCURRENCY:-4}"   # 同时处理的样本数

# ── 代理（DeepSeek 公网 API 需要时取消注释）──────────────────────────────────
# export https_proxy="http://pc.zinyy.tech:7899"
# export no_proxy="127.0.0.1,localhost"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

# ── 检查输入文件 ──────────────────────────────────────────────────────────────
[[ -f "${DATASET}"    ]] || { echo "[ERROR] 数据集不存在: ${DATASET}";    exit 1; }
[[ -f "${INDEX_PATH}" ]] || { echo "[ERROR] BM25 索引不存在: ${INDEX_PATH}"; exit 1; }
[[ -f "${ROOT}/secrets.json" ]] || {
    echo "[ERROR] secrets.json 不存在，请参考 secrets.json.example 创建"
    exit 1
}

# ── 检查本地 vLLM ─────────────────────────────────────────────────────────────
echo "[INFO] 检查本地 vLLM (${LOCAL_VLLM_URL}) ..."
if ! curl -sf "${LOCAL_VLLM_URL}/models" -o /dev/null; then
    echo "[ERROR] 本地 vLLM 未响应: ${LOCAL_VLLM_URL}"
    echo "        请先启动 vLLM (Qwen3-8B)"
    exit 1
fi
echo "[INFO] 本地 vLLM 正常"

# ── 打印配置 ──────────────────────────────────────────────────────────────────
read DEEPSEEK_MODEL_DISPLAY SUMMARY_MODEL_DISPLAY < <(python3 -c "
import json
s = json.load(open('secrets.json'))
print(s.get('DEEPSEEK_MODEL','deepseek-chat'), s.get('SUMMARY_AGENT_MODEL','qwen_auto'))
" 2>/dev/null || echo "deepseek-chat qwen_auto")

echo ""
echo "=================================================="
echo "  Main Agent:     DeepSeek (${DEEPSEEK_MODEL_DISPLAY})"
echo "  Summary Agent:  本地 vLLM (${SUMMARY_MODEL_DISPLAY} @ ${LOCAL_VLLM_URL})"
echo "  数据集:         ${DATASET}"
echo "  索引:           ${INDEX_PATH}"
echo "  输出目录:       ${OUTPUT_DIR}/"
echo "=================================================="
echo ""

mkdir -p "${OUTPUT_DIR}"

# ── 激活 conda 环境（可选）───────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-server}"
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV}" 2>/dev/null || true
fi

exec python -u run_serial.py \
    --agent-config configs/main_agent_deepseek.yaml \
    --dataset     "${DATASET}"      \
    --index-path  "${INDEX_PATH}"   \
    --output-dir  "${OUTPUT_DIR}"   \
    --concurrency "${CONCURRENCY}"  \
    --no-eval \
    "$@"
