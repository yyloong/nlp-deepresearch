#!/usr/bin/env bash
# Run all previously-correct queries (union across all runs).
# Usage:
#   bash run_correct.sh
#   bash run_correct.sh --limit 5           # extra args forwarded to run_serial.py
#   DATASET=data/dataset.jsonl bash run_correct.sh
ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── 固定默认值 ──
DATASET="${ROOT}/browsecomp_plus_hard50.jsonl"
INDEX_PATH="${ROOT}/indexes/browsecomp_plus_bm25.sqlite"
MODEL="${MODEL:-qwen_auto}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
OUTPUT_DIR="${ROOT}/runs"
MAX_TURNS="${MAX_TURNS:-10}"
MAX_TOKENS="${MAX_TOKENS:-8912}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-1}"
SEARCH_K="${SEARCH_K:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NO_THINK="${NO_THINK:-0}"


unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY 2>/dev/null || true
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Collect correct query_ids from all runs
CORRECT_IDS=$(python3 -c "
import json, glob

ids = set()
for f in sorted(glob.glob('runs/*/eval_correct.json')):
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                ids.add(row['query_id'])
print(','.join(sorted(ids, key=int)))
")

NUM=$(echo "$CORRECT_IDS" | tr ',' '\n' | wc -l)
echo "=== Collected $NUM correct query_ids ==="
echo "IDs: $CORRECT_IDS"
echo ""


exec python run_serial.py \
    --dataset "${DATASET}" \
    --query-ids "${CORRECT_IDS}" \
    --index-path "${INDEX_PATH}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-turns "${MAX_TURNS}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-tool-calls-per-turn "${MAX_TOOL_CALLS}" \
    --no-condense-thinking \
    --no-strip-thinking \
    --search-k "${SEARCH_K}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    "${NO_THINK_FLAG[@]}" \
    "$@"


