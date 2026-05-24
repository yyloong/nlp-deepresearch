#!/usr/bin/env bash
# Run all previously-correct queries (union across all runs).
ROOT="$(cd "$(dirname "$0")" && pwd)"

DATASET="${ROOT}/browsecomp_plus_hard50.jsonl"
INDEX_PATH="${ROOT}/indexes/browsecomp_plus_bm25.sqlite"
MODEL="${MODEL:-qwen_auto}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
OUTPUT_DIR="${ROOT}/runs"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY 2>/dev/null || true
cd "$ROOT"

CORRECT_IDS=$(python3 -c "
import json, glob
ids = set()
for f in sorted(glob.glob('runs/*/eval_correct.json')):
    with open(f) as fh:
        for line in fh:
            if line.strip():
                ids.add(json.loads(line.strip())['query_id'])
print(','.join(sorted(ids, key=int)))
")

echo "=== Collected $(echo "$CORRECT_IDS" | tr ',' '\n' | wc -l) correct query_ids ==="
echo

exec python run_serial.py \
    --dataset "${DATASET}" \
    --query-ids "${CORRECT_IDS}" \
    --index-path "${INDEX_PATH}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    "$@"
