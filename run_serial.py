#!/usr/bin/env python3
"""
Serial (single-instance) agent evaluation — processes questions one at a time.

Eliminates any variability from parallel batching / async interleaving by
using n_envs=1 and executing all model calls sequentially.

Defaults are aligned with run_agent.sh (env vars supported):
    DATASET, INDEX_PATH, MODEL, BASE_URL, OUTPUT_DIR, MAX_TURNS,
    MAX_TOKENS, MAX_TOOL_CALLS, SEARCH_K, EVAL_BATCH_SIZE, NO_THINK

Usage:
    # 完整运行 + 评估（使用环境变量默认值）
    DATASET=browsecomp_plus_hard50.jsonl \\
    INDEX_PATH=indexes/browsecomp_plus_bm25.sqlite \\
    python run_serial.py

    # 命令行覆盖
    python run_serial.py \\
        --dataset data/browsecomp_plus_hard50.jsonl \\
        --index-path indexes/browsecomp_plus_bm25.sqlite \\
        --model qwen_auto --limit 10

    # 跳过评估
    python run_serial.py --no-eval --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.eval_async import evaluate_trajectories  # noqa: E402

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Serial agent evaluation — one question at a time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── 与 run_agent.sh 对齐的参数（支持环境变量覆盖）──
    p.add_argument("--dataset", default=os.environ.get("DATASET", ""),
                   help="Dataset jsonl path (env: DATASET)")
    p.add_argument("--index-path", default=os.environ.get("INDEX_PATH", ""),
                   help="BM25 SQLite index path (env: INDEX_PATH)")
    p.add_argument("--model", default=os.environ.get("MODEL", "qwen_auto"),
                   help="vLLM model name (env: MODEL)")
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "runs"))
    p.add_argument("--max-turns", type=int, default=int(os.environ.get("MAX_TURNS", "10")))
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "10240")))
    p.add_argument("--max-tool-calls-per-turn", type=int,
                   default=int(os.environ.get("MAX_TOOL_CALLS", "1")))
    p.add_argument("--search-k", type=int, default=int(os.environ.get("SEARCH_K", "5")))
    p.add_argument("--snippet-max-chars", type=int, default=1200)
    p.add_argument("--eval-batch-size", type=int, default=1,
                   help="Eval batch size (default: 1 for serial eval)")
    p.add_argument("--eval-model", default=None, help="Eval model (defaults to --model)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    p.add_argument("--query-ids", type=str, default=None,
                   help="Comma-separated list of query IDs to run (e.g. '442,26,471')")
    p.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    p.add_argument("--eval-only", type=str, default=None,
                   help="Only run eval on an existing submission.jsonl (provide path to submission)")
    p.add_argument("--no-verify", action="store_true", help="Disable verify agent (submit_answer returns mock success)")
    # ── Serial-only 或额外参数 ──
    p.add_argument("--max-context", type=int, default=40960,
                   help="Max context window for auto-condensation")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--think-trunc-no-think", action="store_true", default=False,
                   help="When think-block is truncated, first retry with thinking disabled "
                        "before falling back to RETRY_NUDGE")
    return p


async def _main_async(args: argparse.Namespace) -> None:
    from agent.dataset_utils import load_jsonl
    from agent.agent_loop import generate_trajectories

    # ── Eval-only mode: load existing submission.jsonl and evaluate ──
    if args.eval_only:
        records = load_jsonl(args.eval_only)
        print(f"Loaded {len(records)} records from {args.eval_only}", flush=True)
        if not args.dataset:
            sys.exit("ERROR: --dataset is required for eval")
        eval_model = args.eval_model or args.model
        eval_path = str(Path(args.eval_only).parent / "eval.jsonl")
        summary, details = await evaluate_trajectories(
            records=records,
            dataset_path=args.dataset,
            model=eval_model,
            base_url=args.base_url,
            api_key=args.api_key,
            eval_batch_size=args.eval_batch_size,
            temperature=0.0,
            max_tokens=8192,
            output_path=eval_path,
        )
        print(f"\nAccuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total_queries']})")
        print(f"Avg tool calls/query: {summary['avg_tool_calls_per_query']}")
        print(f"Eval saved: {eval_path}")
        return

    # Validate required args
    if not args.dataset:
        sys.exit("ERROR: --dataset is required (set DATASET env var or pass --dataset)")
    if not args.index_path:
        sys.exit("ERROR: --index-path is required (set INDEX_PATH env var or pass --index-path)")

    print(f"=== Serial Agent (n_envs=1) ===")
    print(f"Dataset:    {args.dataset}")
    print(f"Index:      {args.index_path}")
    print(f"Model:      {args.model}")
    print(f"Base URL:   {args.base_url}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"verify:     {not args.no_verify}")
    print(f"Output:     {args.output_dir}/")
    print(f"===============================")
    print()

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = str(Path(args.output_dir) / f"run_{ts}")
    submission_path = str(Path(output_dir) / "submission.jsonl")

    query_ids = None
    if args.query_ids:
        query_ids = args.query_ids.replace(" ", "").split(",")

    records = await generate_trajectories(
        dataset_path=args.dataset,
        index_path=args.index_path,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        output_path=submission_path,
        n_envs=1,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_context=args.max_context,
        search_k=args.search_k,
        snippet_max_chars=args.snippet_max_chars,
        limit=args.limit,
        query_ids=query_ids,
        max_tool_calls_per_turn=args.max_tool_calls_per_turn,
        think_trunc_no_think=args.think_trunc_no_think,
    )

    if args.no_eval:
        return

    eval_model = args.eval_model or args.model
    eval_path = str(Path(output_dir) / "eval.jsonl")
    summary, details = await evaluate_trajectories(
        records=records,
        dataset_path=args.dataset,
        model=eval_model,
        base_url=args.base_url,
        api_key=args.api_key,
        eval_batch_size=args.eval_batch_size,
        temperature=0.0,
        max_tokens=8192,
        output_path=eval_path,
    )

    print(f"\n{'=' * 50}")
    print(f"Accuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total_queries']})")
    print(f"Avg tool calls/query: {summary['avg_tool_calls_per_query']}")
    print(f"Avg retrieved docs/query: {summary['avg_retrieved_docs_per_query']}")
    print(f"{'=' * 50}")

    eval_map = {d["query_id"]: d["eval_judgment"] for d in details}
    correct_recs = [r for r in records if eval_map.get(r["query_id"]) == "CORRECT"]
    incorrect_recs = [r for r in records if eval_map.get(r["query_id"]) == "INCORRECT"]

    def _write_jsonl(path: Path, recs: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_jsonl(Path(output_dir) / "correct.json", correct_recs)
    _write_jsonl(Path(output_dir) / "incorrect.json", incorrect_recs)

    eval_correct = [d for d in details if d["eval_judgment"] == "CORRECT"]
    eval_incorrect = [d for d in details if d["eval_judgment"] == "INCORRECT"]
    _write_jsonl(Path(output_dir) / "eval_correct.json", eval_correct)
    _write_jsonl(Path(output_dir) / "eval_incorrect.json", eval_incorrect)

    print(f"\nSaved: {len(correct_recs)} correct, {len(incorrect_recs)} incorrect → {output_dir}")


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
