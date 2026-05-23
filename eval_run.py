#!/usr/bin/env python3
"""
Evaluate a completed run. Pass a run directory or defaults to the latest.

Usage:
    python eval_run.py                           # eval latest run
    python eval_run.py runs/run_20260522_225813  # eval specific run
    python eval_run.py --model qwen_auto --batch 16  # custom eval params
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.eval_async import evaluate_trajectories
from agent.dataset_utils import load_jsonl

logger = logging.getLogger(__name__)


def _find_latest_run() -> str:
    runs = sorted(glob.glob("runs/run_*/submission.jsonl"), key=os.path.getmtime)
    if not runs:
        sys.exit("No runs found under runs/")
    return str(Path(runs[-1]).parent)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a completed agent run")
    p.add_argument("run_dir", nargs="?", default=None,
                   help="Path to run directory (default: latest)")
    p.add_argument("--dataset", default=os.environ.get("DATASET", "browsecomp_plus_hard50.jsonl"))
    p.add_argument("--model", default=os.environ.get("MODEL", "qwen_auto"))
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--batch", type=int, default=1, help="Eval batch size")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    return p


async def _main(args: argparse.Namespace) -> None:
    run_dir = args.run_dir or _find_latest_run()
    run_dir = str(Path(run_dir))
    submission_path = str(Path(run_dir) / "submission.jsonl")

    if not os.path.isfile(submission_path):
        sys.exit(f"submission.jsonl not found in {run_dir}")

    records = load_jsonl(submission_path)
    print(f"Run:     {run_dir}")
    print(f"Records: {len(records)}")
    print(f"Dataset: {args.dataset}")
    print(f"Model:   {args.model}")
    print()

    eval_path = str(Path(run_dir) / "eval.jsonl")
    summary, details = await evaluate_trajectories(
        records=records,
        dataset_path=args.dataset,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        eval_batch_size=args.batch,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_path=eval_path,
    )

    print(f"\n{'=' * 50}")
    print(f"Accuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total_queries']})")
    print(f"Avg tool calls/query: {summary['avg_tool_calls_per_query']}")
    print(f"Avg retrieved docs/query: {summary['avg_retrieved_docs_per_query']}")
    print(f"{'=' * 50}")

    # Save correct/incorrect splits
    eval_map = {d["query_id"]: d["eval_judgment"] for d in details}
    correct = [r for r in records if eval_map.get(r["query_id"]) == "CORRECT"]
    incorrect = [r for r in records if eval_map.get(r["query_id"]) == "INCORRECT"]

    def _write_jsonl(path: Path, recs: list[dict[str, Any]]) -> None:
        import json
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_jsonl(Path(run_dir) / "correct.json", correct)
    _write_jsonl(Path(run_dir) / "incorrect.json", incorrect)
    _write_jsonl(Path(run_dir) / "eval_correct.json",
                 [d for d in details if d["eval_judgment"] == "CORRECT"])
    _write_jsonl(Path(run_dir) / "eval_incorrect.json",
                 [d for d in details if d["eval_judgment"] == "INCORRECT"])

    print(f"\nSaved: {len(correct)} correct, {len(incorrect)} incorrect -> {run_dir}")


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stderr)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
