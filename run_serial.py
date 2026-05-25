#!/usr/bin/env python3
"""
Serial (single-instance) agent evaluation — processes questions one at a time.

All agent parameters come from YAML. Only model and base-url are CLI-overridable.
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

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.agent import Agent
from agent.browsecomp_searcher import BrowseCompBM25Searcher
from agent.dataset_utils import load_jsonl
from agent.eval_async import evaluate_trajectories
from agent.tool_func import ToolRegistry
from agent.utils import extract_final_answer
from agent.vllm_client_async import VLLMClientAsync

logger = logging.getLogger(__name__)

# ── Load secrets.json (if present) into os.environ ───────────────────────────
_SECRETS_FILE = Path(__file__).parent / "secrets.json"
if _SECRETS_FILE.exists():
    _secrets = json.loads(_SECRETS_FILE.read_text(encoding="utf-8"))
    for _k, _v in _secrets.items():
        os.environ.setdefault(_k, str(_v))
    print(f"[secrets] Loaded {len(_secrets)} keys from {_SECRETS_FILE}", flush=True)

_TOKENIZER_PATH = "Qwen/Qwen3-8B"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer

_tok: Any = AutoTokenizer.from_pretrained(
    _TOKENIZER_PATH, trust_remote_code=True, local_files_only=True,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Serial agent evaluation — one question at a time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=os.environ.get("DATASET", ""))
    p.add_argument("--index-path", default=os.environ.get("INDEX_PATH", ""))
    p.add_argument("--model", default=os.environ.get("MODEL", "qwen_auto"),
                   help="Model name — overrides YAML (env: MODEL)")
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "runs"))
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--eval-model", default=None, help="Eval model (defaults to --model)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--query-ids", type=str, default=None,
                   help="Comma-separated list of query IDs")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--eval-only", type=str, default=None,
                   help="Only eval existing submission.jsonl")
    p.add_argument("--agent-config", default="configs/main_agent_smart.yaml",
                   help="Path to main agent YAML config")
    return p


def _load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_searcher(index_path: str) -> BrowseCompBM25Searcher:
    return BrowseCompBM25Searcher(index_path=index_path)


async def _main_async(args: argparse.Namespace) -> None:
    from agent.dataset_utils import load_jsonl

    # ── Tokenizer ──
    global _TOKENIZER_PATH, _tok

    # ── Eval-only mode ──
    if args.eval_only:
        records = load_jsonl(args.eval_only)
        print(f"Loaded {len(records)} records from {args.eval_only}", flush=True)
        if not args.dataset:
            sys.exit("ERROR: --dataset is required for eval")
        eval_model = args.eval_model or args.model
        eval_path = str(Path(args.eval_only).parent / "eval.jsonl")
        summary, details = await evaluate_trajectories(
            records=records, dataset_path=args.dataset, model=eval_model,
            base_url=args.base_url, api_key=args.api_key,
            eval_batch_size=args.eval_batch_size, temperature=0.0, max_tokens=8192,
            output_path=eval_path,
        )
        print(f"\nAccuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total_queries']})")
        print(f"Avg tool calls/query: {summary['avg_tool_calls_per_query']}")
        print(f"Eval saved: {eval_path}")
        return

    if not args.dataset:
        sys.exit("ERROR: --dataset is required")
    if not args.index_path:
        sys.exit("ERROR: --index-path is required")

    print(f"=== Serial Agent ===")
    print(f"Dataset:    {args.dataset}")
    print(f"Index:      {args.index_path}")
    print(f"Output:     {args.output_dir}/")
    print(f"===================")
    print()

    rows = load_jsonl(args.dataset, limit=args.limit)
    if args.query_ids:
        ids = set(args.query_ids.replace(" ", "").split(","))
        rows = [r for r in rows if r.get("query_id", "") in ids]
        print(f"Filtered to {len(rows)} queries by query_ids", flush=True)
    total = len(rows)

    # ── Shared client & searcher ──
    _global_client = VLLMClientAsync(base_url=args.base_url, api_key=args.api_key, max_concurrent=10)
    _client_cache: Dict[str, VLLMClientAsync] = {}  # (base_url, api_key) → client
    searcher = build_searcher(args.index_path)

    def _resolve_env(val: str) -> str:
        """Expand ${ENV_VAR} placeholders. Unset vars resolve to empty string."""
        import re
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), val)

    def _get_client(base_url: str, api_key: str) -> VLLMClientAsync:
        key = (base_url, api_key)
        if key not in _client_cache:
            _client_cache[key] = VLLMClientAsync(
                base_url=base_url, api_key=api_key, max_concurrent=10,
            )
        return _client_cache[key]

    # ── Agent factory — supports per-agent base_url/api_key/model in YAML ──
    def _make_agent(config_path: str, override_model: bool = True) -> Agent:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Per-agent API config (YAML keys: base_url, api_key, model)
        agent_base_url = _resolve_env(str(cfg.pop("base_url", ""))) or args.base_url
        agent_api_key  = _resolve_env(str(cfg.pop("api_key",  ""))) or args.api_key
        agent_model    = _resolve_env(str(cfg.get("model",    "")))

        # Determine if using local model (vLLM) or remote API
        is_local = "127.0.0.1" in agent_base_url or "localhost" in agent_base_url
        cfg["_is_local_model"] = is_local

        # Resolve model: env-expanded YAML value wins; fallback to CLI --model for local agents
        if agent_model:
            cfg["model"] = agent_model
        elif override_model and is_local:
            cfg["model"] = args.model

        agent_client = _get_client(agent_base_url, agent_api_key)

        import uuid, tempfile
        patched_path = os.path.join(tempfile.gettempdir(), f"agent_{uuid.uuid4().hex}.yaml")
        with open(patched_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        return Agent(patched_path, client=agent_client, tokenizer=_tok)

    def agent_factory(config_path: str) -> Agent:
        return _make_agent(config_path, override_model=True)

    # ── ToolRegistry with agent_factory ──
    # ── Main agent — created first so ToolRegistry can reference it ──
    main_agent = _make_agent(args.agent_config)
    tool_registry = ToolRegistry(searcher=searcher, agent_factory=agent_factory, main_agent=main_agent)
    main_agent.tool_registry = tool_registry.build_registry(main_agent._tool_names, main_agent._tool_config)

    # ── Output setup ──
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = str(Path(args.output_dir) / f"run_{ts}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    submission_path = str(Path(output_dir) / "submission.jsonl")
    traj_dir = str(Path(output_dir) / "trajectories")
    Path(traj_dir).mkdir(parents=True, exist_ok=True)

    # ── Process each question ──
    records: List[Dict[str, Any]] = []
    try:
        for i, row in enumerate(rows):
            qid = row["query_id"]
            question = row.get("question") or row["query"]
            gold_answer = row.get("answer", "") or row.get("gold_answer", "") or ""

            t0 = time.time()
            q_preview = question[:200].replace("\n", " ")
            if len(question) > 200:
                q_preview += "..."
            print(f"┌{'─'*78}┐\n│ [{i+1}/{total}]  qid={qid}\n├{'─'*78}┤\n│ Question: {q_preview}")
            if gold_answer:
                gold_preview = gold_answer[:150].replace("\n", " ")
                if len(gold_answer) > 150:
                    gold_preview += "..."
                print(f"│ Gold Ans: {gold_preview}")
            print(f"└{'─'*78}┘\n  Running...", flush=True)

            main_agent.trajectory_dir = traj_dir
            main_agent.name = qid
            if tool_registry._verify_agent is not None:
                tool_registry._verify_agent.trajectory_dir = traj_dir
                tool_registry._verify_agent.name = f"{qid}_verify"

            traj = await main_agent.run(question)
            answer = extract_final_answer(traj) or ""

            # Determine finish_reason from trajectory
            finish_reason = "max_turns"
            for i, msg in enumerate(traj):
                if msg.get("role") == "assistant":
                    for t in (msg.get("tool_calls") or []):
                        if t["function"]["name"] == "submit_answer":
                            for j in range(i + 1, min(i + 3, len(traj))):
                                if traj[j].get("role") == "tool":
                                    try:
                                        fb = json.loads(traj[j].get("content", "{}"))
                                        if fb.get("is_correct"):
                                            finish_reason = "submit_answer_confirmed"
                                    except Exception:
                                        pass
                                    break

            elapsed = time.time() - t0
            rec = {
                "query_id": qid, "status": finish_reason,
                "predicted_answer": answer, "messages": traj,
            }
            records.append(rec)

            # Update trajectory file with predicted answer
            traj_path = Path(traj_dir) / f"{qid}.json"
            if traj_path.exists():
                with open(traj_path) as f:
                    traj_data = json.load(f)
                traj_data["predicted_answer"] = answer
                traj_data["status"] = finish_reason
                with open(traj_path, "w") as f:
                    json.dump(traj_data, f, ensure_ascii=False, indent=2)

            main_agent.reset_state()
            if tool_registry._verify_agent is not None:
                tool_registry._verify_agent.reset_state()

            ans_preview = answer[:200].replace("\n", " ")
            if len(answer) > 200:
                ans_preview += "..."
            print(f"  ┌{'─'*76}┐\n  | ✓ [{i+1}/{total}] qid={qid}  {elapsed:.1f}s\n"
                  f"  ├{'─'*76}┤\n  | status: {finish_reason}\n"
                  f"  └{'─'*76}┘\n", flush=True)

        print(f"[generate] {len(records)}/{total} queries done", flush=True)
    finally:
        searcher.connection.close()
        for c in _client_cache.values():
            await c._client.close()
        if not _client_cache:
            await _global_client._client.close()

    with open(submission_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[generate] saved {len(records)} trajectories -> {submission_path}", flush=True)

    if args.no_eval:
        return

    eval_model = args.eval_model or args.model
    eval_path = str(Path(output_dir) / "eval.jsonl")
    summary, details = await evaluate_trajectories(
        records=records, dataset_path=args.dataset, model=eval_model,
        base_url=args.base_url, api_key=args.api_key,
        eval_batch_size=args.eval_batch_size, temperature=0.0, max_tokens=8192,
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
    _write_jsonl(Path(output_dir) / "eval_correct.json",
                 [d for d in details if d["eval_judgment"] == "CORRECT"])
    _write_jsonl(Path(output_dir) / "eval_incorrect.json",
                 [d for d in details if d["eval_judgment"] == "INCORRECT"])

    print(f"\nSaved: {len(correct_recs)} correct, {len(incorrect_recs)} incorrect -> {output_dir}")


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stderr)
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
