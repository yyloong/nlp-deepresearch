#!/usr/bin/env python3
"""
Serial (single-instance) agent evaluation — processes questions one at a time.

Uses the new refactored architecture:
  - agent.py:      Unified Agent class (all agent types)
  - tool_docs.py:  Tool specifications
  - tool_func.py:  Tool implementations (ToolRegistry)
  - configs/*.yaml: Per-agent-type configuration

Usage:
    DATASET=browsecomp_plus_hard50.jsonl \\
    INDEX_PATH=indexes/browsecomp_plus_bm25.sqlite \\
    python run_serial.py

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
from typing import Any, Callable, Dict, List, Optional

import yaml

# Ensure agent/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.agent import Agent
from agent.browsecomp_searcher import BrowseCompBM25Searcher  # noqa: E402
from agent.dataset_utils import load_jsonl  # noqa: E402
from agent.eval_async import evaluate_trajectories  # noqa: E402
from agent.tool_docs import (  # noqa: E402
    build_condense_tool_specs,
    build_main_agent_tool_specs,
    build_search_agent_tool_specs,
    build_sub_summary_tool_specs,
    build_verify_agent_tool_specs,
)
from agent.tool_func import ToolRegistry  # noqa: E402
from agent.utils import extract_final_answer  # noqa: E402
from agent.vllm_client_async import VLLMClientAsync  # noqa: E402

logger = logging.getLogger(__name__)

# Tokenizer
_TOKENIZER_PATH = "Qwen/Qwen3-8B"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402

_tok: Any = AutoTokenizer.from_pretrained(
    _TOKENIZER_PATH, trust_remote_code=True, local_files_only=True,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Serial agent evaluation — one question at a time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    p.add_argument("--snippet-max-tokens", type=int, default=600)
    p.add_argument("--eval-batch-size", type=int, default=1,
                   help="Eval batch size (default: 1 for serial eval)")
    p.add_argument("--eval-model", default=None, help="Eval model (defaults to --model)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    p.add_argument("--query-ids", type=str, default=None,
                   help="Comma-separated list of query IDs to run (e.g. '442,26,471')")
    p.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    p.add_argument("--eval-only", type=str, default=None,
                   help="Only run eval on an existing submission.jsonl")
    p.add_argument("--no-verify", action="store_true", help="Disable verify agent")
    p.add_argument("--max-context", type=int, default=40960,
                   help="Max context window for auto-condensation")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tokenizer-path", default="Qwen/Qwen3-8B")
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
    if args.tokenizer_path != _TOKENIZER_PATH:
        _TOKENIZER_PATH = args.tokenizer_path
        _tok = AutoTokenizer.from_pretrained(_TOKENIZER_PATH, trust_remote_code=True, local_files_only=True)

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

    # Validate required args
    if not args.dataset:
        sys.exit("ERROR: --dataset is required")
    if not args.index_path:
        sys.exit("ERROR: --index-path is required")

    print(f"=== Serial Agent (refactored) ===")
    print(f"Dataset:    {args.dataset}")
    print(f"Index:      {args.index_path}")
    print(f"Model:      {args.model}")
    print(f"Base URL:   {args.base_url}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"verify:     {not args.no_verify}")
    print(f"Output:     {args.output_dir}/")
    print(f"================================")
    print()

    # ── Load dataset ──
    rows = load_jsonl(args.dataset, limit=args.limit)
    if args.query_ids:
        ids = set(args.query_ids.replace(" ", "").split(","))
        rows = [r for r in rows if r.get("query_id", "") in ids]
        print(f"Filtered to {len(rows)} queries by query_ids: {sorted(ids)}", flush=True)
    total = len(rows)

    # ── Create shared client ──
    client = VLLMClientAsync(base_url=args.base_url, api_key=args.api_key, max_concurrent=10)

    # ── Create searcher ──
    searcher = build_searcher(args.index_path)

    # ── Override YAML config with CLI args ──
    def _load_and_patch(config_path: str) -> str:
        """Load YAML, patch with CLI args, write to temp, return path."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["model"] = args.model
        cfg["max_tokens"] = args.max_tokens
        cfg["max_turn"] = args.max_turns
        cfg["max_context"] = args.max_context
        cfg["temperature"] = args.temperature
        cfg["max_tool_calls_per_turn"] = args.max_tool_calls_per_turn
        if "tool_config" not in cfg:
            cfg["tool_config"] = {}
        for key in ("search", "smart_search"):
            if key in cfg["tool_config"] or key in cfg.get("tools", []):
                cfg["tool_config"].setdefault(key, {})
                cfg["tool_config"][key]["search_k"] = args.search_k
                cfg["tool_config"][key]["snippet_max_tokens"] = args.snippet_max_tokens
        # Write patched config
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
            yaml.dump(cfg, tf)
            return tf.name

    # ── Agent factory for call_subagents (light patch: only model/search_k, keep own max_turn etc.) ──
    def agent_factory(config_path: str) -> Agent:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["model"] = args.model
        cfg["max_tokens"] = args.max_tokens
        cfg["temperature"] = args.temperature
        cfg.setdefault("tool_config", {}).setdefault("search", {})
        cfg["tool_config"]["search"]["search_k"] = args.search_k
        cfg["tool_config"]["search"]["snippet_max_tokens"] = args.snippet_max_tokens
        import uuid, tempfile
        patched_path = os.path.join(tempfile.gettempdir(), f"agent_{uuid.uuid4().hex}.yaml")
        with open(patched_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        return Agent(patched_path, client=client, tokenizer=_tok)

    # ── Create ToolRegistry ──
    tool_registry = ToolRegistry(searcher=searcher, agent_factory=agent_factory)

    # ── Create sub_summary agent (if search uses subagent) ──
    main_cfg = _load_yaml_config(args.agent_config)
    use_subagent = (
        main_cfg.get("tool_config", {}).get("search", {}).get("use_subagent_summary", False)
    )
    # Judge agent's search uses subagent_summary — enable it globally (main uses smart_search, unaffected)
    _judge_use_subagent = bool(_load_yaml_config("configs/relevance_judge_agent.yaml")
                               .get("tool_config", {}).get("search", {}).get("use_subagent_summary", False))
    sub_summary_agent = None
    if use_subagent or _judge_use_subagent:
        sub_summary_agent = agent_factory("configs/sub_summary_agent.yaml")
        sub_summary_agent.tool_registry = tool_registry.build_registry("sub_summary")
        tool_registry.set_sub_summary_agent(sub_summary_agent)

    def _get_enable_verify(default: bool = False) -> bool:
        """Read enable_verify from the actual main agent config being used."""
        cfg = _load_yaml_config(args.agent_config)
        return bool(cfg.get("tool_config", {}).get("submit_answer", {}).get("enable_verify", default))

    # ── Create verify agent ──
    verify_agent = None
    if not args.no_verify:
        verify_patched = _load_and_patch("configs/verify_agent.yaml")
        verify_agent = Agent(verify_patched, client=client, tokenizer=_tok)
        verify_agent.tool_registry = tool_registry.build_registry("verify")
        tool_registry.set_verify_agent(verify_agent)

    # ── Create surrender check agent ──
    surrender_check_patched = _load_and_patch("configs/surrender_check_agent.yaml")
    surrender_check_agent = Agent(surrender_check_patched, client=client, tokenizer=_tok)
    surrender_check_agent.tool_registry = tool_registry.build_registry("surrender_check")
    tool_registry.set_surrender_check_agent(surrender_check_agent)

    # ── Create relevance judge agent ──
    relevance_judge_agent = agent_factory("configs/relevance_judge_agent.yaml")
    relevance_judge_agent.tool_registry = tool_registry.build_registry("relevance_judge")
    tool_registry.set_relevance_judge_agent(relevance_judge_agent)

    # ── Configure search (YAML tool_config value takes priority) ──
    _yaml_search_k = int(main_cfg.get("tool_config", {}).get("smart_search", {}).get("search_k",
                         main_cfg.get("tool_config", {}).get("search", {}).get("search_k", 5)))
    tool_registry.configure_search(
        search_k=_yaml_search_k,
        snippet_max_tokens=args.snippet_max_tokens,
        use_subagent_summary=_judge_use_subagent or use_subagent,
    )

    # ── Create main agent ──
    main_patched = _load_and_patch(args.agent_config)
    main_agent = Agent(main_patched, client=client, tokenizer=_tok)
    main_agent.tool_registry = tool_registry.build_registry("main", enable_verify=_get_enable_verify(True))
    tool_registry.set_main_agent(main_agent)

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
            question = row["query"]
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

            # Set trajectory dir for this question (agent saves internally)
            main_agent.trajectory_dir = traj_dir
            main_agent.name = qid
            if verify_agent is not None:
                verify_agent.trajectory_dir = traj_dir
                verify_agent.name = f"{qid}_verify"

            # Run main agent
            traj = await main_agent.run(question)
            answer = extract_final_answer(traj) or ""

            # Determine finish reason and first submission
            finish_reason = "max_turns"
            first_submit_answer = ""
            last_submit_was_accepted = False
            for i, msg in enumerate(traj):
                if msg.get("role") == "assistant":
                    for t in (msg.get("tool_calls") or []):
                        if t["function"]["name"] == "submit_answer":
                            if not first_submit_answer:
                                try:
                                    args = json.loads(t["function"].get("arguments", "{}"))
                                    first_submit_answer = args.get("answer", "")
                                except Exception:
                                    pass
                            # Check tool result for acceptance
                            for j in range(i+1, min(i+3, len(traj))):
                                tm = traj[j]
                                if tm.get("role") == "tool":
                                    try:
                                        fb = json.loads(tm.get("content", "{}"))
                                        if fb.get("is_correct"):
                                            last_submit_was_accepted = True
                                    except Exception:
                                        pass
            if last_submit_was_accepted:
                finish_reason = "submit_answer_confirmed"

            # If max_turns, use first submission (not the desperate last one)
            if finish_reason == "max_turns" and first_submit_answer:
                answer = first_submit_answer

            elapsed = time.time() - t0
            rec = {
                "query_id": qid,
                "status": finish_reason,
                "predicted_answer": answer,
                "messages": traj,
            }
            records.append(rec)

            # Reset agent state for next question
            main_agent.reset_state()
            if verify_agent is not None:
                verify_agent.reset_state()

            ans_preview = answer[:200].replace("\n", " ")
            if len(answer) > 200:
                ans_preview += "..."
            print(
                f"  ┌{'─'*76}┐\n"
                f"  | ✓ [{i+1}/{total}] qid={qid}  finished in {elapsed:.1f}s\n"
                f"  ├{'─'*76}┤\n"
                f"  | status:     {finish_reason}\n"
                f"  └{'─'*76}┘\n",
                flush=True,
            )

        print(f"[generate] {len(records)}/{total} queries done", flush=True)
    finally:
        searcher.connection.close()
        await client._client.close()

    # ── Save submission ──
    with open(submission_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[generate] saved {len(records)} trajectories -> {submission_path}", flush=True)

    # ── Eval ──
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
    eval_correct = [d for d in details if d["eval_judgment"] == "CORRECT"]
    eval_incorrect = [d for d in details if d["eval_judgment"] == "INCORRECT"]
    _write_jsonl(Path(output_dir) / "eval_correct.json", eval_correct)
    _write_jsonl(Path(output_dir) / "eval_incorrect.json", eval_incorrect)

    print(f"\nSaved: {len(correct_recs)} correct, {len(incorrect_recs)} incorrect -> {output_dir}")


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
