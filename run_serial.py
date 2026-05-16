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
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402

# ── Reuse existing infra (no modification to original code) ──
from agent.env import DeepResearchEnv  # noqa: E402
from agent.vllm_client_async import VLLMClientAsync  # noqa: E402
from agent.eval_async import evaluate_trajectories  # noqa: E402
from agent.utils import (  # noqa: E402
    RETRY_NUDGE,
    count_tokens_messages,
    extract_final_answer,
    hard_truncate_tail_tool_messages,
    is_truncated_think_response,
    validate_tool_call,
)
from agent.agent_loop import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    _condense_context,
)

logger = logging.getLogger(__name__)

_TOKENIZER_PATH = "Qwen/Qwen3-8B"
_tok: Any = AutoTokenizer.from_pretrained(
    _TOKENIZER_PATH, trust_remote_code=True, local_files_only=True,
)
MAX_TOOL_RETRIES = 2


# ═══════════════════════════════════════════════════════════════
# Core: process a single question serially
# ═══════════════════════════════════════════════════════════════

async def process_one_question(
    env: DeepResearchEnv,
    client: VLLMClientAsync,
    model: str,
    question: str,
    tools: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    max_context: int,
    extra_payload: Optional[Dict[str, Any]],
    max_tool_calls_per_turn: int = 1,
    think_trunc_no_think: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """Run one question to completion on slot 0, returning (trajectory, finish_reason)."""
    obs: List[Dict[str, Any]] = env.reset_slot(0, question)
    finish_reason = "max_turns"  # default if loop exhausts without break

    for turn in range(env.max_turns):
        # ── 1. Model call ──
        raw = await client.simple_chat(
            model=model,
            messages=obs,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto",
            extra_payload=extra_payload,
        )
        resp: Dict[str, Any] = raw["choices"][0]["message"]

        # ── 2. Think-truncation retry ──
        content = resp.get("content", "") or ""
        tc = resp.get("tool_calls")
        if is_truncated_think_response(content, tc):
            # Record the truncated response in trajectory
            env.append_to_trajectory(0, resp)

            if think_trunc_no_think:
                # Stage 1: retry with thinking DISABLED (avoids the repetition loop)
                no_think_extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
                msgs_no_think = list(obs) + [resp]
                raw = await client.simple_chat(
                    model=model, messages=msgs_no_think,
                    temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice="auto", extra_payload=no_think_extra,
                )
                resp = raw["choices"][0]["message"]
                tc = resp.get("tool_calls")
                print(f"    [retry-think] truncated → retrying with thinking disabled", flush=True)

            if not tc:
                # Stage 2 (or direct fallback): RETRY_NUDGE
                if think_trunc_no_think:
                    env.append_to_trajectory(0, resp)
                env.append_to_trajectory(0, {"role": "user", "content": RETRY_NUDGE})
                msgs = list(obs) + [resp, {"role": "user", "content": RETRY_NUDGE}]
                raw = await client.simple_chat(
                    model=model, messages=msgs,
                    temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice="auto", extra_payload=extra_payload,
                )
                resp = raw["choices"][0]["message"]
                tc = resp.get("tool_calls")
                tag = "no-think retry also failed → " if think_trunc_no_think else ""
                print(f"    [retry-think] {tag}RETRY_NUDGE", flush=True)

        # ── 3. Tool-call validation retry ──
        if tc:
            for retry_num in range(MAX_TOOL_RETRIES):
                all_errors: List[Dict[str, str]] = []
                for tc_item in tc:
                    err = validate_tool_call(tc_item, tools)
                    if err:
                        all_errors.append({
                            "tool_name": tc_item.get("function", {}).get("name", "?"),
                            "message": err,
                        })
                if not all_errors:
                    break

                # Record the failed response + error nudge in trajectory
                env.append_to_trajectory(0, resp)
                msgs = list(obs) + [resp]
                error_lines = [f"- `{e['tool_name']}`: {e['message']}" for e in all_errors]
                nudge = (
                    "Your tool call(s) failed validation:\n\n"
                    + "\n".join(error_lines)
                    + "\n\nPlease correct the error(s) and try again."
                )
                msgs.append({"role": "user", "content": nudge})
                env.append_to_trajectory(0, {"role": "user", "content": nudge})
                print(
                    f"    [retry-tool] attempt {retry_num + 1}/{MAX_TOOL_RETRIES}: "
                    f"{len(all_errors)} error(s)",
                    flush=True,
                )
                raw = await client.simple_chat(
                    model=model, messages=msgs,
                    temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice="auto", extra_payload=extra_payload,
                )
                resp = raw["choices"][0]["message"]
                tc = resp.get("tool_calls")
                if not tc:
                    break

        # ── 4. Enforce max tool calls per turn ──
        tc = resp.get("tool_calls")
        if tc and len(tc) > max_tool_calls_per_turn:
            resp["tool_calls"] = tc[:max_tool_calls_per_turn]

        # ── 5. Execute tools via env.step_single ──
        obs, done = env.step_single(0, resp)
        if done:
            # Determine finish reason from the final response
            tc_final = resp.get("tool_calls")
            finish_reason = "max_turns" if (tc_final and len(tc_final) > 0) else "no_tool_calls"
            break

        # ── 6. Context condensation ──
        used = count_tokens_messages(_tok, obs)
        if used > max_context // 2:
            last = obs[-1] if obs else None
            if last is not None and last.get("role") == "tool":
                hard_truncate_tail_tool_messages(
                    _tok, obs, max_context, label=f"turn {turn + 1}",
                )
                # Sync trajectory tool messages with truncated versions
                env.sync_trajectory_tool_tail(0)
                used_after = count_tokens_messages(_tok, obs)
                if used_after > max_context // 2:
                    condensed = await _condense_context(
                        _tok, obs, client, model, temperature,
                        max_tokens, max_context, extra_payload,
                    )
                    env.set_messages(0, condensed)
                    env.replace_trajectory(0, condensed)
                    obs = condensed

    return env.extract_slot_trajectory(0), finish_reason


# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════

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
    p.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    # ── Serial-only 或额外参数 ──
    p.add_argument("--max-context", type=int, default=40960,
                   help="Max context window for auto-condensation")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-strip-thinking", action="store_true",
                   help="Keep <think> blocks in context")
    p.add_argument("--no-condense-thinking", action="store_true",
                   help="Disable <think> condensation (default: on, matching run_agent.sh)")
    p.add_argument("--no-think", action="store_true",
                   default=os.environ.get("NO_THINK", "0") not in ("0", "false", ""),
                   help="Disable model thinking mode (env: NO_THINK)")
    p.add_argument("--think-trunc-no-think", action="store_true", default=False,
                   help="When think-block is truncated, first retry with thinking disabled "
                        "before falling back to RETRY_NUDGE")
    return p


async def _main_async(args: argparse.Namespace) -> None:
    from agent.dataset_utils import load_jsonl

    # Validate required args (may come from env vars)
    if not args.dataset:
        sys.exit("ERROR: --dataset is required (set DATASET env var or pass --dataset)")
    if not args.index_path:
        sys.exit("ERROR: --index-path is required (set INDEX_PATH env var or pass --index-path)")

    rows = load_jsonl(args.dataset, limit=args.limit)
    total = len(rows)
    condense_thinking = not args.no_condense_thinking

    print(f"Loaded {total} queries from {args.dataset}", flush=True)
    print(f"=== Serial Agent (n_envs=1) ===")
    print(f"Dataset:    {args.dataset}")
    print(f"Index:      {args.index_path}")
    print(f"Model:      {args.model}")
    print(f"Base URL:   {args.base_url}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"max_tool_calls: {args.max_tool_calls_per_turn}")
    print(f"condense_thinking: {condense_thinking}")
    print(f"no_think:   {args.no_think}")
    print(f"Output:     {args.output_dir}/")
    print(f"===============================")
    print()

    # ── Setup ──
    client = VLLMClientAsync(
        base_url=args.base_url, api_key=args.api_key, max_concurrent=1,
    )
    env = DeepResearchEnv(
        index_path=args.index_path,
        n_envs=1,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_turns=args.max_turns,
        search_k=args.search_k,
        snippet_max_chars=args.snippet_max_chars,
        record_trajectory=True,
        strip_thinking=not args.no_strip_thinking,
        condense_thinking=condense_thinking,
    )
    tools = env.tool_specs

    extra_payload: Optional[Dict[str, Any]] = None
    if args.no_think:
        extra_payload = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    records: List[Dict[str, Any]] = []

    # ── Serial processing ──
    t_start = time.time()
    try:
        for i, row in enumerate(rows):
            qid = row["query_id"]
            question = row["query"]
            t0 = time.time()

            traj, finish_reason = await process_one_question(
                env=env,
                client=client,
                model=args.model,
                question=question,
                tools=tools,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_context=args.max_context,
                extra_payload=extra_payload,
                max_tool_calls_per_turn=args.max_tool_calls_per_turn,
                think_trunc_no_think=args.think_trunc_no_think,
            )

            answer = extract_final_answer(traj) or ""
            records.append({
                "query_id": qid,
                "status": finish_reason,
                "predicted_answer": answer,
                "messages": traj,
            })

            elapsed = time.time() - t0
            print(
                f"  [{i + 1}/{total}] qid={qid}  "
                f"turns={sum(1 for m in traj if m.get('role')=='assistant')}  "
                f"{elapsed:.1f}s",
                flush=True,
            )
    finally:
        env.close()
        await client._client.close()

    gen_time = time.time() - t_start
    print(f"\n[done] {len(records)} queries in {gen_time:.1f}s", flush=True)

    # ── Save ──
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = str(run_dir / "submission.jsonl")
    with open(submission_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[save] {submission_path}", flush=True)

    if args.no_eval:
        return

    # ── Evaluate ──
    eval_model = args.eval_model or args.model
    t0 = time.time()
    eval_path = str(run_dir / "eval.jsonl")
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
    eval_time = time.time() - t0

    print(f"\n{'=' * 50}")
    print(f"Evaluation in {eval_time:.1f}s")
    print(f"Accuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total_queries']})")
    print(f"Avg tool calls/query: {summary['avg_tool_calls_per_query']}")
    print(f"Avg retrieved docs/query: {summary['avg_retrieved_docs_per_query']}")
    print(f"{'=' * 50}")

    # ── Split correct / incorrect ──
    eval_map = {d["query_id"]: d["eval_judgment"] for d in details}
    correct_recs = [r for r in records if eval_map.get(r["query_id"]) == "CORRECT"]
    incorrect_recs = [r for r in records if eval_map.get(r["query_id"]) == "INCORRECT"]

    def _write_jsonl(path: Path, recs: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_jsonl(run_dir / "correct.json", correct_recs)
    _write_jsonl(run_dir / "incorrect.json", incorrect_recs)

    eval_correct = [d for d in details if d["eval_judgment"] == "CORRECT"]
    eval_incorrect = [d for d in details if d["eval_judgment"] == "INCORRECT"]
    _write_jsonl(run_dir / "eval_correct.json", eval_correct)
    _write_jsonl(run_dir / "eval_incorrect.json", eval_incorrect)

    print(f"\nSaved: {len(correct_recs)} correct, {len(incorrect_recs)} incorrect → {run_dir}")


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
