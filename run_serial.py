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
# Content logging helpers
# ═══════════════════════════════════════════════════════════════

def _tok_count(text: str) -> int:
    """Count tokens in a string using the module-level tokenizer."""
    return len(_tok.encode(text))


def _extract_think_blocks(text: str) -> "List[str]":
    """Extract ``<think>...</think>`` block contents from text."""
    import re
    blocks: List[str] = []
    for m in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL):
        blocks.append(m.group(1).strip())
    # Also catch unclosed <think>
    m = re.search(r"<think>(.*)$", text, re.DOTALL)
    if m:
        blocks.append(m.group(1).strip() + " [UNCLOSED]")
    return blocks


def _log_content_block(label: str, text: str, max_chars: int = 3000) -> None:
    """Print a labelled content block with word-wrap indentation."""
    if not text:
        return
    suffix = ""
    if len(text) > max_chars:
        n_tok = _tok_count(text)
        text = text[:max_chars]
        suffix = f"\n... [truncated, {n_tok} tokens total]"
    # Indent every line
    for line in text.split("\n"):
        print(f"    │ {line}", flush=True)
    if suffix:
        print(f"    │{suffix}", flush=True)


def _log_json_block(label: str, obj: Any) -> None:
    """Print a labelled JSON block."""
    import json as _json
    text = _json.dumps(obj, ensure_ascii=False, indent=2)
    for line in text.split("\n"):
        print(f"    │ {line}", flush=True)


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
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Run one question to completion on slot 0.

    Returns (trajectory, finish_reason, stats) where stats is a dict with
    per-turn metrics for logging.
    """
    obs: List[Dict[str, Any]] = env.reset_slot(0, question)
    finish_reason = "max_turns"  # default if loop exhausts without break
    turn_stats: List[Dict[str, Any]] = []
    total_retries = 0

    # One-time debug: show think-processing mode
    mode = "condense" if env.condense_thinking else ("strip" if env.strip_thinking else "keep-as-is")
    print(f"  [think-mode] condense_thinking={env.condense_thinking} strip_thinking={env.strip_thinking} → {mode}", flush=True)

    for turn in range(env.max_turns):
        t_turn_start = time.time()
        n_tokens_before = count_tokens_messages(_tok, obs)
        st: Dict[str, Any] = {
            "turn": turn + 1,
            "n_tokens_before": n_tokens_before,
            "think_truncated": False,
            "retries": 0,
            "tool_calls": [],
            "tool_results": [],
            "condensed": False,
        }

        # ── Turn header ──
        n_msgs = len(obs)
        n_system = sum(1 for m in obs if m.get("role") == "system")
        n_user = sum(1 for m in obs if m.get("role") == "user")
        n_assistant = sum(1 for m in obs if m.get("role") == "assistant")
        n_tool = sum(1 for m in obs if m.get("role") == "tool")
        print(
            f"  ┌─ Turn {turn + 1}/{env.max_turns} "
            f"| msgs: {n_msgs} (sys:{n_system} usr:{n_user} asst:{n_assistant} tool:{n_tool}) "
            f"| tokens: {n_tokens_before} "
            f"| {time.strftime('%H:%M:%S')}",
            flush=True,
        )

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
        final_raw_content: str = resp.get("content", "") or ""
        raw_content: str = final_raw_content  # original (may differ after retries)
        raw_tc = resp.get("tool_calls")

        # ── Log raw model response ──
        usage = raw.get("usage", {})
        if usage:
            print(
                f"    │ [model] prompt_tokens={usage.get('prompt_tokens', '?')}  "
                f"completion_tokens={usage.get('completion_tokens', '?')}  "
                f"total={usage.get('total_tokens', '?')}",
                flush=True,
            )

        # Show think blocks summary
        think_blocks = _extract_think_blocks(raw_content)
        if think_blocks:
            total_think_tokens = sum(_tok_count(b) for b in think_blocks)
            print(
                f"    │ [think] {len(think_blocks)} block(s), "
                f"{total_think_tokens} tokens total",
                flush=True,
            )

        # Show non-think content
        non_think = raw_content
        if think_blocks:
            import re
            non_think = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL)
            non_think = re.sub(r"<think>.*$", "", non_think, flags=re.DOTALL).strip()
        if non_think:
            print(f"    │ [content] ({_tok_count(non_think)} tokens):", flush=True)
            _log_content_block("content", non_think, max_chars=2000)
        elif raw_tc:
            print(f"    │ [content] (tool-call only, no text)", flush=True)

        # Show tool calls
        if raw_tc:
            print(f"    │ [tool_calls] {len(raw_tc)} call(s):", flush=True)
            for tci, tc_item in enumerate(raw_tc):
                fn = tc_item.get("function", {})
                name = fn.get("name", "?")
                try:
                    args_parsed = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args_parsed = fn.get("arguments", "{}")
                print(f"    │   [{tci}] {name}", flush=True)
                _log_json_block("args", args_parsed)

        # ── 2. Think-truncation retry ──
        content = raw_content
        tc = raw_tc
        if is_truncated_think_response(content, tc):
            st["think_truncated"] = True
            if content:
                resp["content"] = content[:len(content) // 10] + "\n...[THINK_TRUNCATED]\n</think>"
            env.append_to_trajectory(0, resp)
            print(f"    ⚠ [think-trunc] content truncated ({_tok_count(content)} → {_tok_count(resp['content'])} tokens)", flush=True)

            if think_trunc_no_think:
                st["retries"] += 1
                no_think_extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
                msgs_no_think = list(obs) + [resp]
                raw = await client.simple_chat(
                    model=model, messages=msgs_no_think,
                    temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice="auto", extra_payload=no_think_extra,
                )
                resp = raw["choices"][0]["message"]
                tc = resp.get("tool_calls")
                content = resp.get("content", "") or ""
                final_raw_content = content
                print(f"    ↪ retry (no-think) | content: {_tok_count(content)} tokens | tc: {len(tc) if tc else 0}", flush=True)
                if content:
                    _log_content_block("retry-content", content, max_chars=1000)

            if not tc:
                st["retries"] += 1
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
                content = resp.get("content", "") or ""
                final_raw_content = content
                tag = "no-think→" if think_trunc_no_think else ""
                print(f"    ↪ retry ({tag}RETRY_NUDGE) | content: {_tok_count(content)} tokens | tc: {len(tc) if tc else 0}", flush=True)
                if content:
                    _log_content_block("retry-content", content, max_chars=1000)

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

                st["retries"] += 1
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
                    f"    ⚠ tool validation failed (attempt {retry_num + 1}/{MAX_TOOL_RETRIES}): "
                    f"{len(all_errors)} error(s)",
                    flush=True,
                )
                for e in all_errors:
                    print(f"       {e['tool_name']}: {e['message']}", flush=True)
                raw = await client.simple_chat(
                    model=model, messages=msgs,
                    temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice="auto", extra_payload=extra_payload,
                )
                resp = raw["choices"][0]["message"]
                tc = resp.get("tool_calls")
                final_raw_content = resp.get("content", "") or ""
                if not tc:
                    break

        # ── 4. Enforce max tool calls per turn ──
        tc = resp.get("tool_calls")
        if tc and len(tc) > max_tool_calls_per_turn:
            n_truncated = len(tc) - max_tool_calls_per_turn
            resp["tool_calls"] = tc[:max_tool_calls_per_turn]
            print(f"    ⚠ tool calls truncated ({n_truncated} dropped)", flush=True)

        # Record tool call info for stats
        tc = resp.get("tool_calls")
        if tc:
            for tc_item in tc:
                fn = tc_item.get("function", {})
                st["tool_calls"].append({
                    "name": fn.get("name", "?"),
                    "args_preview": str(fn.get("arguments", ""))[:120],
                })

        # ── 5. Execute tools via env.step_single ──
        # Capture obs length before step to find new tool-result messages
        n_msgs_before_step = len(obs)
        obs, done = env.step_single(0, resp)
        n_tokens_after = count_tokens_messages(_tok, obs) if obs is not None else n_tokens_before
        st["n_tokens_after"] = n_tokens_after
        st["elapsed"] = time.time() - t_turn_start

        # ── Extract and log: condensed assistant content & tool results ──
        # When done=True, step_single returns obs=None (no new messages to show)
        if obs is not None:
            new_msgs = obs[n_msgs_before_step:]
            for nm in new_msgs:
                role = nm.get("role", "")
                if role == "assistant":
                    processed_content = nm.get("content", "") or ""
                    # Only report condensation/strip when the env actually modified the content
                    if env.condense_thinking and final_raw_content and final_raw_content != processed_content:
                        # Condensation mode: think blocks replaced with [Plan] summaries
                        final_think_blocks = _extract_think_blocks(final_raw_content)
                        raw_think_tokens = sum(_tok_count(b) for b in final_think_blocks)
                        if raw_think_tokens > 0:
                            import re
                            final_no_think = re.sub(r"<think>.*?</think>", "", final_raw_content, flags=re.DOTALL)
                            final_no_think = re.sub(r"<think>.*$", "", final_no_think, flags=re.DOTALL)
                            processed_think_tokens = max(_tok_count(processed_content) - _tok_count(final_no_think), 0)
                            print(
                                f"    │ [condensed] think: {raw_think_tokens} → ~{processed_think_tokens} tokens "
                                f"({processed_think_tokens / raw_think_tokens * 100:.0f}%)",
                                flush=True,
                            )
                    elif env.strip_thinking and final_raw_content and final_raw_content != processed_content:
                        # Strip mode: think blocks removed entirely
                        print(
                            f"    │ [stripped] think removed "
                            f"({_tok_count(final_raw_content)} → {_tok_count(processed_content)} tokens)",
                            flush=True,
                        )
                elif role == "tool":
                    tc_id = nm.get("tool_call_id", "?")
                    tool_content = nm.get("content", "") or ""
                    st["tool_results"].append({
                        "tool_call_id": tc_id,
                        "tokens": _tok_count(tool_content),
                    })
                    # Try to parse and display structured
                    try:
                        parsed = json.loads(tool_content)
                        if isinstance(parsed, dict):
                            keys = list(parsed.keys())
                            n_results = len(parsed.get("results", parsed.get("documents", [])))
                            if isinstance(parsed.get("results"), list):
                                n_results = len(parsed["results"])
                            print(
                                f"    │ [tool_result] id={tc_id}  "
                                f"tokens={_tok_count(tool_content)}  keys={keys}  "
                                f"results={n_results}",
                                flush=True,
                            )
                            # Show first result snippet
                            results = parsed.get("results", [])
                            if results and isinstance(results, list) and len(results) > 0:
                                r0 = results[0]
                                if isinstance(r0, dict):
                                    snippet = str(r0.get("snippet", r0.get("content", str(r0)[:300])))
                                    print(f"    │   [0]: {snippet[:200]}", flush=True)
                                else:
                                    print(f"    │   [0]: {str(r0)[:200]}", flush=True)
                            elif "error" in parsed:
                                print(f"    │   ERROR: {str(parsed['error'])[:300]}", flush=True)
                        else:
                            print(
                                f"    │ [tool_result] id={tc_id}  "
                                f"tokens={_tok_count(tool_content)}  type={type(parsed).__name__}",
                                flush=True,
                            )
                            _log_content_block("result", str(parsed)[:500], max_chars=500)
                    except (json.JSONDecodeError, TypeError):
                        print(
                            f"    │ [tool_result] id={tc_id}  "
                            f"tokens={_tok_count(tool_content)}  (raw text)",
                            flush=True,
                        )
                        _log_content_block("result", tool_content, max_chars=300)

        # ── Per-turn summary line ──
        tc_names = [t["name"] for t in st["tool_calls"]]
        tc_str = ", ".join(tc_names) if tc_names else "(none)"
        think_flag = "⚡THINK-TRUNC " if st["think_truncated"] else ""
        retry_flag = f" retries:{st['retries']}" if st["retries"] else ""
        cond_flag = " ↻CONDENSED" if st["condensed"] else ""
        n_tool_results = len(st["tool_results"])
        total_tool_result_tokens = sum(tr["tokens"] for tr in st["tool_results"])
        print(
            f"  └─ turn {turn + 1:2d} done | tokens: {n_tokens_before:5d} → {n_tokens_after:5d} "
            f"(Δ{n_tokens_after - n_tokens_before:+d}) | "
            f"tools: {tc_str} → {n_tool_results} results ({total_tool_result_tokens} tokens)"
            f"{retry_flag}{cond_flag} "
            f"| {st['elapsed']:.1f}s",
            flush=True,
        )

        turn_stats.append(st)
        total_retries += st["retries"]

        if done:
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
                env.sync_trajectory_tool_tail(0)
                used_after = count_tokens_messages(_tok, obs)
                if used_after > max_context // 2:
                    t_cond_start = time.time()
                    condensed = await _condense_context(
                        _tok, obs, client, model, temperature,
                        max_tokens, max_context, extra_payload,
                    )
                    env.set_messages(0, condensed)
                    env.replace_trajectory(0, condensed)
                    obs = condensed
                    st["condensed"] = True
                    print(
                        f"    ↻ context condensed ({used_after} → "
                        f"{count_tokens_messages(_tok, obs)} tokens, "
                        f"{time.time() - t_cond_start:.1f}s)",
                        flush=True,
                    )
        print()  # blank line between turns

    # Compile aggregate stats
    agg_stats = {
        "total_turns": len(turn_stats),
        "total_retries": total_retries,
        "n_think_trunc": sum(1 for s in turn_stats if s["think_truncated"]),
        "n_condensed": sum(1 for s in turn_stats if s["condensed"]),
        "final_tokens": turn_stats[-1]["n_tokens_after"] if turn_stats else 0,
        "tool_calls_total": sum(len(s["tool_calls"]) for s in turn_stats),
        "turn_details": turn_stats,
    }

    return env.extract_slot_trajectory(0), finish_reason, agg_stats


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
    p.add_argument("--query-ids", type=str, default=None,
                   help="Comma-separated list of query IDs to run (e.g. '442,26,471')")
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
    if args.query_ids:
        ids = set(args.query_ids.replace(" ", "").split(","))
        rows = [r for r in rows if r.get("query_id", "") in ids]
        print(f"Filtered to {len(rows)} queries by --query-ids: {sorted(ids)}", flush=True)
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

    # ── Create output directory upfront for incremental saves ──
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = run_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    submission_path = str(run_dir / "submission.jsonl")
    print(f"Output dir: {run_dir}/", flush=True)
    print()

    # ── Serial processing ──
    t_start = time.time()
    try:
        for i, row in enumerate(rows):
            qid = row["query_id"]
            question = row["query"]
            gold_answer = row.get("answer", "") or row.get("gold_answer", "") or ""
            t0 = time.time()

            # ── Per-sample header ──
            q_preview = question[:200].replace("\n", " ")
            if len(question) > 200:
                q_preview += "..."
            gold_preview = gold_answer[:150].replace("\n", " ")
            if len(gold_answer) > 150:
                gold_preview += "..."

            print(f"┌{'─' * 78}┐")
            print(f"│ [{i + 1}/{total}]  qid={qid}")
            print(f"├{'─' * 78}┤")
            print(f"│ Question: {q_preview}")
            if gold_preview:
                print(f"│ Gold Ans: {gold_preview}")
            print(f"└{'─' * 78}┘")
            print(f"  Running...", flush=True)

            traj, finish_reason, stats = await process_one_question(
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
            elapsed = time.time() - t0

            rec = {
                "query_id": qid,
                "status": finish_reason,
                "predicted_answer": answer,
                "messages": traj,
            }
            records.append(rec)

            # ── Per-sample summary ──
            ans_preview = answer[:200].replace("\n", " ")
            if len(answer) > 200:
                ans_preview += "..."

            print(f"  ┌{'─' * 76}┐")
            print(f"  │ ✓ [{i + 1}/{total}] qid={qid}  finished in {elapsed:.1f}s")
            print(f"  ├{'─' * 76}┤")
            print(f"  │ status:     {finish_reason}")
            print(f"  │ turns:      {stats['total_turns']}")
            print(f"  │ retries:    {stats['total_retries']} "
                  f"(think-trunc: {stats['n_think_trunc']}, "
                  f"condensed: {stats['n_condensed']})")
            print(f"  │ tool calls: {stats['tool_calls_total']}")
            print(f"  │ tokens:     {stats['final_tokens']}")
            if answer:
                print(f"  ├{'─' * 76}┤")
                print(f"  │ Answer: {ans_preview}")
            print(f"  └{'─' * 76}┘")
            print(flush=True)

            # ── Incremental save: individual trajectory ──
            traj_path = traj_dir / f"{qid}.json"
            with traj_path.open("w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)

            # ── Incremental save: append to submission.jsonl ──
            with open(submission_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    finally:
        env.close()
        await client._client.close()

    gen_time = time.time() - t_start

    # ── Final summary ──
    n_correct = sum(1 for r in records if r["status"] == "no_tool_calls")
    n_max_turns = sum(1 for r in records if r["status"] == "max_turns")
    total_turns = sum(
        sum(1 for m in r["messages"] if m.get("role") == "assistant")
        for r in records
    )
    print(f"\n{'=' * 80}")
    print(f"  Done: {len(records)} queries in {gen_time:.1f}s "
          f"({gen_time / max(len(records), 1):.1f}s avg)")
    print(f"  Stopped by no_tool_calls: {n_correct}  |  max_turns: {n_max_turns}")
    print(f"  Total assistant turns: {total_turns}")
    print(f"  Saved: {submission_path}")
    print(f"  Trajectories: {traj_dir}/")
    print(f"{'=' * 80}")
    print(flush=True)

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
