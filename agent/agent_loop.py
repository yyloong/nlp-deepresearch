"""
Deep Research Agent Loop

Async per-slot router — 每个 env slot 独立协程，一问结束立刻补下一问。

一键执行：python -m agent.agent_loop --dataset ... --index-path ... --model ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer

_TOKENIZER_PATH = "Qwen/Qwen3-8B"
_tok: Any = AutoTokenizer.from_pretrained(
    _TOKENIZER_PATH, trust_remote_code=True, local_files_only=True,
)

from .env import DeepResearchEnv
from .eval_async import evaluate_trajectories
from .utils import (
    count_tokens_messages,
    extract_final_answer,
    hard_truncate_tail_tool_messages,
    is_truncated_think_response,
    validate_tool_call,
)
from .vllm_client_async import VLLMClientAsync
from .agent import Agent, DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Verbose serial processing (used when n_envs == 1)
# ═══════════════════════════════════════════════════════════════

def _tok_count(text: str) -> int:
    return len(_tok.encode(text))

def _extract_think_blocks(text: str) -> "List[str]":
    blocks: List[str] = []
    for m in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL):
        blocks.append(m.group(1).strip())
    m = re.search(r"<think>(.*)$", text, re.DOTALL)
    if m:
        blocks.append(m.group(1).strip() + " [UNCLOSED]")
    return blocks

def _log_content_block(label: str, text: str, max_chars: int = 3000) -> None:
    if not text:
        return
    suffix = ""
    if len(text) > max_chars:
        n_tok = _tok_count(text)
        text = text[:max_chars]
        suffix = f"\n... [truncated, {n_tok} tokens total]"
    for line in text.split("\n"):
        print(f"    │ {line}", flush=True)
    if suffix:
        print(f"    │{suffix}", flush=True)

def _log_json_block(label: str, obj: Any) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    for line in text.split("\n"):
        print(f"    │ {line}", flush=True)

async def _process_one_question_verbose(
    env: DeepResearchEnv, agent: Agent, question: str, tools: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    import time as _time
    max_context = agent.max_context
    max_tokens = agent.max_tokens
    max_tool_calls_per_turn = agent.max_tool_calls_per_turn
    obs: List[Dict[str, Any]] = env.reset_slot(0, question)
    finish_reason = "max_turns"
    turn_stats: List[Dict[str, Any]] = []
    total_retries = 0

    for turn in range(env.max_turns):
        t_turn_start = _time.time()
        n_tokens_before = count_tokens_messages(_tok, obs)
        st: Dict[str, Any] = {"turn": turn + 1, "n_tokens_before": n_tokens_before,
                              "think_truncated": False, "retries": 0, "tool_calls": [],
                              "tool_results": [], "condensed": False}

        n_msgs = len(obs)
        print(f"  ┌─ Turn {turn + 1}/{env.max_turns} | msgs: {n_msgs} (sys:{sum(1 for m in obs if m.get('role')=='system')} usr:{sum(1 for m in obs if m.get('role')=='user')} asst:{sum(1 for m in obs if m.get('role')=='assistant')} tool:{sum(1 for m in obs if m.get('role')=='tool')}) | tokens: {n_tokens_before} | {_time.strftime('%H:%M:%S')}", flush=True)

        safe_limit = max_context - max_tokens - 2000
        if n_tokens_before > safe_limit:
            t_cs = _time.time()
            condensed = await agent.condense_context(obs, original_question=question)
            env.set_messages(0, condensed)
            env.append_to_trajectory(0, {"role": "user", "content": condensed[1]['content'], "_condensed": True})
            obs = condensed
            st["condensed"] = True
            n_tokens_before = count_tokens_messages(_tok, obs)
            print(f"  ↻ pre-call condensed → {n_tokens_before} tokens ({_time.time() - t_cs:.1f}s)", flush=True)

        if n_tokens_before > safe_limit:
            print(f"  ⚠ context still {n_tokens_before} > {safe_limit} after condense, forcing early stop", flush=True)
            finish_reason = "context_overflow"
            break

        raw = await agent.client.simple_chat(model=agent.model, messages=obs, temperature=agent.temperature,
                                             max_tokens=agent.max_tokens, tools=tools, tool_choice="auto",
                                             extra_payload=agent.extra_payload)
        resp: Dict[str, Any] = raw["choices"][0]["message"]
        final_raw_content: str = resp.get("content", "") or ""
        raw_content: str = final_raw_content
        raw_tc = resp.get("tool_calls")

        usage = raw.get("usage", {})
        if usage:
            print(f"    │ [model] prompt_tokens={usage.get('prompt_tokens','?')}  completion_tokens={usage.get('completion_tokens','?')}  total={usage.get('total_tokens','?')}", flush=True)

        think_blocks = _extract_think_blocks(raw_content)
        if think_blocks:
            print(f"    │ [think] {len(think_blocks)} block(s), {sum(_tok_count(b) for b in think_blocks)} tokens total", flush=True)
        non_think = raw_content
        if think_blocks:
            non_think = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL)
            non_think = re.sub(r"<think>.*$", "", non_think, flags=re.DOTALL).strip()
        if non_think:
            print(f"    │ [content] ({_tok_count(non_think)} tokens):", flush=True)
            _log_content_block("content", non_think, max_chars=2000)
        elif raw_tc:
            print(f"    │ [content] (tool-call only, no text)", flush=True)

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

        content = raw_content
        tc = raw_tc
        if is_truncated_think_response(content, tc):
            st["think_truncated"] = True
            st["retries"] += 1
            print(f"    ⚠ [think-trunc] content truncated ({_tok_count(content)} tokens)", flush=True)
            resp = await agent.retry_think_truncation(obs, resp, tools, lambda m: env.append_to_trajectory(0, m))
            tc = resp.get("tool_calls")
            content = resp.get("content", "") or ""
            final_raw_content = content
            print(f"    ↪ retry | content: {_tok_count(content)} tokens | tc: {len(tc) if tc else 0}", flush=True)
            if content:
                _log_content_block("retry-content", content, max_chars=1000)

        if tc:
            all_errors_init: List[Dict[str, str]] = []
            for tc_item in tc:
                err = validate_tool_call(tc_item, tools)
                if err:
                    all_errors_init.append({"tool_name": tc_item.get("function", {}).get("name", "?"), "message": err})
            if all_errors_init:
                st["retries"] += 1
                print(f"    ⚠ tool validation failed: {len(all_errors_init)} error(s)", flush=True)
                for e in all_errors_init:
                    print(f"       {e['tool_name']}: {e['message']}", flush=True)
                resp = await agent.retry_tool_validation(obs, resp, tools, lambda m: env.append_to_trajectory(0, m))
                tc = resp.get("tool_calls")
                final_raw_content = resp.get("content", "") or ""

        tc = resp.get("tool_calls")
        if tc and len(tc) > max_tool_calls_per_turn:
            n_truncated = len(tc) - max_tool_calls_per_turn
            resp = agent.enforce_max_tool_calls(resp)
            print(f"    ⚠ tool calls truncated ({n_truncated} dropped)", flush=True)

        tc = resp.get("tool_calls")
        if tc:
            for tc_item in tc:
                fn = tc_item.get("function", {})
                st["tool_calls"].append({"name": fn.get("name", "?"), "args_preview": str(fn.get("arguments", ""))[:120]})

        n_msgs_before_step = len(obs)
        obs, done = await env.step_single(0, resp)
        n_tokens_after = count_tokens_messages(_tok, obs) if obs is not None else n_tokens_before
        st["n_tokens_after"] = n_tokens_after
        st["elapsed"] = _time.time() - t_turn_start

        if obs is None:
            for m in reversed(env._instances[0].trajectory):
                if m.get("role") == "tool" and "is_correct" in (m.get("content", "") or ""):
                    try:
                        fb = json.loads(m["content"])
                        verdict = "CORRECT" if fb.get("is_correct") else "INCORRECT"
                        print(f"    │ [verify] {verdict} — {fb.get('reason','')[:200]}", flush=True)
                        if fb.get("suggestions"):
                            print(f"    │ [verify] suggestions: {fb['suggestions'][:200]}", flush=True)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break
        if obs is not None:
            for nm in obs[n_msgs_before_step:]:
                role = nm.get("role", "")
                if role == "tool":
                    tc_id = nm.get("tool_call_id", "?")
                    tool_content = nm.get("content", "") or ""
                    st["tool_results"].append({"tool_call_id": tc_id, "tokens": _tok_count(tool_content)})
                    try:
                        parsed = json.loads(tool_content)
                        if isinstance(parsed, dict):
                            keys = list(parsed.keys())
                            print(f"    │ [tool_result] id={tc_id}  tokens={_tok_count(tool_content)}  keys={keys}", flush=True)
                            if "error" in parsed:
                                print(f"    │   ERROR: {str(parsed['error'])[:300]}", flush=True)
                    except (json.JSONDecodeError, TypeError):
                        print(f"    │ [tool_result] id={tc_id}  tokens={_tok_count(tool_content)}  (raw text)", flush=True)

        tc_names = [t["name"] for t in st["tool_calls"]]
        tc_str = ", ".join(tc_names) if tc_names else "(none)"
        retry_flag = f" retries:{st['retries']}" if st["retries"] else ""
        cond_flag = " ↻CONDENSED" if st["condensed"] else ""
        print(f"  └─ turn {turn + 1:2d} done | tokens: {n_tokens_before:5d} → {n_tokens_after:5d} (Δ{n_tokens_after - n_tokens_before:+d}) | tools: {tc_str} → {len(st['tool_results'])} results ({sum(tr['tokens'] for tr in st['tool_results'])} tokens){retry_flag}{cond_flag} | {st['elapsed']:.1f}s", flush=True)
        turn_stats.append(st)
        total_retries += st["retries"]

        if done:
            tc_final = resp.get("tool_calls")
            tc_names_final = [t["function"]["name"] for t in tc_final] if tc_final else []
            finish_reason = "submit_answer_confirmed" if "submit_answer" in tc_names_final else "max_turns"
            break

        used = count_tokens_messages(_tok, obs)
        safe_limit = max_context - max_tokens - 2000
        if used > safe_limit * 0.7:
            last = obs[-1] if obs else None
            if last is not None and last.get("role") == "tool":
                hard_truncate_tail_tool_messages(_tok, obs, max_context, label=f"turn {turn + 1}")
                env.sync_trajectory_tool_tail(0)
                used_after = count_tokens_messages(_tok, obs)
                if used_after > max_context // 2:
                    t_cond_start = _time.time()
                    condensed = await agent.condense_context(obs)
                    env.set_messages(0, condensed)
                    env.append_to_trajectory(0, {"role": "user", "content": condensed[1]['content'], "_condensed": True})
                    obs = condensed
                    st["condensed"] = True
                    print(f"    ↻ context condensed ({used_after} → {count_tokens_messages(_tok, obs)} tokens, {_time.time() - t_cond_start:.1f}s)", flush=True)
        print()

    return env.extract_slot_trajectory(0), finish_reason, {
        "total_turns": len(turn_stats), "total_retries": total_retries,
        "n_think_trunc": sum(1 for s in turn_stats if s["think_truncated"]),
        "n_condensed": sum(1 for s in turn_stats if s["condensed"]),
        "final_tokens": turn_stats[-1]["n_tokens_after"] if turn_stats else 0,
        "tool_calls_total": sum(len(s["tool_calls"]) for s in turn_stats),
        "turn_details": turn_stats,
    }


# ═══════════════════════════════════════════════════════════════
# Async per-slot router
# ═══════════════════════════════════════════════════════════════

async def _run_one_question_async(
    slot_id: int, qidx: int, question: str, env: DeepResearchEnv, agent: Agent,
    tools: List[Dict[str, Any]], result_queue: "asyncio.Queue[tuple[int, List[Dict[str, Any]]]]",
    *, done_counter=None, n_total=0, done_lock=None,
) -> None:
    traj = await agent.run_question(env, slot_id, question, tools)
    await result_queue.put((qidx, traj))
    if done_counter is not None and done_lock is not None and n_total > 0:
        async with done_lock:
            done_counter[0] += 1
            cur = done_counter[0]
        if cur % 10 == 0 or cur == n_total:
            print(f"  [router] {cur}/{n_total} queries done", flush=True)


async def run_agent_async_router(
    env: DeepResearchEnv, agent: Agent, questions: List[str],
) -> List[List[Dict[str, Any]]]:
    import asyncio as _asyncio
    n_total = len(questions)
    n_workers = min(env.n_envs, n_total)
    tools = env.tool_specs
    pending: "_asyncio.Queue[tuple[int, str]]" = _asyncio.Queue()
    results: "_asyncio.Queue[tuple[int, List[Dict[str, Any]]]]" = _asyncio.Queue()
    for qidx, q in enumerate(questions):
        await pending.put((qidx, q))
    done_counter: List[int] = [0]
    done_lock = _asyncio.Lock()

    async def _worker(slot_id: int) -> None:
        while True:
            try:
                qidx, q = pending.get_nowait()
            except _asyncio.QueueEmpty:
                return
            await _run_one_question_async(slot_id=slot_id, qidx=qidx, question=q, env=env,
                                          agent=agent, tools=tools, result_queue=results,
                                          done_counter=done_counter, n_total=n_total, done_lock=done_lock)

    workers = [_worker(i) for i in range(n_workers)]
    await _asyncio.gather(*workers)
    result_dict: Dict[int, List[Dict[str, Any]]] = {}
    for _ in range(n_total):
        qidx, traj = await results.get()
        result_dict[qidx] = traj
    return [result_dict[i] for i in range(n_total)]


# ═══════════════════════════════════════════════════════════════
# 批量轨迹生成
# ═══════════════════════════════════════════════════════════════

async def generate_trajectories(
    dataset_path: str, index_path: str, model: str, base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy", output_path: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT, n_envs: int = 4, max_turns: int = 10,
    temperature: float = 0.0, max_tokens: int = 4096, max_context: int = 40960,
    search_k: int = 5, snippet_max_chars: int = 1200,
    extra_payload: Optional[Dict[str, Any]] = None, limit: Optional[int] = None,
    query_ids: Optional[List[str]] = None,
    max_tool_calls_per_turn: int = 1, think_trunc_no_think: bool = False,
) -> List[Dict[str, Any]]:
    from .dataset_utils import load_jsonl
    rows = load_jsonl(dataset_path, limit=limit)
    if query_ids:
        ids = set(query_ids)
        rows = [r for r in rows if r.get("query_id", "") in ids]
        print(f"Filtered to {len(rows)} queries by query_ids: {sorted(ids)}", flush=True)
    total = len(rows)
    client = VLLMClientAsync(base_url=base_url, api_key=api_key, max_concurrent=max(n_envs, 10))

    agent = Agent(client=client, model=model, tokenizer=_tok, max_tokens=max_tokens,
                  temperature=temperature, max_context=max_context, extra_payload=extra_payload,
                  max_tool_calls_per_turn=max_tool_calls_per_turn, think_trunc_no_think=think_trunc_no_think)
    verify_agent = Agent(client=client, model=model, tokenizer=_tok, max_tokens=max_tokens, temperature=temperature)
    sub_agent = Agent(
        client=client, model=model, tokenizer=_tok,
        max_tokens=4096, max_context=32768, temperature=0.0,
    )

    env = DeepResearchEnv(index_path=index_path, n_envs=n_envs, system_prompt=system_prompt,
                          max_turns=max_turns, search_k=search_k, snippet_max_chars=snippet_max_chars,
                          record_trajectory=True, verify_agent=verify_agent, sub_agent=sub_agent)  # sub_agent disabled

    records: List[Dict[str, Any]] = []
    try:
        all_questions = [r["query"] for r in rows]
        all_qids = [r["query_id"] for r in rows]

        if n_envs == 1:
            tools = env.tool_specs
            traj_dir = ""
            if output_path:
                traj_dir = str(Path(output_path).parent / "trajectories")
                Path(traj_dir).mkdir(parents=True, exist_ok=True)
            for i, row in enumerate(rows):
                qid = row["query_id"]
                question = row["query"]
                gold_answer = row.get("answer", "") or row.get("gold_answer", "") or ""
                t0 = time.time()
                q_preview = question[:200].replace("\n", " ")
                if len(question) > 200: q_preview += "..."
                print(f"┌{'─'*78}┐\n│ [{i+1}/{total}]  qid={qid}\n├{'─'*78}┤\n│ Question: {q_preview}")
                if gold_answer:
                    gold_preview = gold_answer[:150].replace("\n", " ")
                    if len(gold_answer) > 150: gold_preview += "..."
                    print(f"│ Gold Ans: {gold_preview}")
                print(f"└{'─'*78}┘\n  Running...", flush=True)

                traj, finish_reason, stats = await _process_one_question_verbose(env=env, agent=agent, question=question, tools=tools)
                answer = extract_final_answer(traj) or ""
                elapsed = time.time() - t0
                rec = {"query_id": qid, "status": finish_reason, "predicted_answer": answer, "messages": traj}
                records.append(rec)

                if traj_dir:
                    with Path(traj_dir, f"{qid}.json").open("w", encoding="utf-8") as f:
                        json.dump(rec, f, ensure_ascii=False, indent=2)
                    # Save condense messages
                    if agent._condense_sessions:
                        with Path(traj_dir, f"{qid}_condense.json").open("w", encoding="utf-8") as f:
                            json.dump({"query_id": qid, "sessions": agent._condense_sessions}, f, ensure_ascii=False, indent=2)
                    # Save verify messages
                    if verify_agent._verify_msgs:
                        with Path(traj_dir, f"{qid}_verify.json").open("w", encoding="utf-8") as f:
                            json.dump({"query_id": qid, "messages": verify_agent._verify_msgs}, f, ensure_ascii=False, indent=2)
                    # Save verify condense messages
                    if verify_agent._condense_sessions:
                        with Path(traj_dir, f"{qid}_verify_condense.json").open("w", encoding="utf-8") as f:
                            json.dump({"query_id": qid, "sessions": verify_agent._condense_sessions}, f, ensure_ascii=False, indent=2)
                    agent.reset_trajectories()
                    verify_agent.reset_trajectories()

                ans_preview = answer[:200].replace("\n", " ")
                if len(answer) > 200: ans_preview += "..."
                print(f"  ┌{'─'*76}┐\n  │ ✓ [{i+1}/{total}] qid={qid}  finished in {elapsed:.1f}s\n  ├{'─'*76}┤\n  │ status:     {finish_reason}\n  │ turns:      {stats['total_turns']}\n  │ retries:    {stats['total_retries']} (think-trunc: {stats['n_think_trunc']}, condensed: {stats['n_condensed']})\n  │ tool calls: {stats['tool_calls_total']}\n  │ tokens:     {stats['final_tokens']}")
                if answer: print(f"  ├{'─'*76}┤\n  │ Answer: {ans_preview}")
                print(f"  └{'─'*76}┘\n", flush=True)
        else:
            trajs = await run_agent_async_router(env=env, agent=agent, questions=all_questions)
            for row, traj in zip(rows, trajs):
                answer = extract_final_answer(traj) or ""
                finish_reason = "unknown"
                for m in reversed(traj):
                    if m.get("role") == "assistant":
                        tc = m.get("tool_calls")
                        finish_reason = "max_turns" if (tc and len(tc) > 0) else "no_tool_calls"
                        break
                records.append({"query_id": row["query_id"], "status": finish_reason, "predicted_answer": answer, "messages": traj})

        print(f"[generate] {len(records)}/{total} queries done", flush=True)
    finally:
        env.close()
        await client._client.close()

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[generate] saved {len(records)} trajectories → {output_path}", flush=True)
    return records


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deep Research Agent — 一键轨迹生成 + 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--index-path", required=True)
    p.add_argument("--model", default="qwen_auto")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-context", type=int, default=40960)
    p.add_argument("--max-tool-calls-per-turn", type=int, default=1)
    p.add_argument("--search-k", type=int, default=5)
    p.add_argument("--snippet-max-chars", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--eval-model", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--think-trunc-no-think", action="store_true", default=False)
    p.add_argument("--tokenizer-path", default="Qwen/Qwen3-8B")
    return p


async def _main_async(args: argparse.Namespace) -> None:
    global _TOKENIZER_PATH, _tok
    if args.tokenizer_path != _TOKENIZER_PATH:
        _TOKENIZER_PATH = args.tokenizer_path
        _tok = AutoTokenizer.from_pretrained(_TOKENIZER_PATH, trust_remote_code=True, local_files_only=True)
    output_dir = Path(args.output_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    submission_path = str(run_dir / "submission.jsonl")
    eval_path = str(run_dir / "eval.jsonl")
    t0 = time.time()
    records = await generate_trajectories(
        dataset_path=args.dataset, index_path=args.index_path, model=args.model,
        base_url=args.base_url, api_key=args.api_key, output_path=submission_path,
        n_envs=args.n_envs, max_turns=args.max_turns, temperature=args.temperature,
        max_tokens=args.max_tokens, max_context=args.max_context,
        search_k=args.search_k, snippet_max_chars=args.snippet_max_chars,
        limit=args.limit, max_tool_calls_per_turn=args.max_tool_calls_per_turn,
        think_trunc_no_think=args.think_trunc_no_think)
    gen_time = time.time() - t0
    print(f"\n[done] generated {len(records)} trajectories in {gen_time:.1f}s", flush=True)
    if args.no_eval: return
    eval_model = args.eval_model or args.model
    t0 = time.time()
    summary, details = await evaluate_trajectories(records=records, dataset_path=args.dataset,
        model=eval_model, base_url=args.base_url, api_key=args.api_key,
        eval_batch_size=args.eval_batch_size, temperature=0.0, max_tokens=8192, output_path=eval_path)
    eval_time = time.time() - t0
    print(f"\n{'='*50}\nEvaluation complete in {eval_time:.1f}s\nAccuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total_queries']})\nAvg tool calls/query: {summary['avg_tool_calls_per_query']}\nAvg retrieved docs/query: {summary['avg_retrieved_docs_per_query']}\n{'='*50}")
    eval_map = {d["query_id"]: d["eval_judgment"] for d in details}
    correct_records = [r for r in records if eval_map.get(r["query_id"]) == "CORRECT"]
    incorrect_records = [r for r in records if eval_map.get(r["query_id"]) == "INCORRECT"]
    for name, recs in [("correct.json", correct_records), ("incorrect.json", incorrect_records),
                        ("eval_correct.json", [d for d in details if d["eval_judgment"]=="CORRECT"]),
                        ("eval_incorrect.json", [d for d in details if d["eval_judgment"]=="INCORRECT"])]:
        with (run_dir / name).open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved: {len(correct_records)} correct, {len(incorrect_records)} incorrect → {run_dir}")


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
