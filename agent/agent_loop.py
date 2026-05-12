"""
Deep Research Agent Loop

支持两种模式：
1. 经典模式：run_agent_loop / run_agent — 内部管理模型调用和工具执行。
2. 环境模式：run_agent_with_env — 使用 DeepResearchEnv 解耦模型推理和工具执行，
   适配 RL 训练框架。

一键执行：python -m agent.agent_loop --dataset ... --index-path ... --model ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .browsecomp_searcher import BrowseCompBM25Searcher
from .env import DeepResearchEnv
from .tools import build_searcher, get_agent_tool_specs_and_registry
from .vllm_client_async import VLLMClientAsync

logger = logging.getLogger(__name__)

# ── Tokenizer (lazy load, shared across calls) ──
_tokenizer: Any = None


def _get_tokenizer(model_path: str = "Qwen/Qwen3-8B") -> Any:
    global _tokenizer
    if _tokenizer is None:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
    return _tokenizer


def count_tokens(obj: Union[str, List[Dict[str, Any]]], model_path: str = "Qwen/Qwen3-8B") -> int:
    """Count tokens using the actual tokenizer."""
    tok = _get_tokenizer(model_path)
    if isinstance(obj, list):
        text = tok.apply_chat_template(obj, tokenize=False, add_generation_prompt=False)
    else:
        text = obj
    return len(tok.encode(text))

DEFAULT_SYSTEM_PROMPT = """\
You are a Deep Research Agent. Your task is to find the correct answer to a complex \
question by thoroughly searching a document corpus. You MUST conduct a multi-round \
investigation — a single search is never sufficient for these questions.

Required research process:
1. Decompose the question: identify all entities, events, and relationships mentioned.
2. Search for each clue independently — different phrasings, different angles.
3. When snippets look promising, call `get_document` to read the full text.
4. Cross-check: verify each finding against at least one other document.

Available tools:
- `search`: BM25 index lookup (returns docid, score, snippet).
- `get_document`: retrieve a full document by docid.

Answer format:
Explanation: <step-by-step reasoning citing specific documents and evidence>
Exact Answer: <concise final answer>\
"""

# Model function protocol: (messages, tools) -> assistant_msg (dict)
ModelFn = Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Any]

# ── Context condensation ──────────────────────

CONDENSE_PROMPT = """\
You are a research progress summarizer. Compress the conversation history into a \
concise progress record. Keep all facts, names, dates, numbers, and document IDs \
that are relevant to answering the question. Be thorough on facts, concise in wording.

Include:
1. **Original question** (verbatim)
2. **Tools called**: every tool invocation with its arguments, in order
3. **Key findings**: specific evidence gathered, with document IDs
4. **What remains to be found**"""


_TOKENIZER_PATH = "Qwen/Qwen3-8B"


def _init_tokenizer(path: str) -> None:
    global _TOKENIZER_PATH
    _TOKENIZER_PATH = path
    _get_tokenizer(path)  # eager load


def _count_tokens(msg_list: List[Dict[str, Any]]) -> int:
    return count_tokens(msg_list, model_path=_TOKENIZER_PATH)


async def _condense_context(
    messages: List[Dict[str, Any]],
    model_fn: ModelFn,
    max_tokens: int,
    max_context: int,
) -> List[Dict[str, Any]]:
    """Condense conversation history using token-accurate truncation.

    Uses the tokenizer to truncate the transcript so the condense call itself
    stays within the context window. Rebuilds as: [system, user, summary_user_msg].
    """
    if len(messages) <= 4:
        return messages

    # Serialize everything after system + user into one transcript
    transcript_lines: List[str] = []
    for m in messages[2:]:
        role = m.get("role", "?")
        content = str(m.get("content", "") or "")
        tc = m.get("tool_calls")
        if tc:
            for t in tc:
                fn = t.get("function", {})
                content += f"\n[TOOL_CALL: {fn.get('name', '?')}({fn.get('arguments', '')})]"
        transcript_lines.append(f"[{role}]: {content}")

    transcript = "\n\n".join(transcript_lines)

    condense_messages = [
        {"role": "system", "content": CONDENSE_PROMPT},
        {"role": "user", "content": f"Compress:\n\n{transcript}"},
    ]

    resp = await model_fn(condense_messages, tools=[])
    summary = resp.get("content", "")

    # Summary goes into a user message — it's new context for the agent
    summary_msg: Dict[str, Any] = {
        "role": "user",
        "content": (
            f"[PROGRESS SUMMARY — prior conversation compressed]\n"
            f"Original question: {messages[1]['content']}\n\n"
            f"{summary}"
        ),
    }

    condensed: List[Dict[str, Any]] = [
        messages[0],   # system prompt
        summary_msg,   # user: summary of everything so far
    ]

    before = _count_tokens(messages)
    after = _count_tokens(condensed)
    print(f"  [condense] {before} → {after} tokens ({len(messages)} → {len(condensed)} messages)", flush=True)
    return condensed


# ═══════════════════════════════════════════════════════════════
# Classic agent loop (模型推理 + 工具执行内部耦合)
# ═══════════════════════════════════════════════════════════════

async def run_agent_loop(
    question: str,
    searcher: BrowseCompBM25Searcher,
    client: VLLMClientAsync,
    model: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_turns: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    search_k: int = 5,
    snippet_max_chars: int = 1200,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    tools, registry = get_agent_tool_specs_and_registry(
        searcher=searcher, k=search_k, snippet_max_chars=snippet_max_chars
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for turn in range(max_turns):
        logger.debug("Turn %d: sending %d messages", turn + 1, len(messages))

        response = await client.simple_chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto",
            extra_payload=extra_payload,
        )

        choice = response["choices"][0]
        if "message" not in choice:
            logger.warning("Turn %d: no message in response", turn + 1)
            break

        assistant_msg: Dict[str, Any] = choice["message"]
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            logger.debug("Turn %d: agent finished (finish_reason=%s)", turn + 1, choice.get("finish_reason", ""))
            break

        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            args = json.loads(fn["arguments"])
            call_id: str = tc.get("id", "")

            if name not in registry:
                logger.warning("Turn %d: unknown tool %r", turn + 1, name)
                tool_result = json.dumps({"error": f"unknown tool: {name}"})
            else:
                try:
                    raw = registry[name](**args)
                    tool_result = json.dumps(raw, ensure_ascii=False)
                except Exception as exc:
                    logger.warning("Turn %d: tool %r failed: %s", turn + 1, name, exc)
                    tool_result = json.dumps({"error": str(exc)})

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": tool_result,
            })
    else:
        logger.warning("Agent reached max_turns (%d) without finishing", max_turns)

    return messages


async def run_agent(
    question: str,
    index_path: str,
    model: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_turns: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    search_k: int = 5,
    snippet_max_chars: int = 1200,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    searcher = build_searcher(index_path)
    client = VLLMClientAsync(base_url=base_url, api_key=api_key)
    try:
        return await run_agent_loop(
            question=question,
            searcher=searcher,
            client=client,
            model=model,
            system_prompt=system_prompt,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            search_k=search_k,
            snippet_max_chars=snippet_max_chars,
            extra_payload=extra_payload,
        )
    finally:
        await client._client.close()


# ═══════════════════════════════════════════════════════════════
# Env-based agent loop (模型推理与工具执行解耦)
# ═══════════════════════════════════════════════════════════════

async def run_agent_with_env(
    env: DeepResearchEnv,
    model_fn: ModelFn,
    questions: List[str],
    max_context: int = 40960,
    max_tokens: int = 4096,
) -> List[List[Dict[str, Any]]]:
    """使用 DeepResearchEnv 运行 agent loop。

    每轮对所有活跃实例并行调用模型（asyncio.gather），最大化 GPU 利用率。
    使用 tokenizer 精确计数，上下文余量不足时自动压缩。
    """
    obs, infos = env.reset(questions)
    tools = env.tool_specs

    for _ in range(env.max_turns):
        # 收集活跃实例
        active = [(i, o) for i, o in enumerate(obs) if o is not None]
        if not active:
            break

        # 检查 + 压缩：超过半满就压缩，保证 condense 调用自身不溢出
        async def _prepare(idx: int, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            used = _count_tokens(msgs)
            if used > max_context // 2 and len(msgs) > 4:
                print(f"  [condense] instance {idx}: {used}/{max_context} tokens → condensing", flush=True)
                return await _condense_context(msgs, model_fn, max_tokens, max_context)
            return msgs

        messages_list = await asyncio.gather(*[
            _prepare(i, m) for i, m in active
        ])

        # 每条消息的 token 数（用于日志）
        for idx, (i, _) in enumerate(active):
            tc = _count_tokens(messages_list[idx])
            if tc > max_context * 0.8:
                print(f"  [warn] instance {i}: {tc} input tokens, close to limit {max_context}", flush=True)

        # 并行调用模型
        async def _call_with_retry(msgs: List[Dict[str, Any]], idx: int) -> Dict[str, Any]:
            try:
                return await model_fn(msgs, tools)
            except Exception as e:
                err = str(e)
                if "context length" in err.lower() or "input_tokens" in err.lower():
                    # Context overflow — condense aggressively and retry once
                    print(f"  [overflow] instance {idx}: condensing and retrying...", flush=True)
                    condensed = await _condense_context(msgs, model_fn, max_tokens, max_context)
                    return await model_fn(condensed, tools)
                raise

        indices, msgs_list = zip(*active)
        responses = await asyncio.gather(*[
            _call_with_retry(m, i) for m, i in zip(messages_list, indices)
        ])

        actions: List[Any] = [None] * len(obs)
        for idx, resp in zip(indices, responses):
            actions[idx] = resp

        next_obs, rewards, dones, infos = env.step(actions)

        if all(dones):
            break
        obs = next_obs

    return env.get_trajectories()


def make_vllm_model_fn(
    client: VLLMClientAsync,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> ModelFn:
    """创建适配 DeepResearchEnv 的 model_fn（基于 VLLMClientAsync）。"""

    async def _fn(
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        response = await client.simple_chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto",
            extra_payload=extra_payload,
        )
        return response["choices"][0]["message"]

    return _fn


# ═══════════════════════════════════════════════════════════════
# 批量轨迹生成
# ═══════════════════════════════════════════════════════════════

async def generate_trajectories(
    dataset_path: str,
    index_path: str,
    model: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy",
    output_path: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    n_envs: int = 4,
    max_turns: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_context: int = 40960,
    search_k: int = 5,
    snippet_max_chars: int = 1200,
    extra_payload: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    strip_thinking: bool = True,
) -> List[Dict[str, Any]]:
    from .dataset_utils import load_jsonl

    rows = load_jsonl(dataset_path, limit=limit)
    total = len(rows)
    client = VLLMClientAsync(base_url=base_url, api_key=api_key)
    model_fn = make_vllm_model_fn(
        client=client, model=model,
        temperature=temperature, max_tokens=max_tokens,
        extra_payload=extra_payload,
    )

    env = DeepResearchEnv(
        index_path=index_path,
        n_envs=n_envs,
        system_prompt=system_prompt,
        max_turns=max_turns,
        search_k=search_k,
        snippet_max_chars=snippet_max_chars,
        record_trajectory=True,
        strip_thinking=strip_thinking,
    )

    records: List[Dict[str, Any]] = []

    try:
        for batch_start in range(0, total, n_envs):
            batch_rows = rows[batch_start:batch_start + n_envs]
            batch_questions = [r["query"] for r in batch_rows]

            trajs = await run_agent_with_env(
                env=env, model_fn=model_fn, questions=batch_questions,
                max_context=max_context, max_tokens=max_tokens,
            )

            for row, traj in zip(batch_rows, trajs):
                answer = extract_final_answer(traj) or ""
                records.append({
                    "query_id": row["query_id"],
                    "status": "completed",
                    "predicted_answer": answer,
                    "messages": traj,
                })

            done = batch_start + len(batch_rows)
            print(f"[generate] {done}/{total} queries done", flush=True)
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
# 异步批量评估
# ═══════════════════════════════════════════════════════════════

EVAL_SYSTEM_PROMPT = """You are an expert evaluator for question-answering systems.
Your task is to judge whether a predicted answer is semantically equivalent to the gold (reference) answer.

Rules:
- Ignore case differences, punctuation variations, and extra whitespace.
- Treat abbreviations and full forms as equivalent (e.g., "US" = "United States").
- If the predicted answer contains the gold answer as a substring (or vice versa) and the extra content does not change the meaning, treat as CORRECT.
- If the predicted answer is a valid alternative phrasing of the gold answer, treat as CORRECT.
- If the predicted answer is clearly wrong, incomplete in a meaningful way, or contradicts the gold answer, treat as INCORRECT.

Reply in exactly this format:
Judgment: CORRECT
Reasoning: <one sentence explaining your decision>"""


def _parse_eval_judgment(text: str) -> Tuple[str, str]:
    m = re.search(r'Judgment:\s*(CORRECT|INCORRECT)', text, re.IGNORECASE)
    judgment = m.group(1).upper() if m else "INCORRECT"
    m2 = re.search(r'Reasoning:\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    reasoning = m2.group(1).strip() if m2 else ""
    return judgment, reasoning


async def evaluate_trajectories(
    records: List[Dict[str, Any]],
    dataset_path: str,
    model: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy",
    eval_batch_size: int = 16,
    temperature: float = 0.0,
    max_tokens: int = 256,
    output_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """异步批量评估 — 使用 asyncio.gather 并行调用 eval 模型，打满 GPU。"""
    from .dataset_utils import load_jsonl

    dataset = load_jsonl(dataset_path)
    gold_map = {row["query_id"]: row["answer"] for row in dataset}
    question_map = {row["query_id"]: row.get("query", "") for row in dataset}

    client = VLLMClientAsync(base_url=base_url, api_key=api_key)
    details: List[Dict[str, Any]] = []
    correct = 0
    total = 0

    async def _eval_one(sub: Dict[str, Any]) -> Dict[str, Any]:
        qid = sub.get("query_id", "")
        gold = gold_map.get(qid, "")
        question = question_map.get(qid, "")
        pred = sub.get("predicted_answer", "")

        if not pred:
            pred = extract_final_answer(sub.get("messages", [])) or ""

        if not gold:
            return {
                "query_id": qid, "question": question,
                "gold_answer": "", "predicted_answer": pred,
                "eval_judgment": "INCORRECT",
                "eval_reasoning": "No gold answer found.",
                "eval_model_response": "",
                "trajectory_stats": {},
                "status": sub.get("status", "unknown"),
            }

        if not pred:
            return {
                "query_id": qid, "question": question,
                "gold_answer": gold, "predicted_answer": "",
                "eval_judgment": "INCORRECT",
                "eval_reasoning": "No predicted answer.",
                "eval_model_response": "",
                "trajectory_stats": _trajectory_stats(sub.get("messages", [])),
                "status": sub.get("status", "unknown"),
            }

        eval_msgs = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}"},
        ]

        try:
            resp = await client.simple_chat(
                model=model, messages=eval_msgs,
                temperature=temperature, max_tokens=max_tokens,
            )
            eval_text = resp["choices"][0]["message"]["content"]
            judgment, reasoning = _parse_eval_judgment(eval_text)
        except Exception as e:
            eval_text = f"ERROR: {e}"
            judgment, reasoning = "INCORRECT", str(e)

        return {
            "query_id": qid, "question": question,
            "gold_answer": gold, "predicted_answer": pred,
            "eval_judgment": judgment,
            "eval_reasoning": reasoning,
            "eval_model_response": eval_text,
            "trajectory_stats": _trajectory_stats(sub.get("messages", [])),
            "status": sub.get("status", "unknown"),
        }

    try:
        for batch_start in range(0, len(records), eval_batch_size):
            batch = records[batch_start:batch_start + eval_batch_size]
            results = await asyncio.gather(*[_eval_one(r) for r in batch])
            for d in results:
                if d["eval_judgment"] == "CORRECT":
                    correct += 1
                total += 1
                details.append(d)

            done = min(batch_start + eval_batch_size, len(records))
            print(f"[eval] {done}/{len(records)} done", flush=True)
    finally:
        await client._client.close()

    accuracy = correct / total if total > 0 else 0.0
    all_tc = [d["trajectory_stats"].get("num_tool_calls", 0) for d in details]
    all_docs = [d["trajectory_stats"].get("num_retrieved_docs", 0) for d in details]

    summary: Dict[str, Any] = {
        "total_queries": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(accuracy, 4),
        "avg_tool_calls_per_query": round(sum(all_tc) / total, 2) if total > 0 else 0,
        "avg_retrieved_docs_per_query": round(sum(all_docs) / total, 2) if total > 0 else 0,
        "total_tool_calls": sum(all_tc),
        "total_retrieved_docs": sum(all_docs),
        "eval_model": model,
    }

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[eval] saved results → {output_path}", flush=True)

    return summary, details


def _trajectory_stats(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    tc = 0
    docids: List[str] = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tc += len(msg["tool_calls"])
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "docid" in item:
                            docids.append(item["docid"])
                elif isinstance(parsed, dict) and "docid" in parsed:
                    docids.append(parsed["docid"])
    return {
        "num_tool_calls": tc,
        "num_assistant_messages": sum(1 for m in messages if m.get("role") == "assistant"),
        "num_tool_messages": sum(1 for m in messages if m.get("role") == "tool"),
        "num_retrieved_docs": len(docids),
        "unique_retrieved_docids": len(set(docids)),
    }


def extract_final_answer(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return None


# ═══════════════════════════════════════════════════════════════
# 一键执行入口
# ═══════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deep Research Agent — 一键轨迹生成 + 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agent.agent_loop \\
      --dataset browsecomp_plus_hard50.jsonl \\
      --index-path indexes/browsecomp_plus_bm25.sqlite \\
      --model qwen_auto --max-tokens 4096 --n-envs 4

  python -m agent.agent_loop \\
      --dataset browsecomp_plus_hard50.jsonl \\
      --index-path indexes/browsecomp_plus_bm25.sqlite \\
      --model qwen_auto --limit 10 --no-eval
        """,
    )
    p.add_argument("--dataset", required=True, help="数据集 jsonl 路径")
    p.add_argument("--index-path", required=True, help="BM25 SQLite 索引路径")
    p.add_argument("--model", default="qwen_auto", help="vLLM 模型名")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM 服务地址")
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--output-dir", default="runs", help="输出目录")
    p.add_argument("--n-envs", type=int, default=4, help="并行 env 实例数")
    p.add_argument("--max-turns", type=int, default=10, help="最大 tool-calling 轮数")
    p.add_argument("--max-tokens", type=int, default=4096, help="每轮模型最大 token 数")
    p.add_argument("--max-context", type=int, default=40960, help="模型最大上下文长度（用于自动压缩判断）")
    p.add_argument("--search-k", type=int, default=5, help="search 返回文档数")
    p.add_argument("--snippet-max-chars", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--eval-batch-size", type=int, default=16, help="评估并行数")
    p.add_argument("--eval-model", default=None, help="评估模型（默认同 --model）")
    p.add_argument("--limit", type=int, default=None, help="限制处理条数")
    p.add_argument("--no-eval", action="store_true", help="跳过评估")
    p.add_argument("--no-strip-thinking", action="store_true", help="保留 <think> 块在上下文中（默认 strip）")
    p.add_argument("--tokenizer-path", default="Qwen/Qwen3-8B", help="Tokenizer 模型路径（用于精确 token 计数）")
    return p


async def _main_async(args: argparse.Namespace) -> None:
    _init_tokenizer(args.tokenizer_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    submission_path = str(output_dir / f"submission_{ts}.jsonl")
    eval_path = str(output_dir / f"eval_{ts}.jsonl")

    # ── 1. 生成轨迹 ──
    t0 = time.time()
    records = await generate_trajectories(
        dataset_path=args.dataset,
        index_path=args.index_path,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        output_path=submission_path,
        n_envs=args.n_envs,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_context=args.max_context,
        search_k=args.search_k,
        snippet_max_chars=args.snippet_max_chars,
        limit=args.limit,
        strip_thinking=not args.no_strip_thinking,
    )
    gen_time = time.time() - t0
    print(f"\n[done] generated {len(records)} trajectories in {gen_time:.1f}s", flush=True)

    if args.no_eval:
        return

    # ── 2. 评估 ──
    eval_model = args.eval_model or args.model
    t0 = time.time()
    summary, details = await evaluate_trajectories(
        records=records,
        dataset_path=args.dataset,
        model=eval_model,
        base_url=args.base_url,
        api_key=args.api_key,
        eval_batch_size=args.eval_batch_size,
        temperature=0.0,
        max_tokens=256,
        output_path=eval_path,
    )
    eval_time = time.time() - t0

    # ── 3. 打印结果 ──
    print(f"\n{'='*50}")
    print(f"Evaluation complete in {eval_time:.1f}s")
    print(f"Accuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total_queries']})")
    print(f"Avg tool calls/query: {summary['avg_tool_calls_per_query']}")
    print(f"Avg retrieved docs/query: {summary['avg_retrieved_docs_per_query']}")
    print(f"{'='*50}")

    # 错误案例
    errors = [d for d in details if d["eval_judgment"] == "INCORRECT"]
    if errors:
        print(f"\nIncorrect ({len(errors)}):")
        for d in errors[:10]:
            print(f"  [{d['query_id']}] pred={d['predicted_answer'][:80]}...")


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
