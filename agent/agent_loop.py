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
import os
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

from .browsecomp_searcher import BrowseCompBM25Searcher
from .env import DeepResearchEnv
from .eval_async import evaluate_trajectories
from .tools import build_searcher, get_agent_tool_specs_and_registry
from .utils import (
    RETRY_NUDGE,
    count_tokens_messages,
    extract_final_answer,
    hard_truncate_tail_tool_messages,
    is_truncated_think_response,
)
from .vllm_client_async import VLLMClientAsync

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a Deep Research Agent. Your task is to find the correct answer to a complex \
question by searching a document corpus.

CRITICAL RULES — you MUST follow these:
1. ALWAYS call `search` or `get_document` on your first turn. Never output a final \
answer without first using at least one tool. You do NOT know the answer in advance.
2. Keep your thinking concise — plan your next tool call in 1-2 sentences max. \
Long analysis without acting is forbidden. Act first, then think about results.
3. Conduct multi-round investigation: use DIFFERENT search queries with DIFFERENT \
phrasings. A single search is never enough. Aim for 3+ distinct searches.
4. When snippets look relevant, call `get_document` to read the full document.
5. Cross-check every finding against at least one other independent source.

Available tools:
- `search`: BM25 index lookup (returns docid, score, snippet).
- `get_document`: retrieve a full document by docid.

Answer format (on your FINAL turn — when you are ready to answer):
YOU MUST output exactly in this format, with both sections present:
Explanation: <step-by-step reasoning citing specific docids and evidence>
Exact Answer: <your final concise answer>
Do NOT include anything after "Exact Answer:" — no extra commentary.\
"""

# ── Context condensation ──────────────────────

CONDENSE_PROMPT = """\
You are a research progress summarizer. Compress the conversation history into a \
concise but complete progress record. Preserve ALL factual details — names, dates, \
numbers, document IDs, and key snippets. Do NOT summarize or paraphrase evidence; \
copy important findings verbatim. Be thorough on facts, concise in wording.

Structure your output as follows:

1. **Original question** (verbatim)
2. **Searches performed** (list every search query with the docids it returned)
3. **Documents retrieved** (for each docid read via get_document, keep the full \
   document text or at minimum all factual claims, names, dates, and numbers)
4. **Key findings** (specific evidence gathered, cross-references verified)
5. **What remains to be found** (specific missing pieces needed to answer)

CRITICAL: Do NOT lose any document ID or factual detail. If a document contains a \
name, date, or number that might be relevant, keep it verbatim."""


async def _condense_context(
    tok: Any,
    messages: List[Dict[str, Any]],
    client: VLLMClientAsync,
    model: str,
    temperature: float,
    max_tokens: int,
    max_context: int,
    extra_payload: Optional[Dict[str, Any]] = None,
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

    resp = await client.simple_chat(
        model=model,
        messages=condense_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=[],
        tool_choice="auto",
        extra_payload=extra_payload,
    )
    summary = resp["choices"][0]["message"].get("content", "")

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

    before = count_tokens_messages(tok, messages)
    after = count_tokens_messages(tok, condensed)
    print(f"  [condense] {before} → {after} tokens ({len(messages)} → {len(condensed)} messages)", flush=True)
    return condensed


# ═══════════════════════════════════════════════════════════════
# Env-based agent loop (模型推理与工具执行解耦)
# ═══════════════════════════════════════════════════════════════

async def run_agent_with_env(
    env: DeepResearchEnv,
    client: VLLMClientAsync,
    model: str,
    questions: List[str],
    max_context: int = 40960,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> List[List[Dict[str, Any]]]:
    """使用 DeepResearchEnv 运行 agent loop。

    每轮：模型推理 → env.step（追加 assistant + tool）→ token 检查 → 压缩。
    压缩在 env.step 之后，避免提前压缩导致的 len guard 死锁。
    """
    obs, infos = env.reset(questions)
    tools = env.tool_specs

    for _ in range(env.max_turns):
        # 收集活跃实例
        active = [(i, o) for i, o in enumerate(obs) if o is not None]
        if not active:
            break

        # 1. 并行模型调用
        indices, msgs_list = zip(*active)
        raw = await asyncio.gather(*[
            client.simple_chat(
                model=model,
                messages=m,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice="auto",
                extra_payload=extra_payload,
            )
            for m in msgs_list
        ])
        responses = [r["choices"][0]["message"] for r in raw]

        # 1.5 截断检测与重试（防止模型输出超长 <think> 块导致无工具调用）
        for i in range(len(responses)):
            resp = responses[i]
            content = resp.get("content", "") or ""
            tool_calls = resp.get("tool_calls")
            idx = indices[i]
            if is_truncated_think_response(content, tool_calls):
                msgs = list(obs[idx]) if obs[idx] is not None else []
                msgs.append({
                    "role": "user",
                    "content": RETRY_NUDGE,
                })
                retry_raw = await client.simple_chat(
                    model=model,
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice="auto",
                    extra_payload=extra_payload,
                )
                retry_resp = retry_raw["choices"][0]["message"]
                responses[i] = retry_resp
                print(f"  [retry] instance {idx}: truncated think detected, nudging model to call tools", flush=True)

        # 2. env.step — 追加 assistant 消息 + tool 结果
        actions: List[Any] = [None] * len(obs)
        for idx, resp in zip(indices, responses):
            actions[idx] = resp

        next_obs, rewards, dones, infos = env.step(actions)

        if all(dones):
            break

        # 3. 检查 + 压缩（在 env.step 之后，消息数自然 > 2）
        async def _maybe_condense(
            idx: int, msgs: Optional[List[Dict[str, Any]]]
        ) -> Optional[List[Dict[str, Any]]]:
            if msgs is None:
                return None
            used = count_tokens_messages(_tok, msgs)
            if used > max_context // 2:
                last = msgs[-1] if msgs else None
                is_tool_tail = last is not None and last.get("role") == "tool"
                if not is_tool_tail:
                    raise RuntimeError(
                        f"instance {idx}: context at {used} tokens (>"
                        f"{max_context // 2}) but last message role is "
                        f"{last.get('role') if isinstance(last, dict) else last!r}; "
                        "expected tool messages at tail after env.step — this should not happen."
                    )
                hard_truncate_tail_tool_messages(_tok, msgs, max_context)
                used_after = count_tokens_messages(_tok, msgs)
                if used_after > max_context:
                    raise RuntimeError(
                        f"instance {idx}: {used_after} tokens still exceed max_context="
                        f"{max_context} after hard-truncating the trailing tool block; "
                        "likely oversized older tool/assistant payloads or missing prior "
                        "condense — this should not happen."
                    )
                print(
                    f"  [condense] instance {idx}: {used}/{max_context} tokens "
                    f"(after tool hard-cap {used_after}) → condensing",
                    flush=True,
                )
                condensed = await _condense_context(
                    _tok, msgs, client, model, temperature, max_tokens, max_context, extra_payload,
                )
                env.set_messages(idx, condensed)
                return condensed
            return msgs

        obs = await asyncio.gather(*[
            _maybe_condense(i, o) for i, o in enumerate(next_obs)
        ])

    return env.get_trajectories()


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
                env=env,
                client=client,
                model=model,
                questions=batch_questions,
                max_context=max_context,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_payload=extra_payload,
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
    global _TOKENIZER_PATH, _tok
    if args.tokenizer_path != _TOKENIZER_PATH:
        _TOKENIZER_PATH = args.tokenizer_path
        _tok = AutoTokenizer.from_pretrained(
            _TOKENIZER_PATH, trust_remote_code=True, local_files_only=True,
        )

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
