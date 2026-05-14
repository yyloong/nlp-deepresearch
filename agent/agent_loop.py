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
    validate_tool_call,
)
from .vllm_client_async import VLLMClientAsync

logger = logging.getLogger(__name__)

MAX_TOOL_RETRIES = 2  # max retries per turn when tool call validation fails

DEFAULT_SYSTEM_PROMPT = """\
You are a Deep Research Agent. Answer complex questions by searching a document corpus \
using `search` and `get_document`. Every answer must be grounded in retrieved evidence.

**Important Rules:**
1.You are allowed to call one tool per turn.
2.search tool is used to get the relevant documents,and get_document tool is used to get the detailed information of the document.
3.You should collect information step by step,make sure all the answer has its evidence and always have a full understanding of the whole context before you propose a conclusion.

### BM25 SEARCH ENGINE PRINCIPLES & QUERY GUIDELINES ###

**Understanding the BM25 Indexer (The Principles):**
The `search` tool is powered by a **BM25 (Bag-of-Words)** indexer. You MUST adapt your queries to its mechanical nature:
1. **Zero Semantic Understanding:** BM25 does not understand meaning, grammar, or intent. It only counts exact word occurrences. A query like "who was the ruler" will literally search for documents containing the word "who".
2. **High IDF (Inverse Document Frequency) Rules All:** BM25 gives massively higher scores to RARE words (unique nouns, weird names, specific IDs) and penalizes common words (verbs, adjectives, prepositions). 
3. **Math & Logic Blindness:** BM25 cannot calculate relations. "Over 500", "before 2020", or "mother of" are ignored logically. It just searches for the literal string "over" and "500".
4. **Query Dilution:** The more words you put in a query, the more the BM25 score is diluted by irrelevant matches. Shorter, highly-targeted keyword clusters win.

**Actionable BM25 Search Rules:**

1. **Decompose Multi-Hop Questions (Prevent Query Dilution):**
   - NEVER put all constraints into one query. BM25 will fail if you ask it to match 10 different facts at once. Break it down into sequential steps.
   - *Bad:* "wizard who won the Dragon Taming Cup in the 3rd Era worked at a magical academy built by the Elf King"
   - *Good Step 1:* First, find the academy -> "magical academy" "Elf King"
   - *Good Step 2:* Then, find the person -> "[Name of Academy from Step 1]" "Dragon Taming Cup" "3rd Era"

2. **Extract High-IDF Nouns ONLY:**
   - Strip out ALL conversational language, verbs, and relational phrases. Only keep the rarest nouns and entities. 
   - *Bad:* "a cybernetic pirate who secretly smuggled a glowing pineapple into a spaceship"
   - *Good:* pirate "glowing pineapple" spaceship (Drop "who secretly smuggled")

3. **Strip Relational Operators for Numbers/Dates:**
   - Remove comparative words. Keep only exact digits or unique text descriptors.
   - *Bad:* "a vampire born before the year 800 whose creator was a legendary blacksmith"
   - *Good:* vampire creator blacksmith 800 (Drop "born before the year")

4. **Target the Most UNIQUE Identifier First:**
   - Always start your search with the rarest combination of words (Highest IDF tokens) to quickly narrow down the BM25 results.
   - *Example:* If looking for "a three-headed dog guarding a neon castle during the Great Meteor Shower", start with -> "three-headed dog" "neon castle" "Great Meteor Shower"

5. **Iterative BM25 Refinement:**
   - If snippets are irrelevant, your keywords might be too strict or slightly mismatched in phrasing. DO NOT just repeat the query. 
   - *Strategy:* Drop the least important keywords, or try noun synonyms that might appear in a formal document (e.g., if "cash payment" fails, try "financial settlement" or "compensation").

You MUST work in the following order:

1. Search for specific entities (names, places, dates) rather than long descriptive phrases.
2. After getting results, extract names/entities from them and use those for your next search.
3. If a snippet looks even partially relevant,use **call get_document** to read more detailed information and collect information in a more comprehensive way and in a full context,some times information can be totally different in different context,only reading snippets can be misleading.
4. After get the detailed document, find key **relevant** information.
5. Then if there are other documents that you haven't check detailedly, you should continue to search and get the detailed information.
6.If you find all documents can not provided enough information,keep searching,refine your search query and consider if you can search from a different angle then rewrite your search query and for more useful documents again.
7.The answer **MUST** match the question perfectly otherwise you should continue to search.

You are not allowed to output the following content unless you are totally confident about your answer:

**I am sure that the answer is totally correct,and the evidence is**
evidence:
Evidence Mapping (list each claim and its source):
  Claim 1: <what I assert>
    → Source: docid=<X>, quote="<exact supporting text from the document>"
  Claim 2: ...
  (add more claims as needed)
answer:
Explanation: <step-by-step reasoning, citing docids for each claim>
Exact Answer: <concise final answer>
"""

# ── Context condensation ──────────────────────

CONDENSE_PROMPT = """\
You are a research progress summarizer. Compress the conversation history into a \
concise but complete progress record. Preserve ALL factual details — names, dates, \
numbers, document IDs, and key snippets. Do NOT summarize or paraphrase evidence; \
copy important findings verbatim. Be thorough on facts, concise in wording.

Structure your output as follows:

1. **Original question** (verbatim)
2. **Clues from question** (list every distinct clue / constraint that needs verification)
3. **Clue verification status** (for each clue: ✓ verified by docid X, or ✗ still unknown)
4. **Searches performed** (list every search query with the docids it returned)
5. **Documents retrieved** (for each docid read via get_document, keep the full \
   document text or at minimum all factual claims, names, dates, and numbers)
6. **Key findings** (specific evidence gathered, cross-references verified)
7. **What remains to be found** (specific missing pieces needed to answer)

CRITICAL: Do NOT lose any document ID or factual detail. If a document contains a \
name, date, or number that might be relevant, keep it verbatim. The clue verification \
status is the most important section — it tells the agent what still needs work."""


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
    max_tool_calls_per_turn: int = 1,
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

        # 1.6 工具调用校验 + 重试（检测未知工具名 / 缺失必填参数，省一轮 turn）
        for i in range(len(responses)):
            resp = responses[i]
            idx = indices[i]
            tool_calls = resp.get("tool_calls")
            if not tool_calls:
                continue

            for retry_num in range(MAX_TOOL_RETRIES):
                all_errors: List[Dict[str, str]] = []
                for tc in tool_calls:
                    err = validate_tool_call(tc, tools)
                    if err:
                        all_errors.append({
                            "tool_name": tc.get("function", {}).get("name", "?"),
                            "message": err,
                        })

                if not all_errors:
                    break

                # 构建带错误信息的 nudge
                msgs = list(obs[idx]) if obs[idx] is not None else []
                msgs.append(resp)  # 失败的那条 assistant 消息

                error_lines = []
                for e in all_errors:
                    error_lines.append(f"- `{e['tool_name']}`: {e['message']}")

                nudge = (
                    f"Your tool call(s) failed validation with the following error(s):\n\n"
                    + "\n".join(error_lines)
                    + "\n\nPlease correct the error(s) and call the tool(s) again "
                    "with valid arguments."
                )
                msgs.append({"role": "user", "content": nudge})

                print(
                    f"  [tool-retry] instance {idx} attempt {retry_num + 1}/{MAX_TOOL_RETRIES}: "
                    f"{len(all_errors)} validation error(s)",
                    flush=True,
                )

                retry_raw = await client.simple_chat(
                    model=model,
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice="auto",
                    extra_payload=extra_payload,
                )
                resp = retry_raw["choices"][0]["message"]
                responses[i] = resp
                tool_calls = resp.get("tool_calls")

                if not tool_calls:
                    # 模型放弃调用工具 — 交给 env.step 处理
                    break

        # 2. env.step — 追加 assistant 消息 + tool 结果
        actions: List[Any] = [None] * len(obs)
        for idx, resp in zip(indices, responses):
            tc = resp.get("tool_calls")
            if tc and len(tc) > max_tool_calls_per_turn:
                resp["tool_calls"] = tc[:max_tool_calls_per_turn]
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
                hard_truncate_tail_tool_messages(_tok, msgs, max_context, label=f"instance {idx}")
                used_after = count_tokens_messages(_tok, msgs)

                # If truncation alone brought us back under threshold, skip condensation
                if used_after <= max_context // 2:
                    print(
                        f"  [condense] instance {idx}: {used}→{used_after}/{max_context} tokens "
                        f"(after tool hard-cap, under threshold → skip)",
                        flush=True,
                    )
                    return msgs

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
# Router-based agent loop — 保持所有 env slot 满载
# ═══════════════════════════════════════════════════════════════

async def run_agent_router(
    env: DeepResearchEnv,
    client: VLLMClientAsync,
    model: str,
    questions: List[str],
    max_context: int = 40960,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    extra_payload: Optional[Dict[str, Any]] = None,
    max_tool_calls_per_turn: int = 1,
) -> List[List[Dict[str, Any]]]:
    """Router-based agent loop — keeps all env slots filled.

    When an instance finishes, it immediately starts the next pending question
    in that slot, maximising parallel throughput.

    Returns trajectories in the same order as *questions*.
    """
    n_envs: int = env.n_envs
    n_total: int = len(questions)
    n_initial: int = min(n_envs, n_total)
    tools: List[Dict[str, Any]] = env.tool_specs

    # ── Per-slot bookkeeping ──
    slot_qidx: Dict[int, int] = {}                # slot → original question index
    trajectories: Dict[int, List[Dict[str, Any]]] = {}  # qidx → completed trajectory
    next_qidx: int = 0                             # next question to assign

    # ── Fill initial slots ──
    initial_qs = questions[:n_initial]
    for slot in range(n_initial):
        slot_qidx[slot] = slot
    next_qidx = n_initial

    obs, _infos = env.reset(initial_qs)

    # Expand obs to full n_envs (unused slots are None)
    obs = list(obs) + [None] * (n_envs - n_initial)

    while slot_qidx or (next_qidx < n_total):
        # ── 1. Collect active instances ──
        active = [(i, o) for i, o in enumerate(obs) if o is not None]
        if not active:
            # All current instances done but more questions pending — start fresh batch
            fresh_qs: List[str] = []
            fresh_slots: List[int] = []
            for slot in range(n_envs):
                if next_qidx < n_total:
                    fresh_qs.append(questions[next_qidx])
                    fresh_slots.append(slot)
                    slot_qidx[slot] = next_qidx
                    next_qidx += 1
            if not fresh_qs:
                break
            env.reset(fresh_qs)
            obs = [None] * n_envs
            for slot in fresh_slots:
                obs[slot] = env.get_slot_messages(slot)
            continue

        indices, msgs_list = zip(*active)

        # ── 2. Parallel model calls ──
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

        # ── 2.5 Think truncation retry ──
        for i in range(len(responses)):
            resp = responses[i]
            content = resp.get("content", "") or ""
            tool_calls = resp.get("tool_calls")
            idx = indices[i]
            if is_truncated_think_response(content, tool_calls):
                msgs = list(obs[idx]) if obs[idx] is not None else []
                msgs.append({"role": "user", "content": RETRY_NUDGE})
                retry_raw = await client.simple_chat(
                    model=model, messages=msgs,
                    temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice="auto", extra_payload=extra_payload,
                )
                responses[i] = retry_raw["choices"][0]["message"]

        # ── 2.6 Tool call validation retry ──
        for i in range(len(responses)):
            resp = responses[i]
            idx = indices[i]
            tool_calls = resp.get("tool_calls")
            if not tool_calls:
                continue
            for retry_num in range(MAX_TOOL_RETRIES):
                all_errors: List[Dict[str, str]] = []
                for tc in tool_calls:
                    err = validate_tool_call(tc, tools)
                    if err:
                        all_errors.append({
                            "tool_name": tc.get("function", {}).get("name", "?"),
                            "message": err,
                        })
                if not all_errors:
                    break
                msgs = list(obs[idx]) if obs[idx] is not None else []
                msgs.append(resp)
                error_lines = [f"- `{e['tool_name']}`: {e['message']}" for e in all_errors]
                nudge = (
                    "Your tool call(s) failed validation:\n\n"
                    + "\n".join(error_lines)
                    + "\n\nPlease correct the error(s) and try again."
                )
                msgs.append({"role": "user", "content": nudge})
                retry_raw = await client.simple_chat(
                    model=model, messages=msgs,
                    temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice="auto", extra_payload=extra_payload,
                )
                resp = retry_raw["choices"][0]["message"]
                responses[i] = resp
                tool_calls = resp.get("tool_calls")
                if not tool_calls:
                    break

        # ── 3. env.step ──
        actions: List[Any] = [None] * n_envs
        for idx, resp in zip(indices, responses):
            tc = resp.get("tool_calls")
            if tc and len(tc) > max_tool_calls_per_turn:
                resp["tool_calls"] = tc[:max_tool_calls_per_turn]
            actions[idx] = resp
        next_obs, _rewards, dones, _infos = env.step(actions)

        # ── 4. Handle completions + refill ──
        for slot in range(n_envs):
            if dones[slot] and slot in slot_qidx:
                qidx = slot_qidx.pop(slot)
                trajectories[qidx] = env.extract_slot_trajectory(slot)

                if next_qidx < n_total:
                    # Refill this slot immediately
                    slot_qidx[slot] = next_qidx
                    next_obs[slot] = env.reset_slot(slot, questions[next_qidx])
                    next_qidx += 1

        # ── 5. Context condensation (per active instance) ──
        async def _maybe_condense(
            idx: int, msgs: Optional[List[Dict[str, Any]]]
        ) -> Optional[List[Dict[str, Any]]]:
            if msgs is None:
                return None
            used = count_tokens_messages(_tok, msgs)
            if used <= max_context // 2:
                return msgs

            last = msgs[-1] if msgs else None
            is_tool_tail = last is not None and last.get("role") == "tool"
            if not is_tool_tail:
                return msgs  # don't condense if tail isn't tool messages

            hard_truncate_tail_tool_messages(_tok, msgs, max_context, label=f"instance {idx}")
            used_after = count_tokens_messages(_tok, msgs)
            if used_after <= max_context // 2:
                return msgs  # truncation sufficed

            condensed = await _condense_context(
                _tok, msgs, client, model, temperature, max_tokens, max_context, extra_payload,
            )
            env.set_messages(idx, condensed)
            return condensed

        obs = await asyncio.gather(*[
            _maybe_condense(i, o) for i, o in enumerate(next_obs)
        ])

    # ── Return in original order ──
    return [trajectories[i] for i in range(n_total)]


# ═══════════════════════════════════════════════════════════════
# Async per-slot router — 每个 slot 独立协程，一问结束立刻补下一问
# ═══════════════════════════════════════════════════════════════

async def _run_one_question_async(
    slot_id: int,
    qidx: int,
    question: str,
    env: DeepResearchEnv,
    client: VLLMClientAsync,
    model: str,
    tools: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    max_context: int,
    extra_payload: Optional[Dict[str, Any]],
    result_queue: "asyncio.Queue[tuple[int, List[Dict[str, Any]]]]",
    *,
    done_counter: Optional[List[int]] = None,
    n_total: int = 0,
    done_lock: Optional["asyncio.Lock"] = None,
    max_tool_calls_per_turn: int = 1,
) -> None:
    """Run a single question to completion in one slot.

    The slot's lifecycle: model → retries → step_single → condense → loop.
    When done, pushes ``(qidx, trajectory)`` into *result_queue*.
    If *done_counter* / *done_lock* / *n_total* are provided, prints
    progress every 10 completions.
    """
    import asyncio as _asyncio

    obs: List[Dict[str, Any]] = env.reset_slot(slot_id, question)

    for _ in range(env.max_turns):
        # ── Model call ──
        raw = await client.simple_chat(
            model=model, messages=obs,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice="auto", extra_payload=extra_payload,
        )
        resp: Dict[str, Any] = raw["choices"][0]["message"]

        # ── Think truncation retry ──
        content = resp.get("content", "") or ""
        tc = resp.get("tool_calls")
        if is_truncated_think_response(content, tc):
            msgs = list(obs) + [{"role": "user", "content": RETRY_NUDGE}]
            raw = await client.simple_chat(
                model=model, messages=msgs,
                temperature=temperature, max_tokens=max_tokens,
                tools=tools, tool_choice="auto", extra_payload=extra_payload,
            )
            resp = raw["choices"][0]["message"]
            tc = resp.get("tool_calls")

        # ── Tool validation retry ──
        if tc:
            for _retry_num in range(MAX_TOOL_RETRIES):
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
                msgs = list(obs) + [resp]
                error_lines = [f"- `{e['tool_name']}`: {e['message']}" for e in all_errors]
                nudge = (
                    "Your tool call(s) failed validation:\n\n"
                    + "\n".join(error_lines)
                    + "\n\nPlease correct the error(s) and try again."
                )
                msgs.append({"role": "user", "content": nudge})
                raw = await client.simple_chat(
                    model=model, messages=msgs,
                    temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice="auto", extra_payload=extra_payload,
                )
                resp = raw["choices"][0]["message"]
                tc = resp.get("tool_calls")
                if not tc:
                    break

        # ── Enforce max tool calls per turn ──
        tc = resp.get("tool_calls")
        if tc and len(tc) > max_tool_calls_per_turn:
            resp["tool_calls"] = tc[:max_tool_calls_per_turn]

        # ── env.step_single ──
        obs, done = env.step_single(slot_id, resp)

        if done:
            break

        # ── Context condensation ──
        used = count_tokens_messages(_tok, obs)
        if used > max_context // 2:
            last = obs[-1] if obs else None
            is_tool_tail = last is not None and last.get("role") == "tool"
            if is_tool_tail:
                hard_truncate_tail_tool_messages(_tok, obs, max_context, label=f"slot {slot_id}")
                used_after = count_tokens_messages(_tok, obs)
                if used_after > max_context // 2:
                    condensed = await _condense_context(
                        _tok, obs, client, model, temperature,
                        max_tokens, max_context, extra_payload,
                    )
                    env.set_messages(slot_id, condensed)
                    obs = condensed

    # ── Report result ──
    traj = env.extract_slot_trajectory(slot_id)
    await result_queue.put((qidx, traj))

    if done_counter is not None and done_lock is not None and n_total > 0:
        async with done_lock:
            done_counter[0] += 1
            cur = done_counter[0]
        if cur % 10 == 0 or cur == n_total:
            print(f"  [router] {cur}/{n_total} queries done", flush=True)


async def run_agent_async_router(
    env: DeepResearchEnv,
    client: VLLMClientAsync,
    model: str,
    questions: List[str],
    max_context: int = 40960,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    extra_payload: Optional[Dict[str, Any]] = None,
    max_tool_calls_per_turn: int = 1,
) -> List[List[Dict[str, Any]]]:
    """Fully-async per-slot router — no idle time between questions.

    Each env slot runs as an independent coroutine.  When a slot finishes
    one question it immediately pulls the next from the queue.  Model calls
    and tool execution never block sibling slots.

    Returns trajectories in the same order as *questions*.
    """
    import asyncio as _asyncio

    n_total = len(questions)
    n_workers = min(env.n_envs, n_total)
    tools = env.tool_specs

    pending: "_asyncio.Queue[tuple[int, str]]" = _asyncio.Queue()
    results: "_asyncio.Queue[tuple[int, List[Dict[str, Any]]]]" = _asyncio.Queue()

    for qidx, q in enumerate(questions):
        await pending.put((qidx, q))

    # Shared progress counter so workers can print incrementally
    done_counter: List[int] = [0]
    done_lock = _asyncio.Lock()

    async def _worker(slot_id: int) -> None:
        while True:
            try:
                qidx, q = pending.get_nowait()
            except _asyncio.QueueEmpty:
                return
            await _run_one_question_async(
                slot_id=slot_id,
                qidx=qidx,
                question=q,
                env=env,
                client=client,
                model=model,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                max_context=max_context,
                extra_payload=extra_payload,
                result_queue=results,
                done_counter=done_counter,
                n_total=n_total,
                done_lock=done_lock,
                max_tool_calls_per_turn=max_tool_calls_per_turn,
            )

    workers = [_worker(i) for i in range(n_workers)]
    await _asyncio.gather(*workers)

    # Collect results in original order
    result_dict: Dict[int, List[Dict[str, Any]]] = {}
    for _ in range(n_total):
        qidx, traj = await results.get()
        result_dict[qidx] = traj

    return [result_dict[i] for i in range(n_total)]


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
    condense_thinking: bool = False,
    max_tool_calls_per_turn: int = 1,
) -> List[Dict[str, Any]]:
    from .dataset_utils import load_jsonl

    rows = load_jsonl(dataset_path, limit=limit)
    total = len(rows)
    client = VLLMClientAsync(base_url=base_url, api_key=api_key, max_concurrent=n_envs)

    env = DeepResearchEnv(
        index_path=index_path,
        n_envs=n_envs,
        system_prompt=system_prompt,
        max_turns=max_turns,
        search_k=search_k,
        snippet_max_chars=snippet_max_chars,
        record_trajectory=True,
        strip_thinking=strip_thinking,
        condense_thinking=condense_thinking,
    )

    records: List[Dict[str, Any]] = []

    try:
        all_questions = [r["query"] for r in rows]
        all_qids = [r["query_id"] for r in rows]

        trajs = await run_agent_async_router(
            env=env,
            client=client,
            model=model,
            questions=all_questions,
            max_context=max_context,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_payload=extra_payload,
            max_tool_calls_per_turn=max_tool_calls_per_turn,
        )

        for row, traj in zip(rows, trajs):
            answer = extract_final_answer(traj) or ""
            records.append({
                "query_id": row["query_id"],
                "status": "completed",
                "predicted_answer": answer,
                "messages": traj,
            })

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
    p.add_argument("--max-tool-calls-per-turn", type=int, default=1, help="每轮最大 tool call 数（超出的会被截断）")
    p.add_argument("--search-k", type=int, default=5, help="search 返回文档数")
    p.add_argument("--snippet-max-chars", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--eval-batch-size", type=int, default=16, help="评估并行数")
    p.add_argument("--eval-model", default=None, help="评估模型（默认同 --model）")
    p.add_argument("--limit", type=int, default=None, help="限制处理条数")
    p.add_argument("--no-eval", action="store_true", help="跳过评估")
    p.add_argument("--no-strip-thinking", action="store_true", help="保留 <think> 块在上下文中（默认 strip）")
    p.add_argument("--condense-thinking", action="store_true", help="压缩 <think> 块为简洁的计划摘要而非完全 strip（保留核心目的和规划）")
    p.add_argument("--no-think", action="store_true", help="禁用模型 thinking 模式（通过 extra_payload 传入 chat_template_kwargs）")
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
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    submission_path = str(run_dir / "submission.jsonl")
    eval_path = str(run_dir / "eval.jsonl")

    # ── 构建 extra_payload ──
    extra_payload: Optional[Dict[str, Any]] = None
    if args.no_think:
        extra_payload = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

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
        extra_payload=extra_payload,
        limit=args.limit,
        strip_thinking=not args.no_strip_thinking,
        condense_thinking=args.condense_thinking,
        max_tool_calls_per_turn=args.max_tool_calls_per_turn,
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
        max_tokens=8192,
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

    # ── 4. 拆分保存 correct / incorrect ──
    eval_map = {d["query_id"]: d["eval_judgment"] for d in details}
    correct_records = [r for r in records if eval_map.get(r["query_id"]) == "CORRECT"]
    incorrect_records = [r for r in records if eval_map.get(r["query_id"]) == "INCORRECT"]

    # 轨迹
    correct_path = run_dir / "correct.json"
    incorrect_path = run_dir / "incorrect.json"
    with correct_path.open("w", encoding="utf-8") as f:
        for rec in correct_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with incorrect_path.open("w", encoding="utf-8") as f:
        for rec in incorrect_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # eval 评估详情
    eval_correct = [d for d in details if d["eval_judgment"] == "CORRECT"]
    eval_incorrect = [d for d in details if d["eval_judgment"] == "INCORRECT"]
    eval_correct_path = run_dir / "eval_correct.json"
    eval_incorrect_path = run_dir / "eval_incorrect.json"
    with eval_correct_path.open("w", encoding="utf-8") as f:
        for d in eval_correct:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with eval_incorrect_path.open("w", encoding="utf-8") as f:
        for d in eval_incorrect:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nSaved: {len(correct_records)} correct → {correct_path}")
    print(f"Saved: {len(incorrect_records)} incorrect → {incorrect_path}")
    print(f"Saved: {len(eval_correct)} eval correct → {eval_correct_path}")
    print(f"Saved: {len(eval_incorrect)} eval incorrect → {eval_incorrect_path}")

    # 错误案例摘要
    if incorrect_records:
        print(f"\nIncorrect ({len(incorrect_records)}):")
        for d in details:
            if d["eval_judgment"] == "INCORRECT":
                print(f"  [{d['query_id']}] pred={d['predicted_answer'][:80]}...")
                if len([x for x in details if x['eval_judgment']=='INCORRECT']) > 10:
                    if d == [x for x in details if x['eval_judgment']=='INCORRECT'][9]:
                        print(f"  ... and {len(incorrect_records)-10} more")
                        break


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
