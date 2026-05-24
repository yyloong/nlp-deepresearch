"""
condense_long.py — 对超长 SFT 样本做离线摘要压缩（复用 Agent.condense 逻辑）

算法（从前往后滑动压缩）：
  对于 token 数超过 target_length 的样本，反复执行：
  1. 从 remaining 开头取最大能放进 target_length 的前缀 front
     → 直接存为训练样本（完整可学，≤ target_length，无需截断）
  2. 把 front 送去 LLM 压缩（condense nudge → submit_condensed_summary）
     → 若模型成功调用工具，存为 condense process 训练样本
  3. remaining = [sys, summary_user] + front 之后的消息
  4. 重复，直到 remaining ≤ target_length
     → 存为最终训练样本（含最终 answer），不再压缩

每个原始超长样本可产生多条训练样本：
  - 若压缩 k 轮：产生 k 个 front 片段 + 最多 k 个 condense process + 1 个最终 continuation
  - 所有样本均 ≤ target_length，可直接用于训练

用法:
    python train/condense_long.py \
        --input  train/sft_data.jsonl \
        --output train/sft_data_condensed.jsonl \
        --agent_config configs/main_agent_smart.yaml \
        --tokenizer Qwen/Qwen3-8B \
        --target_length 16384 \
        --base_url http://127.0.0.1:8000/v1 \
        --model qwen3-8b \
        --concurrency 8
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

from agent.tool_docs import submit_condensed_summary_tool_spec
from agent.vllm_client_async import VLLMClientAsync

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Token counting（与 train.py 保持一致）
# ─────────────────────────────────────────────────────────────────────────────

def count_tokens(tokenizer, messages: List[Dict]) -> int:
    total = 0
    for m in messages:
        role    = m["role"]
        content = m.get("content") or ""
        if role == "tool":
            content = f"<tool_response>\n{content}\n</tool_response>"
        text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        total += len(tokenizer.encode(text, add_special_tokens=False))
    return total


def find_front_chunk(tokenizer, messages: List[Dict], budget: int) -> int:
    """
    从对话开头向后贪心，找最大前缀使 messages[:end_i] 的 token ≤ budget。

    合法的切分点在 **tool 消息之后**：
      front = [..., asst(tool_call), tool]   ← 以完整工具调用+响应结尾
      back  = [asst(继续推理), ...]           ← 以 assistant 消息开头

    压缩完 front 后，summary_user 拼上 back 构成合法对话：
      [sys, summary_user, asst, ...]

    返回 end_i，即 front = messages[:end_i]。
    """
    acc     = 0
    last_ok = 0  # 最近一个合法切分点（在 tool 之后）

    for i, m in enumerate(messages):
        t = count_tokens(tokenizer, [m])
        if acc + t > budget:
            break
        acc += t
        if m["role"] == "tool":
            last_ok = i + 1  # tool 消息之后是合法切分点

    return last_ok


# ─────────────────────────────────────────────────────────────────────────────
# Self-condense（完全对应 Agent.condense()）
# ─────────────────────────────────────────────────────────────────────────────

CONDENSE_NUDGE = (
    "Your context is getting long. Before continuing, please summarize your progress so far. "
    "Report: (1) which tools you used and which documents you retrieved, "
    "(2) your current reasoning strategy, "
    "(3) key findings with docid references, "
    "(4) what still needs to be found. "
    "Call submit_condensed_summary to submit your summary."
)


def _trim_front_to_budget(
    tokenizer,
    front_messages: List[Dict],
    budget: int,
) -> List[Dict]:
    """
    把前半段从头部截断，确保总 token ≤ budget。
    始终保留 system 消息（index=0）和最新的若干条消息（尾部优先）。
    """
    if count_tokens(tokenizer, front_messages) <= budget:
        return front_messages

    system_msg = front_messages[0]
    system_tok = count_tokens(tokenizer, [system_msg])
    remaining   = budget - system_tok

    kept = []
    for m in reversed(front_messages[1:]):
        t = count_tokens(tokenizer, [m])
        if t <= remaining:
            kept.insert(0, m)
            remaining -= t
        else:
            break  # 从头部截掉放不下的部分

    return [system_msg] + kept


CONDENSE_RETRY_NUDGE = (
    "Please call the submit_condensed_summary tool now to submit your summary. "
    "Fill in all required fields: tool_summary, key_thoughts, key_findings, and remaining_to_find."
)

_CONDENSE_LABELS = [
    ("key_thoughts",      "### Reasoning"),
    ("tool_summary",      "### Tools & Documents"),
    ("key_findings",      "### Key Findings"),
    ("remaining_to_find", "### Remaining"),
]

MAX_CONDENSE_RETRIES = 3


def _parse_condense_response(resp: Dict) -> str:
    """从 API 响应中解析 submit_condensed_summary 的结构化摘要。"""
    for tc in (resp.get("tool_calls") or []):
        fn = tc.get("function", {})
        if fn.get("name") == "submit_condensed_summary":
            try:
                args = json.loads(fn.get("arguments", "{}"))
                parts = [
                    f"{label}\n{args[k].strip()}"
                    for k, label in _CONDENSE_LABELS
                    if args.get(k, "").strip()
                ]
                if parts:
                    return "\n\n".join(parts)
            except Exception:
                pass
    return ""


def _build_assistant_msg(resp: Dict) -> Dict:
    msg: Dict = {"role": "assistant"}
    if resp.get("content"):
        msg["content"] = resp["content"]
    if resp.get("tool_calls"):
        msg["tool_calls"] = resp["tool_calls"]
    return msg


async def self_condense_front(
    client: VLLMClientAsync,
    model: str,
    tokenizer,
    front_messages: List[Dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    api_context_limit: int = 40960,
) -> tuple[str, Dict | None]:
    """
    把前半段消息发给模型，追加 condense nudge，让模型调用
    submit_condensed_summary 提交结构化摘要。

    若模型未调用工具，追加提醒消息重试（最多 MAX_CONDENSE_RETRIES 次）。
    重试的上下文仅用于引导本次 API 调用，不会保存到训练样本中。

    返回：
      (summary_text, first_assistant_msg)
      - summary_text       : 结构化摘要字符串，失败时为空串
      - first_assistant_msg: 第一次响应的 assistant 消息（含 tool_calls），
                             用于构建 condense process 训练样本
    """
    condense_nudge_msg = {"role": "user", "content": CONDENSE_NUDGE}
    nudge_tok      = count_tokens(tokenizer, [condense_nudge_msg])
    front_budget   = api_context_limit - max_tokens - nudge_tok - 512
    front_messages = _trim_front_to_budget(tokenizer, front_messages, front_budget)

    condense_tools = [submit_condensed_summary_tool_spec()]
    # 初始对话：front + condense nudge
    working_msgs   = list(front_messages) + [condense_nudge_msg]

    first_assistant_msg: Dict | None = None  # 保存第一次响应，用于训练样本

    for attempt in range(1, MAX_CONDENSE_RETRIES + 1):
        async with semaphore:
            try:
                raw = await client.simple_chat(
                    model=model,
                    messages=working_msgs,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    tools=condense_tools,
                    tool_choice="auto",
                )
            except Exception as e:
                import traceback
                logger.error(
                    f"Condense API call failed (attempt {attempt}/{MAX_CONDENSE_RETRIES}):\n"
                    f"  Type   : {type(e).__name__}\n"
                    f"  Message: {e}\n"
                    f"  Traceback:\n"
                    + "".join(f"    {l}" for l in traceback.format_exc().splitlines(keepends=True))
                )
                return "", None

        choices = raw.get("choices") or []
        if not choices:
            logger.warning(f"Condense API returned empty choices (attempt {attempt})")
            return "", None

        resp = choices[0].get("message") or {}
        assistant_msg = _build_assistant_msg(resp)

        # 保存第一次响应（用于训练样本，反映模型真实的首次尝试）
        if first_assistant_msg is None:
            first_assistant_msg = assistant_msg

        analysis = _parse_condense_response(resp)
        if analysis:
            # 成功：模型正确调用了 submit_condensed_summary
            if attempt > 1:
                logger.debug(f"Condense succeeded on retry attempt {attempt}")
            return analysis, first_assistant_msg

        # 模型未调用工具：追加提醒，重试（重试上下文不进入训练样本）
        logger.debug(
            f"Condense attempt {attempt}: model did not call tool, "
            f"appending retry nudge."
        )
        working_msgs = working_msgs + [
            assistant_msg,
            {"role": "user", "content": CONDENSE_RETRY_NUDGE},
        ]

    logger.warning(
        f"Condense failed after {MAX_CONDENSE_RETRIES} attempts "
        f"(model never called submit_condensed_summary)."
    )
    return "", None


# ─────────────────────────────────────────────────────────────────────────────
# 单样本处理
# ─────────────────────────────────────────────────────────────────────────────

def extract_question(messages: List[Dict]) -> str:
    """
    提取原始问题（完整内容，不截断）。
    SFT 数据的第一条 user message 即为原始问题，与框架中 self._current_question 一致。
    """
    for m in messages:
        if m["role"] == "user":
            return (m.get("content") or "").strip()
    return ""


MAX_CONDENSE_ROUNDS = 20  # 防止死循环的上限


async def process_sample(
    client: VLLMClientAsync,
    model: str,
    tokenizer,
    messages: List[Dict],
    target_length: int,
    condense_max_tokens: int,
    semaphore: asyncio.Semaphore,
    api_context_limit: int = 40960,
    initial_tokens: int = 0,
) -> List[List[Dict]]:
    """
    从前往后滑动压缩，直到剩余对话 ≤ target_length：

    每轮：
      1. 从 remaining 开头，取最大能放进 target_length 的前缀 front
         → 直接存为训练样本①（完整可学，包含这段所有 tool-use）
      2. 把 front 送去压缩（condense nudge → submit_condensed_summary）
         → 若成功调用工具，存训练样本②（condense process）
      3. remaining = [sys, summary_user] + back（front 之后的消息）
      4. 重复，直到 remaining ≤ target_length
         → 存为最终训练样本（含最终 answer），不再压缩

    返回：所有训练样本的列表（每个元素 ≤ target_length）。
    initial_tokens: 第一轮直接使用预计算的 token 数，避免重复 tokenize 大对话。
    """
    question  = extract_question(messages)
    sys_msg   = messages[0]
    remaining = list(messages)
    all_versions: List[List[Dict]] = []

    # 第一轮用调用方预计算的 token 数（来自缓存），避免在事件循环里阻塞性地 tokenize 整个对话
    total = initial_tokens

    for round_idx in range(MAX_CONDENSE_ROUNDS):
        if round_idx > 0:
            # remaining 已经被压缩重建，需要重新计算；放到线程池避免阻塞事件循环
            total = await asyncio.to_thread(count_tokens, tokenizer, remaining)

        if total <= target_length:
            # remaining 已足够短，直接存为最终训练样本（含最终 answer）
            all_versions.append(remaining)
            logger.debug(
                f"Round {round_idx + 1}: remaining fits ({total} tokens), "
                f"saved as final sample."
            )
            break

        # ── 从开头取最大前缀（≤ target_length，末尾必须是 tool 消息） ──────
        # 放入线程池，避免逐条 tokenize 阻塞事件循环
        end_i = await asyncio.to_thread(find_front_chunk, tokenizer, remaining, target_length)

        if end_i <= 1:
            # 单条消息本身超过 target_length，无法再切分，直接停止。
            # 之前已产出的样本正常保留，本轮 remaining 不可用，丢弃。
            logger.warning(
                f"Round {round_idx + 1}: single message exceeds target_length; "
                f"stopping (keeping {len(all_versions)} samples already produced)."
            )
            break

        front = remaining[:end_i]   # [sys, m1...mk]，token ≤ target_length
        back  = remaining[end_i:]   # [m_{k+1}, ...]，不含 system

        logger.debug(
            f"Round {round_idx + 1}: total={total}, "
            f"front={len(front)} msgs, back={len(back)} msgs"
        )

        # ── 训练样本①：front 直接存 ──────────────────────────────────────
        all_versions.append(front)

        # ── 压缩 front → summary ──────────────────────────────────────────
        summary, assistant_msg = await self_condense_front(
            client, model, tokenizer,
            front, condense_max_tokens, semaphore,
            api_context_limit=api_context_limit,
        )

        if not summary or assistant_msg is None:
            # condense 失败：无法构造合法的下一轮对话（back 以 assistant 开头，
            # 没有 summary_user 就缺少上下文），直接停止本样本的处理。
            # front 已作为 sample① 保存，后续片段本轮无法产出。
            logger.debug(
                f"Round {round_idx + 1}: condense failed; "
                f"cannot build valid continuation without summary. Stopping."
            )
            break

        # ── 训练样本②：condense process（仅当模型实际调用了工具时保存） ───
        if assistant_msg.get("tool_calls"):
            condense_process: List[Dict] = list(front) + [
                {"role": "user", "content": CONDENSE_NUDGE},
                assistant_msg,
            ]
            all_versions.append(condense_process)
        else:
            logger.debug(
                f"Round {round_idx + 1}: model used plain-text fallback, "
                f"skipping condense process sample."
            )

        # ── 构建下一轮 remaining ──────────────────────────────────────────
        summary_user: Dict[str, Any] = {
            "role": "user",
            "content": (
                f"{question}\n\n"
                f"[Research Progress Summary]\n"
                f"Your earlier context has been compressed into the following structured summary. "
                f"Use it to continue your research efficiently without repeating work already done.\n\n"
                f"{summary}\n\n"
                f"Continue your research from where you left off."
            ),
        }
        remaining = [sys_msg, summary_user] + back
        logger.debug(
            f"  → new remaining: {len(remaining)} msgs (token count computed next round)"
        )
    else:
        logger.warning(
            f"Reached MAX_CONDENSE_ROUNDS={MAX_CONDENSE_ROUNDS}; "
            f"saving remaining as-is ({total} tokens)."
        )
        all_versions.append(remaining)

    return all_versions


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    logger.info(f"Loading tokenizer from {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=Path(args.tokenizer).is_dir(),
    )

    logger.info(f"Loading data from {args.input} ...")
    samples: List[Dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    logger.info(f"Total samples: {len(samples)}")

    # ── token 数缓存：避免每次重新 tokenize ────────────────────────────────
    cache_path = Path(args.input).with_suffix(".token_counts.json")
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(cached) == len(samples):
            token_counts = cached
            logger.info(f"Loaded token counts from cache: {cache_path}")
        else:
            logger.info(
                f"Cache size mismatch ({len(cached)} vs {len(samples)}), recomputing ..."
            )
            token_counts = None
    else:
        token_counts = None

    if token_counts is None:
        import os
        from multiprocessing.pool import ThreadPool
        from functools import partial
        from tqdm import tqdm

        num_threads = min(os.cpu_count() or 1, 16)
        logger.info(f"Counting tokens with {num_threads} threads ...")
        _count_fn = partial(count_tokens, tokenizer)
        with ThreadPool(num_threads) as pool:
            token_counts = list(tqdm(
                pool.imap(_count_fn, [s["messages"] for s in samples]),
                total=len(samples),
                desc="Counting tokens",
                unit="sample",
                dynamic_ncols=True,
            ))
        cache_path.write_text(json.dumps(token_counts), encoding="utf-8")
        logger.info(f"Token counts saved to cache: {cache_path}")

    # 超过 max_input_tokens 的样本本轮跳过（留待后续批次处理）
    if args.max_input_tokens > 0:
        pairs        = [(s, t) for s, t in zip(samples, token_counts) if t <= args.max_input_tokens]
        n_skipped    = len(samples) - len(pairs)
        samples      = [p[0] for p in pairs]
        token_counts = [p[1] for p in pairs]
        if n_skipped:
            logger.info(
                f"Skipped {n_skipped} samples exceeding max_input_tokens={args.max_input_tokens} "
                f"(not included in output; process separately)"
            )

    long_indices = [i for i, t in enumerate(token_counts) if t > args.target_length]
    logger.info(
        f"Samples to process: {len(samples)} total  |  "
        f"need condense: {len(long_indices)}  |  "
        f"already short: {len(samples) - len(long_indices)}"
    )

    if not long_indices:
        logger.info("Nothing to condense, writing output as-is.")
        all_output = samples
    else:
        client    = VLLMClientAsync(
            base_url=args.base_url,
            api_key=args.api_key,
            max_concurrent=args.concurrency,
        )
        semaphore = asyncio.Semaphore(args.concurrency)

        async def _process(i: int) -> tuple[int, List[List[Dict]], int]:
            orig_tokens = token_counts[i]
            versions    = await process_sample(
                client, args.model, tokenizer,
                samples[i]["messages"],
                args.target_length,
                args.condense_max_tokens,
                semaphore,
                api_context_limit=args.api_context_limit,
                initial_tokens=orig_tokens,  # 避免第一轮重复 tokenize 整个对话
            )
            return i, versions, orig_tokens

        tasks   = [_process(i) for i in long_indices]
        results = await tqdm_asyncio.gather(*tasks, desc="Condensing", unit="sample")

        long_set = set(long_indices)
        # 短样本直接保留
        all_output: List[Dict] = [s for i, s in enumerate(samples) if i not in long_set]
        n_short = len(all_output)

        # 长样本：用所有 versions 完全替换原始样本（不保留原始超长版本）
        total_condensed = 0
        for i, versions, orig_tok in results:
            if not versions:
                continue
            meta = {k: samples[i][k] for k in samples[i] if k != "messages"}
            for v in versions:
                all_output.append({"messages": v, **meta})
                total_condensed += 1

        logger.info(
            f"Condense complete: {len(long_indices)} long samples → "
            f"{total_condensed} training samples "
            f"(front slices + condense process + final continuations)"
        )
        logger.info(
            f"Output composition: {n_short} short (kept as-is) + "
            f"{total_condensed} condensed = {len(all_output)} total"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in all_output:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Written {len(all_output)} samples to {out_path}")

    # 压缩后统计
    still_long = sum(
        1 for s in all_output
        if count_tokens(tokenizer, s["messages"]) > args.target_length
    )
    logger.info(
        f"Still exceeding {args.target_length} tokens: "
        f"{still_long} / {len(all_output)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Condense long SFT samples using self-condense (submit_condensed_summary)"
    )
    parser.add_argument("--input",               required=True,  help="输入 sft_data.jsonl")
    parser.add_argument("--output",              required=True,  help="输出 jsonl 路径")
    parser.add_argument("--tokenizer",           required=True,  help="tokenizer 路径或 HF repo")
    parser.add_argument("--target_length",       type=int, default=16384,
                        help="目标序列长度上限（token 数）")
    parser.add_argument("--base_url",            default="http://127.0.0.1:8000/v1",
                        help="vLLM API base_url")
    parser.add_argument("--api_key",             default="dummy", help="API key")
    parser.add_argument("--model",               required=True,  help="用于压缩的模型名称")
    parser.add_argument("--concurrency",         type=int, default=8, help="并发请求数")
    parser.add_argument("--condense_max_tokens", type=int, default=1024,
                        help="condense 摘要的最大输出 token 数")
    parser.add_argument("--api_context_limit",   type=int, default=40960,
                        help="condense 模型的最大 context 长度（用于截断前半段）")
    parser.add_argument("--max_input_tokens",    type=int, default=0,
                        help="跳过原始 token 数超过此值的样本（0=不过滤）。"
                             "用于分批处理，先处理较短样本")
    args = parser.parse_args()

    asyncio.run(main(args))
