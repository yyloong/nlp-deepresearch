"""
condense_long.py — 对超长 SFT 样本做离线摘要压缩（复用 Agent.condense 逻辑）

对于 token 数超过 target_length 的样本，将其拆分为：
  前半段：system + 早期 tool-use 轮次（探索过程）
  后半段：剩余轮次（含最终 reasoning + answer）

把前半段作为一个完整对话喂给 Agent.condense()，得到 submit_condensed_summary
的四字段结构化摘要，拼合成：
  [system, 摘要_user, 后半段...]

这样既保留了最终 answer，又不完全丢弃早期探索信息，且摘要格式与
正式 condense 完全一致（同一个 submit_condensed_summary 工具调用）。

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


def find_split_point(tokenizer, messages: List[Dict], target_length: int) -> int:
    """
    找最大切分索引 split_i，满足：
      1. messages[split_i:] 的 token ≤ target_length
      2. messages[split_i] 的 role 是 'user' 或 'assistant'（不能是 'tool'）
         —— 避免后半段以孤立的 tool response 开头（缺少前置 tool_call）
      3. 后半段至少含一条 assistant 消息（有可学习 token）

    strategy：从尾部向前贪心积累；找到合法候选后，若该位置是 'tool' 消息，
    则继续向前推直到找到 'user' 或 'assistant' 边界。
    """
    n = len(messages)
    acc = 0
    candidate = n  # 默认：全部归入前半段

    for i in range(n - 1, 0, -1):
        t = count_tokens(tokenizer, [messages[i]])
        if acc + t <= target_length:
            acc += t
            candidate = i
        else:
            break

    # 向前移动直到 candidate 落在合法边界（user 或 assistant）
    split_i = candidate
    while split_i < n and messages[split_i]["role"] == "tool":
        split_i += 1  # 跳过 tool 消息，让它归入前半段（随所属 tool_call 一起被 condense）

    # 如果跳过 tool 后超出范围，退到最后一个 assistant 位置
    if split_i >= n:
        for k in range(n - 1, 0, -1):
            if messages[k]["role"] in ("user", "assistant"):
                split_i = k
                break

    # 确保后半段有 assistant 消息（有可学习 token）
    if not any(m["role"] == "assistant" for m in messages[split_i:]):
        for k in range(n - 1, split_i - 1, -1):
            if messages[k]["role"] == "assistant":
                split_i = k
                break

    return split_i


# ─────────────────────────────────────────────────────────────────────────────
# Self-condense（完全对应 Agent.condense()）
# ─────────────────────────────────────────────────────────────────────────────

CONDENSE_NUDGE = (
    "Your context has been long so far"
    "Please summarize your progress for better perform in the following turn. "
    "Report what tools you used, your key findings with docid references, "
    "your reasoning strategy, and what remains to be found. "
    "You should call submit_condensed_summary to submit your summary."
)


async def self_condense_front(
    client: VLLMClientAsync,
    model: str,
    tokenizer,
    front_messages: List[Dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, Dict | None]:
    """
    把前半段消息发给模型，追加 condense nudge，让模型调用
    submit_condensed_summary 提交结构化摘要。
    完全复用 Agent.condense() 的提示和 tool spec。

    返回：
      (summary_text, assistant_resp_msg)
      - summary_text      : 拼好的摘要字符串，失败时为空串
      - assistant_resp_msg: 模型原始 assistant 消息（含 tool_calls），
                            用于构建 condense 过程的 SFT 训练样本
    """
    condense_nudge_msg = {"role": "user", "content": CONDENSE_NUDGE}
    condense_msgs = list(front_messages) + [condense_nudge_msg]
    condense_tools = [submit_condensed_summary_tool_spec()]

    async with semaphore:
        try:
            raw = await client.simple_chat(
                model=model,
                messages=condense_msgs,
                temperature=0.0,
                max_tokens=max_tokens,
                tools=condense_tools,
                tool_choice="auto",
            )
        except Exception as e:
            logger.warning(f"Condense API call failed: {e}")
            return "", None

    choices = raw.get("choices") or []
    if not choices:
        return "", None
    resp = choices[0].get("message") or {}

    # 解析 submit_condensed_summary 工具调用（与 agent.py 格式完全一致）
    _LABELS = [
        ("tool_summary",      "### Tools & Documents"),
        ("key_thoughts",      "### Reasoning"),
        ("key_findings",      "### Key Findings"),
        ("remaining_to_find", "### Remaining"),
    ]
    analysis = ""
    for tc in (resp.get("tool_calls") or []):
        fn = tc.get("function", {})
        if fn.get("name") == "submit_condensed_summary":
            try:
                args = json.loads(fn.get("arguments", "{}"))
                parts = [
                    f"{label}\n{args[k].strip()}"
                    for k, label in _LABELS
                    if args.get(k, "").strip()
                ]
                analysis = "\n\n".join(parts)
            except Exception:
                pass

    # fallback：取 content（去掉 think blocks）
    if not analysis:
        raw_content = resp.get("content", "") or ""
        raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        analysis = raw_content or "(progress summary unavailable)"

    # 构建 assistant 消息（与框架消息格式一致）
    assistant_msg: Dict = {"role": "assistant"}
    if resp.get("content"):
        assistant_msg["content"] = resp["content"]
    if resp.get("tool_calls"):
        assistant_msg["tool_calls"] = resp["tool_calls"]

    return analysis, assistant_msg


# ─────────────────────────────────────────────────────────────────────────────
# 单样本处理
# ─────────────────────────────────────────────────────────────────────────────

def extract_question(messages: List[Dict]) -> str:
    """
    提取原始问题。SFT 数据的 user message 通常以原始问题开头（第一行），
    与框架中 self._current_question 保持一致。
    """
    for m in messages:
        if m["role"] == "user":
            content = (m.get("content") or "").strip()
            # 取第一个空行之前的内容作为问题（避免带上 context 注释等）
            first_para = content.split("\n\n")[0].strip()
            return first_para[:600]
    return ""


MAX_CONDENSE_ROUNDS = 5  # 最多压缩轮数，防止死循环


async def process_sample(
    client: VLLMClientAsync,
    model: str,
    tokenizer,
    messages: List[Dict],
    target_length: int,
    condense_max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> List[List[Dict]]:
    """
    循环压缩超长样本，收集所有中间压缩结果作为独立训练样本。

    每轮产生一个新的 [system, 摘要_user, 后半段]，即使仍超长也记录下来
    （后续训练时截断处理）。最后一轮产生满足 target_length 的结果。

    返回：所有中间/最终压缩版本的列表，每个元素都是一份独立的训练样本。
    """
    total    = count_tokens(tokenizer, messages)
    question = extract_question(messages)
    all_versions: List[List[Dict]] = []  # 收集每轮压缩结果

    for round_idx in range(MAX_CONDENSE_ROUNDS):
        if total <= target_length:
            break

        split_i = find_split_point(tokenizer, messages, target_length)
        front   = messages[:split_i]   # 含 system
        back    = messages[split_i:]   # 后半段（无 system）

        logger.debug(
            f"Round {round_idx + 1}: total={total} tokens, split_i={split_i}, "
            f"front={len(front)} msgs, back={len(back)} msgs"
        )

        # 若前半段只剩 system（无法再切），从后半段尾部贪心保留后退出
        if len(front) <= 1:
            logger.debug("Front has only system msg; hard-truncating back half.")
            kept_back: List[Dict] = []
            budget = target_length
            for m in reversed(back):
                t = count_tokens(tokenizer, [m])
                if t <= budget:
                    kept_back.insert(0, m)
                    budget -= t
                else:
                    break
            messages = [messages[0]] + kept_back
            all_versions.append(messages)
            break

        summary, assistant_msg = await self_condense_front(
            client, model, tokenizer,
            front, condense_max_tokens, semaphore,
        )

        if not summary or assistant_msg is None:
            logger.debug(f"Round {round_idx + 1}: condense returned empty; dropping front.")
            messages = [messages[0]] + back
            total    = count_tokens(tokenizer, messages)
            continue

        # ── 训练样本 A：前半段 + condense nudge + assistant(submit_condensed_summary)
        # 教模型「如何在上下文过长时做自我压缩」
        # system 保持原样，训练信号来自 nudge→tool_call 的对话 pattern
        # 仅当模型确实调用了 submit_condensed_summary（有 tool_calls）才保存，
        # 纯文本 fallback 的响应不具备工具调用训练价值
        if assistant_msg.get("tool_calls"):
            condense_process_sample: List[Dict] = list(front) + [
                {"role": "user", "content": CONDENSE_NUDGE},
                assistant_msg,
            ]
            all_versions.append(condense_process_sample)
        else:
            logger.debug(f"Round {round_idx + 1}: model used plain text fallback, skipping Sample A.")

        # ── 训练样本 B：system + summary_user + 后半段
        # 教模型「在 condensed context 下继续工作」
        # summary_user 格式与 agent.py condense() 完全一致
        summary_user: Dict[str, Any] = {
            "role": "user",
            "content": (
                f"{question}\n\n"
                f"Following is your previous progress:\n\n{summary}"
            ),
        }
        messages = [messages[0], summary_user] + back
        total    = count_tokens(tokenizer, messages)
        all_versions.append(messages)

        logger.debug(
            f"Round {round_idx + 1}: saved 2 samples "
            f"(condense_process + condensed_continuation), "
            f"continuation={total} tokens, {len(messages)} msgs"
        )

    if total > target_length:
        logger.warning(
            f"Sample still {total} > {target_length} tokens after "
            f"{MAX_CONDENSE_ROUNDS} rounds; last version kept (train.py will truncate)."
        )

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

    # 统计超长样本
    token_counts = [count_tokens(tokenizer, s["messages"]) for s in samples]
    long_indices = [i for i, t in enumerate(token_counts) if t > args.target_length]
    logger.info(
        f"Samples exceeding {args.target_length} tokens: "
        f"{len(long_indices)} / {len(samples)} "
        f"({100 * len(long_indices) / len(samples):.1f}%)"
    )

    extra_samples: List[Dict] = []

    if not long_indices:
        logger.info("Nothing to condense, writing output as-is.")
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
            )
            return i, versions, orig_tokens

        tasks   = [_process(i) for i in long_indices]
        results = await tqdm_asyncio.gather(*tasks, desc="Condensing", unit="sample")

        # 用所有压缩版本替换/扩展原始样本
        extra_samples: List[Dict] = []
        total_new = 0
        for i, versions, orig_tok in results:
            if not versions:
                continue
            # 用最终版本（最短）替换原样本
            samples[i]["messages"] = versions[-1]
            # 其余中间版本作为新样本追加
            for v in versions[:-1]:
                meta = {k: samples[i][k] for k in samples[i] if k != "messages"}
                extra_samples.append({"messages": v, **meta})
                total_new += 1

        logger.info(
            f"Condense complete: {len(long_indices)} long samples → "
            f"{len(long_indices)} final(continuation) + {total_new} others"
            f"(condense_process + intermediate_continuation) = "
            f"{len(long_indices) + total_new} condensed training samples total"
        )

    # 写出：原始短样本 + 压缩后样本（含所有中间版本）
    all_output = samples + extra_samples
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in all_output:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Written {len(all_output)} samples to {out_path} (was {len(samples)})")

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
    args = parser.parse_args()

    asyncio.run(main(args))
