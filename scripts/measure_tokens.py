#!/usr/bin/env python3
"""Measure token consumption of a trajectory to diagnose why condensation triggers so early."""

import json
import sys
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer

TOKENIZER_PATH = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH, trust_remote_code=True, local_files_only=True,
)


def count_tokens_messages(msg_list):
    """Exact same logic as agent.utils.count_tokens_messages."""
    text = json.dumps(msg_list, ensure_ascii=False)
    return len(tok.encode(text))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/submission_20260513_124203.jsonl"

    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for idx, rec in enumerate(records[:8]):
        qid = rec["query_id"]
        msgs = rec.get("messages", [])

        total = count_tokens_messages(msgs)
        print(f"\n{'='*60}")
        print(f"  query_id={qid}  TOTAL: {total} tokens  ({len(msgs)} messages)")
        print(f"{'='*60}")

        running = 0
        for i, m in enumerate(msgs):
            role = m.get("role", "?")
            content = m.get("content", "") or ""
            tc = m.get("tool_calls")
            prefix = [msgs[:i+1]]
            cum = count_tokens_messages(prefix[0])
            incr = cum - running
            running = cum

            if role == "tool":
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        desc = f"[{len(parsed)} docs]"
                    elif isinstance(parsed, dict) and "error" in parsed:
                        desc = f"[error: {parsed['error'][:50]}]"
                    else:
                        desc = f"[dict]"
                except json.JSONDecodeError:
                    desc = content[:60] + "..." if len(content) > 60 else content
            elif role == "assistant":
                stripped = content[:80].replace("\n", "\\n")
                desc = f'"{stripped}..."' if len(content) > 80 else f'"{content}"'
                if tc:
                    desc += f" + {len(tc)} tool_calls"
            elif role == "system":
                desc = f"[system prompt, {len(content)} chars]"
            else:
                desc = content[:80].replace("\n", "\\n") + ("..." if len(content) > 80 else "")

            bar = "█" * min(50, incr // 80)
            print(f"  [{i:2d}] {role:10s}  +{incr:>5d} → {cum:>6d}  {bar}  {desc}")

        # Also compute raw content vs JSON overhead
        raw_total = 0
        for m in msgs:
            c = m.get("content", "") or ""
            raw_total += len(tok.encode(c, add_special_tokens=False))
        overhead = total - raw_total
        print(f"  {'─'*50}")
        print(f"  Raw content tokens: {raw_total}  JSON overhead: {overhead}  ({overhead/max(1,total)*100:.0f}%)")


if __name__ == "__main__":
    main()
