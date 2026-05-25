#!/usr/bin/env python3
"""
诊断 LiteLLM 代理是否正确传递 tool 消息。

用法:
    python check_tool_support.py                   # 从 secrets.json 读配置
    python check_tool_support.py --base-url ...    # 手动指定
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# 加载 secrets.json
_secrets_file = Path(__file__).parent / "secrets.json"
if _secrets_file.exists():
    for k, v in json.loads(_secrets_file.read_text()).items():
        os.environ.setdefault(k, str(v))

import argparse
from agent.vllm_client_async import VLLMClientAsync

TOOL_SPEC = [{
    "type": "function",
    "function": {
        "name": "get_number",
        "description": "Returns a number",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "integer", "description": "the number"}
            },
            "required": ["value"]
        }
    }
}]

# Round 1: ask model to call the tool
MESSAGES_ROUND1 = [
    {"role": "system", "content": "You are a helpful assistant. When asked for a number, call the get_number tool."},
    {"role": "user",   "content": "Please call get_number with value=42."},
]

# Round 2: inject tool result and ask model to confirm
def build_round2(tool_call_id: str) -> list:
    return MESSAGES_ROUND1 + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {"name": "get_number", "arguments": '{"value": 42}'}
            }]
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": '{"result": 42}'
        },
        {
            "role": "user",
            "content": "What number did the tool return? Reply with just the number."
        }
    ]


async def run_check(client: VLLMClientAsync, model: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Model:    {model}")
    print(f"  is_local: {client._is_local}")
    print(f"{'='*60}\n")

    # ── Round 1: tool call ──────────────────────────────────────────────────
    print("[Round 1] Asking model to call get_number(42)...")
    r1 = await client.simple_chat(
        model=model,
        messages=MESSAGES_ROUND1,
        tools=TOOL_SPEC,
        tool_choice="auto",
        max_tokens=256,
        temperature=0.0,
    )
    msg1 = r1["choices"][0]["message"]
    tool_calls = msg1.get("tool_calls") or []

    if not tool_calls:
        print("  [FAIL] Model did not make a tool call in Round 1.")
        print(f"  Response content: {msg1.get('content','(empty)')[:300]}")
        print("\n  => Tool calling API itself is not working (tools param ignored or model cannot call tools).")
        return

    tc = tool_calls[0]
    call_id = tc["id"]
    call_args = json.loads(tc["function"].get("arguments", "{}"))
    print(f"  [OK] Tool call received: {tc['function']['name']}({call_args})  id={call_id}")

    # ── Round 2: inject tool result ─────────────────────────────────────────
    print("\n[Round 2] Sending tool result (role=tool) and asking model to confirm...")
    msgs2 = build_round2(call_id)
    r2 = await client.simple_chat(
        model=model,
        messages=msgs2,
        tools=TOOL_SPEC,
        tool_choice="auto",
        max_tokens=64,
        temperature=0.0,
    )
    msg2 = r2["choices"][0]["message"]
    content2 = (msg2.get("content") or "").strip()
    print(f"  Model reply: '{content2}'")

    if "42" in content2:
        print("\n  [PASS] Tool message correctly forwarded — model saw the tool result.")
    else:
        print("\n  [FAIL] Tool message likely BLOCKED — model did not reference '42' from the tool result.")
        if not client._flatten_tools:
            print("  => Retrying with flatten_tools_for_proxy=True ...")
            await run_check_flatten(client._client._base_url, str(client._client.api_key), model)
        else:
            print("  Even flatten mode failed — check proxy/model configuration.")

    print()


async def run_check_flatten(base_url: str, api_key: str, model: str) -> None:
    """Re-run Round 2 using flatten_tools_for_proxy mode."""
    from agent.vllm_client_async import _flatten_tool_messages
    call_id = "test_flatten_id"
    msgs2 = build_round2(call_id)
    flat_msgs = _flatten_tool_messages(msgs2)
    print("\n  [Flatten mode] Flattened messages:")
    for m in flat_msgs:
        print(f"    role={m['role']}: {str(m.get('content',''))[:120]}")

    client2 = VLLMClientAsync(
        base_url=base_url, api_key=api_key,
        flatten_tools_for_proxy=True,
    )
    r = await client2.simple_chat(
        model=model,
        messages=flat_msgs,
        tools=TOOL_SPEC,
        tool_choice="auto",
        max_tokens=64,
        temperature=0.0,
    )
    content = (r["choices"][0]["message"].get("content") or "").strip()
    print(f"\n  [Flatten mode] Model reply: '{content}'")
    if "42" in content:
        print("  [PASS] Flatten mode works — tool results delivered as user messages.")
        print("  => flatten_tools_for_proxy=True is already enabled in run_serial.py for remote APIs.")
    else:
        print("  [FAIL] Even flatten mode failed — the model may not support tool calling at all via this proxy.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key",  default=os.environ.get("DEEPSEEK_API_KEY", "dummy"))
    p.add_argument("--model",    default=os.environ.get("DEEPSEEK_MODEL", "qwen_auto"))
    p.add_argument("--no-skip-transform", action="store_true",
                   help="Disable x-litellm-skip-transform header (test without it)")
    args = p.parse_args()

    client = VLLMClientAsync(
        base_url=args.base_url,
        api_key=args.api_key,
        skip_litellm_transform=not args.no_skip_transform,
    )

    print(f"Testing tool message support...")
    print(f"  base_url:              {args.base_url}")
    print(f"  skip_litellm_transform: {not args.no_skip_transform}")

    asyncio.run(run_check(client, args.model))


if __name__ == "__main__":
    main()
