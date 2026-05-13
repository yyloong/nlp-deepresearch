"""Shared token-budget helpers for agent loops and tool payloads."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

_TOOL_TRUNC_MARKER = "\n...[TRUNCATED_BY_CONTEXT_GUARD]"
_JSON_TOOL_SHELL_SLACK = 24


def count_tokens_messages(tok: Any, msg_list: List[Dict[str, Any]]) -> int:
    """Token count of ``json.dumps(msg_list)`` (proxy aligned with agent loop usage)."""
    text = json.dumps(msg_list, ensure_ascii=False)
    return len(tok.encode(text))


def tool_content_token_len(tok: Any, s: str) -> int:
    """Token length of a raw string (e.g. tool ``content``) without chat template."""
    return len(tok.encode(s, add_special_tokens=False))


def truncate_utf8_prefix_to_token_budget(tok: Any, s: str, max_tokens: int) -> str:
    """Keep a UTF-8 prefix of ``s`` whose tokenized length is at most ``max_tokens``."""
    if max_tokens <= 0:
        return ""
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return s
    m_ids = tok.encode(_TOOL_TRUNC_MARKER, add_special_tokens=False)
    marker_toks = len(m_ids)
    if marker_toks >= max_tokens:
        return tok.decode(ids[:max_tokens], skip_special_tokens=True)
    body_toks = max_tokens - marker_toks
    body = tok.decode(ids[:body_toks], skip_special_tokens=True)
    return body + _TOOL_TRUNC_MARKER


def json_with_empty_strings(obj: Any) -> Any:
    """Same JSON shape as ``obj`` but every string value replaced with ``\"\"``."""
    if isinstance(obj, str):
        return ""
    if isinstance(obj, dict):
        return {k: json_with_empty_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_with_empty_strings(x) for x in obj]
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    return str(obj)


def fit_tool_json_under_cap(tok: Any, obj: Any, cap_tokens: int) -> str:
    """Shrink parsed tool JSON so ``json.dumps(obj)`` uses at most ``cap_tokens`` tokens."""

    def _longest_string_cell(o: Any) -> Optional[Tuple[Any, Union[str, int]]]:
        best_toks = -1
        best_cell: Optional[Tuple[Any, Union[str, int]]] = None

        def walk(x: Any) -> None:
            nonlocal best_toks, best_cell
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(v, str):
                        tl = tool_content_token_len(tok, v)
                        if tl > best_toks:
                            best_toks = tl
                            best_cell = (x, k)
                    else:
                        walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(o)
        return best_cell

    obj = json.loads(json.dumps(obj))
    for _ in range(4096):
        ser = json.dumps(obj, ensure_ascii=False)
        total = tool_content_token_len(tok, ser)
        if total <= cap_tokens:
            return ser

        shell_toks = tool_content_token_len(
            tok, json.dumps(json_with_empty_strings(obj), ensure_ascii=False)
        )
        room = cap_tokens - shell_toks - _JSON_TOOL_SHELL_SLACK

        if room < 1:
            if isinstance(obj, list) and len(obj) > 1:
                obj.pop()
                continue
            if isinstance(obj, list) and len(obj) == 1:
                obj.pop()
                return "[]"
            cell = _longest_string_cell(obj)
            if cell is not None:
                parent, key = cell
                parent[key] = ""
                continue
            return ser

        cell = _longest_string_cell(obj)
        if cell is None:
            return ser
        parent, key = cell
        s = parent[key]
        if not isinstance(s, str) or not s:
            if isinstance(obj, list) and len(obj) > 1:
                obj.pop()
                continue
            return ser

        prev_total = total
        parent[key] = truncate_utf8_prefix_to_token_budget(tok, s, max(1, room))
        ser2 = json.dumps(obj, ensure_ascii=False)
        if tool_content_token_len(tok, ser2) >= prev_total:
            if isinstance(obj, list) and len(obj) > 1:
                obj.pop()
                continue
            parent[key] = ""
    return json.dumps(obj, ensure_ascii=False)


def hard_truncate_tail_tool_messages(
    tok: Any,
    messages: List[Dict[str, Any]],
    max_context: int,
) -> None:
    """Mutates trailing consecutive ``role==tool`` messages in place (see agent loop)."""
    tail: List[Dict[str, Any]] = []
    for m in reversed(messages):
        if m.get("role") == "tool":
            tail.append(m)
        else:
            break
    tail = list(reversed(tail))
    if not tail:
        return
    cap_tokens = max(1, max_context // 4)

    for m in tail:
        c = m.get("content")
        if not isinstance(c, str):
            continue
        if tool_content_token_len(tok, c) <= cap_tokens:
            continue
        try:
            parsed = json.loads(c)
        except json.JSONDecodeError:
            m["content"] = truncate_utf8_prefix_to_token_budget(tok, c, cap_tokens)
        else:
            m["content"] = fit_tool_json_under_cap(tok, parsed, cap_tokens)


def trajectory_stats(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def _strip_think_from_text(text: str) -> str:
    """Remove ``<think>...</think>`` blocks (including unclosed) from a string."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def is_truncated_think_response(
    content: str,
    tool_calls: Optional[List[Any]],
) -> bool:
    """Detect whether a model response is a truncated ``<think>`` block with no tool calls.

    This happens when the model output exceeds ``max_tokens`` and gets cut off
    before it finishes writing ``</think>`` and the tool calls.  The response
    has ``content`` that starts with ``<think>`` but lacks ``</think>``, and
    ``tool_calls`` is ``None`` or empty.
    """
    if not content or not content.strip().startswith("<think>"):
        return False
    if tool_calls and len(tool_calls) > 0:
        return False
    # Truncated if <think> present but </think> missing
    if "</think>" in content:
        # Think is closed but may still be effectively empty if content is
        # just the think block with nothing after it
        stripped = _strip_think_from_text(content)
        if stripped:
            return False  # has content after think, probably not truncated
    return True


RETRY_NUDGE = (
    "Your previous response was truncated before you could call any tools. "
    "You MUST call the `search` tool NOW. Do NOT write a long analysis — "
    "just pick the most important clue from the question and search for it. "
    "Be concise: 1-2 sentences of planning, then call the tool."
)


def extract_final_answer(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the final predicted answer from the trajectory.

    1. Find the last assistant message with non-empty content.
    2. Strip ``<think>...</think>`` blocks (including unclosed ones).
    3. If an "Exact Answer:" marker is found, return only the text after it.
    4. Otherwise return the (stripped) full content.
    """
    import re

    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not content:
            continue
        # Strip <think> blocks (closed and unclosed)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL)
        content = content.strip()
        if not content:
            continue
        # Try to extract the "Exact Answer:" portion
        match = re.search(
            r"Exact\s*Answer\s*:\s*(.+?)$", content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            answer = match.group(1).strip()
            # Remove trailing explanation fluff that sometimes leaks
            answer = re.sub(r"\n\s*Explanation\s*:.*$", "", answer, flags=re.IGNORECASE | re.DOTALL)
            return answer.strip()
        return content
    return None
