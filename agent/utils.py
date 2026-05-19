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


def _is_search_result_list(obj: Any) -> bool:
    """Check if *obj* is a list of dicts containing ``snippet`` or ``text`` fields
    (search or get-document results)."""
    if not isinstance(obj, list) or len(obj) == 0:
        return False
    return all(
        isinstance(item, dict) and ("snippet" in item or "text" in item)
        for item in obj
    )


def _fit_results_evenly(tok: Any, results: List[Dict[str, Any]], cap_tokens: int) -> str:
    """Fit a list of search-result dicts under *cap_tokens* by truncating every
    content field (``snippet`` / ``text``) proportionally, rather than dropping
    tail items."""
    n = len(results)

    # ── Build a shell list with all content strings replaced by "" ──
    shell_list: List[Dict[str, Any]] = []
    for item in results:
        shell: Dict[str, Any] = {}
        for k, v in item.items():
            if isinstance(v, str) and k in ("snippet", "text"):
                shell[k] = ""
            else:
                shell[k] = v
        shell_list.append(shell)

    shell_toks = tool_content_token_len(
        tok, json.dumps(shell_list, ensure_ascii=False)
    )
    budget = cap_tokens - shell_toks - _JSON_TOOL_SHELL_SLACK

    if budget <= 0:
        # Even the structural shell exceeds budget — must drop items
        while len(results) > 1:
            results.pop()
            shell_list.pop()
            shell_toks = tool_content_token_len(
                tok, json.dumps(shell_list, ensure_ascii=False)
            )
            budget = cap_tokens - shell_toks - _JSON_TOOL_SHELL_SLACK
            if budget > 0:
                break
        # For the last item, aggressively truncate its content field
        if len(results) == 1 and budget <= 0:
            item = results[0]
            for k in ("snippet", "text"):
                if k in item and isinstance(item[k], str) and item[k]:
                    item[k] = item[k][:200] + _TOOL_TRUNC_MARKER
                    break
            return json.dumps(results, ensure_ascii=False)
        if len(results) == 0:
            return "[]"

    # ── Collect content strings with their positions ──
    # (item_idx, key, content_str)
    content_entries: List[Tuple[int, str, str]] = []
    for i, item in enumerate(results):
        for key in ("snippet", "text"):
            if key in item and isinstance(item[key], str) and item[key]:
                content_entries.append((i, key, item[key]))
                break  # one content field per item

    if not content_entries:
        return json.dumps(results, ensure_ascii=False)

    # ── Distribute budget proportionally by current content length ──
    total_chars = sum(len(s) for _, _, s in content_entries)
    if total_chars == 0:
        total_chars = 1  # avoid division by zero

    for i, key, s in content_entries:
        # Proportional budget, but at least 1 token per item
        share = max(1, int(budget * len(s) / total_chars))
        results[i][key] = truncate_utf8_prefix_to_token_budget(tok, s, share)

    return json.dumps(results, ensure_ascii=False)


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

    # ── Early path: search-result lists → truncate snippets evenly ──
    if _is_search_result_list(obj):
        return _fit_results_evenly(tok, obj, cap_tokens)

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
                # Never drop the last item — aggressively truncate it instead
                item = obj[0]
                if isinstance(item, dict):
                    for k in list(item.keys()):
                        if isinstance(item[k], str) and item[k]:
                            item[k] = item[k][:200] + _TOOL_TRUNC_MARKER
                            break
                    else:
                        obj.pop()
                        return "[]"
                continue
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
            # Raw-token truncation didn't shrink the JSON-serialized form enough,
            # usually due to JSON-escaping inflation (e.g. \n → \\n, \" → \\\").
            # Retry with progressively smaller budgets rather than blanking.
            if isinstance(obj, list) and len(obj) > 1:
                obj.pop()
                continue
            for factor in (0.5, 0.25, 0.1, 0.05):
                reduced = max(1, int(room * factor))
                parent[key] = truncate_utf8_prefix_to_token_budget(tok, s, reduced)
                ser3 = json.dumps(obj, ensure_ascii=False)
                if tool_content_token_len(tok, ser3) < prev_total:
                    break  # made progress
            else:
                # Extreme case — keep a minimal prefix rather than blank
                parent[key] = truncate_utf8_prefix_to_token_budget(tok, s, 50)
    return json.dumps(obj, ensure_ascii=False)


def hard_truncate_tail_tool_messages(
    tok: Any,
    messages: List[Dict[str, Any]],
    max_context: int,
    label: str = "",
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
    cap_tokens = max(1, max_context // 3)

    truncated_count = 0
    for m in tail:
        c = m.get("content")
        if not isinstance(c, str):
            continue
        before = tool_content_token_len(tok, c)
        if before <= cap_tokens:
            continue
        try:
            parsed = json.loads(c)
        except json.JSONDecodeError:
            m["content"] = truncate_utf8_prefix_to_token_budget(tok, c, cap_tokens)
        else:
            m["content"] = fit_tool_json_under_cap(tok, parsed, cap_tokens)
        after = tool_content_token_len(tok, m["content"])
        truncated_count += 1
        prefix = f"  [truncate] {label}" if label else "  [truncate]"
        print(f"{prefix} tool result {before}→{after} tokens (cap={cap_tokens})", flush=True)


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
    "You appear to be stuck in a repetitive reasoning loop. "
    "Try to think in a different way"
)


def validate_tool_call(
    tc: Dict[str, Any],
    tool_specs: List[Dict[str, Any]],
) -> Optional[str]:
    """Validate a single tool call against the tool specifications.

    Checks:
    1. Tool name exists in the spec list.
    2. ``arguments`` is valid JSON.
    3. All required parameters (declared in the spec) are present.

    Returns an error message string if validation fails, or ``None`` if valid.
    """
    fn = tc.get("function", {})
    name = fn.get("name", "")
    args_str = fn.get("arguments", "")

    # 1. Tool name exists
    tool_spec = None
    for t in tool_specs:
        if t.get("function", {}).get("name") == name:
            tool_spec = t["function"]
            break

    if tool_spec is None:
        available = [t["function"]["name"] for t in tool_specs]
        return (
            f"Unknown tool '{name}'. "
            f"Available tools: {', '.join(available)}. "
            f"Please use one of the available tools."
        )

    # 2. Valid JSON
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError as e:
        return (
            f"Invalid JSON in arguments: {e}. "
            f"Arguments must be valid JSON. "
            f"Received (truncated): {args_str[:200]}"
        )

    # 3. Required parameters present
    if not isinstance(args, dict):
        return (
            f"Arguments must be a JSON object (dict), got {type(args).__name__}. "
            f"Received: {args_str[:200]}"
        )

    required = tool_spec.get("parameters", {}).get("required", [])
    missing = [p for p in required if p not in args or args[p] is None or args[p] == ""]

    if missing:
        param_info = tool_spec.get("parameters", {}).get("properties", {})
        hints = []
        for p in missing:
            desc = param_info.get(p, {}).get("description", "")
            hints.append(f"'{p}' ({desc})" if desc else f"'{p}'")
        return (
            f"Missing required parameter(s): {', '.join(hints)}. "
            f"Please provide values for: {', '.join(missing)}."
        )

    return None


def extract_final_answer(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the final predicted answer from the trajectory.

    1. First check for ``submit_answer`` tool calls (new flow).
    2. Fall back to text-based extraction from assistant content.
    """
    import re

    # ── Priority 1: submit_answer tool call ──
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            if tc.get("function", {}).get("name") == "submit_answer":
                try:
                    args = json.loads(tc["function"].get("arguments", "{}"))
                    answer = args.get("answer", "").strip()
                    if answer:
                        return answer
                except (json.JSONDecodeError, TypeError):
                    pass

    # ── Priority 2: text-based extraction (legacy / no-tool-call answers) ──
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
            answer = re.sub(r"\n\s*Explanation\s*:.*$", "", answer, flags=re.IGNORECASE | re.DOTALL)
            return answer.strip()
        return content
    return None
