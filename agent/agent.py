"""
Unified Agent class — the single agent implementation shared by ALL agent types.

Every agent (main, search, verify, sub_summary) uses this exact class.
Different behaviors are determined entirely by YAML configuration:
  - system_prompt, tools, end_tool, max_turn, etc.

Key methods:
  - chat():            single model call
  - chat_with_tool_retry(): chat + retry on format/no-tool errors (retry context removed)
  - condense():        context compression via structured summary
  - run():             main agent loop (condense -> chat -> execute tools -> repeat)
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from .tool_docs import (
    call_subagents_tool_spec,
    get_document_tool_spec,
    give_feedback_tool_spec,
    judge_relevance_tool_spec,
    report_surrender_verdict_tool_spec,
    search_tool_spec,
    smart_search_tool_spec,
    submit_answer_tool_spec,
    submit_condensed_summary_tool_spec,
    submit_summary_tool_spec,
)
from .utils import count_tokens_messages, hard_truncate_tail_tool_messages

# ═══════════════════════════════════════════════════════════════
# Default condense prompts (override-able via YAML)
# ═══════════════════════════════════════════════════════════════

class Agent:
    """Unified agent for all roles: main research, search, verify, sub_summary.

    Initialised from a YAML config file. All agent types share this class;
    behaviour is determined by the config (prompt, tools, end_tool, etc.).
    """

    def __init__(
        self,
        config_path: str,
        *,
        client: Any = None,
        tokenizer: Any = None,
        tool_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        name: str = "",
        trajectory_dir: str = "",
    ) -> None:
        # ── Load YAML config ──
        with open(config_path, "r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)

        self.config_path = config_path
        self.name: str = name or cfg.get("agent_type", "agent")
        self.trajectory_dir: str = trajectory_dir
        self.agent_type: str = cfg.get("agent_type", "main")

        # Model / client
        self.client = client
        self.model: str = cfg.get("model", "qwen_auto")
        self.tokenizer = tokenizer

        # Generation params
        self.temperature: float = float(cfg.get("temperature", 0.0))
        self.max_tokens: int = int(cfg.get("max_tokens", 4096))
        self.max_context: int = int(cfg.get("max_context", 40960))
        self.max_turn: int = int(cfg.get("max_turn", 10))
        self.enable_thinking: bool = bool(cfg.get("enable_thinking", True))
        self.max_tool_calls_per_turn: int = int(cfg.get("max_tool_calls_per_turn", 1))
        self.max_tool_retries: int = int(cfg.get("max_tool_retries", 2))
        self.condense_token_threshold: float = float(cfg.get("condense_token_threshold", 0.5))

        # Prompts
        self.system_prompt: str = cfg.get("system_prompt", "").strip()

        # Tool setup
        self.end_tool: str = cfg.get("end_tool", "submit_answer")
        self._tool_names: List[str] = list(cfg.get("tools", []))
        self._tool_config: Dict[str, Any] = dict(cfg.get("tool_config", {}))
        # search_k priority: tool_config.<tool>.search_k > tool_config.search.search_k > default 5
        _search_cfg = self._tool_config.get("smart_search", {}) or self._tool_config.get("search", {})
        self.search_k: int = int(_search_cfg.get("search_k", 5))
        self.tool_specs: List[Dict[str, Any]] = self._build_tool_specs()
        self.tool_registry: Dict[str, Callable[..., Any]] = tool_registry or {}

        # Interpolate placeholders in prompt
        self.system_prompt = self.system_prompt.replace("{search_k}", str(self.search_k))

        # Extra payload for model calls
        # chat_template_kwargs only applies to local Qwen3 via vLLM; remote APIs (DeepSeek etc.) ignore it.
        self.extra_payload: Dict[str, Any] = dict(cfg.get("extra_payload", {}))
        is_local = cfg.get("_is_local_model", True)  # set False by run_serial when using remote API
        if is_local and not self.enable_thinking:
            self.extra_payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        # ── Per-run state ──
        self._current_question: str = ""
        self._trajectory: List[Dict[str, Any]] = []
        self._condense_sessions: List[Dict[str, Any]] = []

    # ── Tool spec builder ─────────────────────────

    def _build_tool_specs(self) -> List[Dict[str, Any]]:
        """Build OpenAI-format tool specs from the agent's YAML tool list."""
        _map: Dict[str, Any] = {
            "search": search_tool_spec,
            "smart_search": smart_search_tool_spec,
            "get_document": get_document_tool_spec,
            "submit_answer": submit_answer_tool_spec,
            "call_subagents": call_subagents_tool_spec,
            "give_feedback": give_feedback_tool_spec,
            "judge_relevance": judge_relevance_tool_spec,
            "submit_summary": submit_summary_tool_spec,
            "submit_condensed_summary": submit_condensed_summary_tool_spec,
            "report_surrender_verdict": report_surrender_verdict_tool_spec,
        }
        specs = []
        for name in self._tool_names:
            fn = _map.get(name)
            if fn is None:
                raise KeyError(f"Unknown tool: {name}")
            try:
                specs.append(fn(self.search_k))
            except TypeError:
                specs.append(fn())
        return specs

    # ═══════════════════════════════════════════════════════════════
    # Model calling
    # ═══════════════════════════════════════════════════════════════

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        """Single async model call. Returns the full raw response dict (includes usage)."""
        return await self.client.simple_chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools if tools is not None else self.tool_specs,
            tool_choice=tool_choice,
            extra_payload=self.extra_payload if self.extra_payload else None,
        )

    # ═══════════════════════════════════════════════════════════════
    # Chat with tool-call retry
    # ═══════════════════════════════════════════════════════════════

    async def chat_with_tool_retry(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        """Call chat() with retry logic for format / no-tool-call errors.

        On retry success the intermediate nudge context is discarded;
        only the final successful assistant message is returned.
        """
        _tools = tools if tools is not None else self.tool_specs
        internal_msgs = list(messages)

        for attempt in range(self.max_tool_retries + 1):
            raw = await self.chat(internal_msgs, _tools, tool_choice)
            resp: Dict[str, Any] = raw.get("choices", [{}])[0].get("message", {})
            usage = raw.get("usage", {})

            tc = resp.get("tool_calls")
            content = resp.get("content", "") or ""

            # Log model response
            prompt_tok = usage.get("prompt_tokens", "?")
            comp_tok = usage.get("completion_tokens", "?")
            print(f"    │ [model] prompt_tokens={prompt_tok}  completion_tokens={comp_tok}", flush=True)

            # Log think blocks (avoid double-counting: if unclosed, don't count closed blocks)
            think_blocks = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
            # Only flag unclosed if </think> is truly missing after the last <think>
            last_think = content.rfind("<think>")
            last_close = content.rfind("</think>")
            unclosed = last_think > last_close
            if think_blocks or unclosed:
                n_blocks = len(think_blocks) + (1 if unclosed else 0)
                total_tok = sum(len(self.tokenizer.encode(b)) for b in think_blocks)
                if unclosed and last_think >= 0:
                    total_tok += len(self.tokenizer.encode(content[last_think + len("<think>"):]))
                print(f"    │ [think] {n_blocks} block(s), {total_tok} tokens total", flush=True)

            # Strip think blocks for content display
            non_think = content
            if think_blocks or unclosed:
                non_think = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                non_think = re.sub(r"<think>.*$", "", non_think, flags=re.DOTALL).strip()
            if non_think:
                n_tok = len(self.tokenizer.encode(non_think))
                print(f"    │ [content] ({n_tok} tokens):", flush=True)
                for line in non_think.split("\n")[:20]:
                    print(f"    │ {line}", flush=True)
            elif tc:
                print(f"    │ [content] (tool-call only, no text)", flush=True)

            if tc:
                print(f"    │ [tool_calls] {len(tc)} call(s):", flush=True)
                for tci, tc_item in enumerate(tc):
                    fn = tc_item.get("function", {})
                    name = fn.get("name", "?")
                    try:
                        args_parsed = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args_parsed = fn.get("arguments", "{}")
                    args_preview = json.dumps(args_parsed, ensure_ascii=False, indent=2)
                    print(f"    │   [{tci}] {name}", flush=True)
                    for line in args_preview.split("\n")[:10]:
                        print(f"    │     {line}", flush=True)

            # ── Check: completion truncated (hit max_tokens) ──
            comp_tokens = int(usage.get("completion_tokens", 0))
            content_str = resp.get("content", "") or ""
            think_open = content_str.count("<think>")
            think_close = content_str.count("</think>")
            truncated_think = think_open > think_close and comp_tokens >= self.max_tokens * 0.9

            if truncated_think and tc and attempt < self.max_tool_retries:
                print(f"    ⚠ completion truncated ({comp_tokens}/{self.max_tokens} tokens), unclosed think, retrying", flush=True)
                internal_msgs.append(resp)
                nudge = {
                    "role": "user",
                    "content": "Your response was cut off. Based on what you have so far, call your tools now. Be concise."
                }
                internal_msgs.append(nudge)
                continue

            # ── Check: no tool calls ──
            if not tc:
                if attempt < self.max_tool_retries:
                    nudge = {
                        "role": "user",
                        "content": (
                            "You did not call any tool. You MUST call a tool to make progress: "
                            "use the available tools to search, read documents, or submit your answer. "
                            "Do NOT output plain text — always call a tool."
                        ),
                    }
                    internal_msgs.append(nudge)
                    print(f"    ⚠ no tool call, retry {attempt + 1}/{self.max_tool_retries}", flush=True)
                    continue
                return resp

            # ── Check: tool call validation ──
            from .utils import validate_tool_call

            all_errors: List[Dict[str, str]] = []
            for tc_item in tc:
                err = validate_tool_call(tc_item, _tools)
                if err:
                    all_errors.append({
                        "tool_name": tc_item.get("function", {}).get("name", "?"),
                        "message": err,
                    })
            if all_errors:
                if attempt < self.max_tool_retries:
                    error_lines = [f"- `{e['tool_name']}`: {e['message']}" for e in all_errors]
                    nudge = {
                        "role": "user",
                        "content": (
                            "Your tool call(s) failed validation:\n\n"
                            + "\n".join(error_lines)
                            + "\n\nPlease correct the error(s) and try again."
                        ),
                    }
                    internal_msgs.append(nudge)
                    print(f"    ⚠ tool validation failed: {len(all_errors)} error(s), retry {attempt + 1}/{self.max_tool_retries}", flush=True)
                    for e in all_errors:
                        print(f"       {e['tool_name']}: {e['message']}", flush=True)
                    continue
                return resp

            # ── Success ──
            return resp

        return resp

    # ═══════════════════════════════════════════════════════════════
    # Context condensation
    # ═══════════════════════════════════════════════════════════════

    async def condense(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Self-condense: ask the model to summarize its own progress.

        Injects a user message asking the model to summarize, then the model
        calls submit_condensed_summary. The model sees its own context naturally
        -- no preprocessing of think blocks or tool calls needed.
        Returns ``[system_msg, summary_user_msg]``.
        """
        if len(messages) <= 4:
            return messages

        question = self._current_question
        before = count_tokens_messages(self.tokenizer, messages)

        condense_nudge = {
            "role": "user",
            "content": (
                "Your context is getting long. Before continuing, please summarize your progress so far. "
                "Report: (1) which tools you used and which documents you retrieved, "
                "(2) your current reasoning strategy, "
                "(3) key findings with docid references, "
                "(4) what still needs to be found. "
                "Call submit_condensed_summary to submit your summary."
            ),
        }
        condense_msgs = list(messages) + [condense_nudge]
        condense_tools = [submit_condensed_summary_tool_spec()]

        print(f"  [condense] self-condense (context~{before} tokens)...", flush=True)

        analysis = ""
        session_msgs: List[Dict[str, Any]] = []

        try:
            resp = await self.chat_with_tool_retry(condense_msgs, condense_tools, "auto")
        except Exception:
            analysis = "(condense failed)"
        else:
            for t in (resp.get("tool_calls") or []):
                fn = t.get("function", {})
                if fn.get("name") == "submit_condensed_summary":
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                        _LABELS = [
                            ("key_thoughts",       "### Reasoning"),
                            ("tool_summary",       "### Tools & Documents"),
                            ("key_findings",       "### Key Findings"),
                            ("remaining_to_find",  "### Remaining"),
                        ]
                        parts = [
                            f"{label}\n{args[k].strip()}"
                            for k, label in _LABELS
                            if args.get(k, "").strip()
                        ]
                        analysis = "\n\n".join(parts)
                    except Exception:
                        pass
            if not analysis:
                raw = resp.get("content", "") or ""
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                analysis = raw or "(progress summary unavailable)"

        session_msgs.append(resp)

        summary_msg: Dict[str, Any] = {
            "role": "user",
            "content": (
                f"{question}\n\n"
                f"[Research Progress Summary]\n"
                f"Your earlier context has been compressed into the following structured summary. "
                f"Use it to continue your research efficiently without repeating work already done.\n\n"
                f"{analysis}\n\n"
                f"Continue your research from where you left off."
            ),
        }

        condensed: List[Dict[str, Any]] = [messages[0], summary_msg]

        after = count_tokens_messages(self.tokenizer, condensed)
        print(f"  [condense] {before} -> {after} tokens ({len(messages)} -> {len(condensed)} messages)", flush=True)

        self._condense_sessions.append({"messages": session_msgs})
        return condensed


    # ═══════════════════════════════════════════════════════════════
    # Main agent loop
    # ═══════════════════════════════════════════════════════════════

    async def _run_inner(self, user_prompt: str) -> List[Dict[str, Any]]:
        """Execute the full agent loop.

        1. Initialize messages with system prompt + user prompt
        2. Loop:
           a. Condense if token count exceeds threshold
           b. Call model via chat_with_tool_retry
           c. Execute tool calls
           d. Check for end_tool or max_turn
        3. Return trajectory

        Agent communication is entirely through tools — when this agent calls
        a tool like submit_answer or call_subagents, the tool implementation
        may invoke other agents internally.
        """
        self._current_question = user_prompt
        self._trajectory = []
        self._condense_sessions = []

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self._trajectory = [dict(m) for m in messages]

        finish_reason = "max_turns"
        turn_stats: List[Dict[str, Any]] = []

        for turn in range(self.max_turn):
            t_turn_start = time.time()
            n_tokens_before = count_tokens_messages(self.tokenizer, messages)
            n_msgs = len(messages)

            role_counts = {
                "sys": sum(1 for m in messages if m.get("role") == "system"),
                "usr": sum(1 for m in messages if m.get("role") == "user"),
                "asst": sum(1 for m in messages if m.get("role") == "assistant"),
                "tool": sum(1 for m in messages if m.get("role") == "tool"),
            }
            print(
                f"  ┌─ [{self.name}] Turn {turn + 1}/{self.max_turn} | "
                f"msgs: {n_msgs} (sys:{role_counts['sys']} usr:{role_counts['usr']} "
                f"asst:{role_counts['asst']} tool:{role_counts['tool']}) | "
                f"tokens: {n_tokens_before} | {time.strftime('%H:%M:%S')}",
                flush=True,
            )

            # ── Pre-call condense ──
            condense_threshold_tokens = int(self.max_context * self.condense_token_threshold)
            safe_limit = self.max_context - self.max_tokens - 2000

            if n_tokens_before > safe_limit:
                t_cs = time.time()
                condensed = await self.condense(messages)
                messages = condensed
                self._trajectory.append({
                    "role": "user",
                    "content": condensed[1]["content"] if len(condensed) > 1 else "",
                    "_condensed": True,
                })
                n_tokens_before = count_tokens_messages(self.tokenizer, messages)
                print(f"  ↻ pre-call condensed -> {n_tokens_before} tokens ({time.time() - t_cs:.1f}s)", flush=True)

            if n_tokens_before > safe_limit:
                print(f"  ⚠ context still {n_tokens_before} > {safe_limit} after condense, forcing early stop", flush=True)
                finish_reason = "context_overflow"
                break

            # ── Model call with tool retry ──
            resp = await self.chat_with_tool_retry(messages, self.tool_specs)

            # Enforce max tool calls per turn
            tc = resp.get("tool_calls")
            if tc and len(tc) > self.max_tool_calls_per_turn:
                n_truncated = len(tc) - self.max_tool_calls_per_turn
                resp["tool_calls"] = tc[:self.max_tool_calls_per_turn]
                tc = resp["tool_calls"]
                print(f"    ⚠ tool calls truncated ({n_truncated} dropped)", flush=True)

            # Log tool calls
            tc_names = [t["function"]["name"] for t in tc] if tc else []

            # Handle no-tool-call (should not happen with chat_with_tool_retry, but guard)
            if not tc:
                nudge = {
                    "role": "user",
                    "content": (
                        "You did not call any tool. You MUST call a tool to make progress: "
                        "use the available tools to search, read documents, or submit your answer. "
                        "Do NOT output plain text — always call a tool."
                    ),
                }
                messages.append(nudge)
                self._trajectory.append({
                    "role": "user",
                    "content": "[NUDGE] Model returned no tool call. Reminding to use a tool.",
                })
                self._trajectory.append(copy.deepcopy(resp))
                n_tokens_after = count_tokens_messages(self.tokenizer, messages)
                print(f"  └─ turn {turn + 1:2d} done | tokens: {n_tokens_before:5d} -> {n_tokens_after:5d} | tools: (none) | {time.time() - t_turn_start:.1f}s", flush=True)
                continue

            # ── Append assistant message to messages and trajectory ──
            messages.append(resp)
            self._trajectory.append(copy.deepcopy(resp))

            # ── Check for end_tool BEFORE execution ──
            # (end_tool terminates the loop after execution)
            end_tool_called = any(t["function"]["name"] == self.end_tool for t in tc)

            # ── Execute tool calls ──
            for tc_item in tc:
                fn = tc_item["function"]
                name = fn["name"]
                call_id: str = tc_item.get("id", "")
                args_str: str = fn.get("arguments", "{}")

                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError) as exc:
                    tool_result = json.dumps({
                        "error": f"Failed to parse arguments as JSON: {exc}. Received (truncated): {args_str[:300]}",
                    })
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})
                    self._trajectory.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})
                    continue

                if name not in self.tool_registry:
                    tool_result = json.dumps({
                        "error": f"Unknown tool '{name}'. Available: {list(self.tool_registry.keys())}",
                    })
                else:
                    try:
                        fn_impl = self.tool_registry[name]
                        import inspect as _inspect
                        print(f"    [tool-exec] calling {name}({list(args.keys())}) async={_inspect.iscoroutinefunction(fn_impl)}", flush=True)
                        if _inspect.iscoroutinefunction(fn_impl):
                            raw = await fn_impl(**args)
                        else:
                            raw = fn_impl(**args)
                        tool_result = json.dumps(raw, ensure_ascii=False)
                    except TypeError as exc:
                        tool_result = json.dumps({
                            "error": f"Invalid arguments for '{name}': {exc}.",
                        })
                    except Exception as exc:
                        print(f"    [tool-exec] ERROR: {name} failed: {exc}", flush=True)
                        tool_result = json.dumps({
                            "error": f"Tool '{name}' execution failed: {exc}.",
                        })

                # Log tool result
                try:
                    parsed_result = json.loads(tool_result)
                    if isinstance(parsed_result, dict):
                        keys = list(parsed_result.keys())
                        result_tokens = count_tokens_messages(self.tokenizer, [{"role": "user", "content": tool_result}])
                        print(f"    │ [tool_result] id={call_id}  tokens={result_tokens}  keys={keys}", flush=True)
                        if "error" in parsed_result:
                            print(f"    │   ERROR: {str(parsed_result['error'])[:300]}", flush=True)
                        elif name == "search" and isinstance(parsed_result, list):
                            print(f"    │   search returned {len(parsed_result)} items", flush=True)
                        elif name == "call_subagents":
                            print(f"    │   call_subagents returned {len(parsed_result)} results", flush=True)
                        elif name == "submit_answer":
                            verdict = "CORRECT" if parsed_result.get("is_correct") else "INCORRECT"
                            print(f"    │ [verify] {verdict} — {parsed_result.get('reason','')[:200]}", flush=True)
                except (json.JSONDecodeError, TypeError):
                    pass

                messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})
                self._trajectory.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})

            n_tokens_after = count_tokens_messages(self.tokenizer, messages)
            tc_str = ", ".join(tc_names) if tc_names else "(none)"
            elapsed = time.time() - t_turn_start
            print(f"  └─ [{self.name}] turn {turn + 1:2d} done | tokens: {n_tokens_before:5d} -> {n_tokens_after:5d} (Δ{n_tokens_after - n_tokens_before:+d}) | tools: {tc_str} | {elapsed:.1f}s", flush=True)
            turn_stats.append({"turn": turn + 1, "n_tokens_before": n_tokens_before, "n_tokens_after": n_tokens_after, "tool_calls": tc_names})

            if end_tool_called:
                # Check if submit_answer was rejected — if so, continue rather than terminate
                should_stop = True
                if self.end_tool == "submit_answer":
                    # Look at the last tool result for submit_answer
                    for m in reversed(messages):
                        if m.get("role") == "tool":
                            try:
                                fb = json.loads(m.get("content", "{}"))
                                if isinstance(fb, dict) and not fb.get("is_correct", True):
                                    should_stop = False
                                    print(f"    ⚠ [{self.name}] answer REJECTED by verify agent, continuing search...", flush=True)
                            except (json.JSONDecodeError, TypeError):
                                pass
                            break
                if should_stop:
                    finish_reason = f"{self.end_tool}_confirmed"
                    print(f"    ✓ [{self.name}] end_tool '{self.end_tool}' called, terminating loop", flush=True)
                    break

            # ── Post-turn condense check ──
            used = count_tokens_messages(self.tokenizer, messages)
            if used > condense_threshold_tokens:
                last = messages[-1] if messages else None
                if last is not None and last.get("role") == "tool":
                    hard_truncate_tail_tool_messages(
                        self.tokenizer, messages, self.max_context, label=f"turn {turn + 1}",
                    )
                    # Sync trajectory tool tail
                    traj = self._trajectory
                    for i in range(len(traj) - 1, -1, -1):
                        if traj[i].get("role") != "tool":
                            break
                        if i < len(messages) and messages[i].get("role") == "tool":
                            traj[i]["content"] = messages[i]["content"]

                    used_after = count_tokens_messages(self.tokenizer, messages)
                    if used_after > condense_threshold_tokens:
                        t_cond_start = time.time()
                        condensed = await self.condense(messages)
                        messages = condensed
                        self._trajectory.append({
                            "role": "user",
                            "content": condensed[1]["content"] if len(condensed) > 1 else "",
                            "_condensed": True,
                        })
                        print(f"    ↻ context condensed ({used_after} -> {count_tokens_messages(self.tokenizer, messages)} tokens, {time.time() - t_cond_start:.1f}s)", flush=True)

            print()

        # ── Force-submit if max_turns reached without end_tool ──────────────
        if finish_reason == "max_turns":
            _FORCE_MAX_RETRIES = 3
            force_nudges = [
                (
                    f"You have used all {self.max_turn} turns without submitting an answer. "
                    f"Based on everything you have found so far, call {self.end_tool} NOW "
                    f"with your best answer. Do not search anymore."
                ),
                f"You MUST call {self.end_tool} right now. Stop everything else and submit your best answer immediately.",
                f"FINAL WARNING: call {self.end_tool} with whatever answer you have. This is your last chance.",
            ]
            print(f"  ⚑ [{self.name}] max_turns reached — forcing final {self.end_tool} call", flush=True)
            force_messages = list(messages)
            for attempt in range(_FORCE_MAX_RETRIES):
                nudge = force_nudges[min(attempt, len(force_nudges) - 1)]
                force_messages.append({"role": "user", "content": nudge})
                if attempt == 0:
                    self._trajectory.append({"role": "user", "content": nudge})
                try:
                    forced = await self.chat_with_tool_retry(force_messages)
                    self._trajectory.append(forced)
                    called_end = any(
                        tc["function"]["name"] == self.end_tool
                        for tc in (forced.get("tool_calls") or [])
                    )
                    if called_end:
                        finish_reason = f"{self.end_tool}_confirmed"
                        print(f"  ✓ [{self.name}] force-submit succeeded (attempt {attempt + 1})", flush=True)
                        break
                    print(f"  ↻ [{self.name}] force-submit attempt {attempt + 1}: {self.end_tool} not called, retrying…", flush=True)
                    force_messages.append({"role": "assistant", "content": forced.get("content") or ""})
                except Exception as e:
                    print(f"  ✗ [{self.name}] force-submit attempt {attempt + 1} error: {e}", flush=True)
                    break

        total_turns = len(turn_stats)
        total_tokens = count_tokens_messages(self.tokenizer, self._trajectory)
        self._run_stats = {"total_turns": total_turns, "total_tokens": total_tokens, "finish_reason": finish_reason}
        print(f"  [{self.name}] done | turns={total_turns} tokens={total_tokens} reason={finish_reason}", flush=True)

        return list(self._trajectory)

    async def run(self, question: str) -> List[Dict[str, Any]]:
        """Public entry point — wraps _run_inner with guaranteed trajectory save."""
        try:
            return await self._run_inner(question)
        finally:
            self._save_trajectory()

    # ═══════════════════════════════════════════════════════════════
    # Trajectory helpers
    # ═══════════════════════════════════════════════════════════════

    def _save_trajectory(self) -> None:
        """Save current trajectory and condense sessions to disk."""
        if not self.trajectory_dir:
            return
        try:
            Path(self.trajectory_dir).mkdir(parents=True, exist_ok=True)

            traj_file = Path(self.trajectory_dir, f"{self.name}.json")
            with open(traj_file, "w", encoding="utf-8") as f:
                json.dump({
                    "name": self.name, "agent_type": self.agent_type,
                    **getattr(self, "_run_stats", {}),
                    "messages": self._trajectory,
                }, f, ensure_ascii=False, indent=2)
            print(f"    [{self.name}] trajectory saved -> {traj_file}", flush=True)

            if self._condense_sessions:
                cond_file = Path(self.trajectory_dir, f"{self.name}_condense.json")
                with open(cond_file, "w", encoding="utf-8") as f:
                    json.dump({"name": self.name, "sessions": self._condense_sessions},
                              f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"    [{self.name}] failed to save trajectory: {e}", flush=True)

    def get_trajectory(self) -> List[Dict[str, Any]]:
        return list(self._trajectory)

    def get_condense_sessions(self) -> List[Dict[str, Any]]:
        return list(self._condense_sessions)

    def reset_state(self) -> None:
        self._current_question = ""
        self._trajectory = []
        self._condense_sessions = []
