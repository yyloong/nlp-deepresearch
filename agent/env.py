"""
Deep Research Environment — RL-friendly wrapper for the BrowseComp agent loop.

Supports batched parallel env instances. Each instance manages an independent
conversation with the search/index tool backend. The env does NOT call the model;
it only executes tools and maintains state. The training framework is responsible
for feeding observations to the policy and passing generated actions back to step().

Typical training loop::

    env = DeepResearchEnv(
        index_path="indexes/browsecomp_plus_bm25.sqlite",
        n_envs=4,
        max_turns=10,
    )
    obs, infos = env.reset(questions)
    for _ in range(max_turns):
        actions = policy.generate(obs)          # model produces assistant messages
        next_obs, rewards, dones, infos = env.step(actions)
        # ... accumulate trajectory data ...
        if all(dones):
            break
        obs = next_obs
    trajectories = env.get_trajectories()
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .browsecomp_searcher import BrowseCompBM25Searcher, snippetize
from .tools import build_searcher
from .utils import count_tokens_messages, truncate_utf8_prefix_to_token_budget
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Per-instance state
# ──────────────────────────────────────────────
@dataclass
class EnvInstance:
    """Internal state for a single environment instance."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    done: bool = False
    turn: int = 0
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


# ──────────────────────────────────────────────
# Main Environment
# ──────────────────────────────────────────────
class DeepResearchEnv:
    """Vectorised environment for deep-research agent training.

    Parameters
    ----------
    index_path : str
        Path to the pre-built BM25 SQLite index.
    n_envs : int
        Number of parallel env instances (batch size for step).
    system_prompt : str
        System prompt prepended to every conversation.
    max_turns : int
        Maximum tool-calling turns before forced termination.
    search_k : int
        Default number of documents returned by the ``search`` tool.
    snippet_max_chars : int
        Maximum characters for text snippets returned by ``search``.
    record_trajectory : bool
        Whether to keep full per-instance trajectory logs.
    strip_thinking : bool
        If True, strip ``<think>...</think>`` blocks from assistant messages before
        storing in the conversation history. Saves context space for subsequent turns.
    condense_thinking : bool
        If True, condense ``<think>...</think>`` blocks into concise planning summaries
        instead of fully stripping them. Preserves core purpose and planning intent.
        Takes priority over ``strip_thinking`` when both are True.
    """

    def __init__(
        self,
        index_path: str,
        n_envs: int = 1,
        system_prompt: str = "",
        max_turns: int = 10,
        search_k: int = 5,
        snippet_max_chars: int = 1200,
        record_trajectory: bool = True,
        verify_agent: Any = None,
        enable_verify: bool = True,
        sub_agent: Any = None,
    ) -> None:
        self.n_envs = n_envs
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.search_k = search_k
        self.snippet_max_chars = snippet_max_chars
        self.record_trajectory = record_trajectory

        # Verify-agent config
        self._verify_agent = verify_agent
        self._enable_verify = enable_verify
        self._sub_agent = sub_agent
        self._question: str = ""  # set by reset_slot

        # Shared searcher (thread-safe for reads; tool calls are synchronous)
        self._searcher: BrowseCompBM25Searcher = build_searcher(index_path)
        self._tools, self._registry = self._build_tool_specs()

        # Per-instance state
        self._instances: List[EnvInstance] = []

        # Accumulated trajectories, indexed by instance_id (not completion order)
        self._finished_trajectories: List[Optional[List[Dict[str, Any]]]] = []

    # ── Tool registry ──────────────────────────
    @property
    def tool_specs(self) -> List[Dict[str, Any]]:
        """Return the OpenAI-format tool specs for the model."""
        return self._tools

    @property
    def verify_tool_specs(self) -> List[Dict[str, Any]]:
        """Return tool specs for the verify agent (search + get_document + give_feedback)."""
        return self._verify_tools

    def _build_tool_specs(self) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
        """Build OpenAI-format tool specs and callable registry for main agent + verify agent."""

        async def search(
            query: str,
            found: str = "",
            history_found: str = "",
            next_reason: str = "",
        ) -> List[Dict[str, Any]]:
            print(
                f"    [search] query='{query}' k={self.search_k} sub_agent={self._sub_agent is not None}"
                f" | found={found!r} history_found={history_found!r} next_reason={next_reason!r}",
                flush=True,
            )
            docs = self._searcher.search(query, k=self.search_k)
            print(f"    [search] found {len(docs)} docs", flush=True)
            if self._sub_agent is None:
                return [
                    {
                        "docid": doc["docid"],
                        "score": doc["score"],
                        "snippet": snippetize(doc["text"], self.snippet_max_chars),
                        "url": doc.get("url", ""),
                    }
                    for doc in docs
                ]
            # ── Sub-agent processing: read docs in parallel, extract relevant info ──
            import asyncio as _asyncio
            _SUB_PROMPT = (
                "Extract facts from this document that are related to the query, even loosely. "
                "Report specific names, dates, places, and details found in the document. "
                "Do NOT just say 'nothing found' — if the document mentions any entity or fact "
                "that could be connected to the query's topic, report it. "
                "The query is a rough guide, not an exact match requirement. "
                "Call submit_information with what you found."
            )
            _sub_tools = [{
                "type": "function",
                "function": {
                    "name": "submit_information",
                    "description": "Submit relevant facts extracted from the document.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relevant_info": {"type": "string", "description": "Relevant facts found (names, dates, numbers, quotes with docid)"}
                        },
                        "required": ["relevant_info"]
                    }
                }
            }]

            async def _process_one(doc):
                import traceback as _tb
                prefix = f"Query: {query}\n\nDocument (docid={doc['docid']}):\n"
                _sub = self._sub_agent
                _tok = _sub.tokenizer
                _safe_input = (
                    getattr(_sub, "max_context", 32768)
                    - getattr(_sub, "max_tokens", 4096)
                    - 2000
                )
                _fixed = count_tokens_messages(
                    _tok,
                    [
                        {"role": "system", "content": _SUB_PROMPT},
                        {"role": "user", "content": prefix},
                    ],
                )
                _doc_budget = max(256, _safe_input - _fixed - 64)
                _doc_body = truncate_utf8_prefix_to_token_budget(
                    _tok, doc["text"], _doc_budget
                )
                if len(_doc_body) < len(doc["text"]):
                    print(
                        f"    [sub] docid={doc['docid']}: doc truncated to {_doc_budget} tokens",
                        flush=True,
                    )
                msgs = [
                    {"role": "system", "content": _SUB_PROMPT},
                    {"role": "user", "content": prefix + _doc_body},
                ]
                for attempt in range(2):
                    try:
                        resp = await self._sub_agent.call_model(msgs, _sub_tools)
                        tc = resp.get("tool_calls")
                        if tc:
                            for t in tc:
                                if t['function']['name'] == 'submit_information':
                                    import json as _json
                                    args = _json.loads(t['function'].get('arguments','{}'))
                                    info = args.get('relevant_info','').strip()
                                    print(f"    [sub] docid={doc['docid']}: extracted {len(info)} chars", flush=True)
                                    return {"docid": doc['docid'], "summary": info if info else "(nothing relevant found)"}
                        if attempt == 0:
                            msgs.append({"role": "user", "content": "You MUST call submit_information to submit your findings."})
                            print(f"    [sub] docid={doc['docid']}: retry with nudge", flush=True)
                    except Exception as e:
                        print(f"    [sub] docid={doc['docid']}: ERROR {e}", flush=True)
                        _tb.print_exc()
                        break
                print(f"    [sub] docid={doc['docid']}: all attempts failed, using snippet fallback", flush=True)
                return {"docid": doc['docid'], "summary": snippetize(doc['text'], 300)}

            tasks = [_process_one(d) for d in docs]
            results = await _asyncio.gather(*tasks)
            for r in results:
                print(f"    [search-result] docid={r.get('docid','?')}: {r.get('summary','?')}", flush=True)
            return list(results)

        def get_document(docid: str) -> Dict[str, Any]:
            doc = self._searcher.get_document(docid)
            if doc is None:
                return {"docid": docid, "error": "document not found"}
            return doc

        async def submit_answer(answer: str, evidence: str) -> Dict[str, Any]:
            if not self._enable_verify:
                return {"is_correct": True, "reason": "Verification disabled"}
            if self._verify_agent is None:
                return {"error": "No verify agent configured"}
            return await self._verify_agent.run_verify(
                question=self._question, answer=answer, evidence=evidence,
                tools=self._verify_tools, registry=self._verify_registry,
                max_turns_for_verify=self.max_turns,
            )

        def give_feedback(is_correct: bool, reason: str, error_type: str = "") -> Dict[str, Any]:
            """Called by the verify agent to report its verification verdict."""
            rejection_nudge = ""
            if not is_correct:
                if error_type == "wrong_answer":
                    rejection_nudge = "**DO NOT SUBMIT THIS ANSWER AGAIN. It is clearly wrong. Find a DIFFERENT answer.**"
                elif error_type == "insufficient_evidence":
                    rejection_nudge = "**INSUFFICIENT EVIDENCE. You MUST provide richer evidence than before. If you cannot, CHANGE your answer entirely.**"
            return {
                "is_correct": is_correct,
                "reason": (reason + " | " + rejection_nudge).strip(" |") if rejection_nudge else reason,
            }

        # ── Main agent tools ──
        main_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": (
                        f"Search the BM25 index (top-{self.search_k} results). "
                        "Query MUST be 2-3 specific words (names/dates/titles), NOT generic descriptions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "2-3 specific entity names, never generic words"},
                            "found": {"type": "string", "description": "BEFORE this search: what specific names/dates/titles did you find in the LAST search results? List them.NEVER USE a general description."},
                            "history_found": {
                                "type": "string",
                                "description": (
                                    "Cumulative record of ALL specific names/dates/titles found across "
                                    "EVERY previous search so far (not just the last one). "
                                    "Append each turn's new findings; keep prior entries."
                                    "YOU SHOULD PROVIDE SPECIFIC FINDINGS RATHER THAN A GENERAL DESCRIPTION."
                                    "For example,GOOD:B is a monkey king with constrain C,and you find it.BAD:monkey king or monkey king with constrain C "
                                ),
                            },
                            "next_reason": {"type": "string", "description": "Why this search? How does it use what you found to move toward the answer?"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document",
                    "description": "Retrieve a full document by its docid.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {"type": "string", "description": "Document id"},
                        },
                        "required": ["docid"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_answer",
                    "description": (
                        "Submit your final answer for verification. A verify agent will independently "
                        "check your answer against the document corpus and provide feedback. "
                        "If the answer is incorrect, you will receive suggestions and can continue searching."
                        "[IMPORTANT]The verify agent has no memory or knowledge of your context,so you MUST provide whole evidence every time you submit your answer RATHER than the additional evidence from your last submit."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "Your final concise answer",
                            },
                            "evidence": {
                                "type": "string",
                                "description": (
                                    "Chain-of-evidence for each claim in your answer. "
                                    "Format each claim as: Claim N: <what you assert> → Source: docid=X, quote=\"<exact supporting text>\". "
                                    "Example: Claim 1: The author was born in 1864 → Source: docid=19351, quote=\"Mary H. Debenham (1864-1947)\""
                                ),
                            },
                        },
                        "required": ["answer", "evidence"],
                    },
                },
            },
        ]

        # ── Verify agent tools (search + get_document + give_feedback) ──
        verify_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": (
                        f"Search the BrowseComp-Plus BM25 index and return top-{self.search_k} results "
                        "with docid, score, and snippet."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document",
                    "description": "Retrieve a full document by its docid.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {"type": "string", "description": "Document id"},
                        },
                        "required": ["docid"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "give_feedback",
                    "description": "Report your verification verdict.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_correct": {
                                "type": "boolean",
                                "description": "True if fully correct, False otherwise",
                            },
                            "reason": {
                                "type": "string",
                                "description": "What claims are wrong or unsupported",
                            },
                            "error_type": {
                                "type": "string",
                                "enum": ["wrong_answer", "insufficient_evidence"],
                                "description": "If incorrect: 'wrong_answer' means the answer is clearly wrong (change answer). 'insufficient_evidence' means not enough proof (provide richer evidence or change answer).",
                            },
                        },
                        "required": ["is_correct", "reason", "error_type"],
                    },
                },
            },
        ]

        self._verify_tools = verify_tools
        self._verify_registry = {
            "search": search,
            "get_document": get_document,
            "give_feedback": give_feedback,
        }

        return main_tools, {
            "search": search,
            "get_document": get_document,
            "submit_answer": submit_answer,
        }

    # ── Core API ───────────────────────────────
    def reset(self, questions: List[str]) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Reset all env instances with new questions.

        Parameters
        ----------
        questions : list of str
            One question per env instance. Length must be <= ``n_envs``.
            If shorter, the effective batch size for this episode is reduced.

        Returns
        -------
        observations : list of list of dict
            ``observations[i]`` is the initial message list for instance *i*.
        infos : list of dict
            Per-instance metadata (question, data_source, instance_id).
        """
        n = len(questions)
        if n == 0 or n > self.n_envs:
            raise ValueError(
                f"Expected 1..{self.n_envs} questions, got {n}"
            )

        self._active_n = n  # effective batch size for this episode
        self._instances = []
        self._finished_trajectories = [None] * n  # per-instance slots
        observations: List[List[Dict[str, Any]]] = []
        infos: List[Dict[str, Any]] = []

        for i, q in enumerate(questions):
            msgs: List[Dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": q},
            ]
            inst = EnvInstance(messages=msgs, trajectory=list(msgs))
            self._instances.append(inst)
            observations.append(list(msgs))  # shallow copy
            infos.append({
                "instance_id": i,
                "question": q,
            })

        return observations, infos

    async def step(
        self, actions: List[Dict[str, Any]]
    ) -> Tuple[
        List[Optional[List[Dict[str, Any]]]],
        List[float],
        List[bool],
        List[Dict[str, Any]],
    ]:
        """Execute one step across all env instances.

        Parameters
        ----------
        actions : list of dict
            ``actions[i]`` is the assistant message (OpenAI format) for instance *i*.
            Must contain ``"role": "assistant"``.  May contain ``tool_calls``.

        Returns
        -------
        next_observations : list of list of dict or None
            ``next_observations[i]`` is the full message history after tool execution,
            or ``None`` if the instance is already done.
        rewards : list of float
            Placeholder rewards (all 0.0 for now).
        dones : list of bool
            ``True`` when the assistant message has no ``tool_calls`` or
            ``max_turns`` is reached.
        infos : list of dict
            Per-instance metadata including ``tool_calls_count``, ``turn``, etc.
        """
        n = getattr(self, '_active_n', self.n_envs)
        if len(actions) != n:
            raise ValueError(
                f"Expected {n} actions, got {len(actions)}"
            )

        next_obs: List[Optional[List[Dict[str, Any]]]] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []

        for i, (inst, action) in enumerate(zip(self._instances, actions)):
            if inst.done:
                # Already done — return sentinel values
                next_obs.append(None)
                rewards.append(0.0)
                dones.append(True)
                infos.append({"instance_id": i, "done": True, "step_skipped": True})
                continue

            # 1. Process assistant message — always record in trajectory
            assistant_msg: Dict[str, Any] = dict(action)
            assistant_msg.setdefault("role", "assistant")
            inst.trajectory.append(copy.deepcopy(assistant_msg))

            tool_calls = assistant_msg.get("tool_calls")
            tc_count = len(tool_calls) if tool_calls else 0
            info: Dict[str, Any] = {
                "instance_id": i,
                "turn": inst.turn,
                "tool_calls_count": tc_count,
            }

            # 2. Determine if done — only max_turns or submit_answer can end the episode
            #    If no tool calls: inject nudge, keep bad response ONLY in trajectory (not context).
            if not tool_calls:
                inst.trajectory.append({
                    "role": "user",
                    "content": "[NUDGE] Model returned no tool call. Reminding to use a tool.",
                })
                inst.messages.append({
                    "role": "user",
                    "content": (
                        "You did not call any tool. You MUST call a tool to make progress: "
                        "use `search` to find documents, `get_document` to read them, or "
                        "`submit_answer` when you have a final answer with evidence. "
                        "Do NOT output plain text — always call a tool."
                    ),
                })
            else:
                inst.messages.append(assistant_msg)
            inst.turn += 1
            if inst.turn >= self.max_turns:
                inst.done = True
                info["done"] = True
                info["finish_reason"] = "max_turns"
            else:
                info["done"] = False

            # 3. Execute tool calls (if any, and not done)
            if tool_calls:
                for tc in tool_calls:
                    fn = tc["function"]
                    name = fn["name"]
                    call_id: str = tc.get("id", "")
                    args_str: str = fn.get("arguments", "")

                    # Parse arguments (model may produce malformed JSON)
                    try:
                        args = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError) as exc:
                        tool_result = json.dumps({
                            "error": f"Failed to parse arguments as JSON: {exc}. "
                                     f"Arguments must be valid JSON. "
                                     f"Received (truncated): {args_str[:300]}",
                        })
                        inst.messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_result,
                        })
                        inst.trajectory.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_result,
                        })
                        continue

                    if name not in self._registry:
                        available = list(self._registry.keys())
                        tool_result = json.dumps({
                            "error": f"Unknown tool '{name}'. "
                                     f"Available tools: {', '.join(available)}. "
                                     f"Please use one of the available tools.",
                        })
                    else:
                        try:
                            fn_impl = self._registry[name]
                            if inspect.iscoroutinefunction(fn_impl):
                                raw = await fn_impl(**args)
                            else:
                                raw = fn_impl(**args)
                            tool_result = json.dumps(raw, ensure_ascii=False)
                        except TypeError as exc:
                            sig = inspect.signature(self._registry[name])
                            tool_result = json.dumps({
                                "error": f"Invalid arguments for '{name}': {exc}. "
                                         f"Expected signature: {name}{sig}. "
                                         f"Please check the parameter names and types.",
                            })
                        except Exception as exc:
                            logger.warning(
                                "Instance %d turn %d: tool %r failed: %s",
                                i, inst.turn, name, exc,
                            )
                            tool_result = json.dumps({
                                "error": f"Tool '{name}' execution failed: {exc}. "
                                         f"Please try with different arguments or a different approach.",
                            })

                    inst.messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": tool_result,
                    })
                    inst.trajectory.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": tool_result,
                    })

            # 4. Record trajectory if episode ended
            if inst.done and self.record_trajectory:
                self._finished_trajectories[i] = inst.trajectory

            # 5. Build return values
            next_obs.append(list(inst.messages) if not inst.done else None)
            rewards.append(0.0)  # placeholder
            dones.append(inst.done)
            infos.append(info)

        return next_obs, rewards, dones, infos

    async def step_single(
        self, slot_id: int, assistant_msg: Dict[str, Any]
    ) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
        """Process one step for a single instance.

        Returns ``(observation, done)`` where *observation* is the updated
        message list (or ``None`` if the instance is done).
        """
        inst = self._instances[slot_id]
        if inst.done:
            return (None, True)

        # 1. Process assistant message — always record in trajectory
        msg: Dict[str, Any] = dict(assistant_msg)
        msg.setdefault("role", "assistant")
        inst.trajectory.append(copy.deepcopy(msg))

        tool_calls = msg.get("tool_calls")

        # 2. Determine if done — only max_turns or submit_answer can end the episode.
        #    If no tool calls: inject nudge, keep bad response ONLY in trajectory (not context).
        if not tool_calls:
            inst.trajectory.append({
                "role": "user",
                "content": "[NUDGE] Model returned no tool call. Reminding to use a tool.",
            })
            inst.messages.append({
                "role": "user",
                "content": (
                    "You did not call any tool. You MUST call a tool to make progress: "
                    "use `search` to find documents, `get_document` to read them, or "
                    "`submit_answer` when you have a final answer with evidence. "
                    "Do NOT output plain text — always call a tool."
                ),
            })
        else:
            inst.messages.append(msg)

        inst.turn += 1
        if inst.turn >= self.max_turns:
            inst.done = True

        # 3. Execute tool calls (if any)
        if tool_calls:
            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                call_id: str = tc.get("id", "")
                args_str: str = fn.get("arguments", "")

                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError) as exc:
                    tool_result = json.dumps({
                        "error": f"Failed to parse arguments as JSON: {exc}. "
                                 f"Received (truncated): {args_str[:300]}",
                    })
                    inst.messages.append({
                        "role": "tool", "tool_call_id": call_id,
                        "content": tool_result,
                    })
                    inst.trajectory.append({
                        "role": "tool", "tool_call_id": call_id,
                        "content": tool_result,
                    })
                    continue

                if name not in self._registry:
                    available = list(self._registry.keys())
                    tool_result = json.dumps({
                        "error": f"Unknown tool '{name}'. "
                                 f"Available tools: {', '.join(available)}.",
                    })
                else:
                    try:
                        fn_impl = self._registry[name]
                        print(f"    [tool-exec] calling {name}({list(args.keys())}) async={inspect.iscoroutinefunction(fn_impl)}", flush=True)
                        if inspect.iscoroutinefunction(fn_impl):
                            raw = await fn_impl(**args)
                        else:
                            raw = fn_impl(**args)
                        print(f"    [tool-exec] {name} returned {type(raw).__name__} len={len(raw) if hasattr(raw,'__len__') else '?'}", flush=True)
                        tool_result = json.dumps(raw, ensure_ascii=False)
                    except TypeError as exc:
                        sig = inspect.signature(self._registry[name])
                        tool_result = json.dumps({
                            "error": f"Invalid arguments for '{name}': {exc}. "
                                     f"Expected signature: {name}{sig}.",
                        })
                    except Exception as exc:
                        logger.warning(
                            "Slot %d turn %d: tool %r failed: %s",
                            slot_id, inst.turn, name, exc,
                        )
                        tool_result = json.dumps({
                            "error": f"Tool '{name}' execution failed: {exc}.",
                        })

                inst.messages.append({
                    "role": "tool", "tool_call_id": call_id,
                    "content": tool_result,
                })
                inst.trajectory.append({
                    "role": "tool", "tool_call_id": call_id,
                    "content": tool_result,
                })

        # 4. Post-tool-execution: if submit_answer confirmed correctness, episode is done
        if tool_calls and not inst.done:
            for tc in tool_calls:
                if tc["function"]["name"] == "submit_answer":
                    # Check the tool result for is_correct flag
                    for msg in inst.messages:
                        if msg.get("role") == "tool" and msg.get("tool_call_id") == tc.get("id", ""):
                            try:
                                fb = json.loads(msg["content"])
                                if fb.get("is_correct") is True:
                                    inst.done = True
                            except (json.JSONDecodeError, TypeError):
                                pass
                            break

        # 5. Record trajectory if episode ended
        if inst.done and self.record_trajectory:
            self._finished_trajectories[slot_id] = inst.trajectory

        return (list(inst.messages) if not inst.done else None, inst.done)

    # ── Trajectory & reward helpers ────────────
    def get_trajectories(self) -> List[List[Dict[str, Any]]]:
        """Return completed trajectories in instance order and clear the buffer."""
        trajs = [t for t in self._finished_trajectories if t is not None]
        self._finished_trajectories = []
        return trajs

    def compute_reward(
        self,
        instance_id: int,
        predicted_answer: str,
        ground_truth: str,
    ) -> float:
        """Reward interface — placeholder (always returns 0.0).

        Override or replace this method for actual reward computation.
        """
        return 0.0

    def success_evaluator(self, trajectories: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Evaluate success rate from completed trajectories.

        Default implementation extracts the final assistant answer from each
        trajectory. Override for custom evaluation logic.

        Returns a dict with at least ``success_rate`` (np.ndarray).
        """
        import numpy as np

        answers: List[str] = []
        for traj in trajectories:
            for msg in reversed(traj):
                if msg.get("role") == "assistant" and msg.get("content"):
                    answers.append(msg["content"])
                    break
            else:
                answers.append("")

        return {
            "answers": answers,
            "success_rate": np.zeros(len(trajectories), dtype=np.float32),
        }

    # ── Inspection ─────────────────────────────
    def get_active_messages(self) -> List[List[Dict[str, Any]]]:
        """Return the current message history for each non-done instance."""
        return [inst.messages for inst in self._instances if not inst.done]

    def set_messages(self, instance_id: int, messages: List[Dict[str, Any]]) -> None:
        """Replace conversation history for an instance (used after condensation)."""
        self._instances[instance_id].messages = list(messages)

    def append_to_trajectory(self, instance_id: int, msg: Dict[str, Any]) -> None:
        """Append a message to the instance's trajectory log.

        Used during retry loops (think truncation nudge, tool validation errors)
        to keep the trajectory aligned with what the model actually saw.
        """
        self._instances[instance_id].trajectory.append(copy.deepcopy(msg))

    def replace_trajectory(self, instance_id: int, trajectory: List[Dict[str, Any]]) -> None:
        """Replace the entire trajectory for an instance (used after condensation)."""
        self._instances[instance_id].trajectory = list(trajectory)

    def sync_trajectory_tool_tail(self, instance_id: int) -> None:
        """Copy truncated tool message contents from messages to trajectory.

        Called after ``hard_truncate_tail_tool_messages`` mutates
        ``inst.messages`` in place, to keep the trajectory aligned with
        what the model actually sees.
        """
        inst = self._instances[instance_id]
        msgs = inst.messages
        traj = inst.trajectory
        # Walk from the end, syncing consecutive tool messages
        for i in range(len(traj) - 1, -1, -1):
            if traj[i].get("role") != "tool":
                break
            if i < len(msgs) and msgs[i].get("role") == "tool":
                traj[i]["content"] = msgs[i]["content"]

    def reset_slot(self, slot_id: int, question: str) -> List[Dict[str, Any]]:
        """Reset a single slot with a new question (for router-based scheduling).

        Returns the initial observation (messages) for the slot.
        """
        self._question = question  # store for verify agent
        msgs: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        # Ensure lists are large enough (router may skip env.reset())
        needed = slot_id + 1
        if len(self._instances) < needed:
            self._instances.extend([EnvInstance() for _ in range(needed - len(self._instances))])
        if len(self._finished_trajectories) < needed:
            self._finished_trajectories.extend([None] * (needed - len(self._finished_trajectories)))
        self._instances[slot_id] = EnvInstance(messages=msgs, trajectory=list(msgs))
        self._finished_trajectories[slot_id] = None
        return list(msgs)

    def get_slot_messages(self, slot_id: int) -> List[Dict[str, Any]]:
        """Return a shallow copy of the current message list for a slot."""
        return list(self._instances[slot_id].messages)

    def extract_slot_trajectory(self, slot_id: int) -> List[Dict[str, Any]]:
        """Extract and return the full trajectory for a slot."""
        return list(self._instances[slot_id].trajectory)

    def all_done(self) -> bool:
        return all(inst.done for inst in self._instances)

    def close(self) -> None:
        """Release resources (close searcher connection)."""
        if self._searcher is not None:
            self._searcher.connection.close()
            self._searcher = None  # type: ignore[assignment]
        self._instances = []
        self._finished_trajectories = []
