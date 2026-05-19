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

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Default system prompt (same semantics as agent_loop)
# ──────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """\
You are a Deep Research Agent. Your task is to find the correct answer to a complex \
question by searching a document corpus.

CRITICAL RULES — you MUST follow these:
1. ALWAYS call `search` or `get_document` on your first turn. Never output a final \
answer without first using at least one tool.
2. Search for specific entities (names, places, dates) rather than long phrases. \
Use DIFFERENT queries from DIFFERENT angles.
3. After every search, extract entities from results and use them for your next search.
4. When a snippet looks even partially relevant, call get_document to read the full text.

─── SELF-CHECK RULE ───

Before outputting your final answer, complete the Self-Check section honestly. \
If any item is NO, continue searching instead of answering.

Available tools:
- `search`: BM25 index lookup (returns docid, score, snippet).
- `get_document`: retrieve a full document by docid.

Answer format (on your FINAL turn — when you are ready to answer):
YOU MUST output exactly in this format:

Self-Check:
  [READ]      YES/NO — Did I read the full text of any document via get_document?
  [ANGLES]    YES/NO — Did I try multiple different search angles (not just rephrase)?
  [CHAIN]     YES/NO — Did I use entities found in results to guide my next searches?
  [GROUNDED]  YES/NO — Is every factual statement in my answer directly supported by \
text retrieved through tools (not inference, not prior knowledge)?
  [QUOTABLE]  YES/NO — For each claim, can I point to a docid and quote the exact \
supporting sentence?
  [EXHAUST]   YES/NO — (only if giving up) Did I search for different clues before concluding?

Evidence Mapping (list each claim and its source):
  Claim 1: <what I assert>
    → Source: docid=<X>, quote="<exact supporting text>"
  Claim 2: ...
  (add more claims as needed)

Explanation: <step-by-step reasoning citing specific docids and evidence>
Exact Answer: <your final concise answer>
Do NOT include anything after "Exact Answer:" — no extra commentary.\
"""


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
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 10,
        search_k: int = 5,
        snippet_max_chars: int = 1200,
        record_trajectory: bool = True,
        strip_thinking: bool = True,
        condense_thinking: bool = False,
        # Verify-agent dependencies (needed for submit_answer tool)
        client: Any = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        self.n_envs = n_envs
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.search_k = search_k
        self.snippet_max_chars = snippet_max_chars
        self.record_trajectory = record_trajectory
        self.strip_thinking = strip_thinking
        self.condense_thinking = condense_thinking

        # Verify-agent config
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
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

        def search(query: str) -> List[Dict[str, Any]]:
            docs = self._searcher.search(query, k=self.search_k)
            return [
                {
                    "docid": doc["docid"],
                    "score": doc["score"],
                    "snippet": snippetize(doc["text"], self.snippet_max_chars),
                    "url": doc.get("url", ""),
                }
                for doc in docs
            ]

        def get_document(docid: str) -> Dict[str, Any]:
            doc = self._searcher.get_document(docid)
            if doc is None:
                return {"docid": docid, "error": "document not found"}
            return doc

        async def submit_answer(answer: str, evidence: str) -> Dict[str, Any]:
            """Submit final answer for verification. Triggers a verify agent that checks the answer."""
            if self._client is None:
                return {"error": "No model client configured for verification"}
            return await self._run_verify_agent(answer, evidence)

        def give_feedback(is_correct: bool, reason: str, suggestions: str = "") -> Dict[str, Any]:
            """Called by the verify agent to report its verification verdict.
            Only available to the verify agent, NOT to the main agent."""
            return {
                "is_correct": is_correct,
                "reason": reason,
                "suggestions": suggestions,
            }

        # ── Main agent tools (search + get_document + submit_answer) ──
        main_tools = [
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
                    "name": "submit_answer",
                    "description": (
                        "Submit your final answer for verification. A verify agent will independently "
                        "check your answer against the document corpus and provide feedback. "
                        "If the answer is incorrect, you will receive suggestions and can continue searching."
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
                                    "Chain-of-evidence mapping each claim to its source docid and quote. "
                                    "Format: Claim 1: ... → Source: docid=X, quote=\"...\""
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
                    "description": (
                        "Report your verification verdict. Call this AFTER thoroughly checking all claims. "
                        "If the answer is wrong, provide specific, actionable suggestions for improvement."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_correct": {
                                "type": "boolean",
                                "description": "True if the answer is fully correct, False otherwise",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of your verdict",
                            },
                            "suggestions": {
                                "type": "string",
                                "description": (
                                    "If incorrect: specific suggestions for what to search next, "
                                    "which claims to re-examine, or which angle to pursue"
                                ),
                            },
                        },
                        "required": ["is_correct", "reason"],
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

    # ── Verify Agent ──────────────────────────

    _VERIFY_SYSTEM_PROMPT = """\
You are a Verification Agent. Your job is to independently verify whether a proposed answer to a question is correct, using the document corpus.

**Question:** {question}
**Proposed Answer:** {answer}
**Claimed Evidence:**
{evidence}

You have `search` and `get_document` tools. Verify each claim by searching for supporting documents.

**Rules:**
1. For each claim, search for independent evidence — do NOT just trust the claimed evidence.
2. Use get_document to read full documents; snippets alone can be misleading.
3. Check that the answer exactly matches what the question asks (not a related but different question).
4. If the evidence supports all claims and the answer is correct, call `give_feedback(is_correct=True, ...)`.
5. If any claim is unsupported or the answer is wrong, call `give_feedback(is_correct=False, ...)` with specific, actionable suggestions.
6. Do NOT call give_feedback until you have thoroughly searched and verified.
"""

    _VERIFY_MAX_TURNS = 5

    async def _run_verify_agent(self, answer: str, evidence: str) -> Dict[str, Any]:
        """Run a short verify-agent loop that checks the answer and returns feedback."""
        if self._client is None:
            return {"error": "No model client available for verification"}

        verify_prompt = self._VERIFY_SYSTEM_PROMPT.format(
            question=self._question,
            answer=answer,
            evidence=evidence,
        )
        verify_msgs: List[Dict[str, Any]] = [
            {"role": "system", "content": verify_prompt},
            {"role": "user", "content": "Please verify this answer. Search for evidence and then call give_feedback with your verdict."},
        ]
        verify_tools: List[Dict[str, Any]] = self._verify_tools

        for _ in range(self._VERIFY_MAX_TURNS):
            raw = await self._client.simple_chat(
                model=self._model,
                messages=verify_msgs,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                tools=verify_tools,
                tool_choice="auto",
            )
            resp: Dict[str, Any] = raw["choices"][0]["message"]
            verify_msgs.append(resp)

            tc = resp.get("tool_calls")
            if not tc:
                continue  # model didn't call any tool, retry

            for tc_item in tc:
                fn = tc_item["function"]
                name = fn["name"]
                call_id: str = tc_item.get("id", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    verify_msgs.append({
                        "role": "tool", "tool_call_id": call_id,
                        "content": json.dumps({"error": "Invalid JSON arguments"}),
                    })
                    continue

                if name not in self._verify_registry:
                    verify_msgs.append({
                        "role": "tool", "tool_call_id": call_id,
                        "content": json.dumps({"error": f"Unknown tool: {name}"}),
                    })
                    continue

                fn_impl = self._verify_registry[name]
                try:
                    if inspect.iscoroutinefunction(fn_impl):
                        result = await fn_impl(**args)
                    else:
                        result = fn_impl(**args)
                except Exception as exc:
                    result = {"error": str(exc)}

                tool_result = json.dumps(result, ensure_ascii=False)
                verify_msgs.append({
                    "role": "tool", "tool_call_id": call_id,
                    "content": tool_result,
                })

            # If give_feedback was called, return its result
            tc_names = [t["function"]["name"] for t in tc]
            if "give_feedback" in tc_names:
                for tc_item in tc:
                    if tc_item["function"]["name"] == "give_feedback":
                        try:
                            return json.loads(tc_item["function"]["arguments"])
                        except (json.JSONDecodeError, TypeError):
                            return {"is_correct": False, "reason": "Failed to parse feedback arguments", "suggestions": ""}

        # Max turns reached without give_feedback
        return {"is_correct": False, "reason": "Verify agent exceeded max turns without calling give_feedback", "suggestions": "Try providing more detailed evidence or breaking down claims"}

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

            # 1. Append the assistant message to conversation
            assistant_msg: Dict[str, Any] = dict(action)
            assistant_msg.setdefault("role", "assistant")
            original_content = assistant_msg.get("content", "")
            if self.condense_thinking and isinstance(original_content, str) and original_content:
                assistant_msg["content"] = self._condense_think(original_content)
                traj_msg = copy.deepcopy(assistant_msg)
                traj_msg["_original_content"] = original_content  # preserve for debugging
                inst.trajectory.append(traj_msg)
            elif self.strip_thinking and isinstance(original_content, str) and original_content:
                assistant_msg["content"] = self._strip_think(original_content)
                traj_msg = copy.deepcopy(assistant_msg)
                traj_msg["_original_content"] = original_content  # preserve for debugging
                inst.trajectory.append(traj_msg)
            else:
                inst.trajectory.append(copy.deepcopy(assistant_msg))
            inst.messages.append(assistant_msg)
            inst.turn += 1

            tool_calls = assistant_msg.get("tool_calls")
            tc_count = len(tool_calls) if tool_calls else 0
            info: Dict[str, Any] = {
                "instance_id": i,
                "turn": inst.turn,
                "tool_calls_count": tc_count,
            }

            # 2. Determine if done
            if not tool_calls:
                # Model returned final answer — episode finished
                inst.done = True
                info["done"] = True
                info["finish_reason"] = "no_tool_calls"
            elif inst.turn >= self.max_turns:
                inst.done = True
                info["done"] = True
                info["finish_reason"] = "max_turns"
            else:
                info["done"] = False

            # 3. Execute tool calls (if any)
            if tool_calls and not inst.done:
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

        # 1. Append assistant message
        msg: Dict[str, Any] = dict(assistant_msg)
        msg.setdefault("role", "assistant")
        original_content = msg.get("content", "")
        if self.condense_thinking and isinstance(original_content, str) and original_content:
            msg["content"] = self._condense_think(original_content)
            traj_msg = copy.deepcopy(msg)
            traj_msg["_original_content"] = original_content  # preserve for debugging
            inst.trajectory.append(traj_msg)
        elif self.strip_thinking and isinstance(original_content, str) and original_content:
            msg["content"] = self._strip_think(original_content)
            traj_msg = copy.deepcopy(msg)
            traj_msg["_original_content"] = original_content  # preserve for debugging
            inst.trajectory.append(traj_msg)
        else:
            inst.trajectory.append(copy.deepcopy(msg))
        inst.messages.append(msg)
        inst.turn += 1

        tool_calls = msg.get("tool_calls")

        # 2. Determine if done
        if not tool_calls:
            inst.done = True
        elif inst.turn >= self.max_turns:
            inst.done = True

        # 3. Execute tool calls (if any, and not done via max_turns)
        if tool_calls and not inst.done:
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
                        if inspect.iscoroutinefunction(fn_impl):
                            raw = await fn_impl(**args)
                        else:
                            raw = fn_impl(**args)
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

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove ``<think>...</think>`` blocks from assistant content.

        Also handles unclosed ``<think>`` tags (truncated output).
        """
        import re
        # Remove properly closed <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove unclosed <think> (truncated output) — everything after the opening tag
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _condense_think(text: str) -> str:
        """Condense ``<think>...</think>`` blocks into concise planning summaries.

        Instead of fully stripping the model's reasoning, this extracts key sentences
        related to strategy, planning, and core purpose.  The result is a compact
        ``[Plan]`` marker that preserves context for subsequent turns without wasting
        token budget on verbose chain-of-thought.

        Also handles unclosed ``<think>`` tags (truncated output).
        """
        import re

        # ── sentence-split helper ──
        def _split_sentences(t: str) -> "List[str]":
            # Split on ., !, ? followed by whitespace, or on newlines
            raw = re.split(r"(?<=[.!?])\s+|\n+", t)
            return [s.strip() for s in raw if s.strip()]

        # ── planning-relevance patterns ──
        _PLAN_PATTERNS = [
            r"\b(plan|planning|strategy|approach|goal|objective)\b",
            r"\b(need to|have to|must|should|will|going to)\b",
            r"\b(next|first|then|finally|after that)\b",
            r"\b(search|retrieve|look up|find|check|verify|confirm)\b",
            r"\b(hypothes|suspect|believe|assume|guess)\b",
            r"\b(clue|clues|evidence|lead|direction|angle)\b",
            r"\b(key|critical|important|essential|main|core)\b",
            r"\b(question asks|trying to answer|need to know|missing)\b",
            r"\b(let me|I will|I need|I should|we need)\b",
        ]

        def condense_block(think_content: str) -> str:
            content = think_content.strip()
            if not content:
                return ""
            content = " ".join(content.split())

            sentences = _split_sentences(content)
            if not sentences:
                return ""

            scored: "List[Tuple[str,int]]" = []
            for sent in sentences:
                s_lower = sent.lower()
                score = 0
                for pat in _PLAN_PATTERNS:
                    if re.search(pat, s_lower):
                        score += 1
                # Small bonus for edge sentences (often carry the gist)
                if sent is sentences[0] or sent is sentences[-1]:
                    score += 1
                if score > 0:
                    scored.append((sent, score))

            if not scored:
                # Fallback: keep first + last sentence
                result_sents: "List[str]" = []
                if sentences:
                    result_sents.append(sentences[0][:300])
                if len(sentences) > 1:
                    result_sents.append(sentences[-1][:300])
                summary = " ".join(result_sents)
            else:
                scored.sort(key=lambda x: x[1], reverse=True)
                top = scored[:3]                      # at most 3 sentences
                # Restore original order
                top.sort(key=lambda x: sentences.index(x[0]))
                summary = " ".join(s[0] for s in top)

            max_len = 500
            if len(summary) > max_len:
                summary = summary[:max_len - 3] + "..."

            return f"[Plan] {summary}" if summary else ""

        # Replace closed <think>...</think>
        text = re.sub(
            r"<think>(.*?)</think>",
            lambda m: condense_block(m.group(1)),
            text,
            flags=re.DOTALL,
        )
        # Replace unclosed <think> (truncated output)
        text = re.sub(
            r"<think>(.*)$",
            lambda m: condense_block(m.group(1)),
            text,
            flags=re.DOTALL,
        )
        return text.strip()

    # ── Inspection ─────────────────────────────
    def get_active_messages(self) -> List[List[Dict[str, Any]]]:
        """Return the current message history for each non-done instance."""
        return [inst.messages for inst in self._instances if not inst.done]

    def set_messages(self, instance_id: int, messages: List[Dict[str, Any]]) -> None:
        """Replace conversation history for an instance (used after condensation)."""
        self._instances[instance_id].messages = messages

    def append_to_trajectory(self, instance_id: int, msg: Dict[str, Any]) -> None:
        """Append a message to the instance's trajectory log.

        Used during retry loops (think truncation nudge, tool validation errors)
        to keep the trajectory aligned with what the model actually saw.
        """
        self._instances[instance_id].trajectory.append(copy.deepcopy(msg))

    def replace_trajectory(self, instance_id: int, trajectory: List[Dict[str, Any]]) -> None:
        """Replace the entire trajectory for an instance (used after condensation)."""
        self._instances[instance_id].trajectory = trajectory

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
