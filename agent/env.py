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
answer without first using at least one tool. You do NOT know the answer in advance.
2. Keep your thinking concise — plan your next tool call in 1-2 sentences max. \
Long analysis without acting is forbidden. Act first, then think about results.
3. Conduct multi-round investigation: use DIFFERENT search queries with DIFFERENT \
phrasings. A single search is never enough. Aim for 3+ distinct searches.
4. When snippets look relevant, call `get_document` to read the full document.
5. Cross-check every finding against at least one other independent source.

Available tools:
- `search`: BM25 index lookup (returns docid, score, snippet).
- `get_document`: retrieve a full document by docid.

Answer format (on your FINAL turn — when you are ready to answer):
YOU MUST output exactly in this format, with both sections present:
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
    ) -> None:
        self.n_envs = n_envs
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.search_k = search_k
        self.snippet_max_chars = snippet_max_chars
        self.record_trajectory = record_trajectory
        self.strip_thinking = strip_thinking

        # Shared searcher (thread-safe for reads; tool calls are synchronous)
        self._searcher: BrowseCompBM25Searcher = build_searcher(index_path)
        self._tools, self._registry = self._build_tool_specs()

        # Per-instance state
        self._instances: List[EnvInstance] = []

        # Accumulated trajectories, indexed by instance_id (not completion order)
        self._finished_trajectories: List[Optional[List[Dict[str, Any]]]] = []

    # ── Tool registry ──────────────────────────
    def _build_tool_specs(self) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
        """Build OpenAI-format tool specs and callable registry."""

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

        tools = [
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
        ]
        return tools, {"search": search, "get_document": get_document}

    @property
    def tool_specs(self) -> List[Dict[str, Any]]:
        """OpenAI-format tool definitions (for use by the policy model)."""
        return self._tools

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
            inst = EnvInstance(messages=msgs)
            self._instances.append(inst)
            observations.append(list(msgs))  # shallow copy
            infos.append({
                "instance_id": i,
                "question": q,
            })

        return observations, infos

    def step(
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
            if self.strip_thinking and "content" in assistant_msg:
                assistant_msg["content"] = self._strip_think(assistant_msg["content"])
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
                            raw = self._registry[name](**args)
                            tool_result = json.dumps(raw, ensure_ascii=False)
                        except TypeError as exc:
                            # Wrong argument names or counts
                            import inspect
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

            # 4. Record trajectory if episode ended
            if inst.done and self.record_trajectory:
                self._finished_trajectories[i] = list(inst.messages)

            # 5. Build return values
            next_obs.append(list(inst.messages) if not inst.done else None)
            rewards.append(0.0)  # placeholder
            dones.append(inst.done)
            infos.append(info)

        return next_obs, rewards, dones, infos

    def step_single(
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
        if self.strip_thinking and "content" in msg:
            msg["content"] = self._strip_think(msg["content"])
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
                    continue

                if name not in self._registry:
                    available = list(self._registry.keys())
                    tool_result = json.dumps({
                        "error": f"Unknown tool '{name}'. "
                                 f"Available tools: {', '.join(available)}.",
                    })
                else:
                    try:
                        raw = self._registry[name](**args)
                        tool_result = json.dumps(raw, ensure_ascii=False)
                    except TypeError as exc:
                        import inspect
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

        # 4. Record trajectory if episode ended
        if inst.done and self.record_trajectory:
            self._finished_trajectories[slot_id] = list(inst.messages)

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

    # ── Inspection ─────────────────────────────
    def get_active_messages(self) -> List[List[Dict[str, Any]]]:
        """Return the current message history for each non-done instance."""
        return [inst.messages for inst in self._instances if not inst.done]

    def set_messages(self, instance_id: int, messages: List[Dict[str, Any]]) -> None:
        """Replace conversation history for an instance (used after condensation)."""
        self._instances[instance_id].messages = messages

    def reset_slot(self, slot_id: int, question: str) -> List[Dict[str, Any]]:
        """Reset a single slot with a new question (for router-based scheduling).

        Returns the initial observation (messages) for the slot.
        """
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
        self._instances[slot_id] = EnvInstance(messages=msgs)
        self._finished_trajectories[slot_id] = None
        return list(msgs)

    def get_slot_messages(self, slot_id: int) -> List[Dict[str, Any]]:
        """Return a shallow copy of the current message list for a slot."""
        return list(self._instances[slot_id].messages)

    def extract_slot_trajectory(self, slot_id: int) -> List[Dict[str, Any]]:
        """Extract and return the full trajectory for a slot."""
        return list(self._instances[slot_id].messages)

    def all_done(self) -> bool:
        return all(inst.done for inst in self._instances)

    def close(self) -> None:
        """Release resources (close searcher connection)."""
        if self._searcher is not None:
            self._searcher.connection.close()
            self._searcher = None  # type: ignore[assignment]
        self._instances = []
        self._finished_trajectories = []
