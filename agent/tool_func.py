"""
Tool function implementations.

ToolRegistry holds all dependencies (searcher, agent instances) and exposes
bound async methods that can be called directly as tool handlers.

Agent communication flows entirely through tools:
  - main agent calls search -> BM25 results
  - main agent calls call_subagents -> spawns search agents -> collects answers
  - main agent calls submit_answer -> triggers verify agent -> returns verdict
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, Callable, Dict, List, Optional

from .browsecomp_searcher import BrowseCompBM25Searcher, snippetize
from .utils import count_tokens_messages, truncate_utf8_prefix_to_token_budget


class ToolRegistry:
    """Holds tool implementations with bound dependencies.

    Parameters
    ----------
    searcher : BrowseCompBM25Searcher
        The BM25 searcher instance shared by all agents.
    agent_factory : callable
        ``agent_factory(config_path, client, tokenizer) -> Agent``.
        Used by ``call_subagents`` to create search agents on the fly.
    main_agent : Agent, optional
        The main research agent (needed by submit_answer for verify flow).
    """

    def __init__(
        self,
        *,
        searcher: BrowseCompBM25Searcher,
        agent_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._searcher = searcher
        self._agent_factory = agent_factory

        # These are set after construction by the orchestrator
        self._verify_agent: Optional[Any] = None
        self._sub_summary_agent: Optional[Any] = None
        self._main_agent: Optional[Any] = None
        self._surrender_check_agent: Optional[Any] = None

        # Config overrides
        self._search_k: int = 5
        self._snippet_max_chars: int = 1200
        self._use_subagent_summary: bool = False

    # ── Setters for dependencies wired after construction ──

    def set_verify_agent(self, agent: Any) -> None:
        self._verify_agent = agent

    def set_sub_summary_agent(self, agent: Any) -> None:
        self._sub_summary_agent = agent

    def set_main_agent(self, agent: Any) -> None:
        self._main_agent = agent

    def set_surrender_check_agent(self, agent: Any) -> None:
        self._surrender_check_agent = agent

    def configure_search(self, search_k: int, snippet_max_chars: int, use_subagent_summary: bool) -> None:
        self._search_k = search_k
        self._snippet_max_chars = snippet_max_chars
        self._use_subagent_summary = use_subagent_summary

    # ═══════════════════════════════════════════════════════════════
    # Main / Search Agent Tools
    # ═══════════════════════════════════════════════════════════════

    async def search(
        self,
        query: str,
        found: str = "",
        history_found: str = "",
        next_reason: str = "",
    ) -> List[Dict[str, Any]]:
        """BM25 search with optional sub-agent document summarization."""
        print(
            f"    [search] query='{query}' k={self._search_k} "
            f"sub_agent={self._use_subagent_summary} | "
            f"found={found!r} history_found={history_found!r} next_reason={next_reason!r}",
            flush=True,
        )
        docs = self._searcher.search(query, k=self._search_k)
        print(f"    [search] found {len(docs)} docs", flush=True)

        if not self._use_subagent_summary or self._sub_summary_agent is None:
            return [
                {
                    "docid": doc["docid"],
                    "score": doc["score"],
                    "snippet": snippetize(doc["text"], self._snippet_max_chars),
                    "url": doc.get("url", ""),
                }
                for doc in docs
            ]

        # ── Sub-agent processing: read docs in parallel, extract relevant info ──
        sub = self._sub_summary_agent
        tok = sub.tokenizer
        sub_tools = sub.tool_specs
        sub_prompt = sub.system_prompt

        async def _process_one(doc: Dict[str, Any]) -> Dict[str, Any]:
            import traceback as _tb
            prefix = f"Query: {query}\n\nDocument (docid={doc['docid']}):\n"
            safe_input = (
                getattr(sub, "max_context", 32768)
                - getattr(sub, "max_tokens", 4096)
                - 2000
            )
            fixed = count_tokens_messages(
                tok,
                [
                    {"role": "system", "content": sub_prompt},
                    {"role": "user", "content": prefix},
                ],
            )
            doc_budget = max(256, safe_input - fixed - 64)
            doc_body = truncate_utf8_prefix_to_token_budget(tok, doc["text"], doc_budget)
            if len(doc_body) < len(doc["text"]):
                print(f"    [sub] docid={doc['docid']}: doc truncated to {doc_budget} tokens", flush=True)

            msgs = [
                {"role": "system", "content": sub_prompt},
                {"role": "user", "content": prefix + doc_body},
            ]
            for attempt in range(2):
                try:
                    resp = await sub.chat_with_tool_retry(msgs, tools=sub_tools)
                    tc = resp.get("tool_calls")
                    if tc:
                        for t in tc:
                            if t["function"]["name"] == "submit_summary":
                                args = json.loads(t["function"].get("arguments", "{}"))
                                info = args.get("relevant_info", "").strip()
                                print(f"    [sub] docid={doc['docid']}: extracted {len(info)} chars", flush=True)
                                return {"docid": doc["docid"], "summary": info if info else "(nothing relevant found)"}
                    if attempt == 0:
                        msgs.append({"role": "user", "content": "You MUST call submit_summary to submit your findings."})
                        print(f"    [sub] docid={doc['docid']}: retry with nudge", flush=True)
                except Exception as e:
                    print(f"    [sub] docid={doc['docid']}: ERROR {e}", flush=True)
                    _tb.print_exc()
                    break
            print(f"    [sub] docid={doc['docid']}: all attempts failed, using snippet fallback", flush=True)
            return {"docid": doc["docid"], "summary": snippetize(doc["text"], 300)}

        tasks = [_process_one(d) for d in docs]
        results = await asyncio.gather(*tasks)
        for r in results:
            print(f"    [search-result] docid={r.get('docid','?')}: {r.get('summary','?')}", flush=True)
        return list(results)

    def get_document(self, docid: str) -> Dict[str, Any]:
        """Retrieve full document by docid."""
        doc = self._searcher.get_document(docid)
        if doc is None:
            return {"docid": docid, "error": "document not found"}
        print(f"    [get_document] docid={docid} title={doc.get('url','?')[:80]} text={len(doc.get('text',''))} chars", flush=True)
        return doc

    async def call_subagents(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Spawn search agents in parallel for each question. No concurrency limit.

        Each search agent independently searches the corpus and submits findings.
        """
        if self._agent_factory is None:
            return [{"error": "agent_factory not configured"}]

        print(f"    [call_subagents] spawning {len(questions)} search agents in parallel", flush=True)
        for i, q in enumerate(questions):
            print(f"    [call_subagents]   [{i}] {q[:200]}", flush=True)

        async def _run_one(question: str, idx: int) -> Dict[str, Any]:
            try:
                agent = self._agent_factory("configs/search_agent.yaml")
                agent.tool_registry = self  # share tool registry
                print(f"    [subagent-{idx}] started", flush=True)
                traj = await agent.run(question)
                # Extract answer from trajectory
                answer = ""
                for msg in reversed(traj):
                    if msg.get("role") == "assistant":
                        for tc in (msg.get("tool_calls") or []):
                            if tc.get("function", {}).get("name") == "submit_answer":
                                try:
                                    args = json.loads(tc["function"].get("arguments", "{}"))
                                    answer = args.get("answer", "")
                                except (json.JSONDecodeError, TypeError):
                                    pass
                print(f"    [subagent-{idx}] finished, answer={answer[:200]}", flush=True)
                return {"question": question, "answer": answer, "trajectory": traj}
            except Exception as e:
                print(f"    [subagent-{idx}] ERROR: {e}", flush=True)
                traceback.print_exc()
                return {"question": question, "answer": "", "error": str(e)}

        tasks = [_run_one(q, i) for i, q in enumerate(questions)]
        results = await asyncio.gather(*tasks)
        print(f"    [call_subagents] all {len(questions)} agents finished", flush=True)
        return list(results)

    async def submit_answer(self, answer: str, evidence: str) -> Dict[str, Any]:
        """Submit answer for verification. Triggers verify agent if configured.

        Stage 1: Quick surrender check (is this a real answer or giving up?)
        Stage 2: Full verify agent with search + get_document + give_feedback.
        """
        if self._verify_agent is None:
            return {"is_correct": True, "reason": "Verification disabled (no verify agent configured)"}

        question = ""
        if self._main_agent is not None:
            question = getattr(self._main_agent, "_current_question", "")

        print(f"    [submit_answer] answer={answer[:200]}", flush=True)
        print(f"    [submit_answer] evidence={evidence[:300]}", flush=True)

        # ── Stage 1: Quick surrender check via dedicated agent ──
        print(f"    [verify] verify_stage1_start", flush=True)
        if self._surrender_check_agent is not None:
            surrender_traj = await self._surrender_check_agent.run(
                f"Answer to classify:\n{answer}"
            )
            is_pass = True  # default to pass
            for msg in reversed(surrender_traj):
                if msg.get("role") == "assistant":
                    for tc in (msg.get("tool_calls") or []):
                        if tc.get("function", {}).get("name") == "report_surrender_verdict":
                            try:
                                args = json.loads(tc["function"].get("arguments", "{}"))
                                is_pass = args.get("is_pass", True)
                            except (json.JSONDecodeError, TypeError):
                                pass
            if not is_pass:
                print(f"    [verify] stage1 result: SURRENDER", flush=True)
                return {
                    "is_correct": False,
                    "reason": (
                        "Your answer is a surrender statement. The answer EXISTS in the corpus "
                        "-- do NOT give up. If you think a local information is missing, maybe the "
                        "answer you want to verify is not correct and you NEED to CHANGE your target "
                        "and restart. For example, if you think A satisfies the constraints of the "
                        "question but cannot find the target B related to A, it is very likely that "
                        "the object satisfying the constraints is not only A. Do not get stuck on it."
                    ),
                }
            print(f"    [verify] stage1 verdict: PASS", flush=True)
        else:
            print(f"    [verify] stage1 skipped (no surrender_check_agent)", flush=True)

        # ── Stage 2: Full verify agent ──
        result = await self._verify_agent.run(
            f"Question: {question}\n\nProposed Answer: {answer}\n\nClaimed Evidence:\n{evidence}\n\nPlease verify this answer following the workflow."
        )

        # Extract give_feedback result from verify agent trajectory
        verdict = {"is_correct": True, "reason": "Verification completed"}
        for msg in reversed(result):
            if msg.get("role") == "assistant":
                for tc in (msg.get("tool_calls") or []):
                    if tc.get("function", {}).get("name") == "give_feedback":
                        try:
                            verdict = json.loads(tc["function"].get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            pass

        print(f"    [submit_answer] verdict: is_correct={verdict.get('is_correct')} "
              f"error_type={verdict.get('error_type','?')}", flush=True)
        if verdict.get("reason"):
            print(f"    [submit_answer] reason: {verdict['reason'][:300]}", flush=True)

        return verdict

    # ═══════════════════════════════════════════════════════════════
    # Verify Agent Tools
    # ═══════════════════════════════════════════════════════════════

    async def verify_search(self, query: str) -> List[Dict[str, Any]]:
        """Simplified search for verify agent (no found/history_found fields)."""
        print(f"    [verify-search] query='{query}' k={self._search_k}", flush=True)
        docs = self._searcher.search(query, k=self._search_k)
        print(f"    [verify-search] found {len(docs)} docs", flush=True)
        results = [
            {
                "docid": doc["docid"],
                "score": doc["score"],
                "snippet": snippetize(doc["text"], self._snippet_max_chars),
                "url": doc.get("url", ""),
            }
            for doc in docs
        ]
        return results

    def give_feedback(self, is_correct: bool, reason: str, error_type: str = "") -> Dict[str, Any]:
        """Called by verify agent to report verification verdict."""
        rejection_nudge = ""
        if not is_correct:
            if error_type == "wrong_answer":
                rejection_nudge = (
                    "DO NOT SUBMIT THIS ANSWER AGAIN. It is clearly wrong. "
                    "Find a DIFFERENT answer."
                )
            elif error_type == "insufficient_evidence":
                rejection_nudge = (
                    "INSUFFICIENT EVIDENCE. You MUST provide richer evidence than before. "
                    "If you cannot, CHANGE your answer entirely."
                )
        return {
            "is_correct": is_correct,
            "reason": (reason + " | " + rejection_nudge).strip(" |") if rejection_nudge else reason,
        }

    # ═══════════════════════════════════════════════════════════════
    # Registry builders — return {tool_name: callable} per agent type
    # ═══════════════════════════════════════════════════════════════

    def build_registry(self, agent_type: str) -> Dict[str, Callable[..., Any]]:
        """Return a dict mapping tool names to bound callables for the given agent type."""
        if agent_type == "main":
            return {
                "search": self.search,
                "get_document": self.get_document,
                "call_subagents": self.call_subagents,
                "submit_answer": self.submit_answer,
            }
        elif agent_type == "search":
            return {
                "search": self.search,
                "get_document": self.get_document,
                "submit_answer": self.submit_answer,
            }
        elif agent_type == "verify":
            return {
                "search": self.verify_search,
                "get_document": self.get_document,
                "give_feedback": self.give_feedback,
            }
        elif agent_type == "sub_summary":
            return {
                "submit_summary": self._submit_summary_impl,
            }
        elif agent_type == "surrender_check":
            return {
                "report_surrender_verdict": self._report_surrender_verdict_impl,
            }
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

    async def _submit_summary_impl(self, relevant_info: str) -> Dict[str, str]:
        """Submit summary — pass-through; the caller extracts from trajectory."""
        return {"relevant_info": relevant_info}

    def _report_surrender_verdict_impl(self, is_pass: bool, reason: str) -> Dict[str, Any]:
        """Surrender verdict — pass-through; the caller extracts from trajectory."""
        return {"is_pass": is_pass, "reason": reason}
