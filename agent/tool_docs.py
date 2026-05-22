"""
Tool documentation / OpenAI-format tool specifications.

Each function returns an OpenAI-format tool spec dict.
Tool specs are pure data — no implementation logic.
"""

from __future__ import annotations

from typing import Any, Dict, List


def search_tool_spec(search_k: int = 5) -> Dict[str, Any]:
    """Search tool spec — query only, used by all agents."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                f"Search the BM25 index and return top-{search_k} results "
                "with docid, score, and snippet. "
                "Query MUST be 2-3 specific words (names/dates/titles), NOT generic descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "2-3 specific entity names, never generic words"},
                },
                "required": ["query"],
            },
        },
    }


def get_document_tool_spec() -> Dict[str, Any]:
    """Get full document by docid."""
    return {
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
    }


def submit_answer_tool_spec() -> Dict[str, Any]:
    """Submit final answer for verification (main agent and search agent end_tool)."""
    return {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Submit your final answer for verification. A verify agent will independently "
                "check your answer against the document corpus and provide feedback. "
                "If the answer is incorrect, you will receive suggestions and can continue searching. "
                "IMPORTANT: The verify agent has no memory or knowledge of your context, "
                "so you MUST provide whole evidence every time you submit your answer "
                "RATHER than only additional evidence from your last submit."
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
                            "Format each claim as: Claim N: <what you assert> -> Source: docid=X, "
                            "quote='<exact supporting text>'. "
                            "Example: Claim 1: The author was born in 1864 -> "
                            "Source: docid=19351, quote='Mary H. Debenham (1864-1947)'"
                        ),
                    },
                },
                "required": ["answer", "evidence"],
            },
        },
    }


def call_subagents_tool_spec() -> Dict[str, Any]:
    """Spawn parallel search agents to verify candidates."""
    return {
        "type": "function",
        "function": {
            "name": "call_subagents",
            "description": (
                "Spawn multiple search agents in parallel to verify specific questions or "
                "candidates. Each sub-agent independently searches the corpus and returns "
                "findings. Pass a list of specific verification questions — one per candidate "
                "or angle you want to check. Results are returned as a list of answers."
                "Each agent will only receive one of the questions.Please make sure the questions are independent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of specific verification questions. Each question should be "
                            "self-contained with all necessary context for the sub-agent to "
                            "search **independently**. "
                        ),
                    },
                },
                "required": ["questions"],
            },
        },
    }


def give_feedback_tool_spec() -> Dict[str, Any]:
    """Verify agent end_tool — report verification verdict."""
    return {
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
                        "description": (
                            "If incorrect: 'wrong_answer' means the answer is clearly wrong "
                            "(change answer). 'insufficient_evidence' means not enough proof "
                            "(provide richer evidence or change answer)."
                        ),
                    },
                },
                "required": ["is_correct", "reason", "error_type"],
            },
        },
    }


def judge_relevance_tool_spec() -> Dict[str, Any]:
    """Relevance judge agent end_tool — report HELPFUL / IRRELEVANT / CONFUSING."""
    return {
        "type": "function",
        "function": {
            "name": "judge_relevance",
            "description": "Judge whether the document is relevant to the question. You MUST use this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relevance": {
                        "type": "string",
                        "enum": ["HELPFUL", "IRRELEVANT", "CONFUSING"],
                        "description": "HELPFUL: has facts that help answer. IRRELEVANT: unrelated. CONFUSING: similar but different entity — could mislead.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "If HELPFUL: concise summary of relevant facts with specific names/dates/quotes. Otherwise leave empty.",
                    },
                },
                "required": ["relevance"],
            },
        },
    }


def submit_summary_tool_spec() -> Dict[str, Any]:
    """Sub-summary agent end_tool — submit extracted facts."""
    return {
        "type": "function",
        "function": {
            "name": "submit_summary",
            "description": "Submit relevant facts extracted from the document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relevant_info": {
                        "type": "string",
                        "description": "Relevant facts found (names, dates, numbers, quotes with docid)",
                    },
                },
                "required": ["relevant_info"],
            },
        },
    }


def submit_condensed_summary_tool_spec() -> Dict[str, Any]:
    """Condense tool — submit structured progress summary."""
    return {
        "type": "function",
        "function": {
            "name": "submit_condensed_summary",
            "description": "Submit your condensed progress summary. You MUST use this tool — plain text is ignored.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_summary": {
                        "type": "string",
                        "description": (
                            "Summary of tools used: searches (deduplicated), documents read "
                            "with key content, submissions and their feedback"
                        ),
                    },
                    "key_thoughts": {
                        "type": "string",
                        "description": "Core reasoning and strategy in 2-3 sentences",
                    },
                    "key_findings": {
                        "type": "string",
                        "description": "Verified facts with docid references",
                    },
                    "remaining_to_find": {
                        "type": "string",
                        "description": "Missing clues, what to search next",
                    },
                },
                "required": ["tool_summary", "key_thoughts", "key_findings", "remaining_to_find"],
            },
        },
    }


# ── Convenience: build tool spec lists for each agent type ──

def smart_search_tool_spec(search_k: int = 5) -> Dict[str, Any]:
    """Enhanced search tool spec — auto relevance-filtered results."""
    return {
        "type": "function",
        "function": {
            "name": "smart_search",
            "description": (
                f"Search the BM25 index (top-{search_k} results) with automatic relevance filtering. "
                "Each result is evaluated by a judge agent — only genuinely helpful documents are returned. "
                "Irrelevant and confusing documents are automatically dropped. "
                "Returns a dict with 'results' (list of helpful docs, may be empty) and 'hint' "
                "(only set when ALL results were filtered out — use it to guide your next query). "
                "Query MUST be 2-3 specific words."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "2-3 specific entity names, never generic words"},
                },
                "required": ["query"],
            },
        },
    }


def build_main_agent_smart_tool_specs(search_k: int = 5) -> List[Dict[str, Any]]:
    """Tool specs for main agent (smart): smart_search, get_document, submit_answer."""
    return [
        smart_search_tool_spec(search_k),
        get_document_tool_spec(),
        submit_answer_tool_spec(),
    ]


def build_main_agent_tool_specs(search_k: int = 5) -> List[Dict[str, Any]]:
    """Tool specs for main agent: search (query only), call_subagents, submit_answer."""
    return [
        search_tool_spec(search_k),
        call_subagents_tool_spec(),
        submit_answer_tool_spec(),
    ]


def build_search_agent_tool_specs(search_k: int = 5) -> List[Dict[str, Any]]:
    """Tool specs for search agent: search (query only), get_document, submit_answer."""
    return [
        search_tool_spec(search_k),
        get_document_tool_spec(),
        submit_answer_tool_spec(),
    ]


def build_verify_agent_tool_specs(search_k: int = 5) -> List[Dict[str, Any]]:
    """Tool specs for verify agent: search (simple), get_document, give_feedback."""
    return [
        search_tool_spec(search_k),
        get_document_tool_spec(),
        give_feedback_tool_spec(),
    ]


def report_surrender_verdict_tool_spec() -> Dict[str, Any]:
    """Surrender check agent end_tool — report PASS or SURRENDER verdict."""
    return {
        "type": "function",
        "function": {
            "name": "report_surrender_verdict",
            "description": "Report your classification verdict: PASS (factual answer) or SURRENDER (giving up).",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_pass": {
                        "type": "boolean",
                        "description": "True if PASS (factual assertion), False if SURRENDER (giving up / not found)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of the classification",
                    },
                },
                "required": ["is_pass", "reason"],
            },
        },
    }


def build_sub_summary_tool_specs() -> List[Dict[str, Any]]:
    """Tool specs for sub-summary agent: submit_summary only."""
    return [submit_summary_tool_spec()]


def build_relevance_judge_tool_specs(search_k: int = 5) -> List[Dict[str, Any]]:
    """Tool specs for relevance judge agent: search, get_document, judge_relevance."""
    return [
        search_tool_spec(search_k),
        get_document_tool_spec(),
        judge_relevance_tool_spec(),
    ]


def build_surrender_check_tool_specs() -> List[Dict[str, Any]]:
    """Tool specs for surrender check agent: report_surrender_verdict only."""
    return [report_surrender_verdict_tool_spec()]


def build_condense_tool_specs() -> List[Dict[str, Any]]:
    """Tool specs for condense: submit_condensed_summary only."""
    return [submit_condensed_summary_tool_spec()]
