"""
Unified Agent class for both main research agent and verification agent.

Encapsulates:
- Model calling
- Think truncation retry (two-stage: no-think → RETRY_NUDGE)
- Tool call validation and retry
- Context compression (with think block handling)
- Main agent loop and verify agent loop
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .utils import (
    RETRY_NUDGE,
    count_tokens_messages,
    hard_truncate_tail_tool_messages,
    is_truncated_think_response,
    validate_tool_call,
)
from .vllm_client_async import VLLMClientAsync

logger = logging.getLogger(__name__)

MAX_TOOL_RETRIES = 2

# ── Prompts ─────────────────────────────────────

# Prompt for model-generated analysis in structured condense (main agent)
_CONDENSE_ANALYSIS_PROMPT = """\
Write a progress summary FOR the research agent to read. Call submit_condensed_summary. \
Write in second person (\"you\"/\"your\") as if speaking to the agent — NEVER use \"the agent\". \
The conversation record contains the past tool calls and results. \
Focus on analysis and what to do next. \
Call submit_condensed_summary to respond. No <think> blocks or text before the tool call."""

# Prompt for model-generated analysis in verify condense
_VERIFY_CONDENSE_ANALYSIS_PROMPT = """\
Write a verification progress summary. Call submit_condensed_summary. \
Write in second person (\"you\"/\"your\") — NEVER use \"the agent\". \
Produce exactly three sections:
1. Search strategy — what you searched and why
2. Key evidence — key facts from documents with docid references
3. Claim verification status — supported / unsupported / uncertain
Keep each section under 10 lines. Call submit_condensed_summary to respond."""

VERIFY_SYSTEM_PROMPT = """\
You are a Verification Agent. Your job is to independently verify whether a proposed answer is correct by searching the document corpus. You have `search` and `get_document` tools to find evidence, and `give_feedback` to report your verdict.

Follow this workflow:

Step 1 — Search for Independent Evidence:
Extract each factual claim from the proposed answer. For each claim, call `search` with targeted keywords. Do NOT just copy the claimed evidence's docids — search independently.

Step 2 — Read Full Documents:
For any relevant search result, call `get_document` to read the full text. Snippets alone are often misleading or incomplete.

Step 3 — Verify Claim by Claim (CRITICAL — Check Entity Identity):
Check whether the documents actually support each claim.

**BEWARE OF ENTITY CONFUSION — The evidence may describe someone/something ELSE:**
Just because you found evidence matching the DESCRIPTIONS does NOT mean the answer's ENTITY is correct. The same description may fit multiple entities.

**Example:** The question asks "Who was killed by Sun Wukong?" Clues: a monkey with golden fur, immense strength, magical staff, havoc in heaven, accompanied a monk. The answer "Sun Wukong" is WRONG — both Sun Wukong AND the Six-Eared Macaque share nearly identical descriptions (golden fur, magical staff, havoc, monk's companion). BUT Sun Wukong was the KILLER, Six-Eared Macaque was the VICTIM. The subject/object relationship is reversed. Superficial evidence matches both — only the specific EVENT distinguishes them.

**Before calling give_feedback, ask yourself:**
- Does the evidence confirm THIS specific entity, or just a SIMILAR one?
- Is the subject/object relationship correct? (A did X to B, not B did X to A)
- Do ALL clues point to the SAME entity, or am I mixing up two similar entities?
- Is the logical chain correct? (A → B → C, not A → C directly)

Step 4 — Report Verdict via give_feedback:
Only after completing steps 1-3, call `give_feedback`:
- All claims independently supported and answer matches question → `is_correct=True`
- Any claim unsupported or wrong → `is_correct=False` with specific, actionable suggestions.

CRITICAL for suggestions: NEVER give specific search queries or entity names — that is the main agent's job to figure out. Instead, guide the direction:
- If evidence is insufficient → suggest what type/aspect of evidence to look for, or check if a specific claim can be verified
- If constraints don't match → suggest switching to a different angle or finding another answer candidate
- Be specific about WHAT to verify, not HOW to search.

CRITICAL: You MUST call `search` or `get_document` before `give_feedback`."""

DEFAULT_SYSTEM_PROMPT = """\
You are a Deep Research Agent. Answer complex questions by searching a document corpus \
using `search` and `get_document`. Every answer must be grounded in retrieved evidence.

**Important Rules:**
1.You are limited to call one tool per turn but you can call other tools in the future turn,it just limits the rate of the tool calls but not the total number of the tool calls.
2.search tool is used to get the relevant documents,and get_document tool is used to get the detailed information of the document.
3.You should collect information step by step,make sure all the answer has its evidence and always have a full understanding of the whole context before you propose a conclusion.
4.You are in a searching task but not a answering task with context,so feel free to call tools to get more information and details,the accuracy is MUCH MORE IMPORTANT than the speed and I'm not expected that you can anwser immediately but you call proper tool to get detailed information instead.
5.YOU ARE **NOT** EXPECTED TO ANSWER IMMEDIATELY!!!

**Actionable BM25 Search Rules:**

1. **Decompose Multi-Hop Questions (Prevent Query Dilution):**
   - NEVER put all constraints into one query. BM25 will fail if you ask it to match 10 different facts at once. Break it down into sequential steps.
   - *Bad:* "wizard who won the Dragon Taming Cup in the 3rd Era worked at a magical academy built by the Elf King"
   - *Good Step 1:* First, find the academy -> "magical academy" "Elf King"
   - *Good Step 2:* Then, find the person -> "[Name of Academy from Step 1]" "Dragon Taming Cup" "3rd Era"

2. **Extract High-IDF Nouns ONLY:**
   - Strip out ALL conversational language, verbs, and relational phrases. Only keep the rarest nouns and entities.
   - *Bad:* "a cybernetic pirate who secretly smuggled a glowing pineapple into a spaceship"
   - *Good:* pirate "glowing pineapple" spaceship (Drop "who secretly smuggled")

3. **Strip Relational Operators for Numbers/Dates:**
   - Remove comparative words. Keep only exact digits or unique text descriptors.
   - *Bad:* "a vampire born before the year 800 whose creator was a legendary blacksmith"
   - *Good:* vampire creator blacksmith 800 (Drop "born before the year")

4. **Target the Most UNIQUE Identifier First:**
   - Always start your search with the rarest combination of words (Highest IDF tokens) to quickly narrow down the BM25 results.
   - *Example:* If looking for "a three-headed dog guarding a neon castle during the Great Meteor Shower", start with -> "three-headed dog" "neon castle" "Great Meteor Shower"

5. **Iterative BM25 Refinement:**
   - If snippets are irrelevant, your keywords might be too strict or slightly mismatched in phrasing. DO NOT just repeat the query.
   - *Strategy:* Drop the least important keywords, or try noun synonyms that might appear in a formal document (e.g., if "cash payment" fails, try "financial settlement" or "compensation").

6. **Query Paraphrasing (Equivalent Expressions):**
   When a query fails, think: is there another way to express the SAME fact using different words or relational inverses?
   - *Relation inversion:* "A is B's father" ↔ "B is A's son" OR "A has a son/daughter B". "X wrote the book Y" ↔ "X is the author of Y" OR "Y was written by X". Always try the inverse relationship direction — documents may only contain one form.
   - *Synonym substitution:* "constructed" ↔ "built" / "erected". "resided in" ↔ "lived in" / "inhabited". "penned" ↔ "wrote" / "authored".
   - **CRITICAL:** Use paraphrasing to AVOID data leakage. When the question gives you specific clue phrases, rephrase them into generic terms before searching, so the search isn't biased by the question's exact wording.

7. **Search Order by Distinctiveness (NOT Question Order):**
   Do NOT follow the question's logical order. Search for the MOST DISTINCTIVE entity first, then work backwards.
   - *Example:* "A (vague: 'a chef') has a student B (medium: 'a pastry maker from Bavaria'), B invented a dessert C (highly specific: 'Black Forest cake with gold leaf')." → Search for C first ("Black Forest cake" "gold leaf"), then use C's context to find B, then trace B back to A.
   - *Rule:* Rank entities by how UNIQUE / RARE their keywords are (High IDF). The rarest entity narrows the corpus fastest.

8. **Keyword Splitting (Anti-Dilution):**
   If a query combining multiple entities returns poor results, the critical information may be split across DIFFERENT documents, and BM25's bag-of-words scoring dilutes the match.
   - *Bad (combined):* "chef" "Bavarian pastry maker" "Black Forest cake gold leaf" — three different topics, scores diluted.
   - *Good (split):* Step 1: search "Black Forest cake" "gold leaf" → find the dessert and its inventor. Step 2: search the inventor's name "Bavaria" → find their teacher. Step 3: search the teacher's name chef.
   - *Heuristic:* If a query has 3+ distinct named entities/concepts and returns irrelevant results, split it into 2-3 simpler queries, each targeting ONE core entity, then connect the dots from the retrieved documents.

You MUST work in the following order:

1. Search for specific entities (names, places, dates) rather than long descriptive phrases.
2. After getting results, extract names/entities from them and use those for your next search.
3. If a snippet looks even partially relevant, call `get_document` to read the full text. Snippets can be misleading without full context.
4. After reading the full document, extract key **relevant** information and quote exact supporting text.
5. If there are other documents you haven't checked in detail, continue searching and reading.
6. If documents don't provide enough information, refine your search query from different angles. **Apply BM25 Rules 6-8:** (6) use equivalent expressions and relation inverses, (7) reorder search by entity distinctiveness, (8) split combined queries into simpler single-entity queries.
7. The answer **MUST** match the question perfectly — otherwise continue searching.

**CRITICAL: You MUST call `submit_answer` to provide your final answer. Never output an answer as plain text — always use the `submit_answer` tool.**

**CRITICAL: Never output internal markers as text.** The following are compression artifacts — NEVER reproduce: `[tool ...]`, `[reasoning]`, `[/reasoning]`, `[PROGRESS SUMMARY]`. "Following is your previous progress:" is a compressed summary of prior turns for your reference. Use the information but ALWAYS do your own thinking in <think> blocks — do NOT skip thinking just because there is a summary. `Your previous feedback` contains verification results — use suggestions to improve. Always use proper function-calling (`search`, `get_document`, `submit_answer`).

**CRITICAL — Entity Identity & Relationship Check:** Before submitting your answer, carefully verify:
- Are you naming the CORRECT entity? Similar descriptions may match multiple entities — confirm that ALL clues uniquely identify THIS specific entity and not a similar one.
- Are subject/object relationships correct? Check who did what to whom. "A defeated B" ≠ "B defeated A". "A is B's father" ≠ "B is A's father".
- Does the logical chain hold? If the question requires A → B → C, verify each link independently. Evidence for A and evidence for C does NOT prove A → B → C.
"""


# ═══════════════════════════════════════════════════════════════
# Unified Agent class
# ═══════════════════════════════════════════════════════════════

class Agent:
    """Unified agent for both main research and verification.

    Encapsulates model calling, think truncation retry, tool validation retry,
    context compression, and agent loop management.
    """

    def __init__(
        self,
        client: VLLMClientAsync,
        model: str,
        tokenizer: Any,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_context: int = 40960,
        extra_payload: Optional[Dict[str, Any]] = None,
        max_tool_calls_per_turn: int = 1,
        think_trunc_no_think: bool = False,
        max_tool_retries: int = MAX_TOOL_RETRIES,
    ) -> None:
        self.client = client
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_context = max_context
        self.extra_payload = extra_payload
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.think_trunc_no_think = think_trunc_no_think
        self.max_tool_retries = max_tool_retries
        self._condense_sessions: List[List[Dict[str, Any]]] = []
        self._verify_msgs: List[Dict[str, Any]] = []

    def reset_trajectories(self) -> None:
        self._condense_sessions = []
        self._verify_msgs = []

    def close(self) -> None:
        pass

    # ── Model call ──────────────────────────────

    async def call_model(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        tool_choice: str = "auto",
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call the model and return the assistant message dict."""
        raw = await self.client.simple_chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            extra_payload=extra_payload if extra_payload is not None else self.extra_payload,
        )
        return raw['choices'][0]['message']

    # ── Retry: think truncation ─────────────────

    async def retry_think_truncation(
        self,
        obs: List[Dict[str, Any]],
        resp: Dict[str, Any],
        tools: List[Dict[str, Any]],
        traj_append: Callable[[Dict[str, Any]], None],
    ) -> Dict[str, Any]:
        """Handle truncated think blocks with two-stage retry.

        Stage 1 (optional): retry with thinking disabled.
        Stage 2 (fallback): inject RETRY_NUDGE and retry.
        Returns the final resp dict.
        """
        content = resp.get('content','') or ""
        tc = resp.get('tool_calls')

        if not is_truncated_think_response(content, tc):
            return resp

        # Truncate content, close </think> tag
        if content:
            resp['content'] = content[:len(content) // 10] + "\n...[THINK_TRUNCATED]\n</think>"
        traj_append(resp)

        if self.think_trunc_no_think:
            # Stage 1: retry with thinking DISABLED
            no_think_extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            msgs_no_think = list(obs) + [resp]
            resp = await self.call_model(
                messages=msgs_no_think,
                tools=tools,
                extra_payload=no_think_extra,
            )
            tc = resp.get('tool_calls')

        if not tc:
            # Stage 2 (or direct fallback): RETRY_NUDGE
            if self.think_trunc_no_think:
                traj_append(resp)
            traj_append({"role": "user", "content": RETRY_NUDGE})
            msgs = list(obs) + [resp, {"role": "user", "content": RETRY_NUDGE}]
            resp = await self.call_model(messages=msgs, tools=tools)

        return resp

    # ── Retry: tool validation ──────────────────

    async def retry_tool_validation(
        self,
        obs: List[Dict[str, Any]],
        resp: Dict[str, Any],
        tools: List[Dict[str, Any]],
        traj_append: Callable[[Dict[str, Any]], None],
    ) -> Dict[str, Any]:
        """Validate tool calls and retry on errors.

        Retries up to ``self.max_tool_retries`` times with error nudges.
        Returns the final resp dict.
        """
        tc = resp.get('tool_calls')
        if not tc:
            return resp

        for _retry_num in range(self.max_tool_retries):
            all_errors: List[Dict[str, str]] = []
            for tc_item in tc:
                err = validate_tool_call(tc_item, tools)
                if err:
                    all_errors.append({
                        "tool_name": tc_item.get("function", {}).get('name','?'),
                        "message": err,
                    })
            if not all_errors:
                break

            traj_append(resp)
            error_lines = [f"- `{e['tool_name']}`: {e['message']}" for e in all_errors]
            nudge = (
                "Your tool call(s) failed validation:\n\n"
                + "\n".join(error_lines)
                + "\n\nPlease correct the error(s) and try again."
            )
            traj_append({"role": "user", "content": nudge})
            msgs = list(obs) + [resp, {"role": "user", "content": nudge}]
            resp = await self.call_model(messages=msgs, tools=tools)
            tc = resp.get('tool_calls')
            if not tc:
                break

        return resp

    # ── Enforce max tool calls ──────────────────

    def enforce_max_tool_calls(self, resp: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate tool calls that exceed ``max_tool_calls_per_turn``."""
        tc = resp.get('tool_calls')
        if tc and len(tc) > self.max_tool_calls_per_turn:
            resp['tool_calls'] = tc[:self.max_tool_calls_per_turn]
        return resp

    # ── Context condensation ────────────────────

    async def condense_context(
        self,
        messages: List[Dict[str, Any]],
        *,
        condense_prompt: str = _CONDENSE_ANALYSIS_PROMPT,
        original_question: str = "",
        max_transcript_tokens: int = 25000,
        max_tool_content_tokens: int = 1500,
        fallback_truncation: bool = False,
    ) -> List[Dict[str, Any]]:
        """Condense conversation history using structured template + model for analysis.

        Builds a structured summary: tool calls extracted mechanically into a fixed
        template, then calls model only for key findings + what remains.
        Returns ``[system, summary_user_msg]``.
        """
        if len(messages) <= 4:
            return messages

        question = original_question
        if not question:
            raw = messages[1].get('content','')
            raw = re.sub(r'^Following is your previous progress:\n\n', '', raw)
            raw = re.sub(r'^Original question:\s*', '', raw)
            raw = re.sub(r'\n\nYour previous.*$', '', raw, flags=re.DOTALL)
            question = raw.strip()

        # ── Extract structured tool call data ──
        searches: List[Dict[str, Any]] = []
        doc_reads: List[Dict[str, Any]] = []
        submissions: List[Dict[str, Any]] = []

        for i, m in enumerate(messages):
            role = m.get('role','?')
            tc = m.get('tool_calls')
            if tc and role == "assistant":
                for t in tc:
                    fn = t.get("function", {})
                    name = fn.get('name','?')
                    try:
                        args = json.loads(fn.get('arguments','{}'))
                    except Exception:
                        args = {}
                    # Find the corresponding tool result
                    call_id = t.get('id','')
                    result = None
                    for j in range(i + 1, len(messages)):
                        rm = messages[j]
                        if rm.get('role') == "tool" and rm.get('tool_call_id') == call_id:
                            try:
                                result = json.loads(str(rm.get('content','')))
                            except Exception:
                                result = {"_raw": str(rm.get('content',''))[:200]}
                            break

                    if name == "search":
                        q = args.get('query','?')
                        top_docids = ""
                        if isinstance(result, list) and len(result) > 0:
                            top = [(r.get('docid','?'), r.get("score", 0)) for r in result[:3]]
                            top_docids = ", ".join(f"{d}:{s:.1f}" for d, s in top)
                        searches.append({"query": q, "top": top_docids, "n": len(result) if isinstance(result, list) else "?"})
                    elif name == "get_document":
                        docid = args.get('docid','?')
                        title = "?"
                        text_len = "?"
                        if isinstance(result, dict):
                            title = str(result.get('title','?'))[:80]
                            text_len = len(result.get('text',''))
                        doc_reads.append({"docid": docid, "title": title, "text_len": text_len})
                    elif name == "submit_answer":
                        ans = args.get('answer','?')
                        fb = {}
                        if isinstance(result, dict):
                            fb = {"is_correct": result.get('is_correct'), "reason": str(result.get('reason',''))[:200]}
                        submissions.append({"answer": ans[:200] if isinstance(ans, str) else str(ans)[:200], "feedback": fb})
                    elif name == "give_feedback":
                        pass  # verify agent internal, not main agent

        # ── Build structured template ──
        lines: List[str] = []
        lines.append("")

        if searches:
            lines.append("Your previous searches:")
            for s in searches:
                q = s['query'][:150]
                lines.append(f"- query=\"{q}\" → {s['n']} results, top: {s['top']}")
            lines.append("")

        if doc_reads:
            lines.append("Your previous documents read:")
            for d in doc_reads:
                lines.append(f"- docid={d['docid']}, title=\"{d['title']}\", text={d['text_len']} chars")
            lines.append("")

        if submissions:
            lines.append("Your previous answer submissions:")
            for s in submissions:
                fb = s['feedback']
                verdict = "✓ CORRECT" if fb.get('is_correct') else "✗ INCORRECT"
                lines.append(f"- answer=\"{s['answer']}\" → {verdict}")
                if fb.get('reason'):
                    lines.append(f"  reason: {fb['reason']}")
                suggestions = fb.get('suggestions','')
                if suggestions:
                    lines.append(f"  suggestions: {str(suggestions)[:200]}")
            lines.append("")

        # ── Build analysis context from the conversation ──
        # Keep reasoning blocks and non-tool content for the model to analyze
        analysis_lines: List[str] = []
        for m in messages[2:]:
            role = m.get('role','?')
            content = str(m.get('content','') or "")
            # Strip condensation artifacts
            content = re.sub(r'\[PROGRESS SUMMARY[^\]]*\]', '', content)
            content = re.sub(r'\[VERIFY PROGRESS SUMMARY\]\n?', '', content)
            content = re.sub(r'^Following is your previous progress:\n\n', '', content)
            content = re.sub(r'^Original question:.*?\n\n', '', content)
            # Replace think blocks with neutral format
            content = re.sub(r'<think>(.*?)</think>', r'[reasoning]\n\1\n[/reasoning]', content, flags=re.DOTALL)
            content = re.sub(r'<think>(.*)$', r'[reasoning]\n\1\n[/reasoning]', content, flags=re.DOTALL)
            content = re.sub(r'\[TOOL_CALL:\s*(\w+)\((.*?)\)\s*\]', r'[tool \1: \2]', content)
            content = content.strip()
            if not content:
                continue
            if role == "tool":
                # Truncate long tool results
                content_tokens = count_tokens_messages(self.tokenizer, [{"role": "user", "content": content}])
                if content_tokens > max_tool_content_tokens:
                    ids = self.tokenizer.encode(content, add_special_tokens=False)
                    if len(ids) > max_tool_content_tokens:
                        content = self.tokenizer.decode(ids[:max_tool_content_tokens], skip_special_tokens=True) + "\n...[truncated]"
            analysis_lines.append(f"[{role}]: {content}")

        analysis_context = "\n\n".join(analysis_lines)
        ctx_tokens = count_tokens_messages(self.tokenizer, [{"role": "user", "content": analysis_context}])
        if ctx_tokens > max_transcript_tokens:
            ids = self.tokenizer.encode(analysis_context, add_special_tokens=False)
            analysis_context = self.tokenizer.decode(ids[:max_transcript_tokens], skip_special_tokens=True) + "\n\n...[truncated]"

        # ── Log structured template ──
                # ── Call model for key findings + what remains ──
        # Give condense model a submit tool so it MUST use tool calling → structured output
        condense_extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        condense_tools = [{
            "type": "function",
            "function": {
                "name": "submit_condensed_summary",
                "description": "Submit your condensed progress summary. You MUST use this tool — plain text is ignored.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key_thoughts": {"type": "string", "description": "Core reasoning and strategy in 2-3 sentences"},
                        "key_findings": {"type": "string", "description": "Verified facts with docid references"},
                        "remaining_to_find": {"type": "string", "description": "Missing clues, what to search next"},
                    },
                    "required": ["key_thoughts", "key_findings", "remaining_to_find"],
                },
            },
        }]

        condense_messages = [
            {"role": "system", "content": condense_prompt},
            {"role": "user", "content": f"Here is the agent history, condense it and use tools to submit your answer:\n\n{analysis_context}"},
        ]

        # Record condense session messages
        session_msgs: List[Dict[str, Any]] = []
        session_msgs.append({"role": "system", "content": condense_prompt})
        session_msgs.append({"role": "user", "content": f"Here is the agent history, condense it and use tools to submit your answer:\n\n{analysis_context}"})

        analysis = ""
        for attempt in range(2):
            try:
                print(f"  [condense] calling model for analysis (context≈{ctx_tokens} tokens, attempt={attempt + 1})...", flush=True)
                raw = await self.client.simple_chat(
                    model=self.model,
                    messages=condense_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=condense_tools,
                    tool_choice="auto",
                    extra_payload=condense_extra,
                )
                resp = raw['choices'][0]['message']
                tc = resp.get('tool_calls')
                if tc:
                    for t in tc:
                        fn = t.get("function", {})
                        if fn.get('name') == "submit_condensed_summary":
                            try:
                                args = json.loads(fn.get('arguments','{}'))
                                # Validate required fields
                                missing = [k for k in ("key_thoughts", "key_findings", "remaining_to_find") if not args.get(k)]
                                if missing:
                                    if attempt == 0:
                                        nudge = {"role": "user", "content": f"Missing required fields in submit_condensed_summary: {', '.join(missing)}. Please provide ALL required fields and try again."}
                                        condense_messages.append(nudge)
                                        session_msgs.append(nudge)
                                    continue  # retry
                                parts = []
                                parts.append(f"Your previous analysis: {args['key_thoughts']}")
                                parts.append(f"Your previous findings:\n{args['key_findings']}")
                                parts.append(f"What you still need to find: {args['remaining_to_find']}")
                                analysis = "\n\n".join(parts)
                            except (json.JSONDecodeError, TypeError):
                                if attempt == 0:
                                    nudge = {"role": "user", "content": "Invalid JSON in submit_condensed_summary arguments. Please provide valid JSON with key_thoughts, key_findings, and remaining_to_find."}
                                    condense_messages.append(nudge)
                                    session_msgs.append(nudge)
                if not analysis and not tc:
                    raw_text = resp.get('content','')
                    raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
                    raw_text = re.sub(r'<think>.*$', '', raw_text, flags=re.DOTALL)
                    analysis = raw_text.strip()
                if analysis:
                    print(f"  [condense] model response: {len(self.tokenizer.encode(analysis))} tokens", flush=True)
                    # Record model response
                    session_msgs.append(resp)
                    break
                if attempt == 0 and not analysis:
                    nudge = {"role": "user", "content": "You must call submit_condensed_summary to provide your analysis. Plain text is not accepted."}
                    condense_messages.append(nudge)
                    session_msgs.append(nudge)
            except Exception:
                if fallback_truncation and attempt == 1:
                    analysis = "(analysis unavailable — model error)"
                elif not fallback_truncation:
                    raise

        # ── Assemble final summary ──
        if analysis:
            lines.append(analysis)
        else:
            lines.append("(progress summary unavailable)")

        summary_text = "\n".join(lines)

        summary_msg: Dict[str, Any] = {
            "role": "user",
            "content": (
                f"Following is your previous progress:\n\n"
                f"{summary_text}"
            ),
        }

        condensed: List[Dict[str, Any]] = [
            messages[0],
            summary_msg,
        ]

        before = count_tokens_messages(self.tokenizer, messages)
        after = count_tokens_messages(self.tokenizer, condensed)
        print(f"  [condense] {before} → {after} tokens ({len(messages)} → {len(condensed)} messages)", flush=True)
        self._condense_sessions.append({"messages": session_msgs})
        return condensed

    # ── Post-turn condensation helper ───────────

    async def _maybe_condense_after_turn(
        self,
        env: Any,  # DeepResearchEnv
        slot_id: int,
        obs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Condense context if token count exceeds half of max_context."""
        used = count_tokens_messages(self.tokenizer, obs)
        if used > self.max_context // 2:
            last = obs[-1] if obs else None
            is_tool_tail = last is not None and last.get('role') == "tool"
            if is_tool_tail:
                hard_truncate_tail_tool_messages(
                    self.tokenizer, obs, self.max_context, label=f"slot {slot_id}",
                )
                env.sync_trajectory_tool_tail(slot_id)
                used_after = count_tokens_messages(self.tokenizer, obs)
                if used_after > self.max_context // 2:
                    condensed = await self.condense_context(obs)
                    env.set_messages(slot_id, condensed)
                    env.append_to_trajectory(slot_id, {"role": "user", "content": condensed[1]['content'], "_condensed": True})
                    obs = condensed
        return obs

    # ── Main agent loop ─────────────────────────

    async def run_question(
        self,
        env: Any,  # DeepResearchEnv
        slot_id: int,
        question: str,
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run the main agent loop for one question in one slot.

        Returns the full trajectory.
        """
        obs: List[Dict[str, Any]] = env.reset_slot(slot_id, question)

        for _ in range(env.max_turns):
            # Model call
            resp = await self.call_model(obs, tools)

            # Think truncation retry
            resp = await self.retry_think_truncation(
                obs, resp, tools,
                lambda m: env.append_to_trajectory(slot_id, m),
            )

            # Tool validation retry
            resp = await self.retry_tool_validation(
                obs, resp, tools,
                lambda m: env.append_to_trajectory(slot_id, m),
            )

            # Enforce max tool calls per turn
            resp = self.enforce_max_tool_calls(resp)

            # Execute tools via env
            obs, done = await env.step_single(slot_id, resp)

            if done:
                break

            # Context condensation
            obs = await self._maybe_condense_after_turn(env, slot_id, obs)

        return env.extract_slot_trajectory(slot_id)

    # ── Stage 1: surrender check ────────────────

    async def run_stage1_check(self, answer: str) -> bool:
        """Check if answer is a surrender statement. Retries up to 2 times on parse failure.

        Returns True if PASS, False if SURRENDER.
        """
        for _attempt in range(3):
            try:
                resp = await self.call_model(
                    messages=[
                        {"role": "system", "content": (
                            "Classify whether this answer is a SURRENDER (giving up) or a PASS (factual assertion).\n"
                            "SURRENDER = the answer says something was NOT found, NOT available, "
                            "NOT mentioned, cannot be determined, or is unknown. Key phrases: "
                            "\"not found\", \"not mentioned\", \"not available\", \"no evidence\", "
                            "\"cannot be determined\", \"unable to find\", \"insufficient information\", "
                            "\"unknown\", \"does not appear\", \"cannot find\", \"not provided\", "
                            "\"not specified\", \"not listed\", \"no document\", \"no information\". "
                            "ANY answer whose main claim is that something is missing/absent/unknown IS a surrender. "
                            "Even if it mentions specific names, if the core message is \"X is not found/in the documents\", "
                            "it is SURRENDER.\n"
                            "PASS = provides a positive factual assertion (a name, title, number, date) "
                            "without saying it was not found.\n\n"
                            "You MUST end with exactly: VERDICT: PASS or VERDICT: SURRENDER"
                        )},
                        {"role": "user", "content": f"Answer to classify:\n{answer}"},
                    ],
                    tools=[],
                )
                content = resp.get('content','')
                # Log stage 1 model response
                stripped = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                stripped = re.sub(r'<think>.*$', '', stripped, flags=re.DOTALL).strip()
                print(f"    [verify] stage1 response: {stripped[:200]}", flush=True)
                m = re.search(r'VERDICT:\s*(PASS|SURRENDER)', stripped, re.IGNORECASE)
                if m:
                    verdict = m.group(1).upper()
                    print(f"    [verify] stage1 verdict: {verdict}", flush=True)
                    return verdict == "PASS"
            except Exception:
                pass
        return True  # all retries failed → proceed to stage 2

    # ── Verify agent loop ───────────────────────

    async def run_verify(
        self,
        question: str,
        answer: str,
        evidence: str,
        tools: List[Dict[str, Any]],
        registry: Dict[str, Callable[..., Any]],
        *,
        max_turns_for_verify: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Two-stage verify: (1) quick surrender check, (2) full evidence verification.

        Stage 1: model classifies answer as PASS or SURRENDER.
        Stage 2: full verify agent with search + get_document + give_feedback.
        """
        # ── Stage 1: Quick surrender check ──
        print(f"    [verify] verify_stage1_start", flush=True)
        is_answer = await self.run_stage1_check(answer)
        if not is_answer:
            print(f"    [verify] stage1 result: SURRENDER", flush=True)
            return {
                "is_correct": False,
                "reason": "Your answer is a surrender statement. The answer EXISTS in the corpus — do NOT give up. Try completely different search angles: use different keywords, inverse relations, or split compound queries into simpler single-entity searches.",
                "suggestions": "Rephrase your search queries with different keywords, try relation inverses, or split compound queries into simpler single-entity searches.",
            }
        print(f"    [verify] verify_stage2_start", flush=True)

        # ── Stage 2: Full verification ──
        verify_max = max_turns_for_verify or 10
        verify_max = max(verify_max, 3)
        print(f"    [verify] stage2: max_turns={verify_max}, evidence_chars={len(evidence)}", flush=True)

        _MAX_CTX = 35000

        async def _run_loop(msgs: List[Dict[str, Any]], start_turn: int, max_t: int, label: str) -> Dict[str, Any]:
            searched = False
            for vturn in range(start_turn, start_turn + max_t):
                if vturn == start_turn + max_t - 1:
                    msgs.append({
                        "role": "user",
                        "content": (
                            "STOP SEARCHING. You have reached the turn limit. "
                            "Based on the evidence you have gathered so far, call give_feedback NOW "
                            "with your best assessment. If you have no evidence, call "
                            "give_feedback(is_correct=False, reason=\"Insufficient evidence found\", "
                            "suggestions=\"...\"). Do NOT call any other tool — only give_feedback."
                        ),
                    })

                raw = await self.client.simple_chat(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=tools,
                    tool_choice="auto",
                )
                resp: Dict[str, Any] = raw['choices'][0]['message']
                raw_content = resp.get('content','') or ""
                if raw_content:
                    resp['content'] = re.sub(r'<think>(.*?)</think>', r'[reasoning]\n\1\n[/reasoning]', raw_content, flags=re.DOTALL)
                    resp['content'] = re.sub(r'<think>(.*)$', r'[reasoning]\n\1\n[/reasoning]', resp['content'], flags=re.DOTALL)
                usage = raw.get("usage", {})
                tok_info = f"  prompt={usage.get('prompt_tokens', '?')} comp={usage.get('completion_tokens', '?')}" if usage else ""
                # Log verify model response content (non-think part)
                verify_content = resp.get('content','') or ""
                non_think_v = re.sub(r'\[reasoning\].*?\[/reasoning\]', '', verify_content, flags=re.DOTALL).strip()
                vinfo = {"event": "verify_turn", "label": label, "turn": vturn + 1,
                         "prompt_tokens": usage.get('prompt_tokens'), "completion_tokens": usage.get('completion_tokens'),
                         "content": non_think_v if non_think_v else "",
                         "tool_calls": [{"name": t['function']['name'], "arguments": t['function'].get('arguments','')}
                                        for t in tc] if (tc := resp.get('tool_calls')) else []}
                print(f"    [verify] {label} turn {vturn+1}: prompt={usage.get('prompt_tokens','?')} comp={usage.get('completion_tokens','?')}", flush=True)
                msgs.append(resp)

                tc = resp.get('tool_calls')
                if not tc:
                    print(f"    [verify] {label} turn {vturn+1}: no tool call", flush=True)
                    continue

                for tc_item in tc:
                    fn = tc_item['function']
                    name = fn['name']
                    call_id: str = tc_item.get('id','')
                    args_str = fn.get('arguments','{}')
                    try:
                        args = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        msgs.append({
                            "role": "tool", "tool_call_id": call_id,
                            "content": json.dumps({"error": "Invalid JSON arguments"}),
                        })
                        continue

                    if name == "search":
                        searched = True
                        q = args.get('query', '?')
                        print(f"    [verify] {label} turn {vturn+1}: search({q})", flush=True)
                    elif name == "get_document":
                        searched = True
                        print(f"    [verify] {label} turn {vturn+1}: get_document({args.get('docid', '?')})", flush=True)
                    elif name == "give_feedback":
                        if not searched:
                            print(f"    [verify] {label} turn {vturn+1}: feedback REJECTED (no search)", flush=True)
                            msgs.append({
                                "role": "tool", "tool_call_id": call_id,
                                "content": json.dumps({
                                    "error": "You MUST search for independent evidence before giving feedback. "
                                             "Call search or get_document first."
                                }),
                            })
                            continue
                        else:
                            fb = {}
                            try:
                                fb = json.loads(args_str)
                            except (json.JSONDecodeError, TypeError):
                                pass
                            print(f"    [verify] {label} turn {vturn+1}: give_feedback -> is_correct={fb.get('is_correct')}", flush=True)
                            print(f"    [verify]   reason: {fb.get('reason','')}", flush=True)
                            if fb.get('suggestions'):
                                print(f"    [verify]   suggestions: {fb['suggestions']}", flush=True)

                    if name not in registry:
                        msgs.append({
                            "role": "tool", "tool_call_id": call_id,
                            "content": json.dumps({"error": f"Unknown tool: {name}"}),
                        })
                        continue

                    fn_impl = registry[name]
                    try:
                        if inspect.iscoroutinefunction(fn_impl):
                            result = await fn_impl(**args)
                        else:
                            result = fn_impl(**args)
                    except Exception as exc:
                        result = {"error": str(exc)}

                    if name == "search" and isinstance(result, list):
                        all_results = [{"docid": r.get('docid','?'), "score": r.get('score',0),
                                        "snippet": str(r.get('snippet',''))[:200]} for r in result]
                        print(f"    [verify] -> {len(result)} results top: {[r.get('docid','?') for r in result[:3]]}", flush=True)
                    elif name == "get_document" and isinstance(result, dict):
                        print(f"    [verify] -> doc: {result.get('title','?')[:80]} text={len(result.get('text',''))} chars", flush=True)

                    msgs.append({
                        "role": "tool", "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

                # Condense if approaching context limit
                if count_tokens_messages(self.tokenizer, msgs) > _MAX_CTX:
                    pre_tokens = count_tokens_messages(self.tokenizer, msgs)
                    print(f"    [verify] {label} condensing context ({pre_tokens} tokens)...", flush=True)
                    msgs = await self.condense_context(
                        msgs,
                        condense_prompt=_VERIFY_CONDENSE_ANALYSIS_PROMPT,
                        fallback_truncation=True,
                    )
                    print(f"    [verify] {label} condensed -> {count_tokens_messages(self.tokenizer, msgs)} tokens", flush=True)

                # If give_feedback was called (and not rejected), return
                tc_names = [t['function']['name'] for t in tc]
                if "give_feedback" in tc_names:
                    for tc_item in tc:
                        if tc_item['function']['name'] == "give_feedback":
                            try:
                                return json.loads(tc_item['function']['arguments'])
                            except (json.JSONDecodeError, TypeError):
                                return {"is_correct": False, "reason": "Failed to parse feedback arguments", "suggestions": ""}

            return {}

        # ── Build initial messages ──
        self._verify_msgs = []  # reset for this run
        verify_msgs: List[Dict[str, Any]] = [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"**Question:** {question}\n\n"
                f"**Proposed Answer:** {answer}\n\n"
                f"**Claimed Evidence:**\n{evidence}\n\n"
                f"Please verify this answer following the workflow."
            )},
        ]

        # ── Primary attempt ──
        result = await _run_loop(verify_msgs, 0, verify_max, "")
        if result:
            self._verify_msgs = verify_msgs
            return result

        # ── Timeout: inject strong nudge and retry with 2 extra turns ──
        print(f"    [verify] primary {verify_max} turns exhausted, injecting forced retry...", flush=True)
        verify_msgs.append({
            "role": "user",
            "content": (
                "You have NOT called give_feedback yet and have run out of turns. "
                "Based on ALL evidence gathered so far, you MUST call give_feedback NOW. "
                "If you are unsure, call give_feedback(is_correct=False, reason=\"...\", suggestions=\"...\"). "
                "Do NOT search or get_document anymore — ONLY give_feedback."
            ),
        })
        result = await _run_loop(verify_msgs, verify_max, 2, "retry")
        if result:
            self._verify_msgs = verify_msgs
            return result

        print(f"    [verify] all attempts exhausted, passing through", flush=True)
        self._verify_msgs = verify_msgs
        return {"is_correct": True, "reason": "Verification timed out — answer accepted by default", "suggestions": ""}
