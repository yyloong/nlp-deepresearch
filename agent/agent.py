"""
Unified Agent class for both main research agent and verification agent.

Encapsulates:
- Model calling
- Think truncation retry (two-stage: no-think → RETRY_NUDGE)
- Tool call validation and retry
- Context compression (with think block handling)
- Main agent loop and verify agent loop
"""

# ⚠️ CRITICAL: NEVER put <think>, </think>, [reasoning], [/reasoning], or any
# internal format markers in model-facing prompt strings in this file.
# These are internal artifacts and must never leak into model prompts.

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
Extract key information from the conversation into four fields. \
Call submit_condensed_summary with ALL four fields filled in.

tool_summary: Summarize what tools were used and what was found. \
Deduplicate repeated searches -- mention each unique query once with its top results. \
List each unique document read with key content extracted (not just docid). \
Summarize each unique answer submission and the feedback received. \
Be concise but include specific names, docids, and key details from results.

key_thoughts: The reasoning strategy -- what hypotheses were being tested, \
what logical chains were being followed, what clue connections were being explored. \
Capture the thinking process so dead ends are not repeated.

key_findings: What FACTS were actually found in documents. Include specific names, \
dates, numbers, titles, relationships, quotes with docid references. \
Include both supporting and contradictory evidence.

remaining_to_find: Which specific clues from the question are still unsolved. \
Be precise -- not \"identify the club\" but \"find club name starting with B, \
4 syllables, linked to Latin music, connected to surname Franzini\".

RULES:
- Be dense: every sentence should contain a useful fact or insight
- If an answer was rejected, extract what was learned from the feedback
"""

# Prompt for model-generated analysis in verify condense
_VERIFY_CONDENSE_ANALYSIS_PROMPT = """\
Extract key information from the verification conversation into four fields. \
Call submit_condensed_summary with ALL four fields filled in.

tool_summary: Summarize what searches were performed (deduplicated), \
what documents were read with key content, and what feedback was given. \
Be concise but include specific queries, docids, and key details.

key_thoughts: The verification strategy -- why those searches were chosen, \
what hypotheses about the answer were being tested.

key_findings: Key evidence found in documents with docid references. \
Which claims are supported, which are contradicted, which lack evidence.

remaining_to_find: What still needs to be verified -- specific claims \
not yet checked, documents not yet read, angles not yet explored.

RULES:
- Be dense: every sentence should contain a useful fact or insight
"""

VERIFY_SYSTEM_PROMPT = """\
You are a Verification Agent. Your job is to independently verify whether a proposed answer is correct by searching the document corpus. You have `search` and `get_document` tools to find evidence, and `give_feedback` to report your verdict.

Follow this workflow:

Step 1 -- Search for Independent Evidence:
Extract each factual claim from the proposed answer. For each claim, call `search` with targeted keywords. Do NOT just copy the claimed evidence's docids -- search independently.

Step 2 -- Read Full Documents:
For any relevant search result, call `get_document` to read the full text. Snippets alone are often misleading or incomplete.

Step 3 -- Verify Claim by Claim (CRITICAL -- Check Entity Identity):
Check whether the documents actually support each claim.

**BEWARE OF ENTITY CONFUSION -- The evidence may describe someone/something ELSE:**
Just because you found evidence matching the DESCRIPTIONS does NOT mean the answer's ENTITY is correct. The same description may fit multiple entities.

**Example:** The question asks "Who was killed by Sun Wukong?" Clues: a monkey with golden fur, immense strength, magical staff, havoc in heaven, accompanied a monk. The answer "Sun Wukong" is WRONG -- both Sun Wukong AND the Six-Eared Macaque share nearly identical descriptions (golden fur, magical staff, havoc, monk's companion). BUT Sun Wukong was the KILLER, Six-Eared Macaque was the VICTIM. The subject/object relationship is reversed. Superficial evidence matches both -- only the specific EVENT distinguishes them.

**Before calling give_feedback, ask yourself:**
- Does the evidence confirm THIS specific entity, or just a SIMILAR one?
- Is the subject/object relationship correct? (A did X to B, not B did X to A)
- Do ALL clues point to the SAME entity, or am I mixing up two similar entities?
- Is the logical chain correct? (A → B → C, not A → C directly)

Step 4 -- Report Verdict via give_feedback:
Only after completing steps 1-3, call `give_feedback`:
- All claims independently supported and answer matches question → `is_correct=True`
- Any claim unsupported or wrong → `is_correct=False` with specific, actionable suggestions.

CRITICAL for suggestions: NEVER give specific search queries or entity names -- that is the main agent's job to figure out. Instead, guide the direction:
- If evidence is insufficient → suggest what type/aspect of evidence to look for, or check if a specific claim can be verified and **suggest the agent if can not find efficient evidence ,restart from another angle**.
- If constraints don't match → suggest switching to a different angle or finding another answer candidate
- Be specific about WHAT to verify, not HOW to search.

CRITICAL: You MUST call `search` or `get_document` before `give_feedback`."""

DEFAULT_SYSTEM_PROMPT = """\
You are a Deep Research Agent. Answer complex questions by searching a document corpus \
using `search` and `get_document`. Every answer must be grounded in retrieved evidence.

**HOW TO SEARCH:**

BM25 has NO understanding of word order or meaning. Your query is split into individual tokens, and each token is matched independently — a document matching any single token appears in results. Generic tokens match thousands of docs and bury the relevant one. Short queries (2-3 words) using rare, specific tokens work best.

Example search chain:
  Q: "A wizard won the Dragon Taming Cup at an academy built by the Elf King. Name his familiar."
  1. search "Elf King academy" (3 words, rarest clue) → finds "Crystal Tower Academy"
  2. search "Crystal Tower" "Dragon Cup" (entity + next clue) → finds "Gandalf"
  3. search "Gandalf familiar" (2 words) → "Shadowfax". Done.

CRITICAL SEARCH RULES:
1. FIRST search = the RAREST clue word (profession, object, name). NEVER a vague description.
   Vague queries return noise — \"third era\" matches 500+ docs, \"Elf King\" matches 5.
2. 2-3 words per query, NEVER exceed 5. More words = worse.
3. Chain: entity from result + one new clue.
4. If stuck, completely DIFFERENT clue. Never rephrase a failed query.
5. Call get_document on promising results BEFORE searching again. Snippets show only the first part — chapter titles and key details are often deeper in the document. You cannot judge relevance from snippet alone. When reading a book text, look for \"CHAPTER I\" or \"Chapter 1\" to find the exact first chapter title. Quote it precisely.

**Important Rules:**
1.You are limited to call one tool per turn.
2.Use search for documents, get_document for full text.
3.Accuracy over speed. YOU ARE NOT EXPECTED TO ANSWER IMMEDIATELY.

You MUST work in this order:
1. First, summarize what you already know: which clues from the question are you working on? What entities have you found so far? What remains unknown?
2. Pick the most specific piece of information you have right now — a name, a date, a title. Use that to search next. Always prefer concreteness over vagueness.
3. DIG DEEPER into each entity before moving on. Exhaust what you can learn from one finding, then use it to find the next.
4. After each search, update your summary: what new entity/fact did you find? What's still missing?
5. When a result looks relevant, call get_document to read the full text.

**CRITICAL: NEVER guess from prior knowledge.** Every answer MUST be found in retrieved documents. If search results don't contain the answer, search from a completely different angle — do NOT fall back on what you think you know.

**CRITICAL: You MUST call `submit_answer` to provide your final answer.**

**CRITICAL: Never output `[tool ...]`, `[reasoning]`, `[/reasoning]` as text.** "Following is your previous progress:" is a compressed summary -- use it but continue your own thinking. Feedback from submit_answer tells you why your answer was wrong -- use suggestions to improve.

**CRITICAL: If the same answer has been rejected multiple times, change target and restart from a completely different angle.**
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
        condense_extra = None  # let the model think -- compression needs reasoning
        condense_tools = [{
            "type": "function",
            "function": {
                "name": "submit_condensed_summary",
                "description": "Submit your condensed progress summary. You MUST use this tool -- plain text is ignored.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_summary": {"type": "string", "description": "Summary of tools used: searches (deduplicated), documents read with key content, submissions and their feedback"},
                        "key_thoughts": {"type": "string", "description": "Core reasoning and strategy in 2-3 sentences"},
                        "key_findings": {"type": "string", "description": "Verified facts with docid references"},
                        "remaining_to_find": {"type": "string", "description": "Missing clues, what to search next"},
                    },
                    "required": ["tool_summary", "key_thoughts", "key_findings", "remaining_to_find"],
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
                                missing = [k for k in ("tool_summary","key_thoughts","key_findings","remaining_to_find") if not args.get(k)]
                                if missing:
                                    if attempt == 0:
                                        nudge = {"role": "user", "content": f"Missing required fields in submit_condensed_summary: {', '.join(missing)}. Please provide ALL required fields and try again."}
                                        condense_messages.append(nudge)
                                        session_msgs.append(nudge)
                                    continue  # retry
                                parts = []
                                if args.get('tool_summary'):
                                    parts.append(args['tool_summary'])
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
                    analysis = "(analysis unavailable -- model error)"
                elif not fallback_truncation:
                    raise

        # ── Assemble final summary ──
        if not analysis:
            analysis = "(progress summary unavailable)"
        summary_text = analysis

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
                "reason": "Your answer is a surrender statement. The answer EXISTS in the corpus -- do NOT give up. Try completely different search angles: use different keywords, inverse relations, or split compound queries into simpler single-entity searches.",
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
                            "suggestions=\"...\"). Do NOT call any other tool -- only give_feedback."
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
                "Do NOT search or get_document anymore -- ONLY give_feedback."
            ),
        })
        result = await _run_loop(verify_msgs, verify_max, 2, "retry")
        if result:
            self._verify_msgs = verify_msgs
            return result

        print(f"    [verify] all attempts exhausted, passing through", flush=True)
        self._verify_msgs = verify_msgs
        return {"is_correct": True, "reason": "Verification timed out -- answer accepted by default", "suggestions": ""}
