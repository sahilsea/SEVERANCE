"""Real LLM agent adapter: local Ollama ONLY.

COMPLIANCE: document/question content must never leave this machine. There is
NO cloud fallback (no NVIDIA NIM, no Groq) anywhere in this module, by design
-- not merely deprioritized. Do not reintroduce a cloud HTTP call here without
an explicit, deliberate decision to do so; this file previously called
NVIDIA NIM and Groq as fallbacks, which sent real document passage text to
third-party cloud APIs, before that was identified as a compliance violation
and removed.

NON-NEGOTIABLE DESIGN PRINCIPLES:
1. The Agent Protocol takes NO principal, NO grade, and NO compartments.
   An agent never learns who is asking — it only ever receives already-gated passages.
2. Structured output: The model returns citations as structured JSON, NEVER scraped
   or parsed from prose.
3. Every citation must be an exact verbatim substring copied directly from a passage.
4. If citation verification fails in the runner, the failure reason is fed back
   into the next draft attempt prompt.
5. Every model call in this file goes to the local Ollama server only.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import time
from typing import Callable, Optional, Sequence
import httpx
from contracts import Citation, Draft, Passage
from trust.network_monitor import ExternalConnectionBlocked, check_and_record


def _emit(emit: Optional[Callable[[dict], None]], event: dict) -> None:
    """No-op-safe emit helper: every caller (including the test suite's
    MockAgent, and any caller that doesn't pass emit at all) is unaffected
    when emit is None."""
    if emit is not None:
        emit(event)


_JSON_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def partial_json_string_field(raw: str, field: str) -> Optional[str]:
    """Decode the value of string field `field` from a JSON object that may
    still be mid-generation. Returns the decoded prefix available so far
    (stopping before any incomplete escape), or None if the field hasn't
    started yet. The prefix only ever grows as `raw` grows."""
    match = re.search(r'"%s"\s*:\s*"' % re.escape(field), raw)
    if not match:
        return None
    out = []
    i, n = match.end(), len(raw)
    while i < n:
        c = raw[i]
        if c == '"':
            break
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            break
        e = raw[i + 1]
        if e in _JSON_ESCAPES:
            out.append(_JSON_ESCAPES[e])
            i += 2
            continue
        if e != "u":
            out.append(e)
            i += 2
            continue
        if i + 6 > n:
            break
        try:
            code = int(raw[i + 2:i + 6], 16)
        except ValueError:
            break
        if 0xD800 <= code <= 0xDBFF:
            if i + 12 > n or raw[i + 6:i + 8] != "\\u":
                break
            try:
                low = int(raw[i + 8:i + 12], 16)
            except ValueError:
                break
            out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
            i += 12
            continue
        out.append(chr(code))
        i += 6
    return "".join(out)


def _stream_emitter(
    emit: Optional[Callable[[dict], None]], json_field: Optional[str] = None
) -> Optional[Callable[[str], None]]:
    """Build an on_token callback that forwards the model's live output to
    `emit` as {"stage": "stream", "status": "delta", "text": ...} events.
    With `json_field`, only that string field's decoded text is forwarded
    (the model is writing JSON; the user should see the answer, not braces)."""
    if emit is None:
        return None
    raw_parts: list[str] = []
    state = {"sent": 0}

    def on_token(piece: str) -> None:
        if json_field is None:
            emit({"stage": "stream", "status": "delta", "text": piece})
            return
        raw_parts.append(piece)
        text = partial_json_string_field("".join(raw_parts), json_field)
        if text is not None and len(text) > state["sent"]:
            emit({"stage": "stream", "status": "delta", "text": text[state["sent"]:]})
            state["sent"] = len(text)

    return on_token


def _is_vision_refusal(text: str) -> bool:
    """granite3.2-vision's trained non-answer is the bare word "unanswerable"
    (sometimes with punctuation); treat that, or an empty reply, as no answer."""
    cleaned = re.sub(r"[^a-z]", "", (text or "").lower())
    return cleaned in ("", "unanswerable", "unanswerablequestion", "notanswerable")


class RealLlmAgent:
    """Real LLM Agent: local Ollama only (air-gapped, no cloud calls)."""

    def __init__(
        self,
        ollama_api_base: Optional[str] = None,
        ollama_model: Optional[str] = None,
        ollama_fallback_model: Optional[str] = None,
    ):
        self.ollama_api_base = (ollama_api_base or os.getenv("OLLAMA_API_BASE", "http://localhost:11434")).rstrip("/")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "granite4.1:3b")
        self.ollama_vision_model = os.getenv("OLLAMA_VISION_MODEL", "granite3.2-vision")
        # Task-type model routing: a coding request is handled by a different,
        # code-specialized local model than a document/summary question --
        # not just a fallback-on-failure choice, a genuine per-task selection
        # made by classify_intent() before any drafting call happens. Adding
        # another specialized model later (e.g. a stronger reasoning model)
        # is a one-line addition here, not a redesign.
        self.ollama_code_model = os.getenv("OLLAMA_CODE_MODEL", "qwen2.5-coder:3b")
        # A second locally-installed text model tried, still fully locally,
        # only if the primary model produces zero citations. Different model
        # weights fail differently, so this is real added resilience without
        # any data leaving the machine. Set to "" to disable.
        # NOTE: llama3.2:latest was tried as primary and ruled out -- it hangs
        # (>120s, no response) on this app's real system+user prompt size with
        # format=json, even though it answers trivial prompts in under a
        # second. Kept only as a last-resort fallback, not primary.
        self.ollama_fallback_model = (
            ollama_fallback_model
            if ollama_fallback_model is not None
            else os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:latest")
        )
        # Set once Ollama fails within this agent's lifetime (one HTTP request's
        # worth of retries). Without this, a struggling/overloaded Ollama burns
        # its full timeout on EVERY retry attempt — e.g. 5 retries x 60s = up to
        # 5 minutes before the request ever responds. After the first failure we
        # know it's not going to suddenly recover mid-request, so skip it.
        self._ollama_unavailable = False
        # Only try to auto-launch the Ollama app once per request (this agent's
        # lifetime), regardless of how many retries hit Ollama.
        self._ollama_launch_attempted = False

    def classify_intent(self, question: str, recent_context: str = "") -> str:
        """Route a message BEFORE retrieval into one of five categories:

        - "content": a genuine request for information that lives in the
          document corpus. Proceed to retrieval + citation-grounded drafting
          as normal.
        - "capability": a meta-question about the ASSISTANT/SYSTEM itself
          (e.g. "what can you do for me", "what files do you have access to").
          These have no grounding passage anywhere -- no document describes
          the assistant's own capabilities -- so forcing them through RAG
          either abstains with a useless "no match" or, worse, invites the
          model to invent citations for something no document actually says.
          Answered separately with real, non-hallucinated data (see runner.py).
        - "code": a request to write, explain, debug, or review CODE (any
          programming language, scripts, queries, config/IaC). This has no
          grounding passage in the document corpus either -- code correctness
          comes from the model's own reasoning, not a verbatim quote -- so it
          is routed to a separate, code-specialized local model instead of
          the document-drafting model, and skips citation verification
          entirely (there is nothing to verify a citation against). This is
          the actual task-type model routing this app is built around: a
          coding request must never share a model or a citation-verification
          path with a document question.
        - "calculation": a straightforward arithmetic calculation or unit
          conversion (e.g. engineering unit conversions: pressure, volume
          incl. barrels, temperature, length, mass). Routed to a real,
          hand-verified deterministic tool (tools/calculator.py,
          tools/units.py) -- NEVER computed by the model itself, since a
          small local model gets arithmetic wrong often enough that this
          matters. A multi-step calculation that needs real logic (loops,
          conditionals, multiple intermediate values), not just one
          expression or one conversion, is "code" instead -- the sandbox
          actually running real code is more trustworthy there than forcing
          a multi-step calculation through a single tool call.
        - "other": greeting, small talk, filler, or test input with no real
          information need of any kind. Abstain with a plain, honest message.

        Lexical retrieval scores term overlap, not intent -- it can't make
        any of these five distinctions on its own. This needs an actual
        judgment call, which only a model can make.

        Fails OPEN (returns "content") on any classification failure or
        unrecognized output: blocking a genuine question is worse than
        answering an edge case, and retrieval's own relevance scoring remains
        the safety net for truly irrelevant input that slips through.
        """
        system_prompt = (
            "You classify a single user message for a corporate document search assistant. "
            "Respond STRICTLY in valid JSON: {\"category\": \"...\"} using exactly one of these "
            "five values:\n"
            "- \"content\": a genuine request for information that could be answered FROM A "
            "DOCUMENT -- even if short, informally phrased, or mixed in with unrelated chatter. "
            "This INCLUDES any question asking what to do, how to handle, or what the correct "
            "procedure is for a real-world situation or event (an emergency, an incident, a safety "
            "hazard, a compliance requirement, etc.) -- those are content questions about "
            "procedures, NOT questions about the assistant.\n"
            "- \"capability\": a question specifically about the ASSISTANT OR SYSTEM ITSELF -- what "
            "IT (the assistant) can do, what access or functionality IT has, or how IT works, "
            "including remarking on and asking about ITS OWN behavior just now (e.g. its speed, "
            "how it answered, why it did something). This is NEVER about what the USER should do "
            "in some real-world situation.\n"
            "- \"code\": a request to write, generate, explain, debug, fix, refactor, or review "
            "programming code, a script, a query (SQL etc.), or configuration/infrastructure code. "
            "This is about producing or reasoning about CODE, not about documents or the assistant.\n"
            "- \"calculation\": a single arithmetic calculation or unit conversion (pressure, "
            "volume, temperature, length, mass -- including petroleum barrels). NOT for anything "
            "needing multiple steps, loops, or real logic -- that is \"code\" instead.\n"
            "- \"other\": a greeting, small talk, filler text, or test input with no real "
            "information need at all.\n\n"
            "Disambiguating examples (note the difference):\n"
            "- \"what should I do during an emergency\" -> content (asks about a real-world "
            "procedure that a document may describe)\n"
            "- \"what can you do for me\" -> capability (asks about the assistant's own functions)\n"
            "- \"what files do you have access to\" -> capability (asks about the assistant's own "
            "document access, not document content)\n"
            "- \"how do I report a gas leak\" -> content (asks about a real-world procedure)\n"
            "- \"that was fast, how did you do that\" -> capability (remarking on and asking about "
            "the assistant's own speed/behavior, not a document topic)\n"
            "- \"why did you answer that way\" -> capability (asking about the assistant's own "
            "reasoning/behavior)\n"
            "- \"ok what can I do here then ?\" -> capability (casual filler words like 'ok'/'then'/"
            "'so' wrapped around the question do NOT change its category -- strip filler mentally "
            "and classify the real question underneath: 'what can I do here' asks about the "
            "assistant's own functions, same as 'what can you do for me')\n"
            "- \"write a python function to parse this log file\" -> code\n"
            "- \"why does my SQL query return duplicate rows\" -> code\n"
            "- \"fix the bug in this script\" -> code\n"
            "- \"what does this error message mean\" -> code (debugging code, not a document question)\n"
            "- \"what is 450 * 12.5\" -> calculation\n"
            "- \"convert 100 bar to psi\" -> calculation\n"
            "- \"how many barrels is 5000 liters\" -> calculation\n"
            "- \"what's the boiling point of water in fahrenheit\" -> calculation (212, a unit "
            "conversion of a known constant -- NOT a document lookup)\n"
            "- \"write code to convert a whole column of pressure readings from bar to psi\" -> code "
            "(this needs real logic over a dataset, not one single-value conversion)\n\n"
            "Casual conversational wrapping (\"ok\", \"so\", \"then\", \"well\", trailing punctuation) "
            "is never itself a signal -- always classify based on the actual question inside it.\n\n"
            "If a RECENT CONVERSATION is given below, use it ONLY to resolve pronouns/references "
            "('that', 'it', 'there') in the CURRENT message -- never let the PREVIOUS answer's own "
            "topic or tone pull your classification of the CURRENT message. In particular: a "
            "previous answer that reads as a list of procedural/action steps does NOT make the "
            "current message a procedural question too -- 'what can I do here' after ANY previous "
            "answer, including a safety-procedure answer, is still asking about the assistant's own "
            "functions (capability), not asking for more procedure steps (content). Classify the "
            "CURRENT message's own words first; only consult RECENT CONVERSATION to fill in what a "
            "pronoun refers to.\n\n"
            "When genuinely unsure between content and one of the others, answer \"content\"."
        )
        user_prompt = f"CURRENT message to classify: {question}\n\n"
        if recent_context:
            user_prompt += (
                f"RECENT CONVERSATION (reference only, for resolving pronouns in the CURRENT "
                f"message -- do not let its topic influence your classification):\n{recent_context}\n\n"
            )
        user_prompt += "Respond in JSON with a single field 'category', classifying the CURRENT message above."

        raw_response = None
        if not self._ollama_unavailable:
            try:
                # temperature=0.0: this runs exactly once per query with no
                # retry, unlike drafting -- greedy decoding measurably reduced
                # flip-flopping on borderline phrasing in live testing.
                raw_response = self._call_ollama(system_prompt, user_prompt, temperature=0.0, process_label="intent-classification")
            except Exception as e:
                print(f"[RealLlmAgent] Intent classification via Ollama failed ({e}).")

        if raw_response is None:
            return "content"

        try:
            clean = raw_response.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean)
            parsed = json.loads(clean)
            category = str(parsed.get("category", "content")).strip().lower()
            return category if category in ("content", "capability", "code", "calculation", "other") else "content"
        except Exception:
            return "content"

    def extract_tool_call(self, question: str, recent_context: str = "") -> Optional[dict]:
        """Extract a structured tool call for a "calculation" question --
        this is genuine tool-calling, not the model doing arithmetic: the
        model's ONLY job is to identify which deterministic tool applies and
        pull out its arguments; tools/calculator.py or tools/units.py then
        does the actual computation by hand-verified code, never the model's
        own math. Returns None if extraction fails or produces something
        that doesn't match either tool's expected shape -- the caller falls
        back to the code-generation+sandbox path rather than guessing.
        """
        system_prompt = (
            "You extract a structured tool call from a user's calculation or unit-conversion "
            "request. Respond STRICTLY in valid JSON matching exactly ONE of these two shapes:\n\n"
            "For a plain arithmetic calculation:\n"
            '{"tool": "calculator", "expression": "<a valid Python arithmetic expression using '
            'only numbers, + - * / // % ** and parentheses -- no words, no units>"}\n\n'
            "For a unit conversion:\n"
            '{"tool": "unit_converter", "value": <number>, "from_unit": "<code>", "to_unit": "<code>"}\n\n'
            "Valid unit codes -- use EXACTLY these short codes, not full words:\n"
            "- Pressure: pa, kpa, mpa, bar, psi, atm\n"
            "- Temperature: c, f, k\n"
            "- Volume: ml, l, m3, gal, bbl (bbl = petroleum barrel, the standard refinery volume "
            "unit -- 158.987 liters)\n"
            "- Length: mm, cm, m, km, in, ft, yd, mi\n"
            "- Mass: mg, g, kg, ton, lb, oz\n\n"
            "Examples:\n"
            '- "what is 450 * 12.5" -> {"tool": "calculator", "expression": "450 * 12.5"}\n'
            '- "convert 100 bar to psi" -> {"tool": "unit_converter", "value": 100, "from_unit": '
            '"bar", "to_unit": "psi"}\n'
            '- "how many barrels is 5000 liters" -> {"tool": "unit_converter", "value": 5000, '
            '"from_unit": "l", "to_unit": "bbl"}\n'
            '- "what\'s the boiling point of water in fahrenheit" -> {"tool": "unit_converter", '
            '"value": 100, "from_unit": "c", "to_unit": "f"}'
        )
        user_prompt = f"Request: {question}"
        if recent_context:
            user_prompt += f"\n\nRECENT CONVERSATION (for context on a follow-up only):\n{recent_context}"

        raw_response = None
        if not self._ollama_unavailable:
            try:
                raw_response = self._call_ollama(system_prompt, user_prompt, temperature=0.0, process_label="tool-call-extraction")
            except Exception as e:
                print(f"[RealLlmAgent] Tool-call extraction via Ollama failed ({e}).")

        if raw_response is None:
            return None
        try:
            clean = raw_response.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean)
            parsed = json.loads(clean)
        except Exception:
            return None

        if not isinstance(parsed, dict):
            return None
        tool = parsed.get("tool")
        if tool == "calculator" and isinstance(parsed.get("expression"), str):
            return {"tool": "calculator", "expression": parsed["expression"]}
        if (
            tool == "unit_converter"
            and isinstance(parsed.get("value"), (int, float))
            and isinstance(parsed.get("from_unit"), str)
            and isinstance(parsed.get("to_unit"), str)
        ):
            return {
                "tool": "unit_converter",
                "value": parsed["value"],
                "from_unit": parsed["from_unit"],
                "to_unit": parsed["to_unit"],
            }
        return None

    def draft_capability_answer(
        self,
        question: str,
        facts: str,
        recent_context: str = "",
        emit: Optional[Callable[[dict], None]] = None,
    ) -> Optional[str]:
        """Answer a meta-question about the assistant/system itself in natural
        language, using ONLY the given ground-truth facts.

        Unlike draft(), this needs no citation verification -- there's no
        document passage to quote when the question is about the assistant's
        own behavior, not document content. The caller (harness/runner.py)
        supplies `facts` (real capabilities, real document list, real reason
        this particular answer was fast) so the model has something true and
        specific to work from instead of a generic canned blurb, without any
        chance of it inventing a capability or document that doesn't exist.

        `recent_context` is accepted for interface consistency with draft()
        but deliberately NEVER placed in the prompt: tested live and found
        unsafe -- the local model would restate/extend prior turns' document
        content when it saw it here, and unlike draft() this path has NO
        citation verification to catch that. Conversation memory is safe for
        classify_intent (worst case: a misroute, which fails open into the
        citation-checked content path) and for draft() (any leaked claim
        still has to verify against real passages or gets rejected) -- not
        here, so it's intentionally not used.

        Returns None on any failure so the caller can fall back to a plain
        templated answer -- this is a natural-language polish step, not a
        verification-critical path.
        """
        system_prompt = (
            "You are SEVERANCE, a document question-answering assistant for MRPL. A user asked a "
            "question about how you work or what you can do, not about document content. Write a "
            "short (2-5 sentence), direct, honest answer to their SPECIFIC question, using ONLY "
            "the facts provided below. Do not invent any capability, mechanism, or document not "
            "listed in the facts. If the facts don't cover what they're asking, say so plainly "
            "rather than guessing. Respond STRICTLY in valid JSON: {\"answer\": \"...\"}."
        )
        user_prompt = f"FACTS:\n{facts}\n\nUser question: {question}\n\nRespond in JSON with a single field 'answer'."

        raw_response = None
        if not self._ollama_unavailable:
            _emit(emit, {"stage": "stream", "status": "start", "purpose": "capability", "model": self.ollama_model})
            try:
                raw_response = self._call_ollama(
                    system_prompt,
                    user_prompt,
                    process_label="capability-answer",
                    on_token=_stream_emitter(emit, json_field="answer"),
                )
            except Exception as e:
                print(f"[RealLlmAgent] Capability answer via Ollama failed ({e}).")

        if raw_response is None:
            return None

        try:
            clean = raw_response.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean)
            parsed = json.loads(clean)
            answer = parsed.get("answer")
            return str(answer) if answer else None
        except Exception:
            return None

    def draft_code(
        self,
        question: str,
        recent_context: str = "",
        feedback: Optional[str] = None,
        emit: Optional[Callable[[dict], None]] = None,
    ) -> Optional[str]:
        """Answer a coding request (write/explain/debug/review code) using
        the CODE-SPECIALIZED local model (self.ollama_code_model), never the
        document-drafting model -- this is the actual per-task-type model
        routing: a coding request is deliberately handled by a different
        model than a document question, chosen by classify_intent() before
        this is ever called.

        Returns plain Markdown (fenced code blocks, prose explanation), not
        the structured {answer, citations} shape draft() uses -- there is no
        document passage for generated code to be a verbatim substring of,
        so citation verification does not apply here (same reasoning as
        analyze_image() for vision output). The UI must render this as
        AI-generated code, not as a citation-verified document claim.

        `feedback`, if given, is the REAL stderr/error from actually running
        the previous attempt's code in tools/sandbox.py -- not a guess. This
        is what turns code generation into a genuine generate-run-fix loop
        (see harness/runner.py's code branch) instead of a one-shot reply.

        Returns None on any failure so the caller can abstain honestly
        instead of silently returning nothing.
        """
        system_prompt = (
            "You are a careful, precise coding assistant running fully offline on the user's own "
            "machine -- no code or question here ever leaves this machine. Write correct, working "
            "code for the user's request. Use fenced Markdown code blocks (```language ... ```) for "
            "all code, and brief prose only where it adds real value (a short explanation of "
            "non-obvious choices, not a restatement of what the code obviously does). If the request "
            "is ambiguous, make a reasonable assumption, state it briefly, and proceed -- don't just "
            "ask a clarifying question and stop. If PREVIOUS ATTEMPT FEEDBACK is given below, that is "
            "the real error from actually running your last attempt in a sandbox -- fix that exact "
            "problem, don't just rewrite the code differently."
        )
        user_prompt = ""
        if recent_context:
            user_prompt += f"RECENT CONVERSATION (for context on a follow-up request only):\n{recent_context}\n\n"
        user_prompt += f"Request: {question}"
        if feedback:
            user_prompt += f"\n\nPREVIOUS ATTEMPT FEEDBACK (real sandbox error, fix this exact issue):\n{feedback}"

        _emit(emit, {"stage": "model_call", "status": "start", "purpose": "code", "model": self.ollama_code_model})
        if self._ollama_unavailable:
            _emit(emit, {"stage": "model_call", "status": "error", "purpose": "code", "model": self.ollama_code_model, "message": "Ollama unavailable"})
            return None
        _emit(emit, {"stage": "stream", "status": "start", "purpose": "code", "model": self.ollama_code_model})
        try:
            raw_response = self._call_ollama(
                system_prompt,
                user_prompt,
                model=self.ollama_code_model,
                json_mode=False,
                num_predict=1500,
                timeout=120.0,
                process_label="code-generation",
                on_token=_stream_emitter(emit),
            )
        except Exception as e:
            if isinstance(e, httpx.TimeoutException):
                self._ollama_unavailable = True
            print(f"[RealLlmAgent] Code drafting via Ollama ({self.ollama_code_model}) failed ({e}).")
            _emit(emit, {"stage": "model_call", "status": "error", "purpose": "code", "model": self.ollama_code_model, "message": str(e)})
            return None

        answer = raw_response.strip()
        _emit(emit, {"stage": "model_call", "status": "done", "purpose": "code", "model": self.ollama_code_model})
        return answer or None

    def draft(
        self,
        question: str,
        passages: Sequence[Passage],
        feedback: Optional[str] = None,
        recent_context: str = "",
        emit: Optional[Callable[[dict], None]] = None,
    ) -> Draft:
        """Generate structured draft answer and verbatim citations from gated passages."""
        if not passages:
            return Draft(
                answer="No readable documents available to address this inquiry.",
                citations=[],
            )

        # Build prompt containing ONLY allowed passages
        passages_text = ""
        for p in passages:
            passages_text += f"\n--- Document: {p.doc_id} | Page: {p.page} ---\n{p.text}\n"

        system_prompt = (
            "You are SEVERANCE, the sovereign document intelligence synthesis agent for MRPL.\n"
            "Answer using strictly the provided readable passages.\n\n"
            "ANSWER RULES:\n"
            "1. Keep 'answer' concise: about 80-150 words, using concrete detail actually present in the\n"
            "   passages (named items, quantities, procedures, thresholds) — not restated question text.\n"
            "2. Don't repeat the same sentence pattern per item; each point should add a distinct fact.\n"
            "3. Write 'answer' as GitHub-Flavored Markdown ('##' headings, '-' bullets, '**bold**' for key terms).\n"
            "4. Only state facts the passages support. If they're thin or partial, say so briefly.\n\n"
            "CRITICAL CITATION RULES:\n"
            "5. The 'citations' array must NEVER be empty. Every answer, even a partial one, MUST include at\n"
            "   least one verbatim citation from a passage you examined. Never respond with 'citations': [].\n"
            "6. Every citation 'quote' must be an EXACT verbatim substring copied directly from the text —\n"
            "   never altered, summarized, or paraphrased.\n"
            "7. Quotes must be 20-300 characters and at least 5 words.\n"
            "8. Read the page number ONLY from the passage's own '--- Document: <doc_id> | Page: <page> ---'\n"
            "   header. Passage text often has its own numbering (clause/schedule numbers, list entries like\n"
            "   '25. Cause of leakage...') that looks like a page number but is NOT — e.g. under a header\n"
            "   'Page: 49' whose body contains '25. Cause of leakage...', the correct page is 49, never 25.\n"
            "9. Respond STRICTLY in valid JSON with this exact schema:\n"
            "{\n"
            '  "answer": "Concise, Markdown-formatted synthesis grounded in specific passage detail.",\n'
            '  "citations": [\n'
            '    {"doc_id": "document-id", "page": 1, "quote": "exact verbatim text copied from passage"}\n'
            "  ]\n"
            "}"
        )

        user_prompt = f"Available Readable Passages:\n{passages_text}\n\n"
        if recent_context:
            user_prompt += (
                f"RECENT CONVERSATION (for understanding a casual follow-up question only -- "
                f"NEVER a source for citations; every citation must still come from the Available "
                f"Readable Passages above):\n{recent_context}\n\n"
            )
        user_prompt += (
            f"User Question: {question}\n\n"
            "RESPONSE FORMAT REQUIREMENT:\n"
            "Respond STRICTLY in valid JSON with top-level keys 'answer' (a concise, ~80-150 word,\n"
            "Markdown-formatted answer built from specific details in the passages — not a repeated sentence\n"
            "template per item) and 'citations' (a list of verbatim\n"
            "quotation objects from the passages above). Do NOT output custom\n"
            "top-level keys.\n"
        )
        if feedback:
            user_prompt += (
                f"\nIMPORTANT PREVIOUS ATTEMPT FEEDBACK:\n{feedback}\n"
                "The previous citation failed verification. Make sure your quote is a literal, character-for-character "
                "verbatim copy of text in the specified document and page.\n"
            )

        # Local-only resilience strategy: for each locally-installed model in
        # sequence (primary, then a different fallback model if configured),
        # try a normal draft; if it comes back with zero citations, try one
        # narrower "citation repair" call against the SAME model before
        # moving on -- extracting a supporting verbatim quote for an already
        # -written answer is a much easier task for a small model than
        # composing the answer AND citing it in one shot, so this recovers
        # a real share of cases without ever leaving the machine. Only after
        # every local model (and its repair attempt) fails to produce a
        # citation do we give up. See module docstring: no cloud fallback.
        fallback_draft: Optional[Draft] = None
        tried_models = []
        for model_name in (self.ollama_model, self.ollama_fallback_model):
            if not model_name or model_name in tried_models:
                continue
            tried_models.append(model_name)

            if self._ollama_unavailable:
                break

            _emit(emit, {"stage": "model_call", "status": "start", "purpose": "draft", "model": model_name})
            _emit(emit, {"stage": "stream", "status": "start", "purpose": "draft", "model": model_name})
            try:
                raw_response = self._call_ollama(
                    system_prompt,
                    user_prompt,
                    model=model_name,
                    process_label="document-drafting",
                    on_token=_stream_emitter(emit, json_field="answer"),
                )
            except Exception as e:
                if isinstance(e, httpx.TimeoutException):
                    self._ollama_unavailable = True
                print(f"[RealLlmAgent] Ollama ({model_name}) call failed ({e}).")
                _emit(emit, {"stage": "model_call", "status": "error", "purpose": "draft", "model": model_name, "message": str(e)})
                continue

            draft = self._parse_draft(raw_response, passages)
            if draft.citations:
                _emit(emit, {"stage": "model_call", "status": "done", "purpose": "draft", "model": model_name, "citations": len(draft.citations)})
                return draft
            fallback_draft = fallback_draft or draft
            print(f"[RealLlmAgent] Ollama ({model_name}) returned zero citations. Trying citation repair.")
            _emit(emit, {"stage": "model_call", "status": "empty_citations", "purpose": "draft", "model": model_name})

            _emit(emit, {"stage": "model_call", "status": "start", "purpose": "citation_repair", "model": model_name})
            repaired = self._repair_citations(model_name, draft.answer, passages)
            if repaired:
                _emit(emit, {"stage": "model_call", "status": "done", "purpose": "citation_repair", "model": model_name, "citations": len(repaired)})
                return Draft(answer=draft.answer, citations=repaired)
            _emit(emit, {"stage": "model_call", "status": "empty_citations", "purpose": "citation_repair", "model": model_name})

        if fallback_draft is not None:
            # No local model produced a citation, but at least one responded --
            # return its draft so harness/verify.py's real "no citations were
            # provided" feedback flows into the next retry, instead of masking
            # a genuine drafting difficulty as an infrastructure failure.
            return fallback_draft

        # Do NOT return an empty Draft here: harness/verify.py would trivially
        # pass it as "verified" (nothing to check), recording this
        # infrastructure failure as a verified "answered" response. Raising
        # lets the retry loop treat it like any other failed attempt and
        # abstain honestly.
        raise RuntimeError("The local Ollama backend failed to respond.")

    def _repair_citations(
        self, model_name: str, answer: str, passages: Sequence[Passage]
    ) -> list[Citation]:
        """Narrow follow-up call: given an already-written answer, ask the
        SAME local model only to extract verbatim supporting quotes from the
        passages -- a simpler, more constrained task than composing the
        answer and citing it simultaneously, so a small model is more likely
        to get it right. Returns [] on any failure; never raises.
        """
        passages_text = ""
        for p in passages:
            passages_text += f"\n--- Document: {p.doc_id} | Page: {p.page} ---\n{p.text}\n"

        system_prompt = (
            "You are given an ANSWER and the PASSAGES it was based on. Your only job is to find "
            "1 to 3 short EXACT verbatim quotes from the passages that best support the answer. "
            "Each quote must be copied character-for-character from the passage text -- do not "
            "paraphrase, summarize, fix typos, or alter punctuation in any way. Each quote must be "
            "20-300 characters and at least 5 words. Respond STRICTLY in valid JSON:\n"
            "{\n"
            '  "citations": [\n'
            '    {"doc_id": "document-id", "page": 1, "quote": "exact verbatim text copied from passage"}\n'
            "  ]\n"
            "}"
        )
        user_prompt = (
            f"PASSAGES:\n{passages_text}\n\nANSWER:\n{answer}\n\n"
            "Respond in JSON with a single field 'citations' (a list of 1-3 verbatim quotation "
            "objects copied from the passages above)."
        )

        try:
            raw_response = self._call_ollama(system_prompt, user_prompt, model=model_name, process_label="citation-repair")
        except Exception as e:
            print(f"[RealLlmAgent] Citation repair via Ollama ({model_name}) failed ({e}).")
            return []

        try:
            clean = raw_response.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean)
            parsed = json.loads(clean)
            raw_citations = parsed.get("citations", []) if isinstance(parsed, dict) else []
        except Exception:
            return []

        return self._parse_citations(raw_citations)

    def _ensure_ollama_running(self) -> None:
        """Best-effort: if the local Ollama server isn't reachable, try to launch
        the Ollama app (macOS) and give it a few seconds to come up.

        Quitting the Ollama app takes the primary backend offline until someone
        notices and manually relaunches it; this closes that gap automatically
        for the common case (Ollama.app quit, server not running). Only
        attempted once per request, and it never blocks longer than the poll
        window below -- if it can't recover in time, the caller's normal
        request just proceeds to fail this attempt (there is no cloud
        fallback in this module -- see module docstring).
        """
        if self._ollama_launch_attempted:
            return
        self._ollama_launch_attempted = True

        health_url = f"{self.ollama_api_base}/api/tags"
        try:
            check_and_record("ollama-health-check", health_url)
        except ExternalConnectionBlocked:
            return  # never reachable if OLLAMA_API_BASE were misconfigured to a non-loopback host
        try:
            with httpx.Client(timeout=2.0) as client:
                client.get(health_url)
            return  # already up
        except httpx.HTTPError:
            pass

        if platform.system() != "Darwin":
            return  # only know how to auto-launch the macOS app

        try:
            subprocess.Popen(
                ["open", "-a", "Ollama"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[RealLlmAgent] Could not launch Ollama app ({e}).")
            return

        print("[RealLlmAgent] Ollama server unreachable -- launched the Ollama app, waiting for it to come up...")
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                with httpx.Client(timeout=1.5) as client:
                    client.get(health_url)
                print("[RealLlmAgent] Ollama server is back up.")
                return
            except httpx.HTTPError:
                time.sleep(1.0)
        print("[RealLlmAgent] Ollama app launched but server did not come up within 8s; proceeding to fallbacks for this request.")

    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        json_mode: bool = True,
        num_predict: int = 700,
        timeout: float = 60.0,
        process_label: str = "ollama-inference",
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Call local Ollama endpoint with the given model (defaults to self.ollama_model).

        json_mode=False is for callers that want free-form markdown back
        (e.g. generated code) instead of the structured {answer, citations}
        JSON the drafting/classification paths use -- forcing JSON mode onto
        a code response means escaping every newline and quote inside a
        JSON string, which is fragile for a small model and buys nothing
        here since code answers aren't citation-verified anyway.

        `process_label` identifies WHICH task this call is for (intent
        classification, document drafting, code generation, ...) in
        trust/network_monitor.py's real connection log -- purely a labeling
        detail for the Sovereignty Monitor UI, not a security boundary; the
        boundary is check_and_record() itself, called unconditionally below
        regardless of what label was passed.
        """
        self._ensure_ollama_running()
        url = f"{self.ollama_api_base}/api/chat"
        payload = {
            "model": model or self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            # num_predict caps generation length so a verbose or looping
            # response can't run out the full request timeout.
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        if json_mode:
            payload["format"] = "json"
        # Real enforcement, not just a log: raises and never opens the
        # socket below if `url` isn't loopback. See trust/network_monitor.py.
        check_and_record(process_label, url)
        if on_token is None:
            with httpx.Client(timeout=timeout) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                return data["message"]["content"]

        # Streaming: Ollama sends one JSON line per generated chunk; each is
        # passed to on_token as it arrives, and the full text is returned
        # exactly as the non-streaming path would.
        payload["stream"] = True
        parts: list[str] = []
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, json=payload) as res:
                res.raise_for_status()
                for line in res.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(f"Ollama error: {chunk['error']}")
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        parts.append(piece)
                        on_token(piece)
                    if chunk.get("done"):
                        break
        return "".join(parts)

    def analyze_image(self, question: str, image_base64: str) -> str:
        """Ask the local vision model (granite3.2-vision by default) about ONE
        image, returning plain natural-language text.

        This deliberately returns plain text, NOT the structured {answer,
        citations} JSON draft() uses -- there is no verbatim substring of an
        image to verify a "citation" against, so harness/runner.py labels this
        output as an unverified visual observation rather than routing it
        through harness/verify.py at all.

        PROMPTING: granite3.2-vision was trained on document-VQA data where
        "unanswerable" is a standard answer, and it gives exactly that single
        word to most question-style or "read the text" prompts -- tested live
        on a photographed handwritten notebook page. Prompts that open with
        "Describe this image in detail" reliably get a real description, so
        the user's question is folded into that form, with a plain-description
        retry if the model still refuses.
        """
        focused = f"Describe this image in detail, focusing on: {question.strip()}"
        answer = self._vision_call(focused, image_base64)
        if not _is_vision_refusal(answer):
            return answer
        general = self._vision_call("Describe this image in detail.", image_base64)
        if not _is_vision_refusal(general):
            return (
                "*(The vision model could not answer the question directly, so this is its general "
                "description of the image.)*\n\n" + general
            )
        return (
            "The on-device vision model could not interpret this image. Clear, well-lit photos of "
            "printed text, labels, or diagrams work best; dense or messy handwriting is often beyond "
            "what this small local model can read."
        )

    def _vision_call(self, prompt: str, image_base64: str) -> str:
        self._ensure_ollama_running()
        url = f"{self.ollama_api_base}/api/chat"
        payload = {
            "model": self.ollama_vision_model,
            "messages": [{"role": "user", "content": prompt, "images": [image_base64]}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 600},
        }
        # On an 8 GB machine a full-size photo took 40-100s in live testing
        # (model swap-in plus image encoding), hence the long timeout.
        check_and_record("vision-analysis", url)
        with httpx.Client(timeout=180.0) as client:
            res = client.post(url, json=payload)
            try:
                res.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise httpx.HTTPStatusError(
                    f"{exc}. Response body: {res.text[:500]}",
                    request=exc.request,
                    response=exc.response,
                ) from exc
            data = res.json()
            return data["message"]["content"].strip()

    def _parse_draft(self, raw_content: str, passages: Sequence[Passage]) -> Draft:
        """Safely parse and validate structured Draft JSON."""
        clean_content = raw_content.strip()
        # Strip markdown code blocks if present
        if clean_content.startswith("```"):
            clean_content = re.sub(r"^```(?:json)?\n?", "", clean_content)
            clean_content = re.sub(r"\n?```$", "", clean_content)

        try:
            parsed = json.loads(clean_content)
        except Exception:
            # Attempt regex extraction of JSON object
            match = re.search(r"\{.*\}", clean_content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                return Draft(answer=clean_content, citations=[])

        if not isinstance(parsed, dict):
            return Draft(answer=clean_content, citations=[])

        if "answer" in parsed:
            answer = str(parsed["answer"])
        else:
            # If the model returned keys like {"possible_dangers": [...]}:
            # Intelligently convert into clean, human-readable text instead of dumping raw JSON syntax
            lines = []
            for k, v in parsed.items():
                if k == "citations":
                    continue
                title = k.replace("_", " ").title()
                if isinstance(v, list):
                    lines.append(f"{title}:")
                    for item in v:
                        lines.append(f"• {str(item).replace('_', ' ').capitalize()}")
                elif isinstance(v, dict):
                    lines.append(f"{title}:")
                    for sub_k, sub_v in v.items():
                        lines.append(f"• {sub_k.replace('_', ' ').title()}: {sub_v}")
                else:
                    lines.append(f"{title}: {v}")
            answer = "\n".join(lines) if lines else clean_content
        citations = self._parse_citations(parsed.get("citations", []))
        return Draft(answer=answer, citations=citations)

    def _parse_citations(self, raw_citations) -> list[Citation]:
        """Validate a raw parsed-JSON citations list into Citation objects,
        dropping any entry that doesn't fit the expected shape."""
        citations: list[Citation] = []
        if isinstance(raw_citations, list):
            for c in raw_citations:
                if not isinstance(c, dict):
                    continue
                try:
                    doc_id = str(c.get("doc_id", "")).strip()
                    page = int(c.get("page", 1))
                    quote = str(c.get("quote", "")).strip()
                    citations.append(Citation(doc_id=doc_id, page=page, quote=quote))
                except Exception:
                    # Invalid citation dropped
                    pass
        return citations
