"""Bounded verification loop and execution harness.

NON-NEGOTIABLE DESIGN PRINCIPLES:
1. The retry loop is a plain Python for loop whose exit condition is
   deterministic verification.
2. Abstention: If nothing readable matched, DO NOT CALL THE MODEL AT ALL.
   A model with no sources writes fluent hallucinations.
3. Citation verification is pure Python string matching:
   - Check structured output schema.
   - Exact verbatim substring in claimed doc_id and page.
   - If verification fails, retry with the specific failure fed back into the prompt.
4. After hard cap (max_retries), ABSTAIN. Never return an unverified answer.
5. Classification inheritance: An answer inherits the highest tier and union of
   all compartments of EVERY passage placed into the prompt (not just cited ones).
6. Every query outcome (answered or abstained) writes to the hash-chained ledger.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence
from contracts import (
    AskRequest,
    AskResponse,
    Citation,
    Denial,
    Label,
    Passage,
    Principal,
    Tier,
)
from agents.base import Agent
from harness.retrieve import retrieve
from harness.verify import check as verify_check
from ingest.ephemeral import EphemeralUpload
from trust.conversations import append_turn, conversation_owner, create_conversation, get_recent_context
from trust.labels import can_read, inherit_label
from trust.ledger import log as ledger_log
from trust.reports import save_report_data
from tools.calculator import CalculatorError, calculate
from tools.sandbox import extract_code_block, run_python
from tools.units import UnitError, convert_unit

DEFAULT_MAX_RETRIES = 3
CODE_MAX_ATTEMPTS = 3


def _network_destinations(sandbox_result: dict) -> list[str]:
    seen = []
    for a in sandbox_result.get("network_attempts", []):
        dest = f"{a['host']}:{a['port']}" if a["port"] else a["host"]
        if dest not in seen:
            seen.append(dest)
    return seen


def _format_number(value: float) -> str:
    """Render a float without ugly trailing binary-float noise
    (158.987294928 not 158.98729492800001), while still showing real
    precision -- round to 6 decimal places, then strip trailing zeros."""
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _build_capability_facts(corpus: Sequence[Passage], principal: Principal) -> tuple[str, list[str], list[str]]:
    """Ground-truth facts about this assistant's real behavior and the
    caller's real document access -- the ONLY material either the templated
    fallback or the LLM polish step (draft_capability_answer) may draw on.

    Meta-questions about the assistant itself (e.g. "what can you do", "how
    did you answer that so fast") have no grounding passage anywhere -- no
    document describes the assistant's own capabilities -- so these facts
    come from real corpus/clearance data and real pipeline behavior, never
    from the drafting model guessing.
    """
    seen: dict[str, Passage] = {}
    for p in corpus:
        seen.setdefault(p.doc_id, p)

    readable: list[str] = []
    locked: list[str] = []
    for doc_id, p in seen.items():
        title = p.title or doc_id
        tier = p.label.tier.value
        if can_read(principal, p.label):
            readable.append(f"- **{title}** (`{doc_id}`, {tier})")
        else:
            locked.append(f"- {title} (`{doc_id}`, {tier} — requires additional clearance)")

    facts_lines = [
        "- This assistant answers questions about MRPL's internal documents only, and every "
        "content claim it makes must cite a verbatim passage from a document the caller is "
        "cleared to read.",
        "- It cannot hold a general conversation, perform tasks outside the document corpus, or "
        "answer from memory without a citation.",
        "- Every incoming message is first classified as one of: a real document question, a "
        "meta-question about the assistant/system itself, or small talk/filler with no "
        "information need.",
        "- Meta-questions about the assistant (like this one) skip document retrieval and the "
        "citation-verification loop entirely -- that's the ENTIRE reason they return faster than "
        "a real document question, which must search the corpus, draft an answer, and verify "
        "every citation before responding.",
        f"- Readable documents for this caller ({len(readable)}):",
    ]
    facts_lines.extend(f"  {line}" for line in (readable or ["  (none)"]))
    if locked:
        facts_lines.append(f"- Documents that exist but this caller cannot read ({len(locked)}):")
        facts_lines.extend(f"  {line}" for line in locked)

    return "\n".join(facts_lines), readable, locked


def _build_capability_fallback_answer(readable: list[str], locked: list[str]) -> str:
    """Plain templated capability answer, used when the LLM polish step
    (draft_capability_answer) is unavailable or fails."""
    lines = [
        "## What I Can Do",
        "I answer questions about MRPL's internal documents, grounding every claim in a "
        "verbatim, machine-verified citation from a passage you're cleared to read. I don't "
        "hold a conversation, perform tasks outside the document corpus, or answer from memory "
        "without a citation.",
        "",
        "## Documents You Can Currently Read",
    ]
    if readable:
        lines.extend(readable)
    else:
        lines.append("None of the documents in the corpus are currently readable under your active clearance.")

    if locked:
        lines.append("")
        lines.append("## Documents That Exist But You Cannot Read")
        lines.extend(locked)

    lines.append("")
    lines.append(
        "Ask me something specific about one of the readable documents above, or about safety "
        "procedures, RTI disclosures, or compliance records."
    )
    return "\n".join(lines)


def run_query(
    request: AskRequest,
    corpus: Sequence[Passage],
    principal: Principal,
    agent: Agent,
    db_path: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    emit: Optional[Callable[[dict], None]] = None,
) -> AskResponse:
    """Execute the complete SEVERANCE query lifecycle.

    1. Retrieve and two-axis gate candidates (discarding denied text).
    2. Check for immediate abstention (if 0 readable passages, model is NOT called).
    3. Execute bounded drafting and citation verification loop.
    4. Compute classification inheritance over all prompt passages.
    5. Append tamper-evident audit record to SQLite ledger.

    `emit`, if given, is called with a small dict at each REAL stage boundary
    (not a simulated/timed guess) -- e.g. {"stage": "drafting", "status":
    "start", "attempt": 2}. This is what api/routes/ask.py's streaming
    endpoint uses to give the UI a live log that reflects actual backend
    progress instead of a client-side animation. Optional and side-effect-only:
    omitting it (the default) changes nothing about this function's behavior,
    so every existing caller (including the test suite) is unaffected.
    """
    def _emit(event: dict) -> None:
        if emit is not None:
            emit(event)

    question = request.question.strip()

    # Resolve (or start) a persisted conversation for this query. A supplied
    # conversation_id that doesn't exist or isn't owned by this principal is
    # never reused -- silently starting a fresh one instead avoids leaking
    # whether some other user's conversation_id exists at all.
    conversation_id = request.conversation_id
    if conversation_id and conversation_owner(db_path, conversation_id) != principal.person_id:
        conversation_id = None
    if not conversation_id:
        conversation_id = create_conversation(db_path, principal.person_id, question)

    # Short conversation memory: only the last 1-2 turns of THIS conversation
    # (see trust/conversations.py), used purely to interpret casual follow-up
    # phrasing ("what can I do here THEN", referring to what was just
    # discussed). Never a source of facts or citations -- every citation
    # still has to verify against freshly retrieved passages exactly as
    # before. Kept deliberately short: more turns means more prompt text the
    # local model re-reads on every call, and this pipeline is already
    # latency-sensitive.
    recent_context = get_recent_context(db_path, conversation_id)

    # Step 0: Intent routing gate (optional -- only agents that implement it,
    # currently RealLlmAgent, get this check; MockAgent is unaffected).
    # Lexical retrieval scores term overlap, not intent: rambling or filler text
    # that happens to contain a real corpus keyword (e.g. "...this world is
    # full of gas leak") will legitimately match real passages and produce a
    # real, correctly-cited answer even though it wasn't a genuine question --
    # and a meta-question about the assistant itself ("what can you do",
    # "what files do you have access to") has no grounding passage anywhere,
    # so forcing it through RAG either abstains uselessly or invites invented
    # citations. This routes each of the three cases correctly BEFORE spending
    # a retrieval + drafting cycle on it.
    classify_intent = getattr(agent, "classify_intent", None)
    # Real model identity, when the agent exposes one (RealLlmAgent does;
    # MockAgent doesn't), so the live log can show which model is actually
    # running instead of a generic "the model" -- see ui/app.html.
    active_model = getattr(agent, "ollama_model", None)
    if classify_intent is not None:
        _emit({"stage": "intent", "status": "start", "model": active_model})
        try:
            category = classify_intent(question, recent_context)
        except Exception:
            category = "content"  # fail open
        _emit({"stage": "intent", "status": "done", "category": category, "model": active_model})

        if category == "other":
            answer = (
                "That doesn't read as a specific question. Please ask something concrete about "
                "MRPL's documents, safety procedures, or compliance records, and I'll search the "
                "corpus for a grounded, citation-backed answer."
            )
            effective_label = Label(tier=Tier.PUBLIC, compartments=frozenset())
            ledger_entry = ledger_log(
                db_path=db_path,
                actor=principal.person_id,
                action="ASK_ABSTAINED",
                details={"question": question, "reason": "not_a_genuine_question"},
            )
            response = AskResponse(
                status="abstained",
                answer=answer,
                citations=[],
                denials=[],
                effective_label=effective_label,
                ledger_row_id=ledger_entry.row_id,
                conversation_id=conversation_id,
            )
            append_turn(db_path, conversation_id, question, response)
            return response

        if category == "capability":
            _emit({"stage": "capability", "status": "start"})
            facts, readable, locked = _build_capability_facts(corpus, principal)
            answer = None
            draft_capability = getattr(agent, "draft_capability_answer", None)
            if draft_capability is not None:
                try:
                    # NOT passed recent_context: tested and found unsafe -- the
                    # local model would restate/extend prior turns' document
                    # content here, and this path has NO citation verification
                    # (see agents/real.py's draft_capability_answer docstring).
                    answer = draft_capability(question, facts, emit=emit)
                except Exception:
                    answer = None
            if not answer:
                answer = _build_capability_fallback_answer(readable, locked)
            _emit({"stage": "capability", "status": "done"})
            effective_label = Label(tier=Tier.PUBLIC, compartments=frozenset())
            ledger_entry = ledger_log(
                db_path=db_path,
                actor=principal.person_id,
                action="ASK_ANSWERED",
                details={
                    "question": question,
                    "category": "capability",
                    "answer_preview": answer[:100],
                },
            )
            response = AskResponse(
                status="answered",
                answer=answer,
                citations=[],
                denials=[],
                effective_label=effective_label,
                ledger_row_id=ledger_entry.row_id,
                conversation_id=conversation_id,
            )
            save_report_data(
                db_path=db_path,
                ledger_row_id=ledger_entry.row_id,
                person_id=principal.person_id,
                question=question,
                response=response,
            )
            append_turn(db_path, conversation_id, question, response)
            return response

        if category == "calculation":
            # Genuine tool-calling: the model's only job is to identify
            # WHICH tool applies and extract its arguments -- the actual
            # arithmetic/conversion is done by tools/calculator.py or
            # tools/units.py, hand-verified deterministic code, never the
            # model's own math. See agents/real.py::extract_tool_call.
            _emit({"stage": "calculation", "status": "start"})
            extract_tool_call = getattr(agent, "extract_tool_call", None)
            tool_call = None
            if extract_tool_call is not None:
                try:
                    tool_call = extract_tool_call(question, recent_context)
                except Exception as exc:
                    print(f"[runner] Tool-call extraction raised an error: {exc}")
                    tool_call = None

            tool_result_text: Optional[str] = None
            tool_error: Optional[str] = None
            tool_name: Optional[str] = None
            if tool_call is not None:
                tool_name = tool_call["tool"]
                try:
                    if tool_call["tool"] == "calculator":
                        value = calculate(tool_call["expression"])
                        tool_result_text = f"**{tool_call['expression']} = {_format_number(value)}**"
                    else:
                        value = convert_unit(tool_call["value"], tool_call["from_unit"], tool_call["to_unit"])
                        tool_result_text = (
                            f"**{_format_number(tool_call['value'])} {tool_call['from_unit']} "
                            f"= {_format_number(value)} {tool_call['to_unit']}**"
                        )
                except (CalculatorError, UnitError) as exc:
                    tool_error = str(exc)

            if tool_result_text is not None:
                _emit({"stage": "calculation", "status": "done", "tool": tool_name})
                answer = (
                    f"{tool_result_text}\n\n"
                    "*Computed by the local calculator/unit-conversion tool (a deterministic "
                    "function call, not model arithmetic).*"
                )
                effective_label = Label(tier=Tier.PUBLIC, compartments=frozenset())
                ledger_entry = ledger_log(
                    db_path=db_path,
                    actor=principal.person_id,
                    action="ASK_ANSWERED",
                    details={
                        "question": question,
                        "category": "calculation",
                        "tool": tool_name,
                        "answer_preview": answer[:100],
                    },
                )
                response = AskResponse(
                    status="answered",
                    answer=answer,
                    citations=[],
                    denials=[],
                    effective_label=effective_label,
                    ledger_row_id=ledger_entry.row_id,
                    conversation_id=conversation_id,
                )
                save_report_data(
                    db_path=db_path,
                    ledger_row_id=ledger_entry.row_id,
                    person_id=principal.person_id,
                    question=question,
                    response=response,
                )
                append_turn(db_path, conversation_id, question, response)
                return response

            # Extraction failed, or the tool itself rejected the input (e.g.
            # mismatched unit families) -- fall through to the code-
            # generation+sandbox path below rather than a hard abstain: a
            # real Python calculation can very likely still answer this,
            # and that path already has its own honest failure reporting.
            _emit({"stage": "calculation", "status": "fallback", "reason": tool_error or "could not identify a tool call"})
            category = "code"

        if category == "code":
            # Task-type model routing: a coding request is handled by a
            # separate, code-specialized local model (never the document-
            # drafting model), and has no citation-verification step -- there
            # is no document passage for generated code to be a verbatim
            # substring of. See agents/real.py::draft_code's docstring.
            #
            # This is a real generate -> RUN -> fix -> retry loop, not a
            # one-shot reply: if the generated code includes a Python block,
            # it is actually executed in tools/sandbox.py, and a real
            # failure (the real stderr, not a guess) is fed back into the
            # next attempt, up to CODE_MAX_ATTEMPTS times.
            draft_code = getattr(agent, "draft_code", None)
            feedback: Optional[str] = None
            final_answer: Optional[str] = None
            sandbox_result: Optional[dict] = None
            sandbox_attempts_used = 0

            for attempt in range(1, CODE_MAX_ATTEMPTS + 1):
                _emit({"stage": "code", "status": "start", "attempt": attempt, "max_attempts": CODE_MAX_ATTEMPTS})
                answer = None
                if draft_code is not None:
                    try:
                        answer = draft_code(question, recent_context, feedback=feedback, emit=emit)
                    except Exception as exc:
                        print(f"[runner] Code drafting raised an error: {exc}")
                        answer = None
                _emit({"stage": "code", "status": "done" if answer else "error", "attempt": attempt})

                if not answer:
                    # No response at all from the code model this attempt --
                    # an infra hiccup, not a code bug. Retry rather than
                    # treating it the same as a real execution failure.
                    feedback = "The code model did not respond last attempt. Please try again."
                    continue

                final_answer = answer
                code_block = extract_code_block(answer, "python")
                if code_block is None:
                    # Nothing executable was found (not Python, or a
                    # prose-only answer) -- there is no sandbox claim to
                    # make either way, so this is a legitimate final answer.
                    _emit({"stage": "sandbox", "status": "skipped", "attempt": attempt})
                    sandbox_result = None
                    break

                sandbox_attempts_used = attempt
                _emit({"stage": "sandbox", "status": "start", "attempt": attempt})
                sandbox_result = run_python(code_block)
                if sandbox_result["success"]:
                    _emit({"stage": "sandbox", "status": "pass", "attempt": attempt})
                    break
                if sandbox_result["network_attempts"]:
                    # A blocked network call can't be "fixed" by another
                    # attempt -- and feeding it back invites the model to
                    # hardcode made-up data to get a pass. Stop and report.
                    _emit({
                        "stage": "sandbox",
                        "status": "network_blocked",
                        "attempt": attempt,
                        "destinations": _network_destinations(sandbox_result),
                    })
                    break
                _emit({"stage": "sandbox", "status": "fail", "attempt": attempt, "stderr": sandbox_result["stderr"][:300]})
                feedback = (
                    f"Running this code in the sandbox failed (exit code {sandbox_result['returncode']}).\n"
                    f"STDOUT:\n{sandbox_result['stdout'] or '(empty)'}\n"
                    f"STDERR:\n{sandbox_result['stderr'] or '(empty)'}"
                )

            effective_label = Label(tier=Tier.PUBLIC, compartments=frozenset())

            if final_answer is None:
                # The code model never produced a response across every
                # attempt -- a genuine infra failure, not "code that doesn't
                # work." Abstain honestly, same principle as every other
                # failure path in this file.
                ledger_entry = ledger_log(
                    db_path=db_path,
                    actor=principal.person_id,
                    action="ASK_ABSTAINED",
                    details={"question": question, "reason": "code_drafting_failed"},
                )
                response = AskResponse(
                    status="abstained",
                    answer="Response withheld: the local code model was unable to respond.",
                    citations=[],
                    denials=[],
                    effective_label=effective_label,
                    ledger_row_id=ledger_entry.row_id,
                    conversation_id=conversation_id,
                )
                append_turn(db_path, conversation_id, question, response)
                return response

            # Always show the actual final state -- including code that
            # still fails after every retry -- rather than silently hiding
            # a failure. The point of running it is to report the truth,
            # not to only ever show success.
            if sandbox_result is None:
                answer_text = final_answer
            elif sandbox_result["success"]:
                answer_text = (
                    f"{final_answer}\n\n"
                    f"## Sandbox Execution — Passed (attempt {sandbox_attempts_used} of {CODE_MAX_ATTEMPTS})\n\n"
                    f"```\n{sandbox_result['stdout'] or '(no output)'}\n```"
                )
            elif sandbox_result["network_attempts"]:
                destinations = ", ".join(f"`{d}`" for d in _network_destinations(sandbox_result))
                answer_text = (
                    f"{final_answer}\n\n"
                    f"## Sandbox Execution — Blocked: Network Access Attempted\n\n"
                    f"The generated code tried to connect to {destinations}. This workbench is air-gapped, "
                    f"so the connection was stopped before any data left the machine, and the attempt is "
                    f"recorded in the Sovereignty Monitor. The code was not retried, since no rewrite can "
                    f"make external data available offline.\n\n"
                    f"```\n{sandbox_result['stderr'] or '(no error output)'}\n```"
                )
            else:
                status_note = "timed out" if sandbox_result["timed_out"] else f"exit code {sandbox_result['returncode']}"
                answer_text = (
                    f"{final_answer}\n\n"
                    f"## Sandbox Execution — Failed after {sandbox_attempts_used} attempt(s) ({status_note})\n\n"
                    f"```\n{sandbox_result['stderr'] or '(no error output)'}\n```"
                )

            ledger_entry = ledger_log(
                db_path=db_path,
                actor=principal.person_id,
                action="ASK_ANSWERED",
                details={
                    "question": question,
                    "category": "code",
                    "sandbox_ran": sandbox_result is not None,
                    "sandbox_passed": bool(sandbox_result and sandbox_result["success"]),
                    "sandbox_network_blocked": bool(sandbox_result and sandbox_result["network_attempts"]),
                    "answer_preview": answer_text[:100],
                },
            )
            response = AskResponse(
                status="answered",
                answer=answer_text,
                citations=[],
                denials=[],
                effective_label=effective_label,
                ledger_row_id=ledger_entry.row_id,
                conversation_id=conversation_id,
            )
            save_report_data(
                db_path=db_path,
                ledger_row_id=ledger_entry.row_id,
                person_id=principal.person_id,
                question=question,
                response=response,
            )
            append_turn(db_path, conversation_id, question, response)
            return response

    # Step 1: Two-pass retrieval and gating
    _emit({"stage": "retrieval", "status": "start"})
    allowed_passages, denials = retrieve(
        query=question,
        corpus=corpus,
        principal=principal,
        top_k=request.top_k,
    )
    _emit({
        "stage": "retrieval",
        "status": "done",
        "passages_found": len(allowed_passages),
        "denials_found": len(denials),
    })

    # Step 2: Immediate Abstention Check
    if not allowed_passages:
        if denials:
            withheld_docs = ", ".join(f"'{d.doc_id}'" for d in denials)
            answer = (
                f"Access denied: The requested information requires clearance for {withheld_docs}, "
                "which is withheld under the two-axis security gate. No readable passages matched your query."
            )
        else:
            answer = "No relevant documents found matching your query."

        effective_label = Label(tier=Tier.PUBLIC, compartments=frozenset())
        ledger_entry = ledger_log(
            db_path=db_path,
            actor=principal.person_id,
            action="ASK_ABSTAINED",
            details={
                "question": question,
                "reason": "zero_readable_passages",
                "denials": [d.model_dump() for d in denials],
            },
        )
        response = AskResponse(
            status="abstained",
            answer=answer,
            citations=[],
            denials=denials,
            effective_label=effective_label,
            ledger_row_id=ledger_entry.row_id,
            conversation_id=conversation_id,
        )
        append_turn(db_path, conversation_id, question, response)
        return response

    # Step 3: Bounded Drafting & Verification Loop
    feedback: Optional[str] = None
    successful_draft = None

    for attempt in range(1, max_retries + 1):
        _emit({"stage": "drafting", "status": "start", "attempt": attempt, "max_attempts": max_retries})
        try:
            draft = agent.draft(
                question=question,
                passages=allowed_passages,
                feedback=feedback,
                recent_context=recent_context,
                emit=emit,
            )
        except Exception as exc:
            # Infrastructure failure (e.g. both LLM backends unreachable): treat
            # like a failed verification attempt so the loop can retry, and so
            # exhaustion falls through to a clean ABSTAIN instead of a raw 500.
            _emit({"stage": "drafting", "status": "error", "attempt": attempt, "message": str(exc)})
            feedback = f"Attempt {attempt} failed: drafting agent raised an error ({exc})."
            continue
        _emit({"stage": "drafting", "status": "done", "attempt": attempt, "citations_claimed": len(draft.citations)})

        # Pure Python string verification
        _emit({"stage": "verification", "status": "start", "attempt": attempt})
        failure = verify_check(draft.citations, allowed_passages)
        if failure is None:
            # Verification passed completely
            _emit({"stage": "verification", "status": "pass", "attempt": attempt})
            successful_draft = draft
            break

        _emit({"stage": "verification", "status": "fail", "attempt": attempt, "reason": failure})
        # Feed the precise failure reason back into the next attempt
        avail_str = ", ".join(f"doc_id='{p.doc_id}' page {p.page}" for p in allowed_passages)
        feedback = f"Attempt {attempt} verification failed: {failure}. Note: The ONLY readable passages in context are [{avail_str}]. Ensure all quotes are exact verbatim substrings and page numbers match these exactly."

    # Step 4: Outcome Resolution & Classification Inheritance
    if successful_draft is not None:
        # Verified Answer: inherit classification from ALL passages placed into prompt
        effective_label = inherit_label([p.label for p in allowed_passages])
        ledger_entry = ledger_log(
            db_path=db_path,
            actor=principal.person_id,
            action="ASK_ANSWERED",
            details={
                "question": question,
                "citations_count": len(successful_draft.citations),
                "denials_count": len(denials),
                "effective_tier": effective_label.tier.value,
                "effective_compartments": [c.value for c in effective_label.compartments],
                "answer_preview": successful_draft.answer[:100],
            },
        )
        response = AskResponse(
            status="answered",
            answer=successful_draft.answer,
            citations=successful_draft.citations,
            denials=denials,
            effective_label=effective_label,
            ledger_row_id=ledger_entry.row_id,
            conversation_id=conversation_id,
        )
        # Persist the FULL answer/citations/denials for report regeneration.
        # The ledger entry above only carries a truncated preview (it's readable
        # by every authenticated user as a transparency log); this store is only
        # ever read back through the ownership-checked /report/{id} endpoint.
        save_report_data(
            db_path=db_path,
            ledger_row_id=ledger_entry.row_id,
            person_id=principal.person_id,
            question=question,
            response=response,
        )
        append_turn(db_path, conversation_id, question, response)
        return response
    else:
        # Retries exhausted: ABSTAIN. Never return an unverified answer.
        answer = (
            f"Response withheld: The drafting model was unable to provide citations that could be "
            f"verbatim verified against source passages ({feedback})."
        )
        effective_label = Label(tier=Tier.PUBLIC, compartments=frozenset())
        ledger_entry = ledger_log(
            db_path=db_path,
            actor=principal.person_id,
            action="ASK_ABSTAINED",
            details={
                "question": question,
                "reason": "verification_retries_exhausted",
                "final_feedback": feedback,
                "denials_count": len(denials),
            },
        )
        response = AskResponse(
            status="abstained",
            answer=answer,
            citations=[],
            denials=denials,
            effective_label=effective_label,
            ledger_row_id=ledger_entry.row_id,
            conversation_id=conversation_id,
        )
        append_turn(db_path, conversation_id, question, response)
        return response


def run_ephemeral_query(
    question: str,
    upload: EphemeralUpload,
    principal: Principal,
    agent: Agent,
    db_path: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    emit: Optional[Callable[[dict], None]] = None,
    conversation_id: Optional[str] = None,
) -> AskResponse:
    """Answer a question about an ephemeral, session-scoped user upload (an
    image, or a .pptx with mixed text/image content) -- NEVER governed corpus
    content, so NO two-axis clearance gate applies here. This is the user's
    own file, analyzed like a personal document assistant, not corpus data.

    Text content (if any) goes through the SAME drafting + citation
    verification loop as governed-corpus answers in run_query() above --
    every text claim must still be a verbatim substring of the uploaded text.
    Image content (if any) goes to the vision model and is reported in a
    CLEARLY SEPARATE section labeled as an AI description, not a verified
    citation -- there is no verbatim-substring check possible for an image
    the way there is for text, so it must never be presented with the same
    confidence as a citation-verified finding.
    """
    def _emit(event: dict) -> None:
        if emit is not None:
            emit(event)

    if conversation_id and conversation_owner(db_path, conversation_id) != principal.person_id:
        conversation_id = None
    if not conversation_id:
        conversation_id = create_conversation(db_path, principal.person_id, question)

    answer_sections: list[str] = []
    text_citations: list[Citation] = []

    # --- Text channel: identical citation-verification discipline as governed content ---
    if upload.text_chunks:
        passages = [
            Passage(
                doc_id=f"upload:{upload.filename}",
                page=idx + 1,
                title=chunk.label,
                text=chunk.text,
                label=Label(tier=Tier.PUBLIC, compartments=frozenset()),
            )
            for idx, chunk in enumerate(upload.text_chunks)
        ]

        feedback: Optional[str] = None
        successful_draft = None
        for attempt in range(1, max_retries + 1):
            _emit({"stage": "drafting", "status": "start", "attempt": attempt, "max_attempts": max_retries})
            try:
                draft = agent.draft(question=question, passages=passages, feedback=feedback, emit=emit)
            except Exception as exc:
                _emit({"stage": "drafting", "status": "error", "attempt": attempt, "message": str(exc)})
                feedback = f"Attempt {attempt} failed: drafting agent raised an error ({exc})."
                continue
            _emit({"stage": "drafting", "status": "done", "attempt": attempt, "citations_claimed": len(draft.citations)})

            _emit({"stage": "verification", "status": "start", "attempt": attempt})
            failure = verify_check(draft.citations, passages)
            if failure is None:
                _emit({"stage": "verification", "status": "pass", "attempt": attempt})
                successful_draft = draft
                break
            _emit({"stage": "verification", "status": "fail", "attempt": attempt, "reason": failure})
            avail_str = ", ".join(f"doc_id='{p.doc_id}' page {p.page}" for p in passages)
            feedback = (
                f"Attempt {attempt} verification failed: {failure}. Note: The ONLY readable passages in "
                f"context are [{avail_str}]. Ensure all quotes are exact verbatim substrings and page "
                f"numbers match these exactly."
            )

        if successful_draft is not None:
            answer_sections.append("## Document Text Findings (Verbatim Verified)\n\n" + successful_draft.answer)
            text_citations = successful_draft.citations
        else:
            answer_sections.append(
                "## Document Text Findings\n\nThe drafting model could not produce citations that verified "
                "verbatim against the uploaded text, so no text-based finding is reported."
            )

    # --- Image channel: vision model, explicitly and visibly unverified ---
    analyze_image = getattr(agent, "analyze_image", None)
    if upload.images and analyze_image is not None:
        observations: list[str] = []
        for image in upload.images:
            _emit({"stage": "vision", "status": "start", "label": image.label})
            try:
                description = analyze_image(question, image.base64_data)
            except Exception as exc:
                description = f"(Vision analysis failed: {exc})"
            _emit({"stage": "vision", "status": "done", "label": image.label})
            observations.append(f"**{image.label}:** {description}")

        answer_sections.append(
            "## Visual Observations (AI-Described — Not Verbatim-Verified)\n\n"
            "The following are the vision model's own descriptions of the image content. Unlike the text "
            "findings above, these cannot be checked against an exact source string and should be treated "
            "as an aid, not a verified fact.\n\n" + "\n\n".join(observations)
        )
    elif upload.images and analyze_image is None:
        answer_sections.append(
            "## Visual Observations\n\nThis upload contains images, but the active agent backend does not "
            "support image analysis."
        )

    answer = "\n\n".join(answer_sections) if answer_sections else "The uploaded file contained no analyzable text or images."

    effective_label = Label(tier=Tier.PUBLIC, compartments=frozenset())
    ledger_entry = ledger_log(
        db_path=db_path,
        actor=principal.person_id,
        action="ASK_ANSWERED",
        details={
            "question": question,
            "category": "ephemeral_upload",
            "filename": upload.filename,
            "text_chunks": len(upload.text_chunks),
            "images": len(upload.images),
            "answer_preview": answer[:100],
        },
    )
    response = AskResponse(
        status="answered",
        answer=answer,
        citations=text_citations,
        denials=[],
        effective_label=effective_label,
        ledger_row_id=ledger_entry.row_id,
        conversation_id=conversation_id,
    )
    save_report_data(
        db_path=db_path,
        ledger_row_id=ledger_entry.row_id,
        person_id=principal.person_id,
        question=question,
        response=response,
    )
    append_turn(db_path, conversation_id, question, response)
    return response
