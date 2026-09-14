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

from typing import Optional, Sequence
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
from trust.labels import inherit_label
from trust.ledger import log as ledger_log

DEFAULT_MAX_RETRIES = 3


def run_query(
    request: AskRequest,
    corpus: Sequence[Passage],
    principal: Principal,
    agent: Agent,
    db_path: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> AskResponse:
    """Execute the complete SEVERANCE query lifecycle.

    1. Retrieve and two-axis gate candidates (discarding denied text).
    2. Check for immediate abstention (if 0 readable passages, model is NOT called).
    3. Execute bounded drafting and citation verification loop.
    4. Compute classification inheritance over all prompt passages.
    5. Append tamper-evident audit record to SQLite ledger.
    """
    question = request.question.strip()

    # Step 1: Two-pass retrieval and gating
    allowed_passages, denials = retrieve(
        query=question,
        corpus=corpus,
        principal=principal,
        top_k=request.top_k,
    )

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
        return AskResponse(
            status="abstained",
            answer=answer,
            citations=[],
            denials=denials,
            effective_label=effective_label,
            ledger_row_id=ledger_entry.row_id,
        )

    # Step 3: Bounded Drafting & Verification Loop
    feedback: Optional[str] = None
    successful_draft = None

    for attempt in range(1, max_retries + 1):
        draft = agent.draft(
            question=question,
            passages=allowed_passages,
            feedback=feedback,
        )

        # Pure Python string verification
        failure = verify_check(draft.citations, allowed_passages)
        if failure is None:
            # Verification passed completely
            successful_draft = draft
            break

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
        return AskResponse(
            status="answered",
            answer=successful_draft.answer,
            citations=successful_draft.citations,
            denials=denials,
            effective_label=effective_label,
            ledger_row_id=ledger_entry.row_id,
        )
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
        return AskResponse(
            status="abstained",
            answer=answer,
            citations=[],
            denials=denials,
            effective_label=effective_label,
            ledger_row_id=ledger_entry.row_id,
        )
