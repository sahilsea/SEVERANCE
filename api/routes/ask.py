"""Ask and report delivery endpoints.

ZERO ACCESS DECISIONS INSIDE.
Identity is derived strictly from the signed cookie.
Gating is performed by harness/runner.py calling trust/labels.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Response, status
from contracts import (
    AskRequest,
    AskResponse,
    Citation,
    Denial,
    Label,
    Principal,
    Tier,
)
from agents.mock import MockAgent
from auth.deps import current_principal, get_db_path
from deliver.docx import build_report, get_report_filename
from harness.runner import run_query
from ingest.pdf import load_corpus
from trust.ledger import get_db_connection

router = APIRouter(tags=["Document Workbench"])

MANIFEST_PATH = Path(__file__).parent.parent.parent / "corpus" / "manifest.json"
DOCS_DIR = Path(__file__).parent.parent.parent / "corpus" / "documents"

# Global cached corpus with dynamic mtime invalidation
_CORPUS_CACHE = None
_CORPUS_LAST_CHECK = 0.0


def get_corpus(force_reload: bool = False):
    global _CORPUS_CACHE, _CORPUS_LAST_CHECK
    import time

    current_mtime = 0.0
    if DOCS_DIR.exists():
        pdf_mtimes = [p.stat().st_mtime for p in DOCS_DIR.glob("*.pdf")]
        current_mtime = max(pdf_mtimes + [DOCS_DIR.stat().st_mtime, 0.0])

    if _CORPUS_CACHE is None or force_reload or current_mtime > _CORPUS_LAST_CHECK:
        _CORPUS_CACHE = load_corpus(MANIFEST_PATH, DOCS_DIR)
        _CORPUS_LAST_CHECK = max(current_mtime, time.time())
    return _CORPUS_CACHE


def get_agent():
    """Resolve drafting agent backend based on configuration."""
    backend = os.getenv("AGENT_BACKEND", "groq").lower()
    if backend in ["groq", "ollama", "real"]:
        from agents.real import RealLlmAgent
        return RealLlmAgent()
    elif backend == "adk":
        try:
            from agents.adk import AdkAgent
            return AdkAgent()
        except ImportError:
            print("[WARN] Google ADK not installed; falling back to MockAgent.")
            return MockAgent()
    return MockAgent()


@router.post("/ask", response_model=AskResponse)
def ask_question(
    payload: AskRequest,
    principal: Principal = Depends(current_principal),
):
    """Primary workbench query endpoint."""
    db_path = get_db_path()
    corpus = get_corpus()
    agent = get_agent()

    response = run_query(
        request=payload,
        corpus=corpus,
        principal=principal,
        agent=agent,
        db_path=db_path,
    )
    return response


@router.get("/report/{ledger_row_id}")
def download_report(
    ledger_row_id: int,
    principal: Principal = Depends(current_principal),
):
    """Download Word report for an answered query with 3-way classification stamping."""
    db_path = get_db_path()
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ledger WHERE row_id = ?;", (ledger_row_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ledger record not found.")

        action = row["action"]
        if action != "ASK_ANSWERED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot generate deliverable report for query outcome '{action}'. Reports are only generated for verified answers.",
            )

        details = json.loads(row["details"])
        question = details.get("question", "Inquiry")

        # Reconstruct AskResponse from ledger details
        effective_tier = Tier(details.get("effective_tier", Tier.INTERNAL.value))
        effective_comps = frozenset()
        if "effective_compartments" in details:
            from contracts import Compartment
            effective_comps = frozenset(Compartment(c) for c in details["effective_compartments"])

        effective_label = Label(tier=effective_tier, compartments=effective_comps)

        # Mock citations or answer preview for report rebuilding
        ask_response = AskResponse(
            status="answered",
            answer=details.get("answer_preview", "Synthesis complete."),
            citations=[],
            denials=[],
            effective_label=effective_label,
            ledger_row_id=ledger_row_id,
        )

        docx_bytes = build_report(ask_response, question=question)
        filename = get_report_filename(ask_response)

        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    finally:
        conn.close()
