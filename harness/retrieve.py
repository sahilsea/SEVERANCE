"""Two-pass scoring, gating, and abstention engine.

NON-NEGOTIABLE DESIGN PRINCIPLES:
1. Two-pass retrieval:
   - Pass 1: Score EVERY passage first, completely ignoring labels.
     Never filter before scoring — you must know a passage existed in order
     to record that it was withheld.
   - Pass 2: Evaluate the two-axis gate (can_read).
2. Denied passages have their text DISCARDED AT THE GATE.
   Denied text exists nowhere downstream.
3. A Denial object carries a doc_id and a deterministic reason. NEVER text.
4. top_k counts READABLE passages, not candidates. If top matches are denied,
   keep walking until top_k readable passages are found or corpus is exhausted.
5. Collapse denials per document, not per passage.
6. Abstention: If nothing readable matched, DO NOT CALL THE MODEL.
   Abstain and name the document that would have been needed (titles/doc_ids are not content).
"""

from __future__ import annotations

import math
import re
from typing import Sequence
from contracts import Denial, Passage, Principal
from trust.labels import can_read, denial_reason


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())


def score_passage(query_tokens: list[str], passage_tokens: list[str]) -> float:
    """Compute lexical match score (BM25-style term frequency overlap)."""
    if not query_tokens or not passage_tokens:
        return 0.0

    passage_len = len(passage_tokens)
    score = 0.0
    token_counts: dict[str, int] = {}
    for t in passage_tokens:
        token_counts[t] = token_counts.get(t, 0) + 1

    for q in query_tokens:
        count = token_counts.get(q, 0)
        if count > 0:
            # Term frequency saturation
            tf = (count * 2.2) / (count + 1.2 * (0.25 + 0.75 * (passage_len / 100.0)))
            score += tf

    return score


def retrieve(
    query: str,
    corpus: Sequence[Passage],
    principal: Principal,
    top_k: int = 3,
) -> tuple[list[Passage], list[Denial]]:
    """Execute two-pass retrieval: score all candidates, gate, and discard denied text.

    Returns:
        (allowed_passages, collapsed_denials)
    """
    if not corpus:
        return [], []

    query_tokens = tokenize(query)

    # -----------------------------------------------------------------------
    # Pass 1: Score every passage in the corpus completely ignoring labels.
    # -----------------------------------------------------------------------
    scored: list[tuple[float, Passage]] = []
    for passage in corpus:
        p_tokens = tokenize(passage.text) + tokenize(passage.title)
        score = score_passage(query_tokens, p_tokens)
        scored.append((score, passage))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)

    # -----------------------------------------------------------------------
    # Pass 2: Gate candidates against caller's clearance.
    # -----------------------------------------------------------------------
    allowed_passages: list[Passage] = []
    denials_by_doc_id: dict[str, Denial] = {}

    for score, passage in scored:
        # Evaluate deterministic two-axis gate
        if can_read(principal, passage.label):
            if len(allowed_passages) < top_k:
                allowed_passages.append(passage)
        else:
            # TEXT IS DISCARDED IMMEDIATELY AT THE GATE.
            # Record collapsed denial per document ID.
            if passage.doc_id not in denials_by_doc_id:
                reason = denial_reason(principal, passage.label)
                denials_by_doc_id[passage.doc_id] = Denial(
                    doc_id=passage.doc_id,
                    reason=reason,
                    required_label=passage.label,
                )

        # Stop walking if we have gathered top_k readable passages and checked candidates
        if len(allowed_passages) >= top_k:
            break

    return allowed_passages, list(denials_by_doc_id.values())
