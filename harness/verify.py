"""Deterministic citation verification engine.

NON-NEGOTIABLE DESIGN PRINCIPLES:
1. Pure Python string matching, zero model calls.
2. Every quote must appear verbatim in the exact passage it claims,
   matched by BOTH doc_id AND page (never across the whole corpus).
3. Quotes must be 20-300 characters and contain at least 5 words.
4. check() returns None when clean, or a single precise feedback string naming
   the first problem. One clear problem is more fixable by an agent than five.
5. NEVER derive citations by regexing or scraping the model's prose.
"""

from __future__ import annotations

from typing import Optional, Sequence
from contracts import Citation, Passage


def check(
    citations: Sequence[Citation],
    passages: Sequence[Passage],
) -> Optional[str]:
    """Verify that every citation quote appears verbatim in its claimed passage.

    Returns:
        None if all citations are verified.
        A descriptive string explaining the first failure encountered.
    """
    if not citations:
        return None

    # Index passages by (doc_id, page)
    passage_map: dict[tuple[str, int], str] = {
        (p.doc_id, p.page): p.text for p in passages
    }

    for idx, cit in enumerate(citations):
        # 1. Structural requirements
        quote = cit.quote
        if len(quote) < 20:
            return f"Citation {idx + 1} quote is too short ({len(quote)} chars, minimum 20 required)."

        if len(quote) > 300:
            return f"Citation {idx + 1} quote exceeds maximum length ({len(quote)} chars, maximum 300 allowed)."

        words = quote.strip().split()
        if len(words) < 5:
            return f"Citation {idx + 1} quote has only {len(words)} words; minimum 5 words required."

        # 2. Exact passage resolution
        key = (cit.doc_id, cit.page)
        if key not in passage_map:
            return (
                f"Citation {idx + 1} claims doc_id='{cit.doc_id}' page {cit.page}, "
                f"but no passage for that document and page was provided in the prompt context."
            )

        # 3. Verbatim exact substring match within that specific page
        passage_text = passage_map[key]
        normalized_quote = " ".join(quote.strip().split())
        normalized_passage = " ".join(passage_text.strip().split())
        if quote not in passage_text and normalized_quote not in normalized_passage:
            snippet = quote[:50] + ("..." if len(quote) > 50 else "")
            return (
                f"Citation {idx + 1} quote '{snippet}' was NOT found verbatim in "
                f"doc_id='{cit.doc_id}' page {cit.page}."
            )

    return None
