"""Sentence-level grounding check -- the companion to harness/verify.py.

verify.py proves every QUOTE is a verbatim substring of its passage, but it
says nothing about the answer prose around those quotes: a draft can carry
two perfectly valid quotes while its sentences add findings the passages
never state (seen live: an equipment-list citation "supporting" invented
inspection findings). This module scores each answer sentence against the
passages the model was actually given and removes sentences whose content
isn't there.

DESIGN (same principles as verify.py):
- Pure Python, zero model calls, deterministic and auditable.
- A sentence is UNSUPPORTED if under SUPPORT_THRESHOLD of its content words
  (stopwords and generic filler removed, crude suffix-stemmed) appear in the
  passages, or if it states a specific number (above SMALL_NUMBER_MAX) that
  appears nowhere in the passages. Small numbers are usually counts the model
  derived ("3 steps"), so they are ignored rather than required.
- Polarity: if the source sentences a statement is built from say the
  opposite ("do not use water" vs "use water"), the statement is unsupported
  even though every word matches.
- Threshold chosen on real stored answers from this corpus: grounded
  sentences scored 71-100%, invented ones 40-61%.

HONEST LIMIT: this catches invented facts -- entities, actions, numbers that
aren't in the source -- and flipped negations. It cannot catch every false
relationship composed of words the source does contain ("X was used
improperly" when the source merely lists X). It is a filter, not a proof of
entailment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence
from contracts import Passage

SUPPORT_THRESHOLD = 0.70
MIN_CONTENT_TOKENS = 3  # shorter fragments (headings, "Stay calm.") are not scored
SMALL_NUMBER_MAX = 10  # counts/ordinals the model derives; not required in the source
POLARITY_MATCH = 0.6  # a source sentence sharing this much of a statement's content is its basis
_NEGATIONS = {"no", "not", "never", "none", "nor", "cannot", "without"}
# Prohibitions carry the same polarity as "not": "is prohibited" restates "do not".
_NEGATIVE_PREFIXES = ("prohibit", "forbid", "avoid", "refrain", "disallow", "banned", "restrict")

_STOPWORDS = set("""
a an the and or but if then than that this these those there here of in on at to for from by with as is
are was were be been being it its they them their he she his her we our you your i me my not no also can
could should would will may might must shall do does did done have has had into onto over under about after
before during while such which who whom whose what when where why how all any each every some more most other
same so very just only per via etc include includes including included use used using ensure ensures
ensuring implement implementing measure measures attempt provide provided follow following outlined
detailed specific relevant appropriate additionally according page pages document documents
passage passages source sources section states stated mentions mentioned notes noted says said describes
described specifies specified indicates indicated outlines
""".split())

_SUFFIXES = ("ations", "ation", "ings", "ing", "ied", "ies", "ed", "es", "ly", "s")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(*])")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:6]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("n't", " not").replace("n’t", " not"))


def _is_small_number(token: str) -> bool:
    return token.isdigit() and int(token) <= SMALL_NUMBER_MAX


def _content_tokens(text: str) -> list[str]:
    return [
        _stem(t) for t in _words(text)
        if (t.isdigit() and not _is_small_number(t)) or (not t.isdigit() and len(t) >= 3 and t not in _STOPWORDS)
    ]


def _negated(text: str) -> bool:
    return any(w in _NEGATIONS or w.startswith(_NEGATIVE_PREFIXES) for w in _words(text))


def _source_sentences(passages: Sequence[Passage]) -> list[tuple[set[str], bool]]:
    units = []
    for p in passages:
        for sentence in re.split(r"(?<=[.!?;:])\s+|\n+", p.text):
            tokens = set(_content_tokens(sentence))
            if tokens:
                units.append((tokens, _negated(sentence)))
    return units


def _polarity_conflict(sentence: str, tokens: list[str], sources: list[tuple[set[str], bool]]) -> bool:
    """True when the source sentences this statement is built from all carry
    the opposite negation. No close source sentence means the statement
    synthesises across several, and polarity isn't judged."""
    wanted = set(tokens)
    basis = [neg for src, neg in sources if len(wanted & src) >= POLARITY_MATCH * len(wanted)]
    return bool(basis) and _negated(sentence) not in basis


def _split_sentences(line: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BREAK.split(line) if s.strip()]


@dataclass
class GroundingResult:
    checked: int = 0
    unsupported: list[str] = field(default_factory=list)


def check(answer: str, passages: Sequence[Passage]) -> GroundingResult:
    """Score every sentence of `answer` against `passages`."""
    # Attribution ("According to emergency_rulebook, page 52, ...") is legitimate,
    # so each passage's id, title and page number count as source text too.
    source = " ".join(f"{p.doc_id} {p.title} {p.page} {p.text}" for p in passages)
    vocab = set(_content_tokens(source))
    sources = _source_sentences(passages)
    result = GroundingResult()
    for line in answer.splitlines():
        if line.strip().startswith("#"):
            continue  # headings summarise; they are never removed, so never scored
        body = _LIST_MARKER.sub("", line.strip())
        for sentence in _split_sentences(body):
            tokens = _content_tokens(sentence)
            if len(tokens) < MIN_CONTENT_TOKENS:
                continue
            result.checked += 1
            missing = [t for t in tokens if t not in vocab]
            invented_number = any(t.isdigit() and t not in vocab for t in tokens)
            if (
                invented_number
                or 1 - len(missing) / len(tokens) < SUPPORT_THRESHOLD
                or _polarity_conflict(sentence, tokens, sources)
            ):
                result.unsupported.append(sentence)
    return result


def remove_sentences(answer: str, unsupported: Sequence[str]) -> str:
    """Drop the given sentences from `answer`, keeping its Markdown structure.
    List items and paragraphs that become empty are removed, and so is any
    heading left with nothing under it."""
    drop = set(unsupported)
    kept_lines: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            kept_lines.append(line)
            continue
        marker_match = _LIST_MARKER.match(line)
        marker = marker_match.group(0) if marker_match else ""
        body = line[len(marker):] if marker else stripped
        kept = [s for s in _split_sentences(body) if s not in drop]
        if kept:
            kept_lines.append(marker + " ".join(kept))

    # Remove headings that no longer have any content before the next heading.
    result: list[str] = []
    for i, line in enumerate(kept_lines):
        if line.strip().startswith("#"):
            following = next((l for l in kept_lines[i + 1:] if l.strip()), "")
            if not following or following.strip().startswith("#"):
                continue
        result.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip()
