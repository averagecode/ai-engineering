"""Deterministic eval metrics over an agent answer.

Pure functions: no MLflow, no network, no LLM judge. They take the dict produced
by `predict_fn` (derived from `research_assistant.agent.Answer`) plus a case's
ground truth, and return a score. That keeps them fast, reproducible and unit
testable, and lets the same logic back both `mlflow.genai.evaluate` scorers and
plain pytest assertions.

Why deterministic metrics first: most regressions in a RAG system are *retrieval*
regressions (an embedding-model swap, a chunk-size change, a broken filter), and
those are measurable exactly. An LLM judge is neither needed nor trustworthy for
them — especially when the only judge available is the model under test.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

#: Phrases a grounded agent uses when the corpus cannot answer the question.
DECLINE_MARKERS = (
    "do not cover",
    "does not cover",
    "do not contain",
    "does not contain",
    "not covered",
    "no information",
    "not discussed",
    "not mentioned",
    "not addressed",
    "cannot determine",
    "can not determine",
    "unable to find",
    "no passages",
    "not found in",
    "nothing in the",
    "do not discuss",
    "does not discuss",
    "do not mention",
    "does not mention",
    "do not provide",
    "does not provide",
    "no mention",
    "not relevant",
    "unrelated",
)


def _normalise(text: str) -> str:
    """Collapse whitespace so patterns are not defeated by PDF/LLM line breaks."""
    return re.sub(r"\s+", " ", str(text or ""))


def citation_sources(outputs: Mapping[str, Any]) -> list[str]:
    """Filenames the answer cited, e.g. "paper.pdf, p. 3" -> "paper.pdf"."""
    return [str(c).split(",", 1)[0].strip() for c in outputs.get("citations") or []]


def retrieved_pairs(outputs: Mapping[str, Any]) -> set[tuple[str, int]]:
    """(source, page) pairs the retriever actually returned."""
    pairs: set[tuple[str, int]] = set()
    for doc in outputs.get("documents") or []:
        source, page = doc.get("source"), doc.get("page")
        if source is not None and page is not None:
            pairs.add((str(source), int(page)))
    return pairs


# --- did the agent use the index at all? ------------------------------------ #


def retrieved_anything(outputs: Mapping[str, Any]) -> bool:
    """False when the model answered from memory instead of searching.

    This is the metric that catches the failure mode where a small model treats
    "these papers" as ambiguous and replies "which papers?" without searching.
    """
    return bool(outputs.get("tools_used"))


# --- retrieval quality ------------------------------------------------------ #


def retrieved_expected_source(outputs: Mapping[str, Any], sources: Sequence[str]) -> float:
    """Fraction of the expected papers that appear among retrieved documents."""
    if not sources:
        return 1.0
    retrieved = {source for source, _ in retrieved_pairs(outputs)}
    return sum(source in retrieved for source in sources) / len(sources)


def retrieved_expected_page(
    outputs: Mapping[str, Any], sources: Sequence[str], pages: Sequence[int] | None
) -> float:
    """Fraction of expected papers retrieved *on an acceptable page*.

    Stricter than `retrieved_expected_source`: the right paper is not enough if
    the chunk carrying the fact was missed.
    """
    if not sources:
        return 1.0
    if not pages:
        return retrieved_expected_source(outputs, sources)
    retrieved = retrieved_pairs(outputs)
    hit = sum(any((source, page) in retrieved for page in pages) for source in sources)
    return hit / len(sources)


def reciprocal_rank(outputs: Mapping[str, Any], sources: Sequence[str]) -> float:
    """1/rank of the first retrieved chunk from an expected paper, else 0.

    Rewards ranking the right paper first rather than merely somewhere in the top-k.
    """
    if not sources:
        return 0.0
    wanted = set(sources)
    for index, doc in enumerate(outputs.get("documents") or [], start=1):
        if str(doc.get("source")) in wanted:
            return 1.0 / index
    return 0.0


# --- citation integrity ----------------------------------------------------- #


def citations_are_valid(outputs: Mapping[str, Any], corpus: Iterable[str]) -> float:
    """Fraction of cited filenames that actually exist in the corpus.

    Guards the claim that the assistant cannot fabricate a source.
    """
    cited = citation_sources(outputs)
    if not cited:
        return 1.0  # nothing cited cannot be an invalid citation
    known = set(corpus)
    return sum(source in known for source in cited) / len(cited)


def citations_are_grounded(outputs: Mapping[str, Any]) -> float:
    """Fraction of citations that correspond to a document actually retrieved.

    Catches a citation naming a page the agent never read.
    """
    citations = outputs.get("citations") or []
    if not citations:
        return 1.0
    retrieved = {f"{source}, p. {page}" for source, page in retrieved_pairs(outputs)}
    return sum(str(c).strip() in retrieved for c in citations) / len(citations)


def cited_expected_source(outputs: Mapping[str, Any], sources: Sequence[str]) -> float:
    """Fraction of expected papers that the answer actually cited."""
    if not sources:
        return 1.0
    cited = set(citation_sources(outputs))
    return sum(source in cited for source in sources) / len(sources)


# --- answer content --------------------------------------------------------- #


def matches_any_pattern(outputs: Mapping[str, Any], patterns: Sequence[str]) -> bool:
    """Whether the answer text contains any expected fact pattern."""
    if not patterns:
        return True
    text = _normalise(outputs.get("answer"))
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def pattern_coverage(outputs: Mapping[str, Any], patterns: Sequence[str]) -> float:
    """Fraction of expected patterns present — useful for multi-fact answers."""
    if not patterns:
        return 1.0
    text = _normalise(outputs.get("answer"))
    return sum(bool(re.search(p, text, re.IGNORECASE)) for p in patterns) / len(patterns)


def cites_inline(outputs: Mapping[str, Any]) -> bool:
    """Whether the prose carries inline "(paper.pdf, p. N)" attribution.

    Tracked separately because small models reliably ground their answers while
    ignoring the inline-citation instruction; the `Sources` footer stays correct
    either way. Measuring it shows the gap rather than hiding it.
    """
    return bool(re.search(r"\(?[\w.\-]+\.pdf\s*,?\s*p\.?\s*\d+", _normalise(outputs.get("answer"))))


# --- refusal behaviour ------------------------------------------------------ #


def declined(outputs: Mapping[str, Any]) -> bool:
    """Whether the answer explicitly says the corpus does not cover the question."""
    text = _normalise(outputs.get("answer")).lower()
    return any(marker in text for marker in DECLINE_MARKERS)


def refused_out_of_corpus(outputs: Mapping[str, Any]) -> bool:
    """For unanswerable questions: declined, and claimed no paper as support.

    Both halves matter. Declining while still citing a paper would mean the
    agent attributed a non-answer to a real source.
    """
    return declined(outputs) and not (outputs.get("citations") or [])
