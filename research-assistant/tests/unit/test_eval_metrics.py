"""Unit tests for the deterministic eval metrics.

These need no MLflow, no services and no inference: `evals.metrics` is pure
functions over plain dicts. A metric that silently returns the wrong score is
worse than no metric, so each one is pinned on both its passing and failing case.
"""

from __future__ import annotations

import pytest

from evals import metrics

CORPUS = ["a.pdf", "b.pdf"]


def answer(
    text: str = "",
    citations: list[str] | None = None,
    documents: list[dict] | None = None,
    tools: list[str] | None = None,
) -> dict:
    return {
        "answer": text,
        "citations": citations if citations is not None else [],
        "documents": documents if documents is not None else [],
        "tools_used": tools if tools is not None else [],
    }


def doc(source: str, page: int, text: str = "excerpt") -> dict:
    return {"source": source, "page": page, "text": text}


# --- helpers ---------------------------------------------------------------- #


def test_citation_sources_strips_the_page():
    out = answer(citations=["a.pdf, p. 3", "b.pdf, p. 11"])
    assert metrics.citation_sources(out) == ["a.pdf", "b.pdf"]


def test_citation_sources_handles_a_citation_without_a_page():
    assert metrics.citation_sources(answer(citations=["a.pdf"])) == ["a.pdf"]


def test_retrieved_pairs_ignores_incomplete_metadata():
    out = answer(documents=[doc("a.pdf", 1), {"source": "b.pdf"}, {"page": 4}])
    assert metrics.retrieved_pairs(out) == {("a.pdf", 1)}


# --- retrieval happened ----------------------------------------------------- #


def test_retrieved_anything():
    assert metrics.retrieved_anything(answer(tools=["search_papers"])) is True
    assert metrics.retrieved_anything(answer()) is False


# --- retrieval quality ------------------------------------------------------ #


def test_retrieved_expected_source_full_and_partial():
    out = answer(documents=[doc("a.pdf", 1)])
    assert metrics.retrieved_expected_source(out, ["a.pdf"]) == 1.0
    assert metrics.retrieved_expected_source(out, ["a.pdf", "b.pdf"]) == 0.5
    assert metrics.retrieved_expected_source(out, ["b.pdf"]) == 0.0


def test_retrieved_expected_source_with_no_expectation_is_vacuously_true():
    assert metrics.retrieved_expected_source(answer(), []) == 1.0


def test_retrieved_expected_page_is_stricter_than_source():
    out = answer(documents=[doc("a.pdf", 7)])
    # Right paper, wrong page: source-level passes, page-level does not.
    assert metrics.retrieved_expected_source(out, ["a.pdf"]) == 1.0
    assert metrics.retrieved_expected_page(out, ["a.pdf"], [1, 10]) == 0.0
    assert metrics.retrieved_expected_page(out, ["a.pdf"], [7]) == 1.0


def test_retrieved_expected_page_falls_back_when_no_pages_declared():
    out = answer(documents=[doc("a.pdf", 7)])
    assert metrics.retrieved_expected_page(out, ["a.pdf"], None) == 1.0


def test_reciprocal_rank_rewards_earlier_hits():
    first = answer(documents=[doc("a.pdf", 1), doc("b.pdf", 1)])
    third = answer(documents=[doc("b.pdf", 1), doc("b.pdf", 2), doc("a.pdf", 1)])
    assert metrics.reciprocal_rank(first, ["a.pdf"]) == 1.0
    assert metrics.reciprocal_rank(third, ["a.pdf"]) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank(answer(), ["a.pdf"]) == 0.0


# --- citation integrity ----------------------------------------------------- #


def test_citations_are_valid_detects_a_fabricated_source():
    assert metrics.citations_are_valid(answer(citations=["a.pdf, p. 1"]), CORPUS) == 1.0
    assert metrics.citations_are_valid(answer(citations=["ghost.pdf, p. 1"]), CORPUS) == 0.0
    mixed = answer(citations=["a.pdf, p. 1", "ghost.pdf, p. 2"])
    assert metrics.citations_are_valid(mixed, CORPUS) == 0.5


def test_no_citations_cannot_be_invalid():
    assert metrics.citations_are_valid(answer(), CORPUS) == 1.0


def test_citations_are_grounded_detects_an_unread_page():
    grounded = answer(citations=["a.pdf, p. 3"], documents=[doc("a.pdf", 3)])
    ungrounded = answer(citations=["a.pdf, p. 9"], documents=[doc("a.pdf", 3)])
    assert metrics.citations_are_grounded(grounded) == 1.0
    assert metrics.citations_are_grounded(ungrounded) == 0.0


def test_cited_expected_source():
    out = answer(citations=["a.pdf, p. 1"])
    assert metrics.cited_expected_source(out, ["a.pdf"]) == 1.0
    assert metrics.cited_expected_source(out, ["a.pdf", "b.pdf"]) == 0.5


# --- answer content --------------------------------------------------------- #


def test_matches_any_pattern_survives_line_breaks():
    out = answer("the value is 11 ±\n7 per cent of the total")
    assert metrics.matches_any_pattern(out, [r"11\s*±\s*7"]) is True


def test_matches_any_pattern_is_case_insensitive():
    assert metrics.matches_any_pattern(answer("SIMBA and TNG"), ["simba"]) is True


def test_matches_any_pattern_with_no_patterns_is_vacuously_true():
    assert metrics.matches_any_pattern(answer("anything"), []) is True


def test_matches_any_pattern_fails_when_absent():
    assert metrics.matches_any_pattern(answer("no figures here"), [r"5\.3"]) is False


def test_pattern_coverage_is_fractional():
    out = answer("we find 2.9 sigma but not the other value")
    assert metrics.pattern_coverage(out, [r"2\.9", r"5\.3"]) == 0.5


def test_cites_inline_distinguishes_prose_citations_from_a_footer():
    inline = answer("It is 11 per cent (1709.10378v3.pdf, p. 10).")
    footer_only = answer("It is 11 per cent.", citations=["1709.10378v3.pdf, p. 10"])
    assert metrics.cites_inline(inline) is True
    assert metrics.cites_inline(footer_only) is False


# --- refusal ---------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "The indexed papers do not cover this question.",
        "These papers do not discuss exoplanet atmospheres.",
        "There is no information about that in the retrieved excerpts.",
        "I cannot determine that from these papers.",
    ],
)
def test_declined_recognises_refusal_phrasings(text: str):
    assert metrics.declined(answer(text)) is True


def test_declined_is_false_for_a_substantive_answer():
    assert metrics.declined(answer("The filaments account for 11 per cent.")) is False


def test_refused_out_of_corpus_requires_both_halves():
    good = answer("These papers do not cover the Higgs boson.")
    assert metrics.refused_out_of_corpus(good) is True

    # Declining but still citing a paper means attributing a non-answer to it.
    contradictory = answer(
        "These papers do not cover it.", citations=["a.pdf, p. 1"]
    )
    assert metrics.refused_out_of_corpus(contradictory) is False

    assert metrics.refused_out_of_corpus(answer("The answer is 42.")) is False


# --- robustness ------------------------------------------------------------- #


def test_metrics_tolerate_a_missing_or_empty_answer():
    empty: dict = {}
    assert metrics.retrieved_anything(empty) is False
    assert metrics.citations_are_grounded(empty) == 1.0
    assert metrics.cites_inline(empty) is False
    assert metrics.declined(empty) is False
    assert metrics.retrieved_expected_source(empty, ["a.pdf"]) == 0.0
