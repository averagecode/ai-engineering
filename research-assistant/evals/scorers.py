"""MLflow scorers wrapping the deterministic metrics in `evals.metrics`.

Custom scorers receive a subset of {inputs, outputs, expectations, trace,
session}; these use `outputs` (whatever `predict_fn` returned) and
`expectations` (the ground truth from dataset.yaml).

The built-in retrieval scorers MLflow provides — RetrievalGroundedness,
RetrievalRelevance, RetrievalSufficiency, ToolCallCorrectness — all require a
`trace` column, because they read MLflow retriever spans rather than plain
input/output text. Using them means instrumenting the agent with mlflow.tracing
first, so they are deliberately not wired up here; `judge_scorers()` offers the
input/output-only judges instead.
"""

from __future__ import annotations

from typing import Any, Mapping

from mlflow.genai.scorers import Correctness, Guidelines, Scorer, scorer

from . import metrics


def _sources(expectations: Mapping[str, Any] | None) -> list[str]:
    return list((expectations or {}).get("sources") or [])


def _patterns(expectations: Mapping[str, Any] | None) -> list[str]:
    return list((expectations or {}).get("patterns") or [])


# --- did it retrieve? ------------------------------------------------------- #


@scorer(name="retrieved_anything", aggregations=["mean"])
def retrieved_anything(outputs: Mapping[str, Any]) -> bool:
    """The agent consulted the index rather than answering from memory."""
    return metrics.retrieved_anything(outputs)


# --- retrieval quality ------------------------------------------------------ #


@scorer(name="retrieved_expected_source", aggregations=["mean"])
def retrieved_expected_source(
    outputs: Mapping[str, Any], expectations: Mapping[str, Any]
) -> float:
    """Expected paper(s) present among the retrieved chunks."""
    return metrics.retrieved_expected_source(outputs, _sources(expectations))


@scorer(name="retrieved_expected_page", aggregations=["mean"])
def retrieved_expected_page(outputs: Mapping[str, Any], expectations: Mapping[str, Any]) -> float:
    """Expected paper(s) retrieved on a page that actually carries the fact."""
    return metrics.retrieved_expected_page(
        outputs, _sources(expectations), (expectations or {}).get("pages")
    )


@scorer(name="reciprocal_rank", aggregations=["mean"])
def reciprocal_rank(outputs: Mapping[str, Any], expectations: Mapping[str, Any]) -> float:
    """1/rank of the first chunk from an expected paper."""
    return metrics.reciprocal_rank(outputs, _sources(expectations))


# --- citation integrity ----------------------------------------------------- #


@scorer(name="citations_are_valid", aggregations=["mean"])
def citations_are_valid(outputs: Mapping[str, Any], expectations: Mapping[str, Any]) -> float:
    """Every cited filename exists in the corpus (no fabricated sources)."""
    return metrics.citations_are_valid(outputs, (expectations or {}).get("corpus") or [])


@scorer(name="citations_are_grounded", aggregations=["mean"])
def citations_are_grounded(outputs: Mapping[str, Any]) -> float:
    """Every citation corresponds to a chunk the agent actually retrieved."""
    return metrics.citations_are_grounded(outputs)


@scorer(name="cited_expected_source", aggregations=["mean"])
def cited_expected_source(outputs: Mapping[str, Any], expectations: Mapping[str, Any]) -> float:
    """The answer credits the paper the fact really comes from."""
    return metrics.cited_expected_source(outputs, _sources(expectations))


# --- answer content --------------------------------------------------------- #


@scorer(name="contains_expected_fact", aggregations=["mean"])
def contains_expected_fact(outputs: Mapping[str, Any], expectations: Mapping[str, Any]) -> bool:
    """The answer states at least one of the expected figures."""
    return metrics.matches_any_pattern(outputs, _patterns(expectations))


@scorer(name="fact_coverage", aggregations=["mean"])
def fact_coverage(outputs: Mapping[str, Any], expectations: Mapping[str, Any]) -> float:
    """Fraction of expected figures present — matters for multi-fact answers."""
    return metrics.pattern_coverage(outputs, _patterns(expectations))


@scorer(name="cites_inline", aggregations=["mean"])
def cites_inline(outputs: Mapping[str, Any]) -> bool:
    """Prose carries inline (paper.pdf, p. N) attribution, not just a footer."""
    return metrics.cites_inline(outputs)


# --- refusal behaviour ------------------------------------------------------ #


@scorer(name="refused_out_of_corpus", aggregations=["mean"])
def refused_out_of_corpus(outputs: Mapping[str, Any]) -> bool:
    """Declined, and did not attribute the non-answer to any paper."""
    return metrics.refused_out_of_corpus(outputs)


#: Corpus-independent scorers: they assert nothing about which paper answered.
INTEGRITY_SCORERS: list[Scorer] = [
    retrieved_anything,
    citations_are_valid,
    citations_are_grounded,
    cites_inline,
]

#: For cases whose ground truth names a specific paper (bibliographic, scoped_fact).
SCOPED_SCORERS: list[Scorer] = INTEGRITY_SCORERS + [
    retrieved_expected_source,
    retrieved_expected_page,
    reciprocal_rank,
    cited_expected_source,
    contains_expected_fact,
    fact_coverage,
]

#: For paper-agnostic cases: score the answer and integrity, never the source.
OPEN_SCORERS: list[Scorer] = INTEGRITY_SCORERS + [
    contains_expected_fact,
    fact_coverage,
]

#: Scorers for questions the corpus cannot answer.
OUT_OF_CORPUS_SCORERS: list[Scorer] = [
    refused_out_of_corpus,
    citations_are_valid,
]


def judge_scorers(model: str) -> list[Scorer]:
    """LLM-as-judge scorers, which need only inputs/outputs (no trace).

    A caveat worth stating plainly: with a local setup the judge is usually the
    same small model being evaluated, and a 3B model is a weak judge. Treat these
    as a smoke signal and trust the deterministic scorers for regressions.

    `model` is an MLflow judge URI. MLflow has a first-class Ollama provider, so
    "ollama:/qwen2.5:3b" works with no API key (OllamaConfig defaults api_key to
    the literal "ollama" and talks to localhost:11434).
    """
    return [
        Correctness(model=model),
        Guidelines(
            name="grounded_in_excerpts",
            guidelines=[
                "The response must only assert facts that are attributed to one of "
                "the indexed papers, and must not introduce outside knowledge.",
                "If the response says the papers do not cover something, it must not "
                "also cite a paper as the source of an answer.",
            ],
            model=model,
        ),
    ]
