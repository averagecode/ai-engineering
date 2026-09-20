"""Run the eval suite against the live assistant and log results to MLflow.

    python -m evals.run_eval                      # deterministic scorers only
    python -m evals.run_eval --judge              # add LLM-as-judge scorers
    python -m evals.run_eval --kind single_source # a subset, while iterating
    python -m evals.run_eval --limit 3            # smoke run

Requires Weaviate and Ollama to be reachable and the corpus already ingested
(`docker compose up -d weaviate ollama && docker compose run --rm assistant
ingest`). Tracking is local: MLflow writes to ./mlruns, so nothing is sent
anywhere and nothing costs money. Browse it with `mlflow ui`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("evals")

DATASET = Path(__file__).with_name("dataset.yaml")
#: Questions the corpus cannot answer are scored on refusal, not on retrieval.
OUT_OF_CORPUS = "out_of_corpus"

#: Kinds whose ground truth names a specific paper, so source-level metrics apply.
SOURCE_ASSERTING = ("bibliographic", "scoped_fact")

#: Kinds that assert an answer but not which paper supplied it. Scoring these on
#: source metrics would penalise a correct answer drawn from an equally valid
#: paper — the corpus has several papers per topic.
OPEN = ("open_fact", "inventory")


def discover_corpus(data_dir: Path | None = None) -> list[str]:
    """Filenames of every PDF on disk.

    Derived rather than hard-coded: a stale corpus list makes
    `citations_are_valid` report false failures as soon as papers are added, and
    that metric exists precisely to catch fabricated sources.
    """
    directory = data_dir or (PROJECT_ROOT / "data")
    return sorted(p.name for p in directory.glob("*.pdf"))


def load_cases(
    kinds: list[str] | None = None,
    limit: int | None = None,
    ids: list[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Read dataset.yaml and return (corpus, cases)."""
    spec = yaml.safe_load(DATASET.read_text(encoding="utf-8"))
    cases = spec["cases"]
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    if kinds:
        cases = [c for c in cases if c["kind"] in kinds]
    if limit is not None:
        cases = cases[:limit]
    return discover_corpus(), cases


def to_eval_rows(corpus: list[str], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert cases into MLflow's {inputs, expectations} row format.

    `inputs` is passed to `predict_fn` as keyword arguments, so its keys must match
    that function's parameter names.
    """
    rows = []
    for case in cases:
        sources = case.get("sources") or ([case["source"]] if case.get("source") else [])
        rows.append(
            {
                "inputs": {"question": case["question"]},
                "expectations": {
                    "case_id": case["id"],
                    "kind": case["kind"],
                    "sources": sources,
                    "pages": case.get("pages"),
                    "patterns": case.get("patterns") or [],
                    "corpus": corpus,
                    # Consumed by MLflow's built-in Correctness judge.
                    "expected_facts": case.get("expected_facts") or [],
                },
            }
        )
    return rows


def build_predict_fn(verbose: bool = False):
    """Return (predict_fn, close) wired to the live Weaviate + Ollama stack.

    The assistant is built once and reused across rows: rebuilding it per question
    would re-open Weaviate connections and reload the model needlessly.
    """
    from research_assistant import llm as llm_module
    from research_assistant import vectorstore as vs
    from research_assistant.agent import build_assistant
    from research_assistant.config import get_settings

    settings = get_settings()
    llm_module.verify_models(settings)

    client = vs.connect(settings)
    embeddings = llm_module.build_embeddings(settings)
    chat_model = llm_module.build_chat_model(settings)
    store = vs.get_vector_store(client, embeddings, settings)
    assistant = build_assistant(client, store, chat_model, settings, verbose=verbose)

    stats = vs.collection_stats(client, settings.collection_name)
    if stats["chunk_count"] == 0:
        client.close()
        raise SystemExit(
            "The index is empty — run `docker compose run --rm assistant ingest` first."
        )
    logger.info("Evaluating against %s chunks from %s paper(s)", stats["chunk_count"], len(stats["sources"]))

    def predict_fn(question: str) -> dict[str, Any]:
        """Answer one question, flattened into plain data for the scorers."""
        # Each row is an independent question, so conversation history must not
        # leak between them and skew retrieval.
        assistant.reset()
        answer = assistant.ask(question)
        return {
            "answer": answer.answer,
            "citations": list(answer.citations),
            "tools_used": list(answer.tools_used),
            "documents": [
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "text": doc.page_content,
                }
                for doc in answer.documents
            ],
        }

    def close() -> None:
        for model in (embeddings, chat_model):
            llm_module.close_model(model)
        client.close()

    return predict_fn, close


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", action="append", dest="kinds",
        # Derived from the dataset so the choices cannot go stale when kinds change.
        choices=sorted({c["kind"] for c in load_cases()[1]}),
        help="Only run cases of this kind (repeatable).",
    )
    parser.add_argument("--limit", type=int, help="Run at most N cases (smoke run).")
    parser.add_argument("--id", action="append", dest="ids", help="Run only this case id (repeatable).")
    parser.add_argument(
        "--judge", action="store_true",
        help="Also run LLM-as-judge scorers (slow; judge quality caps their value).",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="MLflow judge URI, e.g. ollama:/qwen2.5:7b. Defaults to the configured LLM.",
    )
    parser.add_argument("--experiment", default="research-assistant-evals")
    parser.add_argument("--verbose", action="store_true", help="Show agent tool calls.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    import mlflow
    import mlflow.genai

    from research_assistant.config import get_settings

    from . import scorers as s

    corpus, cases = load_cases(args.kinds, args.limit, args.ids)
    if not cases:
        print("No cases selected.")
        return 1

    # Three scoring groups, because the kinds support different assertions.
    scoped = [c for c in cases if c["kind"] in SOURCE_ASSERTING]
    open_ended = [c for c in cases if c["kind"] in OPEN]
    unanswerable = [c for c in cases if c["kind"] == OUT_OF_CORPUS]

    judge_model = args.judge_model or f"ollama:/{get_settings().llm_model}"
    extra = s.judge_scorers(judge_model) if args.judge else []
    if args.judge:
        print(f"Judge model: {judge_model}")

    mlflow.set_experiment(args.experiment)
    predict_fn, close = build_predict_fn(verbose=args.verbose)

    try:
        for label, group, group_scorers in (
            ("source-asserting (bibliographic + scoped_fact)", scoped, s.SCOPED_SCORERS + extra),
            ("open (paper-agnostic)", open_ended, s.OPEN_SCORERS + extra),
            ("out_of_corpus (must decline)", unanswerable, s.OUT_OF_CORPUS_SCORERS),
        ):
            if not group:
                continue
            print(f"\n=== {label}: {len(group)} case(s) ===")
            result = mlflow.genai.evaluate(
                data=to_eval_rows(corpus, group),
                predict_fn=predict_fn,
                scorers=group_scorers,
            )
            for name, value in sorted((result.metrics or {}).items()):
                if isinstance(value, float):
                    print(f"  {name:48} {value:.3f}")
                else:
                    print(f"  {name:48} {value}")
    finally:
        close()

    print("\nLogged to MLflow. Browse with:  mlflow ui")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
