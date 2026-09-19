"""Integration tests for the real pipeline: PDF -> Ollama embeddings -> Weaviate.

These talk to live services. They skip themselves (via the `live_services`
fixture) when Weaviate or Ollama is not up, so the suite still passes offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_assistant import ingest as ingest_module
from research_assistant import vectorstore as vs
from research_assistant.llm import build_embeddings
from tests.conftest import PAPER_TEXT, write_pdf

pytestmark = pytest.mark.integration


@pytest.fixture
def papers(tmp_path: Path) -> list[Path]:
    """Both sample papers written to a temp directory."""
    names = ["sparse_routing.pdf", "retrieval_depth.pdf"]
    return [
        write_pdf(tmp_path / name, title, paragraphs)
        for (title, paragraphs), name in zip(PAPER_TEXT, names)
    ]


@pytest.fixture
def embeddings(live_services):
    return build_embeddings(live_services)


@pytest.fixture
def ingested(live_client, live_services, embeddings, papers):
    """Both papers indexed in a throwaway Weaviate collection."""
    report = ingest_module.ingest_paths(papers, live_client, embeddings, live_services)
    assert not report.failed, [f.detail for f in report.failed]
    assert len(report.ingested) == 2
    return report


# --- collection setup ------------------------------------------------------- #


def test_collection_is_created_with_the_expected_schema(live_client, live_services):
    vs.ensure_collection(live_client, live_services.collection_name)

    config = live_client.collections.get(live_services.collection_name).config.get()
    property_names = {prop.name for prop in config.properties}

    assert vs.TEXT_KEY in property_names
    assert {"source", "page", "chunk_id", "checksum"} <= property_names
    # Vectors come from Ollama via LangChain, so Weaviate must not vectorise.
    assert config.vectorizer is None or str(config.vectorizer).lower().endswith("none")


# --- embedding + storage ---------------------------------------------------- #


def test_real_embeddings_are_dense_vectors(embeddings):
    vector = embeddings.embed_query("mixture of experts routing")
    assert len(vector) > 100
    assert all(isinstance(value, float) for value in vector[:10])
    assert any(value != 0.0 for value in vector)


def test_ingestion_stores_chunks_with_metadata(ingested, live_client, live_services):
    stats = vs.collection_stats(live_client, live_services.collection_name)

    assert stats["exists"] is True
    assert stats["chunk_count"] == ingested.total_chunks > 0
    assert set(stats["sources"]) == {"sparse_routing.pdf", "retrieval_depth.pdf"}

    collection = live_client.collections.get(live_services.collection_name)
    sample = collection.query.fetch_objects(limit=1, include_vector=True).objects[0]
    assert sample.properties["source"].endswith(".pdf")
    assert isinstance(sample.properties["page"], int)
    assert len(sample.vector["default"]) > 100


# --- retrieval quality ------------------------------------------------------ #


def test_semantic_search_retrieves_the_relevant_paper(
    ingested, live_client, live_services, embeddings
):
    store = vs.get_vector_store(live_client, embeddings, live_services)

    results = store.similarity_search("How much does GSR reduce inference FLOPs?", k=3)

    assert results
    assert any("41" in doc.page_content for doc in results)
    assert any(doc.metadata["source"] == "sparse_routing.pdf" for doc in results)


def test_search_distinguishes_between_the_two_papers(
    ingested, live_client, live_services, embeddings
):
    store = vs.get_vector_store(live_client, embeddings, live_services)

    hallucination = store.similarity_search("hallucination rate versus retrieval depth", k=2)
    routing = store.similarity_search("mixture of experts router load balancing loss", k=2)

    assert hallucination[0].metadata["source"] == "retrieval_depth.pdf"
    assert routing[0].metadata["source"] == "sparse_routing.pdf"


def test_mmr_search_returns_diverse_chunks(ingested, live_client, live_services, embeddings):
    store = vs.get_vector_store(live_client, embeddings, live_services)
    results = store.max_marginal_relevance_search("efficiency of transformers", k=4, fetch_k=10)
    assert len(results) > 1
    assert len({doc.metadata["chunk_id"] for doc in results}) == len(results)


def test_retrieved_chunks_carry_citable_metadata(
    ingested, live_client, live_services, embeddings
):
    from research_assistant.agent import format_citation

    store = vs.get_vector_store(live_client, embeddings, live_services)
    results = store.similarity_search("load-balancing auxiliary loss", k=2)

    for doc in results:
        citation = format_citation(doc)
        assert citation.endswith(tuple("0123456789"))
        assert ".pdf, p. " in citation


# --- idempotency across real runs ------------------------------------------- #


def test_reingesting_unchanged_papers_is_skipped(
    ingested, live_client, live_services, embeddings, papers
):
    before = vs.collection_stats(live_client, live_services.collection_name)["chunk_count"]

    second = ingest_module.ingest_paths(papers, live_client, embeddings, live_services)

    assert len(second.skipped) == 2
    assert second.total_chunks == 0
    after = vs.collection_stats(live_client, live_services.collection_name)["chunk_count"]
    assert after == before


def test_force_reingest_does_not_duplicate_chunks(
    ingested, live_client, live_services, embeddings, papers
):
    before = vs.collection_stats(live_client, live_services.collection_name)["chunk_count"]

    forced = ingest_module.ingest_paths(
        papers, live_client, embeddings, live_services, force=True
    )

    assert len(forced.ingested) == 2
    after = vs.collection_stats(live_client, live_services.collection_name)["chunk_count"]
    assert after == before


def test_an_edited_paper_replaces_its_old_chunks(
    ingested, live_client, live_services, embeddings, papers
):
    edited = papers[1]
    write_pdf(edited, "Retrieval Depth, revised", ["A single short revised page about caching."])

    report = ingest_module.ingest_paths([edited], live_client, embeddings, live_services)

    assert len(report.ingested) == 1
    stats = vs.collection_stats(live_client, live_services.collection_name)
    assert stats["sources"]["retrieval_depth.pdf"] == report.total_chunks

    store = vs.get_vector_store(live_client, embeddings, live_services)
    texts = " ".join(
        doc.page_content
        for doc in store.similarity_search("hallucination rate percentage", k=5)
        if doc.metadata["source"] == "retrieval_depth.pdf"
    )
    assert "22 percent" not in texts  # the superseded content is gone


def test_deleting_one_source_leaves_the_other(ingested, live_client, live_services):
    removed = vs.delete_source(live_client, live_services.collection_name, "sparse_routing.pdf")
    assert removed > 0

    stats = vs.collection_stats(live_client, live_services.collection_name)
    assert set(stats["sources"]) == {"retrieval_depth.pdf"}


def test_ingest_directory_walks_a_real_folder(live_client, live_services, embeddings, tmp_path):
    data_dir = tmp_path / "library"
    data_dir.mkdir()
    write_pdf(data_dir / "cache.pdf", "Caching", ["A paper about key-value cache compression."])
    (data_dir / "readme.txt").write_text("ignored", encoding="utf-8")

    report = ingest_module.ingest_directory(
        live_client, embeddings, live_services, data_dir=data_dir
    )

    assert [r.path.name for r in report.ingested] == ["cache.pdf"]
