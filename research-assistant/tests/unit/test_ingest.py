"""Unit tests for the ingestion pipeline (idempotency, batching, error handling)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from research_assistant import ingest as ingest_module
from research_assistant import vectorstore as vs
from research_assistant.loader import file_checksum
from tests.conftest import write_pdf
from tests.fakes import FakeEmbeddings, FakeStore, FakeWeaviateClient


@pytest.fixture
def store(mocker) -> FakeStore:
    """Replace the real vector store with an in-memory fake."""
    fake = FakeStore()
    mocker.patch.object(vs, "get_vector_store", return_value=fake)
    return fake


@pytest.fixture
def client() -> FakeWeaviateClient:
    return FakeWeaviateClient()


def _run(paths, client, settings, **kwargs):
    return ingest_module.ingest_paths(
        paths, client, FakeEmbeddings(), settings, **kwargs
    )


# --- happy path ------------------------------------------------------------- #


def test_ingest_embeds_and_stores_chunks(client, settings, store, pdf_library):
    report = _run(pdf_library, client, settings)

    assert [r.status for r in report.files] == ["ingested", "ingested"]
    assert report.total_chunks == len(store.documents) > 0
    assert {d.metadata["source"] for d in store.documents} == {
        "sparse_routing.pdf",
        "retrieval_depth.pdf",
    }


def test_ingested_chunks_carry_the_file_checksum(client, settings, store, sample_pdf):
    _run([sample_pdf], client, settings)
    expected = file_checksum(sample_pdf)
    assert all(d.metadata["checksum"] == expected for d in store.documents)


def test_documents_are_added_in_batches(client, settings, store, pdf_library, mocker):
    mocker.patch.object(
        ingest_module,
        "prepare_chunks",
        return_value=[
            Document(page_content=f"chunk {i}", metadata={"source": "big.pdf", "page": 1})
            for i in range(10)
        ],
    )
    _run([pdf_library[0]], client, settings, batch_size=4)
    assert store.add_batches == [4, 4, 2]


# --- idempotency ------------------------------------------------------------ #


def test_unchanged_file_is_skipped_on_reingest(client, settings, store, sample_pdf):
    checksum = file_checksum(sample_pdf)
    client.seed(settings.collection_name, [{"source": sample_pdf.name, "checksum": checksum}])

    report = _run([sample_pdf], client, settings)

    assert [r.status for r in report.files] == ["skipped"]
    assert "unchanged" in report.files[0].detail
    assert store.documents == []


def test_force_reingests_an_unchanged_file(client, settings, store, sample_pdf):
    client.seed(
        settings.collection_name,
        [{"source": sample_pdf.name, "checksum": file_checksum(sample_pdf)}],
    )
    report = _run([sample_pdf], client, settings, force=True)
    assert [r.status for r in report.files] == ["ingested"]
    assert store.documents


def test_changed_file_replaces_its_stale_chunks(client, settings, store, sample_pdf):
    collection = client.seed(
        settings.collection_name,
        [
            {"source": sample_pdf.name, "checksum": "stale"},
            {"source": sample_pdf.name, "checksum": "stale"},
            {"source": "other.pdf", "checksum": "keep"},
        ],
    )
    report = _run([sample_pdf], client, settings)

    assert [r.status for r in report.files] == ["ingested"]
    # Old chunks for this source are gone; the unrelated paper survives.
    assert [obj.properties["source"] for obj in collection.objects] == ["other.pdf"]


# --- degenerate inputs ------------------------------------------------------ #


def test_pdf_without_extractable_text_is_reported_empty(client, settings, store, sample_pdf, mocker):
    mocker.patch.object(ingest_module, "prepare_chunks", return_value=[])
    report = _run([sample_pdf], client, settings)
    assert [r.status for r in report.files] == ["empty"]
    assert "scanned" in report.files[0].detail
    assert store.documents == []


def test_one_bad_pdf_does_not_abort_the_run(client, settings, store, pdf_library, mocker):
    original = ingest_module.prepare_chunks

    def flaky(path, **kwargs):
        if path.name == "sparse_routing.pdf":
            raise ValueError("corrupt xref table")
        return original(path, **kwargs)

    mocker.patch.object(ingest_module, "prepare_chunks", side_effect=flaky)
    report = _run(sorted(pdf_library), client, settings)

    statuses = {r.path.name: r.status for r in report.files}
    assert statuses["sparse_routing.pdf"] == "failed"
    assert statuses["retrieval_depth.pdf"] == "ingested"
    assert "corrupt xref table" in report.failed[0].detail


def test_report_aggregates_counts(client, settings, store, pdf_library):
    report = _run(pdf_library, client, settings)
    assert len(report.ingested) == 2
    assert report.skipped == [] and report.failed == []
    assert report.total_chunks > 0


# --- directory driver ------------------------------------------------------- #


def test_ingest_directory_walks_the_data_dir(client, settings, store, pdf_library):
    report = ingest_module.ingest_directory(client, FakeEmbeddings(), settings)
    assert len(report.ingested) == 2


def test_ingest_directory_with_no_pdfs_returns_empty_report(client, settings, store):
    report = ingest_module.ingest_directory(client, FakeEmbeddings(), settings)
    assert report.files == []
    assert report.total_chunks == 0


def test_ingest_directory_honours_explicit_dir(client, settings, store, tmp_path: Path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    write_pdf(other / "extra.pdf", "Extra", ["An extra paper about caching strategies."])
    report = ingest_module.ingest_directory(
        client, FakeEmbeddings(), settings, data_dir=other
    )
    assert [r.path.name for r in report.ingested] == ["extra.pdf"]


def test_progress_callback_receives_messages(client, settings, store, sample_pdf):
    messages: list[str] = []
    _run([sample_pdf], client, settings, on_progress=messages.append)
    assert any("Chunking" in m for m in messages)
    assert any("Embedding" in m for m in messages)
