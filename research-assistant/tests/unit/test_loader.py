"""Unit tests for PDF discovery, text cleaning and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from research_assistant.loader import (
    chunk_documents,
    clean_text,
    discover_pdfs,
    file_checksum,
    load_and_chunk,
    load_pdf,
)
from tests.conftest import write_pdf


# --- discovery -------------------------------------------------------------- #


def test_discover_pdfs_finds_only_pdfs_sorted(settings, pdf_library):
    found = discover_pdfs(settings.data_dir)
    assert [p.name for p in found] == ["retrieval_depth.pdf", "sparse_routing.pdf"]


def test_discover_pdfs_is_recursive_and_case_insensitive(settings):
    nested = settings.data_dir / "2024"
    nested.mkdir()
    write_pdf(nested / "Deep.PDF", "Nested", ["Nested paper body text."])
    assert [p.name for p in discover_pdfs(settings.data_dir)] == ["Deep.PDF"]


def test_discover_pdfs_on_empty_dir(settings):
    assert discover_pdfs(settings.data_dir) == []


def test_discover_pdfs_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_pdfs(tmp_path / "nope")


# --- checksums -------------------------------------------------------------- #


def test_checksum_is_stable_and_content_sensitive(tmp_path: Path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"same"), b.write_bytes(b"same")
    assert file_checksum(a) == file_checksum(b) == file_checksum(a)
    b.write_bytes(b"different")
    assert file_checksum(a) != file_checksum(b)


# --- cleaning --------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  spaced   out  ", "spaced out"),
        ("tabs\t\tcollapse", "tabs collapse"),
        ("a\n\n\n\n\nb", "a\n\nb"),
        ("distrib-\nuted systems", "distributed systems"),
        ("null\x00byte", "nullbyte"),
        ("", ""),
    ],
)
def test_clean_text(raw: str, expected: str):
    assert clean_text(raw) == expected


def test_clean_text_preserves_paragraph_breaks():
    assert clean_text("First para.\n\nSecond para.") == "First para.\n\nSecond para."


# --- loading ---------------------------------------------------------------- #


def test_load_pdf_yields_one_document_per_page_with_metadata(sample_pdf: Path):
    documents = load_pdf(sample_pdf)
    assert len(documents) == 3
    assert [d.metadata["page"] for d in documents] == [1, 2, 3]
    assert all(d.metadata["source"] == "sparse_routing.pdf" for d in documents)
    assert all(d.metadata["source_path"] == str(sample_pdf) for d in documents)
    assert len({d.metadata["checksum"] for d in documents}) == 1
    assert "Gated Sparse Routing" in documents[0].page_content


def test_load_pdf_drops_pages_without_text(monkeypatch: pytest.MonkeyPatch, sample_pdf: Path):
    """A scanned page extracts as empty and must not become a document."""
    import pypdf

    real_reader = pypdf.PdfReader

    class BlankSecondPage:
        def __init__(self, path):
            self.pages = list(real_reader(path).pages)
            self.pages[1] = type("P", (), {"extract_text": lambda self: "   "})()

    monkeypatch.setattr(pypdf, "PdfReader", BlankSecondPage)
    documents = load_pdf(sample_pdf)
    assert [d.metadata["page"] for d in documents] == [1, 3]


def test_load_pdf_missing_file_raises(tmp_path: Path):
    with pytest.raises(Exception):
        load_pdf(tmp_path / "absent.pdf")


# --- chunking --------------------------------------------------------------- #


def _page(text: str, page: int = 1) -> Document:
    return Document(
        page_content=text,
        metadata={"source": "paper.pdf", "page": page, "checksum": "abc"},
    )


def test_chunk_documents_respects_chunk_size():
    chunks = chunk_documents([_page("word " * 400)], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_chunk_documents_propagates_and_extends_metadata():
    chunks = chunk_documents([_page("sentence. " * 100, page=7)], chunk_size=150, chunk_overlap=10)
    for expected_index, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "paper.pdf"
        assert chunk.metadata["page"] == 7
        assert chunk.metadata["checksum"] == "abc"
        assert chunk.metadata["chunk_index"] == expected_index
        assert len(chunk.metadata["chunk_id"]) == 32
        assert "start_index" in chunk.metadata


def test_chunk_ids_are_unique_and_deterministic():
    pages = [_page("alpha " * 120, 1), _page("beta " * 120, 2)]
    first = chunk_documents(pages, chunk_size=150, chunk_overlap=10)
    second = chunk_documents(pages, chunk_size=150, chunk_overlap=10)
    ids = [c.metadata["chunk_id"] for c in first]
    assert len(set(ids)) == len(ids)
    assert ids == [c.metadata["chunk_id"] for c in second]


def test_chunk_index_counts_per_source():
    pages = [
        Document(page_content="x " * 200, metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="y " * 200, metadata={"source": "b.pdf", "page": 1}),
    ]
    chunks = chunk_documents(pages, chunk_size=150, chunk_overlap=0)
    for source in ("a.pdf", "b.pdf"):
        indexes = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == source]
        assert indexes == list(range(len(indexes)))


def test_chunk_documents_on_empty_input():
    assert chunk_documents([], chunk_size=100, chunk_overlap=0) == []


def test_short_page_becomes_single_chunk():
    chunks = chunk_documents([_page("Short abstract.")], chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short abstract."


def test_load_and_chunk_covers_multiple_pdfs(pdf_library):
    chunks = load_and_chunk(pdf_library, chunk_size=300, chunk_overlap=30)
    sources = {c.metadata["source"] for c in chunks}
    assert sources == {"sparse_routing.pdf", "retrieval_depth.pdf"}
    assert len(chunks) >= len(pdf_library)


def test_load_pdf_returns_empty_for_a_pdf_with_no_text_layer(
    monkeypatch: pytest.MonkeyPatch, sample_pdf: Path
):
    """A fully scanned PDF yields no documents rather than empty ones."""
    import pypdf

    class ScannedPages:
        def __init__(self, path):  # noqa: ARG002
            self.pages = [type("P", (), {"extract_text": lambda self: ""})() for _ in range(3)]

    monkeypatch.setattr(pypdf, "PdfReader", ScannedPages)
    assert load_pdf(sample_pdf) == []
