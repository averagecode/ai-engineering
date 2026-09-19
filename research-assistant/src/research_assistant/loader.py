"""PDF discovery, loading and chunking.

Pure functions over the filesystem and LangChain `Document` objects — no network
calls live here, which keeps this layer fast to unit test.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# Hyphen at a line break, e.g. "distrib-\nuted" -> "distributed".
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def discover_pdfs(data_dir: Path | str) -> list[Path]:
    """Return every PDF under `data_dir`, sorted for deterministic ingestion."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {directory}")
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() == ".pdf" and p.is_file())


def file_checksum(path: Path | str, *, block_size: int = 1 << 20) -> str:
    """SHA-256 of a file, used to detect re-ingestion of unchanged papers."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    """Normalise whitespace artefacts typical of PDF text extraction."""
    text = text.replace("\x00", "")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def load_pdf(path: Path | str) -> list[Document]:
    """Load one PDF into a `Document` per page, with cleaned text and metadata.

    Pages whose text is empty after cleaning (e.g. scanned images) are dropped.
    """
    from pypdf import PdfReader  # imported lazily so unit tests can stub it

    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    checksum = file_checksum(pdf_path)

    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            logger.debug("Skipping empty page %s of %s", page_number, pdf_path.name)
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page": page_number,
                    "checksum": checksum,
                },
            )
        )

    if not documents:
        logger.warning("No extractable text found in %s (scanned PDF?)", pdf_path.name)
    return documents


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Text splitter tuned for academic prose: prefer paragraph, then sentence."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split page documents into embedding-sized chunks.

    Each chunk keeps its parent's metadata and gains `chunk_index` (position
    within the source document) and `chunk_id` (stable, content-addressed).
    """
    if not documents:
        return []

    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(list(documents))

    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        index = per_source_counter.get(source, 0)
        per_source_counter[source] = index + 1
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = _chunk_id(source, index, chunk.page_content)
    return chunks


def _chunk_id(source: str, index: int, content: str) -> str:
    payload = f"{source}:{index}:{content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def load_and_chunk(
    paths: Iterable[Path | str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Convenience pipeline: load each PDF, then chunk it."""
    all_chunks: list[Document] = []
    for path in paths:
        pages = load_pdf(path)
        all_chunks.extend(
            chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    return all_chunks
