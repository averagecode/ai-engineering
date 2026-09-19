"""Ingestion pipeline: PDFs on disk -> chunks -> embeddings -> Weaviate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from langchain_core.documents import Document

from .config import Settings, get_settings
from .loader import chunk_documents, discover_pdfs, file_checksum, load_pdf
from . import vectorstore as vs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.embeddings import Embeddings
    from weaviate import WeaviateClient

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]


@dataclass
class FileResult:
    """Outcome of ingesting a single PDF."""

    path: Path
    chunks: int = 0
    status: str = "ingested"  # ingested | skipped | empty | failed
    detail: str = ""


@dataclass
class IngestReport:
    """Aggregate outcome of an ingestion run."""

    files: list[FileResult] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(f.chunks for f in self.files)

    @property
    def ingested(self) -> list[FileResult]:
        return [f for f in self.files if f.status == "ingested"]

    @property
    def skipped(self) -> list[FileResult]:
        return [f for f in self.files if f.status == "skipped"]

    @property
    def failed(self) -> list[FileResult]:
        return [f for f in self.files if f.status == "failed"]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{len(self.ingested)} ingested, {len(self.skipped)} skipped, "
            f"{len(self.failed)} failed, {self.total_chunks} chunks"
        )


def prepare_chunks(
    path: Path, *, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Load and chunk a single PDF (thin seam for testing and reuse)."""
    pages = load_pdf(path)
    return chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def ingest_paths(
    paths: Sequence[Path],
    client: "WeaviateClient",
    embeddings: "Embeddings",
    settings: Settings | None = None,
    *,
    force: bool = False,
    batch_size: int = 64,
    on_progress: ProgressCallback | None = None,
) -> IngestReport:
    """Embed and upsert the given PDFs.

    A file whose checksum already matches what Weaviate holds is skipped unless
    `force` is set; a changed file has its old chunks deleted before re-insert,
    so re-running ingestion is always safe.
    """
    settings = settings or get_settings()
    report = IngestReport()
    store = vs.get_vector_store(client, embeddings, settings)

    def notify(message: str) -> None:
        logger.info(message)
        if on_progress is not None:
            on_progress(message)

    for path in paths:
        path = Path(path)
        result = FileResult(path=path)
        try:
            checksum = file_checksum(path)
            existing = vs.stored_checksum(client, settings.collection_name, path.name)

            if existing == checksum and not force:
                result.status = "skipped"
                result.detail = "unchanged since last ingestion"
                notify(f"Skipping {path.name} (unchanged)")
                report.files.append(result)
                continue

            if existing is not None:
                removed = vs.delete_source(client, settings.collection_name, path.name)
                notify(f"Replacing {path.name} ({removed} stale chunks removed)")

            notify(f"Chunking {path.name}")
            chunks = prepare_chunks(
                path,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            if not chunks:
                result.status = "empty"
                result.detail = "no extractable text (scanned PDF?)"
                notify(f"No text extracted from {path.name}")
                report.files.append(result)
                continue

            notify(f"Embedding {len(chunks)} chunks from {path.name}")
            for start in range(0, len(chunks), batch_size):
                store.add_documents(chunks[start : start + batch_size])

            result.chunks = len(chunks)
            notify(f"Stored {len(chunks)} chunks from {path.name}")
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the run
            logger.exception("Failed to ingest %s", path)
            result.status = "failed"
            result.detail = str(exc)
        report.files.append(result)

    return report


def ingest_directory(
    client: "WeaviateClient",
    embeddings: "Embeddings",
    settings: Settings | None = None,
    *,
    data_dir: Path | None = None,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> IngestReport:
    """Ingest every PDF found under the configured data directory."""
    settings = settings or get_settings()
    directory = Path(data_dir) if data_dir else settings.data_dir
    pdfs = discover_pdfs(directory)
    if not pdfs:
        logger.warning("No PDFs found in %s", directory)
        return IngestReport()
    return ingest_paths(
        pdfs, client, embeddings, settings, force=force, on_progress=on_progress
    )
