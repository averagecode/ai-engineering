"""Shared fixtures.

Unit tests use fakes for Weaviate and Ollama so the suite runs offline in
milliseconds. Integration tests are marked `integration` and skip themselves
unless the real services are reachable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

from research_assistant.config import Settings, get_settings

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Keep the cached settings singleton from leaking between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Small, fast settings pointed at a temp data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return Settings(
        data_dir=data_dir,
        chunk_size=200,
        chunk_overlap=20,
        retrieval_k=3,
        retrieval_fetch_k=6,
        collection_name="TestChunk",
    )


# --------------------------------------------------------------------------- #
# Sample PDFs
# --------------------------------------------------------------------------- #

PAPER_TEXT = [
    (
        "Sparse Mixture Routing for Efficient Transformers",
        [
            "Abstract. We introduce Gated Sparse Routing (GSR), a mixture-of-experts "
            "router that activates two of thirty-two experts per token. GSR reduces "
            "inference FLOPs by 41 percent on the WikiText-103 benchmark while "
            "matching the perplexity of a dense baseline of equal parameter count.",
            "Method. The router is a single linear projection followed by a top-k "
            "selection with a load-balancing auxiliary loss weighted at 0.01. We train "
            "for 120000 steps with the AdamW optimiser and a peak learning rate of "
            "3e-4 under a cosine decay schedule.",
            "Results. On WikiText-103 GSR reaches a perplexity of 17.4 against 17.6 for "
            "the dense baseline. Throughput improves from 1840 to 3110 tokens per "
            "second on a single A100 GPU. Ablations show that removing the "
            "load-balancing loss collapses routing onto four experts.",
        ],
    ),
    (
        "Retrieval Depth and Hallucination Rate",
        [
            "Abstract. We measure how the number of retrieved passages affects the "
            "hallucination rate of retrieval-augmented generation. Across three "
            "question-answering datasets, hallucinations fall steeply from one to five "
            "passages and then plateau.",
            "Findings. Moving from one to five retrieved passages lowers the "
            "hallucination rate from 22 percent to 9 percent. Beyond ten passages the "
            "rate rises again to 12 percent, which we attribute to distraction by "
            "loosely related context.",
        ],
    ),
]


def write_pdf(path: Path, title: str, paragraphs: list[str]) -> Path:
    """Render a small text-only PDF, one paragraph block per page."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, title=title)
    story: list = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for index, text in enumerate(paragraphs):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(text, styles["BodyText"]))
    doc.build(story)
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A single three-page PDF with known content."""
    title, paragraphs = PAPER_TEXT[0]
    return write_pdf(tmp_path / "sparse_routing.pdf", title, paragraphs)


@pytest.fixture
def pdf_library(settings: Settings) -> list[Path]:
    """Two PDFs inside `settings.data_dir`, plus a non-PDF that must be ignored."""
    paths = []
    for (title, paragraphs), name in zip(
        PAPER_TEXT, ["sparse_routing.pdf", "retrieval_depth.pdf"]
    ):
        paths.append(write_pdf(settings.data_dir / name, title, paragraphs))
    (settings.data_dir / "notes.txt").write_text("not a pdf", encoding="utf-8")
    return paths


# --------------------------------------------------------------------------- #
# Integration service gating
# --------------------------------------------------------------------------- #


def _weaviate_reachable(settings: Settings) -> bool:
    import httpx

    try:
        response = httpx.get(
            f"{settings.weaviate_url}/v1/.well-known/ready", timeout=3.0
        )
        return response.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _ollama_models(settings: Settings) -> list[str] | None:
    from research_assistant.llm import OllamaUnavailableError, list_available_models

    try:
        return list_available_models(settings)
    except OllamaUnavailableError:
        return None


@pytest.fixture(scope="session")
def live_settings() -> Settings:
    """Settings for integration tests, isolated to its own collection."""
    return Settings(
        collection_name=os.environ.get("RA_TEST_COLLECTION", "IntegrationTestChunk"),
        chunk_size=500,
        chunk_overlap=50,
        retrieval_k=4,
        retrieval_fetch_k=8,
    )


@pytest.fixture(scope="session")
def live_services(live_settings: Settings) -> Settings:
    """Skip the test unless Weaviate is up and both Ollama models are pulled."""
    from research_assistant.llm import _model_present

    if not _weaviate_reachable(live_settings):
        pytest.skip(
            f"Weaviate not reachable at {live_settings.weaviate_url} "
            "(start it with: docker compose up -d weaviate)"
        )
    models = _ollama_models(live_settings)
    if models is None:
        pytest.skip(
            f"Ollama not reachable at {live_settings.ollama_base_url} "
            "(start it with: docker compose up -d ollama)"
        )
    missing = [
        model
        for model in (live_settings.llm_model, live_settings.embedding_model)
        if not _model_present(model, models)
    ]
    if missing:
        pytest.skip("Ollama models not pulled: " + ", ".join(missing))
    return live_settings


@pytest.fixture
def live_client(live_services: Settings):
    """Connected Weaviate client whose test collection is dropped on teardown."""
    from research_assistant import vectorstore as vs

    client = vs.connect(live_services)
    vs.delete_collection(client, live_services.collection_name)
    try:
        yield client
    finally:
        try:
            vs.delete_collection(client, live_services.collection_name)
        finally:
            client.close()
