"""Unit tests for Weaviate collection management, against a fake v4 client."""

from __future__ import annotations

import pytest

from research_assistant import vectorstore as vs
from tests.fakes import FakeEmbeddings, FakeWeaviateClient


@pytest.fixture
def client() -> FakeWeaviateClient:
    return FakeWeaviateClient()


ROWS = [
    {"text": "chunk one", "source": "a.pdf", "page": 1, "checksum": "aaa"},
    {"text": "chunk two", "source": "a.pdf", "page": 2, "checksum": "aaa"},
    {"text": "chunk three", "source": "b.pdf", "page": 1, "checksum": "bbb"},
]


# --- collection lifecycle --------------------------------------------------- #


def test_ensure_collection_creates_once(client: FakeWeaviateClient):
    vs.ensure_collection(client, "Chunk")
    vs.ensure_collection(client, "Chunk")
    assert client.collections.created == ["Chunk"]


def test_delete_collection_reports_whether_it_existed(client: FakeWeaviateClient):
    assert vs.delete_collection(client, "Chunk") is False
    vs.ensure_collection(client, "Chunk")
    assert vs.delete_collection(client, "Chunk") is True
    assert client.collections.exists("Chunk") is False


# --- checksums -------------------------------------------------------------- #


def test_stored_checksum_returns_none_without_collection(client: FakeWeaviateClient):
    assert vs.stored_checksum(client, "Chunk", "a.pdf") is None


def test_stored_checksum_returns_none_for_unknown_source(client: FakeWeaviateClient):
    client.seed("Chunk", ROWS)
    assert vs.stored_checksum(client, "Chunk", "missing.pdf") is None


def test_stored_checksum_finds_recorded_value(client: FakeWeaviateClient):
    client.seed("Chunk", ROWS)
    assert vs.stored_checksum(client, "Chunk", "a.pdf") == "aaa"
    assert vs.stored_checksum(client, "Chunk", "b.pdf") == "bbb"


# --- deletion --------------------------------------------------------------- #


def test_delete_source_removes_only_that_source(client: FakeWeaviateClient):
    collection = client.seed("Chunk", ROWS)
    assert vs.delete_source(client, "Chunk", "a.pdf") == 2
    assert [obj.properties["source"] for obj in collection.objects] == ["b.pdf"]


def test_delete_source_is_a_noop_when_absent(client: FakeWeaviateClient):
    client.seed("Chunk", ROWS)
    assert vs.delete_source(client, "Chunk", "ghost.pdf") == 0
    assert vs.delete_source(client, "Other", "a.pdf") == 0


# --- stats ------------------------------------------------------------------ #


def test_collection_stats_when_absent(client: FakeWeaviateClient):
    stats = vs.collection_stats(client, "Chunk")
    assert stats == {"exists": False, "chunk_count": 0, "sources": {}}


def test_collection_stats_groups_by_source(client: FakeWeaviateClient):
    client.seed("Chunk", ROWS)
    stats = vs.collection_stats(client, "Chunk")
    assert stats["exists"] is True
    assert stats["chunk_count"] == 3
    assert stats["sources"] == {"a.pdf": 2, "b.pdf": 1}


def test_collection_stats_sources_are_sorted(client: FakeWeaviateClient):
    client.seed("Chunk", [{"source": name} for name in ["z.pdf", "a.pdf", "m.pdf"]])
    assert list(vs.collection_stats(client, "Chunk")["sources"]) == ["a.pdf", "m.pdf", "z.pdf"]


# --- vector store wiring ---------------------------------------------------- #


def test_get_vector_store_creates_collection_and_binds_embeddings(client, settings, mocker):
    captured = {}

    class DummyStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    mocker.patch("langchain_weaviate.WeaviateVectorStore", DummyStore)
    embeddings = FakeEmbeddings()

    store = vs.get_vector_store(client, embeddings, settings)

    assert isinstance(store, DummyStore)
    assert client.collections.created == [settings.collection_name]
    assert captured["index_name"] == settings.collection_name
    assert captured["text_key"] == vs.TEXT_KEY
    assert captured["embedding"] is embeddings
    assert vs.TEXT_KEY not in captured["attributes"]
    assert "source" in captured["attributes"] and "page" in captured["attributes"]


# --- connection errors ------------------------------------------------------ #


def test_connect_wraps_failures_with_guidance(settings, mocker):
    mocker.patch("weaviate.WeaviateClient", side_effect=OSError("connection refused"))
    with pytest.raises(vs.WeaviateUnavailableError) as excinfo:
        vs.connect(settings)
    message = str(excinfo.value)
    assert settings.weaviate_url in message
    assert "docker compose up" in message


def test_weaviate_client_context_manager_always_closes(settings, mocker):
    fake = FakeWeaviateClient()
    mocker.patch.object(vs, "connect", return_value=fake)

    with pytest.raises(RuntimeError):
        with vs.weaviate_client(settings):
            raise RuntimeError("boom")
    assert fake.closed is True
