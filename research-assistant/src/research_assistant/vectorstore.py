"""Weaviate connection and collection management.

Vectors are produced client-side by Ollama embeddings, so the collection is
created with self-provided vectors and no server-side vectoriser module. That
keeps Weaviate a pure vector index and leaves model choice to LangChain.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from .config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.embeddings import Embeddings
    from langchain_weaviate import WeaviateVectorStore
    from weaviate import WeaviateClient

logger = logging.getLogger(__name__)

TEXT_KEY = "text"
#: Chunk metadata persisted alongside the text, and returned on retrieval.
METADATA_FIELDS = ("source", "source_path", "page", "chunk_index", "chunk_id", "checksum")


class WeaviateUnavailableError(RuntimeError):
    """Raised when the Weaviate instance cannot be reached."""


def connect(settings: Settings | None = None) -> "WeaviateClient":
    """Open a Weaviate v4 client against the configured host."""
    import weaviate
    from weaviate.connect import ConnectionParams

    settings = settings or get_settings()
    try:
        client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_params(
                http_host=settings.weaviate_host,
                http_port=settings.weaviate_http_port,
                http_secure=settings.weaviate_secure,
                grpc_host=settings.weaviate_host,
                grpc_port=settings.weaviate_grpc_port,
                grpc_secure=settings.weaviate_secure,
            ),
            skip_init_checks=False,
        )
        client.connect()
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly error
        raise WeaviateUnavailableError(
            f"Could not connect to Weaviate at {settings.weaviate_url}: {exc}\n"
            "Start it with: docker compose up -d weaviate"
        ) from exc
    return client


@contextmanager
def weaviate_client(settings: Settings | None = None) -> Iterator["WeaviateClient"]:
    """Context manager that always closes the underlying gRPC/HTTP pools."""
    client = connect(settings)
    try:
        yield client
    finally:
        client.close()


def ensure_collection(client: "WeaviateClient", name: str) -> None:
    """Create the chunk collection if it does not exist yet (idempotent)."""
    from weaviate.classes.config import Configure, DataType, Property

    if client.collections.exists(name):
        return

    logger.info("Creating Weaviate collection %s", name)
    client.collections.create(
        name=name,
        description="Chunks of research papers, embedded locally with Ollama.",
        # Vectors are supplied by the client (Ollama embeddings via LangChain),
        # so Weaviate must not try to vectorise anything itself.
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name=TEXT_KEY, data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="source_path", data_type=DataType.TEXT, index_searchable=False),
            Property(name="page", data_type=DataType.INT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="chunk_id", data_type=DataType.TEXT, index_searchable=False),
            Property(name="checksum", data_type=DataType.TEXT, index_searchable=False),
        ],
    )


def delete_collection(client: "WeaviateClient", name: str) -> bool:
    """Drop the collection. Returns True if something was deleted."""
    if not client.collections.exists(name):
        return False
    client.collections.delete(name)
    logger.info("Deleted Weaviate collection %s", name)
    return True


def get_vector_store(
    client: "WeaviateClient",
    embeddings: "Embeddings",
    settings: Settings | None = None,
) -> "WeaviateVectorStore":
    """LangChain vector store bound to the chunk collection."""
    from langchain_weaviate import WeaviateVectorStore

    settings = settings or get_settings()
    ensure_collection(client, settings.collection_name)
    return WeaviateVectorStore(
        client=client,
        index_name=settings.collection_name,
        text_key=TEXT_KEY,
        embedding=embeddings,
        attributes=[field for field in METADATA_FIELDS if field != TEXT_KEY],
    )


def stored_checksum(client: "WeaviateClient", collection_name: str, source: str) -> str | None:
    """Checksum recorded for an already-ingested source file, if any."""
    from weaviate.classes.query import Filter

    if not client.collections.exists(collection_name):
        return None
    collection = client.collections.get(collection_name)
    result = collection.query.fetch_objects(
        filters=Filter.by_property("source").equal(source),
        limit=1,
        return_properties=["checksum"],
    )
    if not result.objects:
        return None
    return result.objects[0].properties.get("checksum")


def delete_source(client: "WeaviateClient", collection_name: str, source: str) -> int:
    """Remove every chunk belonging to one source file. Returns chunks deleted."""
    from weaviate.classes.query import Filter

    if not client.collections.exists(collection_name):
        return 0
    collection = client.collections.get(collection_name)
    result = collection.data.delete_many(
        where=Filter.by_property("source").equal(source)
    )
    return int(getattr(result, "successful", 0) or 0)


def collection_stats(client: "WeaviateClient", collection_name: str) -> dict:
    """Chunk count and per-source breakdown, for `research-assistant status`."""
    if not client.collections.exists(collection_name):
        return {"exists": False, "chunk_count": 0, "sources": {}}

    collection = client.collections.get(collection_name)
    total = collection.aggregate.over_all(total_count=True).total_count or 0

    sources: dict[str, int] = {}
    for obj in collection.iterator(return_properties=["source"]):
        source = str(obj.properties.get("source", "unknown"))
        sources[source] = sources.get(source, 0) + 1

    return {"exists": True, "chunk_count": total, "sources": dict(sorted(sources.items()))}
