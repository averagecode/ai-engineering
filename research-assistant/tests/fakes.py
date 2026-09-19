"""In-memory stand-ins for Weaviate and Ollama, used by the unit tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeEmbeddings(Embeddings):
    """Deterministic hash-based embeddings: no network, stable across runs."""

    def __init__(self, dimensions: int = 16) -> None:
        self.dimensions = dimensions
        self.embed_calls: list[str] = []

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dimensions)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return self._vector(text)


@dataclass
class FakeStore:
    """Minimal `WeaviateVectorStore` surface used by the ingest and agent code."""

    documents: list[Document] = field(default_factory=list)
    search_results: list[Document] = field(default_factory=list)
    add_batches: list[int] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    mmr_supported: bool = True

    def add_documents(self, documents: Sequence[Document], **_: Any) -> list[str]:
        self.documents.extend(documents)
        self.add_batches.append(len(documents))
        return [str(i) for i in range(len(documents))]

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, **_: Any
    ) -> list[Document]:
        if not self.mmr_supported:
            raise RuntimeError("vectors not returned by this backend")
        self.queries.append(query)
        return self.search_results[:k]

    def similarity_search(self, query: str, k: int = 4, **_: Any) -> list[Document]:
        self.queries.append(query)
        return self.search_results[:k]


@dataclass
class _FakeObject:
    properties: dict[str, Any]


class _FakeAggregate:
    def __init__(self, objects: list[_FakeObject]) -> None:
        self._objects = objects

    def over_all(self, total_count: bool = False) -> Any:  # noqa: ARG002
        return type("Result", (), {"total_count": len(self._objects)})()


class _FakeQuery:
    def __init__(self, objects: list[_FakeObject]) -> None:
        self._objects = objects

    def fetch_objects(
        self,
        filters: Any = None,
        limit: int | None = None,
        return_properties: Sequence[str] | None = None,  # noqa: ARG002
    ) -> Any:
        matched = [obj for obj in self._objects if _matches(obj, filters)]
        if limit is not None:
            matched = matched[:limit]
        return type("Result", (), {"objects": matched})()


class _FakeData:
    def __init__(self, collection: "FakeCollection") -> None:
        self._collection = collection

    def delete_many(self, where: Any = None) -> Any:
        kept, removed = [], 0
        for obj in self._collection.objects:
            if _matches(obj, where):
                removed += 1
            else:
                kept.append(obj)
        self._collection.objects = kept
        return type("Result", (), {"successful": removed, "failed": 0})()


def _matches(obj: _FakeObject, filters: Any) -> bool:
    """Support the one filter shape the production code uses: property == value.

    Mirrors `weaviate.classes.query.Filter.by_property(t).equal(v)`, whose value
    object exposes `.target` and `.value`.
    """
    if filters is None:
        return True
    target = getattr(filters, "target", None)
    if target is None:
        return True
    return obj.properties.get(target) == getattr(filters, "value", None)


@dataclass
class FakeCollection:
    name: str
    objects: list[_FakeObject] = field(default_factory=list)

    @property
    def query(self) -> _FakeQuery:
        return _FakeQuery(self.objects)

    @property
    def aggregate(self) -> _FakeAggregate:
        return _FakeAggregate(self.objects)

    @property
    def data(self) -> _FakeData:
        return _FakeData(self)

    def iterator(self, return_properties: Sequence[str] | None = None) -> Iterable[_FakeObject]:  # noqa: ARG002
        return list(self.objects)

    def add(self, **properties: Any) -> None:
        self.objects.append(_FakeObject(properties))


class _FakeCollections:
    def __init__(self) -> None:
        self._collections: dict[str, FakeCollection] = {}
        self.created: list[str] = []
        self.deleted: list[str] = []

    def exists(self, name: str) -> bool:
        return name in self._collections

    def create(self, name: str, **_: Any) -> FakeCollection:
        collection = FakeCollection(name)
        self._collections[name] = collection
        self.created.append(name)
        return collection

    def get(self, name: str) -> FakeCollection:
        if name not in self._collections:
            raise KeyError(f"collection {name!r} does not exist")
        return self._collections[name]

    def delete(self, name: str) -> None:
        self._collections.pop(name, None)
        self.deleted.append(name)


class FakeWeaviateClient:
    """Enough of the v4 client surface for the vectorstore/ingest/agent units."""

    def __init__(self) -> None:
        self.collections = _FakeCollections()
        self.closed = False

    def seed(self, name: str, rows: Sequence[dict[str, Any]]) -> FakeCollection:
        collection = (
            self.collections.get(name)
            if self.collections.exists(name)
            else self.collections.create(name)
        )
        for row in rows:
            collection.add(**row)
        return collection

    def close(self) -> None:
        self.closed = True


class FakeAction:
    """Minimal `AgentAction` stand-in: just the tool name the agent invoked."""

    def __init__(self, tool: str) -> None:
        self.tool = tool


class FakeExecutor:
    """Stands in for `AgentExecutor`, recording the payloads it was invoked with."""

    def __init__(
        self,
        outputs: Sequence[str] | None = None,
        tools_called: Sequence[str] | None = None,
    ) -> None:
        self.outputs = list(outputs or ["a grounded answer"])
        self.tools_called = list(tools_called or [])
        self.calls: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({k: list(v) if isinstance(v, list) else v for k, v in payload.items()})
        output = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        steps = [(FakeAction(name), "observation") for name in self.tools_called]
        return {"output": output, "intermediate_steps": steps}


def make_chunk(source: str, page: int, text: str, index: int = 0) -> Document:
    """A chunk shaped the way the ingestion pipeline produces them."""
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "source_path": f"/data/{source}",
            "page": page,
            "chunk_index": index,
            "chunk_id": f"{source}-{index}",
            "checksum": "deadbeef",
        },
    )


class FakeToolCallingChatModel(BaseChatModel):
    """A scripted tool-calling chat model.

    Replays a fixed list of `AIMessage`s so the real `AgentExecutor` loop — tool
    call, observation, final answer — can be exercised without Ollama.
    """

    responses: list[AIMessage]
    calls: list[list[Any]] = []
    bound_tools: list[Any] = []

    def __init__(self, responses: Sequence[AIMessage], **kwargs: Any) -> None:
        super().__init__(responses=list(responses), calls=[], bound_tools=[], **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolCallingChatModel":
        self.bound_tools.extend(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return ChatResult(generations=[ChatGeneration(message=self.responses[index])])
