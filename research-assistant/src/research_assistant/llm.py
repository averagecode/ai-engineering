"""Ollama-backed models.

Ollama runs entirely on the local machine, so inference and embeddings cost
nothing and need no API key. The helpers here also verify that the requested
models are actually pulled, which is the most common first-run failure.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama server cannot be reached or a model is missing."""


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Embedding model used for both ingestion and query encoding."""
    from langchain_ollama import OllamaEmbeddings

    settings = settings or get_settings()
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Chat model that drives the agent loop (needs tool-calling support)."""
    from langchain_ollama import ChatOllama

    settings = settings or get_settings()
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        client_kwargs={"timeout": settings.request_timeout},
    )


def close_model(model: object) -> None:
    """Release the HTTP connections a ChatOllama / OllamaEmbeddings holds.

    langchain-ollama exposes no public `close`, so this reaches for the
    underlying synchronous httpx client defensively. If a future version
    restructures its internals, the sockets are simply left to interpreter
    shutdown, which is the behaviour we had before.
    """
    import inspect

    inner = getattr(getattr(model, "_client", None), "_client", None)
    close = getattr(inner, "close", None)
    if callable(close) and not inspect.iscoroutinefunction(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - cleanup must never fail a command
            logger.debug("Could not close HTTP client for %s", type(model).__name__, exc_info=True)


def list_available_models(settings: Settings | None = None) -> list[str]:
    """Names of models pulled into the local Ollama server."""
    import httpx

    settings = settings or get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly error
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {settings.ollama_base_url}: {exc}"
        ) from exc
    return [model["name"] for model in response.json().get("models", [])]


def _model_present(required: str, available: list[str]) -> bool:
    """Match a model name, tolerating an omitted `:latest` tag."""
    if required in available:
        return True
    base = required.split(":", 1)[0]
    return any(name == required or name.split(":", 1)[0] == base for name in available)


def verify_models(settings: Settings | None = None) -> None:
    """Fail fast with actionable instructions if a required model is missing."""
    settings = settings or get_settings()
    available = list_available_models(settings)
    missing = [
        model
        for model in (settings.llm_model, settings.embedding_model)
        if not _model_present(model, available)
    ]
    if missing:
        pulls = "\n".join(f"  ollama pull {model}" for model in missing)
        raise OllamaUnavailableError(
            "Missing Ollama model(s): " + ", ".join(missing) + "\nPull them with:\n" + pulls
        )
    logger.debug("Ollama models available: %s", available)
