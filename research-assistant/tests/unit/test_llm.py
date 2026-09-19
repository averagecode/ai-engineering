"""Unit tests for Ollama model wiring and preflight checks."""

from __future__ import annotations

import pytest

from research_assistant import llm as llm_module
from research_assistant.llm import (
    OllamaUnavailableError,
    _model_present,
    build_chat_model,
    build_embeddings,
    list_available_models,
    verify_models,
)


class _Response:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


def _patch_tags(mocker, payload: dict | None = None, error: Exception | None = None):
    import httpx

    if error is not None:
        return mocker.patch.object(httpx, "get", side_effect=error)
    return mocker.patch.object(httpx, "get", return_value=_Response(payload or {}))


# --- model construction ----------------------------------------------------- #


def test_build_embeddings_points_at_configured_ollama(settings):
    embeddings = build_embeddings(settings)
    assert embeddings.model == settings.embedding_model
    assert embeddings.base_url == settings.ollama_base_url


def test_build_chat_model_uses_configured_model_and_temperature(settings):
    settings = settings.model_copy(update={"llm_model": "qwen2.5:7b", "llm_temperature": 0.3})
    chat = build_chat_model(settings)
    assert chat.model == "qwen2.5:7b"
    assert chat.temperature == 0.3
    assert chat.base_url == settings.ollama_base_url


def test_models_need_no_api_key(settings, monkeypatch: pytest.MonkeyPatch):
    """Construction must not depend on any key being present in the environment."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "WEAVIATE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert build_chat_model(settings) is not None
    assert build_embeddings(settings) is not None


# --- listing ---------------------------------------------------------------- #


def test_list_available_models_parses_tag_response(settings, mocker):
    _patch_tags(mocker, {"models": [{"name": "llama3.1:8b"}, {"name": "nomic-embed-text:latest"}]})
    assert list_available_models(settings) == ["llama3.1:8b", "nomic-embed-text:latest"]


def test_list_available_models_on_empty_server(settings, mocker):
    _patch_tags(mocker, {"models": []})
    assert list_available_models(settings) == []


def test_list_available_models_wraps_connection_errors(settings, mocker):
    _patch_tags(mocker, error=OSError("connection refused"))
    with pytest.raises(OllamaUnavailableError, match=settings.ollama_base_url):
        list_available_models(settings)


def test_list_available_models_wraps_http_errors(settings, mocker):
    import httpx

    mocker.patch.object(httpx, "get", return_value=_Response({}, status=500))
    with pytest.raises(OllamaUnavailableError):
        list_available_models(settings)


# --- tag matching ----------------------------------------------------------- #


@pytest.mark.parametrize(
    ("required", "available", "expected"),
    [
        ("llama3.1:8b", ["llama3.1:8b"], True),
        ("nomic-embed-text", ["nomic-embed-text:latest"], True),
        ("nomic-embed-text:latest", ["nomic-embed-text"], True),
        ("llama3.1:8b", ["llama3.2:3b"], False),
        ("llama3.1:8b", [], False),
    ],
)
def test_model_present_tolerates_missing_tags(required, available, expected):
    assert _model_present(required, available) is expected


# --- preflight -------------------------------------------------------------- #


def test_verify_models_passes_when_both_present(settings, mocker):
    _patch_tags(
        mocker,
        {"models": [{"name": settings.llm_model}, {"name": settings.embedding_model}]},
    )
    verify_models(settings)  # must not raise


def test_verify_models_lists_pull_commands_for_missing_models(settings, mocker):
    _patch_tags(mocker, {"models": [{"name": settings.llm_model}]})
    with pytest.raises(OllamaUnavailableError) as excinfo:
        verify_models(settings)
    message = str(excinfo.value)
    assert settings.embedding_model in message
    assert f"ollama pull {settings.embedding_model}" in message
    # The model that IS present must not be listed as missing.
    assert f"ollama pull {settings.llm_model}" not in message


def test_verify_models_reports_both_when_nothing_is_pulled(settings, mocker):
    _patch_tags(mocker, {"models": []})
    with pytest.raises(OllamaUnavailableError) as excinfo:
        verify_models(settings)
    assert str(excinfo.value).count("ollama pull") == 2


def test_verify_models_propagates_server_down(settings, mocker):
    mocker.patch.object(llm_module, "list_available_models", side_effect=OllamaUnavailableError("down"))
    with pytest.raises(OllamaUnavailableError, match="down"):
        verify_models(settings)


# --- cleanup ----------------------------------------------------------------- #


def test_close_model_closes_the_underlying_http_client():
    class Inner:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Model:
        def __init__(self):
            self._client = type("C", (), {"_client": Inner()})()

    model = Model()
    llm_module.close_model(model)
    assert model._client._client.closed is True


def test_close_model_tolerates_unfamiliar_internals():
    """A langchain-ollama version without the expected attributes must not crash."""
    for model in (object(), type("M", (), {"_client": None})()):
        llm_module.close_model(model)  # must not raise


def test_close_model_skips_async_clients():
    """Calling an async close would return an un-awaited coroutine, not close it."""

    class AsyncInner:
        async def close(self):  # pragma: no cover - must never be called
            raise AssertionError("async close should not be invoked")

    model = type("M", (), {"_client": type("C", (), {"_client": AsyncInner()})()})()
    llm_module.close_model(model)  # must not raise or warn


def test_close_model_swallows_close_errors():
    class Boom:
        def close(self):
            raise RuntimeError("already shut down")

    model = type("M", (), {"_client": type("C", (), {"_client": Boom()})()})()
    llm_module.close_model(model)  # must not propagate
