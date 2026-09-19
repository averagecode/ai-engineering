"""Unit tests for settings loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from research_assistant.config import PROJECT_ROOT, Settings, get_settings


def test_defaults_are_local_only():
    # `_env_file=None` so a developer's local .env cannot influence the result;
    # RA_* variables are already stripped by the hermetic-environment fixture.
    settings = Settings(_env_file=None)
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.weaviate_host == "localhost"
    assert settings.data_dir == PROJECT_ROOT / "data"
    # Nothing in the defaults should imply a hosted, keyed service.
    assert "api" not in settings.ollama_base_url


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RA_LLM_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("RA_WEAVIATE_HOST", "weaviate")
    monkeypatch.setenv("RA_CHUNK_SIZE", "900")
    settings = Settings(_env_file=None)
    assert settings.llm_model == "qwen2.5:7b"
    assert settings.weaviate_host == "weaviate"
    assert settings.chunk_size == 900


def test_weaviate_url_respects_scheme():
    assert Settings(weaviate_host="db", weaviate_http_port=9000).weaviate_url == "http://db:9000"
    assert Settings(weaviate_secure=True).weaviate_url.startswith("https://")


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="chunk_overlap must be smaller"):
        Settings(chunk_size=100, chunk_overlap=100)


@pytest.mark.parametrize(
    ("field", "value"),
    [("chunk_size", 0), ("retrieval_k", 0), ("request_timeout", 0), ("llm_temperature", 3.0)],
)
def test_out_of_range_values_rejected(field: str, value: object):
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_data_dir_is_expanded():
    assert not str(Settings(data_dir=Path("~/papers")).data_dir).startswith("~")


def test_get_settings_is_cached():
    get_settings.cache_clear()
    assert get_settings() is get_settings()
