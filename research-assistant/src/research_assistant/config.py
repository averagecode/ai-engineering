"""Central configuration.

Every value has a local-only default, so the app runs with no API keys and no
paid services. Override anything through environment variables (see .env.example).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (…/research-assistant), used to anchor relative paths.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for the research assistant."""

    model_config = SettingsConfigDict(
        env_prefix="RA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths -------------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")

    # --- Ollama (inference + embeddings) ----------------------------------
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="qwen2.5:3b")
    embedding_model: str = Field(default="nomic-embed-text")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # Ollama request timeout in seconds; generation on CPU can be slow.
    request_timeout: float = Field(default=300.0, gt=0)

    # --- Weaviate ----------------------------------------------------------
    weaviate_host: str = Field(default="localhost")
    weaviate_http_port: int = Field(default=8080)
    weaviate_grpc_port: int = Field(default=50051)
    weaviate_secure: bool = Field(default=False)
    collection_name: str = Field(default="ResearchPaperChunk")

    # --- Chunking ----------------------------------------------------------
    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    # --- Retrieval ---------------------------------------------------------
    retrieval_k: int = Field(default=5, gt=0)
    # Candidates fetched before MMR re-ranking; must be >= retrieval_k.
    retrieval_fetch_k: int = Field(default=20, gt=0)
    max_agent_iterations: int = Field(default=6, gt=0)

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @field_validator("data_dir")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser()

    @property
    def weaviate_url(self) -> str:
        scheme = "https" if self.weaviate_secure else "http"
        return f"{scheme}://{self.weaviate_host}:{self.weaviate_http_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call `get_settings.cache_clear()` in tests."""
    return Settings()
