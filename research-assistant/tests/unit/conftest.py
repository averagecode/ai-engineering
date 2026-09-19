"""Unit tests must not depend on ambient configuration.

A deployed environment (the `tests` service in docker-compose sets
`RA_OLLAMA_BASE_URL`, `RA_WEAVIATE_HOST`, …) or a developer's local `.env` would
otherwise leak into `Settings()` and change what these tests observe. Integration
tests deliberately *do* read that configuration, so this fixture is scoped to
`tests/unit/` only.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip every `RA_*` override for the duration of each unit test."""
    for name in [key for key in os.environ if key.startswith("RA_")]:
        monkeypatch.delenv(name, raising=False)
    yield
