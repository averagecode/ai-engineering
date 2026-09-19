"""Integration tests driving the CLI itself against live Weaviate and Ollama."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_assistant import cli as cli_module
from research_assistant import vectorstore as vs
from tests.conftest import PAPER_TEXT, write_pdf

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture
def live_cli(live_client, live_services, tmp_path: Path, mocker):
    """CLI bound to live services, a temp data dir and a throwaway collection."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    title, paragraphs = PAPER_TEXT[0]
    write_pdf(data_dir / "sparse_routing.pdf", title, paragraphs)

    settings = live_services.model_copy(update={"data_dir": data_dir})
    mocker.patch.object(cli_module, "get_settings", return_value=settings)
    # `live_client` owns connection lifecycle; hand the CLI its own connections.
    return settings


def test_status_reports_live_services(live_cli):
    result = runner.invoke(cli_module.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "up" in result.output


def test_ingest_then_ask_end_to_end(live_cli):
    ingest = runner.invoke(cli_module.app, ["ingest"])
    assert ingest.exit_code == 0, ingest.output
    assert "sparse_routing.pdf" in ingest.output
    assert "chunks embedded" in ingest.output

    status = runner.invoke(cli_module.app, ["status"])
    assert "sparse_routing.pdf" in status.output

    ask = runner.invoke(cli_module.app, ["ask", "By how much does GSR reduce inference FLOPs?"])
    assert ask.exit_code == 0, ask.output
    assert "Answer" in ask.output
    assert "sparse_routing.pdf" in ask.output  # a citation was printed


def test_second_ingest_skips_unchanged_papers(live_cli):
    assert runner.invoke(cli_module.app, ["ingest"]).exit_code == 0
    second = runner.invoke(cli_module.app, ["ingest"])
    assert second.exit_code == 0, second.output
    assert "skipped" in second.output


def test_ask_before_ingest_refuses(live_cli):
    result = runner.invoke(cli_module.app, ["ask", "anything"])
    assert result.exit_code == 1
    assert "index is empty" in result.output


def test_reset_empties_the_index(live_cli, live_client, live_services):
    assert runner.invoke(cli_module.app, ["ingest"]).exit_code == 0
    assert vs.collection_stats(live_client, live_services.collection_name)["chunk_count"] > 0

    result = runner.invoke(cli_module.app, ["reset", "--yes"])
    assert result.exit_code == 0

    assert vs.collection_stats(live_client, live_services.collection_name)["chunk_count"] == 0
