"""Unit tests for the CLI: command wiring, exit codes and failure messages.

Weaviate and Ollama are faked, so these run offline. They assert on what the
user sees and on the process exit status, not on internal call plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_assistant import cli as cli_module
from research_assistant import ingest as ingest_module
from research_assistant import llm as llm_module
from research_assistant import vectorstore as vs
from research_assistant.agent import Answer, ResearchAssistant, SourceCollector
from research_assistant.config import get_settings
from tests.fakes import FakeExecutor, FakeStore, FakeWeaviateClient, make_chunk

runner = CliRunner()


@pytest.fixture
def cli_env(settings, mocker):
    """Patch the CLI's collaborators and point settings at a temp data dir."""
    mocker.patch.object(cli_module, "get_settings", return_value=settings)
    mocker.patch.object(llm_module, "verify_models")
    client = FakeWeaviateClient()
    mocker.patch.object(vs, "connect", return_value=client)
    mocker.patch.object(vs, "get_vector_store", return_value=FakeStore())
    mocker.patch.object(llm_module, "build_embeddings", return_value=mocker.Mock())
    mocker.patch.object(llm_module, "build_chat_model", return_value=mocker.Mock())
    return client


def invoke(*args: str):
    return runner.invoke(cli_module.app, list(args))


# --- help / discoverability ------------------------------------------------- #


def test_no_args_shows_help():
    result = invoke()
    assert result.exit_code != 0
    for command in ("ingest", "ask", "chat", "status", "reset"):
        assert command in result.output


def test_each_command_documents_itself():
    for command in ("ingest", "ask", "chat", "status", "reset"):
        result = invoke(command, "--help")
        assert result.exit_code == 0, command
        assert "Usage" in result.output


# --- ingest ----------------------------------------------------------------- #


def test_ingest_reports_each_paper_and_a_total(cli_env, pdf_library):
    result = invoke("ingest")
    assert result.exit_code == 0
    assert "sparse_routing.pdf" in result.output
    assert "retrieval_depth.pdf" in result.output
    assert "chunks embedded" in result.output


def test_ingest_with_no_pdfs_tells_the_user_where_to_put_them(cli_env, settings):
    result = invoke("ingest")
    assert result.exit_code == 0
    assert "No PDFs found" in result.output
    assert str(settings.data_dir) in result.output


def test_ingest_exits_nonzero_when_a_paper_fails(cli_env, pdf_library, mocker):
    mocker.patch.object(ingest_module, "prepare_chunks", side_effect=ValueError("corrupt pdf"))
    result = invoke("ingest")
    assert result.exit_code == 1
    assert "failed" in result.output


def test_ingest_accepts_an_explicit_data_dir(cli_env, tmp_path: Path):
    from tests.conftest import write_pdf

    other = tmp_path / "other"
    other.mkdir()
    write_pdf(other / "extra.pdf", "Extra", ["A paper about vector quantisation."])
    result = invoke("ingest", "--data-dir", str(other))
    assert result.exit_code == 0
    assert "extra.pdf" in result.output


def test_ingest_forwards_the_force_flag(cli_env, pdf_library, mocker):
    spy = mocker.spy(ingest_module, "ingest_directory")
    invoke("ingest", "--force")
    assert spy.call_args.kwargs["force"] is True


def test_ingest_fails_fast_when_a_model_is_missing(cli_env, mocker, pdf_library):
    mocker.patch.object(
        llm_module,
        "verify_models",
        side_effect=llm_module.OllamaUnavailableError("Missing Ollama model(s): nomic-embed-text"),
    )
    result = invoke("ingest")
    assert result.exit_code == 1
    assert "nomic-embed-text" in result.output


def test_ingest_reports_weaviate_being_down(cli_env, mocker, pdf_library):
    mocker.patch.object(
        vs, "weaviate_client", side_effect=vs.WeaviateUnavailableError("Could not connect")
    )
    result = invoke("ingest")
    assert result.exit_code == 1
    assert "Could not connect" in result.output


def test_ingest_reports_a_missing_data_dir(cli_env, tmp_path: Path):
    result = invoke("ingest", "--data-dir", str(tmp_path / "absent"))
    assert result.exit_code == 1
    assert "does not exist" in result.output


# --- ask -------------------------------------------------------------------- #


@pytest.fixture
def indexed(cli_env, settings) -> FakeWeaviateClient:
    """A CLI environment whose index already holds chunks."""
    cli_env.seed(settings.collection_name, [{"source": "a.pdf", "page": 1, "checksum": "x"}])
    return cli_env


def _stub_assistant(mocker, answer_text="GSR cuts FLOPs by 41% (a.pdf, p. 1).", citations=("a.pdf, p. 1",)):
    collector = SourceCollector()

    class Stub(ResearchAssistant):
        def ask(self, question: str) -> Answer:
            return Answer(
                question=question,
                answer=answer_text,
                citations=list(citations),
                documents=[make_chunk("a.pdf", 1, "excerpt")],
            )

    stub = Stub(FakeExecutor(), collector)
    mocker.patch.object(cli_module, "build_assistant", return_value=stub)
    return stub


def test_ask_prints_the_answer_and_sources(indexed, mocker):
    _stub_assistant(mocker)
    result = invoke("ask", "How much does GSR cut FLOPs?")
    assert result.exit_code == 0
    assert "41%" in result.output
    assert "Sources" in result.output
    assert "a.pdf, p. 1" in result.output


def test_ask_can_suppress_sources(indexed, mocker):
    _stub_assistant(mocker)
    result = invoke("ask", "anything", "--no-sources")
    assert result.exit_code == 0
    assert "Sources" not in result.output


def test_ask_notes_when_nothing_was_retrieved(indexed, mocker):
    _stub_assistant(mocker, answer_text="The indexed papers do not cover this.", citations=())
    result = invoke("ask", "Who won the 1998 World Cup?")
    assert result.exit_code == 0
    assert "No passages were retrieved" in result.output


def test_ask_on_an_empty_index_directs_the_user_to_ingest(cli_env, mocker):
    _stub_assistant(mocker)
    result = invoke("ask", "anything")
    assert result.exit_code == 1
    assert "index is empty" in result.output
    assert "ingest" in result.output


def test_ask_closes_the_weaviate_client(indexed, mocker):
    _stub_assistant(mocker)
    invoke("ask", "anything")
    assert indexed.closed is True


def test_ask_closes_the_ollama_http_clients(indexed, mocker):
    """Otherwise every run ends with a ResourceWarning about a leaked socket."""
    _stub_assistant(mocker)
    spy = mocker.spy(llm_module, "close_model")
    invoke("ask", "anything")
    assert spy.call_count == 2  # embeddings + chat model


def test_ingest_closes_the_embeddings_client(cli_env, pdf_library, mocker):
    spy = mocker.spy(llm_module, "close_model")
    invoke("ingest")
    assert spy.call_count == 1


def test_ask_closes_the_client_when_the_index_is_empty(cli_env, mocker):
    _stub_assistant(mocker)
    invoke("ask", "anything")
    assert cli_env.closed is True


def test_ask_reports_ollama_being_down(cli_env, mocker):
    mocker.patch.object(
        llm_module, "verify_models", side_effect=llm_module.OllamaUnavailableError("Ollama down")
    )
    result = invoke("ask", "anything")
    assert result.exit_code == 1
    assert "Ollama down" in result.output


# --- chat ------------------------------------------------------------------- #


def test_chat_answers_then_exits_on_command(indexed, mocker):
    _stub_assistant(mocker)
    result = runner.invoke(cli_module.app, ["chat"], input="What is GSR?\n/exit\n")
    assert result.exit_code == 0
    assert "41%" in result.output


def test_chat_exits_cleanly_on_eof(indexed, mocker):
    _stub_assistant(mocker)
    result = runner.invoke(cli_module.app, ["chat"], input="")
    assert result.exit_code == 0
    assert "Bye" in result.output


def test_chat_reset_clears_history(indexed, mocker):
    stub = _stub_assistant(mocker)
    spy = mocker.spy(stub, "reset")
    result = runner.invoke(cli_module.app, ["chat"], input="/reset\n/exit\n")
    assert result.exit_code == 0
    assert spy.call_count == 1
    assert "Conversation cleared" in result.output


def test_chat_papers_lists_the_index(indexed, mocker):
    _stub_assistant(mocker)
    result = runner.invoke(cli_module.app, ["chat"], input="/papers\n/exit\n")
    assert "a.pdf" in result.output


def test_chat_survives_a_failing_question(indexed, mocker):
    stub = _stub_assistant(mocker)
    mocker.patch.object(stub, "ask", side_effect=RuntimeError("model timed out"))
    result = runner.invoke(cli_module.app, ["chat"], input="a question\n/exit\n")
    assert result.exit_code == 0
    assert "model timed out" in result.output


def test_chat_ignores_blank_input(indexed, mocker):
    stub = _stub_assistant(mocker)
    spy = mocker.spy(stub, "ask")
    runner.invoke(cli_module.app, ["chat"], input="\n\n/exit\n")
    assert spy.call_count == 0


# --- status ----------------------------------------------------------------- #


def test_status_reports_everything_up(cli_env, settings, mocker):
    cli_env.seed(settings.collection_name, [{"source": "a.pdf"}, {"source": "a.pdf"}])
    mocker.patch.object(
        llm_module,
        "list_available_models",
        return_value=[settings.llm_model, settings.embedding_model],
    )
    result = invoke("status")
    assert result.exit_code == 0
    assert "up" in result.output
    assert "a.pdf" in result.output


def test_status_flags_a_missing_model(cli_env, settings, mocker):
    mocker.patch.object(llm_module, "list_available_models", return_value=[settings.llm_model])
    result = invoke("status")
    assert result.exit_code == 0
    assert "missing" in result.output
    assert "ollama pull" in result.output


def test_status_exits_nonzero_when_ollama_is_down(cli_env, mocker):
    mocker.patch.object(
        llm_module,
        "list_available_models",
        side_effect=llm_module.OllamaUnavailableError("Could not reach Ollama"),
    )
    result = invoke("status")
    assert result.exit_code == 1
    assert "down" in result.output


def test_status_exits_nonzero_when_weaviate_is_down(cli_env, settings, mocker):
    mocker.patch.object(
        llm_module,
        "list_available_models",
        return_value=[settings.llm_model, settings.embedding_model],
    )
    mocker.patch.object(
        vs, "weaviate_client", side_effect=vs.WeaviateUnavailableError("Could not connect")
    )
    result = invoke("status")
    assert result.exit_code == 1
    assert "down" in result.output


def test_status_tells_an_empty_index_to_ingest(cli_env, settings, mocker):
    mocker.patch.object(
        llm_module,
        "list_available_models",
        return_value=[settings.llm_model, settings.embedding_model],
    )
    result = invoke("status")
    assert "Nothing indexed yet" in result.output


# --- reset ------------------------------------------------------------------ #


def test_reset_deletes_the_collection_with_confirmation(cli_env, settings):
    cli_env.seed(settings.collection_name, [{"source": "a.pdf"}])
    result = runner.invoke(cli_module.app, ["reset"], input="y\n")
    assert result.exit_code == 0
    assert cli_env.collections.deleted == [settings.collection_name]


def test_reset_aborts_when_declined(cli_env, settings):
    cli_env.seed(settings.collection_name, [{"source": "a.pdf"}])
    result = runner.invoke(cli_module.app, ["reset"], input="n\n")
    assert result.exit_code == 0
    assert "Cancelled" in result.output
    assert cli_env.collections.deleted == []


def test_reset_yes_skips_the_prompt(cli_env, settings):
    cli_env.seed(settings.collection_name, [{"source": "a.pdf"}])
    result = invoke("reset", "--yes")
    assert result.exit_code == 0
    assert "Deleted collection" in result.output


def test_reset_on_a_missing_collection_is_harmless(cli_env, settings):
    result = invoke("reset", "--yes")
    assert result.exit_code == 0
    assert "did not exist" in result.output


# --- remaining failure paths ------------------------------------------------ #


def test_ask_reports_weaviate_being_down(cli_env, mocker):
    mocker.patch.object(
        vs, "connect", side_effect=vs.WeaviateUnavailableError("Could not connect to Weaviate")
    )
    result = invoke("ask", "anything")
    assert result.exit_code == 1
    assert "Could not connect to Weaviate" in result.output


def test_chat_reports_weaviate_being_down(cli_env, mocker):
    mocker.patch.object(
        vs, "connect", side_effect=vs.WeaviateUnavailableError("Could not connect to Weaviate")
    )
    result = runner.invoke(cli_module.app, ["chat"])
    assert result.exit_code == 1
    assert "Could not connect to Weaviate" in result.output


def test_ask_closes_the_client_when_wiring_fails(indexed, mocker):
    """A failure after connecting must not leak the Weaviate connection."""
    mocker.patch.object(cli_module, "build_assistant", side_effect=RuntimeError("bad model"))
    with pytest.raises(RuntimeError, match="bad model"):
        runner.invoke(cli_module.app, ["ask", "anything"], catch_exceptions=False)
    assert indexed.closed is True


def test_chat_papers_when_the_index_is_emptied_mid_session(indexed, mocker):
    """`/papers` must cope if the index is cleared while the session is open."""
    _stub_assistant(mocker)
    mocker.patch.object(
        vs,
        "collection_stats",
        side_effect=[
            {"exists": True, "chunk_count": 1, "sources": {"a.pdf": 1}},  # startup check
            {"exists": True, "chunk_count": 0, "sources": {}},  # /papers, now empty
        ],
    )
    result = runner.invoke(cli_module.app, ["chat"], input="/papers\n/exit\n")
    assert result.exit_code == 0
    assert "Nothing indexed yet" in result.output


def test_reset_reports_weaviate_being_down(cli_env, mocker):
    mocker.patch.object(
        vs, "weaviate_client", side_effect=vs.WeaviateUnavailableError("Could not connect")
    )
    result = invoke("reset", "--yes")
    assert result.exit_code == 1
    assert "Could not connect" in result.output
