"""End-to-end integration: the LangChain agent answering over live services.

Real Ollama inference drives the agent loop, so assertions are about behaviour
that any competent tool-calling model must exhibit — a search was performed, an
answer came back, citations point at the right paper — not about exact wording.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_assistant import ingest as ingest_module
from research_assistant import vectorstore as vs
from research_assistant.agent import build_assistant
from research_assistant.llm import build_chat_model, build_embeddings
from tests.conftest import PAPER_TEXT, write_pdf

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def _module_tmp(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("agent-papers")


@pytest.fixture
def assistant(live_client, live_services, _module_tmp: Path):
    """A fully wired assistant over two freshly indexed papers."""
    papers = [
        write_pdf(_module_tmp / name, title, paragraphs)
        for (title, paragraphs), name in zip(
            PAPER_TEXT, ["sparse_routing.pdf", "retrieval_depth.pdf"]
        )
    ]
    embeddings = build_embeddings(live_services)
    report = ingest_module.ingest_paths(
        papers, live_client, embeddings, live_services, force=True
    )
    assert not report.failed, [f.detail for f in report.failed]

    store = vs.get_vector_store(live_client, embeddings, live_services)
    llm = build_chat_model(live_services)
    return build_assistant(live_client, store, llm, live_services)


def test_chat_model_supports_tool_calling(live_services):
    """The configured model must be able to bind tools, or the agent cannot work."""
    from langchain_core.tools import tool

    @tool
    def noop(query: str) -> str:
        """A no-op tool."""
        return query

    bound = build_chat_model(live_services).bind_tools([noop])
    assert bound is not None


def test_agent_answers_a_factual_question_with_citations(assistant):
    answer = assistant.ask("By how much does GSR reduce inference FLOPs?")

    assert answer.answer.strip()
    # The agent must have retrieved rather than answered from memory.
    assert "search_papers" in answer.tools_used
    assert answer.documents, "the agent answered without calling search_papers"
    assert any(doc.metadata["source"] == "sparse_routing.pdf" for doc in answer.documents)
    assert any("sparse_routing.pdf" in citation for citation in answer.citations)


def test_retrieved_context_contains_the_answer(assistant):
    """Separates retrieval quality from generation quality."""
    answer = assistant.ask("What perplexity does GSR reach on WikiText-103?")

    assert answer.documents
    retrieved = " ".join(doc.page_content for doc in answer.documents)
    assert "17.4" in retrieved


def test_list_papers_tool_reads_the_live_index(live_client, live_services, assistant):
    """Deterministic: the inventory tool over real Weaviate data."""
    from research_assistant.agent import build_list_papers_tool

    result = build_list_papers_tool(live_client, live_services).invoke({})

    assert "sparse_routing.pdf" in result
    assert "retrieval_depth.pdf" in result
    assert "chunks" in result


def test_agent_consults_the_index_when_asked_what_it_has(assistant):
    """Which tool the model picks is up to the model; that it uses one is not."""
    answer = assistant.ask("Which papers do you have access to? List their filenames.")

    assert answer.answer.strip()
    assert answer.tools_used, "the agent answered without consulting the index"
    assert set(answer.tools_used) <= {"search_papers", "list_papers"}


def test_agent_routes_to_the_second_paper(assistant):
    answer = assistant.ask("What happens to the hallucination rate beyond ten passages?")

    assert answer.documents
    assert any(doc.metadata["source"] == "retrieval_depth.pdf" for doc in answer.documents)


def test_agent_does_not_invent_citations_for_out_of_corpus_questions(assistant):
    answer = assistant.ask("What is the capital city of Bolivia?")

    assert answer.answer.strip()
    # Whatever it says, it must not attribute the claim to one of these papers.
    assert "sparse_routing.pdf" not in answer.answer
    assert "retrieval_depth.pdf" not in answer.answer


def test_conversation_history_is_retained_across_turns(assistant):
    assistant.ask("What is GSR?")
    assistant.ask("How many experts does it activate per token?")

    assert len(assistant.history) == 4
    assert assistant.history[0].content == "What is GSR?"


def test_reset_clears_conversation_but_keeps_the_index(assistant, live_client, live_services):
    assistant.ask("What is GSR?")
    assistant.reset()

    assert assistant.history == []
    assert vs.collection_stats(live_client, live_services.collection_name)["chunk_count"] > 0
