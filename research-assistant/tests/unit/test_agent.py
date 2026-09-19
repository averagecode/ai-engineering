"""Unit tests for the agent's tools, citation formatting and conversation state."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from research_assistant import agent as agent_module
from research_assistant.agent import (
    Answer,
    ResearchAssistant,
    SourceCollector,
    build_list_papers_tool,
    build_search_tool,
    format_citation,
    format_documents,
)
from tests.fakes import FakeAction, FakeExecutor, FakeStore, FakeWeaviateClient, make_chunk


# --- citation formatting ---------------------------------------------------- #


def test_format_citation_includes_page():
    assert format_citation(make_chunk("paper.pdf", 4, "text")) == "paper.pdf, p. 4"


def test_format_citation_without_page():
    assert format_citation(Document(page_content="t", metadata={"source": "p.pdf"})) == "p.pdf"


def test_format_citation_without_metadata():
    assert format_citation(Document(page_content="t")) == "unknown source"


def test_format_documents_numbers_and_tags_each_excerpt():
    rendered = format_documents(
        [make_chunk("a.pdf", 1, "First excerpt."), make_chunk("b.pdf", 9, "Second excerpt.")]
    )
    assert "[1] (a.pdf, p. 1)" in rendered
    assert "[2] (b.pdf, p. 9)" in rendered
    assert "First excerpt." in rendered and "Second excerpt." in rendered


def test_format_documents_on_empty_result_tells_the_model_to_say_so():
    rendered = format_documents([])
    assert "No matching passages" in rendered
    assert "does not cover" in rendered


# --- source collector ------------------------------------------------------- #


def test_collector_deduplicates_citations_preserving_order():
    collector = SourceCollector()
    collector.add([make_chunk("b.pdf", 2, "x"), make_chunk("a.pdf", 1, "y")])
    collector.add([make_chunk("b.pdf", 2, "z")])  # same page, different chunk
    assert collector.unique_citations() == ["b.pdf, p. 2", "a.pdf, p. 1"]


def test_collector_reset_clears_documents():
    collector = SourceCollector()
    collector.add([make_chunk("a.pdf", 1, "x")])
    collector.reset()
    assert collector.documents == [] and collector.unique_citations() == []


# --- search tool ------------------------------------------------------------ #


@pytest.fixture
def search_setup(settings):
    store = FakeStore(
        search_results=[
            make_chunk("sparse_routing.pdf", 3, "GSR reduces inference FLOPs by 41 percent.", 0),
            make_chunk("retrieval_depth.pdf", 1, "Hallucinations fall from 22 to 9 percent.", 1),
        ]
    )
    collector = SourceCollector()
    return store, collector, build_search_tool(store, collector, settings)


def test_search_tool_metadata_guides_the_model(search_setup):
    _, _, tool = search_setup
    assert tool.name == "search_papers"
    assert "Semantic search" in tool.description
    assert "page number" in tool.description


def test_search_tool_returns_formatted_excerpts_and_records_sources(search_setup):
    store, collector, tool = search_setup
    result = tool.invoke({"query": "How much does GSR reduce FLOPs?"})

    assert "41 percent" in result
    assert "(sparse_routing.pdf, p. 3)" in result
    assert store.queries == ["How much does GSR reduce FLOPs?"]
    assert collector.unique_citations() == ["sparse_routing.pdf, p. 3", "retrieval_depth.pdf, p. 1"]


def test_search_tool_uses_settings_k_and_fetch_k(settings):
    store = FakeStore(search_results=[make_chunk("a.pdf", i, f"chunk {i}") for i in range(10)])
    collector = SourceCollector()
    tool = build_search_tool(store, collector, settings)
    tool.invoke({"query": "anything"})
    assert len(collector.documents) == settings.retrieval_k


def test_search_tool_falls_back_to_similarity_when_mmr_unsupported(settings):
    store = FakeStore(search_results=[make_chunk("a.pdf", 1, "fallback excerpt")], mmr_supported=False)
    collector = SourceCollector()
    tool = build_search_tool(store, collector, settings)

    result = tool.invoke({"query": "q"})

    assert "fallback excerpt" in result
    assert len(collector.documents) == 1


def test_search_tool_handles_no_results(settings):
    tool = build_search_tool(FakeStore(search_results=[]), SourceCollector(), settings)
    assert "No matching passages" in tool.invoke({"query": "quantum tunnelling in bees"})


# --- list papers tool ------------------------------------------------------- #


def test_list_papers_tool_reports_inventory(settings):
    client = FakeWeaviateClient()
    client.seed(
        settings.collection_name,
        [{"source": "a.pdf"}, {"source": "a.pdf"}, {"source": "b.pdf"}],
    )
    result = build_list_papers_tool(client, settings).invoke({})
    assert "2 indexed paper(s)" in result
    assert "- a.pdf (2 chunks)" in result
    assert "- b.pdf (1 chunks)" in result


def test_list_papers_tool_when_nothing_indexed(settings):
    result = build_list_papers_tool(FakeWeaviateClient(), settings).invoke({})
    assert "No papers are indexed yet" in result
    assert "ingestion" in result


def test_list_papers_tool_takes_no_arguments(settings):
    tool = build_list_papers_tool(FakeWeaviateClient(), settings)
    assert tool.args == {}


# --- assistant facade ------------------------------------------------------- #


def test_ask_returns_answer_with_citations():
    collector = SourceCollector()
    executor = FakeExecutor(["GSR cuts FLOPs by 41% (sparse_routing.pdf, p. 3)."])

    class Recording(FakeExecutor):
        def invoke(self, payload):
            collector.add([make_chunk("sparse_routing.pdf", 3, "excerpt")])
            return super().invoke(payload)

    assistant = ResearchAssistant(Recording(executor.outputs), collector)
    answer = assistant.ask("How much does GSR cut FLOPs?")

    assert isinstance(answer, Answer)
    assert "41%" in answer.answer
    assert answer.citations == ["sparse_routing.pdf, p. 3"]
    assert len(answer.documents) == 1


def test_ask_records_which_tools_the_agent_called():
    assistant = ResearchAssistant(
        FakeExecutor(["done"], tools_called=["list_papers", "search_papers"]), SourceCollector()
    )
    assert assistant.ask("what have you got?").tools_used == ["list_papers", "search_papers"]


def test_ask_reports_no_tools_when_the_model_skipped_retrieval():
    assistant = ResearchAssistant(FakeExecutor(["from memory"]), SourceCollector())
    assert assistant.ask("anything").tools_used == []


def test_tools_used_ignores_malformed_steps():
    from research_assistant.agent import _tools_used

    assert _tools_used([]) == []
    assert _tools_used([(object(), "obs")]) == []


def test_ask_rejects_blank_questions():
    assistant = ResearchAssistant(FakeExecutor(), SourceCollector())
    with pytest.raises(ValueError, match="must not be empty"):
        assistant.ask("   ")


def test_history_is_passed_to_the_executor_on_later_turns():
    # tools_called set so the no-retrieval retry does not fire and add invocations.
    executor = FakeExecutor(["first", "second"], tools_called=["search_papers"])
    assistant = ResearchAssistant(executor, SourceCollector())

    assistant.ask("What is GSR?")
    assistant.ask("And its FLOP saving?")

    assert executor.calls[0]["chat_history"] == []
    history = executor.calls[1]["chat_history"]
    assert [type(m) for m in history] == [HumanMessage, AIMessage]
    assert history[0].content == "What is GSR?"
    assert history[1].content == "first"


def test_history_window_is_bounded():
    assistant = ResearchAssistant(
        FakeExecutor(tools_called=["search_papers"]), SourceCollector(), history_turns=2
    )
    for i in range(5):
        assistant.ask(f"question {i}")
    assert len(assistant.history) == 4
    assert assistant.history[0].content == "question 3"


def test_reset_clears_history_and_sources():
    collector = SourceCollector()
    assistant = ResearchAssistant(FakeExecutor(), collector)
    assistant.ask("anything")
    collector.add([make_chunk("a.pdf", 1, "x")])

    assistant.reset()

    assert assistant.history == []
    assert collector.documents == []


def test_sources_do_not_leak_between_questions():
    collector = SourceCollector()
    collector.add([make_chunk("stale.pdf", 1, "from a previous run")])
    assistant = ResearchAssistant(FakeExecutor(), collector)
    answer = assistant.ask("a fresh question")
    assert answer.citations == []


# --- wiring ----------------------------------------------------------------- #


def test_build_assistant_registers_both_tools(settings, mocker):
    captured = {}

    def fake_executor(llm, tools, cfg, verbose=False):
        captured["tools"] = tools
        captured["verbose"] = verbose
        return FakeExecutor()

    mocker.patch.object(agent_module, "build_agent_executor", side_effect=fake_executor)
    assistant = agent_module.build_assistant(
        FakeWeaviateClient(), FakeStore(), mocker.Mock(), settings, verbose=True
    )

    assert isinstance(assistant, ResearchAssistant)
    assert [t.name for t in captured["tools"]] == ["search_papers", "list_papers"]
    assert captured["verbose"] is True


def test_system_prompt_enforces_grounding_and_citation():
    prompt = agent_module.SYSTEM_PROMPT.lower()
    assert "search_papers" in prompt
    assert "cite" in prompt
    assert "do not speculate" in prompt


# --- agent loop (real AgentExecutor, scripted model) ------------------------ #


def test_agent_executor_calls_the_search_tool_then_answers(settings):
    """Drives the real AgentExecutor loop with a scripted tool-calling model."""
    from langchain_core.messages import AIMessage

    from tests.fakes import FakeToolCallingChatModel

    store = FakeStore(
        search_results=[make_chunk("sparse_routing.pdf", 3, "GSR cuts FLOPs by 41 percent.")]
    )
    collector = SourceCollector()
    tools = [build_search_tool(store, collector, settings)]
    llm = FakeToolCallingChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_papers", "args": {"query": "GSR FLOP saving"}, "id": "call-1"}
                ],
            ),
            AIMessage(content="GSR cuts FLOPs by 41% (sparse_routing.pdf, p. 3)."),
        ]
    )

    executor = agent_module.build_agent_executor(llm, tools, settings)
    result = executor.invoke({"input": "How much does GSR cut FLOPs?", "chat_history": []})

    assert "41%" in result["output"]
    assert store.queries == ["GSR FLOP saving"]
    assert collector.unique_citations() == ["sparse_routing.pdf, p. 3"]
    # One tool call was taken, and it is reported back as an intermediate step.
    assert len(result["intermediate_steps"]) == 1
    assert result["intermediate_steps"][0][0].tool == "search_papers"


def test_agent_executor_stops_at_max_iterations(settings):
    """A model that only ever calls tools must not loop forever."""
    from langchain_core.messages import AIMessage

    from tests.fakes import FakeToolCallingChatModel

    settings = settings.model_copy(update={"max_agent_iterations": 2})
    store = FakeStore(search_results=[make_chunk("a.pdf", 1, "excerpt")])
    tools = [build_search_tool(store, SourceCollector(), settings)]
    llm = FakeToolCallingChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_papers", "args": {"query": "q"}, "id": "loop"}],
            )
        ]
    )

    executor = agent_module.build_agent_executor(llm, tools, settings)
    result = executor.invoke({"input": "go", "chat_history": []})

    assert len(store.queries) <= 2
    assert result["output"]


# --- retry when the model skips retrieval ----------------------------------- #


def test_retries_once_when_the_agent_did_not_search():
    """Observed with qwen2.5:3b: it asks "which papers?" instead of searching."""
    collector = SourceCollector()

    class SkipsThenSearches(FakeExecutor):
        def invoke(self, payload):
            attempt = len(self.calls)
            self.calls.append(dict(payload))
            if attempt == 0:  # first try: no tool call
                return {"output": "Which papers do you mean?", "intermediate_steps": []}
            collector.add([make_chunk("paper.pdf", 4, "the retrieved fact")])
            return {
                "output": "The papers report 30 per cent (paper.pdf, p. 4).",
                "intermediate_steps": [(FakeAction("search_papers"), "obs")],
            }

    executor = SkipsThenSearches()
    assistant = ResearchAssistant(executor, collector)
    answer = assistant.ask("What fraction do these papers account for?")

    assert len(executor.calls) == 2
    assert "30 per cent" in answer.answer
    assert answer.tools_used == ["search_papers"]
    assert answer.citations == ["paper.pdf, p. 4"]


def test_retry_prompt_spells_out_what_these_papers_means():
    executor = FakeExecutor(["no idea which papers"])
    ResearchAssistant(executor, SourceCollector()).ask("Summarise these papers")

    retry_prompt = executor.calls[1]["input"]
    assert "search_papers" in retry_prompt
    assert "Summarise these papers" in retry_prompt
    assert "never ask which papers" in retry_prompt


def test_history_records_the_original_question_not_the_nudge():
    executor = FakeExecutor(["which papers?"])
    assistant = ResearchAssistant(executor, SourceCollector())
    assistant.ask("Summarise these papers")

    assert assistant.history[0].content == "Summarise these papers"
    assert "search_papers" not in str(assistant.history[0].content)


def test_no_retry_when_the_agent_already_searched():
    executor = FakeExecutor(["grounded"], tools_called=["search_papers"])
    assistant = ResearchAssistant(executor, SourceCollector())
    assistant.ask("anything")
    assert len(executor.calls) == 1


def test_first_answer_is_kept_when_the_retry_also_fails_to_search():
    """Don't discard a usable answer for an equally unsourced retry."""
    executor = FakeExecutor(["the original answer", "the retry answer"])
    assistant = ResearchAssistant(executor, SourceCollector())
    answer = assistant.ask("anything")

    assert len(executor.calls) == 2
    assert answer.answer == "the original answer"
    assert answer.tools_used == []


def test_retry_can_be_disabled():
    executor = FakeExecutor(["no search"])
    assistant = ResearchAssistant(executor, SourceCollector(), retry_without_retrieval=False)
    assistant.ask("anything")
    assert len(executor.calls) == 1
