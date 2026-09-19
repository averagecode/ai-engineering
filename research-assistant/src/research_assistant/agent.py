"""The LangChain research agent.

A tool-calling agent is given two tools over the Weaviate index — semantic
search and a paper inventory — and is instructed to answer only from what it
retrieves, citing paper and page. Retrieved chunks are recorded in a
`SourceCollector` so the CLI can print citations next to the answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool

from .config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain.agents import AgentExecutor
    from langchain_weaviate import WeaviateVectorStore
    from weaviate import WeaviateClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research assistant with access to a library of \
research papers that have been indexed into a vector database.

Rules you must follow:
1. Always call `search_papers` before answering a question about the papers. \
Never answer from memory or prior knowledge about a paper's content.
2. If the first search is thin or off-target, rephrase and search again \
(different keywords, a narrower sub-question). Up to three searches is normal.
3. Ground every claim in retrieved text and cite it inline as \
(paper.pdf, p. N). Cite each distinct source you rely on.
4. If the retrieved excerpts do not contain the answer, say plainly that the \
indexed papers do not cover it. Do not speculate or fill gaps from general \
knowledge.
5. Use `list_papers` when asked what is available, or to scope a question to a \
particular paper.
6. Be concise and technical. Quote short phrases where precise wording matters, \
and note explicitly when papers disagree.
"""


@dataclass
class SourceCollector:
    """Records documents returned by the retriever during one agent run."""

    documents: list[Document] = field(default_factory=list)

    def add(self, documents: Sequence[Document]) -> None:
        self.documents.extend(documents)

    def reset(self) -> None:
        self.documents.clear()

    def unique_citations(self) -> list[str]:
        """De-duplicated "source p. N" labels in first-seen order."""
        seen: list[str] = []
        for doc in self.documents:
            label = format_citation(doc)
            if label not in seen:
                seen.append(label)
        return seen


def format_citation(document: Document) -> str:
    """Human-readable citation label for a chunk."""
    source = document.metadata.get("source", "unknown source")
    page = document.metadata.get("page")
    return f"{source}, p. {page}" if page is not None else str(source)


def format_documents(documents: Sequence[Document]) -> str:
    """Render retrieved chunks as a citation-tagged block for the LLM."""
    if not documents:
        return (
            "No matching passages were found in the indexed papers. "
            "Try different keywords, or tell the user the corpus does not cover this."
        )
    blocks = [
        f"[{index}] ({format_citation(doc)})\n{doc.page_content.strip()}"
        for index, doc in enumerate(documents, start=1)
    ]
    return "\n\n".join(blocks)


def build_search_tool(
    store: "WeaviateVectorStore",
    collector: SourceCollector,
    settings: Settings | None = None,
) -> StructuredTool:
    """Semantic search over the paper chunks, with MMR for diverse excerpts."""
    settings = settings or get_settings()
    fetch_k = max(settings.retrieval_fetch_k, settings.retrieval_k)

    def search_papers(query: str) -> str:
        """Search the indexed research papers for passages relevant to a query."""
        try:
            documents = store.max_marginal_relevance_search(
                query, k=settings.retrieval_k, fetch_k=fetch_k
            )
        except Exception:  # noqa: BLE001 - MMR needs vectors some backends omit
            logger.warning("MMR search failed; falling back to similarity search", exc_info=True)
            documents = store.similarity_search(query, k=settings.retrieval_k)
        collector.add(documents)
        return format_documents(documents)

    return StructuredTool.from_function(
        func=search_papers,
        name="search_papers",
        description=(
            "Semantic search over the indexed research papers. Input: a focused "
            "natural-language query or question. Returns numbered excerpts, each "
            "tagged with its paper filename and page number."
        ),
    )


def build_list_papers_tool(
    client: "WeaviateClient", settings: Settings | None = None
) -> StructuredTool:
    """Inventory tool so the agent can report what is actually indexed."""
    from . import vectorstore as vs

    settings = settings or get_settings()

    def list_papers() -> str:
        """List the papers currently indexed, with their chunk counts."""
        stats = vs.collection_stats(client, settings.collection_name)
        if not stats["sources"]:
            return "No papers are indexed yet. The user should add PDFs to data/ and run ingestion."
        lines = [f"- {name} ({count} chunks)" for name, count in stats["sources"].items()]
        return f"{len(stats['sources'])} indexed paper(s):\n" + "\n".join(lines)

    return StructuredTool.from_function(
        func=list_papers,
        name="list_papers",
        description=(
            "List the filenames of every research paper currently indexed, with "
            "how many chunks each contributed. Takes no arguments."
        ),
    )


def build_agent_executor(
    llm: BaseChatModel,
    tools: Sequence[StructuredTool],
    settings: Settings | None = None,
    *,
    verbose: bool = False,
) -> "AgentExecutor":
    """Wire the prompt, model and tools into a tool-calling agent loop."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent

    settings = settings or get_settings()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, list(tools), prompt)
    return AgentExecutor(
        agent=agent,
        tools=list(tools),
        verbose=verbose,
        max_iterations=settings.max_agent_iterations,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


@dataclass
class Answer:
    """One agent response plus the citations it was grounded in."""

    question: str
    answer: str
    citations: list[str]
    documents: list[Document]
    #: Names of the tools the agent invoked, in call order. Empty means the model
    #: answered without consulting the index.
    tools_used: list[str] = field(default_factory=list)


#: Sent when the agent answered without searching. Small models sometimes treat
#: "these papers" as ambiguous and ask which papers instead of looking, so the
#: question is put again with the ambiguity resolved explicitly.
RETRY_INSTRUCTION = (
    "You answered without searching the library. Call search_papers now with a "
    "focused query taken from the question below, then answer only from the "
    'excerpts it returns. "These papers", "the papers" and similar phrases always '
    "mean the papers indexed in the library — never ask which papers are meant, "
    "use list_papers if you need their names.\n\nQuestion: {question}"
)


class ResearchAssistant:
    """Stateful facade over the agent: holds chat history and source tracking."""

    def __init__(
        self,
        executor: "AgentExecutor",
        collector: SourceCollector,
        *,
        history_turns: int = 6,
        retry_without_retrieval: bool = True,
    ) -> None:
        self._executor = executor
        self._collector = collector
        self._history: list[BaseMessage] = []
        self._history_turns = history_turns
        self._retry_without_retrieval = retry_without_retrieval

    @property
    def history(self) -> list[BaseMessage]:
        return list(self._history)

    def reset(self) -> None:
        """Clear conversation history (the vector index is untouched)."""
        self._history.clear()
        self._collector.reset()

    def _run(self, prompt: str) -> tuple[str, list[str]]:
        """One agent invocation. Returns its text and the tools it called."""
        self._collector.reset()
        result: dict[str, Any] = self._executor.invoke(
            {"input": prompt, "chat_history": self._history}
        )
        text = str(result.get("output", "")).strip()
        return text, _tools_used(result.get("intermediate_steps") or [])

    def ask(self, question: str) -> Answer:
        """Answer one question, updating conversation history.

        If the model answers without consulting the index, the question is put
        once more with the ambiguity spelled out — see `RETRY_INSTRUCTION`. The
        better of the two attempts (the one that actually retrieved) is returned.
        """
        if not question.strip():
            raise ValueError("Question must not be empty")

        answer_text, tools_used = self._run(question)

        if not tools_used and self._retry_without_retrieval:
            logger.info("Agent answered without retrieving; retrying with an explicit nudge")
            retry_text, retry_tools = self._run(RETRY_INSTRUCTION.format(question=question))
            if retry_tools:
                answer_text, tools_used = retry_text, retry_tools

        self._history.extend([HumanMessage(question), AIMessage(answer_text)])
        # Keep the window bounded: local models have modest context budgets.
        max_messages = self._history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

        return Answer(
            question=question,
            answer=answer_text,
            citations=self._collector.unique_citations(),
            documents=list(self._collector.documents),
            tools_used=tools_used,
        )


def _tools_used(intermediate_steps: Sequence[Any]) -> list[str]:
    """Tool names from an AgentExecutor's `intermediate_steps`, in call order."""
    names = []
    for step in intermediate_steps:
        action = step[0] if isinstance(step, (tuple, list)) and step else step
        name = getattr(action, "tool", None)
        if name:
            names.append(str(name))
    return names


def build_assistant(
    client: "WeaviateClient",
    store: "WeaviateVectorStore",
    llm: BaseChatModel,
    settings: Settings | None = None,
    *,
    verbose: bool = False,
) -> ResearchAssistant:
    """Assemble the full assistant from its collaborators."""
    settings = settings or get_settings()
    collector = SourceCollector()
    tools = [
        build_search_tool(store, collector, settings),
        build_list_papers_tool(client, settings),
    ]
    executor = build_agent_executor(llm, tools, settings, verbose=verbose)
    return ResearchAssistant(executor, collector)
