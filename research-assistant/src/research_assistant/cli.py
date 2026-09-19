"""Command-line interface.

    research-assistant ingest            # index every PDF in data/
    research-assistant ask "question"    # one-shot question
    research-assistant chat              # interactive session
    research-assistant status            # what is indexed, is everything up?
    research-assistant reset             # drop the index
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from . import ingest as ingest_module
from . import llm as llm_module
from . import vectorstore as vs
from .agent import ResearchAssistant, build_assistant
from .config import get_settings

app = typer.Typer(
    name="research-assistant",
    help="Ask questions about a local library of research papers. Runs fully offline.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _fail(message: str) -> None:
    """Print an error and exit with a non-zero status."""
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)


@app.command()
def ingest(
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", "-d", help="Directory to scan for PDFs (default: ./data)."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-embed papers even if unchanged."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log progress in detail."),
) -> None:
    """Chunk, embed and index every PDF in the data directory."""
    _configure_logging(verbose)
    settings = get_settings()

    try:
        llm_module.verify_models(settings)
    except llm_module.OllamaUnavailableError as exc:
        _fail(str(exc))

    embeddings = llm_module.build_embeddings(settings)
    target = data_dir or settings.data_dir

    try:
        with vs.weaviate_client(settings) as client:
            with console.status("[bold cyan]Ingesting papers…", spinner="dots") as status:
                report = ingest_module.ingest_directory(
                    client,
                    embeddings,
                    settings,
                    data_dir=target,
                    force=force,
                    on_progress=lambda message: status.update(f"[cyan]{message}"),
                )
    except vs.WeaviateUnavailableError as exc:
        _fail(str(exc))
    except FileNotFoundError as exc:
        _fail(str(exc))
    finally:
        llm_module.close_model(embeddings)

    if not report.files:
        console.print(
            f"[yellow]No PDFs found in {target}.[/yellow] Drop some papers there and re-run."
        )
        raise typer.Exit(code=0)

    table = Table(title="Ingestion report", show_lines=False)
    table.add_column("Paper", overflow="fold")
    table.add_column("Status")
    table.add_column("Chunks", justify="right")
    table.add_column("Detail", overflow="fold")
    colours = {"ingested": "green", "skipped": "yellow", "empty": "yellow", "failed": "red"}
    for result in report.files:
        colour = colours.get(result.status, "white")
        table.add_row(
            result.path.name,
            f"[{colour}]{result.status}[/{colour}]",
            str(result.chunks),
            result.detail,
        )
    console.print(table)
    console.print(
        f"[bold]{len(report.ingested)}[/bold] ingested, "
        f"[bold]{len(report.skipped)}[/bold] skipped, "
        f"[bold]{len(report.failed)}[/bold] failed — "
        f"[bold]{report.total_chunks}[/bold] chunks embedded."
    )
    if report.failed:
        raise typer.Exit(code=1)


def _close_all(client, models) -> None:
    """Release the Weaviate connection and every Ollama HTTP client."""
    for model in models:
        llm_module.close_model(model)
    client.close()


def _render_answer(answer, show_sources: bool) -> None:
    console.print(Panel(Markdown(answer.answer or "_(empty response)_"), title="Answer"))
    if show_sources and answer.citations:
        console.print("[bold]Sources[/bold]")
        for citation in answer.citations:
            console.print(f"  • {citation}")
    elif show_sources:
        console.print("[yellow]No passages were retrieved for this answer.[/yellow]")


def _build_assistant_or_fail(settings, verbose: bool):
    """Open Weaviate + Ollama and return (client, assistant, models).

    `models` are the Ollama-backed objects the caller must close alongside the
    Weaviate client. Exits the process on any failure.
    """
    try:
        llm_module.verify_models(settings)
    except llm_module.OllamaUnavailableError as exc:
        _fail(str(exc))

    try:
        client = vs.connect(settings)
    except vs.WeaviateUnavailableError as exc:
        _fail(str(exc))

    # From here on the client and any Ollama models are ours to own: close them on
    # any failure, since the caller only gets a chance to once we hand them back.
    # Note that `typer.Exit` is an Exception subclass, so `_fail` is covered too.
    models: list[object] = []
    try:
        stats = vs.collection_stats(client, settings.collection_name)
        if stats["chunk_count"] == 0:
            _fail(
                "The index is empty. Add PDFs to the data directory and run "
                "`research-assistant ingest` first."
            )
        embeddings = llm_module.build_embeddings(settings)
        models.append(embeddings)
        store = vs.get_vector_store(client, embeddings, settings)
        chat_model = llm_module.build_chat_model(settings)
        models.append(chat_model)
        assistant = build_assistant(client, store, chat_model, settings, verbose=verbose)
    except BaseException:
        _close_all(client, models)
        raise
    return client, assistant, models


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about the indexed papers."),
    show_sources: bool = typer.Option(
        True, "--sources/--no-sources", help="Print the citations behind the answer."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show the agent's reasoning."),
) -> None:
    """Ask a single question and print the answer."""
    _configure_logging(verbose)
    settings = get_settings()
    client, assistant, models = _build_assistant_or_fail(settings, verbose)
    try:
        with console.status("[bold cyan]Thinking…", spinner="dots"):
            answer = assistant.ask(question)
        _render_answer(answer, show_sources)
    finally:
        _close_all(client, models)


@app.command()
def chat(
    show_sources: bool = typer.Option(
        True, "--sources/--no-sources", help="Print the citations behind each answer."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show the agent's reasoning."),
) -> None:
    """Start an interactive, multi-turn session over the indexed papers."""
    _configure_logging(verbose)
    settings = get_settings()
    client, assistant, models = _build_assistant_or_fail(settings, verbose)

    console.print(
        Panel(
            "Ask questions about your indexed papers.\n"
            "[dim]/reset clears the conversation · /papers lists what is indexed · "
            "/exit quits[/dim]",
            title=f"Research assistant · {settings.llm_model}",
        )
    )
    try:
        _chat_loop(assistant, client, settings, show_sources)
    finally:
        _close_all(client, models)


def _chat_loop(
    assistant: ResearchAssistant, client, settings, show_sources: bool
) -> None:
    """REPL body, split out so it can be driven directly in tests."""
    while True:
        try:
            question = console.input("\n[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            return

        if not question:
            continue
        lowered = question.lower()
        if lowered in {"/exit", "/quit", "exit", "quit"}:
            console.print("[dim]Bye.[/dim]")
            return
        if lowered == "/reset":
            assistant.reset()
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        if lowered == "/papers":
            stats = vs.collection_stats(client, settings.collection_name)
            for name, count in stats["sources"].items():
                console.print(f"  • {name} ({count} chunks)")
            if not stats["sources"]:
                console.print("[yellow]Nothing indexed yet.[/yellow]")
            continue

        try:
            with console.status("[bold cyan]Thinking…", spinner="dots"):
                answer = assistant.ask(question)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            err_console.print(f"[red]That question failed:[/red] {exc}")
            continue
        _render_answer(answer, show_sources)


@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log progress in detail."),
) -> None:
    """Report on Ollama, Weaviate and the current contents of the index."""
    _configure_logging(verbose)
    settings = get_settings()

    table = Table(title="Research assistant status")
    table.add_column("Component")
    table.add_column("State")
    table.add_column("Detail", overflow="fold")

    ollama_ok = False
    try:
        models = llm_module.list_available_models(settings)
        ollama_ok = True
        table.add_row("Ollama", "[green]up[/green]", f"{settings.ollama_base_url}")
        for label, model in (("LLM", settings.llm_model), ("Embeddings", settings.embedding_model)):
            present = llm_module._model_present(model, models)
            table.add_row(
                f"{label} model",
                "[green]ready[/green]" if present else "[red]missing[/red]",
                model if present else f"{model} — run: ollama pull {model}",
            )
    except llm_module.OllamaUnavailableError as exc:
        table.add_row("Ollama", "[red]down[/red]", str(exc).splitlines()[0])

    try:
        with vs.weaviate_client(settings) as client:
            stats = vs.collection_stats(client, settings.collection_name)
            table.add_row("Weaviate", "[green]up[/green]", settings.weaviate_url)
            table.add_row(
                "Collection",
                "[green]present[/green]" if stats["exists"] else "[yellow]absent[/yellow]",
                settings.collection_name,
            )
            table.add_row("Chunks", str(stats["chunk_count"]), f"{len(stats['sources'])} paper(s)")
            console.print(table)
            if stats["sources"]:
                console.print("[bold]Indexed papers[/bold]")
                for name, count in stats["sources"].items():
                    console.print(f"  • {name} ({count} chunks)")
            else:
                console.print(
                    "[yellow]Nothing indexed yet.[/yellow] Add PDFs to "
                    f"{settings.data_dir} and run `research-assistant ingest`."
                )
    except vs.WeaviateUnavailableError as exc:
        table.add_row("Weaviate", "[red]down[/red]", str(exc).splitlines()[0])
        console.print(table)
        raise typer.Exit(code=1)

    if not ollama_ok:
        raise typer.Exit(code=1)


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete the Weaviate collection and everything indexed in it."""
    settings = get_settings()
    if not yes:
        confirmed = typer.confirm(
            f"Delete collection '{settings.collection_name}' and all indexed chunks?"
        )
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)
    try:
        with vs.weaviate_client(settings) as client:
            deleted = vs.delete_collection(client, settings.collection_name)
    except vs.WeaviateUnavailableError as exc:
        _fail(str(exc))
    if deleted:
        console.print(f"[green]Deleted collection {settings.collection_name}.[/green]")
    else:
        console.print(f"[yellow]Collection {settings.collection_name} did not exist.[/yellow]")


def _quiet_third_party_warnings() -> None:
    """Hide library warnings the user cannot act on.

    weaviate-client nags about its own version on every connect, langchain-core
    triggers a Pydantic deprecation on every tool invocation, and Ollama's HTTP
    client can emit ResourceWarning at shutdown. None is actionable from the CLI,
    and all of them print around every answer.

    Importing weaviate pulls in authlib, which *prepends* its own
    ``("default", Warning)`` filter — that outranks anything registered earlier,
    so filtering before the import silently has no effect. The heavy modules are
    therefore imported first (every real command loads them anyway) and the
    filters applied afterwards.

    Only the console entry point calls this, so tests and library users keep
    seeing the warnings.
    """
    import importlib

    for module in ("weaviate", "langchain_ollama", "langchain_weaviate", "langchain.agents"):
        try:
            importlib.import_module(module)
        except Exception:  # noqa: BLE001 - a real import failure surfaces later, in context
            logging.getLogger(__name__).debug("Could not pre-import %s", module, exc_info=True)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)


def main() -> None:  # pragma: no cover - entry point shim
    _quiet_third_party_warnings()
    try:
        app()
    except KeyboardInterrupt:
        err_console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
