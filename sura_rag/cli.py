"""
CLI interface for SURA-RAG.

Provides command-line access to the forget pipeline, audit log queries,
and certificate generation.
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="sura",
    help="SURA-RAG: Verified data deletion and leak detection for RAG systems.",
)

console = Console()


@app.command()
def version():
    """Print the sura-rag version."""
    from sura_rag import __version__

    console.print(f"sura-rag v{__version__}")


@app.command()
def audit(
    event_type: str = typer.Option(None, help="Filter by event type"),
    requestor_id: str = typer.Option(None, help="Filter by requestor"),
    regulation: str = typer.Option(None, help="Filter by regulation"),
    db_url: str = typer.Option(
        "sqlite:///sura_audit.db", help="Audit database URL"
    ),
):
    """Query the SURA audit log."""
    from rich.table import Table

    from sura_rag.audit.logger import AuditLogger

    logger = AuditLogger(db_url=db_url)
    entries = logger.query(
        event_type=event_type,
        requestor_id=requestor_id,
        regulation=regulation,
    )

    if not entries:
        console.print("[dim]No audit entries found.[/dim]")
        return

    table = Table(title="SURA Audit Log")
    table.add_column("Timestamp")
    table.add_column("Event Type")
    table.add_column("Doc IDs")
    table.add_column("Requestor")
    table.add_column("Status")

    for entry in entries[:50]:
        table.add_row(
            entry.timestamp,
            entry.event_type,
            ", ".join(entry.doc_ids),
            entry.requestor_id,
            entry.status,
        )

    console.print(table)


@app.command()
def check():
    """Check if Ollama is running and required models are available."""
    from sura_rag.llms.ollama import OllamaAdapter

    adapter = OllamaAdapter()
    if adapter.is_available():
        console.print("[green]Ollama is running and available.[/green]")
    else:
        console.print(
            "[red]Ollama is not running.[/red]\n"
            "Start it with: [bold]ollama serve[/bold]"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
