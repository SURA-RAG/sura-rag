# Run this example: python examples/04_audit_and_certificates.py
# Prerequisites: ollama serve && ollama pull llama3.2:3b

"""
Audit Log & Certificates Example

Demonstrates:
1. Querying the audit log with filters
2. Generating and saving compliance certificates
3. Exporting audit data to pandas DataFrame
"""

import sura_rag as sr
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    print("=== SURA-RAG Audit & Certificates ===\n")

    # Create client (requires Ollama running)
    try:
        adapter = sr.adapters.ChromaDBAdapter(
            collection_name="audit_demo",
            persist_directory="./audit_demo_db",
        )
        client = sr.SuraClient(
            vector_store=adapter,
            config=sr.SuraConfig(enable_rich_logging=False),
            audit_db_url="sqlite:///audit_demo.db",
        )
    except Exception as e:
        print(f"Setup failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        return

    # Query audit log
    console.print("[bold]Audit Log Entries:[/bold]")
    entries = client.audit_log()
    if entries:
        table = Table(title="Audit Log")
        table.add_column("Timestamp")
        table.add_column("Event Type")
        table.add_column("Doc IDs")
        table.add_column("Status")
        for entry in entries[:10]:
            table.add_row(
                entry.timestamp,
                entry.event_type,
                ", ".join(entry.doc_ids),
                entry.status,
            )
        console.print(table)
    else:
        console.print("[dim]No audit entries yet. Run 01_quickstart.py first.[/dim]")

    # Filter by regulation
    gdpr_entries = client.audit_log(regulation="GDPR_Art17")
    console.print(f"\nGDPR entries: {len(gdpr_entries)}")

    # Export to DataFrame (requires pandas)
    try:
        df = client.audit_log(as_dataframe=True)
        console.print(f"\nDataFrame shape: {df.shape}")
        console.print(df.head())
    except ImportError:
        console.print(
            "\n[dim]Install pandas for DataFrame export: "
            "pip install sura-rag[pandas][/dim]"
        )

    console.print("\n[bold green]Audit demo complete![/bold green]")


if __name__ == "__main__":
    main()
