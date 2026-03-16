# Run this example: python examples/01_quickstart.py
# Prerequisites: ollama serve (in another terminal)
#                ollama pull llama3.2:3b
#                ollama pull nomic-embed-text

"""
SURA-RAG Quickstart Example

Demonstrates the full forget pipeline:
1. Create a ChromaDB collection with sample documents
2. Forget a document (GDPR Article 17 request)
3. Verify deletion with leak probing
4. Generate a compliance certificate
5. Demonstrate runtime guardrailing
"""

import sys

import chromadb
from rich.console import Console
from rich.table import Table

import sura_rag as sr

console = Console()


def main():
    # Step 1: Check if Ollama is running
    console.print("\n[bold blue]Step 1:[/bold blue] Checking Ollama connection...")
    llm = sr.llms.OllamaAdapter()
    if not llm.is_available():
        console.print(
            "[bold red]Ollama is not running![/bold red]\n"
            "Start it with: [green]ollama serve[/green]\n"
            "Then pull models:\n"
            "  [green]ollama pull llama3.2:3b[/green]\n"
            "  [green]ollama pull nomic-embed-text[/green]"
        )
        sys.exit(1)
    console.print("[green]Ollama is running.[/green]")

    # Step 2: Create ChromaDB with sample documents
    console.print("\n[bold blue]Step 2:[/bold blue] Creating ChromaDB collection...")
    chroma_client = chromadb.PersistentClient(path="./demo_chroma_db")
    collection = chroma_client.get_or_create_collection(name="demo_collection")

    docs = [
        {
            "id": "doc_001",
            "text": "John Smith is a senior engineer at Acme Corp earning $120,000 per year. His employee ID is EMP-4821. He joined in March 2019.",
        },
        {
            "id": "doc_002",
            "text": "Project Falcon is a confidential product roadmap targeting Q3 2025 launch. The budget is $2.4M and the lead is Sarah Chen.",
        },
        {
            "id": "doc_003",
            "text": "Patient record: Maria Garcia, DOB 1985-03-12, diagnosis hypertension, prescribed lisinopril 10mg daily since January 2024.",
        },
    ]

    # Clear and re-add
    try:
        chroma_client.delete_collection("demo_collection")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name="demo_collection")
    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
    )
    console.print(f"[green]Added {collection.count()} documents to ChromaDB.[/green]")

    # Step 3: Create SuraClient
    console.print("\n[bold blue]Step 3:[/bold blue] Creating SuraClient...")
    adapter = sr.adapters.ChromaDBAdapter(
        collection_name="demo_collection",
        persist_directory="./demo_chroma_db",
    )
    client = sr.SuraClient(
        vector_store=adapter,
        config=sr.SuraConfig(generator_model="llama3.2:3b"),
    )

    # Step 4: Define a simple RAG query function
    def rag_query(query: str) -> str:
        """Simple RAG: retrieve top result and format as answer."""
        results = collection.query(query_texts=[query], n_results=1)
        if results["documents"] and results["documents"][0]:
            return results["documents"][0][0]
        return "No relevant information found."

    # Step 5: Forget doc_001 (John Smith's records)
    console.print("\n[bold blue]Step 5:[/bold blue] Forgetting doc_001 (John Smith)...")
    result = client.forget(
        doc_ids=["doc_001"],
        subject="John Smith salary records",
        requestor_id="user_4821",
        regulation="GDPR_Art17",
        rag_query_fn=rag_query,
        forget_mode=sr.ForgetMode.BALANCED,
        num_probes=8,
    )

    # Step 6: Print results
    console.print("\n[bold blue]Step 6:[/bold blue] Forget Results:")
    table = Table(title="ForgetResult")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Certificate ID", result.certificate_id)
    table.add_row("Status", result.status)
    table.add_row("Vector Deleted", str(result.vector_deleted))
    table.add_row("Probe Verdict", result.probe_result.verdict)
    table.add_row("Composite Score", f"{result.forget_score.composite_score:.4f}")
    table.add_row("Leak Score", f"{result.forget_score.parametric_leak_score:.4f}")
    table.add_row("Guardrail Mode", result.guardrail_mode)
    console.print(table)

    # Step 7: Save compliance certificate
    console.print("\n[bold blue]Step 7:[/bold blue] Saving compliance certificate...")
    cert = client.get_certificate(result.certificate_id)
    if cert:
        console.print(f"[green]Certificate retrieved: {cert.certificate_id}[/green]")

    # Step 8: Demonstrate guardrail wrapping
    console.print("\n[bold blue]Step 8:[/bold blue] Testing guardrail wrapper...")
    safe_rag = client.wrap(rag_query)

    response = safe_rag("What is John Smith's salary?")
    console.print(f"[yellow]Guardrailed response:[/yellow] {response}")

    scan = client._guardrail.last_scan_result
    if scan:
        console.print(f"  Leaked: {scan.leaked}")
        console.print(f"  Action: {scan.action_taken}")
        console.print(f"  Similarity: {scan.similarity_score:.4f}")

    console.print("\n[bold green]Demo complete![/bold green]")


if __name__ == "__main__":
    main()
