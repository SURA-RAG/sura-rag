"""
Sample documents for testing SURA-RAG.

Provides realistic test data and a helper to build an in-memory
ChromaDB collection pre-populated with these documents.
"""

from __future__ import annotations

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "text": (
            "John Smith is a senior engineer at Acme Corp earning $120,000 "
            "per year. His employee ID is EMP-4821. He joined in March 2019."
        ),
        "subject": "John Smith employment record",
    },
    {
        "id": "doc_002",
        "text": (
            "Project Falcon is a confidential product roadmap targeting "
            "Q3 2025 launch. The budget is $2.4M and the lead is Sarah Chen."
        ),
        "subject": "Project Falcon roadmap",
    },
    {
        "id": "doc_003",
        "text": (
            "Patient record: Maria Garcia, DOB 1985-03-12, diagnosis "
            "hypertension, prescribed lisinopril 10mg daily since January 2024."
        ),
        "subject": "Patient Maria Garcia medical record",
    },
]


def build_mock_chromadb(
    collection_name: str = "test_collection",
) -> "chromadb.Collection":
    """Create an in-memory ChromaDB collection with sample documents.

    Args:
        collection_name: Name for the test collection.

    Returns:
        A ChromaDB Collection populated with SAMPLE_DOCUMENTS.
    """
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name)

    collection.add(
        ids=[doc["id"] for doc in SAMPLE_DOCUMENTS],
        documents=[doc["text"] for doc in SAMPLE_DOCUMENTS],
        metadatas=[{"subject": doc["subject"]} for doc in SAMPLE_DOCUMENTS],
    )

    return collection
