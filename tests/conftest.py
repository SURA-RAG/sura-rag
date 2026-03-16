"""
Shared pytest fixtures for SURA-RAG tests.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from tests.fixtures.mock_ollama import MockOllamaAdapter
from tests.fixtures.sample_docs import SAMPLE_DOCUMENTS


@pytest.fixture
def mock_llm():
    """Provide a MockOllamaAdapter with diverse embeddings (clean mode)."""
    return MockOllamaAdapter(high_similarity=False)


@pytest.fixture
def mock_llm_leaky():
    """Provide a MockOllamaAdapter with high-similarity embeddings (leak mode)."""
    return MockOllamaAdapter(high_similarity=True)


@pytest.fixture
def sample_docs():
    """Provide the sample documents list."""
    return SAMPLE_DOCUMENTS


@pytest.fixture
def chroma_collection():
    """Provide an in-memory ChromaDB collection with sample documents."""
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection(name="test_sura")

    collection.add(
        ids=[doc["id"] for doc in SAMPLE_DOCUMENTS],
        documents=[doc["text"] for doc in SAMPLE_DOCUMENTS],
        metadatas=[{"subject": doc["subject"]} for doc in SAMPLE_DOCUMENTS],
    )

    return collection


@pytest.fixture
def chroma_adapter(chroma_collection):
    """Provide a ChromaDBAdapter backed by the in-memory collection."""
    from sura_rag.adapters.chroma import ChromaDBAdapter

    adapter = ChromaDBAdapter.__new__(ChromaDBAdapter)
    adapter._collection = chroma_collection
    adapter._collection_name = "test_sura"
    return adapter


@pytest.fixture
def tmp_db_url(tmp_path):
    """Provide a temporary SQLite database URL."""
    db_path = tmp_path / "test_audit.db"
    return f"sqlite:///{db_path}"


@pytest.fixture
def forget_registry(tmp_db_url):
    """Provide a ForgetRegistry backed by a temporary database."""
    from sura_rag.engines.forget_engine import ForgetRegistry

    return ForgetRegistry(db_url=tmp_db_url)


@pytest.fixture
def sura_config():
    """Provide a default SuraConfig."""
    from sura_rag.config import SuraConfig

    return SuraConfig()
