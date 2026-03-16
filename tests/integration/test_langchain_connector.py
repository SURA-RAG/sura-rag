"""
Integration test for the LangChain connector.

Tests that SuraRetriever correctly filters out forgotten documents.
This test uses mocked LangChain objects since LangChain is an optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sura_rag.engines.forget_engine import ForgetRegistry
from tests.fixtures.mock_ollama import MockOllamaAdapter


@dataclass
class MockDocument:
    """Mock LangChain Document."""
    page_content: str
    metadata: dict = field(default_factory=dict)


class MockRetriever:
    """Mock LangChain retriever that returns all documents."""

    def __init__(self, docs: list[MockDocument]):
        self._docs = docs

    def get_relevant_documents(self, query: str) -> list[MockDocument]:
        return self._docs


class TestLangChainConnector:
    """Tests for SuraRetriever integration."""

    def test_filters_forgotten_documents(self, tmp_db_url):
        """SuraRetriever filters out forgotten documents."""
        # Set up registry with a forgotten document
        llm = MockOllamaAdapter()
        registry = ForgetRegistry(db_url=tmp_db_url)
        registry.add(
            doc_id="doc_001",
            subject="John Smith",
            fingerprint_text="John Smith employment record",
            embedding=llm.embed("John Smith employment record"),
            regulation="GDPR_Art17",
        )

        # Create mock documents
        docs = [
            MockDocument(
                page_content="John Smith earns $120,000",
                metadata={"doc_id": "doc_001"},
            ),
            MockDocument(
                page_content="Project Falcon roadmap",
                metadata={"doc_id": "doc_002"},
            ),
        ]

        mock_retriever = MockRetriever(docs)

        # Create a minimal mock client
        class MockSuraClient:
            _registry = registry

        # Import and test (handles case where langchain not installed)
        from sura_rag.connectors.langchain import SuraRetriever

        # Manually construct without the langchain import check
        retriever = SuraRetriever.__new__(SuraRetriever)
        retriever._base_retriever = mock_retriever
        retriever._sura_client = MockSuraClient()
        retriever._filter_mode = "hard_block"
        retriever._registry = registry

        # Test filtering
        results = retriever.get_relevant_documents("salary info")
        assert len(results) == 1
        assert results[0].metadata["doc_id"] == "doc_002"
