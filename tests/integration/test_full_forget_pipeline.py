"""
End-to-end integration test for the full forget pipeline.

Uses in-memory ChromaDB and MockOllamaAdapter to test the complete
SuraClient.forget() workflow without requiring Ollama.
"""

from __future__ import annotations

import uuid

import chromadb

from sura_rag.adapters.chroma import ChromaDBAdapter
from sura_rag.client import SuraClient
from sura_rag.config import ForgetMode, GuardrailMode, SuraConfig
from sura_rag.engines.forget_engine import ForgetRegistry
from tests.fixtures.mock_ollama import MockOllamaAdapter
from tests.fixtures.sample_docs import SAMPLE_DOCUMENTS


def _build_test_client(tmp_db_url: str):
    """Build a SuraClient with in-memory ChromaDB and mock LLM."""
    # Create in-memory ChromaDB with sample documents
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="integration_test")
    collection.add(
        ids=[doc["id"] for doc in SAMPLE_DOCUMENTS],
        documents=[doc["text"] for doc in SAMPLE_DOCUMENTS],
        metadatas=[{"subject": doc["subject"]} for doc in SAMPLE_DOCUMENTS],
    )

    # Create adapter pointing to the in-memory collection
    adapter = ChromaDBAdapter.__new__(ChromaDBAdapter)
    adapter._collection = collection
    adapter._collection_name = "integration_test"

    mock_llm = MockOllamaAdapter(high_similarity=False)
    config = SuraConfig(enable_rich_logging=False)

    # Build client manually to avoid Ollama check
    client = SuraClient.__new__(SuraClient)
    client._config = config
    client._config.default_guardrail_mode = GuardrailMode.HARD_BLOCK
    client._llm = mock_llm
    client._vector_store = adapter

    from sura_rag.audit.certificate import CertificateGenerator
    from sura_rag.audit.logger import AuditLogger
    from sura_rag.engines.guardrail import RuntimeGuardrail
    from sura_rag.engines.leak_prober import LeakProber

    client._registry = ForgetRegistry(tmp_db_url)
    from sura_rag.engines.forget_engine import ForgetEngine
    client._forget_engine = ForgetEngine(adapter, client._registry, mock_llm)
    client._leak_prober = LeakProber(mock_llm, config)
    client._guardrail = RuntimeGuardrail(client._registry, mock_llm, config)
    client._audit_logger = AuditLogger(tmp_db_url)
    client._cert_generator = CertificateGenerator(tmp_db_url)

    return client, adapter


class TestFullForgetPipeline:
    """End-to-end integration tests for the forget pipeline."""

    def test_forget_deletes_document(self, tmp_db_url):
        """client.forget() deletes the document from ChromaDB."""
        client, adapter = _build_test_client(tmp_db_url)

        assert adapter.document_exists("doc_001")

        result = client.forget(
            doc_ids=["doc_001"],
            subject="John Smith employment record",
            requestor_id="user_4821",
            regulation="GDPR_Art17",
            forget_mode=ForgetMode.FAST,
        )

        assert result.vector_deleted
        assert not adapter.document_exists("doc_001")

    def test_forget_returns_valid_certificate_id(self, tmp_db_url):
        """client.forget() returns a valid UUID certificate_id."""
        client, _ = _build_test_client(tmp_db_url)

        result = client.forget(
            doc_ids=["doc_001"],
            subject="John Smith",
            requestor_id="user_001",
            regulation="GDPR_Art17",
            forget_mode=ForgetMode.FAST,
        )

        # Verify it's a valid UUID
        uuid.UUID(result.certificate_id)
        assert len(result.certificate_id) == 36

    def test_forget_registers_in_registry(self, tmp_db_url):
        """client.forget() registers the doc in the ForgetRegistry."""
        client, _ = _build_test_client(tmp_db_url)

        client.forget(
            doc_ids=["doc_001"],
            subject="John Smith",
            requestor_id="user_001",
            regulation="GDPR_Art17",
            forget_mode=ForgetMode.FAST,
        )

        assert client._registry.is_forgotten("doc_001")

    def test_forget_with_probes(self, tmp_db_url):
        """client.forget() runs probes when rag_query_fn is provided."""
        client, _ = _build_test_client(tmp_db_url)

        def mock_rag(query: str) -> str:
            return "I have no information about that."

        result = client.forget(
            doc_ids=["doc_001"],
            subject="John Smith",
            requestor_id="user_001",
            regulation="GDPR_Art17",
            rag_query_fn=mock_rag,
            forget_mode=ForgetMode.BALANCED,
            num_probes=4,
        )

        assert result.vector_deleted
        assert result.probe_result.total_probes_run > 0
        assert result.status in ("completed", "leaked_guardrailed")

    def test_forget_creates_audit_entry(self, tmp_db_url):
        """client.forget() creates an entry in the audit log."""
        client, _ = _build_test_client(tmp_db_url)

        client.forget(
            doc_ids=["doc_001"],
            subject="John Smith",
            requestor_id="user_001",
            regulation="GDPR_Art17",
            forget_mode=ForgetMode.FAST,
        )

        entries = client.audit_log(event_type="forget")
        assert len(entries) == 1
        assert entries[0].doc_ids == ["doc_001"]

    def test_other_docs_remain_after_forget(self, tmp_db_url):
        """Forgetting doc_001 does not affect doc_002 or doc_003."""
        client, adapter = _build_test_client(tmp_db_url)

        client.forget(
            doc_ids=["doc_001"],
            subject="John Smith",
            requestor_id="user_001",
            regulation="GDPR_Art17",
            forget_mode=ForgetMode.FAST,
        )

        assert adapter.document_exists("doc_002")
        assert adapter.document_exists("doc_003")
        assert adapter.get_collection_size() == 2
