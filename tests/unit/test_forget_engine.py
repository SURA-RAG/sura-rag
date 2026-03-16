"""Tests for the ForgetEngine and ForgetRegistry."""

from __future__ import annotations

import pytest

from sura_rag.engines.forget_engine import ForgetEngine, ForgetRegistry
from sura_rag.exceptions import DocumentNotFoundError


class TestForgetRegistry:
    """Tests for the ForgetRegistry."""

    def test_add_stores_fingerprint(self, forget_registry, mock_llm):
        """ForgetRegistry.add() stores fingerprint correctly."""
        forget_registry.add(
            doc_id="doc_001",
            subject="John Smith",
            fingerprint_text="John Smith is a senior engineer",
            embedding=mock_llm.embed("John Smith is a senior engineer"),
            regulation="GDPR_Art17",
            requestor_id="user_001",
        )

        entries = forget_registry.get_all()
        assert len(entries) == 1
        assert entries[0].doc_id == "doc_001"
        assert entries[0].subject == "John Smith"
        assert len(entries[0].fingerprint_embedding) == 384

    def test_is_forgotten_returns_true_after_add(self, forget_registry, mock_llm):
        """ForgetRegistry.is_forgotten() returns True after add()."""
        assert not forget_registry.is_forgotten("doc_001")

        forget_registry.add(
            doc_id="doc_001",
            subject="test",
            fingerprint_text="test text",
            embedding=mock_llm.embed("test text"),
            regulation="GDPR_Art17",
        )

        assert forget_registry.is_forgotten("doc_001")

    def test_is_forgotten_returns_false_for_unknown(self, forget_registry):
        """ForgetRegistry.is_forgotten() returns False for unknown doc_id."""
        assert not forget_registry.is_forgotten("nonexistent")

    def test_remove_deletes_entry(self, forget_registry, mock_llm):
        """ForgetRegistry.remove() deletes the entry."""
        forget_registry.add(
            doc_id="doc_001",
            subject="test",
            fingerprint_text="test",
            embedding=[0.1] * 384,
            regulation="GDPR_Art17",
        )
        assert forget_registry.is_forgotten("doc_001")

        forget_registry.remove("doc_001")
        assert not forget_registry.is_forgotten("doc_001")


class TestForgetEngine:
    """Tests for the ForgetEngine."""

    def test_delete_removes_doc_from_collection(
        self, chroma_adapter, forget_registry, mock_llm
    ):
        """delete() removes doc from the vector store."""
        engine = ForgetEngine(chroma_adapter, forget_registry, mock_llm)

        assert chroma_adapter.document_exists("doc_001")

        result = engine.delete(
            doc_ids=["doc_001"],
            subject="John Smith",
            requestor_id="user_001",
            regulation="GDPR_Art17",
        )

        assert result["confirmed"]
        assert "doc_001" in result["deleted_ids"]
        assert not chroma_adapter.document_exists("doc_001")

    def test_delete_nonexistent_doc_raises_error(
        self, chroma_adapter, forget_registry, mock_llm
    ):
        """delete() with non-existent doc_id raises DocumentNotFoundError."""
        engine = ForgetEngine(chroma_adapter, forget_registry, mock_llm)

        with pytest.raises(DocumentNotFoundError):
            engine.delete(
                doc_ids=["nonexistent_doc"],
                subject="test",
                requestor_id="user_001",
                regulation="GDPR_Art17",
            )

    def test_delete_confirms_via_document_exists(
        self, chroma_adapter, forget_registry, mock_llm
    ):
        """delete() confirms deletion via document_exists()."""
        engine = ForgetEngine(chroma_adapter, forget_registry, mock_llm)

        result = engine.delete(
            doc_ids=["doc_002"],
            subject="Project Falcon",
            requestor_id="user_002",
            regulation="GDPR_Art17",
        )

        assert result["confirmed"]
        assert not chroma_adapter.document_exists("doc_002")

    def test_delete_stores_fingerprint_in_registry(
        self, chroma_adapter, forget_registry, mock_llm
    ):
        """delete() stores fingerprint in the ForgetRegistry."""
        engine = ForgetEngine(chroma_adapter, forget_registry, mock_llm)

        engine.delete(
            doc_ids=["doc_001"],
            subject="John Smith",
            requestor_id="user_001",
            regulation="GDPR_Art17",
        )

        assert forget_registry.is_forgotten("doc_001")
        entries = forget_registry.get_all()
        assert any(e.doc_id == "doc_001" for e in entries)

    def test_delete_returns_doc_texts(
        self, chroma_adapter, forget_registry, mock_llm
    ):
        """delete() returns the document texts for probe generation."""
        engine = ForgetEngine(chroma_adapter, forget_registry, mock_llm)

        result = engine.delete(
            doc_ids=["doc_001"],
            subject="John Smith",
            requestor_id="user_001",
            regulation="GDPR_Art17",
        )

        assert "doc_001" in result["doc_texts"]
        assert "John Smith" in result["doc_texts"]["doc_001"]
