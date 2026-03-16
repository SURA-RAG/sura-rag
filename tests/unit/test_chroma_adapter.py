"""Tests for the ChromaDB adapter."""

from __future__ import annotations

import pytest

from sura_rag.exceptions import DocumentNotFoundError


class TestChromaDBAdapter:
    """Tests for ChromaDBAdapter."""

    def test_document_exists(self, chroma_adapter):
        """document_exists() returns True for existing documents."""
        assert chroma_adapter.document_exists("doc_001")
        assert chroma_adapter.document_exists("doc_002")
        assert chroma_adapter.document_exists("doc_003")

    def test_document_not_exists(self, chroma_adapter):
        """document_exists() returns False for non-existing documents."""
        assert not chroma_adapter.document_exists("nonexistent")

    def test_get_document_text(self, chroma_adapter):
        """get_document_text() returns the correct text."""
        text = chroma_adapter.get_document_text("doc_001")
        assert "John Smith" in text
        assert "$120,000" in text

    def test_get_document_text_not_found(self, chroma_adapter):
        """get_document_text() raises error for missing document."""
        with pytest.raises(DocumentNotFoundError):
            chroma_adapter.get_document_text("nonexistent")

    def test_get_collection_size(self, chroma_adapter):
        """get_collection_size() returns correct count."""
        assert chroma_adapter.get_collection_size() == 3

    def test_delete_removes_documents(self, chroma_adapter):
        """delete() removes documents from the collection."""
        assert chroma_adapter.document_exists("doc_001")
        chroma_adapter.delete(["doc_001"])
        assert not chroma_adapter.document_exists("doc_001")
        assert chroma_adapter.get_collection_size() == 2

    def test_delete_nonexistent_raises(self, chroma_adapter):
        """delete() raises DocumentNotFoundError for missing docs."""
        with pytest.raises(DocumentNotFoundError):
            chroma_adapter.delete(["nonexistent"])
