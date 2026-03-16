"""
ChromaDB adapter for SURA-RAG.

Provides vector store operations (delete, retrieve, verify) against
a ChromaDB collection, supporting both local and remote instances.
"""

from __future__ import annotations

from sura_rag.adapters.base import BaseVectorAdapter
from sura_rag.exceptions import DocumentNotFoundError


class ChromaDBAdapter(BaseVectorAdapter):
    """Adapter for ChromaDB vector store.

    Supports both local persistent storage and remote ChromaDB servers.

    Args:
        collection_name: Name of the ChromaDB collection to operate on.
        persist_directory: Local directory for persistent storage.
        host: Hostname for remote ChromaDB server (optional).
        port: Port for remote ChromaDB server.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./chroma_db",
        host: str | None = None,
        port: int = 8000,
    ) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB is not installed. Install it with:\n"
                "  pip install sura-rag[cpu]\n"
                "  or: pip install chromadb"
            )

        if host is not None:
            self._client = chromadb.HttpClient(host=host, port=port)
        else:
            self._client = chromadb.PersistentClient(path=persist_directory)

        self._collection = self._client.get_or_create_collection(
            name=collection_name
        )
        self._collection_name = collection_name

    def delete(self, doc_ids: list[str]) -> bool:
        """Delete documents by ID from the ChromaDB collection.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            True if all documents were deleted successfully.
        """
        for doc_id in doc_ids:
            if not self.document_exists(doc_id):
                raise DocumentNotFoundError(
                    f"Document '{doc_id}' not found in collection "
                    f"'{self._collection_name}'"
                )

        self._collection.delete(ids=doc_ids)
        return True

    def get_document_text(self, doc_id: str) -> str:
        """Retrieve the raw text of a document by ID.

        Args:
            doc_id: The document ID to retrieve.

        Returns:
            The document text content.

        Raises:
            DocumentNotFoundError: If the document ID does not exist.
        """
        result = self._collection.get(ids=[doc_id], include=["documents"])
        if not result["ids"]:
            raise DocumentNotFoundError(
                f"Document '{doc_id}' not found in collection "
                f"'{self._collection_name}'"
            )
        return result["documents"][0]

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document ID exists in the collection.

        Args:
            doc_id: The document ID to check.

        Returns:
            True if the document exists, False otherwise.
        """
        result = self._collection.get(ids=[doc_id])
        return len(result["ids"]) > 0

    def get_collection_size(self) -> int:
        """Return the total number of documents in the collection.

        Returns:
            Integer count of documents.
        """
        return self._collection.count()
