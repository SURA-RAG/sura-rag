"""
Abstract base class for vector store adapters.

Any new vector store (Pinecone, Weaviate, Milvus, etc.) must implement
this interface to be compatible with SURA-RAG.
"""

from abc import ABC, abstractmethod


class BaseVectorAdapter(ABC):
    """Abstract base class for all vector store adapters.

    Provides the minimal interface required by the ForgetEngine
    to delete documents, retrieve text for probe generation,
    and verify deletion.
    """

    @abstractmethod
    def delete(self, doc_ids: list[str]) -> bool:
        """Delete documents by ID from the vector store.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            True if all documents were deleted successfully.
        """

    @abstractmethod
    def get_document_text(self, doc_id: str) -> str:
        """Retrieve the raw text of a document by ID.

        Used by the ForgetEngine for probe generation and fingerprinting.

        Args:
            doc_id: The document ID to retrieve.

        Returns:
            The document text content.

        Raises:
            DocumentNotFoundError: If the document ID does not exist.
        """

    @abstractmethod
    def document_exists(self, doc_id: str) -> bool:
        """Check if a document ID exists in the store.

        Args:
            doc_id: The document ID to check.

        Returns:
            True if the document exists, False otherwise.
        """

    @abstractmethod
    def get_collection_size(self) -> int:
        """Return the total number of documents in the store.

        Returns:
            Integer count of documents.
        """
