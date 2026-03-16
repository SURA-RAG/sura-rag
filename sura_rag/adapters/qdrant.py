"""
Qdrant adapter for SURA-RAG.

Provides vector store operations against a Qdrant collection,
supporting both in-memory and remote Qdrant instances.
"""

from __future__ import annotations

from sura_rag.adapters.base import BaseVectorAdapter
from sura_rag.exceptions import DocumentNotFoundError


class QdrantAdapter(BaseVectorAdapter):
    """Adapter for Qdrant vector store.

    Supports in-memory storage, local file storage, and remote Qdrant servers.

    Args:
        collection_name: Name of the Qdrant collection to operate on.
        location: Local storage path or \":memory:\" for in-memory.
        url: URL for remote Qdrant server (optional).
        api_key: API key for remote Qdrant server (optional).
    """

    def __init__(
        self,
        collection_name: str,
        location: str = ":memory:",
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "Qdrant client is not installed. Install it with:\n"
                "  pip install sura-rag\n"
                "  or: pip install qdrant-client"
            )

        if url is not None:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(location=location)

        self._collection_name = collection_name
        self._payloads: dict[str, str] = {}

    def add_documents(
        self, doc_ids: list[str], texts: list[str], vectors: list[list[float]]
    ) -> None:
        """Add documents to the Qdrant collection.

        This is a helper for setting up the collection. The ForgetEngine
        does not call this directly — it is used during initial ingestion.

        Args:
            doc_ids: List of document IDs.
            texts: List of document texts.
            vectors: List of embedding vectors.
        """
        from qdrant_client.models import Distance, PointStruct, VectorParams

        vector_size = len(vectors[0]) if vectors else 384

        # Create collection if it doesn't exist
        collections = [
            c.name for c in self._client.get_collections().collections
        ]
        if self._collection_name not in collections:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE
                ),
            )

        points = []
        for i, (doc_id, text, vector) in enumerate(
            zip(doc_ids, texts, vectors)
        ):
            self._payloads[doc_id] = text
            points.append(
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={"doc_id": doc_id, "text": text},
                )
            )

        self._client.upsert(
            collection_name=self._collection_name, points=points
        )

    def delete(self, doc_ids: list[str]) -> bool:
        """Delete documents by ID from the Qdrant collection.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            True if all documents were deleted successfully.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        for doc_id in doc_ids:
            if not self.document_exists(doc_id):
                raise DocumentNotFoundError(
                    f"Document '{doc_id}' not found in collection "
                    f"'{self._collection_name}'"
                )

        for doc_id in doc_ids:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id", match=MatchValue(value=doc_id)
                        )
                    ]
                ),
            )
            self._payloads.pop(doc_id, None)

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
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id", match=MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )

        points = results[0]
        if not points:
            raise DocumentNotFoundError(
                f"Document '{doc_id}' not found in collection "
                f"'{self._collection_name}'"
            )
        return points[0].payload.get("text", "")

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document ID exists in the collection.

        Args:
            doc_id: The document ID to check.

        Returns:
            True if the document exists, False otherwise.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id", match=MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=1,
        )
        return len(results[0]) > 0

    def get_collection_size(self) -> int:
        """Return the total number of documents in the collection.

        Returns:
            Integer count of documents.
        """
        info = self._client.get_collection(
            collection_name=self._collection_name
        )
        return info.points_count or 0
