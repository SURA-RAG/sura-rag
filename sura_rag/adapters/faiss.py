"""
FAISS adapter for SURA-RAG.

FAISS does not support native deletion, so this adapter implements
a soft-delete pattern using a blocklist file stored alongside the index.
"""

from __future__ import annotations

import json
from pathlib import Path

from sura_rag.adapters.base import BaseVectorAdapter
from sura_rag.exceptions import DocumentNotFoundError


class FAISSAdapter(BaseVectorAdapter):
    """Adapter for FAISS vector store with soft-delete support.

    Since FAISS does not support native document deletion, this adapter
    maintains a blocklist file alongside the index. Documents in the
    blocklist are treated as deleted and excluded from all operations.

    Args:
        index_path: Path to the FAISS index file.
        docstore_path: Path to the JSON document store file mapping
                       doc_ids to their text and index positions.
    """

    def __init__(self, index_path: str, docstore_path: str) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError(
                "FAISS is not installed. Install it with:\n"
                "  pip install sura-rag\n"
                "  or: pip install faiss-cpu"
            )

        self._index_path = Path(index_path)
        self._docstore_path = Path(docstore_path)
        self._blocklist_path = self._index_path.with_suffix(".blocklist.json")

        self._docstore: dict[str, dict] = {}
        if self._docstore_path.exists():
            self._docstore = json.loads(self._docstore_path.read_text())

        self._blocklist: set[str] = set()
        if self._blocklist_path.exists():
            self._blocklist = set(
                json.loads(self._blocklist_path.read_text())
            )

    def _save_blocklist(self) -> None:
        """Persist the blocklist to disk."""
        self._blocklist_path.write_text(
            json.dumps(sorted(self._blocklist), indent=2)
        )

    def _save_docstore(self) -> None:
        """Persist the document store to disk."""
        self._docstore_path.parent.mkdir(parents=True, exist_ok=True)
        self._docstore_path.write_text(
            json.dumps(self._docstore, indent=2)
        )

    def add_documents(
        self, doc_ids: list[str], texts: list[str], vectors: list[list[float]]
    ) -> None:
        """Add documents to the FAISS index and docstore.

        Args:
            doc_ids: List of document IDs.
            texts: List of document texts.
            vectors: List of embedding vectors.
        """
        import faiss
        import numpy as np

        arr = np.array(vectors, dtype="float32")

        if self._index_path.exists():
            index = faiss.read_index(str(self._index_path))
        else:
            index = faiss.IndexFlatL2(arr.shape[1])

        start_idx = index.ntotal
        index.add(arr)

        for i, (doc_id, text) in enumerate(zip(doc_ids, texts)):
            self._docstore[doc_id] = {
                "text": text,
                "index_position": start_idx + i,
            }

        faiss.write_index(index, str(self._index_path))
        self._save_docstore()

    def delete(self, doc_ids: list[str]) -> bool:
        """Soft-delete documents by adding them to the blocklist.

        FAISS indices do not support native deletion. Instead, documents
        are added to a blocklist and excluded from all subsequent operations.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            True if all documents were blocked successfully.
        """
        for doc_id in doc_ids:
            if not self.document_exists(doc_id):
                raise DocumentNotFoundError(
                    f"Document '{doc_id}' not found in FAISS docstore"
                )

        for doc_id in doc_ids:
            self._blocklist.add(doc_id)

        self._save_blocklist()
        return True

    def get_document_text(self, doc_id: str) -> str:
        """Retrieve the raw text of a document by ID.

        Args:
            doc_id: The document ID to retrieve.

        Returns:
            The document text content.

        Raises:
            DocumentNotFoundError: If the document ID does not exist
                                   or has been soft-deleted.
        """
        if doc_id in self._blocklist:
            raise DocumentNotFoundError(
                f"Document '{doc_id}' has been deleted (soft-delete via blocklist)"
            )
        if doc_id not in self._docstore:
            raise DocumentNotFoundError(
                f"Document '{doc_id}' not found in FAISS docstore"
            )
        return self._docstore[doc_id]["text"]

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document ID exists and is not soft-deleted.

        Args:
            doc_id: The document ID to check.

        Returns:
            True if the document exists and is not in the blocklist.
        """
        return doc_id in self._docstore and doc_id not in self._blocklist

    def get_collection_size(self) -> int:
        """Return the number of active (non-deleted) documents.

        Returns:
            Integer count of active documents.
        """
        return len(
            [
                doc_id
                for doc_id in self._docstore
                if doc_id not in self._blocklist
            ]
        )
