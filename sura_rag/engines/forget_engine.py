"""
Forget engine for SURA-RAG.

Orchestrates vector store deletion and manages the persistent forget registry
that stores fingerprints of deleted documents for runtime leak detection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sura_rag.audit.schema import Base, SuraForgetRegistry
from sura_rag.exceptions import DocumentNotFoundError

if TYPE_CHECKING:
    from sura_rag.adapters.base import BaseVectorAdapter
    from sura_rag.llms.base import BaseLLMAdapter


@dataclass
class RegistryEntry:
    """A single entry in the forget registry."""

    doc_id: str
    subject: str
    fingerprint_text: str
    fingerprint_embedding: list[float]
    forgotten_at: str
    regulation: str
    requestor_id: str


class ForgetRegistry:
    """Persistent blocklist of forgotten documents.

    Stored in SQLite via SQLAlchemy. Each entry stores the document ID,
    subject, fingerprint text (first 500 chars), fingerprint embedding,
    timestamp, and regulation.

    Args:
        db_url: SQLAlchemy database URL.
    """

    def __init__(self, db_url: str = "sqlite:///sura_audit.db") -> None:
        self._engine = create_engine(db_url)
        Base.metadata.create_all(self._engine)

    def add(
        self,
        doc_id: str,
        subject: str,
        fingerprint_text: str,
        embedding: list[float],
        regulation: str,
        requestor_id: str = "",
    ) -> None:
        """Add a document to the forget registry.

        Args:
            doc_id: The document ID being forgotten.
            subject: Human-readable subject description.
            fingerprint_text: First 500 chars of the document text.
            embedding: Embedding vector of the fingerprint text.
            regulation: Applicable regulation (e.g., GDPR_Art17).
            requestor_id: Who requested the deletion.
        """
        now = datetime.now(timezone.utc)

        with Session(self._engine) as session:
            existing = (
                session.query(SuraForgetRegistry)
                .filter(SuraForgetRegistry.doc_id == doc_id)
                .first()
            )
            if existing:
                existing.subject = subject
                existing.fingerprint_text = fingerprint_text
                existing.fingerprint_embedding_json = json.dumps(embedding)
                existing.forgotten_at = now
                existing.regulation = regulation
                existing.requestor_id = requestor_id
            else:
                row = SuraForgetRegistry(
                    doc_id=doc_id,
                    subject=subject,
                    fingerprint_text=fingerprint_text,
                    fingerprint_embedding_json=json.dumps(embedding),
                    forgotten_at=now,
                    regulation=regulation,
                    requestor_id=requestor_id,
                )
                session.add(row)
            session.commit()

    def get_all(self) -> list[RegistryEntry]:
        """Retrieve all entries from the forget registry.

        Returns:
            A list of RegistryEntry instances.
        """
        with Session(self._engine) as session:
            rows = session.query(SuraForgetRegistry).all()
            return [
                RegistryEntry(
                    doc_id=row.doc_id,
                    subject=row.subject or "",
                    fingerprint_text=row.fingerprint_text or "",
                    fingerprint_embedding=json.loads(
                        row.fingerprint_embedding_json
                    )
                    if row.fingerprint_embedding_json
                    else [],
                    forgotten_at=row.forgotten_at.isoformat()
                    if row.forgotten_at
                    else "",
                    regulation=row.regulation or "",
                    requestor_id=row.requestor_id or "",
                )
                for row in rows
            ]

    def is_forgotten(self, doc_id: str) -> bool:
        """Check if a document ID is in the forget registry.

        Args:
            doc_id: The document ID to check.

        Returns:
            True if the document has been forgotten.
        """
        with Session(self._engine) as session:
            row = (
                session.query(SuraForgetRegistry)
                .filter(SuraForgetRegistry.doc_id == doc_id)
                .first()
            )
            return row is not None

    def remove(self, doc_id: str) -> None:
        """Remove a document from the forget registry. For testing only.

        Args:
            doc_id: The document ID to remove.
        """
        with Session(self._engine) as session:
            session.query(SuraForgetRegistry).filter(
                SuraForgetRegistry.doc_id == doc_id
            ).delete()
            session.commit()


class ForgetEngine:
    """Orchestrates vector store deletion and registry management.

    Handles the core forget pipeline: verify existence, retrieve text
    for fingerprinting, delete from vector store, and confirm deletion.

    Args:
        adapter: The vector store adapter.
        registry: The forget registry for storing fingerprints.
        llm: The LLM adapter for generating fingerprint embeddings.
    """

    def __init__(
        self,
        adapter: BaseVectorAdapter,
        registry: ForgetRegistry,
        llm: BaseLLMAdapter,
    ) -> None:
        self._adapter = adapter
        self._registry = registry
        self._llm = llm

    def delete(
        self,
        doc_ids: list[str],
        subject: str,
        requestor_id: str,
        regulation: str,
    ) -> dict:
        """Execute the forget pipeline: verify, fingerprint, delete, confirm.

        Args:
            doc_ids: List of document IDs to forget.
            subject: Human-readable description of the data subject.
            requestor_id: Who requested the deletion.
            regulation: Applicable regulation (e.g., GDPR_Art17).

        Returns:
            Dict with keys: deleted_ids, failed_ids, fingerprints_stored,
            doc_texts, confirmed.

        Raises:
            DocumentNotFoundError: If any doc_id is not found in the store.
        """
        deleted_ids: list[str] = []
        failed_ids: list[str] = []
        doc_texts: dict[str, str] = {}
        fingerprints_stored = 0

        # Step 1 & 2: Verify existence and retrieve text
        for doc_id in doc_ids:
            if not self._adapter.document_exists(doc_id):
                raise DocumentNotFoundError(
                    f"Document '{doc_id}' not found in the vector store. "
                    f"Cannot forget a document that doesn't exist."
                )
            doc_texts[doc_id] = self._adapter.get_document_text(doc_id)

        # Step 3: Generate and store fingerprints
        for doc_id, text in doc_texts.items():
            fingerprint_text = text[:500]
            embedding = self._llm.embed(fingerprint_text)
            self._registry.add(
                doc_id=doc_id,
                subject=subject,
                fingerprint_text=fingerprint_text,
                embedding=embedding,
                regulation=regulation,
                requestor_id=requestor_id,
            )
            fingerprints_stored += 1

        # Step 4: Delete from vector store
        try:
            self._adapter.delete(doc_ids)
        except Exception:
            failed_ids.extend(doc_ids)
            return {
                "deleted_ids": [],
                "failed_ids": failed_ids,
                "fingerprints_stored": fingerprints_stored,
                "doc_texts": doc_texts,
                "confirmed": False,
            }

        # Step 5: Confirm deletion
        confirmed = True
        for doc_id in doc_ids:
            if self._adapter.document_exists(doc_id):
                failed_ids.append(doc_id)
                confirmed = False
            else:
                deleted_ids.append(doc_id)

        return {
            "deleted_ids": deleted_ids,
            "failed_ids": failed_ids,
            "fingerprints_stored": fingerprints_stored,
            "doc_texts": doc_texts,
            "confirmed": confirmed,
        }
