"""
Audit logger for SURA-RAG.

Writes all SURA events (forget, probe, guardrail intercepts) to a
persistent database via SQLAlchemy. Provides a query API with filtering.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sura_rag.audit.schema import Base, SuraAuditLog
from sura_rag.models import AuditEntry

if TYPE_CHECKING:
    from sura_rag.models import ForgetResult, ProbeResult, ScanResult


class AuditLogger:
    """Writes all SURA events to SQLite (or Postgres) and provides a query API.

    Args:
        db_url: SQLAlchemy database URL. Defaults to local SQLite file.
    """

    def __init__(self, db_url: str = "sqlite:///sura_audit.db") -> None:
        self._engine = create_engine(db_url)
        Base.metadata.create_all(self._engine)

    def _write_entry(
        self,
        event_type: str,
        doc_ids: list[str],
        requestor_id: str,
        regulation: str | None,
        status: str,
        details: dict,
    ) -> str:
        """Write a single audit entry to the database.

        Args:
            event_type: Type of event (forget, probe, guardrail_intercept, scan).
            doc_ids: Associated document IDs.
            requestor_id: Who initiated the action.
            regulation: Applicable regulation (e.g., GDPR_Art17).
            status: Outcome status.
            details: Additional JSON-serializable details.

        Returns:
            The generated entry_id (UUID).
        """
        entry_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        with Session(self._engine) as session:
            row = SuraAuditLog(
                entry_id=entry_id,
                event_type=event_type,
                timestamp=now,
                doc_ids_json=json.dumps(doc_ids),
                requestor_id=requestor_id,
                regulation=regulation,
                status=status,
                details_json=json.dumps(details, default=str),
            )
            session.add(row)
            session.commit()

        return entry_id

    def log_forget(self, result: ForgetResult) -> None:
        """Log a forget operation to the audit trail.

        Args:
            result: The ForgetResult to log.
        """
        self._write_entry(
            event_type="forget",
            doc_ids=result.doc_ids,
            requestor_id=result.requestor_id,
            regulation=result.regulation,
            status=result.status,
            details={
                "certificate_id": result.certificate_id,
                "vector_deleted": result.vector_deleted,
                "composite_score": result.forget_score.composite_score,
                "verdict": result.probe_result.verdict,
                "guardrail_mode": result.guardrail_mode,
            },
        )

    def log_probe(self, result: ProbeResult, requestor_id: str) -> None:
        """Log a probe operation to the audit trail.

        Args:
            result: The ProbeResult to log.
            requestor_id: Who initiated the probe.
        """
        self._write_entry(
            event_type="probe",
            doc_ids=result.doc_ids,
            requestor_id=requestor_id,
            regulation=None,
            status=result.verdict,
            details={
                "total_probes_run": result.total_probes_run,
                "parametric_leak_score": result.parametric_leak_score,
                "hits": len(result.leakage_hits),
                "duration_seconds": result.probe_duration_seconds,
            },
        )

    def log_guardrail(self, result: ScanResult, requestor_id: str) -> None:
        """Log a guardrail intercept to the audit trail.

        Args:
            result: The ScanResult to log.
            requestor_id: Who triggered the scan.
        """
        self._write_entry(
            event_type="guardrail_intercept",
            doc_ids=[],
            requestor_id=requestor_id,
            regulation=None,
            status="leaked" if result.leaked else "clean",
            details={
                "leaked": result.leaked,
                "similarity_score": result.similarity_score,
                "action_taken": result.action_taken,
                "guardrail_mode": result.guardrail_mode,
                "leaked_spans_count": len(result.leaked_spans),
            },
        )

    def query(
        self,
        event_type: str | None = None,
        requestor_id: str | None = None,
        regulation: str | None = None,
        since: str | None = None,
        status: str | None = None,
        as_dataframe: bool = False,
    ) -> list[AuditEntry]:
        """Query the audit log with optional filters.

        Args:
            event_type: Filter by event type.
            requestor_id: Filter by requestor.
            regulation: Filter by regulation.
            since: Filter events after this ISO date string.
            status: Filter by status.
            as_dataframe: If True, return a pandas DataFrame instead.

        Returns:
            A list of AuditEntry instances, or a DataFrame if requested.

        Raises:
            ImportError: If as_dataframe=True and pandas is not installed.
        """
        with Session(self._engine) as session:
            query = session.query(SuraAuditLog)

            if event_type:
                query = query.filter(SuraAuditLog.event_type == event_type)
            if requestor_id:
                query = query.filter(
                    SuraAuditLog.requestor_id == requestor_id
                )
            if regulation:
                query = query.filter(SuraAuditLog.regulation == regulation)
            if since:
                since_dt = datetime.fromisoformat(since)
                query = query.filter(SuraAuditLog.timestamp >= since_dt)
            if status:
                query = query.filter(SuraAuditLog.status == status)

            rows = query.order_by(SuraAuditLog.timestamp.desc()).all()

            entries = [
                AuditEntry(
                    entry_id=row.entry_id,
                    event_type=row.event_type,
                    timestamp=row.timestamp.isoformat(),
                    doc_ids=json.loads(row.doc_ids_json)
                    if row.doc_ids_json
                    else [],
                    requestor_id=row.requestor_id or "",
                    regulation=row.regulation,
                    status=row.status or "",
                    details=json.loads(row.details_json)
                    if row.details_json
                    else {},
                )
                for row in rows
            ]

        if as_dataframe:
            try:
                import pandas as pd
            except ImportError:
                raise ImportError(
                    "pandas is required for DataFrame export.\n"
                    "Install with: pip install sura-rag[pandas]"
                )
            return pd.DataFrame([e.model_dump() for e in entries])

        return entries
