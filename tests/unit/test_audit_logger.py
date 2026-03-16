"""Tests for the AuditLogger."""

from __future__ import annotations

from sura_rag.audit.logger import AuditLogger
from sura_rag.models import (
    ForgetResult,
    ForgetScore,
    ProbeResult,
    ScanResult,
)


def _make_forget_result() -> ForgetResult:
    """Create a sample ForgetResult for testing."""
    return ForgetResult(
        certificate_id="test-cert-001",
        doc_ids=["doc_001"],
        subject="John Smith",
        requestor_id="user_001",
        regulation="GDPR_Art17",
        timestamp="2024-01-01T00:00:00Z",
        vector_deleted=True,
        probe_result=ProbeResult(
            doc_ids=["doc_001"],
            total_probes_run=10,
            leakage_hits=[],
            parametric_leak_score=0.0,
            verdict="CLEAN",
            probe_duration_seconds=1.5,
        ),
        forget_score=ForgetScore(
            vector_deletion_confirmed=True,
            parametric_leak_score=0.0,
            adversarial_bypass_rate=0.0,
            composite_score=1.0,
            utility_delta=0.0,
        ),
        guardrail_activated=False,
        guardrail_mode="hard_block",
        status="completed",
    )


class TestAuditLogger:
    """Tests for the AuditLogger."""

    def test_log_forget_creates_entry(self, tmp_db_url):
        """log_forget() writes an entry to the audit log."""
        logger = AuditLogger(db_url=tmp_db_url)
        result = _make_forget_result()
        logger.log_forget(result)

        entries = logger.query(event_type="forget")
        assert len(entries) == 1
        assert entries[0].event_type == "forget"
        assert entries[0].doc_ids == ["doc_001"]

    def test_log_probe_creates_entry(self, tmp_db_url):
        """log_probe() writes an entry to the audit log."""
        logger = AuditLogger(db_url=tmp_db_url)
        probe_result = ProbeResult(
            doc_ids=["doc_001"],
            total_probes_run=5,
            leakage_hits=[],
            parametric_leak_score=0.0,
            verdict="CLEAN",
            probe_duration_seconds=0.5,
        )
        logger.log_probe(probe_result, requestor_id="user_001")

        entries = logger.query(event_type="probe")
        assert len(entries) == 1

    def test_log_guardrail_creates_entry(self, tmp_db_url):
        """log_guardrail() writes an entry to the audit log."""
        logger = AuditLogger(db_url=tmp_db_url)
        scan_result = ScanResult(
            original_response="test response",
            output=None,
            leaked=True,
            similarity_score=0.85,
            leaked_spans=["test response"],
            guardrail_mode="hard_block",
            action_taken="hard_blocked",
        )
        logger.log_guardrail(scan_result, requestor_id="user_001")

        entries = logger.query(event_type="guardrail_intercept")
        assert len(entries) == 1
        assert entries[0].status == "leaked"

    def test_query_filters_by_requestor(self, tmp_db_url):
        """query() filters by requestor_id correctly."""
        logger = AuditLogger(db_url=tmp_db_url)
        result = _make_forget_result()
        logger.log_forget(result)

        entries = logger.query(requestor_id="user_001")
        assert len(entries) == 1

        entries = logger.query(requestor_id="unknown")
        assert len(entries) == 0

    def test_query_filters_by_regulation(self, tmp_db_url):
        """query() filters by regulation correctly."""
        logger = AuditLogger(db_url=tmp_db_url)
        result = _make_forget_result()
        logger.log_forget(result)

        entries = logger.query(regulation="GDPR_Art17")
        assert len(entries) == 1

        entries = logger.query(regulation="CCPA")
        assert len(entries) == 0
