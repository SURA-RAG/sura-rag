"""Tests for Pydantic data models."""

from __future__ import annotations

import json

from sura_rag.models import (
    AuditEntry,
    ComplianceCertificate,
    ForgetResult,
    ForgetScore,
    LeakageHit,
    ProbeResult,
    ScanResult,
)


class TestLeakageHit:
    """Tests for the LeakageHit model."""

    def test_create_leakage_hit(self):
        """LeakageHit can be created with valid data."""
        hit = LeakageHit(
            probe_query="What is the salary?",
            llm_response="The salary is $120,000.",
            similarity_score=0.85,
            probe_type="direct",
            leaked_spans=["The salary is $120,000"],
        )
        assert hit.similarity_score == 0.85
        assert hit.probe_type == "direct"

    def test_serialization(self):
        """LeakageHit serializes to JSON correctly."""
        hit = LeakageHit(
            probe_query="test",
            llm_response="response",
            similarity_score=0.5,
            probe_type="paraphrase",
            leaked_spans=[],
        )
        data = hit.model_dump()
        assert "probe_query" in data
        assert data["similarity_score"] == 0.5


class TestForgetScore:
    """Tests for the ForgetScore model."""

    def test_create_forget_score(self):
        """ForgetScore can be created with valid data."""
        score = ForgetScore(
            vector_deletion_confirmed=True,
            parametric_leak_score=0.1,
            adversarial_bypass_rate=0.05,
            composite_score=0.85,
            utility_delta=0.0,
        )
        assert score.composite_score == 0.85

    def test_composite_score_bounded(self):
        """ForgetScore enforces composite_score bounds."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ForgetScore(
                vector_deletion_confirmed=True,
                parametric_leak_score=0.0,
                adversarial_bypass_rate=0.0,
                composite_score=1.5,  # Out of bounds
                utility_delta=0.0,
            )


class TestProbeResult:
    """Tests for the ProbeResult model."""

    def test_create_probe_result(self):
        """ProbeResult can be created with valid data."""
        result = ProbeResult(
            doc_ids=["doc_001"],
            total_probes_run=10,
            leakage_hits=[],
            parametric_leak_score=0.0,
            verdict="CLEAN",
            probe_duration_seconds=1.5,
        )
        assert result.verdict == "CLEAN"
        assert result.total_probes_run == 10


class TestScanResult:
    """Tests for the ScanResult model."""

    def test_scan_result_pass_through(self):
        """ScanResult with pass_through action."""
        result = ScanResult(
            original_response="Hello world",
            output="Hello world",
            leaked=False,
            similarity_score=0.1,
            leaked_spans=[],
            guardrail_mode="hard_block",
            action_taken="pass_through",
        )
        assert not result.leaked
        assert result.output == "Hello world"

    def test_scan_result_hard_blocked(self):
        """ScanResult with hard_blocked action."""
        result = ScanResult(
            original_response="Sensitive data",
            output=None,
            leaked=True,
            similarity_score=0.9,
            leaked_spans=["Sensitive data"],
            guardrail_mode="hard_block",
            action_taken="hard_blocked",
        )
        assert result.leaked
        assert result.output is None


class TestComplianceCertificate:
    """Tests for the ComplianceCertificate model."""

    def test_create_certificate(self):
        """ComplianceCertificate can be created with valid data."""
        cert = ComplianceCertificate(
            certificate_id="test-001",
            issued_at="2024-01-01T00:00:00Z",
            issued_for="user_001",
            regulation="GDPR_Art17",
            doc_ids=["doc_001"],
            subject="Test subject",
            forget_score=ForgetScore(
                vector_deletion_confirmed=True,
                parametric_leak_score=0.0,
                adversarial_bypass_rate=0.0,
                composite_score=1.0,
                utility_delta=0.0,
            ),
            probe_summary="10 probes, 0 leaks",
            guardrail_mode="hard_block",
            sha256_hash="a" * 64,
            raw_pdf_bytes=None,
        )
        assert cert.certificate_id == "test-001"
