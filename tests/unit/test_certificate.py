"""Tests for the CertificateGenerator."""

from __future__ import annotations

from sura_rag.audit.certificate import CertificateGenerator
from sura_rag.models import (
    ForgetResult,
    ForgetScore,
    ProbeResult,
)


def _make_forget_result() -> ForgetResult:
    """Create a sample ForgetResult for testing."""
    return ForgetResult(
        certificate_id="cert-test-001",
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


class TestCertificateGenerator:
    """Tests for CertificateGenerator."""

    def test_generate_json_certificate(self, tmp_db_url):
        """generate() creates a JSON certificate."""
        gen = CertificateGenerator(db_url=tmp_db_url)
        result = _make_forget_result()
        cert = gen.generate(result, format="json")

        assert cert.certificate_id == "cert-test-001"
        assert cert.issued_for == "user_001"
        assert cert.regulation == "GDPR_Art17"
        assert len(cert.sha256_hash) == 64
        assert cert.raw_pdf_bytes is None

    def test_generate_pdf_certificate(self, tmp_db_url):
        """generate() creates a PDF certificate with raw bytes."""
        gen = CertificateGenerator(db_url=tmp_db_url)
        result = _make_forget_result()
        cert = gen.generate(result, format="pdf")

        assert cert.raw_pdf_bytes is not None
        assert len(cert.raw_pdf_bytes) > 0
        # PDF magic bytes
        assert cert.raw_pdf_bytes[:4] == b"%PDF"

    def test_certificate_save_json(self, tmp_db_url, tmp_path):
        """ComplianceCertificate.save() writes JSON file."""
        gen = CertificateGenerator(db_url=tmp_db_url)
        result = _make_forget_result()
        cert = gen.generate(result, format="json")

        path = str(tmp_path / "cert.json")
        cert.save(path)

        import json
        with open(path) as f:
            data = json.load(f)
        assert data["certificate_id"] == "cert-test-001"

    def test_certificate_save_pdf(self, tmp_db_url, tmp_path):
        """ComplianceCertificate.save() writes PDF file."""
        gen = CertificateGenerator(db_url=tmp_db_url)
        result = _make_forget_result()
        cert = gen.generate(result, format="pdf")

        path = str(tmp_path / "cert.pdf")
        cert.save(path)

        with open(path, "rb") as f:
            assert f.read(4) == b"%PDF"

    def test_certificate_persisted_in_db(self, tmp_db_url):
        """generate() stores the certificate in the database."""
        gen = CertificateGenerator(db_url=tmp_db_url)
        result = _make_forget_result()
        gen.generate(result, format="json")

        retrieved = gen.get_certificate("cert-test-001")
        assert retrieved is not None
        assert retrieved.certificate_id == "cert-test-001"

    def test_get_certificate_returns_none_for_missing(self, tmp_db_url):
        """get_certificate() returns None for non-existent ID."""
        gen = CertificateGenerator(db_url=tmp_db_url)
        assert gen.get_certificate("nonexistent") is None
