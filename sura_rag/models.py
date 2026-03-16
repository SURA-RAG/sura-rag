"""
Pydantic data models for all SURA-RAG operations.

These models define the structure of results returned by the forget engine,
leak prober, guardrail scanner, audit logger, and certificate generator.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class LeakageHit(BaseModel):
    """A single instance where a probe query detected potential data leakage."""

    probe_query: str
    llm_response: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    probe_type: str  # "direct", "paraphrase", "contextual", "adversarial"
    leaked_spans: list[str]


class ProbeResult(BaseModel):
    """Aggregated result from running leak probes against forgotten documents."""

    doc_ids: list[str]
    total_probes_run: int
    leakage_hits: list[LeakageHit]
    parametric_leak_score: float = Field(ge=0.0, le=1.0)
    verdict: str  # "CLEAN", "LEAKED", "UNCERTAIN", "SKIPPED"
    probe_duration_seconds: float


class ForgetScore(BaseModel):
    """Composite score measuring how thoroughly a document was forgotten."""

    vector_deletion_confirmed: bool
    parametric_leak_score: float
    adversarial_bypass_rate: float
    composite_score: float = Field(ge=0.0, le=1.0)
    utility_delta: float = 0.0  # 0.0 in Phase 1 (no unlearning yet)


class ForgetResult(BaseModel):
    """Complete result of a forget operation including deletion, probing, and certification."""

    certificate_id: str
    doc_ids: list[str]
    subject: str
    requestor_id: str
    regulation: str
    timestamp: str  # ISO 8601
    vector_deleted: bool
    probe_result: ProbeResult
    forget_score: ForgetScore
    guardrail_activated: bool
    guardrail_mode: str
    status: str  # "completed", "failed", "leaked_guardrailed"


class ScanResult(BaseModel):
    """Result of scanning a single RAG response through the runtime guardrail."""

    original_response: str
    output: str | None  # None if HARD_BLOCK
    leaked: bool
    similarity_score: float
    leaked_spans: list[str]
    guardrail_mode: str
    action_taken: str  # "pass_through", "hard_blocked", "soft_redacted",
    # "warned_passed_through", "fallback_substituted"


class AuditEntry(BaseModel):
    """A single entry in the SURA audit log."""

    entry_id: str
    event_type: str  # "forget", "probe", "guardrail_intercept", "scan"
    timestamp: str
    doc_ids: list[str]
    requestor_id: str
    regulation: str | None
    status: str
    details: dict  # flexible JSON payload


class ComplianceCertificate(BaseModel):
    """A signed compliance certificate for a completed forget operation."""

    certificate_id: str
    issued_at: str
    issued_for: str  # requestor_id
    regulation: str
    doc_ids: list[str]
    subject: str
    forget_score: ForgetScore
    probe_summary: str
    guardrail_mode: str
    sha256_hash: str  # hash of all fields above
    raw_pdf_bytes: bytes | None = None  # set when format="pdf"

    model_config = {"arbitrary_types_allowed": True}

    def save(self, path: str) -> None:
        """Save the certificate to disk as PDF or JSON.

        Args:
            path: File path to write the certificate to. If the certificate
                  contains PDF bytes and the path ends with .pdf, writes
                  raw PDF. Otherwise writes JSON.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if self.raw_pdf_bytes and path.endswith(".pdf"):
            target.write_bytes(self.raw_pdf_bytes)
        else:
            data = self.model_dump(exclude={"raw_pdf_bytes"})
            target.write_text(json.dumps(data, indent=2, default=str))
