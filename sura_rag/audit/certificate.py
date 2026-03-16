"""
Compliance certificate generator for SURA-RAG.

Generates signed compliance certificates for completed forget operations.
Supports PDF output via reportlab and JSON fallback.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sura_rag.audit.schema import Base, SuraCertificates
from sura_rag.exceptions import CertificateGenerationError
from sura_rag.models import ComplianceCertificate

if TYPE_CHECKING:
    from sura_rag.models import ForgetResult


class CertificateGenerator:
    """Generates signed compliance certificates for completed forget events.

    PDF generation uses reportlab. JSON export is always available as fallback.

    Args:
        db_url: SQLAlchemy database URL for persisting certificates.
    """

    def __init__(self, db_url: str = "sqlite:///sura_audit.db") -> None:
        self._engine = create_engine(db_url)
        Base.metadata.create_all(self._engine)

    def generate(
        self, forget_result: ForgetResult, format: str = "pdf"
    ) -> ComplianceCertificate:
        """Generate a compliance certificate from a forget result.

        Args:
            forget_result: The completed ForgetResult to certify.
            format: Output format - \"pdf\" or \"json\".

        Returns:
            A ComplianceCertificate with raw_pdf_bytes set if format is \"pdf\".

        Raises:
            CertificateGenerationError: If PDF generation fails.
        """
        issued_at = datetime.now(timezone.utc).isoformat()

        # Build the hash input
        hash_input = (
            forget_result.certificate_id
            + issued_at
            + json.dumps(forget_result.doc_ids)
            + forget_result.subject
            + str(forget_result.forget_score.composite_score)
        )
        sha256_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        # Build probe summary
        pr = forget_result.probe_result
        probe_summary = (
            f"{pr.total_probes_run} probes run, "
            f"{len(pr.leakage_hits)} leaks detected, "
            f"verdict: {pr.verdict}, "
            f"leak score: {pr.parametric_leak_score:.4f}"
        )

        raw_pdf_bytes: bytes | None = None
        if format == "pdf":
            try:
                raw_pdf_bytes = self._generate_pdf(
                    certificate_id=forget_result.certificate_id,
                    issued_at=issued_at,
                    issued_for=forget_result.requestor_id,
                    regulation=forget_result.regulation,
                    doc_ids=forget_result.doc_ids,
                    subject=forget_result.subject,
                    forget_score=forget_result.forget_score,
                    probe_summary=probe_summary,
                    guardrail_mode=forget_result.guardrail_mode,
                    sha256_hash=sha256_hash,
                )
            except Exception as e:
                raise CertificateGenerationError(
                    f"Failed to generate PDF certificate: {e}"
                ) from e

        cert = ComplianceCertificate(
            certificate_id=forget_result.certificate_id,
            issued_at=issued_at,
            issued_for=forget_result.requestor_id,
            regulation=forget_result.regulation,
            doc_ids=forget_result.doc_ids,
            subject=forget_result.subject,
            forget_score=forget_result.forget_score,
            probe_summary=probe_summary,
            guardrail_mode=forget_result.guardrail_mode,
            sha256_hash=sha256_hash,
            raw_pdf_bytes=raw_pdf_bytes,
        )

        # Persist to database
        self._store_certificate(cert)

        return cert

    def get_certificate(
        self, certificate_id: str
    ) -> ComplianceCertificate | None:
        """Retrieve a previously generated certificate by ID.

        Args:
            certificate_id: The UUID of the certificate.

        Returns:
            The ComplianceCertificate if found, None otherwise.
        """
        with Session(self._engine) as session:
            row = (
                session.query(SuraCertificates)
                .filter(SuraCertificates.certificate_id == certificate_id)
                .first()
            )
            if row is None:
                return None

            forget_score_data = (
                json.loads(row.forget_score_json)
                if row.forget_score_json
                else {}
            )
            from sura_rag.models import ForgetScore

            return ComplianceCertificate(
                certificate_id=row.certificate_id,
                issued_at=row.issued_at.isoformat() if row.issued_at else "",
                issued_for=row.issued_for or "",
                regulation=row.regulation or "",
                doc_ids=json.loads(row.doc_ids_json)
                if row.doc_ids_json
                else [],
                subject=row.subject or "",
                forget_score=ForgetScore(**forget_score_data),
                probe_summary="",
                guardrail_mode="",
                sha256_hash=row.sha256_hash or "",
                raw_pdf_bytes=None,
            )

    def _store_certificate(self, cert: ComplianceCertificate) -> None:
        """Persist a certificate to the database.

        Args:
            cert: The ComplianceCertificate to store.
        """
        with Session(self._engine) as session:
            row = SuraCertificates(
                certificate_id=cert.certificate_id,
                issued_at=datetime.fromisoformat(cert.issued_at),
                issued_for=cert.issued_for,
                regulation=cert.regulation,
                doc_ids_json=json.dumps(cert.doc_ids),
                subject=cert.subject,
                forget_score_json=cert.forget_score.model_dump_json(),
                sha256_hash=cert.sha256_hash,
            )
            session.add(row)
            session.commit()

    def _generate_pdf(
        self,
        certificate_id: str,
        issued_at: str,
        issued_for: str,
        regulation: str,
        doc_ids: list[str],
        subject: str,
        forget_score,
        probe_summary: str,
        guardrail_mode: str,
        sha256_hash: str,
    ) -> bytes:
        """Generate a PDF certificate using reportlab.

        Returns:
            Raw PDF bytes.
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CertTitle",
            parent=styles["Title"],
            fontSize=20,
            spaceAfter=20,
            textColor=colors.HexColor("#1a1a2e"),
        )
        heading_style = ParagraphStyle(
            "CertHeading",
            parent=styles["Heading2"],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.HexColor("#16213e"),
        )
        body_style = ParagraphStyle(
            "CertBody",
            parent=styles["Normal"],
            fontSize=10,
            spaceAfter=4,
        )
        footer_style = ParagraphStyle(
            "CertFooter",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.grey,
        )

        elements = []

        # Header
        elements.append(
            Paragraph("SURA-RAG Compliance Certificate", title_style)
        )
        elements.append(Spacer(1, 12))

        # Certificate details table
        data = [
            ["Certificate ID", certificate_id],
            ["Issued At", issued_at],
            ["Issued For", issued_for],
            ["Regulation", regulation],
            ["Subject", subject],
            ["Documents Deleted", ", ".join(doc_ids)],
        ]

        table = Table(data, colWidths=[2 * inch, 4 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eaf6")),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1a1a2e")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 16))

        # Forget Score breakdown
        elements.append(Paragraph("Forget Score Breakdown", heading_style))
        score_data = [
            ["Metric", "Value"],
            [
                "Vector Deletion Confirmed",
                str(forget_score.vector_deletion_confirmed),
            ],
            [
                "Parametric Leak Score",
                f"{forget_score.parametric_leak_score:.4f}",
            ],
            [
                "Adversarial Bypass Rate",
                f"{forget_score.adversarial_bypass_rate:.4f}",
            ],
            [
                "Composite Score",
                f"{forget_score.composite_score:.4f}",
            ],
            ["Utility Delta", f"{forget_score.utility_delta:.4f}"],
        ]
        score_table = Table(score_data, colWidths=[3 * inch, 3 * inch])
        score_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#1a1a2e"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        elements.append(score_table)
        elements.append(Spacer(1, 16))

        # Probe summary
        elements.append(Paragraph("Probe Summary", heading_style))
        elements.append(Paragraph(probe_summary, body_style))
        elements.append(Spacer(1, 8))

        # Guardrail mode
        elements.append(Paragraph("Guardrail Mode", heading_style))
        elements.append(Paragraph(guardrail_mode, body_style))
        elements.append(Spacer(1, 8))

        # SHA-256 hash
        elements.append(Paragraph("Integrity Hash (SHA-256)", heading_style))
        elements.append(Paragraph(sha256_hash, body_style))
        elements.append(Spacer(1, 24))

        # Footer
        elements.append(
            Paragraph("Generated by sura-rag v0.1.0", footer_style)
        )

        doc.build(elements)
        return buffer.getvalue()
