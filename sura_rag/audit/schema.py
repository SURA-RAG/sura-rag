"""
SQLAlchemy ORM schema for SURA-RAG audit tables.

Defines the database tables for audit logs, forget registry,
and compliance certificates.
"""

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all SURA tables."""


class SuraAuditLog(Base):
    """Audit log table recording all SURA events."""

    __tablename__ = "sura_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_id = Column(String(36), unique=True, nullable=False)
    event_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    doc_ids_json = Column(Text)
    requestor_id = Column(String(255))
    regulation = Column(String(100))
    status = Column(String(50))
    details_json = Column(Text)


class SuraForgetRegistry(Base):
    """Forget registry table storing fingerprints of deleted documents."""

    __tablename__ = "sura_forget_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(500), unique=True, nullable=False)
    subject = Column(String(500))
    fingerprint_text = Column(Text)
    fingerprint_embedding_json = Column(Text)
    forgotten_at = Column(DateTime)
    regulation = Column(String(100))
    requestor_id = Column(String(255))


class SuraCertificates(Base):
    """Compliance certificates table."""

    __tablename__ = "sura_certificates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    certificate_id = Column(String(36), unique=True)
    issued_at = Column(DateTime)
    issued_for = Column(String(255))
    regulation = Column(String(100))
    doc_ids_json = Column(Text)
    subject = Column(String(500))
    forget_score_json = Column(Text)
    sha256_hash = Column(String(64))
