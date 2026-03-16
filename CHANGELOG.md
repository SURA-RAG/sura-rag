# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-03-16

### Added

- **ForgetEngine**: Vector store deletion with persistent fingerprint registry (SQLite-backed)
- **LeakProber**: Multi-strategy probe system with 4 strategies:
  - Direct entity probes — straightforward factual questions
  - Paraphrase probes — rephrased queries using synonyms and restructuring
  - Contextual probes — indirect questions requiring document knowledge
  - Adversarial probes — jailbreak-style prompts to bypass filters
- **RuntimeGuardrail**: Real-time response scanning with 4 enforcement modes:
  - `HARD_BLOCK` — completely suppress leaked responses
  - `SOFT_BLOCK` — redact specific leaked spans
  - `WARN_AND_LOG` — pass through with audit logging
  - `FALLBACK_RESPONSE` — substitute a safe fallback message
- **AuditLogger**: Full audit trail via SQLAlchemy (SQLite or PostgreSQL)
- **CertificateGenerator**: PDF and JSON compliance certificates with SHA-256 signing
- **Vector store adapters**: ChromaDB, Qdrant, FAISS (soft-delete)
- **LLM adapters**: Ollama (fully local), HuggingFace Transformers
- **Framework connectors**: LangChain retriever wrapper, LlamaIndex query engine wrapper
- **SuraClient**: Unified API for forget, probe, guardrail, and certificate operations
- **CLI**: `sura version`, `sura audit`, `sura check` commands via Typer
- **Pydantic v2 models** for all data structures
- **54 tests** (47 unit + 7 integration) with 68% code coverage
- **CI/CD**: GitHub Actions for testing (Python 3.10/3.11/3.12 on Ubuntu + Windows) and PyPI publishing
