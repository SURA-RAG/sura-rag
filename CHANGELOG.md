# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-01

### Added

- Initial release of sura-rag
- ForgetEngine: vector store deletion with fingerprint registry
- LeakProber: multi-strategy probe system (direct, paraphrase, contextual, adversarial)
- RuntimeGuardrail: real-time response scanning with 4 modes (hard block, soft block, warn & log, fallback)
- AuditLogger: SQLite/Postgres audit trail for all operations
- CertificateGenerator: PDF and JSON compliance certificates
- Vector store adapters: ChromaDB, Qdrant, FAISS
- LLM adapters: Ollama (local), HuggingFace Transformers
- Framework connectors: LangChain retriever, LlamaIndex query engine wrapper
- SuraClient: unified API for all operations
- Pydantic models for all data structures
- Comprehensive test suite
