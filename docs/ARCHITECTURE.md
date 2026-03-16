# SURA-RAG Architecture

## Overview

SURA-RAG is a layered library designed to make RAG systems compliant with data deletion regulations (GDPR Article 17, CCPA, etc.). The architecture follows a pipeline pattern where each stage can operate independently or as part of the full forget workflow.

## Core Pipeline

```
[Forget Request] → [ForgetEngine] → [LeakProber] → [RuntimeGuardrail] → [AuditLogger] → [Certificate]
```

### 1. ForgetEngine

Handles vector store deletion and fingerprint storage. When a document is forgotten:

1. The document text is retrieved and its first 500 characters are stored as a fingerprint
2. An embedding of the fingerprint is computed and stored in the ForgetRegistry
3. The document is deleted from the vector store
4. Deletion is confirmed by checking `document_exists()` returns False

### 2. LeakProber

Runs multi-strategy probes against the RAG system to detect residual knowledge:

- **DirectEntityProbe**: Asks straightforward factual questions
- **ParaphraseProbe**: Rephrases questions using synonyms and different structures
- **ContextualProbe**: Asks indirect questions requiring document knowledge
- **AdversarialProbe**: Uses jailbreak-style prompts to bypass filters

### 3. RuntimeGuardrail

Intercepts RAG responses at runtime and checks them against stored fingerprints:

- **HARD_BLOCK**: Completely suppresses leaked responses
- **SOFT_BLOCK**: Redacts specific leaked spans
- **WARN_AND_LOG**: Passes through but logs for audit
- **FALLBACK_RESPONSE**: Substitutes a safe fallback message

### 4. Audit System

All operations are logged to a persistent database (SQLite or Postgres) with full queryability. Compliance certificates can be generated as PDF or JSON.

## Adapter Pattern

Vector stores and LLM backends are abstracted behind minimal interfaces:

- `BaseVectorAdapter`: 4 methods (delete, get_document_text, document_exists, get_collection_size)
- `BaseLLMAdapter`: 3 methods (generate, embed, is_available)

This makes it trivial to add support for new vector stores or LLM providers.

## Dependencies

The library uses a tiered dependency model:

- **Core**: chromadb, qdrant-client, faiss-cpu, ollama, sqlalchemy, reportlab, pydantic
- **CPU extras**: sentence-transformers, torch
- **Framework connectors**: langchain, llama-index
- **HF backend**: transformers, huggingface-hub
