# sura-rag

[![PyPI version](https://img.shields.io/pypi/v/sura-rag.svg)](https://pypi.org/project/sura-rag/)
[![Downloads](https://img.shields.io/pypi/dm/sura-rag.svg)](https://pypi.org/project/sura-rag/)
[![Python versions](https://img.shields.io/pypi/pyversions/sura-rag.svg)](https://pypi.org/project/sura-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/SURA-RAG/sura-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/SURA-RAG/sura-rag/actions/workflows/ci.yml)

**Verified data deletion and runtime leak detection for RAG systems.** GDPR Article 17 compliant forget pipeline with multi-strategy leak probing, runtime guardrailing, and signed compliance certificates. 100% local, zero cloud API required.

---

## The Problem

RAG systems retrieve and present data from vector stores, but when a user exercises their GDPR Article 17 "right to be forgotten," simply deleting a document from the vector store is not enough. The LLM may have memorized fragments during retrieval, cached chunks may persist, and there is no way to verify that the data is truly gone.

**sura-rag** closes this gap with a complete forget pipeline:

```
Delete → Probe → Guardrail → Certify
```

## Quick Install

```bash
# Core (Ollama-based, no GPU required)
pip install sura-rag

# With CPU embeddings (sentence-transformers)
pip install "sura-rag[cpu]"

# With CUDA support (pre-install CUDA torch first)
pip install "sura-rag[cuda]"

# With framework connectors
pip install "sura-rag[langchain]"
pip install "sura-rag[llamaindex]"

# Everything
pip install "sura-rag[all]"
```

## 30-Second Quickstart

```python
import sura_rag as sr

# Connect to your vector store
client = sr.SuraClient(
    vector_store=sr.adapters.ChromaDBAdapter("my_collection"),
    config=sr.SuraConfig(generator_model="llama3.2:3b"),
)

# Forget a document (GDPR Article 17)
result = client.forget(
    doc_ids=["doc_001"],
    subject="John Smith salary records",
    requestor_id="user_4821",
    regulation="GDPR_Art17",
)

print(f"Score: {result.forget_score.composite_score}")  # 0.0–1.0
print(f"Status: {result.status}")                       # "completed"
print(f"Certificate: {result.certificate_id}")          # UUID
```

## How It Works

1. **Delete** — Documents are removed from the vector store and their fingerprints (text + embedding) are stored in a persistent registry.
2. **Probe** — Four probe strategies (direct, paraphrase, contextual, adversarial) query the RAG system to detect residual knowledge leakage.
3. **Guardrail** — A runtime guardrail continuously scans all RAG responses against the fingerprint registry, blocking or redacting any leaked content.
4. **Certify** — A signed compliance certificate (PDF or JSON) with SHA-256 hash is generated as auditable proof of deletion.

## Features

| Feature | Status |
|---------|:------:|
| Vector store deletion (ChromaDB, Qdrant, FAISS) | ✅ |
| Fingerprint registry (SQLite-backed) | ✅ |
| Direct entity probes | ✅ |
| Paraphrase probes | ✅ |
| Contextual probes | ✅ |
| Adversarial probes (jailbreak detection) | ✅ |
| Runtime guardrail — 4 modes (hard block, soft block, warn & log, fallback) | ✅ |
| Audit logging (SQLite / PostgreSQL) | ✅ |
| PDF & JSON compliance certificates | ✅ |
| LangChain retriever connector | ✅ |
| LlamaIndex query engine connector | ✅ |
| LLM backends — Ollama (local) & HuggingFace Transformers | ✅ |
| CLI tool (`sura version`, `sura audit`, `sura check`) | ✅ |

### Planned (Phase 2)

- Parametric unlearning via LoRA fine-tuning
- TOFU benchmark evaluation suite
- Multi-GPU training support

## Compatibility

| Component | Supported |
|-----------|-----------|
| ChromaDB | ≥ 0.5.0 |
| Qdrant | ≥ 1.9.0 |
| FAISS | ≥ 1.8.0 (soft-delete) |
| LangChain | ≥ 0.2.0 |
| LlamaIndex | ≥ 0.10.0 |
| Ollama | ≥ 0.2.0 |
| HuggingFace Transformers | ≥ 4.40.0 |
| PyTorch | ≥ 2.2.0 |
| Python | 3.10 · 3.11 · 3.12 |
| OS | Windows · Linux · macOS |

## Local Setup

### 1. Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from https://ollama.com
```

### 2. Pull models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 3. Start Ollama

```bash
ollama serve
```

### 4. Install sura-rag

```bash
pip install sura-rag

# or for development:
git clone https://github.com/SURA-RAG/sura-rag.git
cd sura-rag
pip install -e ".[dev,cpu]"
```

### 5. Verify installation

```bash
sura version    # prints sura-rag version
sura check      # checks if Ollama is running
```

### 6. Run tests

```bash
pytest tests/unit/ -v
```

## Environment Setup

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

The `.env` file is in `.gitignore` and will never be committed. For the default Ollama backend, no API tokens are required. See `.env.example` for all available settings.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run tests: `pytest tests/unit/ -v`
4. Run linting: `ruff check sura_rag/`
5. Submit a pull request

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use sura-rag in academic research, please cite:

```bibtex
@software{sura_rag_2025,
  title = {sura-rag: Verified Data Deletion and Leak Detection for RAG Systems},
  author = {Saxena, Aditya},
  year = {2025},
  url = {https://github.com/SURA-RAG/sura-rag},
  license = {MIT},
}
```
