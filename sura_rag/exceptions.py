"""
Custom exceptions for the SURA-RAG library.

All exceptions inherit from SuraError, making it easy to catch
any SURA-specific error with a single except clause.
"""


class SuraError(Exception):
    """Base exception for all SURA-RAG errors."""


class OllamaNotRunningError(SuraError):
    """Raised when the Ollama server is unreachable.

    Start Ollama with: ollama serve
    """


class AdapterNotConfiguredError(SuraError):
    """Raised when a vector database adapter is not properly configured."""


class DocumentNotFoundError(SuraError):
    """Raised when a document ID is not found in the vector store."""


class LeakDetectedError(SuraError):
    """Raised in strict mode when a data leak is detected after deletion."""


class CertificateGenerationError(SuraError):
    """Raised when PDF certificate generation fails."""


class UnlearnerNotAvailableError(SuraError):
    """Raised when a Phase 2 feature (parametric unlearning) is called.

    Parametric unlearning (LoRA-based) is planned for Phase 2.
    Phase 1 supports retrieval-layer deletion and runtime guardrailing only.
    """
