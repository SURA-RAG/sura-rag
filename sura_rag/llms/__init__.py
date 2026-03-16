"""LLM adapters for SURA-RAG."""

from sura_rag.llms.base import BaseLLMAdapter
from sura_rag.llms.ollama import OllamaAdapter

__all__ = [
    "BaseLLMAdapter",
    "OllamaAdapter",
]

# HuggingFaceAdapter requires optional deps — import explicitly:
# from sura_rag.llms.hf import HuggingFaceAdapter
