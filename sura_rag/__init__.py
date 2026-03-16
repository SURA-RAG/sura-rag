"""
SURA-RAG: Verified data deletion and leak detection for RAG systems.

GDPR Article 17 compliant forget pipeline with multi-strategy leak probing,
runtime guardrailing, and signed compliance certificates.
100% local, zero cloud API required.

Usage::

    import sura_rag as sr

    client = sr.SuraClient(
        vector_store=sr.adapters.ChromaDBAdapter("my_collection"),
        config=sr.SuraConfig(generator_model="llama3.2:3b"),
    )

    result = client.forget(
        doc_ids=["doc_001"],
        subject="John Smith salary records",
        requestor_id="user_4821",
        regulation="GDPR_Art17",
    )
"""

__version__ = "0.1.0"

from sura_rag.client import SuraClient
from sura_rag.config import (
    ForgetMode,
    GuardrailMode,
    ProbeStrategy,
    SuraConfig,
)

# Submodule namespaces
from sura_rag import adapters
from sura_rag import llms

__all__ = [
    "__version__",
    "SuraClient",
    "SuraConfig",
    "ForgetMode",
    "GuardrailMode",
    "ProbeStrategy",
    "adapters",
    "llms",
]
