"""
LangChain connector for SURA-RAG.

Provides a drop-in retriever replacement that filters out chunks
belonging to forgotten documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sura_rag.client import SuraClient


class SuraRetriever:
    """Drop-in replacement for any LangChain retriever.

    Wraps an existing retriever and filters out chunks belonging to
    forgotten documents. Also scans responses through the guardrail.

    Usage::

        from sura_rag.connectors.langchain import SuraRetriever

        retriever = SuraRetriever(
            base_retriever=chroma.as_retriever(),
            sura_client=client,
            filter_mode="hard_block",
        )
        # Use retriever in any LangChain chain — unchanged

    Args:
        base_retriever: The underlying LangChain retriever.
        sura_client: A configured SuraClient instance.
        filter_mode: How to handle forgotten documents
                     (\"hard_block\", \"soft_block\", \"warn_and_log\").
    """

    def __init__(
        self,
        base_retriever,
        sura_client: SuraClient,
        filter_mode: str = "hard_block",
    ) -> None:
        try:
            from langchain_core.retrievers import BaseRetriever  # noqa: F401
        except ImportError:
            raise ImportError(
                "LangChain is not installed. Install it with:\n"
                "  pip install sura-rag[langchain]\n"
                "  or: pip install langchain langchain-community"
            )

        self._base_retriever = base_retriever
        self._sura_client = sura_client
        self._filter_mode = filter_mode
        self._registry = sura_client._registry

    def get_relevant_documents(self, query: str) -> list:
        """Retrieve documents and filter out forgotten ones.

        Calls the base retriever, then removes any documents whose IDs
        are in the forget registry.

        Args:
            query: The search query.

        Returns:
            Filtered list of LangChain Document objects.
        """
        docs = self._base_retriever.get_relevant_documents(query)
        filtered = []
        for doc in docs:
            doc_id = doc.metadata.get("id") or doc.metadata.get("doc_id", "")
            if doc_id and self._registry.is_forgotten(doc_id):
                continue
            filtered.append(doc)
        return filtered

    async def aget_relevant_documents(self, query: str) -> list:
        """Async version of get_relevant_documents.

        Args:
            query: The search query.

        Returns:
            Filtered list of LangChain Document objects.
        """
        if hasattr(self._base_retriever, "aget_relevant_documents"):
            docs = await self._base_retriever.aget_relevant_documents(query)
        else:
            docs = self._base_retriever.get_relevant_documents(query)

        filtered = []
        for doc in docs:
            doc_id = doc.metadata.get("id") or doc.metadata.get("doc_id", "")
            if doc_id and self._registry.is_forgotten(doc_id):
                continue
            filtered.append(doc)
        return filtered
