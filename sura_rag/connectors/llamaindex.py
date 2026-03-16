"""
LlamaIndex connector for SURA-RAG.

Provides a drop-in wrapper for any LlamaIndex query engine that
scans responses through the SURA guardrail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sura_rag.client import SuraClient
    from sura_rag.models import ScanResult


class SuraQueryEngineWrapper:
    """Drop-in wrapper for any LlamaIndex query engine.

    Passes all query responses through the SURA guardrail for
    leak detection before returning them to the user.

    Usage::

        from sura_rag.connectors.llamaindex import SuraQueryEngineWrapper

        engine = SuraQueryEngineWrapper(
            base_engine=index.as_query_engine(),
            sura_client=client,
        )
        response = engine.query("What is John's salary?")

    Args:
        base_engine: The underlying LlamaIndex query engine.
        sura_client: A configured SuraClient instance.
    """

    def __init__(self, base_engine, sura_client: SuraClient) -> None:
        try:
            import llama_index  # noqa: F401
        except ImportError:
            raise ImportError(
                "LlamaIndex is not installed. Install it with:\n"
                "  pip install sura-rag[llamaindex]\n"
                "  or: pip install llama-index"
            )

        self._base_engine = base_engine
        self._sura_client = sura_client
        self.last_scan: ScanResult | None = None

    def query(self, query_str: str):
        """Query the engine and scan the response through the guardrail.

        Args:
            query_str: The query string.

        Returns:
            The base engine's response object, with text potentially
            modified by the guardrail.
        """
        response = self._base_engine.query(query_str)
        response_text = str(response)

        scan_result = self._sura_client.guardrail(response_text)
        self.last_scan = scan_result

        if scan_result.output is not None:
            # Modify the response text in-place if possible
            if hasattr(response, "response"):
                response.response = scan_result.output
        else:
            if hasattr(response, "response"):
                response.response = self._sura_client._config.fallback_message

        return response

    async def aquery(self, query_str: str):
        """Async version of query.

        Args:
            query_str: The query string.

        Returns:
            The base engine's response object, guardrailed.
        """
        if hasattr(self._base_engine, "aquery"):
            response = await self._base_engine.aquery(query_str)
        else:
            response = self._base_engine.query(query_str)

        response_text = str(response)
        scan_result = self._sura_client.guardrail(response_text)
        self.last_scan = scan_result

        if scan_result.output is not None:
            if hasattr(response, "response"):
                response.response = scan_result.output
        else:
            if hasattr(response, "response"):
                response.response = self._sura_client._config.fallback_message

        return response
