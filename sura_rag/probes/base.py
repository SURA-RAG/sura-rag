"""
Abstract base class for leak detection probes.

Probes generate targeted queries designed to detect whether a RAG system
still retains information from documents that should have been forgotten.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sura_rag.models import LeakageHit

if TYPE_CHECKING:
    from collections.abc import Callable

    from sura_rag.llms.base import BaseLLMAdapter


class BaseProbe(ABC):
    """Abstract base class for all leak detection probes.

    Each probe type implements a different strategy for generating
    queries that attempt to extract forgotten information from the RAG system.

    Args:
        llm_adapter: The LLM adapter used for query generation and scoring.
        threshold: Similarity threshold above which a response is considered
                   a leak (0.0–1.0).
    """

    def __init__(
        self, llm_adapter: BaseLLMAdapter, threshold: float = 0.75
    ) -> None:
        self.llm = llm_adapter
        self.threshold = threshold

    @abstractmethod
    def generate_queries(self, doc_text: str, n: int = 5) -> list[str]:
        """Generate n probe queries targeting the forgotten document.

        Args:
            doc_text: The original document text to probe for.
            n: Number of queries to generate.

        Returns:
            A list of probe query strings.
        """

    @abstractmethod
    def score_response(
        self, doc_text: str, response: str
    ) -> tuple[float, list[str]]:
        """Score a RAG response for information leakage.

        Args:
            doc_text: The original document text that should be forgotten.
            response: The RAG system's response to a probe query.

        Returns:
            A tuple of (similarity_score, leaked_spans) where leaked_spans
            are specific sentences from the response that match the document.
        """

    def probe(
        self, doc_text: str, rag_query_fn: Callable[[str], str], n: int = 5
    ) -> list[LeakageHit]:
        """Generate queries, run them through the RAG system, and detect leaks.

        Args:
            doc_text: The original document text to probe for.
            rag_query_fn: A callable that takes a query string and returns
                          the RAG system's response.
            n: Number of probe queries to generate.

        Returns:
            A list of LeakageHit instances for responses above the threshold.
        """
        hits: list[LeakageHit] = []
        queries = self.generate_queries(doc_text, n)
        for query in queries:
            response = rag_query_fn(query)
            score, spans = self.score_response(doc_text, response)
            if score >= self.threshold:
                hits.append(
                    LeakageHit(
                        probe_query=query,
                        llm_response=response,
                        similarity_score=round(score, 4),
                        probe_type=self.__class__.__name__,
                        leaked_spans=spans,
                    )
                )
        return hits
