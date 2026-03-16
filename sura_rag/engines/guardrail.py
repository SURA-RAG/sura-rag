"""
Runtime guardrail engine for SURA-RAG.

Wraps any RAG query function and intercepts responses at runtime,
checking every response against the ForgetRegistry for leaked content.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from sura_rag.config import GuardrailMode
from sura_rag.guardrails.fallback import fallback_response
from sura_rag.guardrails.hard_block import hard_block
from sura_rag.guardrails.soft_block import soft_block
from sura_rag.guardrails.warn_log import warn_and_log
from sura_rag.llms.ollama import cosine_similarity
from sura_rag.models import ScanResult

if TYPE_CHECKING:
    from sura_rag.config import SuraConfig
    from sura_rag.engines.forget_engine import ForgetRegistry
    from sura_rag.llms.base import BaseLLMAdapter


class RuntimeGuardrail:
    """Wraps any RAG query function and intercepts responses at runtime.

    Checks every response against the ForgetRegistry by computing
    embedding similarity against stored fingerprints.

    Args:
        registry: The forget registry containing deleted document fingerprints.
        llm: The LLM adapter for computing embeddings.
        config: SURA configuration with thresholds and guardrail settings.
    """

    def __init__(
        self,
        registry: ForgetRegistry,
        llm: BaseLLMAdapter,
        config: SuraConfig,
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._config = config
        self.last_scan_result: ScanResult | None = None

    def scan(self, response: str) -> ScanResult:
        """Scan a single response against the forget registry.

        Embeds the response and compares it against all stored fingerprint
        embeddings. If similarity exceeds the threshold, applies the
        configured guardrail mode.

        Args:
            response: The RAG system's response to scan.

        Returns:
            ScanResult indicating whether leakage was detected and what
            action was taken.
        """
        response_embedding = self._llm.embed(response[:500])

        entries = self._registry.get_all()
        best_score = 0.0
        best_entry = None

        for entry in entries:
            if not entry.fingerprint_embedding:
                continue
            sim = cosine_similarity(response_embedding, entry.fingerprint_embedding)
            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_score >= self._config.leak_threshold and best_entry is not None:
            # Extract leaked spans at sentence level
            leaked_spans = self._extract_leaked_spans(
                response, best_entry.fingerprint_text
            )

            output, action_taken = self._apply_mode(
                response,
                self._config.default_guardrail_mode,
                leaked_spans,
            )

            result = ScanResult(
                original_response=response,
                output=output,
                leaked=True,
                similarity_score=round(best_score, 4),
                leaked_spans=leaked_spans,
                guardrail_mode=self._config.default_guardrail_mode.value,
                action_taken=action_taken,
            )
        else:
            result = ScanResult(
                original_response=response,
                output=response,
                leaked=False,
                similarity_score=round(best_score, 4),
                leaked_spans=[],
                guardrail_mode=self._config.default_guardrail_mode.value,
                action_taken="pass_through",
            )

        self.last_scan_result = result
        return result

    def wrap(self, rag_fn: Callable[[str], str]) -> Callable[[str], str]:
        """Return a guardrail-wrapped version of any RAG function.

        Every response from the wrapped function is automatically scanned
        against the forget registry.

        Args:
            rag_fn: The RAG query function to wrap.

        Returns:
            A wrapped callable that scans every response.
        """

        def wrapped(query: str) -> str:
            response = rag_fn(query)
            result = self.scan(response)
            self.last_scan_result = result
            if result.output is None:
                return self._config.fallback_message
            return result.output

        return wrapped

    def _apply_mode(
        self,
        response: str,
        mode: GuardrailMode,
        leaked_spans: list[str],
    ) -> tuple[str | None, str]:
        """Apply the specified guardrail mode to a leaked response.

        Args:
            response: The original response.
            mode: Which guardrail mode to apply.
            leaked_spans: The specific leaked text spans.

        Returns:
            Tuple of (processed_output, action_taken_label).
        """
        if mode == GuardrailMode.HARD_BLOCK:
            return hard_block(response, leaked_spans)
        elif mode == GuardrailMode.SOFT_BLOCK:
            return soft_block(response, leaked_spans)
        elif mode == GuardrailMode.WARN_AND_LOG:
            return warn_and_log(response, leaked_spans)
        elif mode == GuardrailMode.FALLBACK_RESPONSE:
            return fallback_response(
                response, leaked_spans, self._config.fallback_message
            )
        else:
            return hard_block(response, leaked_spans)

    def _extract_leaked_spans(
        self, response: str, fingerprint_text: str
    ) -> list[str]:
        """Extract specific leaked sentence spans from the response.

        Compares each sentence in the response against each sentence in
        the fingerprint text using embedding similarity.

        Args:
            response: The RAG response to check.
            fingerprint_text: The stored fingerprint of the forgotten document.

        Returns:
            List of response sentences that match the fingerprint above
            the span threshold.
        """
        leaked_spans: list[str] = []
        resp_sentences = [s.strip() for s in response.split(". ") if s.strip()]
        doc_sentences = [
            s.strip() for s in fingerprint_text.split(". ") if s.strip()
        ]

        for resp_sent in resp_sentences:
            if len(resp_sent) < 10:
                continue
            resp_vec = self._llm.embed(resp_sent)
            for doc_sent in doc_sentences:
                if len(doc_sent) < 10:
                    continue
                doc_vec = self._llm.embed(doc_sent)
                sim = cosine_similarity(resp_vec, doc_vec)
                if sim >= self._config.span_threshold:
                    leaked_spans.append(resp_sent)
                    break

        return leaked_spans
