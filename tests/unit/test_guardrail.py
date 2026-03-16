"""Tests for the RuntimeGuardrail engine."""

from __future__ import annotations

from sura_rag.config import GuardrailMode, SuraConfig
from sura_rag.engines.forget_engine import ForgetRegistry
from sura_rag.engines.guardrail import RuntimeGuardrail
from tests.fixtures.mock_ollama import MockOllamaAdapter


def _setup_guardrail(
    tmp_db_url: str,
    guardrail_mode: GuardrailMode,
    high_similarity: bool = True,
    threshold: float = 0.01,
) -> tuple[RuntimeGuardrail, ForgetRegistry]:
    """Helper to set up a guardrail with a registry entry."""
    llm = MockOllamaAdapter(high_similarity=high_similarity)
    config = SuraConfig(
        default_guardrail_mode=guardrail_mode,
        leak_threshold=threshold,
        span_threshold=0.01 if high_similarity else 0.99,
    )
    registry = ForgetRegistry(db_url=tmp_db_url)

    # Add a forgotten document
    registry.add(
        doc_id="doc_001",
        subject="John Smith",
        fingerprint_text="John Smith is a senior engineer at Acme Corp earning $120,000 per year",
        embedding=llm.embed("John Smith is a senior engineer"),
        regulation="GDPR_Art17",
    )

    guardrail = RuntimeGuardrail(registry, llm, config)
    return guardrail, registry


class TestRuntimeGuardrail:
    """Tests for RuntimeGuardrail."""

    def test_hard_block_returns_none_output(self, tmp_db_url):
        """HARD_BLOCK returns output=None."""
        guardrail, _ = _setup_guardrail(
            tmp_db_url, GuardrailMode.HARD_BLOCK
        )
        result = guardrail.scan(
            "John Smith earns $120,000 at Acme Corp."
        )
        assert result.leaked
        assert result.output is None
        assert result.action_taken == "hard_blocked"

    def test_soft_block_redacts_leaked_spans(self, tmp_db_url):
        """SOFT_BLOCK replaces leaked_spans with [REDACTED]."""
        guardrail, _ = _setup_guardrail(
            tmp_db_url, GuardrailMode.SOFT_BLOCK
        )
        result = guardrail.scan(
            "John Smith is a senior engineer at Acme Corp earning $120,000 per year"
        )
        assert result.leaked
        assert result.action_taken == "soft_redacted"
        if result.leaked_spans:
            assert "[REDACTED]" in (result.output or "")

    def test_warn_and_log_returns_original(self, tmp_db_url):
        """WARN_AND_LOG returns the original response unchanged."""
        guardrail, _ = _setup_guardrail(
            tmp_db_url, GuardrailMode.WARN_AND_LOG
        )
        original = "John Smith earns $120,000 at Acme Corp."
        result = guardrail.scan(original)
        assert result.leaked
        assert result.output == original
        assert result.action_taken == "warned_passed_through"

    def test_fallback_response_returns_fallback(self, tmp_db_url):
        """FALLBACK_RESPONSE returns the configured fallback message."""
        guardrail, _ = _setup_guardrail(
            tmp_db_url, GuardrailMode.FALLBACK_RESPONSE
        )
        result = guardrail.scan(
            "John Smith earns $120,000 at Acme Corp."
        )
        assert result.leaked
        assert result.output == SuraConfig().fallback_message
        assert result.action_taken == "fallback_substituted"

    def test_scan_returns_clean_for_unrelated_content(self, tmp_db_url):
        """scan() returns leaked=False for content unrelated to forgotten docs."""
        llm = MockOllamaAdapter(high_similarity=False)
        config = SuraConfig(
            default_guardrail_mode=GuardrailMode.HARD_BLOCK,
            leak_threshold=0.99,  # Very high threshold
        )
        registry = ForgetRegistry(db_url=tmp_db_url)
        registry.add(
            doc_id="doc_001",
            subject="John Smith",
            fingerprint_text="John Smith is a senior engineer",
            embedding=llm.embed("John Smith is a senior engineer"),
            regulation="GDPR_Art17",
        )

        guardrail = RuntimeGuardrail(registry, llm, config)
        result = guardrail.scan("The weather today is sunny and warm.")

        assert not result.leaked
        assert result.output == "The weather today is sunny and warm."
        assert result.action_taken == "pass_through"

    def test_wrap_intercepts_responses(self, tmp_db_url):
        """wrap() decorator intercepts and scans responses."""
        guardrail, _ = _setup_guardrail(
            tmp_db_url, GuardrailMode.HARD_BLOCK
        )

        def mock_rag(query: str) -> str:
            return "John Smith earns $120,000 at Acme Corp."

        wrapped = guardrail.wrap(mock_rag)
        result = wrapped("What is John's salary?")

        # Should be blocked or fallback
        assert guardrail.last_scan_result is not None
        assert guardrail.last_scan_result.leaked
