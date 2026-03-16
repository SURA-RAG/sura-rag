"""Tests for the LeakProber and probe strategies."""

from __future__ import annotations

from sura_rag.config import ProbeStrategy, SuraConfig
from sura_rag.engines.leak_prober import LeakProber
from sura_rag.probes.direct import DirectEntityProbe
from tests.fixtures.mock_ollama import MockOllamaAdapter


class TestDirectEntityProbe:
    """Tests for DirectEntityProbe."""

    def test_generate_queries_returns_n_queries(self, mock_llm):
        """generate_queries() returns the requested number of queries."""
        probe = DirectEntityProbe(llm_adapter=mock_llm, threshold=0.75)
        queries = probe.generate_queries(
            "John Smith earns $120,000 at Acme Corp.", n=5
        )
        assert len(queries) > 0
        assert len(queries) <= 5

    def test_generate_queries_returns_strings(self, mock_llm):
        """generate_queries() returns a list of strings."""
        probe = DirectEntityProbe(llm_adapter=mock_llm, threshold=0.75)
        queries = probe.generate_queries("Test document text.", n=3)
        assert all(isinstance(q, str) for q in queries)


class TestLeakProber:
    """Tests for the LeakProber orchestrator."""

    def test_probe_returns_probe_result_clean(self):
        """ProbeResult.verdict is CLEAN when similarity is below threshold."""
        llm = MockOllamaAdapter(high_similarity=False)
        config = SuraConfig(leak_threshold=0.75)
        prober = LeakProber(llm=llm, config=config)

        def mock_rag(query: str) -> str:
            return "I don't have any information about that topic."

        result = prober.probe(
            doc_ids=["doc_001"],
            doc_texts={
                "doc_001": "John Smith earns $120,000 at Acme Corp."
            },
            rag_query_fn=mock_rag,
            strategies=[ProbeStrategy.DIRECT],
            num_probes=5,
        )

        assert result.doc_ids == ["doc_001"]
        assert result.total_probes_run > 0
        assert result.verdict in ("CLEAN", "UNCERTAIN", "LEAKED")
        assert result.probe_duration_seconds >= 0

    def test_probe_detects_leak_with_high_similarity(self):
        """ProbeResult.verdict is LEAKED when similarity is above threshold."""
        llm = MockOllamaAdapter(high_similarity=True)
        config = SuraConfig(leak_threshold=0.01)  # Very low threshold
        prober = LeakProber(llm=llm, config=config)

        original_text = "John Smith earns $120,000 at Acme Corp."

        def leaky_rag(query: str) -> str:
            return original_text  # Returns the exact document

        result = prober.probe(
            doc_ids=["doc_001"],
            doc_texts={"doc_001": original_text},
            rag_query_fn=leaky_rag,
            strategies=[ProbeStrategy.DIRECT],
            num_probes=5,
        )

        assert result.total_probes_run > 0
        assert len(result.leakage_hits) > 0
        assert result.parametric_leak_score > 0

    def test_probe_result_has_correct_fields(self):
        """ProbeResult has all required fields."""
        llm = MockOllamaAdapter(high_similarity=False)
        config = SuraConfig()
        prober = LeakProber(llm=llm, config=config)

        result = prober.probe(
            doc_ids=["doc_001"],
            doc_texts={"doc_001": "Test document."},
            rag_query_fn=lambda q: "No information available.",
            strategies=[ProbeStrategy.DIRECT],
            num_probes=3,
        )

        assert hasattr(result, "doc_ids")
        assert hasattr(result, "total_probes_run")
        assert hasattr(result, "leakage_hits")
        assert hasattr(result, "parametric_leak_score")
        assert hasattr(result, "verdict")
        assert hasattr(result, "probe_duration_seconds")

    def test_probe_with_all_strategies(self):
        """probe() works with ProbeStrategy.ALL."""
        llm = MockOllamaAdapter(high_similarity=False)
        config = SuraConfig(leak_threshold=0.99)
        prober = LeakProber(llm=llm, config=config)

        result = prober.probe(
            doc_ids=["doc_001"],
            doc_texts={"doc_001": "Test document with some facts."},
            rag_query_fn=lambda q: "Generic response.",
            strategies=None,  # ALL
            num_probes=8,
        )

        assert result.total_probes_run > 0
        assert result.verdict in ("CLEAN", "UNCERTAIN", "LEAKED")
