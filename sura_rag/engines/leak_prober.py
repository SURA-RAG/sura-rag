"""
Leak prober engine for SURA-RAG.

Orchestrates all four probe strategies to detect whether a RAG system
still leaks information from documents that have been forgotten.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from sura_rag.config import ProbeStrategy
from sura_rag.models import LeakageHit, ProbeResult
from sura_rag.probes.adversarial import AdversarialProbe
from sura_rag.probes.contextual import ContextualProbe
from sura_rag.probes.direct import DirectEntityProbe
from sura_rag.probes.paraphrase import ParaphraseProbe

if TYPE_CHECKING:
    from sura_rag.config import SuraConfig
    from sura_rag.llms.base import BaseLLMAdapter

_STRATEGY_MAP = {
    ProbeStrategy.DIRECT: DirectEntityProbe,
    ProbeStrategy.PARAPHRASE: ParaphraseProbe,
    ProbeStrategy.CONTEXTUAL: ContextualProbe,
    ProbeStrategy.ADVERSARIAL: AdversarialProbe,
}


class LeakProber:
    """Orchestrates all four probe strategies for leak detection.

    Runs entirely locally via the configured LLM adapter (typically Ollama).

    Args:
        llm: The LLM adapter for query generation and scoring.
        config: SURA configuration with thresholds and defaults.
    """

    def __init__(self, llm: BaseLLMAdapter, config: SuraConfig) -> None:
        self._llm = llm
        self._config = config

    def probe(
        self,
        doc_ids: list[str],
        doc_texts: dict[str, str],
        rag_query_fn: Callable[[str], str],
        strategies: list[ProbeStrategy] | None = None,
        num_probes: int = 15,
    ) -> ProbeResult:
        """Run leak probes against forgotten documents.

        For each strategy, instantiates the correct Probe class and runs
        it against the RAG system to detect information leakage.

        Args:
            doc_ids: List of document IDs being probed.
            doc_texts: Mapping of doc_id to document text.
            rag_query_fn: Callable that takes a query and returns RAG response.
            strategies: Which probe strategies to use. None means ALL.
            num_probes: Total number of probes to run across all strategies.

        Returns:
            ProbeResult with aggregated scores and verdict.
        """
        start_time = time.time()

        if strategies is None or ProbeStrategy.ALL in (strategies or []):
            active_strategies = [
                ProbeStrategy.DIRECT,
                ProbeStrategy.PARAPHRASE,
                ProbeStrategy.CONTEXTUAL,
                ProbeStrategy.ADVERSARIAL,
            ]
        else:
            active_strategies = strategies

        probes_per_strategy = max(1, num_probes // len(active_strategies))
        all_hits: list[LeakageHit] = []
        total_probes_run = 0

        for strategy in active_strategies:
            probe_cls = _STRATEGY_MAP[strategy]
            probe = probe_cls(
                llm_adapter=self._llm,
                threshold=self._config.leak_threshold,
            )

            for doc_id in doc_ids:
                doc_text = doc_texts.get(doc_id, "")
                if not doc_text:
                    continue

                hits = probe.probe(doc_text, rag_query_fn, n=probes_per_strategy)
                all_hits.extend(hits)
                total_probes_run += probes_per_strategy

        # Compute scores
        parametric_leak_score = (
            len(all_hits) / total_probes_run if total_probes_run > 0 else 0.0
        )

        # Determine verdict
        if parametric_leak_score > 0.2:
            verdict = "LEAKED"
        elif parametric_leak_score > 0.05:
            verdict = "UNCERTAIN"
        else:
            verdict = "CLEAN"

        duration = time.time() - start_time

        return ProbeResult(
            doc_ids=doc_ids,
            total_probes_run=total_probes_run,
            leakage_hits=all_hits,
            parametric_leak_score=round(parametric_leak_score, 4),
            verdict=verdict,
            probe_duration_seconds=round(duration, 2),
        )
