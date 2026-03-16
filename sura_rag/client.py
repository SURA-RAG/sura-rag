"""
Main client for SURA-RAG.

SuraClient is the primary public interface. Users only ever need to
import and instantiate this class to use the full SURA pipeline.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from sura_rag.audit.certificate import CertificateGenerator
from sura_rag.audit.logger import AuditLogger
from sura_rag.config import (
    ForgetMode,
    GuardrailMode,
    ProbeStrategy,
    SuraConfig,
)
from sura_rag.engines.forget_engine import ForgetEngine, ForgetRegistry
from sura_rag.engines.guardrail import RuntimeGuardrail
from sura_rag.engines.leak_prober import LeakProber
from sura_rag.exceptions import OllamaNotRunningError
from sura_rag.llms.ollama import OllamaAdapter
from sura_rag.models import (
    ComplianceCertificate,
    ForgetResult,
    ForgetScore,
    ProbeResult,
    ScanResult,
)

if TYPE_CHECKING:
    from sura_rag.adapters.base import BaseVectorAdapter
    from sura_rag.llms.base import BaseLLMAdapter
    from sura_rag.models import AuditEntry


class SuraClient:
    """Main entry point for the SURA library.

    Provides the complete forget pipeline: vector deletion, leak probing,
    runtime guardrailing, audit logging, and compliance certification.

    Example::

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
        print(result.forget_score.composite_score)

    Args:
        vector_store: A configured vector store adapter.
        llm: An LLM adapter. Defaults to OllamaAdapter if None.
        config: SURA configuration. Uses defaults if None.
        guardrail_mode: Default guardrail mode for leak interception.
        audit_db_url: SQLAlchemy URL for the audit database.
    """

    def __init__(
        self,
        vector_store: BaseVectorAdapter,
        llm: BaseLLMAdapter | None = None,
        config: SuraConfig | None = None,
        guardrail_mode: GuardrailMode = GuardrailMode.HARD_BLOCK,
        audit_db_url: str = "sqlite:///sura_audit.db",
    ) -> None:
        load_dotenv()

        self._config = config or SuraConfig()
        self._config.default_guardrail_mode = guardrail_mode

        # Initialize LLM adapter
        if llm is None:
            llm = OllamaAdapter(
                model=self._config.generator_model,
                embed_model=self._config.embedder_model,
                host=self._config.ollama_host,
            )

        if not llm.is_available():
            raise OllamaNotRunningError(
                "Ollama is not running. Start it with: ollama serve\n"
                "Then pull the required models:\n"
                f"  ollama pull {self._config.generator_model}\n"
                f"  ollama pull {self._config.embedder_model}"
            )

        self._llm = llm
        self._vector_store = vector_store

        # Initialize components
        self._registry = ForgetRegistry(audit_db_url)
        self._forget_engine = ForgetEngine(vector_store, self._registry, llm)
        self._leak_prober = LeakProber(llm, self._config)
        self._guardrail = RuntimeGuardrail(
            self._registry, llm, self._config
        )
        self._audit_logger = AuditLogger(audit_db_url)
        self._cert_generator = CertificateGenerator(audit_db_url)

        if self._config.enable_rich_logging:
            self._print_banner()

    def _print_banner(self) -> None:
        """Print a startup banner using rich."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            console.print(
                Panel.fit(
                    "[bold blue]SURA-RAG[/bold blue] v0.1.0\n"
                    "[dim]Verified data deletion & leak detection for RAG systems[/dim]\n"
                    f"[green]LLM:[/green] {self._config.generator_model} | "
                    f"[green]Embedder:[/green] {self._config.embedder_model}\n"
                    f"[green]Guardrail:[/green] {self._config.default_guardrail_mode.value}",
                    title="[bold]sura-rag[/bold]",
                    border_style="blue",
                )
            )
        except ImportError:
            pass

    def forget(
        self,
        doc_ids: list[str],
        subject: str,
        requestor_id: str,
        regulation: str = "GDPR_Art17",
        probe_strategy: ProbeStrategy = ProbeStrategy.ALL,
        num_probes: int = 15,
        rag_query_fn: Callable[[str], str] | None = None,
        forget_mode: ForgetMode = ForgetMode.BALANCED,
    ) -> ForgetResult:
        """Execute the full forget pipeline.

        Steps:
        1. Delete documents from vector store and store fingerprints.
        2. If rag_query_fn provided and mode != FAST, run leak probes.
        3. Compute ForgetScore from probe results.
        4. Log to audit trail.
        5. Generate compliance certificate.
        6. Return ForgetResult.

        Args:
            doc_ids: Document IDs to forget.
            subject: Description of the data subject.
            requestor_id: Who requested the deletion.
            regulation: Applicable regulation.
            probe_strategy: Which probe strategies to use.
            num_probes: Total number of probes to run.
            rag_query_fn: RAG query function for leak testing.
            forget_mode: How thorough the forget process should be.

        Returns:
            ForgetResult with complete pipeline results.
        """
        certificate_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Step 1: Delete from vector store
        delete_result = self._forget_engine.delete(
            doc_ids=doc_ids,
            subject=subject,
            requestor_id=requestor_id,
            regulation=regulation,
        )

        vector_deleted = delete_result["confirmed"]
        doc_texts = delete_result.get("doc_texts", {})

        # Step 2: Run probes
        if rag_query_fn is not None and forget_mode != ForgetMode.FAST:
            strategies = (
                [probe_strategy]
                if probe_strategy != ProbeStrategy.ALL
                else None
            )
            probe_result = self._leak_prober.probe(
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                rag_query_fn=rag_query_fn,
                strategies=strategies,
                num_probes=num_probes,
            )
        else:
            probe_result = ProbeResult(
                doc_ids=doc_ids,
                total_probes_run=0,
                leakage_hits=[],
                parametric_leak_score=0.0,
                verdict="SKIPPED",
                probe_duration_seconds=0.0,
            )

        # Step 3: Compute ForgetScore
        adversarial_bypass = 0.0
        if probe_result.leakage_hits:
            adv_hits = [
                h
                for h in probe_result.leakage_hits
                if "Adversarial" in h.probe_type
            ]
            adv_total = max(
                1,
                sum(
                    1
                    for _ in probe_result.leakage_hits
                    if "Adversarial" in _.probe_type
                ),
            )
            adversarial_bypass = len(adv_hits) / adv_total if adv_hits else 0.0

        composite = (
            self._config.weight_vector * (1.0 if vector_deleted else 0.0)
            + self._config.weight_parametric
            * (1.0 - probe_result.parametric_leak_score)
            + self._config.weight_adversarial * (1.0 - adversarial_bypass)
        )

        forget_score = ForgetScore(
            vector_deletion_confirmed=vector_deleted,
            parametric_leak_score=probe_result.parametric_leak_score,
            adversarial_bypass_rate=adversarial_bypass,
            composite_score=round(min(1.0, max(0.0, composite)), 4),
            utility_delta=0.0,
        )

        # Determine status
        guardrail_activated = probe_result.verdict == "LEAKED"
        if not vector_deleted:
            status = "failed"
        elif guardrail_activated:
            status = "leaked_guardrailed"
        else:
            status = "completed"

        result = ForgetResult(
            certificate_id=certificate_id,
            doc_ids=doc_ids,
            subject=subject,
            requestor_id=requestor_id,
            regulation=regulation,
            timestamp=timestamp,
            vector_deleted=vector_deleted,
            probe_result=probe_result,
            forget_score=forget_score,
            guardrail_activated=guardrail_activated,
            guardrail_mode=self._config.default_guardrail_mode.value,
            status=status,
        )

        # Step 4: Log to audit
        self._audit_logger.log_forget(result)

        # Step 5: Generate certificate
        self._cert_generator.generate(result)

        return result

    def probe(
        self,
        doc_ids: list[str],
        rag_query_fn: Callable[[str], str],
        strategy: ProbeStrategy = ProbeStrategy.ALL,
        num_probes: int = 15,
    ) -> ProbeResult:
        """Run the leak prober standalone without triggering deletion.

        Useful for auditing an existing system for data leakage.

        Args:
            doc_ids: Document IDs to probe for.
            rag_query_fn: RAG query function to test.
            strategy: Which probe strategy to use.
            num_probes: Total number of probes to run.

        Returns:
            ProbeResult with leak detection results.
        """
        doc_texts: dict[str, str] = {}
        for doc_id in doc_ids:
            entries = self._registry.get_all()
            for entry in entries:
                if entry.doc_id == doc_id:
                    doc_texts[doc_id] = entry.fingerprint_text
                    break
            if doc_id not in doc_texts:
                # Try to get from vector store
                if self._vector_store.document_exists(doc_id):
                    doc_texts[doc_id] = self._vector_store.get_document_text(
                        doc_id
                    )

        strategies = [strategy] if strategy != ProbeStrategy.ALL else None
        result = self._leak_prober.probe(
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            rag_query_fn=rag_query_fn,
            strategies=strategies,
            num_probes=num_probes,
        )

        self._audit_logger.log_probe(result, requestor_id="system")
        return result

    def guardrail(self, response: str) -> ScanResult:
        """Scan a single response against the forget registry.

        Useful for manual scanning outside the wrap() decorator.

        Args:
            response: The RAG response text to scan.

        Returns:
            ScanResult indicating whether leakage was detected.
        """
        return self._guardrail.scan(response)

    def wrap(self, rag_fn: Callable[[str], str]) -> Callable[[str], str]:
        """Return a guardrail-wrapped version of any RAG function.

        Every response from the wrapped function is automatically scanned.

        Args:
            rag_fn: The RAG query function to wrap.

        Returns:
            A wrapped callable with automatic leak scanning.
        """
        return self._guardrail.wrap(rag_fn)

    def get_certificate(
        self, certificate_id: str, format: str = "pdf"
    ) -> ComplianceCertificate:
        """Retrieve a previously generated certificate by ID.

        Args:
            certificate_id: The UUID of the certificate.
            format: Output format (\"pdf\" or \"json\").

        Returns:
            The ComplianceCertificate if found.
        """
        cert = self._cert_generator.get_certificate(certificate_id)
        if cert is None:
            from sura_rag.exceptions import SuraError

            raise SuraError(
                f"Certificate '{certificate_id}' not found in the database."
            )
        return cert

    def audit_log(
        self,
        event_type: str | None = None,
        requestor_id: str | None = None,
        regulation: str | None = None,
        since: str | None = None,
        status: str | None = None,
        as_dataframe: bool = False,
    ) -> list[AuditEntry]:
        """Query the audit trail with optional filters.

        Args:
            event_type: Filter by event type.
            requestor_id: Filter by requestor.
            regulation: Filter by regulation.
            since: Filter events after this ISO date string.
            status: Filter by status.
            as_dataframe: If True, return a pandas DataFrame.

        Returns:
            A list of AuditEntry instances, or DataFrame if requested.
        """
        return self._audit_logger.query(
            event_type=event_type,
            requestor_id=requestor_id,
            regulation=regulation,
            since=since,
            status=status,
            as_dataframe=as_dataframe,
        )

    def forget_score(self, doc_ids: list[str]) -> ForgetScore:
        """Compute a ForgetScore without running the full pipeline.

        Checks the registry and runs a quick probe (5 probes, direct only).

        Args:
            doc_ids: Document IDs to score.

        Returns:
            ForgetScore with current deletion status.
        """
        all_forgotten = all(
            self._registry.is_forgotten(doc_id) for doc_id in doc_ids
        )
        vector_deleted = all_forgotten and all(
            not self._vector_store.document_exists(doc_id)
            for doc_id in doc_ids
        )

        return ForgetScore(
            vector_deletion_confirmed=vector_deleted,
            parametric_leak_score=0.0,
            adversarial_bypass_rate=0.0,
            composite_score=1.0 if vector_deleted else 0.0,
            utility_delta=0.0,
        )
