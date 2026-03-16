"""
Configuration models for SURA-RAG.

Provides enums for forget modes, guardrail modes, and probe strategies,
plus the main SuraConfig that controls all library behavior.
"""

from enum import Enum

from pydantic import BaseModel, Field


class ForgetMode(str, Enum):
    """Controls the thoroughness of the forget pipeline."""

    FAST = "fast"  # output filter only, <5s
    BALANCED = "balanced"  # probe + guardrail, ~2min
    THOROUGH = "thorough"  # all probes + adversarial, ~8min


class GuardrailMode(str, Enum):
    """Controls how the runtime guardrail handles detected leaks."""

    HARD_BLOCK = "hard_block"
    SOFT_BLOCK = "soft_block"
    WARN_AND_LOG = "warn_and_log"
    FALLBACK_RESPONSE = "fallback_response"


class ProbeStrategy(str, Enum):
    """Available leak probe strategies."""

    DIRECT = "direct"
    PARAPHRASE = "paraphrase"
    CONTEXTUAL = "contextual"
    ADVERSARIAL = "adversarial"
    ALL = "all"


class SuraConfig(BaseModel):
    """Main configuration for the SURA-RAG library.

    All settings have sensible defaults that work out of the box
    with a local Ollama installation.
    """

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    generator_model: str = "llama3.2:3b"
    embedder_model: str = "nomic-embed-text"

    # Leak detection settings
    leak_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    span_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    default_num_probes: int = Field(default=15, ge=1)
    default_probe_strategy: ProbeStrategy = ProbeStrategy.ALL

    # Guardrail settings
    default_guardrail_mode: GuardrailMode = GuardrailMode.HARD_BLOCK
    fallback_message: str = (
        "I'm sorry, I cannot provide information on that topic."
    )

    # Audit settings
    audit_backend: str = "sqlite"  # "sqlite" or "postgres"
    audit_db_url: str = "sqlite:///sura_audit.db"
    enable_rich_logging: bool = True

    # Scoring weights for composite ForgetScore
    weight_parametric: float = Field(default=0.50, ge=0.0, le=1.0)
    weight_vector: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_adversarial: float = Field(default=0.30, ge=0.0, le=1.0)
