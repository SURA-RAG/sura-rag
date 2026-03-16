"""
Abstract base class for LLM adapters.

Any new LLM backend (OpenAI, vLLM, etc.) must implement this interface
to be compatible with SURA-RAG's probe and guardrail systems.
"""

from abc import ABC, abstractmethod


class BaseLLMAdapter(ABC):
    """Abstract base class for all LLM adapters.

    Provides the minimal interface required by probes and guardrails
    to generate text and compute embeddings.
    """

    @abstractmethod
    def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 400
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt to generate from.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated response string.
        """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM backend is reachable.

        Returns:
            True if the backend is available, False otherwise.
            Never raises exceptions.
        """
