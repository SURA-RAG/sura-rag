"""
Mock Ollama adapter for testing.

Provides deterministic responses without requiring a running Ollama server.
Used in ALL unit tests to ensure tests pass without network access.
"""

from __future__ import annotations

import hashlib
import math

from sura_rag.llms.base import BaseLLMAdapter


class MockOllamaAdapter(BaseLLMAdapter):
    """Mock LLM adapter that returns deterministic responses.

    Never makes network calls. Produces consistent outputs based on
    input hashes for reproducible testing.

    Args:
        embed_dim: Dimension of the fake embedding vectors.
        high_similarity: If True, embed() returns vectors that are similar
                         to each other (for testing leak detection).
                         If False, returns diverse vectors (for testing clean cases).
    """

    def __init__(
        self, embed_dim: int = 384, high_similarity: bool = False
    ) -> None:
        self._embed_dim = embed_dim
        self._high_similarity = high_similarity

    def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 400
    ) -> str:
        """Generate a deterministic fake response based on the prompt hash.

        Args:
            prompt: The input prompt.
            temperature: Ignored in mock.
            max_tokens: Ignored in mock.

        Returns:
            A deterministic fake response string.
        """
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        # If prompt asks for questions, generate fake probe questions
        if "question" in prompt.lower() or "generate" in prompt.lower():
            return (
                "What is the person's salary?\n"
                "When did the employee join?\n"
                "What is the employee ID?\n"
                "What company does the person work for?\n"
                "What is the annual compensation?"
            )

        return f"Mock response for prompt hash {prompt_hash[:8]}"

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic fake embedding vector.

        If high_similarity mode is on, all embeddings are similar
        (useful for testing leak detection). Otherwise, embeddings
        are derived from the text hash to be diverse.

        Args:
            text: The text to embed.

        Returns:
            A list of floats of length embed_dim.
        """
        if self._high_similarity:
            # Return nearly identical vectors for all inputs
            # (simulates leaked content matching the fingerprint)
            base = [0.1] * self._embed_dim
            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Add tiny perturbation based on hash
            for i in range(min(16, self._embed_dim)):
                base[i] += int(text_hash[i], 16) * 0.001
            return base

        # Deterministic but diverse vectors based on text hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        vector: list[float] = []
        for i in range(self._embed_dim):
            # Use rolling hash characters to generate values
            idx = (i * 2) % len(text_hash)
            val = int(text_hash[idx : idx + 2], 16) / 255.0
            vector.append(val)

        # Normalize to unit length
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def is_available(self) -> bool:
        """Always returns True for testing.

        Returns:
            True.
        """
        return True
