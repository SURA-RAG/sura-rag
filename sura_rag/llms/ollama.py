"""
Ollama LLM adapter for SURA-RAG.

Provides text generation and embedding via a local Ollama server.
Zero cloud API required — everything runs on your machine.
"""

from __future__ import annotations

import math

from sura_rag.exceptions import OllamaNotRunningError
from sura_rag.llms.base import BaseLLMAdapter


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses numpy if available for speed, falls back to pure Python.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity score between -1.0 and 1.0.
    """
    try:
        import numpy as np

        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except ImportError:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for Ollama local LLM server.

    Provides text generation and embedding via the Ollama Python client.
    Requires Ollama to be running locally (ollama serve).

    Args:
        model: Model name for text generation (e.g., \"llama3.2:3b\").
        embed_model: Model name for embeddings (e.g., \"nomic-embed-text\").
        host: Ollama server URL.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        embed_model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        try:
            import ollama  # noqa: F401
        except ImportError:
            raise ImportError(
                "Ollama Python client is not installed. Install it with:\n"
                "  pip install ollama"
            )

        self._model = model
        self._embed_model = embed_model
        self._host = host
        self._timeout = timeout

        from ollama import Client

        self._client = Client(host=host, timeout=timeout)

    def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 400
    ) -> str:
        """Generate text from a prompt using Ollama.

        Args:
            prompt: The input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text response.

        Raises:
            OllamaNotRunningError: If the Ollama server is unreachable.
        """
        try:
            response = self._client.generate(
                model=self._model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            return response["response"]
        except Exception as e:
            if "connect" in str(e).lower() or "refused" in str(e).lower():
                raise OllamaNotRunningError(
                    "Ollama is not running. Start it with: ollama serve\n"
                    f"Expected server at: {self._host}\n"
                    "Install Ollama from: https://ollama.com"
                ) from e
            raise

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector using Ollama.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            OllamaNotRunningError: If the Ollama server is unreachable.
        """
        try:
            response = self._client.embeddings(
                model=self._embed_model, prompt=text
            )
            return response["embedding"]
        except Exception as e:
            if "connect" in str(e).lower() or "refused" in str(e).lower():
                raise OllamaNotRunningError(
                    "Ollama is not running. Start it with: ollama serve\n"
                    f"Expected server at: {self._host}\n"
                    "Install Ollama from: https://ollama.com"
                ) from e
            raise

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable.

        Makes a lightweight request to the Ollama API tags endpoint.

        Returns:
            True if Ollama is running, False otherwise. Never raises.
        """
        try:
            self._client.list()
            return True
        except Exception:
            return False
