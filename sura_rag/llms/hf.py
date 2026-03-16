"""
HuggingFace Transformers LLM adapter for SURA-RAG.

Provides text generation and embedding via local HuggingFace models.
Requires a HuggingFace token for gated models like Llama.
"""

from __future__ import annotations

import os

from sura_rag.exceptions import SuraError
from sura_rag.llms.base import BaseLLMAdapter


class HuggingFaceAdapter(BaseLLMAdapter):
    """Adapter for HuggingFace Transformers models.

    Runs models locally using the transformers library. Supports
    CPU and CUDA inference with automatic device selection.

    Args:
        model_name: HuggingFace model ID (e.g., \"meta-llama/Llama-3.2-3B-Instruct\").
        token: HuggingFace API token. Reads from HF_TOKEN env var if None.
        device: Device to run on (\"cpu\", \"cuda\", \"auto\").
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        token: str | None = None,
        device: str = "auto",
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "HuggingFace Transformers is not installed. Install it with:\n"
                "  pip install sura-rag[hf]\n"
                "  or: pip install transformers huggingface-hub accelerate"
            )

        self._token = token or os.environ.get("HF_TOKEN")
        if not self._token:
            raise SuraError(
                "HuggingFace token required. Set it in your .env file.\n"
                "Get your token at: https://huggingface.co/settings/tokens\n"
                "Then add to .env: HF_TOKEN=hf_your_token_here"
            )

        self._model_name = model_name
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=self._token
        )

        load_kwargs: dict = {"token": self._token}
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        elif device == "cuda":
            load_kwargs["device_map"] = "cuda"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        )

        if device == "cpu":
            self._model = self._model.to("cpu")

        # Embedding model (sentence-transformers style)
        self._embed_model = None
        self._embed_tokenizer = None

    def _get_embed_model(self):
        """Lazy-load a sentence-transformers model for embeddings."""
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embed_model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device="cpu" if self._device == "cpu" else None,
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embeddings.\n"
                    "Install with: pip install sura-rag[cpu]"
                )
        return self._embed_model

    def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 400
    ) -> str:
        """Generate text from a prompt using HuggingFace model.

        Args:
            prompt: The input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens to generate.

        Returns:
            The generated text response.
        """
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        outputs = self._model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens (exclude prompt tokens)
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector using sentence-transformers.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        model = self._get_embed_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def is_available(self) -> bool:
        """Check if the HuggingFace model is loaded and ready.

        Returns:
            True if the model is loaded, False otherwise. Never raises.
        """
        try:
            return self._model is not None and self._tokenizer is not None
        except Exception:
            return False
