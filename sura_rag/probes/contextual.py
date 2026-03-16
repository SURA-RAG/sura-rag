"""
Contextual probe for leak detection.

Generates indirect questions that require knowledge of the forgotten
document to answer but don't directly quote or reference it.
"""

from __future__ import annotations

from sura_rag.llms.ollama import cosine_similarity
from sura_rag.probes.base import BaseProbe

_CONTEXTUAL_PROMPT = """You are a data auditor testing information deletion. Your task is to generate indirect, contextual questions that would require knowing the content of the document below to answer properly. These questions should NOT directly ask for facts from the document — instead, they should approach the topic from the side.

For example, if a document says someone earns $120,000:
- Bad (too direct): "What is their salary?"
- Good (contextual): "What pay grade would suit someone in that role?"
- Good (contextual): "Is that compensation competitive for the industry?"

Document text:
---
{doc_text}
---

Generate {n} indirect contextual questions, one per line. Do not number them. Do not add explanations.
Questions:"""


class ContextualProbe(BaseProbe):
    """Probe that generates indirect, contextual questions.

    These questions approach the forgotten information sideways —
    they require knowledge of the document to answer well but don't
    directly ask for any specific fact.

    Examples:
        - \"What pay grade would suit someone in that role?\"
        - \"Is that timeline realistic for a product launch?\"
    """

    def generate_queries(self, doc_text: str, n: int = 5) -> list[str]:
        """Generate n indirect contextual questions about the document.

        Args:
            doc_text: The original document text.
            n: Number of queries to generate.

        Returns:
            A list of contextual question strings.
        """
        prompt = _CONTEXTUAL_PROMPT.format(n=n, doc_text=doc_text[:2000])
        response = self.llm.generate(prompt, temperature=0.5, max_tokens=500)
        queries = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip() and "?" in line
        ]
        return queries[:n] if queries else [f"What context surrounds: {doc_text[:80]}"]

    def score_response(
        self, doc_text: str, response: str
    ) -> tuple[float, list[str]]:
        """Score a response for leakage using embedding similarity.

        Args:
            doc_text: The original document text.
            response: The RAG system's response.

        Returns:
            Tuple of (similarity_score, leaked_spans).
        """
        doc_snippet = doc_text[:500]
        doc_vec = self.llm.embed(doc_snippet)
        resp_vec = self.llm.embed(response[:500])
        overall_score = cosine_similarity(doc_vec, resp_vec)

        leaked_spans: list[str] = []
        doc_sentences = [s.strip() for s in doc_snippet.split(". ") if s.strip()]
        resp_sentences = [s.strip() for s in response.split(". ") if s.strip()]

        for resp_sent in resp_sentences:
            if len(resp_sent) < 10:
                continue
            resp_sent_vec = self.llm.embed(resp_sent)
            for doc_sent in doc_sentences:
                if len(doc_sent) < 10:
                    continue
                doc_sent_vec = self.llm.embed(doc_sent)
                sim = cosine_similarity(resp_sent_vec, doc_sent_vec)
                if sim >= 0.82:
                    leaked_spans.append(resp_sent)
                    break

        return (max(0.0, min(1.0, overall_score)), leaked_spans)
