"""
Soft block guardrail mode.

Redacts leaked spans from the response, replacing them with [REDACTED].
The rest of the response is preserved.
"""


def soft_block(response: str, leaked_spans: list[str]) -> tuple[str, str]:
    """Apply soft block mode: redact leaked spans from the response.

    Args:
        response: The original response.
        leaked_spans: The specific text spans to redact.

    Returns:
        Tuple of (redacted_response, \"soft_redacted\").
    """
    redacted = response
    for span in leaked_spans:
        redacted = redacted.replace(span, "[REDACTED]")
    return (redacted, "soft_redacted")
