"""
Fallback response guardrail mode.

Replaces the entire response with a safe fallback message.
"""


def fallback_response(
    response: str, leaked_spans: list[str], fallback_message: str
) -> tuple[str, str]:
    """Apply fallback mode: replace response with a safe message.

    Args:
        response: The original response (discarded).
        leaked_spans: The leaked text spans (ignored).
        fallback_message: The safe message to return instead.

    Returns:
        Tuple of (fallback_message, \"fallback_substituted\").
    """
    return (fallback_message, "fallback_substituted")
