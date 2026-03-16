"""
Hard block guardrail mode.

Completely blocks the response, returning None. This is the strictest
mode — no potentially leaked content reaches the user.
"""


def hard_block(response: str, leaked_spans: list[str]) -> tuple[None, str]:
    """Apply hard block mode: completely suppress the response.

    Args:
        response: The original response (ignored).
        leaked_spans: The leaked text spans (ignored).

    Returns:
        Tuple of (None, \"hard_blocked\").
    """
    return (None, "hard_blocked")
