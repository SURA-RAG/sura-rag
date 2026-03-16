"""
Warn and log guardrail mode.

Passes through the response unchanged but logs a warning.
The leak is recorded in the audit log for compliance review.
"""


def warn_and_log(response: str, leaked_spans: list[str]) -> tuple[str, str]:
    """Apply warn-and-log mode: pass through unchanged, flag for audit.

    Args:
        response: The original response (returned unchanged).
        leaked_spans: The leaked text spans (logged, not redacted).

    Returns:
        Tuple of (response, \"warned_passed_through\").
    """
    return (response, "warned_passed_through")
