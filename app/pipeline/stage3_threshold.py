"""Stage 3 — Threshold Check.

Evaluates the English comment count after the fetch cycle. Returns a quality mode
tuple ("full", None) or ("degraded", None), or raises ValueError if the count is
too low to proceed with analysis.
"""


def check_threshold(english_count: int) -> tuple[str, None]:
    """Return ("full", None) or ("degraded", None), or raise ValueError if count ≤ 50."""
    if english_count > 100:
        return ("full", None)
    if english_count >= 51:
        return ("degraded", None)
    raise ValueError(
        f"Insufficient English comments found ({english_count} detected). "
        "A minimum of 51 comments is required for analysis."
    )
