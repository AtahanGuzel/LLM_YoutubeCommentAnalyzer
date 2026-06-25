"""Stage 8 — Confidence Computation.

Computes the overall confidence tier (High / Medium / Low) from Eval 2 scores and
data quality signals: processing loss ratio, relevant comment count, and average
criterion score. Returns a tuple of (tier, warning_message | None).
"""


def compute_confidence(
    scores: list[int],
    relevant_count: int,
    processed_count: int,
    total_fetched: int,
) -> tuple[str, str | None]:
    """Return (confidence_tier, warning_message) based on scores and data quality.

    Check order (first match wins):
    1. loss_ratio > 0.25 → Low
    2. relevant_count < 30 → Low
    3. avg >= 4.0 → High
    4. avg >= 2.5 → Medium
    5. else → Low
    """
    loss_ratio = (total_fetched - processed_count) / total_fetched

    if loss_ratio > 0.25:
        return (
            "Low",
            f"Analysis based on {processed_count} of {total_fetched} "
            "comments due to processing errors.",
        )

    if relevant_count < 30:
        return ("Low", "Fewer than 30 comments were relevant to the target product.")

    avg = sum(scores) / len(scores)
    if avg >= 4.0:
        return ("High", None)
    elif avg >= 2.5:
        return ("Medium", "Results should be treated as directional.")
    else:
        return ("Low", "Manual review of comments is recommended.")


def find_score_one_criteria(output_quality: dict) -> list[str]:
    """Return a list of criterion names where score == 1.

    Checks sentiment, pain_points, and competitors from the output_quality dict.
    Returns an empty list if no criterion scores exactly 1.
    """
    criteria = ("sentiment", "pain_points", "competitors")
    return [c for c in criteria if output_quality.get(c, {}).get("score") == 1]
