"""Stage 5 — Code Aggregation.

Aggregates extracted labels in code. Computes sentiment distribution for target
comments, ranks pain points by weighted score (mention_count + log(likes + 1)),
and counts competitor mentions, applying the >=10 threshold for inclusion.
"""

import math
from collections import defaultdict


def compute_sentiment_distribution(labeled_comments: list[dict]) -> dict:
    """Count sentiment for target comments only and return raw counts + percentages.

    competitor_focus and noise comments are excluded entirely.
    Returns {"positive": n, "negative": n, "neutral": n,
             "positive_pct": f, "negative_pct": f, "neutral_pct": f}
    """
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    for comment in labeled_comments:
        if comment.get("comment_type") != "target":
            continue
        sentiment = comment.get("sentiment")
        if sentiment in counts:
            counts[sentiment] += 1

    total = sum(counts.values())
    if total == 0:
        return {
            "positive": 0, "negative": 0, "neutral": 0,
            "positive_pct": 0.0, "negative_pct": 0.0, "neutral_pct": 0.0,
        }

    return {
        "positive": counts["positive"],
        "negative": counts["negative"],
        "neutral": counts["neutral"],
        "positive_pct": round(counts["positive"] / total * 100, 2),
        "negative_pct": round(counts["negative"] / total * 100, 2),
        "neutral_pct": round(counts["neutral"] / total * 100, 2),
    }


def weighted_score(mention_count: int, total_likes: int) -> float:
    """Return mention_count + log(total_likes + 1) per brief Stage 5 formula."""
    return mention_count + math.log(total_likes + 1)


def rank_pain_points(labeled_comments: list[dict]) -> list[dict]:
    """Collect pain points from target comments with negative/neutral sentiment.

    Returns up to 5 pain points in descending weighted score order.
    Each entry: {"pain_point": str, "mention_count": int, "weighted_score": float}
    """
    # Accumulate: pain_point → {mentions, total_likes}
    aggregated: dict[str, dict] = defaultdict(lambda: {"mentions": 0, "total_likes": 0})

    for comment in labeled_comments:
        if comment.get("comment_type") != "target":
            continue
        if comment.get("sentiment") not in ("negative", "neutral"):
            continue
        like_count = comment.get("like_count", 0)
        for pain_point in comment.get("pain_points", []):
            aggregated[pain_point]["mentions"] += 1
            aggregated[pain_point]["total_likes"] += like_count

    if not aggregated:
        return []

    scored = [
        {
            "pain_point": pp,
            "mention_count": data["mentions"],
            "weighted_score": weighted_score(data["mentions"], data["total_likes"]),
        }
        for pp, data in aggregated.items()
    ]
    scored.sort(key=lambda x: x["weighted_score"], reverse=True)
    return scored[:5]


_COMPETITOR_THRESHOLD = 10


def count_competitors(labeled_comments: list[dict]) -> list[dict]:
    """Count unique competitor_mentions across all labeled comments.

    Counts raw comment occurrences (not unique mention strings per comment).
    Returns only competitors with count >= 10, as a list of
    {"competitor": str, "mention_count": int} sorted by descending count.
    """
    counts: dict[str, int] = defaultdict(int)

    for comment in labeled_comments:
        for brand in comment.get("competitor_mentions", []):
            counts[brand] += 1

    result = [
        {"competitor": brand, "mention_count": count}
        for brand, count in counts.items()
        if count >= _COMPETITOR_THRESHOLD
    ]
    result.sort(key=lambda x: x["mention_count"], reverse=True)
    return result
