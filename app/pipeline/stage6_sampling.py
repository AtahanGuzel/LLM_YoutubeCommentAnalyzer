"""Stage 6 — Stratified Sample Selection.

Selects a stratified sample from the labeled comment dataset for evaluation.
Sample composition: 5 per sentiment bucket, 3 per pain point (up to 5), 3 per
competitor, plus 5 random comments. This sample is passed to the evaluation model.
"""

import random


def select_stratified_sample(
    labeled_comments: list[dict],
    pain_points: list[dict],
    competitors: list[dict],
    *,
    seed: int = 0,
) -> list[dict]:
    """Return a deduplicated stratified sample from labeled_comments.

    Selection targets (in order, deduplicating by comment_id):
    - Up to 5 per sentiment bucket (positive, negative, neutral) from target comments
    - Up to 3 comments per identified pain point (up to 5 pain points)
    - Up to 3 comments per identified competitor
    - Up to 5 random comments from the full labeled set

    Comments that appear in an earlier bucket are not repeated in a later bucket.
    If a bucket has fewer than its target, all available comments are taken.
    """
    rng = random.Random(seed)
    seen_ids: set = set()
    sample: list[dict] = []

    def add(comment: dict) -> None:
        cid = comment.get("comment_id")
        if cid not in seen_ids:
            seen_ids.add(cid)
            sample.append(comment)

    # Sentiment buckets — target comments only
    for sentiment in ("positive", "negative", "neutral"):
        bucket = [
            c for c in labeled_comments
            if c.get("comment_type") == "target" and c.get("sentiment") == sentiment
        ]
        for comment in bucket[:5]:
            add(comment)

    # Pain point buckets — target comments with the pain point string
    pain_point_names = [pp["pain_point"] for pp in pain_points[:5]]
    for pp_name in pain_point_names:
        bucket = [
            c for c in labeled_comments
            if c.get("comment_type") == "target"
            and pp_name in c.get("pain_points", [])
        ]
        count = 0
        for comment in bucket:
            if count >= 3:
                break
            add(comment)
            count += 1

    # Competitor buckets — any comment type that mentions the competitor
    competitor_names = [comp["competitor"] for comp in competitors]
    for comp_name in competitor_names:
        bucket = [
            c for c in labeled_comments
            if comp_name in c.get("competitor_mentions", [])
        ]
        count = 0
        for comment in bucket:
            if count >= 3:
                break
            add(comment)
            count += 1

    # Random sanity-check comments from the full labeled set
    remaining = [c for c in labeled_comments if c.get("comment_id") not in seen_ids]
    rng.shuffle(remaining)
    for comment in remaining[:5]:
        add(comment)

    return sample
