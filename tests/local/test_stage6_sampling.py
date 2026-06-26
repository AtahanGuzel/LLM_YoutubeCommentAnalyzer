"""Local tests for stage6_sampling.py — random and stratified sample selection."""

from app.pipeline.stage6_sampling import select_random_sample, select_stratified_sample


def _make_comments(n: int) -> list[dict]:
    return [{"comment_id": i} for i in range(n)]


def _make_labeled_comments(n: int) -> list[dict]:
    sentiments = ("positive", "negative", "neutral")
    comments = []
    for i in range(n):
        comments.append({
            "comment_id": i,
            "text": f"comment {i}",
            "comment_type": "target",
            "sentiment": sentiments[i % 3],
            "pain_points": ["slow delivery"] if i % 5 == 0 else [],
            "competitor_mentions": ["BrandX"] if i % 7 == 0 else [],
            "like_count": i % 10,
        })
    return comments


def test_random_sample_returns_exactly_30():
    comments = _make_comments(60)
    result = select_random_sample(comments)
    assert len(result) == 30
    input_ids = {c["comment_id"] for c in comments}
    for item in result:
        assert item["comment_id"] in input_ids


def test_random_sample_fewer_than_30_returns_all():
    comments = _make_comments(15)
    result = select_random_sample(comments)
    assert len(result) == 15


def test_random_sample_independent_of_stratified():
    comments = _make_labeled_comments(80)
    pain_points = [{"pain_point": "slow delivery", "mention_count": 5, "weighted_score": 5.0}]
    competitors = [{"competitor": "BrandX", "mention_count": 12}]

    sample_a = select_random_sample(comments)
    sample_b = select_stratified_sample(comments, pain_points, competitors)

    assert len(sample_a) > 0
    assert len(sample_b) > 0
