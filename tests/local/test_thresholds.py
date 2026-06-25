"""Local tests for apply_post_extraction_threshold in stage4_extraction.py.

Tests boundary values 50, 51, 100, and 101 against Stage 3 threshold spec:
  ≤ 50  → error outcome
  51–100 → degraded quality mode
  > 100 → full quality mode
"""

from app.pipeline.stage4_extraction import apply_post_extraction_threshold


def test_50_returns_error():
    result = apply_post_extraction_threshold(50)
    assert isinstance(result, dict)
    assert result["outcome"] == "error"
    assert "50" in result["message"]
    assert "51" in result["message"]


def test_51_returns_degraded():
    result = apply_post_extraction_threshold(51)
    assert isinstance(result, tuple)
    assert result[0] == "degraded"
    assert result[1] is None


def test_100_returns_degraded():
    result = apply_post_extraction_threshold(100)
    assert isinstance(result, tuple)
    assert result[0] == "degraded"
    assert result[1] is None


def test_101_returns_full():
    result = apply_post_extraction_threshold(101)
    assert isinstance(result, tuple)
    assert result[0] == "full"
    assert result[1] is None
