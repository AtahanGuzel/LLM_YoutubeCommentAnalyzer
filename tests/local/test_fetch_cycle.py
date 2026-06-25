from unittest.mock import patch, call
import pytest

from app.pipeline.stage2_comment_fetcher import fetch_and_filter_comments
from app.pipeline.stage3_threshold import check_threshold


def _make_english_comments(count, offset=0):
    return [
        {"comment_id": f"id_{offset + i}", "text": f"This is a great product, highly recommend it number {offset + i}", "like_count": 0}
        for i in range(count)
    ]


def _make_nonenglish_comments(count, offset=0):
    return [
        {"comment_id": f"id_{offset + i}", "text": f"Das ist ein tolles Produkt, sehr empfehlenswert Nummer {offset + i}", "like_count": 0}
        for i in range(count)
    ]


def test_stops_at_100_english():
    page = {"items": _make_english_comments(100), "next_page_token": "page2"}
    with patch("app.pipeline.stage2_comment_fetcher.fetch_comments_page", return_value=page) as mock_fetch:
        result = fetch_and_filter_comments("some_video_id")
    assert len(result) == 100
    mock_fetch.assert_called_once()


def test_stops_at_hard_cap():
    page = {"items": _make_nonenglish_comments(100), "next_page_token": "next"}
    with patch("app.pipeline.stage2_comment_fetcher.fetch_comments_page", return_value=page) as mock_fetch:
        result = fetch_and_filter_comments("some_video_id")
    assert len(result) == 0
    assert mock_fetch.call_count == 5


def test_stops_at_no_next_page():
    page1 = {"items": _make_english_comments(30, offset=0), "next_page_token": "page2"}
    page2 = {"items": _make_english_comments(30, offset=30), "next_page_token": None}
    with patch("app.pipeline.stage2_comment_fetcher.fetch_comments_page", side_effect=[page1, page2]) as mock_fetch:
        result = fetch_and_filter_comments("some_video_id")
    assert len(result) == 60
    assert mock_fetch.call_count == 2


def test_threshold_full():
    assert check_threshold(101) == ("full", None)


def test_threshold_degraded():
    assert check_threshold(51) == ("degraded", None)
    assert check_threshold(100) == ("degraded", None)


def test_threshold_reject():
    with pytest.raises(ValueError, match="Insufficient English comments found \\(50 detected\\)"):
        check_threshold(50)
    with pytest.raises(ValueError, match="Insufficient English comments found \\(0 detected\\)"):
        check_threshold(0)
