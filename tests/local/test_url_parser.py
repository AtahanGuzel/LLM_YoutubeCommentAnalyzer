from app.pipeline.stage0_url_parser import extract_video_id


def test_watch_format():
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_short_url_format():
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_shorts_format():
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_live_format():
    assert extract_video_id("https://www.youtube.com/live/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_non_youtube_url_returns_none():
    assert extract_video_id("https://www.vimeo.com/123456789") is None


def test_empty_string_returns_none():
    assert extract_video_id("") is None


def test_missing_video_id_returns_none():
    assert extract_video_id("https://www.youtube.com/watch?v=") is None
