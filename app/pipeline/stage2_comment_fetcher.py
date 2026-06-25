"""Stage 2 — Comment Fetching (Fetch → Filter → Cycle).

Fetches top-level comments from the YouTube Data API in pages of 100, applies
langdetect language filtering to retain English comments only, and stops when
english_count reaches 100, total_fetched reaches 500, or no next page token remains.
"""

from langdetect import detect, DetectorFactory

from app.utils.youtube_client import fetch_comments_page

DetectorFactory.seed = 0

_ENGLISH_TARGET = 100
_TOTAL_CAP = 500


def is_english(text: str) -> bool:
    """Return True if langdetect identifies the text as English, False otherwise."""
    try:
        return detect(text) == "en"
    except Exception:
        return False


def fetch_and_filter_comments(video_id: str, *, client=None) -> list[dict]:
    """Fetch and return English comments via the fetch-filter cycle.

    Fetches pages of 100 top-level comments using fetch_comments_page, applies
    langdetect filtering, and stops when 100 English comments are collected,
    500 total comments are fetched, or no next page token remains.

    Args:
        video_id: YouTube video ID to fetch comments for.
        client: Optional pre-built YouTube API client; creates one via
                get_youtube_client() if not provided.

    Returns:
        List of comment dicts with keys: comment_id, text, like_count.
        May contain fewer than 100 comments if the video has limited English content.
    """
    english_comments: list[dict] = []
    total_fetched = 0
    page_token = None

    while True:
        page = fetch_comments_page(video_id, page_token, client=client)
        batch = page["items"]
        total_fetched += len(batch)

        for comment in batch:
            if is_english(comment["text"]):
                english_comments.append(comment)
                if len(english_comments) >= _ENGLISH_TARGET:
                    return english_comments

        page_token = page["next_page_token"]
        if page_token is None or total_fetched >= _TOTAL_CAP:
            return english_comments
