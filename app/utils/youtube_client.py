"""YouTube Data API v3 client and fetch cycle utilities.

Provides the configured YouTube API client, metadata fetch function,
and comment fetch cycle used by the pipeline's comment fetching stage.
"""

import os

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()


def get_youtube_client():
    """Return a YouTube Data API v3 resource object keyed with YOUTUBE_API_KEY."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    return build("youtube", "v3", developerKey=api_key)


def fetch_video_metadata(video_id: str, *, client=None) -> dict:
    """Fetch and return title, description, and tags for a YouTube video by ID."""
    youtube = client or get_youtube_client()
    try:
        response = youtube.videos().list(
            part="snippet",
            id=video_id,
        ).execute()
    except HttpError as e:
        if e.status_code == 403:
            raise PermissionError(
                "This video is private or age-restricted and cannot be accessed."
            ) from e
        raise

    if not response.get("items"):
        raise ValueError(f"Video not found: '{video_id}'.")

    snippet = response["items"][0]["snippet"]
    return {
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "tags": snippet.get("tags", []),
    }


def fetch_comments_page(video_id: str, page_token: str | None, *, client=None) -> dict:
    """Fetch one page of up to 100 top-level comments, returning items and next_page_token."""
    youtube = client or get_youtube_client()
    kwargs = dict(
        part="snippet",
        videoId=video_id,
        order="relevance",
        maxResults=100,
        textFormat="plainText",
    )
    if page_token:
        kwargs["pageToken"] = page_token

    try:
        response = youtube.commentThreads().list(**kwargs).execute()
    except HttpError as e:
        if e.status_code == 403:
            reason = ""
            try:
                import json
                body = json.loads(e.content.decode())
                errors = body.get("error", {}).get("errors", [])
                if errors:
                    reason = errors[0].get("reason", "")
            except Exception:
                pass
            if reason == "commentsDisabled":
                raise PermissionError("This video has comments disabled.") from e
            raise PermissionError(
                "This video is private or age-restricted and cannot be accessed."
            ) from e
        raise

    items = []
    for thread in response.get("items", []):
        top = thread["snippet"]["topLevelComment"]["snippet"]
        items.append({
            "comment_id": thread["id"],
            "text": top.get("textDisplay", ""),
            "like_count": top.get("likeCount", 0),
        })

    return {
        "items": items,
        "next_page_token": response.get("nextPageToken"),
    }
