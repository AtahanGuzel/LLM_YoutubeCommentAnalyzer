"""Stage 0 — Input & URL Parsing.

Extracts the video ID from a YouTube URL. Handles all four supported formats:
youtube.com/watch?v={id}, youtu.be/{id}, youtube.com/shorts/{id}, youtube.com/live/{id}.
"""

import re
from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str) -> str | None:
    """Return the YouTube video ID from a URL, or None if the URL is unrecognised."""
    parsed = urlparse(url)

    if parsed.netloc in ("www.youtube.com", "youtube.com"):
        if parsed.path == "/watch":
            params = parse_qs(parsed.query)
            ids = params.get("v", [])
            return ids[0] if ids else None

        match = re.match(r"^/(?:shorts|live)/([A-Za-z0-9_-]+)", parsed.path)
        if match:
            return match.group(1)

    if parsed.netloc == "youtu.be":
        vid = parsed.path.lstrip("/")
        return vid if vid else None

    return None
