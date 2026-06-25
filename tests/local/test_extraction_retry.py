"""Local tests for process_chunk_with_retry in stage4_extraction.py.

Covers: success after two RateLimitErrors, and None after all retries exhausted.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import groq as groq_lib
import httpx

from app.pipeline.stage4_extraction import process_chunk_with_retry


_VALID_RESPONSE = json.dumps([
    {
        "comment_id": 1,
        "comment_type": "target",
        "sentiment": "positive",
        "pain_points": [],
        "competitor_mentions": [],
    }
])

_CHUNK = [{"comment_id": 1, "text": "Great product!"}]


def _rate_limit_error():
    request = httpx.Request("GET", "https://api.groq.com")
    response = httpx.Response(429, request=request)
    return groq_lib.RateLimitError("rate limit exceeded", response=response, body=None)


def _completion(content: str):
    completion = MagicMock()
    completion.choices[0].message.content = content
    return completion


def test_retry_succeeds_on_third_attempt():
    """Raises RateLimitError on attempts 1 and 2, succeeds on attempt 3."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _rate_limit_error(),
        _rate_limit_error(),
        _completion(_VALID_RESPONSE),
    ]

    async def _run():
        with patch("app.pipeline.stage4_extraction.get_groq_client", return_value=mock_client), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            return await process_chunk_with_retry(_CHUNK, "Test Product")

    result = asyncio.run(_run())
    assert result is not None
    assert len(result) == 1
    assert result[0]["comment_type"] == "target"
    assert result[0]["sentiment"] == "positive"


def test_retry_exhausted_returns_none():
    """Raises RateLimitError on all 3 attempts — should return None."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _rate_limit_error(),
        _rate_limit_error(),
        _rate_limit_error(),
    ]

    async def _run():
        with patch("app.pipeline.stage4_extraction.get_groq_client", return_value=mock_client), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            return await process_chunk_with_retry(_CHUNK, "Test Product")

    result = asyncio.run(_run())
    assert result is None
