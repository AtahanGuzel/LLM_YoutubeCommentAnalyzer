"""Stage 4 — Chunked Extraction.

Sends comments to the extraction model in chunks of 25 for structured label extraction.
Each comment is classified with comment_type, sentiment, pain_points, and
competitor_mentions. Chunks are retried up to 3 times with exponential backoff.
"""

import asyncio
import json
import re

import groq as groq_lib

from app.utils.groq_client import (
    EXTRACTION_MODEL,
    get_groq_client,
    parse_extraction_response,
    validate_extraction_response,
)
from app.pipeline.prompts.extraction import EXTRACTION_PROMPT

_CHUNK_SIZE = 25


def chunk_comments(comments: list[dict]) -> list[list[dict]]:
    """Split a comment list into chunks of 25, final chunk absorbs any remainder.

    Non-final chunks are exactly 25 items. The final chunk is 25 + remainder so
    there is never a dangling sub-25 chunk at the end (e.g. 76 → [25, 25, 26]).
    """
    n = len(comments)
    if n == 0:
        return []

    n_complete = n // _CHUNK_SIZE
    remainder = n % _CHUNK_SIZE

    if remainder == 0:
        return [comments[i * _CHUNK_SIZE:(i + 1) * _CHUNK_SIZE] for i in range(n_complete)]

    if n_complete == 0:
        return [comments]

    # Has complete chunks plus remainder: merge remainder into final chunk
    leading = [comments[i * _CHUNK_SIZE:(i + 1) * _CHUNK_SIZE] for i in range(n_complete - 1)]
    final = comments[(n_complete - 1) * _CHUNK_SIZE:]
    return leading + [final]


async def process_chunk_with_retry(
    chunk: list[dict],
    primary_product: str,
    max_retries: int = 3,
) -> list[dict] | None:
    """Call the extraction model on one chunk with retry and exponential backoff.

    Returns the parsed list of labeled comment dicts on success.
    Returns None (never raises) after all retries are exhausted.

    RateLimitError: exponential backoff — 2^attempt seconds (1s, 2s, 4s).
    InternalServerError / APITimeoutError: fixed 1-second delay then retry.
    """
    client = get_groq_client()
    prompt = EXTRACTION_PROMPT.replace("{primary_product}", primary_product)
    user_message = json.dumps(
        [{"comment_id": c["comment_id"], "text": c["text"]} for c in chunk]
    )

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = completion.choices[0].message.content
            # Extract JSON from markdown code fences if present
            fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if fence_match:
                raw = fence_match.group(1).strip()
            else:
                # Fall back to extracting the outermost JSON array from raw text
                start = raw.find("[")
                end = raw.rfind("]")
                if start != -1 and end > start:
                    raw = raw[start:end + 1]
            parsed = parse_extraction_response(raw)
            # Handle object-wrapped array e.g. {"labels": [...]}
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
                else:
                    parsed = None
            if parsed is not None and validate_extraction_response(parsed):
                return parsed
            return None
        except groq_lib.RateLimitError:
            await asyncio.sleep(2 ** attempt)
        except (groq_lib.InternalServerError, groq_lib.APITimeoutError):
            await asyncio.sleep(1)
    return None


async def extract_labels(
    comments: list[dict],
    primary_product: str,
) -> tuple[list[dict], int]:
    """Process all comment chunks and return (labeled_comments, processed_count).

    Chunks that return None from process_chunk_with_retry are discarded.
    After extraction, each labeled comment is enriched with like_count from the
    original comments list (joined by comment_id; defaults to 0 if not found).
    If all chunks fail, returns ([], 0).
    """
    chunks = chunk_comments(comments)
    labeled: list[dict] = []

    for chunk in chunks:
        result = await process_chunk_with_retry(chunk, primary_product)
        if result is not None:
            labeled.extend(result)

    like_lookup = {c["comment_id"]: c.get("like_count", 0) for c in comments}
    for item in labeled:
        item["like_count"] = like_lookup.get(item.get("comment_id"), 0)

    return labeled, len(labeled)


def apply_post_extraction_threshold(processed_count: int):
    """Return quality mode tuple or error dict based on post-extraction processed_count.

    >100 → ("full", None)
    51–100 → ("degraded", None)
    ≤50 → {"outcome": "error", "message": "..."}
    """
    if processed_count > 100:
        return ("full", None)
    if processed_count >= 51:
        return ("degraded", None)
    return {
        "outcome": "error",
        "message": (
            f"Insufficient English comments found ({processed_count} detected). "
            "A minimum of 51 comments is required for analysis."
        ),
    }
