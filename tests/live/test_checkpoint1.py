# Product promotion video used: "iPhone 17 Review: No Asterisks!" (MKBHD)
# URL: https://www.youtube.com/watch?v=rng_yUSwrgU
# Chosen because it is a high-engagement product review video with 100+ English
# comments and classifies as is_promotion: true / product_confidence: high.

import asyncio

from app.main import run_pipeline

PRODUCT_PROMO_URL = "https://www.youtube.com/watch?v=rng_yUSwrgU"


def test_checkpoint1_full_pipeline():
    """Smoke test: run_pipeline returns primary_product, video_title, and English comments."""
    result = asyncio.run(run_pipeline(PRODUCT_PROMO_URL))

    assert result.get("outcome") == "proceed", (
        f"Expected proceed outcome, got: {result.get('outcome')} — {result.get('message')}"
    )
    assert result.get("primary_product"), "primary_product must be a non-empty string"
    assert result.get("video_title"), "video_title must be a non-empty string"
    comments = result.get("comments", [])
    assert len(comments) > 0, "comments list must be non-empty"
