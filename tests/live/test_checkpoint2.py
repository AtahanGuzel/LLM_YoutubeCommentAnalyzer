# Product promotion video used: "iPhone 17 Review: No Asterisks!" (MKBHD)
# URL: https://www.youtube.com/watch?v=rng_yUSwrgU
# Live call: 1 YouTube metadata unit (stage 1) + 1 comment page unit (stage 2)
# + Groq extraction calls (stage 4, Llama 4 Scout, ~4 chunks for 100 comments)

import asyncio

from app.main import run_pipeline

PRODUCT_PROMO_URL = "https://www.youtube.com/watch?v=rng_yUSwrgU"


def test_checkpoint2_stages4_and_5():
    """Smoke test: pipeline stages 4-5 produce sentiment_distribution, pain_points, competitors."""
    result = asyncio.run(run_pipeline(PRODUCT_PROMO_URL))

    assert result.get("outcome") == "proceed", (
        f"Expected proceed outcome, got: {result.get('outcome')} — {result.get('message')}"
    )

    sd = result.get("sentiment_distribution")
    assert isinstance(sd, dict), "sentiment_distribution must be a dict"
    assert "positive" in sd, "sentiment_distribution must have 'positive' key"
    assert "negative" in sd, "sentiment_distribution must have 'negative' key"
    assert "neutral" in sd, "sentiment_distribution must have 'neutral' key"

    assert isinstance(result.get("pain_points"), list), "pain_points must be a list"
    assert isinstance(result.get("competitors"), list), "competitors must be a list"
