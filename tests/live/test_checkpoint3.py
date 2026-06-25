# Product promotion video used: "iPhone 17 Review: No Asterisks!" (MKBHD)
# URL: https://www.youtube.com/watch?v=rng_yUSwrgU
# Live call: 1 YouTube metadata unit (stage 1) + 1 comment page unit (stage 2)
# + Groq extraction calls (stage 4) + 1 Groq evaluation call (stage 7)

import asyncio

from app.main import run_pipeline

PRODUCT_PROMO_URL = "https://www.youtube.com/watch?v=rng_yUSwrgU"
_VALID_TIERS = {"High", "Medium", "Low"}


def test_checkpoint3_stages6_7_8():
    """Smoke test: pipeline stages 6-8 produce confidence tier and output_quality evidence."""
    result = asyncio.run(run_pipeline(PRODUCT_PROMO_URL))

    assert result.get("outcome") == "proceed", (
        f"Expected proceed outcome, got: {result.get('outcome')} — {result.get('message')}"
    )

    # Stage 8 — confidence tier
    tier = result.get("confidence_tier")
    assert tier in _VALID_TIERS, f"confidence_tier must be High/Medium/Low, got: {tier!r}"

    # Stage 7 — output_quality evidence strings for all three criteria
    oq = result.get("output_quality")
    assert isinstance(oq, dict), "output_quality must be a dict"
    for criterion in ("sentiment", "pain_points", "competitors"):
        assert criterion in oq, f"output_quality missing '{criterion}' key"
        evidence = oq[criterion].get("evidence")
        assert isinstance(evidence, str) and evidence, (
            f"output_quality['{criterion}']['evidence'] must be a non-empty string"
        )

    # label_quality must NOT be in the returned result
    assert "label_quality" not in result, "label_quality must not be present in run_pipeline return value"
