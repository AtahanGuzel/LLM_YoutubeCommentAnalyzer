# Product promotion video used: "iPhone 17 Review: No Asterisks!" (MKBHD)
# URL: https://www.youtube.com/watch?v=rng_yUSwrgU
# Live call: 1 YouTube metadata unit (stage 1) + 1 comment page unit (stage 2)
# + Groq extraction calls (stage 4) + 1 Groq evaluation call (stage 7)

import asyncio
import json
import logging

from app.main import run_pipeline

PRODUCT_PROMO_URL = "https://www.youtube.com/watch?v=rng_yUSwrgU"
_VALID_TIERS = {"High", "Medium", "Low"}


def test_checkpoint3_stages6_7_8(caplog):
    """Smoke test: pipeline stages 6-8 produce confidence tier and output_quality evidence."""
    with caplog.at_level(logging.INFO, logger="app.pipeline.stage7_evaluation"):
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

    # label_quality and eval1_verdicts must NOT be in the returned result
    assert "label_quality" not in result, "label_quality must not be present in run_pipeline return value"
    assert "eval1_verdicts" not in result, "eval1_verdicts must not be present in run_pipeline return value"

    # Stage 7 — Eval 1 verdict array must be present in app.pipeline.stage7_evaluation log
    stage7_records = [
        r for r in caplog.records
        if r.name == "app.pipeline.stage7_evaluation"
    ]
    verdict_array_found = False
    for record in stage7_records:
        msg = record.getMessage()
        start = msg.find("[")
        if start != -1:
            try:
                data = json.loads(msg[start:])
                if (
                    isinstance(data, list)
                    and len(data) > 0
                    and all(
                        isinstance(v, dict)
                        and "comment_id" in v
                        and "correct" in v
                        and "issue" in v
                        for v in data
                    )
                ):
                    verdict_array_found = True
                    break
            except (json.JSONDecodeError, ValueError):
                continue
    assert verdict_array_found, (
        "Expected Eval 1 verdict array in app.pipeline.stage7_evaluation log, "
        f"but found records: {[r.getMessage() for r in stage7_records]}"
    )
