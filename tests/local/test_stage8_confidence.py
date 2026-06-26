"""Local tests for compute_confidence in stage8_confidence.py — Eval 1 failure rate behavior."""

from app.pipeline.stage8_confidence import compute_confidence

_BASE = dict(scores=[5, 5, 5], relevant_count=50, processed_count=90, total_fetched=100)


def test_high_failure_rate_forces_low():
    tier, _ = compute_confidence(**_BASE, eval1_failure_rate=0.30)
    assert tier == "Low"


def test_moderate_failure_rate_caps_high_to_medium():
    tier, _ = compute_confidence(**_BASE, eval1_failure_rate=0.20)
    assert tier == "Medium"


def test_low_failure_rate_allows_high():
    tier, _ = compute_confidence(**_BASE, eval1_failure_rate=0.10)
    assert tier == "High"


def test_low_failure_rate_medium_scores():
    tier, _ = compute_confidence(scores=[3, 3, 3], relevant_count=50, processed_count=90, total_fetched=100, eval1_failure_rate=0.10)
    assert tier == "Medium"


def test_low_failure_rate_low_scores():
    tier, _ = compute_confidence(scores=[1, 1, 1], relevant_count=50, processed_count=90, total_fetched=100, eval1_failure_rate=0.10)
    assert tier == "Low"
