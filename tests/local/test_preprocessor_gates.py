"""Local tests for apply_decision_gate in stage1_preprocessor.py.

Covers all four decision paths:
  - is_promotion false → reject
  - is_promotion true + product_confidence low + candidates → ask_user with candidates
  - is_promotion true + product_confidence low + no candidates → ask_user without candidates
  - is_promotion true + product_confidence high → proceed
"""

from app.pipeline.stage1_preprocessor import apply_decision_gate


def test_not_promotion_returns_reject():
    response = {
        "is_promotion": False,
        "reasoning": "This is a cooking tutorial.",
        "primary_product": None,
        "product_confidence": "low",
        "candidates": None,
    }
    result = apply_decision_gate(response)
    assert result["outcome"] == "reject"
    assert result["message"] == (
        "This video does not appear to be a product promotion video. "
        "The tool only analyzes product review and promotion content."
    )


def test_low_confidence_with_candidates_returns_ask_user():
    response = {
        "is_promotion": True,
        "reasoning": "Multiple products present.",
        "primary_product": None,
        "product_confidence": "low",
        "candidates": ["iPhone 17", "Samsung Galaxy S25"],
    }
    result = apply_decision_gate(response)
    assert result["outcome"] == "ask_user"
    assert "iPhone 17" in result["message"]
    assert "Samsung Galaxy S25" in result["message"]
    assert result["candidates"] == ["iPhone 17", "Samsung Galaxy S25"]


def test_low_confidence_no_candidates_returns_ask_user():
    response = {
        "is_promotion": True,
        "reasoning": "Product unclear.",
        "primary_product": None,
        "product_confidence": "low",
        "candidates": None,
    }
    result = apply_decision_gate(response)
    assert result["outcome"] == "ask_user"
    assert "Please specify the product name you want to analyze." in result["message"]
    assert result["candidates"] is None


def test_high_confidence_returns_proceed():
    response = {
        "is_promotion": True,
        "reasoning": "Clear iPhone review.",
        "primary_product": "iPhone 17",
        "product_confidence": "high",
        "candidates": None,
    }
    result = apply_decision_gate(response)
    assert result["outcome"] == "proceed"
    assert result["primary_product"] == "iPhone 17"
