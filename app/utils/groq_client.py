"""Groq API client, model constants, parsers, validators, and reliability layer.

Provides the configured Groq client, authoritative model string constants,
response parsing helpers, and the retry/backoff logic used by extraction and
evaluation stages.
"""

import os
import json
from dotenv import load_dotenv
import groq

load_dotenv()


PREPROCESSING_MODEL = "openai/gpt-oss-120b"
EXTRACTION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
EVALUATION_MODEL = "openai/gpt-oss-120b"


class LLMResponseParseError(Exception):
    pass


class PreProcessingResponseError(Exception):
    pass


def parse_llm_json(text: str) -> dict | list:
    """Parse a JSON string, raising LLMResponseParseError on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMResponseParseError(str(e)) from e


_OUTPUT_QUALITY_CRITERIA = {"sentiment", "pain_points", "competitors"}
_CRITERION_FIELDS = {"evidence", "gaps", "score"}


def validate_evaluation_response(data: dict) -> bool:
    """Return True if data contains both label_quality and output_quality scorecards with required fields."""
    if not isinstance(data, dict):
        return False
    if "label_quality" not in data or "output_quality" not in data:
        return False
    lq = data["label_quality"]
    lq_required = {"evaluated_count", "passed", "failed", "failures"}
    if not lq_required.issubset(lq.keys()):
        return False
    oq = data["output_quality"]
    if not _OUTPUT_QUALITY_CRITERIA.issubset(oq.keys()):
        return False
    for criterion in _OUTPUT_QUALITY_CRITERIA:
        if not _CRITERION_FIELDS.issubset(oq[criterion].keys()):
            return False
    return True


_VALID_COMMENT_TYPES = {"target", "competitor_focus", "noise"}
_VALID_SENTIMENTS = {"positive", "negative", "neutral", None}


def validate_extraction_response(data: list) -> bool:
    """Return True if data is a list of comment dicts with valid comment_type and sentiment values."""
    if not isinstance(data, list):
        return False
    for item in data:
        if not isinstance(item, dict):
            return False
        required_keys = {"comment_id", "comment_type", "sentiment", "pain_points", "competitor_mentions"}
        if not required_keys.issubset(item.keys()):
            return False
        if item["comment_type"] not in _VALID_COMMENT_TYPES:
            return False
        if item["sentiment"] not in _VALID_SENTIMENTS:
            return False
        if not isinstance(item["pain_points"], list):
            return False
        if not isinstance(item["competitor_mentions"], list):
            return False
    return True


def validate_preprocessing_response(data: dict) -> bool:
    """Return True if data has all required pre-processing fields with valid values."""
    required_keys = {"is_promotion", "reasoning", "primary_product", "product_confidence", "candidates"}
    if not required_keys.issubset(data.keys()):
        return False
    if not isinstance(data["is_promotion"], bool):
        return False
    if data["product_confidence"] not in ("high", "low"):
        return False
    return True


def parse_extraction_response(text: str):
    """Parse extraction model JSON output, returning None instead of raising on failure."""
    try:
        return parse_llm_json(text)
    except LLMResponseParseError:
        return None


def parse_preprocessing_response(text: str):
    """Parse pre-processing model JSON output, raising PreProcessingResponseError on failure."""
    try:
        return parse_llm_json(text)
    except LLMResponseParseError:
        raise PreProcessingResponseError(
            "Video classification failed due to an unexpected model response. Please try again."
        )


def get_groq_client() -> groq.Groq:
    """Return a Groq client configured with GROQ_API_KEY."""
    api_key = os.getenv("GROQ_API_KEY")
    return groq.Groq(api_key=api_key)
