"""Stage 7 — Evaluation.

Sends the stratified sample and aggregated output to the evaluation model in a single
call. Produces two scorecards: Eval 1 (label quality, backend log only) and Eval 2
(output quality, feeds PDF).
"""

import json
import logging
import re

from app.utils.groq_client import (
    EVALUATION_MODEL,
    get_groq_client,
    parse_llm_json,
    LLMResponseParseError,
)
from app.pipeline.prompts.eval1_label_quality import EVAL1_PROMPT
from app.pipeline.prompts.eval2_output_quality import EVAL2_PROMPT

logger = logging.getLogger(__name__)


class EvaluationResponseError(Exception):
    pass


def _format_sample(sample: list[dict]) -> str:
    """Format sampled comments with their labels for the evaluation prompt."""
    lines = []
    for c in sample:
        label = {
            "comment_type": c.get("comment_type"),
            "sentiment": c.get("sentiment"),
            "pain_points": c.get("pain_points", []),
            "competitor_mentions": c.get("competitor_mentions", []),
        }
        lines.append(
            f"[{c.get('comment_id')}] {c.get('text', '')}\n"
            f"  Label: {json.dumps(label)}"
        )
    return "\n\n".join(lines)


def _format_aggregated(
    sentiment_distribution: dict,
    pain_points: list[dict],
    competitors: list[dict],
) -> str:
    """Format the Stage 5 aggregated output for the evaluation prompt."""
    return json.dumps(
        {
            "sentiment_distribution": sentiment_distribution,
            "pain_points": pain_points,
            "competitors": competitors,
        },
        indent=2,
    )


def _extract_json_from_raw(raw: str):
    """Extract JSON text from raw LLM response, stripping markdown fences."""
    fence_matches = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if fence_matches:
        return fence_matches[0].strip()
    start = raw.find("[") if raw.find("[") != -1 and (raw.find("{") == -1 or raw.find("[") < raw.find("{")) else raw.find("{")
    if start == -1:
        return raw
    end_bracket = raw.rfind("]")
    end_brace = raw.rfind("}")
    end = max(end_bracket, end_brace)
    if end > start:
        return raw[start:end + 1]
    return raw


async def evaluate_label_quality(
    sample_a: list[dict],
    labeled_comments: list[dict],
) -> list[dict]:
    """Call the evaluation model to audit label quality for Sample A.

    Presents raw comment text before each assigned label to mitigate position
    bias. Returns the full verdict array (list of dicts with comment_id, correct,
    issue). Raises EvaluationResponseError on malformed output.
    """
    lines = []
    for c in sample_a:
        label = {
            "comment_type": c.get("comment_type"),
            "sentiment": c.get("sentiment"),
            "pain_points": c.get("pain_points", []),
            "competitor_mentions": c.get("competitor_mentions", []),
        }
        lines.append(
            f"[{c.get('comment_id')}] {c.get('text', '')}\n"
            f"  Label: {json.dumps(label)}"
        )
    formatted_sample = "\n\n".join(lines)

    prompt = (
        EVAL1_PROMPT
        .replace("{n}", str(len(sample_a)))
        .replace("{sampled_comments}", formatted_sample)
    )

    client = get_groq_client()
    completion = client.chat.completions.create(
        model=EVALUATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = completion.choices[0].message.content
    raw = _extract_json_from_raw(raw)

    try:
        data = parse_llm_json(raw)
    except LLMResponseParseError as e:
        raise EvaluationResponseError(
            f"Eval 1 model returned unparseable JSON: {e}"
        ) from e

    if not isinstance(data, list) or not all(
        isinstance(v, dict) and "comment_id" in v and "correct" in v and "issue" in v
        for v in data
    ):
        raise EvaluationResponseError(
            "Eval 1 response is not a valid verdict array."
        )

    logger.info("Eval 1 verdicts: %s", json.dumps(data))
    return data


def compute_eval1_failure_rate(verdicts: list[dict]) -> float:
    """Return the fraction of verdicts where correct is False."""
    failed = sum(1 for v in verdicts if not v["correct"])
    return failed / len(verdicts)


async def evaluate_output_quality(
    sample_b: list[dict],
    aggregated_output: dict,
) -> dict:
    """Call the evaluation model to score output quality using Sample B.

    Presents raw comment text and assigned labels before the aggregated output
    to mitigate position bias. Returns the output_quality dict (with sentiment,
    pain_points, competitors sub-keys each containing evidence, gaps, score).
    Raises EvaluationResponseError on malformed output.
    """
    formatted_sample = _format_sample(sample_b)
    formatted_aggregated = json.dumps(aggregated_output, indent=2)

    primary_product = aggregated_output.get("primary_product", "the target product")

    prompt = (
        EVAL2_PROMPT
        .replace("{primary_product}", primary_product)
        .replace("{sampled_comments}", formatted_sample)
        .replace("{aggregated_output}", formatted_aggregated)
    )

    client = get_groq_client()
    completion = client.chat.completions.create(
        model=EVALUATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = completion.choices[0].message.content
    raw = _extract_json_from_raw(raw)

    try:
        data = parse_llm_json(raw)
    except LLMResponseParseError as e:
        raise EvaluationResponseError(
            f"Eval 2 model returned unparseable JSON: {e}"
        ) from e

    if not isinstance(data, dict) or "output_quality" not in data:
        raise EvaluationResponseError(
            "Eval 2 response missing required output_quality key."
        )
    oq = data["output_quality"]
    for criterion in ("sentiment", "pain_points", "competitors"):
        if criterion not in oq:
            raise EvaluationResponseError(
                f"Eval 2 output_quality missing criterion: {criterion}"
            )
        for field in ("evidence", "gaps", "score"):
            if field not in oq[criterion]:
                raise EvaluationResponseError(
                    f"Eval 2 output_quality.{criterion} missing field: {field}"
                )

    return oq


