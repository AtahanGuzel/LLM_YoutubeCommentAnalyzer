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
    validate_evaluation_response,
    LLMResponseParseError,
)
from app.pipeline.prompts.evaluation import EVALUATION_PROMPT

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


async def evaluate(
    sample: list[dict],
    primary_product: str,
    sentiment_distribution: dict,
    pain_points: list[dict],
    competitors: list[dict],
) -> dict:
    """Call the evaluation model and return the full two-scorecard parsed dict.

    The prompt places raw comment text + assigned labels BEFORE the aggregated output
    to mitigate position bias. Raises EvaluationResponseError if the response cannot
    be parsed or fails schema validation.
    """
    formatted_sample = _format_sample(sample)
    formatted_aggregated = _format_aggregated(
        sentiment_distribution, pain_points, competitors
    )

    prompt = (
        EVALUATION_PROMPT
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
    # Extract JSON from markdown code fences if present, same approach as stage4_extraction.py
    fence_matches = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if fence_matches:
        if len(fence_matches) == 1:
            raw = fence_matches[0].strip()
        else:
            # Multiple fence blocks: merge all JSON objects into one dict
            merged: dict = {}
            for block in fence_matches:
                try:
                    block_data = json.loads(block.strip())
                    if isinstance(block_data, dict):
                        merged.update(block_data)
                except (json.JSONDecodeError, ValueError):
                    pass
            raw = json.dumps(merged) if merged else fence_matches[0].strip()
    else:
        # Fall back to extracting the outermost JSON object from raw text
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            raw = raw[start:end + 1]

    try:
        data = parse_llm_json(raw)
    except LLMResponseParseError as e:
        raise EvaluationResponseError(
            f"Evaluation model returned unparseable JSON: {e}"
        ) from e

    if not validate_evaluation_response(data):
        raise EvaluationResponseError(
            "Evaluation response missing required label_quality or output_quality keys."
        )

    return data


def _log_label_quality(evaluation_result: dict) -> None:
    """Write the label_quality block to the application log at INFO level."""
    label_quality = evaluation_result.get("label_quality", {})
    logger.info(
        "Eval 1 — label_quality: %s",
        json.dumps(label_quality),
    )


async def run_evaluation(
    sample: list[dict],
    primary_product: str,
    sentiment_distribution: dict,
    pain_points: list[dict],
    competitors: list[dict],
) -> dict:
    """Run the full evaluation call and return only the output_quality scorecard.

    label_quality is routed to the application log only and is not returned.
    Returns {"output_quality": {sentiment: ..., pain_points: ..., competitors: ...}}.
    """
    result = await evaluate(
        sample, primary_product, sentiment_distribution, pain_points, competitors
    )
    _log_label_quality(result)
    return {"output_quality": result["output_quality"]}
