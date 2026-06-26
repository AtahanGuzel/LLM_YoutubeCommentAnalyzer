"""FastAPI entry point for the YouTube Comment Analyzer service."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.pipeline.stage0_url_parser import extract_video_id
from app.pipeline.stage1_preprocessor import apply_decision_gate, classify_video
from app.pipeline.stage2_comment_fetcher import fetch_and_filter_comments
from app.pipeline.stage3_threshold import check_threshold
from app.pipeline.stage4_extraction import apply_post_extraction_threshold, extract_labels
from app.pipeline.stage5_aggregation import (
    compute_sentiment_distribution,
    count_competitors,
    rank_pain_points,
)
from app.pipeline.stage6_sampling import select_random_sample, select_stratified_sample
from app.pipeline.stage7_evaluation import (
    evaluate_label_quality,
    compute_eval1_failure_rate,
    evaluate_output_quality,
)
from app.pipeline.stage8_confidence import compute_confidence, find_score_one_criteria
from app.pipeline.stage9_pdf import generate_pdf
from app.utils.youtube_client import fetch_video_metadata

_OUTPUT_DIR = Path(__file__).parent.parent / "output"

logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/")
def health_check():
    """Return service liveness status."""
    return {"status": "ok"}


async def run_pipeline(video_url: str, product_name: str | None = None) -> dict:
    """Run the full analysis pipeline from URL to filtered English comment list.

    Path A (product_name is None): classifies video via stage1, returns early on
    reject/ask_user outcomes, otherwise proceeds to comment fetch and threshold check.

    Path B (product_name provided): skips stage1 classification, fetches metadata
    for video_title only, then proceeds directly to comment fetch and threshold check.
    """
    video_id = extract_video_id(video_url)
    if video_id is None:
        return {"outcome": "reject", "message": "Invalid or unrecognised YouTube URL."}

    if product_name is None:
        # Path A: run pre-processing classification
        classification = await classify_video(video_id)
        gate = apply_decision_gate(classification)

        if gate["outcome"] in ("reject", "ask_user"):
            return gate

        primary_product = gate["primary_product"]
        video_title = classification["video_title"]
    else:
        # Path B: skip classification, fetch metadata for title only
        primary_product = product_name
        metadata = fetch_video_metadata(video_id)
        video_title = metadata["title"]

    comments = fetch_and_filter_comments(video_id)
    quality_mode, _ = check_threshold(len(comments))

    # Stage 4 — chunked extraction
    labeled_comments, processed_count = await extract_labels(comments, primary_product)
    post_threshold = apply_post_extraction_threshold(processed_count)
    if isinstance(post_threshold, dict) and post_threshold.get("outcome") == "error":
        return post_threshold

    quality_mode = post_threshold[0]

    # Stage 5 — code aggregation
    sentiment_distribution = compute_sentiment_distribution(labeled_comments)
    pain_points = rank_pain_points(labeled_comments)
    competitors = count_competitors(labeled_comments)

    # Stage 6 — sample selection (two independent samples)
    sample_a = select_random_sample(labeled_comments)
    sample_b = select_stratified_sample(labeled_comments, pain_points, competitors)

    # Stage 7 — Eval 1: label quality audit (verdict array → log + failure rate)
    verdicts = await evaluate_label_quality(sample_a, labeled_comments)
    eval1_failure_rate = compute_eval1_failure_rate(verdicts)
    logger.info("Eval 1 — verdicts: %s", json.dumps(verdicts))

    # Stage 7 — Eval 2: output quality scoring (feeds PDF scorecard)
    aggregated_output = {
        "primary_product": primary_product,
        "sentiment_distribution": sentiment_distribution,
        "pain_points": pain_points,
        "competitors": competitors,
    }
    output_quality = await evaluate_output_quality(sample_b, aggregated_output)

    # Stage 8 — confidence computation
    scores = [output_quality[c]["score"] for c in ("sentiment", "pain_points", "competitors")]
    relevant_count = (
        sentiment_distribution["positive"]
        + sentiment_distribution["negative"]
        + sentiment_distribution["neutral"]
    )
    confidence_tier, confidence_warning = compute_confidence(
        scores, relevant_count, processed_count, len(comments), eval1_failure_rate
    )
    score_one_warnings = find_score_one_criteria(output_quality)

    # Stage 9 — PDF generation
    analysis_timestamp = datetime.now(timezone.utc)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = analysis_timestamp.strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"report_{video_id}_{ts_str}.pdf"
    pdf_path = str(_OUTPUT_DIR / pdf_filename)

    generate_pdf(
        output_path=pdf_path,
        video_title=video_title,
        video_url=video_url,
        analysis_timestamp=analysis_timestamp,
        comments_fetched=len(comments),
        processed_count=processed_count,
        relevant_count=relevant_count,
        quality_mode=quality_mode,
        primary_product=primary_product,
        sentiment_distribution=sentiment_distribution,
        pain_points=pain_points,
        competitors=competitors,
        confidence_tier=confidence_tier,
        confidence_warning=confidence_warning,
        output_quality=output_quality,
    )

    return {
        "outcome": "proceed",
        "video_id": video_id,
        "video_title": video_title,
        "primary_product": primary_product,
        "quality_mode": quality_mode,
        "comments_fetched": len(comments),
        "processed_count": processed_count,
        "sentiment_distribution": sentiment_distribution,
        "pain_points": pain_points,
        "competitors": competitors,
        "confidence_tier": confidence_tier,
        "confidence_warning": confidence_warning,
        "score_one_warnings": score_one_warnings,
        "output_quality": output_quality,
        "pdf_path": pdf_path,
    }


class AnalyzeRequest(BaseModel):
    video_url: str
    product_name: str | None = None


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Analyze a YouTube product promotion video's comments."""
    result = await run_pipeline(request.video_url, request.product_name)

    if result.get("outcome") == "reject":
        return JSONResponse(status_code=422, content={"detail": result["message"]})

    if result.get("outcome") == "ask_user":
        return JSONResponse(
            status_code=200,
            content={
                "needs_clarification": True,
                "candidates": result.get("candidates"),
                "message": result["message"],
            },
        )

    return FileResponse(
        path=result["pdf_path"],
        media_type="application/pdf",
        filename=os.path.basename(result["pdf_path"]),
    )
