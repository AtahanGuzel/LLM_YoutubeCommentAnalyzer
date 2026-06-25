"""Stage 1 — Pre-processing (Metadata Classification).

Fetches video metadata (title, description, tags) and sends it to the pre-processing
model to classify whether the video is a physical product promotion and extract the
primary product. Implements the decision gate that routes to reject, ask_user, or proceed.
"""

from app.utils.groq_client import (
    PREPROCESSING_MODEL,
    PreProcessingResponseError,
    get_groq_client,
    parse_preprocessing_response,
    validate_preprocessing_response,
)
from app.utils.youtube_client import fetch_video_metadata
from app.pipeline.prompts.preprocessing import PREPROCESSING_PROMPT


async def classify_video(video_id: str, *, client=None) -> dict:
    """Fetch metadata for video_id and classify it using the pre-processing model.

    Returns the parsed response dict with all five fields:
    is_promotion, reasoning, primary_product, product_confidence, candidates.
    Raises PreProcessingResponseError if the model response cannot be parsed or validated.
    """
    metadata = fetch_video_metadata(video_id, client=client)

    user_message = (
        f"Title: {metadata['title']}\n"
        f"Description: {metadata['description']}\n"
        f"Tags: {', '.join(metadata['tags'])}"
    )

    groq = get_groq_client()
    completion = groq.chat.completions.create(
        model=PREPROCESSING_MODEL,
        messages=[
            {"role": "system", "content": PREPROCESSING_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content
    data = parse_preprocessing_response(raw)

    if not validate_preprocessing_response(data):
        raise PreProcessingResponseError(
            "Video classification failed due to an unexpected model response. Please try again."
        )

    data["video_title"] = metadata["title"]
    return data


def apply_decision_gate(response: dict) -> dict:
    """Evaluate the classify_video response and return a structured outcome dict.

    Outcomes:
      {"outcome": "reject", "message": "..."}
      {"outcome": "ask_user", "message": "...", "candidates": [...] | None}
      {"outcome": "proceed", "primary_product": "..."}
    """
    if not response.get("is_promotion"):
        return {
            "outcome": "reject",
            "message": (
                "This video does not appear to be a product promotion video. "
                "The tool only analyzes product review and promotion content."
            ),
        }

    if response.get("product_confidence") == "high":
        return {
            "outcome": "proceed",
            "primary_product": response["primary_product"],
        }

    candidates = response.get("candidates")
    if candidates:
        return {
            "outcome": "ask_user",
            "message": (
                "We couldn't confidently identify the primary product. "
                f"We found: {candidates}. Which would you like to analyze?"
            ),
            "candidates": candidates,
        }
    else:
        return {
            "outcome": "ask_user",
            "message": (
                "We couldn't confidently identify the primary product. "
                "Please specify the product name you want to analyze."
            ),
            "candidates": None,
        }
