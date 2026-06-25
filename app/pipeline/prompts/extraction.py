EXTRACTION_PROMPT = """You are analyzing comments about: {primary_product}

For each comment, classify comment_type as:
- "target"           : contains an opinion, experience, or observation
                       about {primary_product}
- "competitor_focus" : primarily discusses another product or brand
                       rather than {primary_product}
- "noise"            : no meaningful product-related content
                       (reactions, jokes, emojis only, off-topic filler)

Rules:
- Only assign sentiment when comment_type is "target"
- sentiment is null for competitor_focus and noise
- pain_points only populated for target comments with negative/neutral sentiment
- competitor_mentions populated for competitor_focus and target comments
  that reference other brands

For each comment output JSON with this schema:
{
  "comment_id": <id from input>,
  "comment_type": "target" | "competitor_focus" | "noise",
  "sentiment": "positive" | "negative" | "neutral" | null,
  "pain_points": ["string", ...],
  "competitor_mentions": ["brand name", ...]
}

Output a JSON array containing one object per comment."""
