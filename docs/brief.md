# Project Brief: LLM-Based YouTube Product Promotion Video Comment Analyzer
**Version:** 4.0 — Revised  
**Scope:** v1 Physical Products Only  
**Status:** Ready for Implementation

---

## Overview

An LLM-based comment analysis tool that takes a YouTube product promotion video URL,
fetches and filters its comments, extracts structured labels per comment using Llama 4 Scout 17B,
aggregates the results in code, evaluates output quality using openai/gpt-oss-120b as a judge,
and generates a PDF report with charts and an evaluation scorecard.

---

## Target Audience

- Brand Managers
- Brand Representatives
- Marketing Professionals

---

## Tech Stack

| Layer | Choice |
|---|---|
| Backend framework | FastAPI (async-native) |
| LLM provider | Groq API |
| Extraction model | meta-llama/llama-4-scout-17b-16e-instruct |
| Evaluation model | openai/gpt-oss-120b |
| Language detection | langdetect |
| Chart generation | Matplotlib + Seaborn |
| PDF generation | ReportLab |
| YouTube data | YouTube Data API v3 |

---

## Scope Constraints

- **v1 only covers physical products.** Games, software, and digital services are deferred
  to a future version. The pre-processing classifier should be aware of this and may reject
  non-physical-product videos.
- English comments only.
- Top-level comments only. Replies are ignored in v1.
- Spam and bot filtering is not implemented in v1.

---

## Full Pipeline

### Stage 0 — Input & URL Parsing

The user provides a YouTube video URL. The system extracts the video ID from the URL.
All four YouTube URL formats must be handled:

```
youtube.com/watch?v={id}
youtu.be/{id}
youtube.com/shorts/{id}
youtube.com/live/{id}
```

---

### Stage 1 — Pre-processing (Metadata Classification)

Pre-processing runs before any comment is fetched. The video ID is used to retrieve
metadata (title, description, tags) from the YouTube Data API. This metadata is sent
to openai/gpt-oss-120b in a single call that performs two tasks simultaneously:

1. Classify whether the video is a product promotion video
2. Extract the primary product being analyzed

**Why gpt-oss-120b for pre-processing:** This is one small call (~300 tokens) with negligible
quota impact. The classification accuracy benefit justifies using the stronger model
here, and it keeps Scout 17B exclusively responsible for extraction work.

**Pre-processing prompt structure:**

```
Think step by step:
1. What is the central topic of this video?
2. Is a specific physical product the main subject?
3. Is the video's purpose to review, showcase, compare, or promote that product?
4. If yes, what is the exact product name?
5. If multiple products are present, is one clearly the primary subject?

Output JSON only:
{
  "is_promotion": true | false,
  "reasoning": "one sentence",
  "primary_product": "exact product name or null",
  "product_confidence": "high" | "low",
  "candidates": ["product1", "product2"] | null
}
```

**Pre-processing decision logic:**

```
is_promotion: false
  → Reject. Return user error: "This video does not appear to be a product
    promotion video. The tool only analyzes product review and promotion content."

is_promotion: true AND product_confidence: high
  → Proceed to comment fetching with primary_product locked.

is_promotion: true AND product_confidence: low
  → Pause. Ask user:
    "We couldn't confidently identify the primary product.
     [If candidates exist]: We found: {candidate list}. Which would you like to analyze?
     [If no candidates]: Please specify the product name you want to analyze."
  → Resume after user input.
```

Note: `confidence` is not tracked for the promotion classification itself —
only `is_promotion` true/false drives that gate. The `product_confidence`
field only has two values: `high` and `low`. Medium was removed as it collapsed
into the same behavior as low.

**Error states handled at this stage:**
- Video not found → return user error
- Private or age-restricted video (403) → return user error:
  "This video is private or age-restricted and cannot be accessed."
- Comments disabled → return user error:
  "This video has comments disabled."

---

### Stage 2 — Comment Fetching (Fetch → Filter → Cycle)

Comments are fetched only after pre-processing resolves successfully and
primary product is confirmed. This avoids the UX problem of fetching first
then asking the user a configuration question.

**Fetch parameters:**
- `order: relevance` (YouTube's engagement-weighted sort — no native like-count sort exists)
- Top-level comments only (no replies)
- 100 comments per API page (1 quota unit per page)

**Fetch-filter cycle:**

```
fetch 100 comments
        ↓
langdetect filter (English only, seed=0 for determinism)
        ↓
english_count >= 100?   → stop, proceed to threshold check
english_count < 100?    → fetch next 100 (nextPageToken)
no nextPageToken?       → stop, proceed with what we have
total_fetched >= 500?   → stop, proceed with what we have (hard cap)
```

**Caps:**
- Target: 100 English comments before stopping
- Hard cap: 500 total comments fetched per analysis (5 API quota units)
- Daily quota: YouTube API provides 10,000 units/day. At 5 units max per analysis,
  this supports up to 2,000 analyses per day.

**Language detection implementation:**
```python
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except:
        return False
```

---

### Stage 3 — Threshold Check

Applied against English comment count after the fetch cycle completes.

```
English comments > 100   → proceed, "full" quality mode
English comments 51–100  → proceed, "degraded" quality mode (noted in PDF header)
English comments ≤ 50    → reject. Return user error:
                           "Insufficient English comments found ({n} detected).
                            A minimum of 51 comments is required for analysis."
```

---

### Stage 4 — Chunked Extraction (Llama 4 Scout 17B)

Comments are sent to Llama 4 Scout 17B in chunks of 25 for structured label extraction.
The model classifies each comment relative to the primary product.

**Comment label schema:**

```json
{
  "comment_id": 12,
  "comment_type": "target" | "competitor_focus" | "noise",
  "sentiment": "positive" | "negative" | "neutral" | null,
  "pain_points": ["string", ...],
  "competitor_mentions": ["brand name", ...]
}
```

**Label definitions provided in the extraction prompt:**

```
You are analyzing comments about: {primary_product}

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
```

**Downstream routing by label:**

```
target            → sentiment aggregation + pain point counting
competitor_focus  → competitor mention counting only
noise             → excluded from all aggregation
```

**Chunk failure handling:**

Each chunk is retried up to 3 times before being marked as failed:

```python
async def process_chunk_with_retry(chunk, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await call_groq(chunk)
        except RateLimitError:
            await asyncio.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
        except (ServerError, TimeoutError):
            await asyncio.sleep(1)
    return None
```

After all retries, apply the same threshold to successfully processed comments:

```
processed > 100   → proceed normally
processed 51–100  → proceed, flagged as degraded
processed ≤ 50    → fail with error message
```

---

### Stage 5 — Code Aggregation

All counting and ranking happens in code after extraction. The LLM never counts.

**Sentiment distribution:**
Count comments with `comment_type: target` only. Compute percentage breakdown
across positive / negative / neutral. Comments labeled competitor_focus or noise
are excluded from sentiment entirely.

**Pain point ranking (weighted score):**

```python
import math

def weighted_score(mention_count: int, total_likes: int) -> float:
    return mention_count + math.log(total_likes + 1)
```

Frequency drives ranking. Like count provides a soft boost via log scaling.
Top 5 pain points by weighted score are selected for the report.
If zero pain points are identified, the section renders as:
"No pain points identified in the analyzed comments."

**Competitor threshold:**
Competitor brands are counted by raw comment count only (no like weighting).
The threshold rule is a hard business rule and must not be fuzzy.

```
brand mention count >= 10  → included in report
brand mention count 1–9    → excluded, not shown
brand mention count 0      → report section shows:
                             "No competitors meeting the minimum threshold
                              (10 mentions) were identified."
```

---

### Stage 6 — Evaluation Calls (No Sampling)

Sampling has been removed from the pipeline. Both evaluation calls operate
directly on the full labeled comment dataset. No sample selection step exists.

Eval 1 receives all labeled comments (up to 100). Eval 2 receives Eval 1 output —
it is not an independent pipeline call against raw comments.

---

### Stage 7 — Evaluation

**Eval 1 Call — Label quality (engineering diagnostic)**

Function name: `evaluate_label_quality`

Receives all labeled comments from the extraction stage. Calls gpt-oss-120b
to audit every comment's assigned labels against the raw comment text.

Token usage: ~9,100 tokens per call. Exceeds the 8,000 TPM ceiling by ~1,100 tokens.
Retry logic and exponential backoff handle the rate limit hit gracefully.
Expected wait: ~1 minute.

**Eval 1 prompt structure:**

```
You are auditing the label quality of a comment extraction model.
The model was asked to analyze comments about: {primary_product}

For each comment below, check whether the assigned labels are correct.
Identify the first field that is wrong, if any. Do not evaluate downstream
fields if comment_type is incorrect.

Output JSON only — one entry per comment.
```

**Eval 1 output schema (per comment):**

```json
[
  {
    "comment_id": "001",
    "issue_field": null,
    "issue": null
  },
  {
    "comment_id": "012",
    "issue_field": "competitor_mentions",
    "issue": "comment says 'switched from Sony' but competitor_mentions is empty"
  }
]
```

Field rules:
- `issue_field`: `comment_type` | `sentiment` | `pain_points` | `competitor_mentions` | `null`
- `issue`: one sentence description of the problem, or `null` if correct
- Single issue per comment — hierarchical check order: `comment_type` first,
  downstream fields only evaluated if `comment_type` is correct

**Eval 1 aggregate summary (computed by judge as part of same response):**

```json
{
  "comment_type_failures": 3,
  "sentiment_failures": 2,
  "pain_points_failures": 4,
  "competitor_mentions_failures": 1,
  "total_failure_rate": 0.10
}
```

**Eval 1 failure rate computation (code):**

Function name: `compute_eval1_failure_rate`

```python
def compute_eval1_failure_rate(aggregate: dict) -> float:
    return aggregate["total_failure_rate"]
```

**Eval 1 output routing:**
- Full verdict array (all comments, correct and incorrect) → backend log at INFO level
- Aggregate summary → passed to Eval 2 for narrative generation
- Failed cases only (non-null issue_field entries) → passed to Eval 2
- Total failure rate float → passed to Stage 8 confidence computation
- Nothing from Eval 1 appears in the PDF directly

Engineering interpretation: failure rate above ~15% indicates the extraction
prompt needs revision. Failures clustered on a specific field indicate targeted
prompt issues.

---

**Eval 2 Call — Narrative generation (feeds PDF)**

Function name: `evaluate_output_quality`

Eval 2 is a narrative generation call, not a scoring call. It receives Eval 1
aggregate summary and the failed cases list. It generates business-language
summary sentences for each criterion that appear in the PDF scorecard.

No rubric. No 1–5 scores. No sample of raw comments.

**Eval 2 input:**

```
Eval 1 aggregate:
  comment_type failures: {n}
  sentiment failures: {n}
  pain_points failures: {n}
  competitor_mentions failures: {n}
  total failure rate: {%}

Failed cases:
  [comment_id, issue_field, issue_string per failed comment]
```

Only failed cases are sent — correct verdicts are excluded.
Typical token cost: ~500–1,000 tokens per call. Well within TPM ceiling.

**Eval 2 output schema:**

```json
{
  "sentiment_summary": "business language sentence",
  "pain_points_summary": "business language sentence",
  "competitor_summary": "business language sentence",
  "confidence_insight": "overall pattern observation if meaningful, or null"
}
```

Summary sentences are written for a non-technical brand manager audience.
They describe what the failure patterns mean for result reliability,
not what went wrong technically.

Example outputs:
- ✓ "Sentiment classification is well supported across the analyzed comments."
- ⚠ "Some competitor references may have been missed — findings should be treated as directional."
- ⚠ "Pain point analysis is based on limited negative comments — interpret with caution."

`confidence_insight` is populated only when failure patterns across fields
share a common root cause worth surfacing. Otherwise null.

---

### Stage 8 — Confidence Computation (Code)

Computed from Eval 1 failure rate and data quality signals.
Eval 2 scores are not used — Eval 2 is narrative only.
Check order is fixed — earlier checks short-circuit the function.

```python
def compute_confidence(
    relevant_count: int,
    processed_count: int,
    total_fetched: int,
    eval1_failure_rate: float
) -> tuple[str, str | None]:

    loss_ratio = (total_fetched - processed_count) / total_fetched

    if loss_ratio > 0.25:
        return "Low", f"Analysis based on {processed_count} of {total_fetched} \
                        comments due to processing errors."

    if relevant_count < 30:
        return "Low", "Fewer than 30 comments were relevant to the target product."

    if eval1_failure_rate > 0.20:
        return "Low", "Extraction quality is low. \
                        Manual review of comment labels is recommended."

    if eval1_failure_rate > 0.10:
        return "Medium", "Results should be treated as directional."

    return "High", None
```

**Eval 1 failure rate thresholds:**

```
failure_rate > 0.20  → confidence forced to Low
failure_rate > 0.10  → confidence Medium
failure_rate ≤ 0.10  → confidence High
```

Additionally, any single field with a failure rate above 10% triggers a
per-section warning (⚠) in the PDF scorecard regardless of overall confidence tier.

---

### Stage 9 — PDF Report Generation (ReportLab + Matplotlib/Seaborn)

**PDF page layout:**

```
1. Header
2. Sentiment section (Chart 1)
3. Pain Points section (Chart 2)
4. Competitor Analysis section (Chart 3)
5. Evaluation Scorecard
6. Footer
```

**Header content:**
```
Video title
Video URL
Analysis date
Comments fetched:           {n}
Successfully processed:     {n}  [⚠ if chunks failed]
Relevant to [{product}]:    {n}
Quality mode: Full / Degraded
```

**Confidence banner (appears directly below header):**
```
High   → no banner
Medium → yellow banner: "Analysis confidence is moderate.
                         One or more findings have limited sample support.
                         Treat as directional."
Low    → red banner:    "Analysis confidence is low.
                         Manual review of comments is recommended
                         before acting on these findings."
```

**Chart 1 — Sentiment distribution:**
- Type: Donut chart
- Segments: Positive / Neutral / Negative
- Center label: "{n} relevant comments"
- Legend: each segment label with percentage
- Colors: green / gray / red
- Note: competitor_focus and noise comments excluded from this chart entirely

**Chart 2 — Pain points:**
- Type: Horizontal bar chart
- Y axis: pain point labels ranked 1–5 by weighted score
- X axis: weighted score
- Secondary label on each bar: "mentioned in {n} comments"
- Replaced by text note if zero pain points identified

**Chart 3 — Competitor mentions:**
- Type: Horizontal bar chart
- Y axis: competitor brand names
- X axis: raw mention count
- Reference line at x=10 (threshold)
- Replaced by text note if no competitors meet threshold

**Evaluation Scorecard section:**

```
Overall confidence: [HIGH / MEDIUM / LOW]  (large, prominent)

Per-criterion breakdown:
Sentiment     [✓ / ⚠]  {sentiment_summary from Eval 2}
Pain points   [✓ / ⚠]  {pain_points_summary from Eval 2}
Competitors   [✓ / ⚠]  {competitor_summary from Eval 2}

✓ = field failure rate ≤ 10%
⚠ = field failure rate > 10%
```

Summary sentences come from Eval 2 narrative output.
Raw failure rates and numeric scores are not shown in the PDF.
They are logged to the backend only.
Eval 1 verdict array is never shown in the PDF. It is logged to the backend only.

**Footer:**
```
Extraction model: meta-llama/llama-4-scout-17b-16e-instruct
Evaluation model: openai/gpt-oss-120b
Analysis timestamp: {datetime}
Comment fetch parameters: relevance sort, top-level only, max 500 fetched
```

---

## Data Flow Summary

```
User provides YouTube URL
        ↓
Extract video ID (handle all 4 URL formats)
        ↓
Fetch metadata (title, description, tags)
        ↓
gpt-oss-120b pre-processing call
  ├─ not promotion      → reject with message
  ├─ product_confidence low → ask user to specify product
  └─ product_confidence high → proceed
        ↓
Fetch-langdetect cycle (target: 100 English, cap: 500 total)
        ↓
Threshold check on English comment count
  └─ ≤ 50 → reject with message
        ↓
Scout 17B chunked extraction (25 comments/chunk, retry up to 3×)
  └─ chunk failures → recheck threshold on processed count
        ↓
Code aggregation
  ├─ sentiment distribution (target comments only)
  ├─ pain points ranked by weighted score (log likes + count)
  └─ competitors by raw count, ≥10 threshold
        ↓
gpt-oss-120b Eval 1 call (all labeled comments, up to 100)
  ├─ per-comment verdict (issue_field + issue) → backend log
  ├─ aggregate summary (failures per field + total rate) → Eval 2 + Stage 8
  └─ failed cases list → Eval 2
        ↓
gpt-oss-120b Eval 2 call (Eval 1 aggregate + failed cases)
  └─ narrative summaries per field + confidence insight → PDF scorecard
        ↓
Confidence computation (code)
  ├─ data quality signals (loss ratio, relevant count)
  └─ Eval 1 failure rate (drives confidence tier)
        ↓
PDF generation (ReportLab + Matplotlib/Seaborn)
        ↓
Return PDF to user via FastAPI file response
```

---

## API Quota Budget

| Operation | Units | Notes |
|---|---|---|
| Metadata fetch | 1 unit | Per analysis |
| Comment fetch cycle | 1–5 units | Per 100-comment page, max 5 pages |
| Max per analysis | 6 units | |
| Daily capacity | ~1,600 analyses | At 10,000 unit daily quota |

---

## Groq Token Budget

| Call | Model | Tokens | TPM ceiling | Est. wait |
|---|---|---|---|---|
| Pre-processing | openai/gpt-oss-120b | ~400 | 8,000 | negligible |
| Per extraction chunk | meta-llama/llama-4-scout-17b-16e-instruct | ~3,500 | 30,000 | negligible |
| 8 chunks total | meta-llama/llama-4-scout-17b-16e-instruct | ~28,000 | 30,000 | ~1 min at free tier |
| Eval 1 (all labeled comments, up to 100) | openai/gpt-oss-120b | ~9,100 | 8,000 | ~1 min |
| Eval 2 (narrative from Eval 1 failures) | openai/gpt-oss-120b | ~500–1,000 | 8,000 | negligible |

Note: All three gpt-oss-120b calls are sequential. Eval 1 consistently exceeds
the 8,000 TPM ceiling by ~1,100 tokens — retry logic handles this. Eval 2 is
well within the ceiling. Total gpt-oss-120b tokens per pipeline run: ~10,000–10,500.

---

## Out of Scope for v1

- Games and software products
- Reply comment analysis
- Spam and bot filtering
- Multi-language analysis
- Progress indication / streaming UX during 5-minute extraction wait
- Results caching or storage between sessions
- Concurrent multi-user quota management
