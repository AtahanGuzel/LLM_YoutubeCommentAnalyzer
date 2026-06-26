# Project Brief: LLM-Based YouTube Product Promotion Video Comment Analyzer
**Version:** 3.0 — Revised  
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

### Stage 6 — Sample Selection (Two Independent Samples)

Before calling the gpt-oss-120b evaluator, two independent samples are selected in code
from the labeled comment dataset. Each sample serves a different evaluation call.
The two samples are fully independent — no deduplication is applied between them.
A comment may appear in both samples.

**Sample A — Random sample (for Eval 1 — label quality):**

30 comments selected uniformly at random from the full labeled comment dataset.
No stratification. The random sample gives an unbiased estimate of extraction
model failure rate across the actual comment distribution.

Random selection uses a fixed seed (seed=0) for reproducibility. Selection is without replacement — no comment appears twice within Sample A. If the total labeled comment set contains fewer than 30 comments, all available comments are taken.

Size rationale: at the gpt-oss-120b TPM ceiling of 8,000 tokens, with ~87 tokens
per comment (input + output) and ~400 tokens of prompt overhead, 30 comments
fits comfortably within the per-call token budget while remaining readable
in a single human review session.

**Sample B — Stratified sample (for Eval 2 — output quality):**

- 5 comments per sentiment bucket (positive, negative, neutral)
- 3 comments per identified pain point (up to 5 pain points)
- 3 comments per identified competitor
- 5 random comments (sanity check)

Typical total: ~35–50 comments. Deduplication applied within Sample B only
by comment_id — no comment appears twice inside this sample.
If a bucket has fewer comments than its target count, all available comments
are taken with no padding.

Sample B gives the judge coverage across every dimension it will score,
which is why stratification is appropriate here but not for Eval 1.

---

**Eval 1 output schema:**

```json
[
  {
    "comment_id": "001",
    "correct": true,
    "issue": null
  },
  {
    "comment_id": "012",
    "correct": false,
    "issue": "competitor Sony missed; comment says 'switched from Sony' but competitor_mentions is empty"
  }
]
```

**Eval 1 failure rate computation (code):**

Function name: `compute_eval1_failure_rate`

```python
def compute_eval1_failure_rate(verdicts: list[dict]) -> float:
    failed = sum(1 for v in verdicts if not v["correct"])
    return failed / len(verdicts)
```

**Eval 1 output routing:**
- Full verdict array (all 30 comments, correct and incorrect) → backend log at INFO level
- Failure rate float → passed to Stage 8 confidence computation via `compute_eval1_failure_rate`
- Failure patterns (issue strings grouped by type) → backend log
- Nothing from Eval 1 appears in the PDF

Engineering interpretation: failure rate above ~15% indicates the extraction
prompt needs revision. Failure patterns clustered on a specific label type
(all competitor misses, all sentiment ambiguities) indicate targeted prompt issues.

---

**Eval 2 Call — Output quality (feeds PDF)**

Function name: `evaluate_output_quality`

Uses Sample B (stratified sample). The judge receives the stratified sample
with raw text and assigned labels, followed by the final aggregated output,
and scores three criteria on a 1–5 scale.

Raw comments are presented before the Scout 17B output to mitigate position bias.

**Eval 2 prompt structure:**

You are evaluating the output quality of a comment analysis model.

Present evidence before stating any score. Do not state scores until

you have completed the evidence review for each criterion.
TASK: Analyze YouTube comments about {primary_product} for sentiment,

pain points, and competitor mentions.
SAMPLED COMMENTS WITH SCOUT 17B LABELS:

[comment_id, text, assigned label for each sample]
SCOUT 17B FINAL AGGREGATED OUTPUT:

[sentiment distribution, pain points list, competitors list]
Output JSON only in the structure below.

**Eval 2 output schema:**

```json
{
  "output_quality": {
    "sentiment": {
      "evidence": "string — specific observations from samples",
      "gaps": "string — contradictions or missed patterns, or null",
      "score": 1
    },
    "pain_points": {
      "evidence": "string",
      "gaps": "string or null",
      "score": 1
    },
    "competitors": {
      "evidence": "string",
      "gaps": "string or null",
      "score": 1
    }
  }
}
```

**Anchored rubric provided to the judge:**

*Criterion 1 — Sentiment classification accuracy:*
5 → Assigned sentiment is clearly supported by the majority of sampled comments.

No meaningful contradicting evidence present.

4 → Well supported. A small number of samples suggest mild ambiguity but

don't contradict the overall classification.

3 → Partially supported. Roughly equal evidence exists for an alternative

classification.

2 → Weakly supported. Stronger evidence in samples points toward a different

classification.

1 → Directly contradicts the majority of sampled comments.


*Criterion 2 — Pain point identification accuracy:*
5 → All identified pain points are directly evidenced in samples.

No recurring problem pattern in samples was missed.

4 → Well evidenced. One minor issue is slightly mischaracterized or a

low-signal pattern was missed.

3 → Most pain points valid but one is not well supported, or one clearly

recurring issue is absent.

2 → Multiple pain points weakly supported or a high-frequency problem

visible in samples does not appear in output.

1 → Identified pain points do not reflect sampled comment content.

*Criterion 3 — Competitor identification accuracy:*
5 → All competitors in sampled comments correctly captured.

Threshold rule correctly applied.

4 → Correctly identified with minor characterization variance.

No missed competitor from samples.

3 → One competitor visible in samples absent from output, or one included

competitor weakly evidenced.

2 → Multiple competitors missed or incorrectly included relative to samples.

1 → Competitor output does not reflect sampled comment content.

---


### Stage 8 — Confidence Computation (Code)

Computed from Eval 2 scores, Eval 1 failure rate, and data quality signals.
Check order is fixed — earlier checks short-circuit the function.

```python
def compute_confidence(
    scores: list[int],
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

    if eval1_failure_rate > 0.25:
        return "Low", "Extraction quality is low. \
                        Manual review of comment labels is recommended."

    avg = sum(scores) / len(scores)

    if avg >= 4.0:
        if eval1_failure_rate > 0.15:
            return "Medium", "Extraction quality is moderate. \
                               Results should be treated as directional."
        return "High", None
    elif avg >= 2.5:
        return "Medium", "Results should be treated as directional."
    else:
        return "Low", "Manual review of comments is recommended."
```

**Eval 1 failure rate thresholds:**
```
failure_rate > 0.25  → confidence forced to Low regardless of Eval 2 scores
failure_rate > 0.15  → confidence capped at Medium (High becomes Medium)
failure_rate ≤ 0.15  → no cap applied, Eval 2 scores drive confidence normally
```

Additionally, any single Eval 2 criterion scoring 1 triggers a per-section warning
in the PDF regardless of overall average.

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
Sentiment     [✓ / ⚠]  {evidence sentence from judge}
Pain points   [✓ / ⚠]  {evidence sentence from judge}
Competitors   [✓ / ⚠]  {evidence sentence from judge}

✓ = score 4–5    ⚠ = score 1–3
```

Raw numeric scores are not shown in the PDF. They are logged to the backend only.
Eval 1 results (verdict array, failure rate, failure patterns) are never shown
in the PDF. They are logged to the backend only.

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
Two independent sample selections (code)
  ├─ Sample A: 30 random comments → Eval 1
  └─ Sample B: stratified sample → Eval 2
        ↓
gpt-oss-120b Eval 1 call (Sample A)
  └─ verdict per comment (correct/false + issue) → backend log
  └─ failure rate float → Stage 8
        ↓
gpt-oss-120b Eval 2 call (Sample B)
  └─ output quality scores + evidence → Stage 8 + PDF scorecard
        ↓
Confidence computation (code)
  ├─ data quality signals (loss ratio, relevant count)
  ├─ Eval 1 failure rate (caps confidence at Medium or Low)
  └─ Eval 2 scores (drive tier when no cap applies)
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
| Eval 1 (30 random comments) | openai/gpt-oss-120b | ~3,000 | 8,000 | negligible |
| Eval 2 (stratified sample) | openai/gpt-oss-120b | ~3,000–3,700 | 8,000 | negligible |

Note: All three gpt-oss-120b calls are sequential. Each individual call fits within
the 8,000 TPM ceiling. Total gpt-oss-120b tokens per pipeline run: ~6,400–7,100.

---

## Out of Scope for v1

- Games and software products
- Reply comment analysis
- Spam and bot filtering
- Multi-language analysis
- Progress indication / streaming UX during 5-minute extraction wait
- Results caching or storage between sessions
- Concurrent multi-user quota management
