# Architectural Decisions

## Format
Each entry: Decision, Why, Alternatives considered, Source

Entries are ordered by when they were made — brief clarification phase first,
then hole resolution phase, then execution phase.

---

## BRIEF CLARIFICATION PHASE

---

### D01 — Chunked batching for comment extraction

**Decision:** Send comments to 8B in chunks of 25 rather than a single prompt
or one-comment-at-a-time.

**Why:** LLMs cannot reliably count. A single prompt asking "what are the top 5
most frequently mentioned problems?" produces plausible-sounding results with no
verifiable relationship to actual occurrence counts. Code must do the counting.
Per-comment processing is too API-call-intensive against Groq's rate limits.
Chunked batching keeps counting in code while staying within quota.

**Calculation:** 200 comments ÷ 25 per chunk = 8 calls. 8 × ~3,500 tokens =
28,000 tokens. At 30K TPM (Scout 17B) this completes in ~1 minute.

**Alternatives considered:**
- Single prompt: LLM hallucinated frequency counts, rejected
- Per-comment (1 call each): 200 RPD cost per analysis, rejected
- Larger chunks: token overflow risk at tail of comment length distribution

---

### D02 — Stratified sampling for 70B evaluation

**Decision:** Select a stratified sample of ~35 comments before the evaluation
call rather than sending all raw comments.

**Why:** Sending all 200 comments to 70B exceeds the 6,000–8,000 TPM ceiling
in a single call. Stratified sampling gives the judge targeted evidence — it
verifies specific labeling decisions against source text rather than re-deriving
everything from scratch. This makes the evaluation more meaningful, not less.

**Sample composition:**
- 5 per sentiment bucket (positive, negative, neutral)
- 3 per identified pain point (up to 5)
- 3 per identified competitor
- 5 random (sanity check)

**Alternatives considered:**
- Send all raw comments: exceeds TPM ceiling, rejected
- Gold-labeled test set: overkill for runtime evaluation, appropriate for
  development benchmarking only, rejected for runtime use

---

### D03 — Two-scorecard evaluation architecture

**Decision:** One 70B call performs two evaluations: Eval 1 (label quality,
pass/fail per sampled comment) and Eval 2 (output quality, 1–5 per criterion).

**Why:** Label quality and output quality are two different failure modes with
different diagnostic value. Label quality tells you if the extraction prompt is
working. Output quality tells you if the final report is trustworthy. Separating
them gives observability at two pipeline layers in one call. Eval 1 stays in the
backend log. Eval 2 feeds the PDF.

**Alternatives considered:**
- Single scorecard on final output only: can't distinguish extraction failure
  from aggregation failure, rejected
- Separate 70B calls for each eval: doubles quota cost unnecessarily, rejected

---

### D04 — Pre-processing uses 70B, not 8B

**Decision:** Promotional video classification and primary product extraction
use openai/gpt-oss-120b, not the extraction model.

**Why:** One small call (~300 tokens) with negligible quota impact. The
classification accuracy benefit justifies using the stronger model. Keeps 8B
exclusively responsible for extraction work — clean separation of roles.
Also, classification runs before any comment is fetched so the cost is trivial.

**Alternatives considered:**
- 8B for pre-processing: equally capable for binary classification but misses
  the clean role separation. Rejected in favour of stronger model at no
  meaningful cost.
- Code-level keyword detection (title patterns, disclosure signals): too many
  edge cases. "Elden Ring review" vs product review, "vs" in unrelated titles,
  sponsored gameplay videos. LLM handles all cases in one shot. Rejected.

---

### D05 — Pre-processing extended to extract primary product

**Decision:** The pre-processing call returns both is_promotion and
primary_product in one call, not two separate calls.

**Why:** The model already reads title + description + tags to classify
promotional intent. It has everything needed to identify the product in the same
read. Zero extra input tokens, one call instead of two.

**Output schema:**
```json
{
  "is_promotion": true | false,
  "reasoning": "one sentence",
  "primary_product": "exact product name or null",
  "product_confidence": "high" | "low",
  "candidates": ["product1", "product2"] | null
}
```

**Alternatives considered:**
- Separate classification and product extraction calls: doubled API cost
  for no benefit, rejected

---

### D06 — Product confidence has two values, not three

**Decision:** product_confidence is "high" | "low" only. Medium removed.

**Why:** Medium collapsed into the same behavior as low (ask user to confirm).
Three values with two behaviors is unnecessary complexity in the schema.
Every field should drive a distinct action.

**Alternatives considered:**
- Medium → proceed automatically: asymmetric risk. A false high confidence
  on the wrong product corrupts the entire pipeline silently. Rejected.
- Medium → show confirmation: same behavior as low. Redundant value. Removed.

---

### D07 — Promotion classification confidence field removed

**Decision:** No confidence field on is_promotion. Only is_promotion true/false
drives that gate.

**Why:** The confidence value was never wired into any decision. It was schema
noise. Removed to keep every field action-driving.

**Alternatives considered:**
- Low is_promotion confidence → ask user: adds friction for a gate that's
  either pass or fail. Rejected.

---

### D08 — Comment fetch-filter cycle

**Decision:** Fetch 100 comments → langdetect filter → if English count < 100
and pages remain, fetch next 100. Hard cap at 500 total fetched.

**Why:** A fixed 200-comment cap silently shrinks the dataset when non-English
comments are present. A video with 300 total comments where 160 are non-English
would yield only 80 English from a 200-comment fetch. The cycle guarantees
the target English count is reached or comments are genuinely exhausted.

**Caps:**
- Target: 100 English comments
- Hard cap: 500 total fetched (5 YouTube API quota units per analysis)

**Alternatives considered:**
- Fixed cap at 200: silently shrinks dataset, rejected
- Most-liked sort: YouTube API doesn't support it natively — requires
  fetch-then-sort which pays full quota cost anyway. Rejected.
- relevanceLanguage parameter: soft bias, not a hard filter. Rejected.

**Chosen sort:** orderBy=relevance (YouTube's engagement-weighted sort)

---

### D09 — Threshold applied to English comment count only

**Decision:** The 50/100 threshold gates are evaluated against English-filtered
comment count, not total fetched count.

**Why:** The threshold is a quality guarantee. Applying it to total comments
including non-English ones would allow a video with 80 total (30 English) to
pass the >50 threshold, producing analysis from 30 comments. The threshold
must reflect the comments actually available for analysis.

---

### D10 — Irrelevant comments handled by LLM with two distinct labels

**Decision:** Two labels — competitor_focus and noise — rather than one
irrelevant label or code-level filtering.

**Why:** These are semantically distinct categories with different downstream
routing. competitor_focus comments feed competitor analysis. noise comments
are excluded from everything. Conflating them into one label loses information.
Code-level filtering (minimum token count, keyword patterns) creates its own
edge cases and is the wrong tool for semantic judgment.

**Label routing:**
- target → sentiment + pain points + competitor mentions
- competitor_focus → competitor mentions only
- noise → excluded from all aggregation

**Alternatives considered:**
- Single irrelevant label: loses the competitor signal in competitor_focus
  comments, rejected
- Code-level noise filter (min token count): "lol" passes at 1 token,
  "first comment" passes at 2, still semantic judgment required. Rejected.
- totalReplyCount as engagement signal: replies reflect disagreement as
  often as agreement. Ambiguous signal, rejected.

---

### D11 — Logarithmic weighting for pain point ranking

**Decision:** Pain point weighted score = mention_count + log(total_likes + 1)

**Why:** Frequency drives ranking, likes provide a soft boost. Raw like count
as a multiplier would let one viral comment dominate. Log scaling prevents this
while still reflecting that a highly liked comment represents broader agreement
than its author alone.

**Alternatives considered:**
- Raw comment count only: ignores like signal, loses information
- Raw like count multiplier: single viral comment dominates, rejected
- Reply count weighting: ambiguous (agreement vs disagreement), rejected

---

### D12 — Competitor threshold uses raw comment count, not weighted score

**Decision:** The ≥10 comment rule for competitors uses raw comment count,
no logarithmic weighting.

**Why:** The competitor threshold is a hard business rule defined in the brief.
Applying like weighting would make it fuzzy. A brand manager needs to be able
to say "this competitor appears because it was mentioned in at least 10 comments"
— a clear, auditable criterion. Weighted scores don't support this.

---

### D13 — Chunk failure handling — retry first, threshold after

**Decision:** 3 retries with exponential backoff per failed chunk. After all
retries exhausted, return None for that chunk. Re-apply 50/100 threshold to
successfully processed comment count.

**Why:** Most Groq failures are transient (429 rate limit, 500 server error,
timeout). Retry handles 95%+ of real failures before they become data loss.
After genuine failure, proceeding if >50 comments were processed preserves
completed work. Hard failing the entire analysis because 1 of 8 chunks failed
is disproportionate.

**Retry backoff:** 1s, 2s, 4s (2^attempt)

**Post-retry threshold:**
- processed > 100 → proceed, full mode
- processed 51–100 → proceed, degraded mode
- processed ≤ 50 → fail with error

---

### D14 — Malformed LLM response handling — three differentiated handlers

**Decision:** Three separate exception behaviors by call type, not one generic
handler.

**Why:** Call types have different failure consequences. Pre-processing failure
blocks everything — raise immediately, surface to user. Extraction chunk failure
is partial — retry first, then return None (feeds existing chunk failure logic).
Evaluation failure comes after all work is done — return None, generate PDF
without scorecard, confidence forced to Low.

**Handlers:**
- PreProcessingResponseError: raised immediately
- Extraction chunk: return None after retries (retriable failure path)
- Evaluation: return None, PDF renders fallback text

**Alternatives considered:**
- Single generic exception handler: loses the differentiation between
  pipeline-blocking and non-blocking failures, rejected

---

### D15 — Confidence tier computed from three inputs

**Decision:** Confidence computation takes: Eval 2 scores, relevant_count,
and (processed_count / total_fetched) loss ratio.

**Why:** A single eval score doesn't capture all quality signals. A video
with a high eval score but only 15 relevant comments is not High confidence.
A video with good relevance but 30% chunk data loss is not High confidence.
All three signals are needed for an honest quality tier.

**Thresholds:**
- loss_ratio > 0.25 → Low
- relevant_count < 30 → Low
- avg score ≥ 4.0 → High
- avg score 2.5–3.9 → Medium
- avg score < 2.5 → Low

---

### D16 — PDF always generates, never blocked by low eval score

**Decision:** PDF generation proceeds regardless of evaluation score or
confidence tier. Low confidence surfaces as a red banner, not a blocked export.

**Why:** The user waited ~1 minute for extraction. Blocking the PDF after all
that work because the evaluator returned a low score is bad UX and wastes the
analysis. The confidence tier is a quality signal, not a gate. Brand managers
decide whether to act on a low-confidence report, not the system.

---

### D17 — Sentiment distribution excludes competitor_focus and noise

**Decision:** Sentiment donut chart is built from target-labeled comments only.

**Why:** Sentiment only has meaning relative to the target product. A positive
comment about a competitor is not a positive signal for the target product.
Including it would inflate positive sentiment incorrectly.

**PDF header label:** "N relevant comments" (not "N comments analyzed") when
competitor_focus or noise comments are present.

---

### D18 — Evaluation rubric uses chain-of-thought forcing

**Decision:** Judge must output evidence and gaps before stating a score.
Score cannot be stated until evidence review is complete.

**Why:** If the judge outputs a score directly, it pattern-matches to a number
and rationalizes backward. Forcing evidence-first commits the model to
observations before the verdict, producing scores grounded in sample content
rather than prior agreement bias.

**Position bias mitigation:** Raw sampled comments presented before the 8B
output in the evaluation prompt. The judge anchors to what it sees first —
raw evidence should come first.

---

### D19 — Three known LLM judge biases documented

**Decision:** Prompt design actively counters three biases.

**Position bias:** raw evidence before model output in prompt
**Verbosity bias:** structured JSON output schema prevents reward for length
**Self-enhancement bias:** anchored rubric forces evaluation against sample
content, not against judge's own analytical preferences

---

### D20 — Replies ignored for v1

**Decision:** Only top-level comments are fetched and analyzed. Replies excluded.

**Why:** Capturing reply context (threading) is complex for an 8B model.
Reply count as an engagement signal is ambiguous — replies reflect disagreement
as often as agreement. The cost/benefit for v1 doesn't justify the complexity.

**Deferred:** Reply analysis is a v2 consideration.

---

### D21 — Scope limited to physical products for v1

**Decision:** The pre-processing classifier is instructed to accept physical
products only. Games and software are deferred.

**Why:** The boundary of what counts as a "product promotion video" needed to
be locked for consistent classifier behavior. Physical products are the primary
use case for the target audience (brand managers). Games and software introduce
ambiguous cases that would require separate handling.

---

### D22 — Spam and bot filtering ignored for v1

**Decision:** No pre-filtering of spam, bot, or duplicate comments in v1.

**Why:** Spam detection adds significant complexity. For v1 at prototype scale
with relevance-sorted comments, the top-engaged comments are unlikely to be
predominantly spam. The confidence tier surfaces quality concerns implicitly.

---

### D23 — langdetect with deterministic seed

**Decision:** Use langdetect library with DetectorFactory.seed = 0.

**Why:** langdetect is probabilistic by default — the same text can return
different language codes between runs. Seed=0 makes detection fully
deterministic, ensuring two identical analysis runs produce identical
filtered comment sets.

**Failure handling:** Short comments (<4 words) and emoji-only comments that
fail detection are excluded. These carry no analytical value regardless.

**Alternatives considered:**
- lingua-py: more accurate on short texts, deterministic by default, but
  heavier dependency and requires specifying candidate languages upfront.
  Tradeoff not worth it for v1.
- YouTube API relevanceLanguage parameter: soft bias, not a hard filter.
  Non-English comments still appear. Rejected.
- LLM-based detection: overkill by every measure. Rejected.

---

### D24 — PDF layout order

**Decision:** Header → Sentiment → Pain Points → Competitors → Eval Scorecard
→ Footer

**Why:** Brand managers read reports top to bottom. Insights should come before
the quality signal. A brand manager reading page one should see findings first,
not a model score they have to interpret before trusting the rest of the report.
The eval scorecard at the bottom is for analysts who want to dig in.

---

### D25 — PDF chart types

**Decision:**
- Sentiment: donut chart (center label shows N relevant comments)
- Pain points: horizontal bar chart (natural language labels need horizontal space)
- Competitors: horizontal bar chart with reference line at x=10

**Why:** Chart type follows data shape, not style preference. Three mutually
exclusive proportions = donut. Ranked items with text labels = horizontal bar.
Competitor bars include a threshold reference line so the minimum bar is
contextually explained without requiring a footnote.

---

## MODEL SELECTION PHASE

---

### D26 — Extraction model: llama-4-scout-17b-16e-instruct

**Decision:** Use meta-llama/llama-4-scout-17b-16e-instruct for comment
extraction instead of llama-3.1-8b-instant.

**Why:** Scout's 30K TPM reduces the ~28K token extraction job from ~5 minutes
(at 6K TPM on 8B) to ~1 minute. The brief explicitly flagged the 5-minute wait
as a deferred UX problem — Scout solves it. Scout is also a 17B model, more
accurate than 8B for nuanced semantic tasks like comment_type classification.
Not deprecated.

**RPD tradeoff:** Scout has 1K RPD vs 8B's 14.4K RPD. During development this
means ~250 full pipeline runs per day vs ~3,600. Accepted as a development
discipline constraint given the TPM and accuracy benefits.

**Alternatives considered:**
- llama-3.1-8b-instant: best RPD, but deprecated and 5-minute wait. Rejected.
- openai/gpt-oss-20b: better TPM than 8B but 1K RPD same as Scout, no
  meaningful advantage over Scout for this task. Rejected.

---

### D27 — Evaluation and pre-processing model: openai/gpt-oss-120b

**Decision:** Use openai/gpt-oss-120b for both pre-processing classification
and evaluation.

**Why:** Strongest model available on Groq, not deprecated. TPM drops from
12K (llama-3.3-70b) to 8K but the evaluation call is ~4,700 tokens — well
within 8K. Native structured outputs support makes JSON schema compliance
more reliable. Strictly better than 70B with no meaningful limit downside.

**Alternatives considered:**
- llama-3.3-70b-versatile: deprecated. Rejected.
- qwen/qwen3.6-27b: reasoning model, comparable limits to gpt-oss-120b,
  but gpt-oss-120b is stronger for this specific structured output task.
  Rejected.

---

## EXECUTION PHASE (PHASE 1)

---

### D28 — pytest added as dev dependency

**Decision:** pytest installed and added to requirements.txt as an implicit
dev dependency despite not being in the original tech stack.

**Why:** The dev plan explicitly required python -m pytest for done condition
verification. It's a dev dependency only — never ships to production. No impact
on pipeline code.

---

### D29 — CLAUDE.md file naming is authoritative over dev plan

**Decision:** Pipeline stage file names follow CLAUDE.md, not the dev plan,
where they conflicted.

**Correct names:**
- stage4_extraction.py (not stage4_extractor.py)
- stage5_aggregation.py (not stage5_aggregator.py)
- stage6_sampling.py (not stage6_sampler.py)
- stage7_evaluation.py (not stage7_evaluator.py)
- stage9_pdf.py (not stage9_pdf_generator.py)

**Why:** CLAUDE.md is explicitly marked "authoritative — never deviate."
Dev plan had a naming inconsistency. CLAUDE.md wins.

---

### D30 — Test files in tests/local/ and tests/live/ subdirectories

**Decision:** No-API-cost tests in tests/local/, real API call tests in
tests/live/. Not in tests/ root directly.

**Why:** Separating by API cost allows running the safe test suite without
risking quota consumption during development. pytest discovers both
subdirectories recursively — done conditions still satisfied.

---

### D31 — Tech stack

**Final approved stack:**
```
groq                       LLM API client
google-api-python-client   YouTube Data API v3
langdetect                 Language detection
reportlab                  PDF generation
matplotlib                 Chart rendering
seaborn                    Chart aesthetics (on top of matplotlib)
fastapi                    API backend (async-native)
uvicorn                    ASGI server
pytest                     Dev dependency — test runner
```

**Why FastAPI over Flask/Django:** Async-native is non-negotiable. The pipeline
has concurrent async operations (parallel chunk calls, YouTube pagination cycling,
Groq API calls). FastAPI is built exactly for this. Industry standard for AI
API services.

**Why ReportLab over WeasyPrint/pdfkit:** Chart-to-PDF integration path is
cleaner — charts generate into BytesIO buffers, ReportLab embeds directly.
WeasyPrint requires temp files or base64-encoded HTML. No external binary
dependencies unlike pdfkit.

**Why Matplotlib + Seaborn over Plotly:** Static output natively (no kaleido
dependency). Plotly's interactivity is wasted in PDF output. Seaborn provides
better visual defaults on top of Matplotlib's backend.
