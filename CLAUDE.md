# YouTube Comment Analyzer — Project Context

## What this system does
An LLM-based tool that takes a YouTube product promotion video URL, fetches and
filters English comments, extracts structured labels per comment using an extraction
model, aggregates results in code, evaluates output quality using a judge model,
and generates a PDF report with charts and an evaluation scorecard.

## Full specification
All architectural decisions, schemas, pipeline stages, error states, and output
specifications are in: docs/brief.md
Consult it when a task references brief details. Do not invent specifications
not present in it.

## Active phase
Phase 3 — Testing and Prompt Refinement
 
## Model strings (authoritative — never substitute)
Extraction:  meta-llama/llama-4-scout-17b-16e-instruct
Evaluation:  openai/gpt-oss-120b

## Tech stack (approved libraries only)
groq, google-api-python-client, langdetect, reportlab,
matplotlib, seaborn, fastapi, uvicorn, python-dotenv

## File structure (authoritative — never deviate)
youtube-comment-analyzer/
├── .env                          # API keys — never commit
├── .env.example                  # Key names only, no values
├── .gitignore
├── .python-version
├── CLAUDE.md
├── README.md
├── requirements.txt
├── docs/
│   └── brief.md
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI entry point
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── stage0_url_parser.py
│   │   ├── stage1_preprocessor.py
│   │   ├── stage2_comment_fetcher.py
│   │   ├── stage3_threshold.py
│   │   ├── stage4_extraction.py
│   │   ├── stage5_aggregation.py
│   │   ├── stage6_sampling.py
│   │   ├── stage7_evaluation.py
│   │   ├── stage8_confidence.py
│   │   └── stage9_pdf.py
│   └── utils/
│       ├── __init__.py
│       ├── groq_client.py        # Groq client, parsers, validators, constants
│       └── youtube_client.py     # YouTube API client, fetch cycle
├── tests/
│   ├── __init__.py
│   ├── local/                    # No API cost tests
│   └── live/                     # Real API call tests
└── output/                      # Generated PDFs

## Hard constraints (never violate)
- NEVER hardcode API keys or secrets anywhere in code
- NEVER create files outside the file structure above
- NEVER install libraries not in the approved tech stack
- NEVER proceed past a task if its done condition is not met
- NEVER combine two tasks into one step
- NEVER make design decisions — if something is unspecified, stop and ask
- NEVER use model strings other than those listed above
- NEVER write prompt content in Phase 1 — prompt strings are Phase 2
  deliverables. Phase 1 functions must accept prompt strings as parameters
- NEVER make live API calls except in the designated live test tasks
- NEVER exceed the Phase 1 live test budget: one YouTube metadata call,
  one YouTube comment fetch call, one Groq extraction model call,
  one Groq evaluation model call
- NEVER commit the .env file to version control

## Decisions log
(Append here after each session: what was decided and why.
Format: [Task N] decision made — one line.)

[Session 1 — Tasks 1–24] Pipeline file names use CLAUDE.md authoritative names (stage4_extraction.py etc.), not task-plan names.
[Session 1 — Tasks 1–24] Test files placed in tests/local/ subdirectory per CLAUDE.md structure; pytest discovers them recursively.
[Session 1 — Tasks 1–24] pytest installed as implicit dev dependency (not in approved tech stack but required by task plan).
[Session 3 — Task 45] First metadata call used video ID dMCjzFbHSW0 which does not exist (empty items response); second call used dQw4w9WgXcQ (Rick Astley) which is confirmed public with non-empty title/description/tags — this consumed two YouTube metadata quota units total.
[Session 3 — Tasks 45–46] Live test video dQw4w9WgXcQ used for both metadata and comment fetch; it is a music video not a product promotion video, but satisfies the API verification requirements (non-empty title, description, tags, and 100 comments with required fields).
[Session 1 — Pre-processing prompt] Pre-processing prompt corrected to recognize review videos as is_promotion: true per brief Stage 1 Step 3.
[Session 4 — Tasks 31–39] SFNS.ttf (SF Pro) used instead of Arial Unicode for SFNS font; Arial Unicode lacks ⚠ (U+26A0) — SFNS has both ⚠ and ✓ (U+2713).
[Session 4 — Tasks 31–39] stage9_pdf.py implements all PDF sections; generate_pdf function is the single entry point called from main.py after Stage 8.
[Session 5 — Tasks 40–44] All local tests pass (28 total); generate_pdf and fetch_and_filter_comments docstrings expanded to include params and return; active phase advanced to Phase 3.
[Phase 2.5 — Sessions 1–2] Five architectural changes: (1) Stage 6 gained select_random_sample alongside select_stratified_sample — seed=0 for reproducibility, returns all available comments if fewer than 30, no shared logic with stratified sampler; (2) evaluation.py split into eval1_label_quality.py (Eval 1 prompt) and eval2_output_quality.py (Eval 2 content moved); (3) stage7_evaluation.py split into evaluate_label_quality, compute_eval1_failure_rate, and evaluate_output_quality — combined evaluate() and run_evaluation() removed; (4) compute_confidence gained required eval1_failure_rate parameter with no default, with two new threshold checks at > 0.25 (force Low) and > 0.15 (cap High to Medium); (5) run_pipeline now calls Stage 6 twice (sample_a and sample_b) and Stage 7 twice (label quality then output quality), passing eval1_failure_rate to Stage 8.
[Phase 2.5 — Session 2] test_checkpoint1.py assertion fixed: result.get("comments") replaced with result.get("comments_fetched") > 0 since run_pipeline returns the count, not the list — user decision.
