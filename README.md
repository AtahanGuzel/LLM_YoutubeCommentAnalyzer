# YouTube Comment Analyzer

An LLM-based tool that takes a YouTube product promotion video URL, fetches and filters English comments, extracts structured labels per comment, aggregates results, evaluates output quality with a judge model, and generates a PDF report.

## Prerequisites

- Python 3.14.2 (managed via `.python-version`)
- YouTube Data API v3 key
- Groq API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```
YOUTUBE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

## How to run

**Required environment variables** — set these in `.env` before starting:

```
GROQ_API_KEY=your_groq_api_key
YOUTUBE_API_KEY=your_youtube_data_api_v3_key
```

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Then send an analysis request:

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

The endpoint returns a PDF file. Reports are also saved to the `output/` directory
with timestamped filenames in the format:

```
output/report_{video_id}_{YYYYMMDD_HHMMSS}.pdf
```

For example: `output/report_rng_yUSwrgU_20260624_190619.pdf`

## Running tests

```bash
python -m pytest tests/local/
```

## Project structure

```
app/
  main.py                   FastAPI entry point
  pipeline/
    stage0_url_parser.py    Extract video ID from URL
    stage1_preprocessor.py  Metadata classification (Phase 2)
    stage2_comment_fetcher.py  Fetch and filter English comments
    stage3_threshold.py     Enforce minimum comment count
    stage4_extraction.py    Chunked LLM label extraction (Phase 2)
    stage5_aggregation.py   Code aggregation of labels (Phase 2)
    stage6_sampling.py      Stratified sample selection (Phase 2)
    stage7_evaluation.py    Judge model evaluation (Phase 2)
    stage8_confidence.py    Confidence computation (Phase 2)
    stage9_pdf.py           PDF report generation (Phase 2)
  utils/
    groq_client.py          Groq client, model constants, parsers, validators
    youtube_client.py       YouTube API client and fetch cycle
tests/
  local/                    No-API-cost unit tests
  live/                     Real API call tests
output/                     Generated PDFs
```

## Models

| Role | Model |
|---|---|
| Extraction | `meta-llama/llama-4-scout-17b-16e-instruct` |
| Evaluation / Pre-processing | `openai/gpt-oss-120b` |

## Current status

Phase 2 complete. The full pipeline runs end-to-end from URL input to PDF output. All nine pipeline stages are implemented and wired together. Phase 3 covers testing and prompt refinement.
