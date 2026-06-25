# Product promotion video used: "iPhone 17 Review: No Asterisks!" (MKBHD)
# URL: https://www.youtube.com/watch?v=rng_yUSwrgU
# Live call: 1 YouTube metadata unit (stage 1) + 1 comment page unit (stage 2)
# + Groq extraction calls (stage 4) + 1 Groq evaluation call (stage 7)
# Phase 2 running quota: 39 units consumed before this test; this test adds 2 YouTube units.

import asyncio
import os
import re
from pathlib import Path

from app.main import run_pipeline

PRODUCT_PROMO_URL = "https://www.youtube.com/watch?v=rng_yUSwrgU"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
PDF_PATTERN = re.compile(r"^report_[A-Za-z0-9_\-]+_\d{8}_\d{6}\.pdf$")


def test_checkpoint4_pdf_generation():
    """Smoke test: full pipeline produces a non-empty PDF in output/ with correct filename."""
    result = asyncio.run(run_pipeline(PRODUCT_PROMO_URL))

    assert result.get("outcome") == "proceed", (
        f"Expected proceed outcome, got: {result.get('outcome')} — {result.get('message')}"
    )

    pdf_path_str = result.get("pdf_path")
    assert pdf_path_str is not None, "pdf_path missing from run_pipeline result"

    pdf_path = Path(pdf_path_str)

    # File must exist and be non-empty
    assert pdf_path.exists(), f"PDF file not found: {pdf_path}"
    assert pdf_path.stat().st_size > 0, f"PDF file is empty: {pdf_path}"

    # Filename must match report_{video_id}_{timestamp}.pdf pattern
    assert PDF_PATTERN.match(pdf_path.name), (
        f"PDF filename does not match expected pattern: {pdf_path.name}"
    )

    # File must be inside the output/ directory
    assert pdf_path.parent.resolve() == OUTPUT_DIR.resolve(), (
        f"PDF not in expected output dir: {pdf_path.parent}"
    )

    print(f"\nPDF generated: {pdf_path.name} ({pdf_path.stat().st_size} bytes)")
