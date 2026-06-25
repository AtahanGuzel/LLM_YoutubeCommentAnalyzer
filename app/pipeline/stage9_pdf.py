"""Stage 9 — PDF Report Generation.

Generates a PDF report using ReportLab and Matplotlib/Seaborn. Includes a header,
sentiment donut chart, pain points bar chart, competitor bar chart, evaluation
scorecard, and footer. Output is written to the output/ directory.
"""

from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image,
)

from app.utils.groq_client import EXTRACTION_MODEL, EVALUATION_MODEL

# SFNS (SF Pro) supports ⚠ (U+26A0) and ✓ (U+2713) needed in this report.
# Arial Unicode does not include ⚠; SFNS covers both special characters.
_SFNS_FONT = "/System/Library/Fonts/SFNS.ttf"
_BOLD_FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
pdfmetrics.registerFont(TTFont("SFNS", _SFNS_FONT))
pdfmetrics.registerFont(TTFont("ArialBold", _BOLD_FONT))

_PAGE_W, _PAGE_H = A4
_MARGIN = 2.5 * cm
_CONTENT_W = _PAGE_W - 2 * _MARGIN
_LABEL_COL = 5.8 * cm
_VALUE_COL = _CONTENT_W - _LABEL_COL


def generate_pdf(
    output_path: str,
    video_title: str,
    video_url: str,
    analysis_timestamp: datetime,
    comments_fetched: int,
    processed_count: int,
    relevant_count: int,
    quality_mode: str,
    primary_product: str,
    sentiment_distribution: dict,
    pain_points: list[dict],
    competitors: list[dict],
    confidence_tier: str,
    confidence_warning: str | None,
    output_quality: dict,
) -> None:
    """Generate the full PDF analysis report and write it to output_path.

    Builds a ReportLab document containing: header table, optional confidence
    banner, sentiment donut chart, pain points bar chart, competitor bar chart,
    evaluation scorecard, and footer. Writes the finished PDF to output_path.

    Args:
        output_path: Filesystem path for the output PDF file.
        video_title: YouTube video title displayed in the header.
        video_url: Full YouTube video URL displayed in the header.
        analysis_timestamp: Datetime of this analysis run (UTC).
        comments_fetched: Total number of English comments retrieved.
        processed_count: Comments successfully processed by the extraction model.
        relevant_count: Target-type comments relevant to the primary product.
        quality_mode: 'full' or 'degraded' — shown capitalized in the header.
        primary_product: Confirmed product name used in the 'Relevant to' header row.
        sentiment_distribution: Dict with positive/negative/neutral counts and percentages.
        pain_points: List of pain point dicts (pain_point, mention_count, weighted_score).
        competitors: List of competitor dicts (competitor, mention_count).
        confidence_tier: 'High', 'Medium', or 'Low' — drives banner and scorecard.
        confidence_warning: Warning message for Medium/Low tiers, or None for High.
        output_quality: Dict of criterion dicts (evidence, gaps, score) keyed by
                        'sentiment', 'pain_points', 'competitors'.

    Returns:
        None. The PDF is written to output_path as a side effect.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = _make_styles()
    story = []

    story.append(Paragraph("YouTube Comment Analysis Report", styles["report_title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#333333")))
    story.append(Spacer(1, 0.4 * cm))

    story.extend(_build_header(
        styles, video_title, video_url, analysis_timestamp,
        comments_fetched, processed_count, relevant_count,
        quality_mode, primary_product,
    ))

    story.extend(_build_confidence_banner(styles, confidence_tier))

    story.append(Paragraph("Sentiment Analysis", styles["section_heading"]))
    story.append(_build_chart1_sentiment(sentiment_distribution))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Pain Points", styles["section_heading"]))
    story.extend(_build_chart2_pain_points(styles, pain_points))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Competitor Mentions", styles["section_heading"]))
    story.extend(_build_chart3_competitors(styles, competitors))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Evaluation Scorecard", styles["section_heading"]))
    story.extend(_build_scorecard(styles, confidence_tier, output_quality))
    story.append(Spacer(1, 0.8 * cm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#AAAAAA")))
    story.append(Spacer(1, 0.3 * cm))
    story.extend(_build_footer(styles, analysis_timestamp))

    doc.build(story)


def _make_styles() -> dict:
    """Return a dict of named ParagraphStyle instances used throughout the PDF."""
    base = getSampleStyleSheet()
    return {
        "report_title": ParagraphStyle(
            "report_title",
            parent=base["Title"],
            fontName="ArialBold",
            fontSize=18,
            spaceAfter=4,
        ),
        "header_label": ParagraphStyle(
            "header_label",
            parent=base["Normal"],
            fontName="ArialBold",
            fontSize=10,
        ),
        "header_value": ParagraphStyle(
            "header_value",
            parent=base["Normal"],
            fontName="SFNS",
            fontSize=10,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            parent=base["Heading2"],
            fontName="ArialBold",
            fontSize=13,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="SFNS",
            fontSize=10,
        ),
        "banner_medium": ParagraphStyle(
            "banner_medium",
            parent=base["Normal"],
            fontName="SFNS",
            fontSize=10,
            textColor=colors.HexColor("#7A5C00"),
        ),
        "banner_low": ParagraphStyle(
            "banner_low",
            parent=base["Normal"],
            fontName="SFNS",
            fontSize=10,
            textColor=colors.HexColor("#7A1A00"),
        ),
        "scorecard_tier": ParagraphStyle(
            "scorecard_tier",
            parent=base["Normal"],
            fontName="ArialBold",
            fontSize=15,
            spaceAfter=6,
        ),
        "scorecard_row": ParagraphStyle(
            "scorecard_row",
            parent=base["Normal"],
            fontName="SFNS",
            fontSize=10,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontName="SFNS",
            fontSize=8,
            textColor=colors.HexColor("#555555"),
        ),
    }


_CHART_W = _CONTENT_W * 0.72 / 28.35  # points → inches for matplotlib
_CHART_H = _CHART_W * 0.6


def _fig_to_image(fig, width_pts: float) -> Image:
    """Convert a Matplotlib figure to a ReportLab Image at the given point width."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth = width_pts
    img.drawHeight = width_pts * aspect
    return img


def _build_chart1_sentiment(sentiment_distribution: dict) -> Image:
    """Return a donut chart Image for the sentiment distribution.

    Segments: Positive (green) / Neutral (gray) / Negative (red).
    Center label: '{n} relevant comments'.
    Legend: each segment with its percentage.
    """
    positive = sentiment_distribution.get("positive", 0)
    neutral = sentiment_distribution.get("neutral", 0)
    negative = sentiment_distribution.get("negative", 0)
    total = positive + neutral + negative

    pos_pct = sentiment_distribution.get("positive_pct", 0)
    neu_pct = sentiment_distribution.get("neutral_pct", 0)
    neg_pct = sentiment_distribution.get("negative_pct", 0)

    sizes = [positive, neutral, negative]
    labels = [
        f"Positive ({pos_pct:.1f}%)",
        f"Neutral ({neu_pct:.1f}%)",
        f"Negative ({neg_pct:.1f}%)",
    ]
    chart_colors = ["#4CAF50", "#9E9E9E", "#F44336"]

    fig, ax = plt.subplots(figsize=(_CHART_W, _CHART_H))

    wedges, _ = ax.pie(
        sizes,
        colors=chart_colors,
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.5},
        counterclock=False,
    )

    ax.text(
        0, 0, f"{total}\nrelevant\ncomments",
        ha="center", va="center",
        fontsize=9, fontweight="bold", color="#333333",
        multialignment="center",
    )

    ax.legend(
        wedges,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=False,
    )

    ax.set_aspect("equal")
    return _fig_to_image(fig, _CONTENT_W)


def _build_chart2_pain_points(styles: dict, pain_points: list[dict]) -> list:
    """Return pain points chart or fallback text as a list of flowables.

    Horizontal bar chart: pain point labels on Y axis (ranked 1–5 by descending
    weighted score), weighted score on X axis, secondary 'mentioned in N comments'
    label on each bar.
    Falls back to brief-specified text when pain_points is empty.
    """
    if not pain_points:
        return [Paragraph(
            "No pain points identified in the analyzed comments.",
            styles["body"],
        )]

    labels = [pp["pain_point"] for pp in pain_points]
    scores = [pp["weighted_score"] for pp in pain_points]
    mention_counts = [pp["mention_count"] for pp in pain_points]

    # Reverse so highest score is at top
    labels = labels[::-1]
    scores = scores[::-1]
    mention_counts = mention_counts[::-1]

    n = len(labels)
    bar_height = 0.55
    fig_h = max(1.8, n * 0.6 + 0.8)
    fig, ax = plt.subplots(figsize=(_CHART_W * 1.1, fig_h))

    bars = ax.barh(range(n), scores, height=bar_height, color="#5C85D6")

    # Secondary label on each bar
    for i, (bar, mc) in enumerate(zip(bars, mention_counts)):
        ax.text(
            bar.get_width() + 0.05 * max(scores),
            bar.get_y() + bar.get_height() / 2,
            f"mentioned in {mc} comments",
            va="center", ha="left", fontsize=8, color="#555555",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Weighted score", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    # Extend x limit to fit secondary labels
    ax.set_xlim(0, max(scores) * 1.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return [_fig_to_image(fig, _CONTENT_W)]


def _build_chart3_competitors(styles: dict, competitors: list[dict]) -> list:
    """Return competitor mentions chart or fallback text as a list of flowables.

    Horizontal bar chart: brand names on Y axis, raw mention count on X axis,
    vertical reference line at x=10 (threshold).
    Falls back to brief-specified text when competitors list is empty.
    """
    if not competitors:
        return [Paragraph(
            "No competitors meeting the minimum threshold (10 mentions) were identified.",
            styles["body"],
        )]

    brands = [c["competitor"] for c in competitors]
    counts = [c["mention_count"] for c in competitors]

    # Reverse so highest count is at top
    brands = brands[::-1]
    counts = counts[::-1]

    n = len(brands)
    bar_height = 0.55
    fig_h = max(1.8, n * 0.6 + 0.8)
    fig, ax = plt.subplots(figsize=(_CHART_W * 1.1, fig_h))

    ax.barh(range(n), counts, height=bar_height, color="#E07B39")

    # Reference line at x=10 (threshold)
    ax.axvline(x=10, color="#333333", linestyle="--", linewidth=1, label="Threshold (10)")

    ax.set_yticks(range(n))
    ax.set_yticklabels(brands, fontsize=9)
    ax.set_xlabel("Mention count", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return [_fig_to_image(fig, _CONTENT_W)]


def _build_scorecard(
    styles: dict,
    confidence_tier: str,
    output_quality: dict,
) -> list:
    """Return the evaluation scorecard as a list of flowables.

    Shows overall confidence tier prominently, then per-criterion rows with
    ✓ (score 4–5) or ⚠ (score 1–3) and the evidence string from the judge.
    No raw numeric scores are included.
    """
    elements = []

    # Overall confidence — large, prominent
    elements.append(
        Paragraph(f"Overall confidence: {confidence_tier.upper()}", styles["scorecard_tier"])
    )
    elements.append(Spacer(1, 0.3 * cm))

    # Per-criterion rows
    criteria = [
        ("Sentiment", "sentiment"),
        ("Pain points", "pain_points"),
        ("Competitors", "competitors"),
    ]

    rows = []
    for label, key in criteria:
        criterion = output_quality.get(key, {})
        score = criterion.get("score", 0)
        evidence = criterion.get("evidence", "")
        indicator = "✓" if score >= 4 else "⚠"
        rows.append([
            Paragraph(f"{label}:", styles["scorecard_row"]),
            Paragraph(indicator, styles["scorecard_row"]),
            Paragraph(evidence, styles["scorecard_row"]),
        ])

    label_col = 2.5 * cm
    indicator_col = 0.8 * cm
    evidence_col = _CONTENT_W - label_col - indicator_col

    table = Table(rows, colWidths=[label_col, indicator_col, evidence_col])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#F9F9F9"), colors.white]),
    ]))
    elements.append(table)
    return elements


def _build_footer(styles: dict, analysis_timestamp: datetime) -> list:
    """Return footer section as a list of flowables.

    Contains four fields per brief Stage 9 footer spec:
    extraction model, evaluation model, analysis timestamp, fetch parameters.
    """
    ts_str = analysis_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"Extraction model: {EXTRACTION_MODEL}",
        f"Evaluation model: {EVALUATION_MODEL}",
        f"Analysis timestamp: {ts_str}",
        "Comment fetch parameters: relevance sort, top-level only, max 500 fetched",
    ]
    return [Paragraph("<br/>".join(lines), styles["footer"])]


_MEDIUM_MSG = (
    "Analysis confidence is moderate. "
    "One or more findings have limited sample support. "
    "Treat as directional."
)
_LOW_MSG = (
    "Analysis confidence is low. "
    "Manual review of comments is recommended "
    "before acting on these findings."
)


def _build_confidence_banner(styles: dict, confidence_tier: str) -> list:
    """Return a confidence banner as a list of flowables.

    High  → empty list (no banner rendered)
    Medium → yellow-styled banner with the exact Medium message from the brief
    Low    → red-styled banner with the exact Low message from the brief
    """
    if confidence_tier == "High":
        return []

    if confidence_tier == "Medium":
        text = _MEDIUM_MSG
        text_style = styles["banner_medium"]
        bg_color = colors.HexColor("#FFF8DC")
        border_color = colors.HexColor("#C8A000")
    else:
        text = _LOW_MSG
        text_style = styles["banner_low"]
        bg_color = colors.HexColor("#FFF0F0")
        border_color = colors.HexColor("#C80000")

    banner_table = Table(
        [[Paragraph(text, text_style)]],
        colWidths=[_CONTENT_W],
    )
    banner_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_color),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))

    return [banner_table, Spacer(1, 0.4 * cm)]


def _build_header(
    styles: dict,
    video_title: str,
    video_url: str,
    analysis_timestamp: datetime,
    comments_fetched: int,
    processed_count: int,
    relevant_count: int,
    quality_mode: str,
    primary_product: str,
) -> list:
    """Return header section as a list of flowables.

    The 'Successfully processed' row shows a ⚠ indicator when processed_count
    is less than comments_fetched (i.e., one or more extraction chunks failed).
    quality_mode is displayed capitalized ('Full' or 'Degraded').
    """
    chunk_failures = processed_count < comments_fetched
    processed_str = f"{processed_count} ⚠" if chunk_failures else str(processed_count)
    date_str = analysis_timestamp.strftime("%Y-%m-%d")
    mode_display = quality_mode.capitalize()

    rows = [
        [Paragraph("Video title:", styles["header_label"]),
         Paragraph(video_title, styles["header_value"])],
        [Paragraph("Video URL:", styles["header_label"]),
         Paragraph(video_url, styles["header_value"])],
        [Paragraph("Analysis date:", styles["header_label"]),
         Paragraph(date_str, styles["header_value"])],
        [Paragraph("Comments fetched:", styles["header_label"]),
         Paragraph(str(comments_fetched), styles["header_value"])],
        [Paragraph("Successfully processed:", styles["header_label"]),
         Paragraph(processed_str, styles["header_value"])],
        [Paragraph(f"Relevant to [{primary_product}]:", styles["header_label"]),
         Paragraph(str(relevant_count), styles["header_value"])],
        [Paragraph("Quality mode:", styles["header_label"]),
         Paragraph(mode_display, styles["header_value"])],
    ]

    table = Table(rows, colWidths=[_LABEL_COL, _VALUE_COL])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))

    return [table, Spacer(1, 0.5 * cm)]
