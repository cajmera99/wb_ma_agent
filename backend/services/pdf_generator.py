"""
PDF report generator — 10 pages, one acquirer per page.

Uses reportlab's Platypus (flow-based layout) so each page reflows
naturally rather than requiring manual coordinate positioning.

Output: backend/output/{run_id}.pdf
"""

import os
import json
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepInFrame,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

from backend.models.target import TargetProfile


def _esc(text: str) -> str:
    """Escape XML special characters for reportlab Paragraph markup."""
    if not text:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

OUTPUT_DIR = Path("backend/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Usable content width: letter (8.5") minus left + right margins (0.6" each)
PAGE_W = 7.3 * inch

# William Blair palette (approximate)
WB_BLUE = HexColor("#003087")
WB_LIGHT_BLUE = HexColor("#EAF0F8")
WB_GRAY = HexColor("#F5F5F5")
WB_DARK_GRAY = HexColor("#444444")
WB_RED = HexColor("#C0392B")
WB_AMBER = HexColor("#D68910")
WB_GREEN = HexColor("#1E8449")

SEVERITY_COLORS = {"High": WB_RED, "Medium": WB_AMBER, "Low": WB_GREEN}
CONVICTION_COLORS = {"High": WB_GREEN, "Medium": WB_AMBER, "Low": WB_RED}

# Hardcoded hex strings — never use HexColor.__int__() which is unreliable across versions
SEVERITY_HEX = {"High": "#C0392B", "Medium": "#D68910", "Low": "#1E8449"}
CONVICTION_HEX = {"High": "#1E8449", "Medium": "#D68910", "Low": "#C0392B"}


def _build_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title",
        parent=base["Title"],
        fontSize=22,
        textColor=white,
        leading=28,
        spaceAfter=6,
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub",
        parent=base["Normal"],
        fontSize=11,
        textColor=white,
        leading=16,
    )
    styles["page_header"] = ParagraphStyle(
        "page_header",
        parent=base["Normal"],
        fontSize=14,
        textColor=white,
        leading=18,
        fontName="Helvetica-Bold",
    )
    styles["section_label"] = ParagraphStyle(
        "section_label",
        parent=base["Normal"],
        fontSize=8,
        textColor=WB_BLUE,
        fontName="Helvetica-Bold",
        spaceAfter=3,
        spaceBefore=10,
    )
    styles["body"] = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontSize=9,
        textColor=WB_DARK_GRAY,
        leading=14,
        spaceAfter=4,
    )
    styles["small"] = ParagraphStyle(
        "small",
        parent=base["Normal"],
        fontSize=8,
        textColor=WB_DARK_GRAY,
        leading=12,
    )
    styles["footer"] = ParagraphStyle(
        "footer",
        parent=base["Normal"],
        fontSize=7,
        textColor=HexColor("#999999"),
        alignment=TA_CENTER,
    )
    return styles


def _conviction_badge(conviction: str) -> str:
    hex_color = CONVICTION_HEX.get(conviction, "#444444")
    return f'<font color="{hex_color}"><b>■ {conviction} Conviction</b></font>'


def _score_cell(label: str, score: float, styles: dict) -> list:
    """Returns [label_paragraph, score_paragraph] for one score grid cell."""
    if score >= 70:
        hex_col = "#1E8449"
    elif score >= 40:
        hex_col = "#D68910"
    else:
        hex_col = "#C0392B"
    label_p = Paragraph(
        f'<font size="7" color="#003087"><b>{label.upper()}</b></font>',
        styles["small"],
    )
    score_p = Paragraph(
        f'<font size="11" color="{hex_col}"><b>{score:.0f}</b></font>'
        f'<font size="7" color="#888888"> / 100</font>',
        styles["small"],
    )
    return [label_p, score_p]


def _build_cover_page(target: TargetProfile, styles: dict, flowables: list) -> None:
    """Cover page with target profile summary."""
    # Blue header banner simulated with a Table
    cover_data = [[
        Paragraph("M&amp;A Acquirer Identification Report", styles["cover_title"]),
    ]]
    cover_table = Table(cover_data, colWidths=[PAGE_W])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WB_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 30),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 30),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
    ]))
    flowables.append(cover_table)
    flowables.append(Spacer(1, 0.3 * inch))

    flowables.append(Paragraph("Target Company Profile", styles["section_label"]))
    flowables.append(HRFlowable(width="100%", thickness=1, color=WB_BLUE))
    flowables.append(Spacer(1, 0.1 * inch))

    target_data = [
        ["Sector", target.sector],
        ["Enterprise Value", f"~${target.deal_size_mm:,.0f}MM"],
        ["Geography", target.geography],
        ["Ownership", target.ownership],
        ["Profile", target.profile_description or "—"],
    ]

    t = Table(target_data, colWidths=[1.8 * inch, PAGE_W - 1.8 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), WB_BLUE),
        ("TEXTCOLOR", (1, 0), (1, -1), WB_DARK_GRAY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, WB_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    flowables.append(t)
    flowables.append(Spacer(1, 0.25 * inch))

    # ── Scoring Methodology ────────────────────────────────────────────────────
    flowables.append(Paragraph("Acquirer Scoring Methodology", styles["section_label"]))
    flowables.append(HRFlowable(width="100%", thickness=1, color=WB_BLUE))
    flowables.append(Spacer(1, 0.08 * inch))

    # Paragraph styles for table cells — required for text to wrap within column bounds
    hdr_style = ParagraphStyle("sc_hdr", fontSize=8, textColor=white,
                               fontName="Helvetica-Bold", leading=10)
    dim_style = ParagraphStyle("sc_dim", fontSize=8, textColor=WB_BLUE,
                               fontName="Helvetica-Bold", leading=10)
    wt_style  = ParagraphStyle("sc_wt",  fontSize=8, textColor=WB_DARK_GRAY,
                               fontName="Helvetica-Bold", leading=10, alignment=TA_CENTER)
    def_style = ParagraphStyle("sc_def", fontSize=8, textColor=WB_DARK_GRAY,
                               leading=11)

    def row(dim, wt, defn, is_header=False):
        s = hdr_style if is_header else dim_style
        d = hdr_style if is_header else def_style
        w = hdr_style if is_header else wt_style
        return [Paragraph(dim, s), Paragraph(wt, w), Paragraph(defn, d)]

    COL_DIM = 1.5 * inch
    COL_WT  = 0.72 * inch
    COL_DEF = PAGE_W - COL_DIM - COL_WT

    scoring_rows = [
        row("Dimension", "Weight", "Definition", is_header=True),
        row("Sector Affinity",    "35%",
            "Primary-sector matches score 1.0, adjacent sectors 0.7, secondary 0.3; "
            "normalized across all of the acquirer's historical deals."),
        row("Deal Size Match",    "20%",
            "Gaussian decay from the acquirer's median deal size to the target EV; "
            "an exact match scores 1.0 and falls off symmetrically as the gap widens."),
        row("Rationale Alignment","20%",
            "Share of the acquirer's top rationale tags (e.g. Platform Build, Bolt-on, "
            "Cost Synergies) that are classified as high-relevance for this target."),
        row("Recency",            "10%",
            "Blends exponential time-decay since last deal (−0.15/year) with recent "
            "deal count; 3+ transactions since 2022 receives the maximum score."),
        row("Outcome Quality",    "10%",
            "Ratio of Closed to total deals — reflects the acquirer's track record "
            "of completing announced transactions."),
        row("Ownership Match",    "5%",
            "Share of prior targets whose ownership type matches this target; "
            "Private and PE-Backed are treated as equivalent."),
    ]

    scoring_table = Table(
        scoring_rows,
        colWidths=[COL_DIM, COL_WT, COL_DEF],
    )
    scoring_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  WB_BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, WB_GRAY]),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 7),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW",      (0, 0), (-1, -1), 0.25, HexColor("#DDDDDD")),
        ("LINEAFTER",      (0, 0), (1, -1),  0.25, HexColor("#DDDDDD")),
    ]))
    flowables.append(scoring_table)
    flowables.append(Spacer(1, 0.2 * inch))

    flowables.append(Paragraph(
        "Each dimension scores 0–100. The composite score is a weighted sum. "
        "Dimension scores are color-coded: "
        "<font color='#1E8449'><b>green ≥ 70</b></font>  "
        "<font color='#D68910'><b>amber ≥ 40</b></font>  "
        "<font color='#C0392B'><b>red &lt; 40</b></font>  "
        "<b>·</b>  "
        "Conviction level derives from composite score: "
        "<font color='#1E8449'><b>High &gt; 80</b></font>  "
        "<font color='#D68910'><b>Medium 50–79</b></font>  "
        "<font color='#C0392B'><b>Low &lt; 50</b></font>",
        styles["small"],
    ))
    flowables.append(Spacer(1, 0.2 * inch))

    # Footer is drawn on every page via the canvas callback in generate_pdf()


def _build_acquirer_page(rationale: dict, styles: dict, flowables: list) -> None:
    """One page per acquirer. Renders all 6 sections."""
    name = rationale.get("acquirer_name", "Unknown")
    acq_type = rationale.get("acquirer_type", "Strategic")
    score = rationale.get("composite_score", 0)
    rank = rationale.get("rank", "?")
    conviction = rationale.get("conviction_level", "Medium")

    # If this was a failed rationale, render an error page
    if "error" in rationale:
        flowables.append(Paragraph(
            f"#{rank} — {name} — Rationale generation failed: {rationale['error']}",
            styles["body"],
        ))
        flowables.append(PageBreak())
        return

    # Collect this page's content in a local list so it can be wrapped in KeepInFrame.
    # KeepInFrame(mode='shrink') scales the entire page down proportionally when content
    # would overflow, preventing the blank extra page that occurs when Section 2 prose
    # runs long and pushes Section 6 onto a second page.
    page_content = []

    # ── Header Banner ──────────────────────────────────────────────────────────
    header_data = [[
        Paragraph(f"#{rank}  {name}", styles["page_header"]),
        Paragraph(
            f"{acq_type} | Score: {score:.1f}/100 | {_conviction_badge(conviction)}",
            ParagraphStyle("hdr_right", parent=styles["page_header"],
                           fontSize=8, leading=10, alignment=TA_RIGHT),
        ),
    ]]
    header_table = Table(header_data, colWidths=[PAGE_W * 0.44, PAGE_W * 0.56])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WB_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    page_content.append(header_table)
    page_content.append(Spacer(1, 0.1 * inch))

    # ── Score Grid ─────────────────────────────────────────────────────────────
    sub = rationale.get("sub_scores", {})
    dims = [
        ("Sector Affinity",   sub.get("sector", 0)),
        ("Deal Size Match",   sub.get("deal_size", 0)),
        ("Rationale Align",   sub.get("rationale", 0)),
        ("Recency",           sub.get("recency", 0)),
        ("Outcome Quality",   sub.get("outcome", 0)),
        ("Ownership Match",   sub.get("ownership", 0)),
    ]
    # 6 dimensions in a 3-column × 2-row grid; each cell has label + score
    score_grid = [
        [_score_cell(dims[0][0], dims[0][1], styles),
         _score_cell(dims[1][0], dims[1][1], styles),
         _score_cell(dims[2][0], dims[2][1], styles)],
        [_score_cell(dims[3][0], dims[3][1], styles),
         _score_cell(dims[4][0], dims[4][1], styles),
         _score_cell(dims[5][0], dims[5][1], styles)],
    ]
    # Each cell is a nested Table (label row + score row)
    col3 = PAGE_W / 3
    def _cell_table(cell_content):
        t = Table([[cell_content[0]], [cell_content[1]]], colWidths=[col3])
        t.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ]))
        return t

    outer_grid = Table(
        [[_cell_table(score_grid[0][0]),
          _cell_table(score_grid[0][1]),
          _cell_table(score_grid[0][2])],
         [_cell_table(score_grid[1][0]),
          _cell_table(score_grid[1][1]),
          _cell_table(score_grid[1][2])]],
        colWidths=[col3, col3, col3],
    )
    outer_grid.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), WB_LIGHT_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("LINEBELOW",     (0, 0), (-1, 0), 0.5, HexColor("#D0DCF0")),
        ("LINEAFTER",     (0, 0), (1, -1), 0.5, HexColor("#D0DCF0")),
    ]))
    page_content.append(outer_grid)
    page_content.append(Spacer(1, 0.1 * inch))

    # ── Section 1: Acquirer Overview ───────────────────────────────────────────
    page_content.append(Paragraph("1. ACQUIRER OVERVIEW", styles["section_label"]))
    page_content.append(HRFlowable(width="100%", thickness=0.5, color=WB_BLUE))
    page_content.append(Paragraph(_esc(rationale.get("acquirer_overview", "")), styles["body"]))

    # ── Section 2: Strategic Fit Thesis ────────────────────────────────────────
    page_content.append(Paragraph("2. STRATEGIC FIT THESIS", styles["section_label"]))
    page_content.append(HRFlowable(width="100%", thickness=0.5, color=WB_BLUE))
    page_content.append(Paragraph(_esc(rationale.get("strategic_fit_thesis", "")), styles["body"]))

    # ── Section 3: Precedent Activity ──────────────────────────────────────────
    page_content.append(Paragraph("3. PRECEDENT ACTIVITY  —  last 5 deals shown, sector-relevant first", styles["section_label"]))
    page_content.append(HRFlowable(width="100%", thickness=0.5, color=WB_BLUE))

    deals = rationale.get("precedent_deals", [])[:5]
    if deals:
        deal_rows = [["Transaction", "Size ($MM)", "Type", "EV/EBITDA", "Outcome"]]
        for d in deals:
            ev_ebitda = f"{d.get('ev_ebitda_multiple', 'N/A')}x" if d.get("ev_ebitda_multiple") else "N/A"
            deal_rows.append([
                _esc(f"{d.get('target_company', '?')} ({d.get('deal_year', '?')})"),
                f"${d.get('deal_size_mm', 0):,.0f}",
                _esc(d.get("deal_type", "?")),
                ev_ebitda,
                _esc(d.get("outcome", "?").strip().rstrip("},. {")),
            ])
        deals_table = Table(
            deal_rows,
            colWidths=[2.5 * inch, 0.85 * inch, 1.5 * inch, 1.0 * inch, 1.15 * inch],
        )
        deals_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), WB_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, WB_GRAY]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.25, HexColor("#DDDDDD")),
        ]))
        page_content.append(deals_table)
    else:
        page_content.append(Paragraph("No precedent deal data available.", styles["small"]))

    # ── Section 4: Valuation Context ───────────────────────────────────────────
    page_content.append(Paragraph("4. VALUATION CONTEXT", styles["section_label"]))
    page_content.append(HRFlowable(width="100%", thickness=0.5, color=WB_BLUE))
    val = rationale.get("valuation_context", {})
    if val:
        ev_ebitda = val.get("median_ev_ebitda", "N/A")
        ev_rev = val.get("median_ev_revenue", "N/A")
        count = val.get("deal_count_in_range", 0)
        note = val.get("note", "")
        val_text = (
            f"Market median EV/EBITDA: <b>{_esc(str(ev_ebitda))}x</b>  |  "
            f"Market median EV/Revenue: <b>{_esc(str(ev_rev))}x</b>  |  "
            f"Based on <b>{count}</b> comparable closed transactions.  {_esc(note)}"
        )
        page_content.append(Paragraph(val_text, styles["body"]))

    # ── Section 5: Risk Flags ──────────────────────────────────────────────────
    page_content.append(Paragraph("5. RISK FLAGS", styles["section_label"]))
    page_content.append(HRFlowable(width="100%", thickness=0.5, color=WB_BLUE))

    _severity_order = {"High": 0, "Medium": 1, "Low": 2}
    risk_flags = sorted(
        rationale.get("risk_flags", []),
        key=lambda f: _severity_order.get(f.get("severity", "Low"), 2),
    )[:2]
    for flag in risk_flags:
        severity = flag.get("severity", "Medium")
        hex_col = SEVERITY_HEX.get(severity, "#444444")
        risk_text = (
            f'<font color="{hex_col}"><b>[{severity}]</b></font> '
            f'<b>{_esc(flag.get("risk_type", "Risk"))}</b> — {_esc(flag.get("description", ""))}'
        )
        page_content.append(Paragraph(risk_text, styles["body"]))

    # ── Section 6: Conviction Level ────────────────────────────────────────────
    hex_conv = CONVICTION_HEX.get(conviction, "#444444")
    page_content.append(Paragraph("6. CONVICTION LEVEL", styles["section_label"]))
    page_content.append(HRFlowable(width="100%", thickness=0.5, color=WB_BLUE))
    page_content.append(Paragraph(
        f'<font color="{hex_conv}"><b>{conviction}</b></font>  — '
        f'{_esc(rationale.get("conviction_rationale", ""))}',
        styles["body"],
    ))

    # Wrap in KeepInFrame to enforce hard one-page limit per acquirer.
    # 9.6" = letter (11") minus top margin (0.6") minus bottom margin (0.6") minus buffer.
    # mode='shrink' scales content proportionally rather than overflowing to a blank page.
    kif = KeepInFrame(maxWidth=PAGE_W, maxHeight=9.6 * inch, content=page_content, mode='shrink')
    flowables.append(kif)
    flowables.append(PageBreak())


def _draw_page_footer(canvas, doc):
    """Draws the confidentiality footer on every page at a fixed canvas position."""
    date_str = datetime.utcnow().strftime("%B %d, %Y")
    text = f"Prepared: {date_str}  |  Confidential — For Internal Use Only"
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(HexColor("#AAAAAA"))
    canvas.drawCentredString(doc.pagesize[0] / 2, 0.3 * inch, text)
    canvas.restoreState()


def generate_pdf(run_id: str, target: TargetProfile, rationales: list[dict]) -> str:
    """
    Generate the 10-page PDF. Returns the file path.
    Called synchronously (via run_in_executor from the async route).
    """
    pdf_path = str(OUTPUT_DIR / f"{run_id}.pdf")
    styles = _build_styles()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"M&A Acquirer Report — {target.sector}",
    )

    flowables = []

    # Cover page
    _build_cover_page(target, styles, flowables)
    flowables.append(PageBreak())

    # Sort by composite score descending so conviction level always matches position:
    # High conviction acquirers appear first, Low conviction last. The LLM rerank
    # determines which 10 make the shortlist; composite score determines their order.
    sorted_rationales = sorted(rationales, key=lambda r: -r.get("composite_score", 0))
    for i, rationale in enumerate(sorted_rationales):
        rationale = {**rationale, "rank": i + 1}
        _build_acquirer_page(rationale, styles, flowables)

    doc.build(flowables, onFirstPage=_draw_page_footer, onLaterPages=_draw_page_footer)
    return pdf_path
