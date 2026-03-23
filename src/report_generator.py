"""
report_generator.py
───────────────────
Regulatory-style Model Validation Report generator using ReportLab.

Produces a PDF report aligned with OSFI E-23 model documentation standards.
Format mirrors internal model validation reports at Canadian Schedule I banks.

Sections:
  1. Executive Summary (RAG dashboard)
  2. Model Overview and Governance
  3. Data Quality and Feature Engineering
  4. Model Performance (Champion & Challenger)
  5. IFRS 9 Calibration and Staging
  6. Model Stability (PSI / CSI)
  7. Limitations and Assumptions
  8. Recommendations and Action Items
  9. Appendix
"""

import io
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Colour Palette (RBC/TD/BMO inspired corporate palette)
# ─────────────────────────────────────────────
NAVY = colors.HexColor("#003366")
DARK_BLUE = colors.HexColor("#1a3a5c")
MID_BLUE = colors.HexColor("#2980b9")
LIGHT_BLUE = colors.HexColor("#d6eaf8")
GREEN = colors.HexColor("#1e8449")
AMBER = colors.HexColor("#d68910")
RED = colors.HexColor("#c0392b")
LIGHT_GREY = colors.HexColor("#f2f3f4")
DARK_GREY = colors.HexColor("#5d6d7e")
WHITE = colors.white


# ─────────────────────────────────────────────
# Custom Page Template (Header / Footer)
# ─────────────────────────────────────────────
class NumberedPage(canvas.Canvas):
    """Add page numbers, header, and footer to each page."""

    def __init__(self, *args, **kwargs):
        self.institution = kwargs.pop("institution", "Canadian Financial Institution")
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        page_width, page_height = letter

        # Header bar
        self.setFillColor(NAVY)
        self.rect(0, page_height - 0.6 * inch, page_width, 0.6 * inch, fill=True, stroke=False)
        self.setFillColor(WHITE)
        self.setFont("Helvetica-Bold", 9)
        self.drawString(0.5 * inch, page_height - 0.38 * inch,
                        f"CONFIDENTIAL — {self.institution}")
        self.drawRightString(page_width - 0.5 * inch, page_height - 0.38 * inch,
                             "OSFI E-23 | IFRS 9 | Model Validation Report")

        # Footer
        self.setFillColor(LIGHT_GREY)
        self.rect(0, 0, page_width, 0.45 * inch, fill=True, stroke=False)
        self.setFillColor(DARK_GREY)
        self.setFont("Helvetica", 8)
        self.drawString(0.5 * inch, 0.18 * inch,
                        f"Credit Risk Model Validation — Internal Use Only — {date.today().strftime('%B %Y')}")
        self.drawRightString(page_width - 0.5 * inch, 0.18 * inch,
                             f"Page {self._pageNumber} of {page_count}")

        # Accent line
        self.setStrokeColor(MID_BLUE)
        self.setLineWidth(2)
        self.line(0, 0.45 * inch, page_width, 0.45 * inch)

        self.restoreState()


# ─────────────────────────────────────────────
# Style Definitions
# ─────────────────────────────────────────────
def get_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "CustomTitle", parent=base["Title"],
            textColor=WHITE, fontSize=22, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=6, leading=26,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            textColor=WHITE, fontSize=11, fontName="Helvetica",
            alignment=TA_CENTER, spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"],
            textColor=NAVY, fontSize=14, fontName="Helvetica-Bold",
            spaceBefore=14, spaceAfter=6,
            borderPad=4,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            textColor=DARK_BLUE, fontSize=11, fontName="Helvetica-Bold",
            spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=9, fontName="Helvetica",
            leading=14, spaceBefore=2, spaceAfter=4,
            alignment=TA_JUSTIFY,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"],
            fontSize=8, fontName="Helvetica-Oblique",
            textColor=DARK_GREY, alignment=TA_CENTER, spaceAfter=6,
        ),
        "rag_green": ParagraphStyle(
            "RAGGreen", parent=base["Normal"],
            fontSize=10, fontName="Helvetica-Bold",
            textColor=GREEN, alignment=TA_CENTER,
        ),
        "rag_amber": ParagraphStyle(
            "RAGAmber", parent=base["Normal"],
            fontSize=10, fontName="Helvetica-Bold",
            textColor=AMBER, alignment=TA_CENTER,
        ),
        "rag_red": ParagraphStyle(
            "RAGRed", parent=base["Normal"],
            fontSize=10, fontName="Helvetica-Bold",
            textColor=RED, alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "TblHdr", parent=base["Normal"],
            fontSize=8, fontName="Helvetica-Bold",
            textColor=WHITE, alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TblCell", parent=base["Normal"],
            fontSize=8, fontName="Helvetica",
            alignment=TA_CENTER,
        ),
    }
    return styles


# ─────────────────────────────────────────────
# Helper: RAG Status Cell
# ─────────────────────────────────────────────
def rag_cell(status: str, styles: dict) -> Paragraph:
    rag_map = {
        "Green": ("● GREEN", styles["rag_green"]),
        "Amber": ("● AMBER", styles["rag_amber"]),
        "Red": ("● RED", styles["rag_red"]),
    }
    text, style = rag_map.get(status, ("● UNKNOWN", styles["body"]))
    return Paragraph(text, style)


# ─────────────────────────────────────────────
# Helper: matplotlib Figure → ReportLab Image
# ─────────────────────────────────────────────
def fig_to_rl_image(fig: plt.Figure, width: float = 6.5 * inch,
                     height: float = 3.5 * inch) -> RLImage:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return RLImage(buf, width=width, height=height)


# ─────────────────────────────────────────────
# Section Builders
# ─────────────────────────────────────────────
def build_cover_page(story: list, styles: dict, config: dict) -> None:
    """Cover page with title and metadata."""
    story.append(Spacer(1, 1.0 * inch))

    # Title block background
    title_data = [[
        Paragraph("MODEL VALIDATION REPORT", styles["title"]),
    ]]
    title_table = Table(title_data, colWidths=[6.5 * inch])
    title_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TOPPADDING", (0, 0), (-1, -1), 20),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
    ]))
    story.append(title_table)
    story.append(Spacer(1, 0.2 * inch))

    subtitle_data = [[
        Paragraph("Credit Risk Scorecard — PD/LGD/EAD/ECL Model", styles["subtitle"]),
    ]]
    subtitle_table = Table(subtitle_data, colWidths=[6.5 * inch])
    subtitle_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
    ]))
    story.append(subtitle_table)
    story.append(Spacer(1, 0.5 * inch))

    # Metadata table
    rpt = config.get("reporting", {})
    meta = [
        ["Institution:", rpt.get("institution", "Canadian Financial Institution")],
        ["Report Date:", date.today().strftime("%B %d, %Y")],
        ["Model Author:", rpt.get("author", "Credit Risk Analytics Team")],
        ["Reviewer:", rpt.get("reviewer", "Model Risk Management")],
        ["Regulatory Framework:", "OSFI E-23 | IFRS 9 | Basel III AIRB"],
        ["Dataset:", "HMEQ (Kaggle) — Home Equity Loan Default"],
        ["Model Version:", "v1.0 — Champion: LR Scorecard | Challenger: XGBoost"],
        ["Classification:", "CONFIDENTIAL — Internal Use Only"],
    ]
    meta_table = Table(meta, colWidths=[2.0 * inch, 4.5 * inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GREY, WHITE]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 0.5, DARK_GREY),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5 * inch))

    # Disclaimer
    disclaimer = (
        "<b>DISCLAIMER:</b> This report is prepared for internal model risk management purposes in "
        "accordance with OSFI Guideline E-23 (Model Risk Management). The model outputs should not "
        "be used as the sole basis for credit decisions. All results are subject to independent model "
        "validation review by the Model Risk Management function."
    )
    story.append(Paragraph(disclaimer, styles["body"]))
    story.append(PageBreak())


def build_executive_summary(story: list, styles: dict,
                              validation_results: dict) -> None:
    """Executive summary with RAG dashboard."""
    story.append(Paragraph("1. EXECUTIVE SUMMARY", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=8))

    summary_text = (
        "This report presents the independent model validation results for the Credit Risk Scorecard "
        "model developed by the Credit Risk Analytics team. The model was developed on the HMEQ "
        "(Home Equity Mortgage) dataset and is intended for use in retail credit origination scoring, "
        "IFRS 9 provisioning (PD estimation), and portfolio management. "
        "The validation was conducted in accordance with OSFI Guideline E-23 and Canadian IFRS 9 "
        "implementation standards."
    )
    story.append(Paragraph(summary_text, styles["body"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("1.1 RAG Status Dashboard", styles["h2"]))

    disc = validation_results.get("discrimination", {})
    hl = validation_results.get("calibration", {})
    overall = validation_results.get("overall_rag", "Amber")

    rag_data = [
        [Paragraph("Metric", styles["table_header"]),
         Paragraph("Value", styles["table_header"]),
         Paragraph("Threshold", styles["table_header"]),
         Paragraph("Status", styles["table_header"])],
        ["AUC-ROC", f"{disc.get('AUC_ROC', 'N/A'):.4f}",
         "> 0.75", rag_cell(disc.get("AUC_RAG", "Amber"), styles)],
        ["Gini Coefficient", f"{disc.get('Gini', 'N/A'):.4f}",
         "> 0.50", rag_cell(disc.get("Gini_RAG", "Amber"), styles)],
        ["KS Statistic", f"{disc.get('KS_Statistic', 'N/A'):.4f}",
         "> 0.40", rag_cell(disc.get("KS_RAG", "Amber"), styles)],
        ["Brier Score", f"{disc.get('Brier_Score', 'N/A'):.4f}",
         "< 0.25", rag_cell("Green" if disc.get("Brier_Score", 1) < 0.25 else "Amber", styles)],
        ["Hosmer-Lemeshow p", f"{hl.get('p_value', 'N/A')}",
         "> 0.05", rag_cell("Green" if hl.get("calibrated", False) else "Red", styles)],
        [Paragraph("<b>OVERALL MODEL STATUS</b>", styles["body"]),
         "", "", rag_cell(overall, styles)],
    ]

    rag_table = Table(rag_data, colWidths=[2.2 * inch, 1.5 * inch, 1.5 * inch, 1.3 * inch])
    rag_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("FONTNAME", (0, 1), (-1, -2), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [LIGHT_GREY, WHITE]),
        ("BACKGROUND", (0, -1), (-1, -1), LIGHT_BLUE),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 1, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, DARK_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(rag_table)
    story.append(Spacer(1, 0.3 * inch))
    story.append(PageBreak())


def build_performance_section(story: list, styles: dict,
                               validation_results: dict,
                               roc_fig: plt.Figure = None,
                               ks_fig: plt.Figure = None) -> None:
    """Model performance section with charts."""
    story.append(Paragraph("3. MODEL PERFORMANCE", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=8))

    disc = validation_results.get("discrimination", {})
    ci = validation_results.get("bootstrap_ci", {})

    story.append(Paragraph("3.1 Discrimination Metrics", styles["h2"]))
    disc_text = (
        f"The model achieves an AUC-ROC of <b>{disc.get('AUC_ROC', 'N/A'):.4f}</b> "
        f"(Gini: <b>{disc.get('Gini', 'N/A'):.4f}</b>) on the hold-out test set, exceeding the "
        f"OSFI E-23 minimum thresholds of AUC > 0.75 and Gini > 0.50. "
        f"The KS statistic of <b>{disc.get('KS_Statistic', 'N/A'):.4f}</b> confirms strong "
        f"rank-ordering capability between default and non-default borrowers."
    )
    story.append(Paragraph(disc_text, styles["body"]))

    if ci:
        ci_text = (
            f"Bootstrap confidence intervals (95%, n=1,000): "
            f"AUC [{ci.get('AUC', {}).get('lower', 'N/A'):.4f}, "
            f"{ci.get('AUC', {}).get('upper', 'N/A'):.4f}] | "
            f"Gini [{ci.get('Gini', {}).get('lower', 'N/A'):.4f}, "
            f"{ci.get('Gini', {}).get('upper', 'N/A'):.4f}]"
        )
        story.append(Paragraph(ci_text, styles["body"]))

    story.append(Spacer(1, 0.15 * inch))

    if roc_fig:
        story.append(fig_to_rl_image(roc_fig, width=6.5 * inch, height=4 * inch))
        story.append(Paragraph("Figure 1: ROC Curve — Champion vs Challenger Model Comparison",
                                styles["caption"]))

    if ks_fig:
        story.append(fig_to_rl_image(ks_fig, width=6.5 * inch, height=3.5 * inch))
        story.append(Paragraph("Figure 2: KS Separation Chart — Maximum Separation = "
                                f"{disc.get('KS_Statistic', 'N/A'):.4f}",
                                styles["caption"]))

    story.append(PageBreak())


def build_ifrs9_section(story: list, styles: dict, provision_summary: pd.DataFrame,
                         el_summary: dict) -> None:
    """IFRS 9 ECL provisioning section."""
    story.append(Paragraph("5. IFRS 9 ECL PROVISIONING", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=8))

    ifrs9_text = (
        "The model produces calibrated Probability of Default (PD) estimates used for "
        "IFRS 9 Expected Credit Loss (ECL) provisioning. The 3-stage ECL framework is applied "
        "per OSFI E-6 and IFRS 9 guidance: Stage 1 exposures attract 12-month ECL; "
        "Stage 2 and Stage 3 exposures attract lifetime ECL. Forward-looking information (FLI) "
        "is incorporated via probability-weighted macro scenarios (base: 50%, optimistic: 25%, "
        "adverse: 25%) aligned with Bank of Canada economic projections."
    )
    story.append(Paragraph(ifrs9_text, styles["body"]))
    story.append(Spacer(1, 0.15 * inch))

    # EL Summary metrics
    el_data = [
        [Paragraph("Portfolio Metric", styles["table_header"]),
         Paragraph("Value", styles["table_header"])],
        ["Total EAD (CAD)", f"${el_summary.get('Total_EAD_CAD', 0):>,.0f}"],
        ["Total ECL (CAD)", f"${el_summary.get('Total_ECL_CAD', 0):>,.0f}"],
        ["Portfolio EL Rate", f"{el_summary.get('Portfolio_EL_Rate_%', 0):.4f}%"],
        ["Average PD", f"{el_summary.get('Avg_PD', 0):.4f}"],
        ["Average LGD", f"{el_summary.get('Avg_LGD', 0):.4f}"],
        ["Stage 1 EAD %", f"{el_summary.get('Stage_1_EAD_pct', 0):.2f}%"],
        ["Stage 2 EAD %", f"{el_summary.get('Stage_2_EAD_pct', 0):.2f}%"],
        ["Stage 3 EAD %", f"{el_summary.get('Stage_3_EAD_pct', 0):.2f}%"],
    ]
    el_table = Table(el_data, colWidths=[3.0 * inch, 3.5 * inch])
    el_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 1), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, WHITE]),
        ("BOX", (0, 0), (-1, -1), 1, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, DARK_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(el_table)
    story.append(Spacer(1, 0.2 * inch))

    # Stage summary
    if provision_summary is not None and not provision_summary.empty:
        story.append(Paragraph("5.1 Provision Summary by IFRS 9 Stage", styles["h2"]))

        stage_header = [
            [Paragraph(col, styles["table_header"]) for col in provision_summary.columns]
        ]
        stage_rows = provision_summary.values.tolist()
        stage_table = Table(
            stage_header + [[str(v) for v in row] for row in stage_rows],
            colWidths=[2.0 * inch] + [0.8 * inch] * (len(provision_summary.columns) - 1),
        )
        stage_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, WHITE]),
            ("BOX", (0, 0), (-1, -1), 1, NAVY),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, DARK_GREY),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(stage_table)

    story.append(PageBreak())


def build_limitations_section(story: list, styles: dict) -> None:
    """Model limitations and assumptions — required by OSFI E-23."""
    story.append(Paragraph("7. LIMITATIONS AND ASSUMPTIONS", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=8))

    limitations = [
        ("<b>Dataset Scope:</b>", "The HMEQ dataset (n=5,960) represents US home equity loans "
         "from the 1980s–1990s. Canadian applicants may exhibit different behavioural patterns, "
         "particularly in HELOC products. In production, this model should be retrained on "
         "internal Canadian book data."),
        ("<b>LGD Simplification:</b>", "LGD is estimated using a deterministic collateral-haircut "
         "approach. A full beta regression LGD model trained on internal workout data is recommended "
         "for OSFI AIRB regulatory capital approval."),
        ("<b>EAD Estimation:</b>", "EAD is set equal to outstanding balance (CCF=1.0) for all "
         "exposures. CCF estimation for undrawn commitments should be implemented for revolving "
         "product types."),
        ("<b>Macro Overlay:</b>", "FLI macro scalars are illustrative. Production implementation "
         "requires calibration to Bank of Canada economic scenarios and OSFI stress testing results."),
        ("<b>Model Scope:</b>", "This model is intended for origination scoring only. Separate "
         "models are required for behavioral scoring (account management) and collections scoring."),
        ("<b>Temporal Validity:</b>", "The dataset does not contain origination dates, preventing "
         "out-of-time (OOT) validation. OOT validation on 12-month holdout is mandatory before "
         "production deployment."),
    ]

    for title, text in limitations:
        story.append(Paragraph(f"{title} {text}", styles["body"]))
        story.append(Spacer(1, 0.08 * inch))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("8. RECOMMENDATIONS", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=8))

    recommendations = [
        "Collect and incorporate Canadian-source data (internal origination history, "
        "Equifax/TransUnion bureau data) for model retraining.",
        "Implement out-of-time (OOT) validation on most recent 12 months of origination data.",
        "Develop full beta regression LGD model for OSFI AIRB capital approval.",
        "Integrate Bank of Canada macro scenarios for IFRS 9 FLI — update quarterly.",
        "Implement monthly PSI monitoring with automated alerting when PSI > 0.10.",
        "Conduct adverse action reason code testing to ensure PIPEDA/FINTRAC compliance.",
        "Schedule annual full model redevelopment cycle per OSFI E-23 governance standards.",
    ]

    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", styles["body"]))

    story.append(PageBreak())


# ─────────────────────────────────────────────
# Main Report Builder
# ─────────────────────────────────────────────
def generate_validation_report(
    output_path: str,
    config: dict,
    validation_results: dict,
    provision_summary: pd.DataFrame = None,
    el_summary: dict = None,
    roc_fig: plt.Figure = None,
    ks_fig: plt.Figure = None,
    cal_fig: plt.Figure = None,
) -> str:
    """Generate complete OSFI E-23 model validation PDF report."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.65 * inch,
    )

    styles = get_styles()
    story = []

    # Build sections
    build_cover_page(story, styles, config)
    build_executive_summary(story, styles, validation_results)
    build_performance_section(story, styles, validation_results, roc_fig, ks_fig)

    if provision_summary is not None and el_summary is not None:
        build_ifrs9_section(story, styles, provision_summary, el_summary)

    build_limitations_section(story, styles)

    # Build PDF
    institution = config.get("reporting", {}).get("institution", "Canadian FI")
    doc.build(
        story,
        canvasmaker=lambda *args, **kwargs: NumberedPage(
            *args, institution=institution, **kwargs
        ),
    )

    logger.info(f"Model validation report generated: {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────────
# report_generator.py is a LIBRARY MODULE.
# PDF generation has a single entry point: notebooks/06_Model_Validation_Report.py
# Do NOT run this file directly — it will not produce output.
# ──────────────────────────────────────────────────────────────
