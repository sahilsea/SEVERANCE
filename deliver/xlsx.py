"""Deliverable Excel (.xlsx) report generation -- the spreadsheet sibling of
deliver/docx.py's Word report, same underlying data (trust/reports.py),
same classification stamping, same refusal to build a report for an
abstained response. A structured table (citations, denials) reads more
naturally as a spreadsheet than as Word prose, which is the real reason
this exists alongside the Word report rather than replacing it -- different
deliverable shapes for different uses, not a redundant second format.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from contracts import AskResponse, Tier

_BOLD_STRIP_RE = re.compile(r"\*\*(.+?)\*\*")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")
_BULLET_RE = re.compile(r"^[-*]\s+(.*)$")

# Stamping color palette by Tier -- same mapping as deliver/docx.py, in
# openpyxl's ARGB hex form instead of python-docx's RGBColor.
TIER_FILL_COLORS: dict[Tier, str] = {
    Tier.SECRET: "FFB91C1C",
    Tier.CONFIDENTIAL: "FFC2410C",
    Tier.INTERNAL: "FF1D4ED8",
    Tier.PUBLIC: "FF047857",
}


def get_classification_banner_text(response: AskResponse) -> str:
    tier_str = response.effective_label.tier.value.upper()
    comps = response.effective_label.compartments
    if comps:
        comp_str = ", ".join(sorted(c.value.upper() for c in comps))
        return f"CLASSIFICATION: {tier_str} // COMPARTMENTS: [{comp_str}]"
    return f"CLASSIFICATION: {tier_str} // UNRESTRICTED DISTRIBUTION"


def get_report_filename(response: AskResponse) -> str:
    row_id = response.ledger_row_id or "Unrecorded"
    tier_str = response.effective_label.tier.value.upper()
    return f"SEVERANCE_Report_Row{row_id}_{tier_str}.xlsx"


def _plain_text(markdown_line: str) -> str:
    """Strip '**bold**'/'#' markdown markers down to plain text -- a cell
    doesn't render markdown, so leaving the literal characters in would be
    worse than just stripping them."""
    line = _HEADING_RE.sub(r"\1", markdown_line)
    line = _BULLET_RE.sub(r"\1", line)
    return _BOLD_STRIP_RE.sub(r"\1", line).strip()


def build_report(
    response: AskResponse,
    question: str,
    output_path: Optional[str | Path] = None,
) -> bytes:
    """Build an Excel workbook: a Summary sheet (question, answer, metadata)
    and, when present, a Citations sheet and a Withheld Documents sheet --
    the same underlying verified data as deliver/docx.py's Word report.

    Refuses to build a report for an abstained response, same as the Word
    report -- there is nothing verified to hand out for that case.
    """
    if response.status == "abstained":
        raise ValueError("Cannot generate deliverable report for an abstained query response.")

    tier = response.effective_label.tier
    fill_color = TIER_FILL_COLORS.get(tier, "FF000000")
    banner_text = get_classification_banner_text(response)

    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"

    # -----------------------------------------------------------------------
    # Classification banner (mirrors the Word report's top banner + header/footer)
    # -----------------------------------------------------------------------
    summary.merge_cells("A1:D1")
    banner_cell = summary["A1"]
    banner_cell.value = banner_text
    banner_cell.font = Font(bold=True, size=12, color="FFFFFFFF")
    banner_cell.fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")
    banner_cell.alignment = Alignment(horizontal="center", vertical="center")
    summary.row_dimensions[1].height = 22

    # -----------------------------------------------------------------------
    # Metadata block
    # -----------------------------------------------------------------------
    row = 3
    meta_rows = [
        ("Query", question),
        ("Effective Classification", response.effective_label.tier.value.upper()),
        (
            "Effective Compartments",
            ", ".join(sorted(c.value.upper() for c in response.effective_label.compartments)) or "None",
        ),
        ("Ledger Row ID", f"#{response.ledger_row_id or 'Unrecorded'}"),
    ]
    for label, value in meta_rows:
        summary.cell(row=row, column=1, value=label).font = Font(bold=True)
        summary.cell(row=row, column=2, value=value)
        row += 1

    row += 1
    summary.cell(row=row, column=1, value="Verified Synthesized Findings").font = Font(bold=True, size=13)
    row += 1

    for raw_line in response.answer.split("\n"):
        line = _plain_text(raw_line)
        if not line:
            continue
        cell = summary.cell(row=row, column=1, value=line)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        summary.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
        row += 1

    summary.column_dimensions["A"].width = 100
    for col in "BCD":
        summary.column_dimensions[col].width = 20

    # -----------------------------------------------------------------------
    # Citations sheet -- the verbatim-verified proof, one row per citation
    # -----------------------------------------------------------------------
    if response.citations:
        cit_sheet = wb.create_sheet("Citations")
        headers = ["Document ID", "Page", "Verbatim Quoted Passage"]
        for col_idx, header in enumerate(headers, start=1):
            cell = cit_sheet.cell(row=1, column=col_idx, value=header)
            cell.font = Font(bold=True, color="FFFFFFFF")
            cell.fill = PatternFill(start_color="FF404040", end_color="FF404040", fill_type="solid")
        for row_idx, cit in enumerate(response.citations, start=2):
            cit_sheet.cell(row=row_idx, column=1, value=cit.doc_id)
            cit_sheet.cell(row=row_idx, column=2, value=cit.page)
            quote_cell = cit_sheet.cell(row=row_idx, column=3, value=cit.quote)
            quote_cell.alignment = Alignment(wrap_text=True, vertical="top")
        cit_sheet.column_dimensions["A"].width = 28
        cit_sheet.column_dimensions["B"].width = 10
        cit_sheet.column_dimensions["C"].width = 90
        cit_sheet.freeze_panes = "A2"

    # -----------------------------------------------------------------------
    # Withheld Documents sheet (denials without content -- same principle as
    # the Word report: names and reasons only, never withheld text itself)
    # -----------------------------------------------------------------------
    if response.denials:
        den_sheet = wb.create_sheet("Withheld Documents")
        headers = ["Document ID", "Reason Withheld"]
        for col_idx, header in enumerate(headers, start=1):
            cell = den_sheet.cell(row=1, column=col_idx, value=header)
            cell.font = Font(bold=True, color="FFFFFFFF")
            cell.fill = PatternFill(start_color="FF404040", end_color="FF404040", fill_type="solid")
        for row_idx, denial in enumerate(response.denials, start=2):
            den_sheet.cell(row=row_idx, column=1, value=denial.doc_id)
            den_sheet.cell(row=row_idx, column=2, value=denial.reason)
        den_sheet.column_dimensions["A"].width = 28
        den_sheet.column_dimensions["B"].width = 70

    bio = io.BytesIO()
    wb.save(bio)
    file_bytes = bio.getvalue()

    if output_path:
        Path(output_path).write_bytes(file_bytes)

    return file_bytes
