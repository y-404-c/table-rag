"""文档化渲染：把 TAT-QA 的 2D array 渲染成带真实扰动的 PDF。

渲染工具用 reportlab，解析工具用 pdfplumber —— 刻意不同源（需求文档 3.2 的缓解手段），
但"程序渲染 → 程序解析"的循环论证风险仍存在，报告中必须如实披露。

扰动开关：跨页断表（表头重复）、合并单元格（SPAN）、双栏排版（表格左栏 + 正文右栏）、
单元格空白与列间距随机化。全部由 seed 控制，可复现。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from tablerag.config import RenderSettings
from tablerag.domain.text import parse_number
from tablerag.ingest.dataset import Context, infer_header_rows

_LATIN_FIX = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
}


def pdf_safe(text: str) -> str:
    out = []
    for ch in str(text):
        if ch in _LATIN_FIX:
            out.append(_LATIN_FIX[ch])
        elif ord(ch) < 256:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


@dataclass
class RenderResult:
    path: Path
    meta: dict = field(default_factory=dict)


def _cell_text(value: str, rng: random.Random, settings: RenderSettings) -> str:
    """单元格文字（保留原有字符，空白扰动交给 padding 完成，避免污染文本内容）。"""
    return pdf_safe(value).strip()


def _column_widths(cells: list[list[str]], total_width: float, font_size: float) -> list[float]:
    """按"列内最长不可断词"分配宽度：数字不能被折断，长标签允许按空格换行。"""
    width = max((len(row) for row in cells), default=1)
    widths: list[float] = []
    for col in range(width):
        longest_token = 0
        for row in cells:
            if col >= len(row):
                continue
            for token in str(row[col]).split():
                longest_token = max(longest_token, len(token))
        widths.append(max(30.0, longest_token * font_size * 0.58 + 6))
    total = sum(widths)
    if total > total_width:
        widths = [w * total_width / total for w in widths]
    return widths


def _build_table(grid: list[list[str]], rng: random.Random, settings: RenderSettings,
                 width: float) -> tuple[Table, list[dict]]:
    cell_style = ParagraphStyle(
        "cell", fontName="Helvetica", fontSize=settings.font_size,
        leading=settings.font_size + 1.6, splitLongWords=1,
    )
    numeric_style = ParagraphStyle("cell_right", parent=cell_style, alignment=TA_RIGHT)

    data: list[list] = []
    for row_idx, row in enumerate(grid):
        rendered_row: list = []
        for col_idx, cell in enumerate(row):
            text = _cell_text(cell, rng, settings)
            if not text:
                rendered_row.append("")
                continue
            style = numeric_style if parse_number(text) is not None and row_idx > 0 else cell_style
            escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            rendered_row.append(Paragraph(escaped, style))
        data.append(rendered_row)

    col_widths = _column_widths(grid, width, settings.font_size)
    table = Table(data, colWidths=col_widths, repeatRows=1, splitByRow=1)
    style_commands = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), settings.font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 if settings.whitespace else 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2 if settings.whitespace else 5),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if settings.whitespace:
        for row_idx in range(len(grid)):
            for col_idx in range(len(grid[row_idx])):
                if rng.random() < 0.15:
                    pad = rng.choice([3, 5, 7])
                    style_commands.append(
                        ("LEFTPADDING", (col_idx, row_idx), (col_idx, row_idx), pad)
                    )
    spans: list[dict] = []
    if settings.merged_cells:
        header_rows = min(infer_header_rows(grid), len(grid))
        for row_idx in range(header_rows):
            row = grid[row_idx]
            for col_idx in range(len(row) - 1, 0, -1):
                if not str(row[col_idx]).strip() and str(row[col_idx - 1]).strip():
                    style_commands.append(
                        ("SPAN", (col_idx - 1, row_idx), (col_idx, row_idx))
                    )
                    spans.append({"row": row_idx, "col": col_idx - 1, "colspan": 2})
                    break
    table.setStyle(TableStyle(style_commands))
    return table, spans


def render_context(
    ctx: Context,
    out_path: Path,
    settings: RenderSettings | None = None,
    caption: str = "Exhibit 99.1 — Financial Table",
) -> RenderResult:
    settings = settings or RenderSettings()
    rng = random.Random(f"{settings.seed}:{ctx.ctx_id}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title_style = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=10, leading=13,
                                 spaceAfter=4)
    body_style = ParagraphStyle("body", fontName="Helvetica", fontSize=8.5, leading=11)
    caption_text = pdf_safe(caption or "Financial Table")

    if settings.two_column and ctx.n_cols <= 5:
        doc = BaseDocTemplate(
            str(out_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=16 * mm, bottomMargin=16 * mm, title=caption_text,
        )
        gap = 6 * mm
        frame_width = (A4[0] - 36 * mm - gap) / 2
        frame_height = A4[1] - 32 * mm
        frames = [
            Frame(18 * mm, 16 * mm, frame_width, frame_height, id="left",
                  leftPadding=0, rightPadding=0),
            Frame(18 * mm + frame_width + gap, 16 * mm, frame_width, frame_height, id="right",
                  leftPadding=0, rightPadding=0),
        ]
        doc.addPageTemplates([PageTemplate(id="two_column", frames=frames)])
        table, spans = _build_table(ctx.cells, rng, settings, frame_width)
        story = [Paragraph(caption_text, title_style), Spacer(1, 4), table, FrameBreak()]
        for paragraph in ctx.paragraphs:
            story += [Paragraph(pdf_safe(paragraph["text"]), body_style), Spacer(1, 5)]
        layout = "two_column"
    else:
        doc = SimpleDocTemplate(
            str(out_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=16 * mm, bottomMargin=16 * mm, title=caption_text,
        )
        content_width = A4[0] - 36 * mm
        table, spans = _build_table(ctx.cells, rng, settings, content_width)
        story = [Paragraph(caption_text, title_style), Spacer(1, 4)]
        for paragraph in ctx.paragraphs[:1]:
            story += [Paragraph(pdf_safe(paragraph["text"]), body_style), Spacer(1, 6)]
        if settings.page_breaks and ctx.n_rows > 6:
            split_at = max(2, ctx.n_rows // 2)
            head = ctx.cells[:1]
            top_part = head + ctx.cells[1:split_at]
            bottom_part = head + ctx.cells[split_at:]
            top_table, top_spans = _build_table(top_part, rng, settings, content_width)
            bottom_table, bottom_spans = _build_table(
                bottom_part, rng, settings, content_width
            )
            story += [top_table, PageBreak(), Paragraph("(continued)", body_style),
                      bottom_table]
            spans = top_spans + bottom_spans
        else:
            story.append(table)
        story.append(Spacer(1, 8))
        for paragraph in ctx.paragraphs:
            story += [Paragraph(pdf_safe(paragraph["text"]), body_style), Spacer(1, 5)]
        layout = "single_column"

    doc.build(story)
    return RenderResult(
        path=out_path,
        meta={
            "layout": layout,
            "seed": settings.seed,
            "disturbances": {
                "page_breaks": bool(settings.page_breaks and ctx.n_rows > 6),
                "merged_cells": spans,
                "two_column": layout == "two_column",
                "whitespace": settings.whitespace,
            },
            "n_rows": ctx.n_rows,
            "n_cols": ctx.n_cols,
        },
    )


def render_paragraph_safe(text: str) -> str:
    return pdf_safe(text)
