"""PDF 解析：pdfplumber 提取表格网格 + 表格外正文段落，恢复页码/行列。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pdfplumber

from tablerag.domain.models import Paragraph, ParsedDoc, TableGrid
from tablerag.domain.text import normalize_cell

logger = logging.getLogger(__name__)

TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "intersection_tolerance": 3,
}


def _clean_grid(raw: list[list[Any]]) -> list[list[str]]:
    width = max((len(row) for row in raw), default=0)
    grid = []
    for row in raw:
        cells = [normalize_cell(str(cell)) if cell is not None else "" for cell in row]
        cells += [""] * (width - len(cells))
        grid.append(cells)
    return grid


def _words_outside(page, bboxes: list[tuple[float, float, float, float]]) -> list[str]:
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    kept = []
    for word in words:
        cx = (word["x0"] + word["x1"]) / 2
        cy = (word["top"] + word["bottom"]) / 2
        inside = any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in bboxes)
        if not inside:
            kept.append(word)
    kept.sort(key=lambda w: (round(w["top"], 1), w["x0"]))
    lines: list[str] = []
    current: list[dict] = []
    last_top = None
    for word in kept:
        if last_top is None or abs(word["top"] - last_top) <= 3:
            current.append(word)
        else:
            lines.append(" ".join(w["text"] for w in current))
            current = [word]
        last_top = word["top"]
    if current:
        lines.append(" ".join(w["text"] for w in current))
    return [normalize_cell(line) for line in lines if len(normalize_cell(line)) > 3]


def parse_pdf(
    path: Path | str,
    doc_id: str,
    merge_continued: bool = True,
    header_rows: int = 1,
) -> ParsedDoc:
    path = Path(path)
    tables: list[TableGrid] = []
    paragraphs: list[Paragraph] = []
    paragraph_order = 0
    with pdfplumber.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            found = page.find_tables(TABLE_SETTINGS)
            bboxes: list[tuple[float, float, float, float]] = []
            for table_index, table in enumerate(found):
                raw = table.extract()
                grid = _clean_grid(raw or [])
                if not grid or not any(any(c for c in row) for row in grid):
                    continue
                bboxes.append(tuple(table.bbox))
                tables.append(
                    TableGrid(
                        table_id=f"{doc_id}:p{page_number}t{table_index}",
                        doc_id=doc_id,
                        title="",
                        page=page_number,
                        cells=grid,
                        n_rows=len(grid),
                        n_cols=len(grid[0]) if grid else 0,
                        header_rows=min(header_rows, len(grid)),
                        meta={"bbox": list(table.bbox)},
                    )
                )
            for line in _words_outside(page, bboxes):
                paragraph_order += 1
                paragraphs.append(
                    Paragraph(
                        pid=f"{doc_id}:p{page_number}l{paragraph_order}",
                        doc_id=doc_id,
                        page=page_number,
                        order=paragraph_order,
                        text=line,
                    )
                )

    if merge_continued:
        tables = _merge_continued_tables(tables)

    return ParsedDoc(
        doc_id=doc_id,
        source=str(path),
        method="structured",
        n_pages=len(set(t.page for t in tables) | {p.page for p in paragraphs}) or 1,
        tables=tables,
        paragraphs=paragraphs,
        meta={"parser": "pdfplumber", "table_settings": TABLE_SETTINGS,
              "merged_continued": merge_continued},
    )


def _merge_continued_tables(tables: list[TableGrid]) -> list[TableGrid]:
    """跨页断表恢复：相邻页、同列数、且新表首行等于上一表表头的，视为续页拼接。"""
    if len(tables) <= 1:
        return tables
    merged: list[TableGrid] = []
    for table in tables:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and table.n_cols == prev.n_cols
            and table.page is not None
            and prev.page is not None
            and table.page == prev.page + 1
            and table.cells
            and prev.cells
            and [normalize_cell(c) for c in table.cells[0]]
            == [normalize_cell(c) for c in prev.cells[0]]
        ):
            prev.cells.extend(table.cells[1:])
            prev.n_rows = len(prev.cells)
            prev.meta.setdefault("merged_pages", [prev.page]).append(table.page)
            continue
        merged.append(table)
    return merged
