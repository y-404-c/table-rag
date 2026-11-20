"""TAT-QA 获取与加载（官方原始 JSON，含 2D 表格 + 段落 + 标注）。"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tablerag.domain.models import Paragraph, ParsedDoc, TableGrid, Unit
from tablerag.domain.text import flatten_text

logger = logging.getLogger(__name__)

RAW_FILES = {
    "train": "tatqa_dataset_train.json",
    "dev": "tatqa_dataset_dev.json",
    "test": "tatqa_dataset_test.json",
    "test_gold": "tatqa_dataset_test_gold.json",
}
MIRRORS = [
    "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/",
    "https://cdn.jsdelivr.net/gh/NExTplusplus/TAT-QA@master/dataset_raw/",
]


@dataclass
class GoldQuestion:
    qid: str
    question: str
    answers: list[str]
    answer_type: str
    answer_from: str
    rel_paragraphs: list[str] = field(default_factory=list)
    scale: str = ""
    derivation: str = ""
    req_comparison: bool = False

    def scaled_value(self) -> float | None:
        from tablerag.domain.text import parse_number, scale_factor

        for answer in self.answers:
            value = parse_number(answer)
            if value is not None:
                return value * scale_factor(self.scale)
        return None


@dataclass
class Context:
    ctx_id: str
    cells: list[list[str]]
    paragraphs: list[dict[str, Any]]
    questions: list[GoldQuestion]

    @property
    def n_rows(self) -> int:
        return len(self.cells)

    @property
    def n_cols(self) -> int:
        return len(self.cells[0]) if self.cells else 0


def download(raw_dir: Path, splits: Iterable[str] = ("train", "dev"), force: bool = False) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in splits:
        filename = RAW_FILES[split]
        target = raw_dir / filename
        if target.exists() and target.stat().st_size > 1000 and not force:
            paths[split] = target
            continue
        errors: list[str] = []
        for base in MIRRORS:
            url = base + filename
            try:
                logger.info("下载 %s", url)
                with urllib.request.urlopen(url, timeout=120) as resp:
                    data = resp.read()
                if len(data) < 1000:
                    raise ValueError("响应过短")
                target.write_bytes(data)
                paths[split] = target
                break
            except Exception as exc:  # noqa: BLE001 - 多镜像依次尝试
                errors.append(f"{url} -> {exc}")
        else:
            raise RuntimeError("TAT-QA 下载失败，请手动下载后放入 data/raw：\n" + "\n".join(errors))
    return paths


def _paragraph_list(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"uid": str(p.get("uid", "")), "order": int(p.get("order", idx + 1)),
         "text": str(p.get("text", ""))}
        for idx, p in enumerate(raw)
    ]


def _gold_question(raw: dict[str, Any]) -> GoldQuestion:
    answer = raw.get("answer", [])
    answers = [flatten_text(a) for a in answer] if isinstance(answer, list) else [flatten_text(answer)]
    rel = raw.get("rel_paragraphs") or []
    return GoldQuestion(
        qid=str(raw.get("uid", "")),
        question=str(raw.get("question", "")),
        answers=[a for a in answers if a],
        answer_type=str(raw.get("answer_type", "")),
        answer_from=str(raw.get("answer_from", "")),
        rel_paragraphs=[str(r) for r in rel],
        scale=str(raw.get("scale", "") or ""),
        derivation=str(raw.get("derivation", "") or ""),
        req_comparison=bool(raw.get("req_comparison", False)),
    )


def load_contexts(raw_dir: Path, splits: Iterable[str] = ("train", "dev")) -> list[Context]:
    contexts: list[Context] = []
    seen: set[str] = set()
    for split in splits:
        path = raw_dir / RAW_FILES[split]
        if not path.exists():
            raise FileNotFoundError(f"缺少数据文件 {path}，先运行 tr dataset fetch")
        entries = json.loads(path.read_text(encoding="utf-8"))
        for entry in entries:
            table = entry.get("table") or {}
            uid = str(table.get("uid") or "")
            cells = table.get("table") or []
            if not uid or not cells or not cells[0]:
                continue
            if uid in seen:
                continue
            seen.add(uid)
            width = len(cells[0])
            normalized = [
                [flatten_text(col) for col in row] + [""] * max(0, width - len(row))
                for row in cells
            ]
            contexts.append(
                Context(
                    ctx_id=uid,
                    cells=normalized,
                    paragraphs=_paragraph_list(entry.get("paragraphs") or []),
                    questions=[_gold_question(q) for q in entry.get("questions") or []],
                )
            )
    return contexts


def context_to_doc(ctx: Context, header_rows: int | None = None) -> ParsedDoc:
    """把 TAT-QA 的 2D array 直接转成"结构化真值文档"（解析层的对照基准）。"""
    doc_id = f"tatqa:{ctx.ctx_id}"
    table = TableGrid(
        table_id=f"{doc_id}:t0",
        doc_id=doc_id,
        title=f"TAT-QA table {ctx.ctx_id[:8]}",
        page=1,
        cells=ctx.cells,
        n_rows=ctx.n_rows,
        n_cols=ctx.n_cols,
        header_rows=header_rows or infer_header_rows(ctx.cells),
        meta={"source": "tatqa_gold", "ctx_id": ctx.ctx_id},
    )
    paragraphs = [
        Paragraph(
            pid=f"{doc_id}:p{p['order']}",
            doc_id=doc_id,
            page=1,
            order=p["order"],
            text=p["text"],
        )
        for p in ctx.paragraphs
    ]
    return ParsedDoc(
        doc_id=doc_id,
        source=f"tatqa:{ctx.ctx_id}",
        method="structured",
        n_pages=1,
        tables=[table],
        paragraphs=paragraphs,
        meta={"ctx_id": ctx.ctx_id},
    )


def infer_header_rows(cells: list[list[str]], max_header_rows: int = 3) -> int:
    """判断表格前几行是表头。

    TAT-QA 的财报表常见形态：第 0 行是跨列标题（多为空），第 1 行是年份。
    规则：首行空单元格过半、且下一行出现年份或短标签时，认为有 2 行表头。
    """
    from tablerag.domain.text import extract_years, normalize_cell

    if not cells:
        return 1
    first = [normalize_cell(c) for c in cells[0]]
    if len(cells) < 2:
        return 1
    second = [normalize_cell(c) for c in cells[1]]
    empty_ratio = sum(1 for c in first if not c) / max(1, len(first))
    years = len(extract_years(" ".join(second)))
    short_labels = sum(1 for c in second if c and len(c) <= 12)
    if (empty_ratio >= 0.5 and (years >= 2 or short_labels >= len(second) - 1)) or years >= 2:
        return min(2, max_header_rows, len(cells) - 1)
    return 1


def combine_header(cells: list[list[str]], header_rows: int) -> list[str]:
    """多行表头合并成单行标签（title / sub-label），用于行文本与列定位。"""
    if not cells:
        return []
    header_rows = max(1, min(header_rows, len(cells)))
    width = max(len(row) for row in cells[:header_rows])
    combined: list[str] = []
    for col in range(width):
        parts = [
            cells[row][col].strip()
            for row in range(header_rows)
            if col < len(cells[row]) and cells[row][col].strip()
        ]
        combined.append(" / ".join(parts))
    return combined


def build_units(doc: ParsedDoc) -> list[Unit]:
    """把表格按行、段落按段切成检索单元。表格行保留列名，便于数值/专名精确匹配。"""
    from tablerag.domain.text import make_row_text, normalize_cell

    units: list[Unit] = []
    for table in doc.tables:
        header = combine_header(table.cells, table.header_rows)
        for row_idx in range(table.header_rows, table.n_rows):
            row = table.cells[row_idx] if row_idx < len(table.cells) else []
            if not any(normalize_cell(c) for c in row):
                continue
            text = make_row_text(header, row, row_idx)
            units.append(
                Unit(
                    unit_id=f"{table.table_id}:r{row_idx}",
                    doc_id=doc.doc_id,
                    kind="table_row",
                    text=text,
                    page=table.page,
                    table_id=table.table_id,
                    row=row_idx,
                    col=None,
                    meta={"header": header, "table_title": table.title},
                )
            )
        summary = "table " + table.title + " | " + " | ".join(c for c in header if c)
        units.append(
            Unit(
                unit_id=f"{table.table_id}:summary",
                doc_id=doc.doc_id,
                kind="table_summary",
                text=summary,
                page=table.page,
                table_id=table.table_id,
                row=None,
                col=None,
                meta={"header": header},
            )
        )
    for paragraph in doc.paragraphs:
        units.append(
            Unit(
                unit_id=paragraph.pid,
                doc_id=doc.doc_id,
                kind="paragraph",
                text=paragraph.text,
                page=paragraph.page,
                table_id=None,
                row=None,
                col=None,
                meta={"order": paragraph.order},
            )
        )
    return units
