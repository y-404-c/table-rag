"""领域模型：解析结果、检索单元、证据、答案。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

UnitKind = Literal["table_row", "table_cell", "paragraph", "table_summary"]


class TableGrid(BaseModel):
    table_id: str
    doc_id: str
    title: str = ""
    page: int | None = None
    cells: list[list[str]]
    n_rows: int
    n_cols: int
    header_rows: int = 1
    meta: dict[str, Any] = Field(default_factory=dict)

    def cell(self, row: int, col: int) -> str:
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return self.cells[row][col]
        return ""


class Paragraph(BaseModel):
    pid: str
    doc_id: str
    page: int | None = None
    order: int = 0
    text: str = ""


class ParsedDoc(BaseModel):
    doc_id: str
    source: str = ""
    method: Literal["structured", "flatten"] = "structured"
    n_pages: int = 0
    tables: list[TableGrid] = Field(default_factory=list)
    paragraphs: list[Paragraph] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class Unit(BaseModel):
    """检索最小单元。表格按行切（保留列名），段落按段切。"""

    unit_id: str
    doc_id: str
    kind: UnitKind
    text: str
    page: int | None = None
    table_id: str | None = None
    row: int | None = None
    col: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class ScoredUnit(BaseModel):
    unit: Unit
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else (self.rrf_score or 0.0)


class Citation(BaseModel):
    doc_id: str
    table_id: str | None = None
    page: int | None = None
    row: int | None = None
    col: int | None = None
    text: str = ""


class ComputeResult(BaseModel):
    op: str
    value: float | None = None
    display: str = ""
    expression: str = ""
    inputs: list[dict[str, Any]] = Field(default_factory=list)

    def to_prompt(self) -> str:
        if self.display:
            return f"{self.expression} = {self.display}"
        return f"{self.expression} = {self.value}"


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refused_reason: str = ""
    used_compute: bool = False
    compute: ComputeResult | None = None
    route: str = ""
    trace_id: str = ""
    latency_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    hyde_rounds: int = 0
