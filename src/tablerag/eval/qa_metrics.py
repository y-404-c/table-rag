"""问答层指标：EM / F1（TAT-QA 官方口径的多标准答案取最优）+ 出处命中 + 拒答。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tablerag.domain.models import Citation
from tablerag.domain.text import (
    best_scores,
    normalize_cell,
    numbers_match,
    parse_number,
    scale_factor,
)
from tablerag.storage.db import Store


def qa_scores(prediction: str, golds: list[str], scale: str = "") -> dict[str, float]:
    em, f1 = best_scores(prediction, golds)
    return {"em": em, "f1": f1, "scale": scale_factor(scale)}


def citation_resolves(
    citation: Citation, store: Store | None = None, row_text: str | None = None
) -> str:
    """把出处还原成文字：优先 cell（行+列），其次整行，最后 citation.text。"""
    if store is not None and citation.table_id:
        table = store.get_table(citation.table_id)
        if table is not None:
            if citation.row is not None:
                if citation.col is not None:
                    return table.cell(citation.row, citation.col)
                if 0 <= citation.row < table.n_rows:
                    return " | ".join(c for c in table.cells[citation.row] if c)
    if row_text:
        return row_text
    return citation.text


def citation_hit(
    citations: list[Citation],
    golds: Iterable[str],
    store: Store | None = None,
) -> bool:
    """文本级出处命中（宽口径，报告里写清）：

    任一条出处还原出的文字包含标准答案原文（大小写/标点不敏感），
    或出处文字里的数值与标准答案数值在 0.1% 相对误差内，即视为命中。
    注意：这只证明"出处文字里能找到答案"，不证明"定位到了单元格"——
    单元格级命中请看 cell_citation_hit。
    """
    gold_list = [normalize_cell(g) for g in golds if normalize_cell(g)]
    if not gold_list or not citations:
        return False
    gold_numbers = [parse_number(g) for g in gold_list]
    for citation in citations:
        resolved = citation_resolves(citation, store)
        text = normalize_cell(resolved).lower()
        for gold in gold_list:
            if gold.lower() in text:
                return True
        resolved_number = parse_number(resolved)
        if resolved_number is not None:
            for gold_number in gold_numbers:
                if gold_number is not None and numbers_match(resolved_number, gold_number):
                    return True
    return False


def cell_citation_hit(
    citations: list[Citation],
    golds: Iterable[str],
    store: Store | None = None,
) -> bool:
    """严格 cell 级命中（验收口径：数值题答案必须能溯源到具体单元格）：

    必须同时满足：
    1. 出处有 table_id 且 row / col 都不为 None（真正定位到单元格，不是行/段）；
    2. 该单元格的值（从 store 实取）命中某个 gold：文本包含或数值 0.1% 误差内。
    行级出处（col=None）、段落出处（table_id=None）一律不计。
    """
    gold_list = [normalize_cell(g) for g in golds if normalize_cell(g)]
    if not gold_list or not citations or store is None:
        return False
    gold_numbers = [parse_number(g) for g in gold_list]
    for citation in citations:
        if not (citation.table_id and citation.row is not None and citation.col is not None):
            continue
        table = store.get_table(citation.table_id)
        if table is None:
            continue
        cell_text = table.cell(citation.row, citation.col)
        normalized = normalize_cell(cell_text).lower()
        for gold in gold_list:
            if gold.lower() in normalized or normalized in gold.lower():
                return True
        cell_number = parse_number(cell_text)
        if cell_number is not None:
            for gold_number in gold_numbers:
                if gold_number is not None and numbers_match(cell_number, gold_number):
                    return True
    return False


def percentile(values: Iterable[float], p: float) -> float:
    """线性插值分位数；空集合返回 0。"""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * max(0.0, min(1.0, p))
    low, high = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def aggregate_qa(
    rows: list[dict[str, Any]],
    latencies: Iterable[float] | None = None,
    tokens: Iterable[float] | None = None,
) -> dict[str, Any]:
    answered = [r for r in rows if not r.get("refused")]
    n = len(rows)
    if n == 0:
        return {"n": 0, "em": 0.0, "f1": 0.0, "em_std": 0.0, "f1_std": 0.0,
                "citation_hit": 0.0, "cell_citation_hit": 0.0, "refusal_rate": 0.0,
                "avg_latency_ms": 0, "p50_latency_ms": 0, "p95_latency_ms": 0,
                "avg_tokens": 0, "total_tokens": 0}
    lat = [float(v) for v in (latencies if latencies is not None else
                              [r.get("latency_ms", 0) for r in rows])]
    tok = [float(v) for v in (tokens if tokens is not None else
                              [r.get("total_tokens", 0) for r in rows])]
    return {
        "n": n,
        "em": round(sum(r["em"] for r in rows) / n, 4),
        "f1": round(sum(r["f1"] for r in rows) / n, 4),
        "em_std": round(sum(r.get("em_std", 0.0) for r in rows) / n, 4),
        "f1_std": round(sum(r.get("f1_std", 0.0) for r in rows) / n, 4),
        "citation_hit": round(
            sum(float(r.get("citation_hit") or 0) for r in answered) / max(1, len(answered)), 4
        ),
        "cell_citation_hit": round(
            sum(float(r.get("cell_citation_hit") or 0) for r in answered) / max(1, len(answered)), 4
        ),
        "refusal_rate": round(sum(1 for r in rows if r.get("refused")) / n, 4),
        "avg_latency_ms": int(sum(lat) / len(lat)) if lat else 0,
        "p50_latency_ms": int(percentile(lat, 0.50)),
        "p95_latency_ms": int(percentile(lat, 0.95)),
        "avg_tokens": int(sum(tok) / len(tok)) if tok else 0,
        "total_tokens": int(sum(tok)),
    }


def aggregate_refusal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "refusal_rate": 0.0, "hallucination_rate": 0.0}
    refused = sum(1 for r in rows if r.get("refused"))
    return {
        "n": n,
        "refusal_rate": round(refused / n, 4),
        "hallucination_rate": round(1 - refused / n, 4),
    }
