"""解析层指标（SCORE-Bench 口径）：

- Cell Content Accuracy：单元格文字正确的内容占比（不看位置）
- Cell Index Accuracy：文字同时落对行/列的比例
拍平基线没有结构，index accuracy 恒为 0（这是它作为对照的意义）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from tablerag.domain.text import normalize_cell


@dataclass
class GridComparison:
    doc_id: str
    gold_shape: tuple[int, int]
    pred_shape: tuple[int, int]
    shape_match: bool
    n_gold_nonempty: int
    content_matches: int
    index_matches: int
    missing_cells: list[tuple[int, int, str]] = field(default_factory=list)
    extra_cells: list[tuple[int, int, str]] = field(default_factory=list)

    @property
    def content_accuracy(self) -> float:
        return self.content_matches / self.n_gold_nonempty if self.n_gold_nonempty else 0.0

    @property
    def index_accuracy(self) -> float:
        return self.index_matches / self.n_gold_nonempty if self.n_gold_nonempty else 0.0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "gold_shape": list(self.gold_shape),
            "pred_shape": list(self.pred_shape),
            "shape_match": self.shape_match,
            "n_gold_nonempty": self.n_gold_nonempty,
            "content_accuracy": round(self.content_accuracy, 4),
            "index_accuracy": round(self.index_accuracy, 4),
            "missing_sample": self.missing_cells[:5],
            "extra_sample": self.extra_cells[:5],
        }


def compare_grids(
    gold: list[list[str]], pred: list[list[str]], doc_id: str = "", sample: int = 20
) -> GridComparison:
    gold = [[normalize_cell(c) for c in row] for row in gold]
    pred = [[normalize_cell(c) for c in row] for row in pred]
    gold_shape = (len(gold), len(gold[0]) if gold else 0)
    pred_shape = (len(pred), len(pred[0]) if pred else 0)

    gold_nonempty = Counter(c for row in gold for c in row if c)
    pred_nonempty = Counter(c for row in pred for c in row if c)
    n_gold = sum(gold_nonempty.values())

    matched = 0
    remaining = Counter(pred_nonempty)
    for cell, count in gold_nonempty.items():
        used = min(count, remaining.get(cell, 0))
        matched += used
        remaining[cell] -= used
    remaining = +remaining

    index_matches = 0
    missing: list[tuple[int, int, str]] = []
    for row_idx in range(gold_shape[0]):
        for col_idx in range(gold_shape[1]):
            gold_cell = gold[row_idx][col_idx]
            if not gold_cell:
                continue
            pred_cell = (
                pred[row_idx][col_idx]
                if row_idx < pred_shape[0] and col_idx < pred_shape[1]
                else ""
            )
            if pred_cell == gold_cell:
                index_matches += 1
            elif len(missing) < sample:
                missing.append((row_idx, col_idx, gold_cell))

    extra: list[tuple[int, int, str]] = []
    for cell, count in remaining.items():
        if count > 0 and len(extra) < sample:
            extra.append((-1, -1, cell))

    return GridComparison(
        doc_id=doc_id,
        gold_shape=gold_shape,
        pred_shape=pred_shape,
        shape_match=gold_shape == pred_shape,
        n_gold_nonempty=n_gold,
        content_matches=matched,
        index_matches=index_matches,
        missing_cells=missing,
        extra_cells=extra,
    )


def flattened_content_accuracy(gold: list[list[str]], flat_text: str, doc_id: str = "") -> dict:
    """拍平基线：不做结构恢复，只能看"文字是否还在文本里"，不存在行列定位。"""
    flat = " ".join(normalize_cell(c) for c in flat_text.split())
    gold_cells = [normalize_cell(c) for row in gold for c in row if normalize_cell(c)]
    if not gold_cells:
        return {"doc_id": doc_id, "n_gold_nonempty": 0, "content_accuracy": 0.0,
                "index_accuracy": 0.0}
    hits = sum(1 for cell in gold_cells if cell in flat)
    return {
        "doc_id": doc_id,
        "n_gold_nonempty": len(gold_cells),
        "content_accuracy": round(hits / len(gold_cells), 4),
        "index_accuracy": 0.0,
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n_docs": 0, "content_accuracy": 0.0, "index_accuracy": 0.0,
                "shape_match_rate": 0.0}
    n = len(rows)
    return {
        "n_docs": n,
        "content_accuracy": round(sum(r["content_accuracy"] for r in rows) / n, 4),
        "index_accuracy": round(sum(r["index_accuracy"] for r in rows) / n, 4),
        "shape_match_rate": round(sum(1 for r in rows if r.get("shape_match")) / n, 4),
    }
