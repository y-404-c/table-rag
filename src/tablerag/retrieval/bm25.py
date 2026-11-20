"""BM25 精确匹配路：列名 / 数字 / 专名在向量空间里不稳定，这一路负责兜住。"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from tablerag.domain.models import Unit
from tablerag.domain.text import tokenize


class BM25Index:
    def __init__(self, units: list[Unit]):
        self.units = units
        self.corpus = [tokenize(u.text) for u in units]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, top_k: int = 20, doc_id: str | None = None) -> list[tuple[Unit, float]]:
        if self.bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(self.units)), key=lambda i: scores[i], reverse=True)
        results: list[tuple[Unit, float]] = []
        for idx in ranked:
            unit = self.units[idx]
            if doc_id and unit.doc_id != doc_id:
                continue
            score = float(scores[idx])
            if score <= 0:
                break
            results.append((unit, score))
            if len(results) >= top_k:
                break
        return results

    def __len__(self) -> int:
        return len(self.units)
