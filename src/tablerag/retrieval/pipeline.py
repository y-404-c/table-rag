"""检索管线：BM25 + 向量 → RRF → CrossEncoder 精排，每层可单独关闭（消融）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tablerag.config import Settings
from tablerag.domain.models import ParsedDoc, ScoredUnit, Unit
from tablerag.ingest.dataset import build_units
from tablerag.llm.embeddings import Embedder
from tablerag.llm.rerank import Reranker
from tablerag.retrieval.bm25 import BM25Index
from tablerag.retrieval.rrf import rrf_fuse
from tablerag.retrieval.vector import VectorIndex
from tablerag.storage.db import Store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalFlags:
    bm25: bool = True
    vector: bool = True
    rerank: bool = True


@dataclass
class RetrievalTrace:
    bm25_hits: int = 0
    vector_hits: int = 0
    fused: int = 0
    reranked: int = 0
    rerank_used: bool = False
    notes: list[str] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, settings: Settings, store: Store, embedder: Embedder,
                 reranker: Reranker):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.vector = VectorIndex(settings.chroma_dir, embedder)
        self._bm25: BM25Index | None = None
        self._unit_map: dict[str, Unit] = {}
        self.default_flags = RetrievalFlags()

    # ---------- 索引 ----------
    def _refresh_bm25(self) -> None:
        units = self.store.get_units()
        self._bm25 = BM25Index(units)
        self._unit_map = {u.unit_id: u for u in units}

    def index_docs(self, docs: list[ParsedDoc], embed: bool = True) -> dict[str, int]:
        units: list[Unit] = []
        for doc in docs:
            doc_units = build_units(doc)
            units.extend(doc_units)
            self.store.save_doc(doc, doc_units)
        indexed = 0
        if embed and units:
            indexed = self.vector.upsert(units)
        self._bm25 = None
        return {"docs": len(docs), "units": len(units), "embedded": indexed}

    def ensure_loaded(self) -> None:
        if self._bm25 is None:
            self._refresh_bm25()

    def reset(self) -> None:
        try:
            self.vector.client.delete_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001
            pass
        self.vector = VectorIndex(self.settings.chroma_dir, self.embedder)
        self._bm25 = None

    # ---------- 检索 ----------
    def search(
        self,
        query: str,
        doc_id: str | None = None,
        top_k: int | None = None,
        flags: RetrievalFlags | None = None,
    ) -> tuple[list[ScoredUnit], RetrievalTrace]:
        self.ensure_loaded()
        flags = flags or self.default_flags
        cfg = self.settings.retrieval
        top_k = top_k or cfg.final_top_k
        trace = RetrievalTrace()
        bm25_hits: list[tuple[Unit, float]] = []
        vector_hits: list[tuple[str, float]] = []

        if flags.bm25 and self._bm25 is not None:
            bm25_hits = self._bm25.search(query, cfg.bm25_top_k, doc_id=doc_id)
        if flags.vector:
            try:
                vector_hits = self.vector.search(query, cfg.vector_top_k, doc_id=doc_id)
            except Exception as exc:  # noqa: BLE001 - 向量路失败不能让整条链路挂掉
                trace.notes.append(f"vector_failed: {exc}")
                logger.warning("向量检索失败：%s", exc)

        trace.bm25_hits = len(bm25_hits)
        trace.vector_hits = len(vector_hits)
        if not bm25_hits and not vector_hits:
            return [], trace

        fused = rrf_fuse(
            [[unit.unit_id for unit, _ in bm25_hits], [uid for uid, _ in vector_hits]],
            k=cfg.rrf_k,
        )
        trace.fused = len(fused)

        bm25_scores = {unit.unit_id: score for unit, score in bm25_hits}
        vector_scores = dict(vector_hits)
        candidates: list[ScoredUnit] = []
        for unit_id, rrf_score in fused:
            unit = self._unit_map.get(unit_id) or self.store_unit(unit_id)
            if unit is None:
                continue
            candidates.append(
                ScoredUnit(
                    unit=unit,
                    bm25_score=bm25_scores.get(unit_id),
                    vector_score=vector_scores.get(unit_id),
                    rrf_score=rrf_score,
                )
            )

        if flags.rerank and self.reranker.enabled and candidates:
            head = candidates[: min(len(candidates), max(top_k * 3, 20))]
            scores = self.reranker.score(query, [c.unit.text for c in head])
            if scores is not None:
                trace.rerank_used = True
                for candidate, score in zip(head, scores, strict=False):
                    candidate.rerank_score = score
                candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

        trace.reranked = min(len(candidates), top_k)
        return candidates[:top_k], trace

    def store_unit(self, unit_id: str) -> Unit | None:
        """映射里没有说明索引是旧的：重建一次再查，避免每条候选都全表扫描。"""
        unit = self._unit_map.get(unit_id)
        if unit is not None:
            return unit
        self._refresh_bm25()
        return self._unit_map.get(unit_id)

    def stats(self) -> dict:
        return {
            "units": self.store.count_units(),
            "chroma": self.vector.count(),
            "bm25": len(self._bm25) if self._bm25 else 0,
            "rerank": self.reranker.stats,
        }


COLLECTION_NAME = "units"


def build_retriever(settings: Settings, store: Store, cache=None) -> HybridRetriever:
    embedder = Embedder(settings.embedding, cache=cache)
    reranker = Reranker(settings.rerank)
    return HybridRetriever(settings, store, embedder, reranker)
