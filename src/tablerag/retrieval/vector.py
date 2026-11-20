"""向量召回路：Chroma 持久化 + DashScope embedding。"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import chromadb

from tablerag.domain.models import Unit
from tablerag.llm.embeddings import Embedder

logger = logging.getLogger(__name__)

COLLECTION = "units"


class VectorIndex:
    def __init__(self, persist_dir: Path, embedder: Embedder):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedder = embedder
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        # Chroma 的底层是 SQLite，并发评测时串行化集合读写，避免 database is locked
        self._lock = threading.Lock()

    @staticmethod
    def _metadata(unit: Unit) -> dict:
        return {
            "doc_id": unit.doc_id,
            "kind": unit.kind,
            "page": -1 if unit.page is None else int(unit.page),
            "table_id": unit.table_id or "",
            "row": -1 if unit.row is None else int(unit.row),
            "col": -1 if unit.col is None else int(unit.col),
        }

    def upsert(self, units: list[Unit], batch_size: int = 64) -> int:
        total = 0
        for start in range(0, len(units), batch_size):
            batch = units[start:start + batch_size]
            vectors = self.embedder.embed([u.text for u in batch])
            with self._lock:
                self.collection.upsert(
                    ids=[u.unit_id for u in batch],
                    embeddings=vectors,
                    documents=[u.text for u in batch],
                    metadatas=[self._metadata(u) for u in batch],
                )
            total += len(batch)
        return total

    def search(self, query: str, top_k: int = 20, doc_id: str | None = None) -> list[tuple[str, float]]:
        vector = self.embedder.embed_one(query)
        kwargs: dict = {
            "query_embeddings": [vector],
            "n_results": max(1, top_k),
            "include": ["distances", "metadatas", "documents"],
        }
        if doc_id:
            kwargs["where"] = {"doc_id": doc_id}
        with self._lock:
            result = self.collection.query(**kwargs)
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        # cosine distance -> similarity
        return [(uid, 1.0 - float(dist)) for uid, dist in zip(ids, distances, strict=False)]

    def delete_doc(self, doc_id: str) -> None:
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma 删除失败 doc=%s: %s", doc_id, exc)

    def count(self) -> int:
        return int(self.collection.count())
