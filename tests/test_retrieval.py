from tablerag.config import RerankSettings, Settings
from tablerag.domain.models import Paragraph, ParsedDoc, TableGrid
from tablerag.llm.rerank import Reranker
from tablerag.retrieval.bm25 import BM25Index
from tablerag.retrieval.pipeline import HybridRetriever, RetrievalFlags
from tablerag.retrieval.vector import VectorIndex
from tablerag.storage.db import Store
from tablerag.ingest.dataset import build_units


def _doc(doc_id: str, rows: list[list[str]], paragraph: str = "") -> ParsedDoc:
    table = TableGrid(
        table_id=f"{doc_id}:t1", doc_id=doc_id, title="Financials", page=1,
        cells=rows, n_rows=len(rows), n_cols=len(rows[0]), header_rows=1,
    )
    paragraphs = (
        [Paragraph(pid=f"{doc_id}:p1", doc_id=doc_id, page=1, order=0, text=paragraph)]
        if paragraph else []
    )
    return ParsedDoc(doc_id=doc_id, source="unit-test", n_pages=1,
                     tables=[table], paragraphs=paragraphs)


class FakeEmbedder:
    """三维修空间：在 revenue / profit / cost 上做词频，完全离线。"""

    @staticmethod
    def _vec(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("revenue")),
            float(lowered.count("profit")),
            float(lowered.count("cost")) + 0.01,
        ]

    def embed(self, texts):
        return [self._vec(text) for text in texts]

    def embed_one(self, text):
        return self._vec(text)


def test_bm25_ranks_exact_numeric_row_first():
    units = build_units(_doc("d1", [["", "2019"], ["Revenue", "1,234"], ["Cost", "800"]]))
    index = BM25Index(units)
    hits = index.search("revenue in 2019", top_k=3)
    assert hits
    assert "revenue" in hits[0][0].text.lower()


def test_bm25_respects_doc_filter():
    units = build_units(_doc("d1", [["", "2019"], ["Revenue", "1,234"]]))
    units += build_units(_doc("d2", [["", "2019"], ["Revenue", "999"]]))
    units += build_units(_doc("d3", [["", "2019"], ["Cost", "800"]]))
    index = BM25Index(units)
    hits = index.search("revenue", top_k=10, doc_id="d2")
    assert hits and all(unit.doc_id == "d2" for unit, _ in hits)


def test_vector_index_filters_by_doc(tmp_path):
    index = VectorIndex(tmp_path / "chroma", FakeEmbedder())
    units = build_units(_doc("d1", [["", "2019"], ["Revenue", "1,234"]]))
    units += build_units(_doc("d2", [["", "2019"], ["Profit", "50"]]))
    index.upsert(units)
    hits = index.search("revenue", top_k=5, doc_id="d1")
    assert hits
    all_units = {unit.unit_id: unit for unit in units}
    assert all(all_units[uid].doc_id == "d1" for uid, _ in hits)


def test_hybrid_retriever_supports_layer_ablation(tmp_path):
    settings = Settings(data_dir=tmp_path, var_dir=tmp_path)
    store = Store(tmp_path / "store.sqlite")
    retriever = HybridRetriever(
        settings, store, FakeEmbedder(), Reranker(RerankSettings(enabled=False))
    )
    retriever.index_docs([
        _doc("d1", [["", "2019"], ["Revenue", "1,234"]], paragraph="Revenue grew in 2019."),
        _doc("d2", [["", "2019"], ["Cost", "800"]], paragraph="Cost of sales declined."),
    ])

    structured, trace = retriever.search(
        "revenue 2019", doc_id="d1", flags=RetrievalFlags(bm25=True, vector=True, rerank=False)
    )
    assert structured
    assert trace.bm25_hits > 0 and trace.vector_hits > 0
    assert all(item.unit.doc_id == "d1" for item in structured)

    bm25_only, _ = retriever.search(
        "revenue", doc_id="d1", flags=RetrievalFlags(bm25=True, vector=False, rerank=False)
    )
    assert bm25_only and all(item.vector_score is None for item in bm25_only)

    vector_only, _ = retriever.search(
        "revenue", doc_id="d1", flags=RetrievalFlags(bm25=False, vector=True, rerank=False)
    )
    assert vector_only and all(item.bm25_score is None for item in vector_only)
