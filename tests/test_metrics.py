from tablerag.domain.models import Citation, ParsedDoc, TableGrid
from tablerag.eval.parse_metrics import aggregate, compare_grids, flattened_content_accuracy
from tablerag.eval.qa_metrics import cell_citation_hit, citation_hit
from tablerag.ingest.dataset import build_units
from tablerag.retrieval.rrf import rrf_fuse
from tablerag.storage.db import Store


def test_rrf_prefers_documents_ranked_high_in_both_lists():
    fused = rrf_fuse([["a", "b", "c"], ["a", "b", "d"]], k=60)
    assert [item[0] for item in fused][:2] == ["a", "b"]
    assert fused[0][1] > fused[1][1]


def test_rrf_breaks_ties_by_first_seen():
    fused = rrf_fuse([["a", "b"], ["b", "a"]], k=60)
    assert set(item[0] for item in fused[:2]) == {"a", "b"}


def test_rrf_scores_are_monotonic():
    fused = rrf_fuse([["a", "b"], ["a"]], k=60)
    assert fused[0][1] > fused[1][1]


def test_compare_grids_exact():
    grid = [["", "2019"], ["Sales", "100"]]
    result = compare_grids(grid, [row[:] for row in grid])
    assert result.content_accuracy == 1.0
    assert result.index_accuracy == 1.0
    assert result.shape_match


def test_compare_grids_detects_shifted_cells():
    gold = [["", "2019"], ["Sales", "100"]]
    pred = [["", "2019"], ["Sales", "101"]]
    result = compare_grids(gold, pred)
    assert result.content_accuracy < 1.0
    assert result.index_accuracy < 1.0
    assert result.n_gold_nonempty == 3


def test_flatten_baseline_has_no_index_accuracy():
    gold = [["", "2019"], ["Sales", "100"]]
    result = flattened_content_accuracy(gold, "Sales 100 2019")
    assert result["content_accuracy"] == 1.0
    assert result["index_accuracy"] == 0.0


def test_aggregate_handles_empty():
    assert aggregate([])["n_docs"] == 0


def _store_with_table(tmp_path):
    store = Store(tmp_path / "c.db")
    cells = [["", "2019"], ["Revenue", "100"]]
    doc = ParsedDoc(
        doc_id="doc",
        tables=[TableGrid(table_id="doc:t0", doc_id="doc", cells=cells, n_rows=2, n_cols=2)],
    )
    store.save_doc(doc, build_units(doc))
    return store


def test_cell_citation_hit_requires_full_coordinates(tmp_path):
    store = _store_with_table(tmp_path)
    # 行级出处（col=None）：文本口径命中，但 cell 口径不命中——这正是要分开报告的原因
    row_cite = Citation(doc_id="doc", table_id="doc:t0", page=1, row=1, col=None, text="Revenue 100")
    assert citation_hit([row_cite], ["100"], store) is True
    assert cell_citation_hit([row_cite], ["100"], store) is False

    # 真正定位到单元格（row+col 齐全）且值匹配 → cell 口径命中
    cell_cite = Citation(doc_id="doc", table_id="doc:t0", page=1, row=1, col=1, text="100")
    assert cell_citation_hit([cell_cite], ["100"], store) is True

    # 坐标齐全但值不匹配 → 不命中
    wrong = Citation(doc_id="doc", table_id="doc:t0", page=1, row=1, col=0, text="Revenue")
    assert cell_citation_hit([wrong], ["100"], store) is False


def test_cell_citation_hit_without_store_is_false():
    cite = Citation(doc_id="doc", table_id="doc:t0", page=1, row=1, col=1, text="100")
    assert cell_citation_hit([cite], ["100"], store=None) is False
