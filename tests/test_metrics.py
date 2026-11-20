from tablerag.eval.parse_metrics import aggregate, compare_grids, flattened_content_accuracy
from tablerag.retrieval.rrf import rrf_fuse


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
