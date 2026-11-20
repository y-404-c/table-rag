from tablerag.domain.models import ParsedDoc, TableGrid
from tablerag.agent.tools import ToolError, dispatch, load_frame, run_table_op
from tablerag.ingest.dataset import build_units
from tablerag.storage.db import Store

CELLS = [
    ["", "2019", "2018"],
    ["Revenue", "8,657", "8,244"],
    ["Operating profit", "1,000", "900"],
    ["Percentage change", "5.0%", "3.2%"],
]


def _store(tmp_path) -> Store:
    store = Store(tmp_path / "t.db")
    table = TableGrid(table_id="doc:t0", doc_id="doc", page=2, cells=CELLS,
                      n_rows=len(CELLS), n_cols=3, header_rows=1)
    store.save_doc(ParsedDoc(doc_id="doc", tables=[table]), build_units(
        ParsedDoc(doc_id="doc", tables=[table])
    ))
    return store


def test_load_frame_dedupes_headers(tmp_path):
    store = _store(tmp_path)
    table = store.get_table("doc:t0")
    frame = load_frame(table)
    assert len(frame) == 3
    assert "__row__" in frame.columns


def test_growth_rate(tmp_path):
    store = _store(tmp_path)
    result = run_table_op(store, {"op": "growth_rate", "table_id": "doc:t0",
                                  "row": "Revenue", "from_col": "2018", "to_col": "2019"})
    assert round(result.value, 2) == round((8657 - 8244) / 8244 * 100, 2)
    assert len(result.inputs) == 2


def test_mean_row(tmp_path):
    store = _store(tmp_path)
    result = run_table_op(store, {"op": "mean_row", "table_id": "doc:t0", "row": "Revenue",
                                  "cols": ["2019", "2018"]})
    assert result.value == (8657 + 8244) / 2


def test_lookup_returns_cell_text(tmp_path):
    store = _store(tmp_path)
    result = run_table_op(store, {"op": "lookup", "table_id": "doc:t0",
                                  "row": "Operating profit", "col": "2019"})
    assert result.value == 1000
    assert result.inputs[0]["page"] == 2


def test_unknown_column_raises(tmp_path):
    store = _store(tmp_path)
    try:
        dispatch("table_calc", {"op": "lookup", "table_id": "doc:t0",
                                "row": "Revenue", "col": "1999"}, store)
    except ToolError as exc:
        assert "找不到列" in str(exc)
    else:
        raise AssertionError("应当抛出 ToolError")


def test_unknown_table_raises(tmp_path):
    store = _store(tmp_path)
    try:
        run_table_op(store, {"op": "lookup", "table_id": "nope", "row": 0, "col": 0})
    except ToolError:
        pass
    else:
        raise AssertionError("应当抛出 ToolError")
