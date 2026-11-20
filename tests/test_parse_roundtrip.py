from tablerag.config import RenderSettings
from tablerag.eval.parse_metrics import compare_grids
from tablerag.ingest.dataset import Context
from tablerag.ingest.parser import parse_pdf
from tablerag.ingest.render import render_context


def _context(cells: list[list[str]]) -> Context:
    return Context(ctx_id="roundtrip", cells=cells, paragraphs=[], questions=[])


def test_render_then_parse_recovers_grid(tmp_path):
    cells = [
        ["", "2019", "2018"],
        ["Revenue", "1,234", "1,100"],
        ["Cost of sales", "800", "760"],
        ["Gross profit", "434", "340"],
    ]
    pdf_path = tmp_path / "roundtrip.pdf"
    render_context(
        _context(cells),
        pdf_path,
        RenderSettings(page_breaks=False, merged_cells=False, two_column=False, whitespace=True),
    )
    parsed = parse_pdf(pdf_path, doc_id="parse:roundtrip")
    assert parsed.tables, "解析器没有恢复出任何表格"
    best = max(parsed.tables, key=lambda t: t.n_rows * t.n_cols)
    comparison = compare_grids(cells, best.cells, doc_id="roundtrip")
    assert comparison.content_accuracy >= 0.99
    assert comparison.index_accuracy >= 0.99
    assert comparison.shape_match


def test_render_with_disturbances_still_parses_core_cells(tmp_path):
    cells = [
        ["", "2019", "2018"],
        ["Revenue", "1,234", "1,100"],
        ["Cost of sales", "800", "760"],
    ]
    pdf_path = tmp_path / "disturbed.pdf"
    render_context(_context(cells), pdf_path, RenderSettings())
    parsed = parse_pdf(pdf_path, doc_id="parse:disturbed")
    assert parsed.tables
    best = max(parsed.tables, key=lambda t: t.n_rows * t.n_cols)
    comparison = compare_grids(cells, best.cells, doc_id="disturbed")
    # 有扰动时不要求完全一致，但内容恢复率必须明显高于随机
    assert comparison.content_accuracy >= 0.8
