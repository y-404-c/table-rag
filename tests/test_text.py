from tablerag.domain.text import (
    best_scores,
    exact_match,
    extract_years,
    f1_score,
    make_row_text,
    normalize_cell,
    parse_number,
    parse_scaled_number,
    tokenize,
)
from tablerag.ingest.dataset import combine_header, infer_header_rows


def test_normalize_cell_collapses_whitespace():
    assert normalize_cell("  Net\u00a0 income \n 2020 ") == "Net income 2020"


def test_parse_number_handles_currency_and_parentheses():
    assert parse_number("$1,200") == 1200
    assert parse_number("(1,200)") == -1200
    assert parse_number("12.5%") == 12.5
    assert parse_number("n/a") is None


def test_parse_scaled_number():
    assert parse_scaled_number("12.5 million") == 12_500_000


def test_exact_match_numeric_tolerance():
    assert exact_match("1,200", "$1200")
    assert exact_match("18,384.50", "18384.5")
    # EM 严格：0.1% 的数值差异不放过（F1 才给部分分）
    assert not exact_match("1,200", "1201")


def test_f1_partial_overlap():
    assert f1_score("total revenue", "total revenue was") > 0
    assert f1_score("net income", "gross profit") == 0


def test_best_scores_takes_best_of_gold_list():
    em, f1 = best_scores("5.0%", ["5%", "4.9%"])
    assert em == 1.0 and f1 == 1.0


def test_tokenize_keeps_numeric_tokens():
    tokens = tokenize("Revenue grew to $1,200 in 2019")
    assert "1,200" in tokens and "1200" in tokens and "2019" in tokens


def test_extract_years():
    assert extract_years("from 2018 to 2019") == ["2018", "2019"]


def test_make_row_text_includes_header_names():
    text = make_row_text(["", "2019", "2018"], ["Sales", "100", "90"], 3)
    assert text.startswith("row3")
    assert "2019: 100" in text


def test_infer_header_rows_for_year_row():
    cells = [["", "", "Years Ended"], ["", "2019", "2018"], ["Sales", "100", "90"]]
    assert infer_header_rows(cells) == 2
    combined = combine_header(cells, 2)
    assert combined[1] == "2019"
