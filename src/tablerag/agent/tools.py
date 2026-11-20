"""Agent 工具：表格检索引擎 + pandas 计算。

计算工具的输入是"表 + 行标签 + 列标签"，不是任意 pandas 表达式：
LLM 只能选操作和坐标，实际计算由代码执行、可审计、可复算。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tablerag.domain.models import ComputeResult, TableGrid
from tablerag.domain.text import normalize_cell, parse_number
from tablerag.ingest.dataset import combine_header
from tablerag.storage.db import Store


class ToolError(ValueError):
    pass


def load_frame(table: TableGrid) -> pd.DataFrame:
    header_rows = max(1, min(table.header_rows, table.n_rows))
    header = combine_header(table.cells, header_rows)
    rows = table.cells[header_rows:]
    width = table.n_cols
    header = (header + ["" for _ in range(len(header), width)])[:width]
    # 空表头 / 重复表头（"CHANGE" 出现两次很常见）必须去重，否则列定位会拿到 DataFrame
    seen: dict[str, int] = {}
    unique_header: list[str] = []
    for idx, name in enumerate(header):
        base = name or f"col{idx}"
        if base in seen:
            seen[base] += 1
            unique_header.append(f"{base}#{seen[base]}")
        else:
            seen[base] = 0
            unique_header.append(base)
    data = [row + [""] * (width - len(row)) for row in rows]
    frame = pd.DataFrame(data, columns=unique_header)
    frame.insert(0, "__row__", list(range(header_rows, header_rows + len(data))))
    return frame


def _resolve_col(frame: pd.DataFrame, col: Any) -> str:
    if isinstance(col, int):
        candidates = [c for c in frame.columns if c != "__row__"]
        if 0 <= col < len(candidates):
            return candidates[col]
        raise ToolError(f"列索引越界: {col}")
    needle = normalize_cell(str(col)).lower()
    for name in frame.columns:
        if name == "__row__":
            continue
        if normalize_cell(str(name)).lower() == needle:
            return name
    for name in frame.columns:
        if name != "__row__" and needle and needle in normalize_cell(str(name)).lower():
            return name
    raise ToolError(f"找不到列: {col}；可选列: {[c for c in frame.columns if c != '__row__']}")


def _resolve_row(frame: pd.DataFrame, row: Any) -> int:
    """返回 DataFrame 位置索引（iloc）。支持行号（原表行索引）与首列标签。"""
    if isinstance(row, int):
        if row in frame["__row__"].tolist():
            return int(frame.index[frame["__row__"] == row][0])
        if 0 <= row < len(frame):
            return row
        raise ToolError(f"行号越界: {row}，共 {len(frame)} 行")
    needle = normalize_cell(str(row)).lower()
    label_col = [c for c in frame.columns if c != "__row__"][0]
    for idx, value in enumerate(frame[label_col].tolist()):
        if needle and needle in normalize_cell(str(value)).lower():
            return idx
    raise ToolError(f"找不到行: {row}")


def _value(frame: pd.DataFrame, row_pos: int, col_name: str) -> float:
    raw = frame.iloc[row_pos][col_name]
    value = parse_number(str(raw))
    if value is None:
        raise ToolError(f"单元格不是数值: {raw!r}（列 {col_name}）")
    return value


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def run_table_op(store: Store, args: dict[str, Any]) -> ComputeResult:
    table_id = str(args.get("table_id", ""))
    table = store.get_table(table_id)
    if table is None:
        raise ToolError(f"表不存在: {table_id}")
    frame = load_frame(table)
    if frame.empty:
        raise ToolError(f"表为空: {table_id}")

    op = str(args.get("op", "")).strip().lower()
    inputs: list[dict[str, Any]] = []

    def cell(row_pos: int, col_name: str) -> tuple[float, dict]:
        value = _value(frame, row_pos, col_name)
        record = {
            "table_id": table.table_id,
            "row": int(frame.iloc[row_pos]["__row__"]),
            "col": list(frame.columns).index(col_name) - 1,
            "col_name": col_name,
            "value": value,
            "text": str(frame.iloc[row_pos][col_name]),
            "page": table.page,
        }
        inputs.append(record)
        return value, record

    if op in {"lookup", "value"}:
        row_pos = _resolve_row(frame, args.get("row"))
        col_name = _resolve_col(frame, args.get("col"))
        value, record = cell(row_pos, col_name)
        return ComputeResult(op=op, value=value, display=f"{record['text']}",
                             expression=f"{table.table_id}[{record['row']},{col_name}]",
                             inputs=inputs)

    if op in {"difference", "change"}:
        row_pos = _resolve_row(frame, args.get("row"))
        from_col = _resolve_col(frame, args.get("from_col"))
        to_col = _resolve_col(frame, args.get("to_col"))
        start, _ = cell(row_pos, from_col)
        end, _ = cell(row_pos, to_col)
        value = end - start
        return ComputeResult(op=op, value=value, display=_fmt(value),
                             expression=f"{to_col} - {from_col}", inputs=inputs)

    if op in {"growth_rate", "percentage_change", "pct_change"}:
        row_pos = _resolve_row(frame, args.get("row"))
        from_col = _resolve_col(frame, args.get("from_col"))
        to_col = _resolve_col(frame, args.get("to_col"))
        start, _ = cell(row_pos, from_col)
        end, _ = cell(row_pos, to_col)
        if start == 0:
            raise ToolError("基期为 0，增长率无定义")
        value = (end - start) / abs(start) * 100
        return ComputeResult(op=op, value=value, display=f"{value:.2f}%",
                             expression=f"({_fmt(end)} - {_fmt(start)}) / |{_fmt(start)}| × 100%",
                             inputs=inputs)

    if op in {"ratio", "divide"}:
        num_row = _resolve_row(frame, args.get("row"))
        num_col = _resolve_col(frame, args.get("col"))
        den_row = _resolve_row(frame, args.get("den_row", args.get("row")))
        den_col = _resolve_col(frame, args.get("den_col"))
        num, _ = cell(num_row, num_col)
        den, _ = cell(den_row, den_col)
        if den == 0:
            raise ToolError("分母为 0")
        value = num / den
        return ComputeResult(op=op, value=value, display=_fmt(value),
                             expression=f"{_fmt(num)} / {_fmt(den)}", inputs=inputs)

    if op in {"mean_row", "average_row", "mean_cells"}:
        row_pos = _resolve_row(frame, args.get("row"))
        cols = args.get("cols") or args.get("col")
        if isinstance(cols, str):
            cols = [cols]
        if not isinstance(cols, list) or not cols:
            raise ToolError("mean_row 需要 cols 列表")
        values = []
        for col in cols:
            value, _ = cell(row_pos, _resolve_col(frame, col))
            values.append(value)
        series = pd.Series(values, dtype="float64")
        value = float(series.mean())
        return ComputeResult(op="mean_row", value=value, display=_fmt(value),
                             expression="(" + " + ".join(_fmt(v) for v in values) + f") / {len(values)}",
                             inputs=inputs)

    if op in {"sum", "total", "mean", "average"}:
        col_name = _resolve_col(frame, args.get("col"))
        rows = args.get("rows") or args.get("row")
        positions = (
            [_resolve_row(frame, r) for r in rows]
            if isinstance(rows, list)
            else list(range(len(frame)))
        )
        values = []
        for pos in positions:
            value, _ = cell(pos, col_name)
            values.append(value)
        series = pd.Series(values, dtype="float64")
        value = float(series.sum() if op in {"sum", "total"} else series.mean())
        agg = "sum" if op in {"sum", "total"} else "mean"
        return ComputeResult(op=agg, value=value, display=_fmt(value),
                             expression=f"pandas.Series.sum/mean({col_name})", inputs=inputs)

    if op in {"max_row", "min_row", "argmax", "argmin"}:
        col_name = _resolve_col(frame, args.get("col"))
        best_pos = None
        best_value = None
        for pos in range(len(frame)):
            value = parse_number(str(frame.iloc[pos][col_name]))
            if value is None:
                continue
            if best_value is None or (value > best_value if op in {"max_row", "argmax"} else value < best_value):
                best_value, best_pos = value, pos
        if best_pos is None:
            raise ToolError(f"列 {col_name} 没有可比较的数值")
        label_col = [c for c in frame.columns if c != "__row__"][0]
        label = str(frame.iloc[best_pos][label_col])
        record = {
            "table_id": table.table_id,
            "row": int(frame.iloc[best_pos]["__row__"]),
            "col": list(frame.columns).index(col_name) - 1,
            "col_name": col_name,
            "value": best_value,
            "text": str(frame.iloc[best_pos][col_name]),
            "page": table.page,
        }
        inputs.append(record)
        direction = "最大" if op in {"max_row", "argmax"} else "最小"
        return ComputeResult(op=op, value=best_value, display=label,
                             expression=f"{direction}值所在行：{label} ({_fmt(best_value)})",
                             inputs=inputs)

    raise ToolError(f"不支持的操作: {op}")


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TABLE_CALC_SPEC = _schema(
    "table_calc",
    "在指定表格上执行一次可审计的数值计算。必须给出 table_id，以及行/列标签（用表头文字，"
    "不要用 LLM 心算）。",
    {
        "op": {
            "type": "string",
            "enum": ["lookup", "difference", "growth_rate", "ratio", "sum", "mean",
                     "mean_row", "max_row", "min_row"],
        },
        "table_id": {"type": "string"},
        "row": {"type": "string", "description": "行标签（首列文字）"},
        "col": {"type": "string", "description": "列标签（表头文字）"},
        "cols": {"type": "array", "items": {"type": "string"},
                 "description": "mean_row 用：同一行要平均的多列标签"},
        "from_col": {"type": "string"},
        "to_col": {"type": "string"},
        "den_row": {"type": "string"},
        "den_col": {"type": "string"},
        "rows": {"type": "array", "items": {"type": "string"}},
    },
    ["op", "table_id"],
)


def tool_specs() -> list[dict]:
    return [TABLE_CALC_SPEC]


def dispatch(name: str, args: dict[str, Any], store: Store) -> ComputeResult:
    if name != "table_calc":
        raise ToolError(f"未知工具: {name}")
    return run_table_op(store, args)
