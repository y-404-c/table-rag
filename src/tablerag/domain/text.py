"""文本规范化、分词、数值解析、EM/F1。"""

from __future__ import annotations

import re
import string
import unicodedata

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"[-+]?\(?\$?\d[\d,]*(?:\.\d+)?\)?%?")
_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+(?:[.,][0-9]+)*%?")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

SCALE_FACTORS = {
    "": 1.0,
    "unit": 1.0,
    "units": 1.0,
    "ones": 1.0,
    "thousand": 1e3,
    "thousands": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "trillion": 1e12,
    "percent": 1.0,
    "percentage": 1.0,
    "%": 1.0,
}


def normalize_ws(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    return _WS_RE.sub(" ", text).strip()


def normalize_cell(text: str) -> str:
    """单元格文本规范化：全角转半角、去零宽、内部连续空白压成一个空格。

    渲染扰动会故意插入多余空白；如果比较时不做这一层规范化，"内容对不对"和
    "排版空格"会混在一起，指标失去意义。
    """
    return normalize_ws(text)


def _strip_punct(text: str) -> str:
    return "".join(ch for ch in text if ch not in set(string.punctuation))


def normalize_answer(text: str) -> str:
    """SQuAD 口径 + 数值写法统一。大小写/标点/冠词不敏感。"""
    text = normalize_ws(str(text))
    text = _ARTICLES_RE.sub(" ", text)
    text = _strip_punct(text)
    return normalize_ws(text.lower())


def parse_number(text: str) -> float | None:
    """从字符串中取第一个数值。支持 $1,200 / (1,200) / 12.5% / -3。"""
    if text is None:
        return None
    raw = normalize_ws(str(text))
    if not raw:
        return None
    match = _NUM_RE.search(raw)
    if not match:
        return None
    token = match.group(0)
    negative = token.strip().startswith("(") and token.strip().endswith(")")
    token = token.replace("$", "").replace(",", "").replace("%", "").strip("()")
    try:
        value = float(token)
    except ValueError:
        return None
    return -value if negative else value


def parse_scaled_number(text: str) -> float | None:
    """数值 + 量词（million/billion/percent）合并。"""
    value = parse_number(text)
    if value is None:
        return None
    lowered = normalize_ws(str(text)).lower()
    for word, factor in SCALE_FACTORS.items():
        if not word:
            continue
        if re.search(rf"\b{re.escape(word)}\b", lowered) or lowered.endswith("%"):
            return value * factor
    return value


def scale_factor(scale: str | None) -> float:
    return SCALE_FACTORS.get(normalize_ws(scale or "").lower(), 1.0)


def tokenize(text: str) -> list[str]:
    """BM25 分词：英文按词、数字整体保留（千分位/小数/百分号），并补充去标点形式。"""
    lowered = normalize_ws(text).lower()
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(lowered):
        tokens.append(token)
        bare = token.replace(",", "").rstrip("%")
        if bare != token:
            tokens.append(bare)
    return tokens


def extract_years(text: str) -> list[str]:
    return re.findall(r"\b(?:19|20)\d{2}\b", text)


def numbers_match(a: float, b: float, rel: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    if abs(a - b) <= abs_tol:
        return True
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


def exact_match(pred: str, gold: str) -> bool:
    if normalize_answer(pred) == normalize_answer(gold):
        return True
    pa, ga = parse_number(pred), parse_number(gold)
    if pa is not None and ga is not None:
        # EM 要严：只接受四舍五入级别的差异
        return numbers_match(pa, ga, rel=1e-6, abs_tol=1e-6)
    return False


def f1_score(pred: str, gold: str) -> float:
    pa, ga = parse_number(pred), parse_number(gold)
    if pa is not None and ga is not None:
        return 1.0 if numbers_match(pa, ga) else 0.0
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common: dict[str, int] = {}
    for token in gold_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_scores(pred: str, golds: list[str]) -> tuple[float, float]:
    """多标准答案取最优（TAT-QA 官方口径）。"""
    if not golds:
        return 0.0, 0.0
    em = max(exact_match(pred, g) for g in golds)
    f1 = max(f1_score(pred, g) for g in golds)
    return float(em), float(f1)


def flatten_text(value: object) -> str:
    if isinstance(value, list):
        return " / ".join(normalize_cell(str(v)) for v in value if str(v).strip())
    return normalize_cell(str(value if value is not None else ""))


def make_row_text(header: list[str], row: list[str], row_index: int) -> str:
    """表格行转检索文本：列名 + 值，保留行号便于溯源。"""
    parts: list[str] = []
    for idx, value in enumerate(row):
        value = normalize_cell(value)
        if not value:
            continue
        name = normalize_cell(header[idx]) if idx < len(header) and header[idx] else f"col{idx}"
        parts.append(f"{name}: {value}")
    return f"row{row_index} | " + " | ".join(parts)
