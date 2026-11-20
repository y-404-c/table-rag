"""多轮对话演示：从 test 集自动挑文档，构造「那 X 年呢」式追问，真实跑 Checkpointer。

选择规则（不手写题目，避免自造数据）：
- 同一文档内有两道问题，去掉年份后模板完全一致、年份不同（如 2018 / 2019）；
- 第 1 轮问较晚年份，记下答案；
- 第 2 轮只问 "And what about <较早年份>?"（主语、指标名全部省略）；
- 第 3 轮问 "What is the difference between the two values?"（无 gold，检验是否复用上文）。
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, Sequence

from tablerag.config import Settings
from tablerag.eval.runner import _dump
from tablerag.ingest.dataset import Context
from tablerag.service import AppContext

logger = logging.getLogger(__name__)

YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def _template(question: str) -> tuple[str, str] | None:
    years = YEAR_RE.findall(question)
    if len(years) != 1:
        return None
    return YEAR_RE.sub("{Y}", question).strip().lower(), years[0]


def _find_pairs(ctx: Context) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for question in ctx.questions:
        parsed = _template(question.question)
        if parsed is None:
            continue
        template, year = parsed
        groups[template].append((year, question))
    pairs = []
    for template, items in groups.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda item: item[0])
        earlier, later = items[0], items[-1]
        if earlier[0] == later[0]:
            continue
        pairs.append(
            {
                "template": template,
                "earlier_year": earlier[0],
                "later_year": later[0],
                "earlier": earlier[1],
                "later": later[1],
            }
        )
    return pairs


def pick_demos(contexts_by_id: dict[str, Context], ctx_ids: Sequence[str],
               n: int = 3) -> list[dict[str, Any]]:
    demos: list[dict[str, Any]] = []
    for ctx_id in ctx_ids:
        ctx = contexts_by_id.get(ctx_id)
        if ctx is None or not ctx.cells:
            continue
        pairs = _find_pairs(ctx)
        if not pairs:
            continue
        pair = max(pairs, key=lambda item: (item["later"].answer_from == "table",
                                           item["earlier"].answer_from == "table"))
        demos.append({"ctx": ctx, "pair": pair})
        if len(demos) >= n:
            break
    return demos


def _turn_record(index: int, question: str, answer) -> dict[str, Any]:
    return {
        "turn": index,
        "question": question,
        "answer": answer.answer,
        "route": answer.route,
        "refused": answer.refused,
        "citations": [c.model_dump() for c in answer.citations],
        "latency_ms": answer.latency_ms,
        "tokens": answer.token_usage.get("total_tokens", 0),
    }


def run_multi_turn_demo(
    app: AppContext,
    contexts_by_id: dict[str, Context],
    ctx_ids: Sequence[str],
    n: int = 3,
) -> dict[str, Any]:
    selections = pick_demos(contexts_by_id, ctx_ids, n=n)
    agent = app.agent  # 带 Checkpointer：多轮状态真的落 SQLite
    demos: list[dict[str, Any]] = []
    for item in selections:
        ctx = item["ctx"]
        pair = item["pair"]
        doc_id = f"tatqa:{ctx.ctx_id}"
        thread_id = f"demo-{ctx.ctx_id}"
        later = pair["later"]
        earlier = pair["earlier"]
        turns = [
            later.question,
            f"And what about {pair['earlier_year']}?",
            (f"What is the difference between the {pair['later_year']} value and the "
             f"{pair['earlier_year']} value?"),
        ]
        records = []
        for index, question in enumerate(turns, start=1):
            answer = agent.ask(question, doc_id=doc_id, thread_id=thread_id)
            record = _turn_record(index, question, answer)
            if index == 1:
                record["gold"] = later.answers
            elif index == 2:
                record["gold"] = earlier.answers
            records.append(record)
        demos.append(
            {
                "ctx_id": ctx.ctx_id,
                "doc_id": doc_id,
                "thread_id": thread_id,
                "title": f"{pair['template'][:80]}（{pair['earlier_year']} vs {pair['later_year']}）",
                "turns": records,
            }
        )
        logger.info("多轮演示完成：%s", ctx.ctx_id)
    payload = {"n": len(demos), "model": app.settings.llm.model, "demos": demos}
    _dump(app.settings, "multi_turn_demo", payload)
    _write_markdown(app.settings, payload)
    return payload


def _write_markdown(settings: Settings, payload: dict[str, Any]) -> None:
    lines = ["# 多轮对话演示（脚本自动生成）", ""]
    lines.append(f"模型：`{payload.get('model')}`；共 {payload.get('n')} 段对话。")
    lines.append("")
    for demo in payload.get("demos", []):
        lines += [f"## {demo['title']}", "", f"thread_id：`{demo['thread_id']}`", ""]
        for turn in demo["turns"]:
            lines.append(f"**Q{turn['turn']}**：{turn['question']}")
            lines.append("")
            lines.append(f"> {turn['answer']}")
            lines.append("")
            lines.append(
                f"route=`{turn['route']}` refused={turn['refused']} "
                f"tokens={turn['tokens']} latency={turn['latency_ms']}ms"
            )
            lines.append("")
    (settings.reports_dir / "multi_turn_demo.md").write_text("\n".join(lines), encoding="utf-8")


def save_demo_json(settings: Settings, payload: dict[str, Any]) -> None:
    (settings.reports_dir / "multi_turn_demo.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
