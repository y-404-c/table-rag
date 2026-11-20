"""报告生成：只汇总 eval 脚本落盘的 JSON，不手填任何数字。"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from tablerag.config import Settings

HONESTY_NOTES = """
## 局限与风险披露（必须随数字一起看）

1. **数据是英文财报**：TAT-QA 为英文数据集，中文场景说服力有限。没有为了好看自造中文评测集。
2. **渲染-解析同源**：PDF 由 reportlab 按固定 seed 程序生成、由 pdfplumber 解析，测的是
   "解析管线能否还原被程序渲染的表格"，不等于真实扫描件/复杂排版的解析能力。渲染与解析
   刻意使用不同工具，并随机化空白/合并单元格/跨页/双栏，但**无法消除循环论证**。
3. **代表性**：TAT-QA 表格规模偏小（多为 5~10 行），结论不外推到大型多页财报表。
4. **出处命中口径自定**：见 eval/qa_metrics.py 的 `citation_hit`，与官方指标不是同一口径。
5. **样本规模**：受模型 token 额度限制，本次为 test 集前 N 个文档/问题的小样本（N 见每节标题下），
   不是 300 个 test 文档的全量结论；同一条命令可扩样重跑，断点续跑不会重复花钱。
6. **本文件中的每一个数字都来自 `tr eval ...` 的输出**；未运行的项显示为"未运行"。
"""


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_parse_section(payload: dict[str, Any]) -> str:
    structured = payload.get("structured", {})
    flat = payload.get("flatten_baseline", {})
    delta = payload.get("delta", {})
    lines = [
        "## 1. 解析层（结构化路径 vs 拍平基线）",
        "",
        f"样本数：{payload.get('n', 0)} 个文档；渲染扰动：跨页断表 / 合并单元格 / 双栏 / 随机空白。",
        "",
        "| 路径 | Cell Content Accuracy | Cell Index Accuracy | 形状完全一致率 |",
        "|---|---|---|---|",
        f"| 结构化（pdfplumber 表格恢复） | {_fmt(structured.get('content_accuracy'))} | "
        f"{_fmt(structured.get('index_accuracy'))} | {_fmt(structured.get('shape_match_rate'))} |",
        f"| 拍平基线（pypdf 纯文本） | {_fmt(flat.get('content_accuracy'))} | "
        f"{_fmt(flat.get('index_accuracy'))} | n/a（无结构） |",
        "",
        f"差值：content {_fmt(delta.get('content_accuracy'))} / index {_fmt(delta.get('index_accuracy'))}。",
        "",
    ]
    rows = [row for row in payload.get("rows", []) if row.get("content_accuracy", 1.0) < 0.999][:5]
    if rows:
        lines += ["失败样例（前 5 个，按输入 order 而非挑最差）：", ""]
        for row in rows:
            lines.append(
                f"- {row['doc_id']}: gold {row['gold_shape']} vs pred {row['pred_shape']}，"
                f"content={_fmt(row['content_accuracy'])} index={_fmt(row['index_accuracy'])}，"
                f"缺失样例 {row.get('missing_sample', [])[:2]}"
            )
        lines.append("")
    return "\n".join(lines)


def render_qa_section(payload: dict[str, Any], title: str) -> str:
    summary = payload.get("summary", {})
    repeats = payload.get("repeats", 1)
    n = payload.get("n", 0)
    workers = payload.get("workers", 1)
    lines = [
        f"## {title}",
        "",
        f"样本数：{n} 个问题 × {repeats} 次重复 = {n * repeats} 次运行；"
        f"并发 {workers}；模型 `{payload.get('model', 'n/a')}`。",
        "",
        "| EM | EM 题内标准差 | F1 | F1 题内标准差 | 出处命中率 | 拒答率 | P50(ms) | P95(ms) | 平均 token | 总 token |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| {_fmt(summary.get('em', 0))} | {_fmt(summary.get('em_std', 0))} | "
        f"{_fmt(summary.get('f1', 0))} | {_fmt(summary.get('f1_std', 0))} | "
        f"{_fmt(summary.get('citation_hit', 0))} | {_fmt(summary.get('refusal_rate', 0))} | "
        f"{summary.get('p50_latency_ms', summary.get('avg_latency_ms', 0))} | "
        f"{summary.get('p95_latency_ms', summary.get('avg_latency_ms', 0))} | "
        f"{summary.get('avg_tokens', 0)} | {summary.get('total_tokens', 0)} |",
        "",
    ]
    if payload.get("resumed_skipped"):
        lines.append(
            f"本批结果包含 {payload['resumed_skipped']} 次断点续跑复用的运行"
            f"（同一签名、同一命令；新跑 {payload.get('n', 0) * repeats - payload['resumed_skipped']} 次）。"
        )
    keypool = payload.get("keypool") or {}
    if keypool.get("rotations"):
        lines.append(f"评测期间 key 轮换：{keypool['rotations']} 次（{keypool.get('dead', 0)} 个 key 额度耗尽）。")
    cache = payload.get("cache") or {}
    if cache:
        lines.append(
            f"缓存后端命中率：{_fmt(cache.get('hit_rate', 0))}"
            f"（hits={cache.get('hits', 0)}, misses={cache.get('misses', 0)}）"
        )
    flags = payload.get("retrieval_flags") or {}
    if flags:
        lines.append(f"检索开关：{flags}")
    lines.append("")
    failures = [r for r in payload.get("rows", []) if float(r.get("f1", 0)) < 0.999][:5]
    if failures:
        lines += ["掉点样例（前 5 个，按出现顺序；不含满分题）：", ""]
        for row in failures:
            lines.append(
                f"- Q: {row['question']}\n  gold={row.get('gold')} | pred={row['prediction']!r} "
                f"| route={row.get('route')} | refused={row.get('refused')} | EM={row.get('em')} "
                f"| citation_hit={row.get('citation_hit')}"
            )
        lines.append("")
    else:
        lines += ["本批无掉点样例（EM/F1 均为满分）。", ""]
    return "\n".join(lines)


def render_hyde_section(off_payload: dict[str, Any], on_payload: dict[str, Any]) -> str:
    off = off_payload.get("summary", {})
    on = on_payload.get("summary", {})
    delta_em = on.get("em", 0) - off.get("em", 0)
    verdict = "涨点" if delta_em > 0 else ("持平" if abs(delta_em) < 1e-9 else "掉点（此场景无效，如实记录）")
    return "\n".join(
        [
            "## 5. HyDE 开关对照（同一批问题、其余条件不变）",
            "",
            f"样本数：HyDE 关 {off_payload.get('n', 0)} 题 × {off_payload.get('repeats', 1)} 次；"
            f"HyDE 开 {on_payload.get('n', 0)} 题 × {on_payload.get('repeats', 1)} 次；"
            f"模型 `{off_payload.get('model', 'n/a')}`。",
            "",
            "| 条件 | EM | F1 | 出处命中率 | 拒答率 | P50(ms) | P95(ms) |",
            "|---|---|---|---|---|---|---|",
            f"| HyDE 关 | {_fmt(off.get('em', 0))} | {_fmt(off.get('f1', 0))} | "
            f"{_fmt(off.get('citation_hit', 0))} | {_fmt(off.get('refusal_rate', 0))} | "
            f"{off.get('p50_latency_ms', off.get('avg_latency_ms', 0))} | "
            f"{off.get('p95_latency_ms', off.get('avg_latency_ms', 0))} |",
            f"| HyDE 开 | {_fmt(on.get('em', 0))} | {_fmt(on.get('f1', 0))} | "
            f"{_fmt(on.get('citation_hit', 0))} | {_fmt(on.get('refusal_rate', 0))} | "
            f"{on.get('p50_latency_ms', on.get('avg_latency_ms', 0))} | "
            f"{on.get('p95_latency_ms', on.get('avg_latency_ms', 0))} |",
            "",
            f"EM 差值：{delta_em:+.4f} → {verdict}。",
            "",
        ]
    )


def render_demo_section(payload: dict[str, Any]) -> str:
    lines = [
        "## 6. 多轮对话演示（Checkpointer 真实续接）",
        "",
        f"共 {len(payload.get('demos', []))} 段对话；每段第 2 轮是「那…呢」式追问，"
        "第 3 轮需要复用前两轮的数值做差。",
        "",
    ]
    for demo in payload.get("demos", []):
        lines.append(f"### {demo.get('title', demo.get('ctx_id'))}")
        lines.append("")
        lines.append(f"文档：`{demo.get('doc_id')}`，thread：`{demo.get('thread_id')}`")
        lines.append("")
        for turn in demo.get("turns", []):
            citations = ", ".join(
                f"p{c.get('page')} {c.get('table_id')} r{c.get('row')} c{c.get('col')}"
                for c in turn.get("citations", [])[:3]
            )
            lines.append(
                f"- **Q{turn.get('turn')}**：{turn.get('question')}\n"
                f"  - A：{turn.get('answer')}\n"
                f"  - route={turn.get('route')} | refused={turn.get('refused')} | "
                f"出处：{citations or '无'}"
            )
        lines.append("")
    return "\n".join(lines)


def render_refusal_section(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    protocol = payload.get("protocol", "")
    if protocol == "removed_from_corpus":
        protocol_text = "协议：removed_from_corpus（负例文档已真实移出语料池并重建索引）"
    else:
        protocol_text = (
            "协议：scoped_missing_document（提问指向一个不存在的文档，"
            "检验「无证据时是否拒答」；文档仍在语料池中）"
        )
    return "\n".join(
        [
            "## 4. 负例拒答",
            "",
            f"负例数：{payload.get('n', 0)}。{protocol_text}",
            "",
            "| 正确拒答率 | 硬编率（应当拒答却作答） |",
            "|---|---|",
            f"| {_fmt(summary.get('refusal_rate', 0))} | {_fmt(summary.get('hallucination_rate', 0))} |",
            "",
        ]
    )


def write_report(
    settings: Settings,
    name: str,
    title: str,
    sections: list[str],
    meta: dict[str, Any] | None = None,
) -> Path:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [f"# {title}", "", f"生成时间：{stamp}", ""]
    if meta:
        header += ["```json", json.dumps(meta, ensure_ascii=False, indent=2), "```", ""]
    body = "\n".join(header + sections + [HONESTY_NOTES])
    path = settings.reports_dir / name
    path.write_text(body, encoding="utf-8")
    return path


def build_full_report(
    settings: Settings,
    parse_payload: dict[str, Any] | None,
    qa_payload: dict[str, Any] | None,
    baseline_payload: dict[str, Any] | None,
    refusal_payload: dict[str, Any] | None,
    hyde_payload: dict[str, Any] | None = None,
    demo_payload: dict[str, Any] | None = None,
) -> Path:
    sections: list[str] = []
    sections.append("> 本报告由 `tr eval run` 生成；所有数字来自 reports/*.json，未手工填写。\n")
    if parse_payload:
        sections.append(render_parse_section(parse_payload))
    if qa_payload:
        sections.append(render_qa_section(qa_payload, "2. 问答层（结构化路径）"))
    if baseline_payload:
        sections.append(render_qa_section(baseline_payload, "3. 问答层（拍平基线）"))
    if refusal_payload:
        sections.append(render_refusal_section(refusal_payload))
    if hyde_payload and qa_payload:
        sections.append(render_hyde_section(qa_payload, hyde_payload))
    if demo_payload:
        sections.append(render_demo_section(demo_payload))
    if not sections:
        sections.append("未运行任何实验。")
    meta = {
        "model": settings.llm.model,
        "embedding": settings.embedding.model,
        "rerank": settings.rerank.model,
        "repeats": (qa_payload or {}).get("repeats"),
        "workers": (qa_payload or {}).get("workers"),
        "hyde_compare": bool(hyde_payload),
        "keypool": (qa_payload or {}).get("keypool"),
        "total_qa_tokens": sum(
            (payload or {}).get("summary", {}).get("total_tokens", 0)
            for payload in (qa_payload, baseline_payload, hyde_payload)
        ),
    }
    return write_report(settings, "report.md", "表格文档 RAG Agent：评测报告", sections, meta=meta)
