"""评测执行器：解析层 / 问答层 / 负例拒答，所有数字都从这里产出。

长跑友好设计（评测一次要上千次 LLM 调用，免费额度会中途耗尽）：
- 每个「问题 × 重复次数」的结果立即追加到 reports/*_runs.jsonl；
- 重跑时按 key 跳过已完成项，换 key 后可以接着跑，不重复花钱；
- QA 支持并发（workers），Agent 不落 checkpoint，SQLite/Chroma 访问已加锁。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from pypdf import PdfReader

from tablerag.config import Settings
from tablerag.domain.models import ParsedDoc
from tablerag.eval.negatives import build_negatives
from tablerag.eval.parse_metrics import (
    aggregate,
    compare_grids,
    flattened_content_accuracy,
)
from tablerag.eval.qa_metrics import (
    aggregate_qa,
    aggregate_refusal,
    citation_hit,
    qa_scores,
)
from tablerag.ingest.dataset import Context, context_to_doc
from tablerag.ingest.flatten import flatten_pdf
from tablerag.ingest.parser import parse_pdf
from tablerag.ingest.render import render_context
from tablerag.service import AppContext

logger = logging.getLogger(__name__)

_WRITE_LOCK = threading.Lock()


# ---------------- 通用：断点续跑 ----------------
def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "run"


def _signature(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _run_log_path(settings: Settings, kind: str, signature: str) -> Path:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    return settings.reports_dir / f"{kind}_{_slug(settings.llm.model)}_{signature}_runs.jsonl"


def _load_done(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("signature") != signature:
            continue
        key = record.get("run_key")
        if key:
            done[str(key)] = record
    return done


def _append(path: Path, record: dict[str, Any]) -> None:
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------- 解析层 ----------------
def run_parse_eval(
    contexts_by_id: dict[str, Context],
    ctx_ids: Sequence[str],
    settings: Settings,
    limit: int | None = None,
) -> dict[str, Any]:
    ids = [cid for cid in ctx_ids if cid in contexts_by_id]
    if limit:
        ids = ids[:limit]
    rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for ctx_id in ids:
        ctx = contexts_by_id[ctx_id]
        pdf_path = settings.rendered_dir / f"{ctx_id}.pdf"
        result = render_context(ctx, pdf_path, settings.render)
        parsed = parse_pdf(pdf_path, doc_id=f"parse:{ctx_id}")
        pred_cells: list[list[str]] = []
        if parsed.tables:
            best = max(parsed.tables, key=lambda t: t.n_rows * t.n_cols)
            pred_cells = best.cells
        comparison = compare_grids(ctx.cells, pred_cells, doc_id=ctx_id)
        row = comparison.to_dict()
        row.update({"layout": result.meta.get("layout"),
                    "disturbances": result.meta.get("disturbances", {}),
                    "n_parsed_tables": len(parsed.tables)})
        rows.append(row)

        flat_text = "\n".join((page.extract_text() or "") for page in PdfReader(str(pdf_path)).pages)
        flat_rows.append(flattened_content_accuracy(ctx.cells, flat_text, doc_id=ctx_id))

    summary = aggregate(rows)
    flat_summary = aggregate(flat_rows)
    payload = {
        "n": len(rows),
        "structured": summary,
        "flatten_baseline": flat_summary,
        "delta": {
            "content_accuracy": round(summary["content_accuracy"] - flat_summary["content_accuracy"], 4),
            "index_accuracy": round(summary["index_accuracy"] - flat_summary["index_accuracy"], 4),
        },
        "rows": rows,
        "flatten_rows": flat_rows,
    }
    _dump(settings, "parse_eval", payload)
    return payload


# ---------------- 问答层 ----------------
def _doc_for(app: AppContext, ctx: Context, flatten: bool) -> ParsedDoc:
    if flatten:
        pdf_path = app.settings.rendered_dir / f"{ctx.ctx_id}.pdf"
        if not pdf_path.exists():
            render_context(ctx, pdf_path, app.settings.render)
        return flatten_pdf(pdf_path, doc_id=f"tatqa:{ctx.ctx_id}:flat")
    return context_to_doc(ctx)


def index_contexts(app: AppContext, contexts: Sequence[Context], flatten: bool = False,
                   ids: Sequence[str] | None = None) -> dict[str, int]:
    selected = [c for c in contexts if ids is None or c.ctx_id in set(ids)]
    docs = [_doc_for(app, ctx, flatten) for ctx in selected]
    return app.retriever.index_docs(docs)


def run_qa_eval(
    app: AppContext,
    contexts_by_id: dict[str, Context],
    ctx_ids: Sequence[str],
    limit: int | None = None,
    questions_per_ctx: int = 1,
    hyde: bool | None = None,
    flatten: bool = False,
    open_retrieval: bool = False,
    use_rerank: bool = True,
    use_bm25: bool = True,
    use_vector: bool = True,
    repeats: int = 1,
    workers: int = 1,
    resume: bool = True,
    tag: str = "",
) -> dict[str, Any]:
    from tablerag.retrieval.pipeline import RetrievalFlags

    ids = [cid for cid in ctx_ids if cid in contexts_by_id]
    if limit:
        ids = ids[:limit]
    repeats = max(1, repeats)
    if not ids:
        return {"n": 0, "rows": [], "summary": aggregate_qa([]), "repeats": repeats}

    # 只补齐本次实验用到、但还没入库的文档；已入库的不重复向量化（省 token）
    mode = "flat" if flatten else "structured"
    missing = [ctx for ctx in (contexts_by_id[cid] for cid in ids)
               if not app.store.has_doc(
                   f"tatqa:{ctx.ctx_id}:flat" if flatten else f"tatqa:{ctx.ctx_id}")]
    if missing:
        index_contexts(app, missing, flatten=flatten)
    app.retriever.ensure_loaded()

    flags = RetrievalFlags(bm25=use_bm25, vector=use_vector, rerank=use_rerank)
    app.retriever.default_flags = flags
    signature = _signature(
        {
            "model": app.settings.llm.model,
            "mode": mode,
            "hyde": bool(hyde),
            "open": open_retrieval,
            "flags": flags.__dict__,
            "qpc": questions_per_ctx,
            "tag": tag,
        }
    )
    run_path = _run_log_path(app.settings, f"qa_{mode}", signature)
    done = _load_done(run_path, signature) if resume else {}

    tasks: list[dict[str, Any]] = []
    for ctx_id in ids:
        ctx = contexts_by_id[ctx_id]
        doc_id = None if open_retrieval else (
            f"tatqa:{ctx_id}:flat" if flatten else f"tatqa:{ctx_id}"
        )
        for question in ctx.questions[:questions_per_ctx]:
            for rep in range(repeats):
                tasks.append(
                    {
                        "run_key": f"{question.qid}#{rep}",
                        "qid": question.qid,
                        "ctx_id": ctx_id,
                        "doc_id": doc_id,
                        "question": question.question,
                        "gold": question.answers,
                        "scale": question.scale,
                        "answer_type": question.answer_type,
                        "answer_from": question.answer_from,
                        "rep": rep,
                    }
                )

    pending = [task for task in tasks if task["run_key"] not in done]
    skipped = len(tasks) - len(pending)
    if skipped:
        logger.info("断点续跑：跳过已完成 %d 项，剩余 %d 项", skipped, len(pending))

    agent = app.eval_agent

    def execute(task: dict[str, Any]) -> dict[str, Any]:
        answer = agent.ask(task["question"], doc_id=task["doc_id"], hyde=hyde)
        scored = qa_scores(answer.answer, task["gold"], task["scale"])
        record = {
            "signature": signature,
            "run_key": task["run_key"],
            "qid": task["qid"],
            "ctx_id": task["ctx_id"],
            "rep": task["rep"],
            "question": task["question"],
            "gold": task["gold"],
            "answer_type": task["answer_type"],
            "answer_from": task["answer_from"],
            "prediction": answer.answer,
            "refused": answer.refused,
            "route": answer.route,
            "hyde_rounds": answer.hyde_rounds,
            "used_compute": answer.used_compute,
            "em": scored["em"],
            "f1": scored["f1"],
            "citation_hit": citation_hit(answer.citations, task["gold"], app.store),
            "latency_ms": answer.latency_ms,
            "total_tokens": answer.token_usage.get("total_tokens", 0),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _append(run_path, record)
        return record

    new_records: list[dict[str, Any]] = []
    started = time.time()
    if workers <= 1:
        for index, task in enumerate(pending, start=1):
            new_records.append(execute(task))
            if index % 10 == 0 or index == len(pending):
                logger.info("QA 进度 %d/%d（已用 %.0fs）", index, len(pending), time.time() - started)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute, task): task for task in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    new_records.append(future.result())
                except Exception as exc:  # noqa: BLE001 - 单题失败不拖垮整批
                    logger.error("QA 失败 %s：%s", task["run_key"], exc)
                    new_records.append(
                        {
                            **{k: task[k] for k in
                               ("qid", "ctx_id", "question", "gold", "answer_type", "answer_from")},
                            "signature": signature, "run_key": task["run_key"], "rep": task["rep"],
                            "prediction": "", "refused": False, "route": "error",
                            "hyde_rounds": 0, "used_compute": False, "em": 0.0, "f1": 0.0,
                            "citation_hit": False, "latency_ms": 0, "total_tokens": 0,
                            "error": str(exc),
                        }
                    )
                if index % 10 == 0 or index == len(pending):
                    logger.info("QA 进度 %d/%d（已用 %.0fs）", index, len(pending), time.time() - started)

    records = list(done.values()) + new_records
    rows = _aggregate_question_rows(records, tasks)
    summary = aggregate_qa(
        rows,
        latencies=[r.get("latency_ms", 0) for r in records],
        tokens=[r.get("total_tokens", 0) for r in records],
    )
    payload = {
        "n": len(rows),
        "repeats": repeats,
        "workers": workers,
        "mode": mode,
        "model": app.settings.llm.model,
        "hyde": hyde,
        "resumed_skipped": skipped,
        "run_log": str(run_path),
        "elapsed_s": round(time.time() - started, 1),
        "summary": summary,
        "cache": app.cache.stats(),
        "keypool": app.keypool.snapshot() if app.keypool else None,
        "retrieval_flags": flags.__dict__,
        "rows": rows,
    }
    name = f"qa_eval{'_hyde' if hyde else ''}{'_flat' if flatten else ''}"
    _dump(app.settings, name, {**payload, "rows": records})
    return payload


def _aggregate_question_rows(
    records: list[dict[str, Any]], tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    order = {task["run_key"]: index for index, task in enumerate(tasks)}
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_qid.setdefault(str(record["qid"]), []).append(record)

    rows: list[dict[str, Any]] = []
    for qid in sorted(by_qid, key=lambda q: order.get(f"{q}#0", 10**9)):
        recs = sorted(by_qid[qid], key=lambda r: int(r.get("rep", 0)))
        first = recs[0]
        ems = [float(r.get("em", 0.0)) for r in recs]
        f1s = [float(r.get("f1", 0.0)) for r in recs]
        rows.append(
            {
                "qid": qid,
                "ctx_id": first.get("ctx_id"),
                "question": first.get("question"),
                "gold": first.get("gold"),
                "answer_type": first.get("answer_type"),
                "answer_from": first.get("answer_from"),
                "prediction": first.get("prediction"),
                "refused": any(r.get("refused") for r in recs),
                "route": first.get("route"),
                "hyde_rounds": first.get("hyde_rounds", 0),
                "used_compute": first.get("used_compute", False),
                "em": round(statistics.fmean(ems), 4) if ems else 0.0,
                "f1": round(statistics.fmean(f1s), 4) if f1s else 0.0,
                "em_std": round(statistics.pstdev(ems), 4) if len(ems) > 1 else 0.0,
                "f1_std": round(statistics.pstdev(f1s), 4) if len(f1s) > 1 else 0.0,
                "citation_hit": round(
                    statistics.fmean([float(r.get("citation_hit") or 0) for r in recs]), 4
                ) if recs else 0.0,
                "latency_ms": int(statistics.fmean([r.get("latency_ms", 0) for r in recs])),
                "total_tokens": int(statistics.fmean([r.get("total_tokens", 0) for r in recs])),
                "runs": [r.get("prediction") for r in recs] if len(recs) > 1 else [],
            }
        )
    return rows


# ---------------- 拒答 ----------------
def run_refusal_eval(
    app: AppContext,
    contexts: Sequence[Context],
    ctx_ids: Sequence[str],
    n: int = 50,
    rebuild_index: bool = False,
    hyde: bool | None = None,
    workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """负例协议（两种，报告里必须写清用的哪种）：

    - scoped（默认）：按"该文档不存在"提问（doc_id 指向空文档），检索必然无证据，
      检验的是"无证据时是否拒答"。成本低，不破坏现有索引。
    - rebuild：重建索引并把这批文档真实移出语料池，检验开放检索下是否误从别的文档硬编。
      成本高（需要重新向量化整个语料池）。
    """
    from tablerag.eval.negatives import build_negatives

    negatives = build_negatives(contexts, ctx_ids, n=n)
    removed = {item["ctx_id"] for item in negatives}
    if rebuild_index:
        app.retriever.reset()
        keep = [ctx for ctx in contexts if ctx.ctx_id not in removed]
        index_contexts(app, keep)
        protocol = "removed_from_corpus"
    else:
        protocol = "scoped_missing_document"

    signature = _signature(
        {"model": app.settings.llm.model, "kind": "refusal", "protocol": protocol,
         "hyde": bool(hyde)}
    )
    run_path = _run_log_path(app.settings, "refusal", signature)
    done = _load_done(run_path, signature) if resume else {}
    agent = app.eval_agent

    def execute(item: dict[str, Any]) -> dict[str, Any]:
        doc_id = None if rebuild_index else f"tatqa:{item['ctx_id']}:removed"
        answer = agent.ask(item["question"], doc_id=doc_id, hyde=hyde)
        record = {
            "signature": signature,
            "run_key": item["qid"],
            "qid": item["qid"],
            "ctx_id": item["ctx_id"],
            "question": item["question"],
            "gold": item["gold_answers"],
            "prediction": answer.answer,
            "refused": answer.refused,
            "refused_reason": answer.refused_reason,
            "route": answer.route,
            "latency_ms": answer.latency_ms,
            "total_tokens": answer.token_usage.get("total_tokens", 0),
        }
        _append(run_path, record)
        return record

    pending = [item for item in negatives if item["qid"] not in done]
    new_records: list[dict[str, Any]] = []
    if pending:
        if workers <= 1:
            for item in pending:
                new_records.append(execute(item))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(execute, item) for item in pending]
                for future in as_completed(futures):
                    try:
                        new_records.append(future.result())
                    except Exception as exc:  # noqa: BLE001
                        logger.error("负例失败：%s", exc)
    records = list(done.values()) + new_records

    payload = {
        "n": len(records),
        "protocol": protocol,
        "model": app.settings.llm.model,
        "summary": aggregate_refusal(records),
        "removed_docs": sorted(removed),
        "run_log": str(run_path),
        "rows": records,
    }
    _dump(app.settings, "refusal_eval", payload)
    return payload


# ---------------- 工具 ----------------
def _dump(settings: Settings, name: str, payload: dict[str, Any]) -> None:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = payload.get("rows") or []
    if rows:
        keys = list(rows[0].keys())
        with (settings.reports_dir / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in keys})


def mean_std(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.pstdev(values):.4f}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
