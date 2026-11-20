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
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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
    cell_citation_hit,
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


# 参与断点签名的源文件：改 prompt / 评测口径 / 检索逻辑后必须让签名变化，
# 否则重跑会静默复用旧结果（test 集污染的根因）。
_CODE_FILES = [
    "agent/graph.py", "agent/prompts.py", "agent/tools.py", "agent/state.py",
    "eval/runner.py", "eval/qa_metrics.py", "eval/negatives.py", "eval/report.py",
    "retrieval/pipeline.py", "retrieval/bm25.py", "retrieval/vector.py", "retrieval/rrf.py",
    "domain/models.py", "domain/text.py", "storage/cache.py", "storage/db.py",
    "ingest/dataset.py", "ingest/parser.py", "ingest/flatten.py", "ingest/splits.py",
    "llm/client.py", "llm/embeddings.py", "llm/rerank.py",
]


def _code_version() -> str:
    """关键源文件内容的哈希；改 prompt 或代码即变化，写进断点签名防污染复用。"""
    root = Path(__file__).resolve().parent.parent  # src/tablerag
    digest = hashlib.sha1()
    for rel in _CODE_FILES:
        path = root / rel
        if path.exists():
            digest.update(rel.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


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
    code_version = _code_version()
    signature = _signature(
        {
            "model": app.settings.llm.model,
            "mode": mode,
            "hyde": bool(hyde),
            "open": open_retrieval,
            "flags": flags.__dict__,
            "qpc": questions_per_ctx,
            "tag": tag,
            "code": code_version,
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
            "cell_citation_hit": cell_citation_hit(answer.citations, task["gold"], app.store),
            "latency_ms": answer.latency_ms,
            "total_tokens": answer.token_usage.get("total_tokens", 0),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _append(run_path, record)
        return record

    # 只报告本批次的 embedding 缓存命中：清零后再跑，避免断点续跑复用同一批
    # query 导致 hit_rate=1.0 的假象（不是系统缓存效果）。
    app.cache.reset()
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
                            "citation_hit": False, "cell_citation_hit": False,
                            "latency_ms": 0, "total_tokens": 0,
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
        "code_version": code_version,
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
                "cell_citation_hit": round(
                    statistics.fmean([float(r.get("cell_citation_hit") or 0) for r in recs]), 4
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
    protocol: str = "scoped",
    pool_ids: Sequence[str] | None = None,
    corpus_size: int = 200,
) -> dict[str, Any]:
    """负例协议（报告里必须写清用的哪种，数字不能混为一谈）：

    - scoped_missing_document（默认，送分协议）：doc_id 指向不存在的文档，检索必然空，
      100% 拒答是预期的，**不代表真实拒答率**。成本低，不破坏现有索引。
    - in_corpus_no_answer（诚实协议）：问题来自 doc A，但只把 corpus_pool 里的文档
      （与 A 不相交）建索引并开放检索——文档都在库里，但答案确实不在库里。检验的是
      "检索到了相似文档但没答案时是否拒答"，这才是真实拒答能力。
    - removed_from_corpus：重建索引把负例文档真实移出语料池（旧协议，保留兼容）。
    """
    from tablerag.eval.negatives import build_in_corpus_negatives

    by_id = {ctx.ctx_id: ctx for ctx in contexts}
    if protocol == "in_corpus":
        if not pool_ids:
            raise ValueError("in_corpus 协议需要 pool_ids（语料池文档）")
        negatives = build_in_corpus_negatives(contexts, ctx_ids, pool_ids, n=n)
        pool_list = [cid for cid in pool_ids if cid in by_id][:corpus_size]
        missing_pool = [cid for cid in pool_list if not app.store.has_doc(f"tatqa:{cid}")]
        if missing_pool:
            index_contexts(app, [by_id[cid] for cid in missing_pool])
        app.retriever.ensure_loaded()
        protocol_name = "in_corpus_no_answer"
        removed: set[str] = set()
        corpus_docs = pool_list
    else:
        negatives = build_negatives(contexts, ctx_ids, n=n)
        removed = {item["ctx_id"] for item in negatives}
        if rebuild_index:
            app.retriever.reset()
            keep = [ctx for ctx in contexts if ctx.ctx_id not in removed]
            index_contexts(app, keep)
            protocol_name = "removed_from_corpus"
        else:
            protocol_name = "scoped_missing_document"
        corpus_docs = []

    code_version = _code_version()
    signature = _signature(
        {"model": app.settings.llm.model, "kind": "refusal", "protocol": protocol_name,
         "hyde": bool(hyde), "code": code_version}
    )
    run_path = _run_log_path(app.settings, "refusal", signature)
    done = _load_done(run_path, signature) if resume else {}
    agent = app.eval_agent
    app.cache.reset()

    def execute(item: dict[str, Any]) -> dict[str, Any]:
        if protocol_name == "scoped_missing_document":
            doc_id = f"tatqa:{item['ctx_id']}:removed"
        else:
            doc_id = None  # in_corpus / removed_from_corpus：开放检索
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
        "protocol": protocol_name,
        "code_version": code_version,
        "model": app.settings.llm.model,
        "summary": aggregate_refusal(records),
        "removed_docs": sorted(removed),
        "corpus_docs": len(corpus_docs),
        "run_log": str(run_path),
        "rows": records,
    }
    _dump(app.settings, "refusal_eval", payload)
    return payload


# ---------------- 开放召回（从语料库检索，不是给定文档内） ----------------
def _cell_in_topk(
    candidates: list, gold_doc_id: str, golds: list[str], store: Any
) -> bool:
    """top-k 召回里是否有一个单元来自 gold 文档、且该行有单元格命中 gold 答案。"""
    from tablerag.domain.text import normalize_cell, numbers_match, parse_number

    gold_norm = [normalize_cell(g).lower() for g in golds if normalize_cell(g)]
    gold_numbers = [parse_number(g) for g in gold_norm]
    for scored in candidates:
        unit = scored.unit
        if unit.doc_id != gold_doc_id or not unit.table_id or unit.row is None:
            continue
        table = store.get_table(unit.table_id)
        if table is None:
            continue
        for col in range(table.n_cols):
            cell = table.cell(unit.row, col)
            if not cell.strip():
                continue
            ncell = normalize_cell(cell).lower()
            for g in gold_norm:
                if g in ncell or ncell in g:
                    return True
            cn = parse_number(cell)
            if cn is not None:
                for gn in gold_numbers:
                    if gn is not None and numbers_match(cn, gn):
                        return True
    return False


def run_retrieval_eval(
    app: AppContext,
    contexts_by_id: dict[str, Context],
    ctx_ids: Sequence[str],
    pool_ids: Sequence[str],
    limit: int | None = None,
    questions_per_ctx: int = 1,
    top_ks: tuple[int, ...] = (5, 20),
    corpus_size: int = 500,
    resume: bool = True,
    tag: str = "",
) -> dict[str, Any]:
    """开放召回评测：不传 doc_id，在语料库（pool + 本题 test 文档）上召回，
    算 doc recall@k 和 cell recall@k。这才真正检验双路召回 + 精排的技能栏。

    注意：索引语料库要向量化大量文档，成本高；corpus_size 控制规模，报告里如实披露。
    """
    code_version = _code_version()
    signature = _signature(
        {"model": app.settings.llm.model, "kind": "retrieval",
         "code": code_version, "qpc": questions_per_ctx,
         "top_ks": list(top_ks), "corpus_size": corpus_size, "tag": tag}
    )
    run_path = _run_log_path(app.settings, "retrieval", signature)
    done = _load_done(run_path, signature) if resume else {}

    test_in = [cid for cid in ctx_ids if cid in contexts_by_id]
    if limit:
        test_in = test_in[:limit]
    pool_in = [cid for cid in pool_ids if cid in contexts_by_id][:corpus_size]
    all_corpus = test_in + pool_in
    missing = [cid for cid in all_corpus if not app.store.has_doc(f"tatqa:{cid}")]
    if missing:
        index_contexts(app, [contexts_by_id[cid] for cid in missing])
    app.retriever.ensure_loaded()
    app.cache.reset()

    max_k = max(top_ks)
    records: list[dict[str, Any]] = []
    for ctx_id in test_in:
        ctx = contexts_by_id[ctx_id]
        gold_doc = f"tatqa:{ctx_id}"
        for question in ctx.questions[:questions_per_ctx]:
            run_key = question.qid
            if run_key in done:
                records.append(done[run_key])
                continue
            candidates, trace = app.retriever.search(
                question.question, doc_id=None, top_k=max_k
            )
            doc_ids = [c.unit.doc_id for c in candidates]
            doc_hit = {f"doc@{k}": gold_doc in doc_ids[:k] for k in top_ks}
            cell_hit = {
                f"cell@{k}": _cell_in_topk(candidates[:k], gold_doc, question.answers, app.store)
                for k in top_ks
            }
            record = {
                "signature": signature,
                "run_key": run_key,
                "qid": run_key,
                "ctx_id": ctx_id,
                "question": question.question,
                "gold": question.answers,
                "answer_from": question.answer_from,
                **doc_hit,
                **cell_hit,
                "n_retrieved": len(candidates),
                "bm25_hits": trace.bm25_hits,
                "vector_hits": trace.vector_hits,
                "rerank_used": trace.rerank_used,
            }
            _append(run_path, record)
            records.append(record)

    n = len(records) or 1
    summary: dict[str, Any] = {"n": len(records), "corpus_docs": len(all_corpus)}
    for k in top_ks:
        summary[f"doc_recall@{k}"] = round(
            sum(1 for r in records if r.get(f"doc@{k}")) / n, 4
        )
        summary[f"cell_recall@{k}"] = round(
            sum(1 for r in records if r.get(f"cell@{k}")) / n, 4
        )
    payload = {
        "n": len(records),
        "corpus_docs": len(all_corpus),
        "code_version": code_version,
        "top_ks": list(top_ks),
        "summary": summary,
        "run_log": str(run_path),
        "rows": records,
    }
    _dump(app.settings, "retrieval_eval", payload)
    return payload


# ---------------- 消融（W2 验收项） ----------------
ABLATION_VARIANTS = [
    ("full", True, True, True),
    ("no_rerank", False, True, True),
    ("no_bm25", True, False, True),
    ("no_vector", True, True, False),
]


def run_ablation(
    app: AppContext,
    contexts_by_id: dict[str, Context],
    ctx_ids: Sequence[str],
    limit: int = 15,
    questions_per_ctx: int = 1,
    repeats: int = 1,
    workers: int = 1,
    tag: str = "ablation",
    resume: bool = True,
) -> dict[str, Any]:
    """在同一批问题上跑检索开关变体，产出 EM/F1/cell 命中/延迟对照表（W2 验收项）。"""
    results: list[dict[str, Any]] = []
    for name, rerank, bm25, vector in ABLATION_VARIANTS:
        payload = run_qa_eval(
            app, contexts_by_id, ctx_ids, limit=limit,
            questions_per_ctx=questions_per_ctx, repeats=repeats, workers=workers,
            use_rerank=rerank, use_bm25=bm25, use_vector=vector,
            resume=resume, tag=f"{tag}_{name}",
        )
        s = payload.get("summary", {})
        results.append(
            {
                "variant": name,
                "rerank": rerank,
                "bm25": bm25,
                "vector": vector,
                "n": payload.get("n", 0),
                "em": s.get("em", 0.0),
                "f1": s.get("f1", 0.0),
                "citation_hit": s.get("citation_hit", 0.0),
                "cell_citation_hit": s.get("cell_citation_hit", 0.0),
                "p50_latency_ms": s.get("p50_latency_ms", 0),
                "p95_latency_ms": s.get("p95_latency_ms", 0),
                "total_tokens": s.get("total_tokens", 0),
            }
        )
    payload_out = {
        "tag": tag,
        "code_version": _code_version(),
        "variants": results,
    }
    _dump(app.settings, "ablation_eval", payload_out)
    return payload_out


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
