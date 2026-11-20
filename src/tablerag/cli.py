"""命令行入口：tr dataset|render|index|ask|chat|eval|stats|serve。"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from tablerag.config import get_settings
from tablerag.ingest.dataset import (
    context_to_doc,
    download,
    load_contexts,
)
from tablerag.ingest.splits import build_split_payload, load_splits, write_splits
from tablerag.service import build_context

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _split_ids(splits: dict, name: str) -> list[str]:
    if name == "pool":
        return list(splits["corpus_pool"])
    if name == "all":
        return list(splits["dev"]) + list(splits["test"]) + list(splits["corpus_pool"])
    return list(splits[name])


def cmd_dataset(args) -> None:
    settings = get_settings()
    if args.action == "fetch":
        paths = download(settings.raw_dir, splits=args.splits.split(","), force=args.force)
        print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))
        return
    contexts = load_contexts(settings.raw_dir, splits=("train", "dev"))
    if args.action == "stats":
        n_questions = sum(len(c.questions) for c in contexts)
        print(json.dumps({"contexts": len(contexts), "questions": n_questions,
                          "sample_shape": [contexts[0].n_rows, contexts[0].n_cols]},
                         ensure_ascii=False, indent=2))
        return
    if args.action == "split":
        payload = build_split_payload(contexts, seed=args.seed, n_dev=args.n_dev,
                                      n_test=args.n_test)
        path = write_splits(settings.splits_path, payload, force=args.force)
        print(f"已冻结划分：{path}\n{json.dumps(payload['counts'], ensure_ascii=False)}")


def cmd_render(args) -> None:
    settings = get_settings()
    splits = load_splits(settings.splits_path)
    contexts = {c.ctx_id: c for c in load_contexts(settings.raw_dir, splits=("train", "dev"))}
    ids = [cid for cid in _split_ids(splits, args.split) if cid in contexts]
    if args.limit:
        ids = ids[:args.limit]
    from tablerag.ingest.render import render_context

    for ctx_id in ids:
        path = settings.rendered_dir / f"{ctx_id}.pdf"
        render_context(contexts[ctx_id], path, settings.render)
        print(f"{ctx_id} -> {path}")


def cmd_index(args) -> None:
    settings = get_settings()
    splits = load_splits(settings.splits_path)
    contexts = {c.ctx_id: c for c in load_contexts(settings.raw_dir, splits=("train", "dev"))}
    ids = [cid for cid in _split_ids(splits, args.split) if cid in contexts]
    if args.limit:
        ids = ids[:args.limit]
    app = build_context(settings)
    if args.reset:
        app.retriever.reset()
    if args.flatten:
        from tablerag.eval.runner import index_contexts

        stats = index_contexts(app, [contexts[cid] for cid in ids], flatten=True)
    else:
        stats = app.retriever.index_docs([context_to_doc(contexts[cid]) for cid in ids])
    print(json.dumps({**stats, **app.retriever.stats()}, ensure_ascii=False, indent=2))


def cmd_ask(args) -> None:
    settings = get_settings()
    app = build_context(settings, with_agent=True)
    answer = app.agent.ask(args.question, doc_id=args.doc, thread_id=args.thread,
                           hyde=(True if args.hyde else None))
    print(answer.model_dump_json(indent=2))


def cmd_chat(args) -> None:
    settings = get_settings()
    app = build_context(settings, with_agent=True)
    print("输入问题，Ctrl+C 退出。")
    try:
        while True:
            question = input("> ").strip()
            if not question:
                continue
            answer = app.agent.ask(question, doc_id=args.doc, thread_id=args.thread)
            print(f"[{answer.route}] {answer.answer}")
            for citation in answer.citations:
                print(f"  - p{citation.page} {citation.table_id} r{citation.row} c{citation.col}")
    except (KeyboardInterrupt, EOFError):
        print()


def cmd_eval(args) -> None:
    settings = get_settings()
    from tablerag.eval.report import build_full_report
    from tablerag.eval.runner import (
        run_ablation,
        run_parse_eval,
        run_qa_eval,
        run_refusal_eval,
        run_retrieval_eval,
    )

    splits = load_splits(settings.splits_path)
    contexts = load_contexts(settings.raw_dir, splits=("train", "dev"))
    by_id = {c.ctx_id: c for c in contexts}
    test_ids = [cid for cid in splits["test"] if cid in by_id]
    pool_ids = [cid for cid in splits["corpus_pool"] if cid in by_id]

    if args.action == "parse":
        payload = run_parse_eval(by_id, test_ids, settings, limit=args.limit)
        print(json.dumps({k: payload[k] for k in ("n", "structured", "flatten_baseline", "delta")},
                         ensure_ascii=False, indent=2))
        return

    app = build_context(settings, with_agent=True)
    qa_kwargs = {
        "limit": args.limit,
        "questions_per_ctx": args.questions,
        "flatten": args.flatten,
        "open_retrieval": args.open,
        "use_rerank": not args.no_rerank,
        "use_bm25": not args.no_bm25,
        "use_vector": not args.no_vector,
        "repeats": args.repeats,
        "workers": args.workers,
        "resume": not args.no_resume,
        "tag": args.tag,
    }
    if args.action == "qa":
        payload = run_qa_eval(
            app, by_id, test_ids, hyde=(True if args.hyde else None), **qa_kwargs
        )
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        print(f"逐次运行记录：{payload.get('run_log')}")
        return
    if args.action == "refusal":
        payload = run_refusal_eval(
            app, contexts, test_ids, n=args.limit, rebuild_index=args.rebuild,
            workers=args.workers, resume=not args.no_resume,
            protocol=args.protocol, pool_ids=pool_ids, corpus_size=args.corpus_size,
        )
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        print(f"协议：{payload.get('protocol')}；逐次运行记录：{payload.get('run_log')}")
        return
    if args.action == "retrieval":
        payload = run_retrieval_eval(
            app, by_id, test_ids, pool_ids, limit=args.limit,
            questions_per_ctx=args.questions, corpus_size=args.corpus_size,
            resume=not args.no_resume, tag=args.tag,
        )
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return
    if args.action == "ablation":
        payload = run_ablation(
            app, by_id, test_ids, limit=args.limit, questions_per_ctx=args.questions,
            repeats=args.repeats, workers=args.workers, tag=args.tag,
            resume=not args.no_resume,
        )
        print(json.dumps(payload["variants"], ensure_ascii=False, indent=2))
        return
    if args.action == "demo":
        from tablerag.eval.demo import run_multi_turn_demo

        payload = run_multi_turn_demo(app, by_id, test_ids, n=args.limit)
        print(json.dumps({"n": payload["n"],
                          "docs": [d["ctx_id"] for d in payload["demos"]]},
                         ensure_ascii=False, indent=2))
        return
    if args.action == "run":
        parse_payload = run_parse_eval(by_id, test_ids, settings, limit=args.parse_limit)
        qa_payload = run_qa_eval(app, by_id, test_ids, limit=args.qa_limit,
                                 questions_per_ctx=args.questions,
                                 repeats=args.repeats, workers=args.workers,
                                 resume=not args.no_resume, tag=args.tag)
        baseline_payload = run_qa_eval(app, by_id, test_ids, limit=args.qa_limit,
                                       questions_per_ctx=args.questions, flatten=True,
                                       repeats=args.repeats, workers=args.workers,
                                       resume=not args.no_resume, tag=args.tag)
        refusal_payload = run_refusal_eval(app, contexts, test_ids, n=args.refusal_limit,
                                           workers=args.workers, resume=not args.no_resume)
        in_corpus_refusal_payload = None
        if args.honest_refusal:
            in_corpus_refusal_payload = run_refusal_eval(
                app, contexts, test_ids, n=args.refusal_limit, workers=args.workers,
                resume=not args.no_resume, protocol="in_corpus", pool_ids=pool_ids,
                corpus_size=args.corpus_size,
            )
        hyde_payload = None
        if args.hyde:
            hyde_payload = run_qa_eval(app, by_id, test_ids, limit=args.qa_limit,
                                       questions_per_ctx=args.questions, hyde=True,
                                       repeats=args.hyde_repeats, workers=args.workers,
                                       resume=not args.no_resume, tag=args.tag)
        retrieval_payload = None
        if args.retrieval:
            retrieval_payload = run_retrieval_eval(
                app, by_id, test_ids, pool_ids, limit=args.qa_limit,
                questions_per_ctx=args.questions, corpus_size=args.corpus_size,
                resume=not args.no_resume, tag=args.tag,
            )
        ablation_payload = None
        if args.ablation:
            ablation_payload = run_ablation(
                app, by_id, test_ids, limit=args.qa_limit,
                questions_per_ctx=args.questions, repeats=args.repeats,
                workers=args.workers, tag=args.tag, resume=not args.no_resume,
            )
        demo_payload = None
        if not args.no_demo:
            from tablerag.eval.demo import run_multi_turn_demo

            demo_payload = run_multi_turn_demo(app, by_id, test_ids, n=args.demo_limit)
        path = build_full_report(
            settings, parse_payload, qa_payload, baseline_payload, refusal_payload,
            hyde_payload=hyde_payload, demo_payload=demo_payload,
            retrieval_payload=retrieval_payload, ablation_payload=ablation_payload,
            in_corpus_refusal_payload=in_corpus_refusal_payload,
        )
        print(f"报告已生成：{path}")
        print(f"token 合计：QA {qa_payload['summary'].get('total_tokens', 0)} / "
              f"拍平 {baseline_payload['summary'].get('total_tokens', 0)} / "
              f"HyDE {(hyde_payload or {}).get('summary', {}).get('total_tokens', 0)}")


def cmd_stats(args) -> None:
    settings = get_settings()
    app = build_context(settings)
    payload = {
        "index": app.retriever.stats(),
        "cache": {"backend": app.cache.name, "stats": app.cache.stats()},
        "docs": len(app.store.list_docs()),
        "splits": load_splits(settings.splits_path)["counts"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("tablerag.api.app:app", host=args.host, port=args.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tr", description="表格文档 RAG Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    dataset = sub.add_parser("dataset", help="下载 / 划分 TAT-QA")
    dataset.add_argument("action", choices=["fetch", "split", "stats"])
    dataset.add_argument("--splits", default="train,dev")
    dataset.add_argument("--force", action="store_true")
    dataset.add_argument("--seed", type=int, default=42)
    dataset.add_argument("--n-dev", type=int, default=200, dest="n_dev")
    dataset.add_argument("--n-test", type=int, default=300, dest="n_test")
    dataset.set_defaults(func=cmd_dataset)

    render = sub.add_parser("render", help="把 2D 表格渲染成带扰动的 PDF")
    render.add_argument("--split", default="test", choices=["dev", "test", "pool", "all"])
    render.add_argument("--limit", type=int, default=10)
    render.set_defaults(func=cmd_render)

    index = sub.add_parser("index", help="解析并建立索引")
    index.add_argument("build", nargs="?", default="build")
    index.add_argument("--split", default="test", choices=["dev", "test", "pool", "all"])
    index.add_argument("--limit", type=int, default=100)
    index.add_argument("--flatten", action="store_true")
    index.add_argument("--reset", action="store_true")
    index.set_defaults(func=cmd_index)

    ask = sub.add_parser("ask", help="单轮提问")
    ask.add_argument("question")
    ask.add_argument("--doc", default=None)
    ask.add_argument("--hyde", action="store_true")
    ask.add_argument("--thread", default=None)
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="多轮对话（Checkpointer）")
    chat.add_argument("--doc", default=None)
    chat.add_argument("--thread", default="cli-thread")
    chat.set_defaults(func=cmd_chat)

    evaluate = sub.add_parser("eval", help="评测")
    evaluate.add_argument("action", choices=["parse", "qa", "refusal", "retrieval",
                                             "ablation", "demo", "run"])
    evaluate.add_argument("--limit", type=int, default=20)
    evaluate.add_argument("--questions", type=int, default=1)
    evaluate.add_argument("--hyde", action="store_true",
                          help="qa: 打开 HyDE；run: 额外跑一组 HyDE 开对照")
    evaluate.add_argument("--flatten", action="store_true")
    evaluate.add_argument("--open", action="store_true")
    evaluate.add_argument("--no-rerank", action="store_true")
    evaluate.add_argument("--no-bm25", action="store_true")
    evaluate.add_argument("--no-vector", action="store_true")
    evaluate.add_argument("--rebuild", action="store_true")
    evaluate.add_argument("--repeats", type=int, default=1)
    evaluate.add_argument("--hyde-repeats", type=int, default=1, dest="hyde_repeats",
                          help="run 模式下 HyDE 对照的重复次数")
    evaluate.add_argument("--workers", type=int, default=1, help="并发数（评测用，建议 4~6）")
    evaluate.add_argument("--no-resume", action="store_true", dest="no_resume",
                          help="忽略已有 *_runs.jsonl，全部重跑")
    evaluate.add_argument("--tag", default="", help="区分不同实验的断点文件后缀")
    evaluate.add_argument("--parse-limit", type=int, default=20, dest="parse_limit")
    evaluate.add_argument("--qa-limit", type=int, default=10, dest="qa_limit")
    evaluate.add_argument("--refusal-limit", type=int, default=50, dest="refusal_limit")
    evaluate.add_argument("--demo-limit", type=int, default=3, dest="demo_limit")
    evaluate.add_argument("--no-demo", action="store_true", dest="no_demo")
    evaluate.add_argument("--protocol", default="scoped", choices=["scoped", "in_corpus"],
                          help="refusal: 负例协议（scoped 送分 / in_corpus 诚实）")
    evaluate.add_argument("--corpus-size", type=int, default=200, dest="corpus_size",
                          help="in_corpus 负例 / 开放召回的语料库文档数")
    evaluate.add_argument("--retrieval", action="store_true",
                          help="run: 额外跑开放召回评测（doc/cell recall@k）")
    evaluate.add_argument("--ablation", action="store_true",
                          help="run: 额外跑检索消融对照（W2 验收项）")
    evaluate.add_argument("--honest-refusal", action="store_true", dest="honest_refusal",
                          help="run: 额外跑 in_corpus 诚实负例（真实拒答率）")
    evaluate.set_defaults(func=cmd_eval)

    stats = sub.add_parser("stats", help="索引与缓存统计")
    stats.set_defaults(func=cmd_stats)

    serve = sub.add_parser("serve", help="启动 FastAPI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
