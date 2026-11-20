"""负例构造：测"该不该答"（需求文档 4.2）。

两种协议，难度天差地别，报告里不能混：
- removed_context：问题来自 doc A，把 A 移出语料（scoped 协议下直接指向空文档）。
  检索必然空 → 100% 拒答是送分，不代表真实拒答率。
- in_corpus_no_answer：问题来自 doc A，但语料池里只有 pool 文档（与 A 不相交）。
  文档都在库里，只是答案不在库里——这才是真实场景的"该拒答"。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from tablerag.ingest.dataset import Context


def build_negatives(
    contexts: Sequence[Context],
    ctx_ids: Sequence[str],
    n: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    by_id = {ctx.ctx_id: ctx for ctx in contexts}
    rng = random.Random(seed)
    candidates = [cid for cid in ctx_ids if cid in by_id and by_id[cid].questions]
    rng.shuffle(candidates)
    negatives: list[dict[str, Any]] = []
    for ctx_id in candidates[:n]:
        ctx = by_id[ctx_id]
        question = rng.choice(ctx.questions)
        negatives.append(
            {
                "qid": question.qid,
                "ctx_id": ctx_id,
                "question": question.question,
                "gold_answers": question.answers,
                "gold_scale": question.scale,
                "kind": "removed_context",
            }
        )
    return negatives


def build_in_corpus_negatives(
    contexts: Sequence[Context],
    ctx_ids: Sequence[str],
    pool_ids: Sequence[str],
    n: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """诚实负例：问题取自 ctx_ids（如 test 集），但检索语料只用 pool_ids（与 ctx_ids
    不相交）。答案所在文档不在语料里，但语料里有大量真实的相似表格——模型必须拒答，
    不能从别的文档硬编。比 scoped 协议难得多。"""
    by_id = {ctx.ctx_id: ctx for ctx in contexts}
    pool_set = set(pool_ids)
    rng = random.Random(seed)
    # 只取答案来自表格的问题（数值/专名），开放检索更可能误召回相似表格，负例才有区分度
    candidates = [
        cid for cid in ctx_ids
        if cid in by_id and by_id[cid].questions and cid not in pool_set
    ]
    rng.shuffle(candidates)
    negatives: list[dict[str, Any]] = []
    for ctx_id in candidates[:n]:
        ctx = by_id[ctx_id]
        question = rng.choice(ctx.questions)
        negatives.append(
            {
                "qid": question.qid,
                "ctx_id": ctx_id,
                "question": question.question,
                "gold_answers": question.answers,
                "gold_scale": question.scale,
                "gold_from": question.answer_from,
                "kind": "in_corpus_no_answer",
            }
        )
    return negatives
