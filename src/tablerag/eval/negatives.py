"""负例构造：取测试集问题，把对应文档从语料池移除，正确答案在语料中不存在。

没有负例就无法测"该不该答"（需求文档 4.2）。移除式负例比改数字更干净：
问题的每个词都合法，只是答案不在库里。
"""

from __future__ import annotations

import random
from typing import Any, Sequence

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
