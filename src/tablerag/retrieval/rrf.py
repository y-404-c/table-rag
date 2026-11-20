"""RRF（Reciprocal Rank Fusion）：只吃排名不吃分数，免手工调权重。"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def rrf_fuse(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, unit_id in enumerate(ranked, start=1):
            scores[unit_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
