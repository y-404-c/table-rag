"""数据划分：固定 seed、冻结落盘、之后不得改动。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

from tablerag.ingest.dataset import Context

DEFAULT_SEED = 42
DEFAULT_N_DEV = 200
DEFAULT_N_TEST = 300


def build_split_payload(
    contexts: Sequence[Context],
    seed: int = DEFAULT_SEED,
    n_dev: int = DEFAULT_N_DEV,
    n_test: int = DEFAULT_N_TEST,
) -> dict[str, Any]:
    ids = sorted(ctx.ctx_id for ctx in contexts)
    if len(ids) < n_dev + n_test + 10:
        raise ValueError(f"上下文数量 {len(ids)} 不足以划分 dev={n_dev} + test={n_test}")
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    dev = sorted(shuffled[:n_dev])
    test = sorted(shuffled[n_dev:n_dev + n_test])
    pool = sorted(shuffled[n_dev + n_test:])
    assert not set(dev) & set(test)
    assert not set(dev) & set(pool)
    assert not set(test) & set(pool)
    return {
        "seed": seed,
        "n_dev": n_dev,
        "n_test": n_test,
        "counts": {"dev": len(dev), "test": len(test), "corpus_pool": len(pool)},
        "dev": dev,
        "test": test,
        "corpus_pool": pool,
    }


def write_splits(path: Path, payload: dict[str, Any], force: bool = False) -> Path:
    """划分一旦冻结就不允许覆盖（需求文档 3.3）。"""
    if path.exists() and not force:
        raise FileExistsError(f"{path} 已存在且已冻结；确需重建请显式传 --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_splits(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，先运行 tr dataset split")
    return json.loads(path.read_text(encoding="utf-8"))


def split_of(splits: dict[str, Any], ctx_id: str) -> str:
    if ctx_id in set(splits["dev"]):
        return "dev"
    if ctx_id in set(splits["test"]):
        return "test"
    return "corpus_pool"
