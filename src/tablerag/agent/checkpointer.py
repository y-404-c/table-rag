"""Checkpointer：多轮追问的状态持久化。

默认 SQLite（langgraph-checkpoint-sqlite，Windows 可直接跑）。
Redis 后端按需求文档预留：需要安装 langgraph-checkpoint-redis 与运行中的 Redis，
不可用时显式降级并记录日志，不静默丢上下文。
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from tablerag.config import Settings

logger = logging.getLogger(__name__)


class CheckpointerFactory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._conn: sqlite3.Connection | None = None

    def build(self):
        backend = (self.settings.cache.backend or "sqlite").lower()
        if backend == "redis":
            try:
                from langgraph.checkpoint.redis import RedisSaver  # type: ignore

                saver = RedisSaver.from_conn_string(self.settings.cache.url)
                logger.info("Checkpointer 使用 Redis：%s", self.settings.cache.url)
                return saver
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis Checkpointer 不可用（%s），降级为 SQLite", exc)
        return self.sqlite()

    def sqlite(self):
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = Path(self.settings.var_dir) / "checkpoints.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._conn is None:
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
        return SqliteSaver(self._conn)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
