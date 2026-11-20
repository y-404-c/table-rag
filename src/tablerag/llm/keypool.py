"""API key 池：额度耗尽 / 触发限流时自动换 key，并在进程内记住已失效的 key。

评测是一次性长跑（上千次调用），免费额度很容易在跑到一半时耗尽。
这里只做三件事：
1. 按顺序使用 key；
2. 遇到 401/403/额度耗尽 → 标记该 key 失效并换下一个；
3. 遇到 429 → 优先换 key，全部都在限流时由调用方退避重试。

线程安全：评测允许并发，切换 key 与读取快照都要加锁。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class KeysExhausted(RuntimeError):
    """所有 key 都不可用（失效或额度耗尽）。"""


def parse_keys(raw: str) -> list[str]:
    """支持逗号 / 分号 / 换行分隔；去重且保持顺序。"""
    if not raw:
        return []
    normalized = raw.replace(";", ",").replace("\n", ",").replace("\r", ",")
    seen: set[str] = set()
    keys: list[str] = []
    for part in normalized.split(","):
        key = part.strip().strip('"').strip("'")
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


class KeyPool:
    def __init__(self, keys: Sequence[str]):
        self._keys = [key for key in keys if key]
        if not self._keys:
            raise ValueError("KeyPool 至少需要一个 api key")
        self._index = 0
        self._dead: set[int] = set()
        self._lock = threading.Lock()
        self.rotations = 0

    @property
    def size(self) -> int:
        return len(self._keys)

    def current(self) -> str:
        with self._lock:
            self._skip_dead_locked()
            return self._keys[self._index]

    def mask(self, key: str | None = None) -> str:
        key = key or self.current()
        return f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "***"

    def rotate(self, reason: str = "", mark_dead: bool = False) -> bool:
        """切到下一个可用 key；全部不可用返回 False。"""
        with self._lock:
            if mark_dead:
                self._dead.add(self._index)
            if len(self._dead) >= len(self._keys):
                return False
            start = self._index
            for step in range(1, len(self._keys) + 1):
                nxt = (start + step) % len(self._keys)
                if nxt not in self._dead:
                    self._index = nxt
                    self.rotations += 1
                    logger.warning(
                        "切换 api key -> %s（原因：%s）", self.mask(self._keys[nxt]), reason or "unknown"
                    )
                    return True
            return False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._keys),
                "active_index": self._index,
                "dead": len(self._dead),
                "rotations": self.rotations,
            }

    def _skip_dead_locked(self) -> None:
        if self._index not in self._dead:
            return
        for step in range(1, len(self._keys) + 1):
            nxt = (self._index + step) % len(self._keys)
            if nxt not in self._dead:
                self._index = nxt
                return


def classify_error(exc: BaseException) -> str:
    """把异常归类为：rotate（换 key）/ retry（退避重试）/ fatal（直接失败）。

    只按可观察的 HTTP 状态与错误文本判断，不依赖供应商私有错误码。
    """
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status in (401, 403):
        return "rotate"
    if status == 429:
        return "rotate" if any(
            token in text for token in ("quota", "exhausted", "arrearage", "limit exceeded", "free tier")
        ) else "retry"
    if any(token in text for token in ("quota", "free tier", "insufficient balance", "arrearage")):
        return "rotate"
    if status is not None and 500 <= int(status) < 600:
        return "retry"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "retry"
    if any(
        token in text
        for token in (
            "timeout", "timed out", "connection", "connect", "ssl", "eof",
            "network", "reset", "broken pipe", "temporarily unavailable",
        )
    ):
        return "retry"
    return "fatal"
