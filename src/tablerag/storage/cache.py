"""缓存抽象：SQLite（默认）/ Redis / 内存。带命中率统计。"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Protocol

from tablerag.storage.db import Store

logger = logging.getLogger(__name__)


def cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CacheBackend(Protocol):
    name: str

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl_s: int) -> None: ...


class MetricsMixin:
    hits: int = 0
    misses: int = 0

    def record(self, hit: bool) -> None:
        if hit:
            type(self).hits += 1
        else:
            type(self).misses += 1

    @classmethod
    def stats(cls) -> dict[str, float]:
        total = cls.hits + cls.misses
        return {
            "hits": cls.hits,
            "misses": cls.misses,
            "hit_rate": (cls.hits / total) if total else 0.0,
        }


class SqliteCache(MetricsMixin):
    name = "sqlite"

    def __init__(self, store: Store, ttl_s: int = 0):
        self.store = store
        self.ttl_s = ttl_s

    def get(self, key: str) -> str | None:
        value = self.store.cache_get(key)
        self.record(value is not None)
        return value

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        self.store.cache_set(key, value, ttl_s if ttl_s is not None else self.ttl_s)


class MemoryCache(MetricsMixin):
    name = "memory"

    def __init__(self, max_items: int = 2048, ttl_s: int = 0):
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._data: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None:
            self.record(False)
            return None
        value, expires = item
        if expires and expires < time.time():
            self._data.pop(key, None)
            self.record(False)
            return None
        self._data.move_to_end(key)
        self.record(True)
        return value

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        ttl = ttl_s if ttl_s is not None else self.ttl_s
        self._data[key] = (value, time.time() + ttl if ttl else 0.0)
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)


class RedisCache(MetricsMixin):
    name = "redis"

    def __init__(self, url: str, ttl_s: int = 0):
        import redis  # 延迟导入，未装 redis 依赖时不影响 SQLite 路径

        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.client.ping()
        self.ttl_s = ttl_s

    def get(self, key: str) -> str | None:
        value = self.client.get(key)
        self.record(value is not None)
        return value

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        ttl = ttl_s if ttl_s is not None else self.ttl_s
        if ttl and ttl > 0:
            self.client.setex(key, ttl, value)
        else:
            self.client.set(key, value)


def build_cache(backend: str, store: Store, url: str, ttl_s: int) -> Any:
    """按配置构造缓存；Redis 不可用时显式降级到 SQLite 并留下日志。"""
    backend = (backend or "sqlite").lower()
    if backend == "redis":
        try:
            return RedisCache(url, ttl_s)
        except Exception as exc:  # noqa: BLE001 - 连接失败要降级而不是崩
            logger.warning("Redis 不可用（%s），降级为 SQLite 缓存", exc)
            return SqliteCache(store, ttl_s)
    if backend == "memory":
        return MemoryCache(ttl_s=ttl_s)
    return SqliteCache(store, ttl_s)


def cached_json(cache: Any, key: str, producer: Callable[[], Any], ttl_s: int | None = None) -> Any:
    """读缓存 → 未命中则生产并写回。缓存只做加速，不改变语义。"""
    raw = cache.get(key)
    if raw is not None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    value = producer()
    try:
        cache.set(key, json.dumps(value, ensure_ascii=False, default=str), ttl_s)
    except Exception as exc:  # noqa: BLE001
        logger.warning("缓存写入失败：%s", exc)
    return value
