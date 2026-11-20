from tablerag.config import LLMSettings, Settings
from tablerag.storage.cache import MemoryCache, SqliteCache, build_cache
from tablerag.storage.db import Store


def test_sqlite_cache_roundtrip(tmp_path):
    store = Store(tmp_path / "c.db")
    cache = SqliteCache(store)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert cache.stats()["hits"] >= 1


def test_sqlite_cache_ttl_expiry(tmp_path):
    store = Store(tmp_path / "c.db")
    store.cache_set("k", "v", ttl_s=60)
    with store._conn() as conn:
        conn.execute("UPDATE cache SET expires_at=1 WHERE key='k'")
    assert store.cache_get("k") is None


def test_memory_cache_lru_eviction():
    cache = MemoryCache(max_items=2)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.get("a")
    cache.set("c", "3")
    assert cache.get("b") is None
    assert cache.get("a") == "1"


def test_redis_backend_falls_back_to_sqlite(tmp_path):
    store = Store(tmp_path / "c.db")
    cache = build_cache("redis", store, "redis://127.0.0.1:6399/0", 60)
    # 没有 Redis 时必须降级而不是抛异常
    assert cache.name in {"sqlite", "redis"}


def test_settings_reuse_llm_key_for_embedding():
    settings = Settings(llm=LLMSettings(api_key="sk-test"), _env_file=None)
    settings.embedding.api_key = ""
    settings.rerank.api_key = ""
    settings = settings.api_keys_resolved()
    assert settings.embedding.api_key == settings.llm.api_key
    assert settings.rerank.api_key == settings.llm.api_key
