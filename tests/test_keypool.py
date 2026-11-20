from tablerag.config import LLMSettings
from tablerag.llm.keypool import KeyPool, classify_error, parse_keys


class FakeHTTPError(Exception):
    def __init__(self, status: int, message: str = ""):
        super().__init__(message or f"HTTP {status}")
        self.status_code = status


def test_parse_keys_supports_comma_semicolon_newline_and_dedup():
    raw = "a, b;c\n a ,, d"
    assert parse_keys(raw) == ["a", "b", "c", "d"]


def test_settings_resolved_keys_falls_back_to_single_key():
    settings = LLMSettings(api_key="single")
    assert settings.resolved_keys() == ["single"]
    settings = LLMSettings(api_key="single", api_keys="k1,k2")
    assert settings.resolved_keys() == ["k1", "k2"]


def test_keypool_rotates_and_marks_dead():
    pool = KeyPool(["k1", "k2", "k3"])
    assert pool.current() == "k1"
    assert pool.rotate(reason="quota", mark_dead=True)
    assert pool.current() == "k2"
    assert pool.rotate(reason="quota", mark_dead=True)
    assert pool.current() == "k3"
    assert not pool.rotate(reason="quota", mark_dead=True)
    assert pool.snapshot()["dead"] == 3


def test_keypool_survives_rate_limit_rotation_without_marking_dead():
    pool = KeyPool(["k1", "k2"])
    pool.rotate(reason="429", mark_dead=False)
    assert pool.current() == "k2"
    assert pool.rotate(reason="429", mark_dead=False)
    assert pool.current() == "k1"
    assert pool.snapshot()["dead"] == 0


def test_classify_error_maps_auth_quota_and_transient():
    assert classify_error(FakeHTTPError(401)) == "rotate"
    assert classify_error(FakeHTTPError(403)) == "rotate"
    assert classify_error(FakeHTTPError(429, "Free quota exhausted")) == "rotate"
    assert classify_error(FakeHTTPError(429, "too many requests")) == "retry"
    assert classify_error(FakeHTTPError(500)) == "retry"
    assert classify_error(TimeoutError("timeout")) == "retry"
    assert classify_error(ValueError("bad json")) == "fatal"


def test_usage_scope_is_thread_local():
    """并发评测时每个线程的 token 计量必须互不污染。"""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from tablerag.llm.client import ChatClient

    client = ChatClient(LLMSettings(api_key="offline-key"))

    def worker(prompt_tokens: int):
        client.begin_scope()
        client._record_usage(SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=1))
        client._record_usage(SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=1))
        return client.scope_usage()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(worker, 10).result()
        second = pool.submit(worker, 100).result()

    assert first["prompt_tokens"] == 20
    assert second["prompt_tokens"] == 200
    assert first["total_tokens"] == 22
    assert client.usage["prompt_tokens"] == 220
