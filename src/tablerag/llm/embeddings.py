"""向量化：DashScope text-embedding-v3（OpenAI 兼容端点）+ 文本哈希缓存。

与 ChatClient 共用同一套 key 轮换规则：额度耗尽自动换 key，429 退避重试。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Sequence

import httpx

from tablerag.config import EmbeddingSettings
from tablerag.llm.keypool import KeyPool, classify_error

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class EmbeddingHTTPError(EmbeddingError):
    def __init__(self, status: int, body: str):
        super().__init__(f"embedding HTTP {status}: {body[:300]}")
        self.status_code = status


class Embedder:
    def __init__(self, settings: EmbeddingSettings, cache=None, keypool: KeyPool | None = None):
        self.settings = settings
        self.cache = cache
        self.total_tokens = 0
        keys = [settings.api_key.strip()] if settings.api_key.strip() else []
        self.keypool = keypool or (KeyPool(keys) if keys else None)

    @staticmethod
    def _hash(text: str, model: str) -> str:
        return hashlib.sha256(f"{model}::{text}".encode("utf-8")).hexdigest()

    def _request(self, api_key: str, texts: Sequence[str]) -> tuple[list[list[float]], int]:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": self.settings.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        with httpx.Client(timeout=self.settings.timeout_s) as client:
            resp = client.post(
                f"{self.settings.base_url.rstrip('/')}/embeddings", json=payload, headers=headers
            )
        if resp.status_code >= 400:
            raise EmbeddingHTTPError(resp.status_code, resp.text)
        data = resp.json()
        vectors = [item["embedding"] for item in data["data"]]
        return vectors, int((data.get("usage") or {}).get("total_tokens", 0) or 0)

    def _request_with_retry(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.settings.api_key and not self.keypool:
            raise EmbeddingError("未配置 embedding api_key")
        retries = 0
        while True:
            api_key = self.keypool.current() if self.keypool else self.settings.api_key
            try:
                vectors, tokens = self._request(api_key, texts)
                self.total_tokens += tokens
                return vectors
            except Exception as exc:  # noqa: BLE001
                kind = classify_error(exc)
                if kind == "rotate" and self.keypool and self.keypool.rotate(
                    reason=f"embedding: {exc}"[:120], mark_dead=True
                ):
                    continue
                if kind == "retry" and retries < 2:
                    retries += 1
                    time.sleep(min(2 ** retries, 10))
                    continue
                raise EmbeddingError(str(exc)) from exc

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        pending: list[int] = []
        for idx, text in enumerate(texts):
            if self.cache is None:
                pending.append(idx)
                continue
            raw = self.cache.get(self._hash(text, self.settings.model))
            if raw is None:
                pending.append(idx)
                continue
            results[idx] = json.loads(raw)

        batch_size = max(1, self.settings.batch_size)
        for start in range(0, len(pending), batch_size):
            batch_idx = pending[start:start + batch_size]
            vectors = self._request_with_retry([texts[i] for i in batch_idx])
            for idx, vector in zip(batch_idx, vectors, strict=True):
                results[idx] = vector
                if self.cache is not None:
                    self.cache.set(self._hash(texts[idx], self.settings.model), json.dumps(vector))
        return [vector or [] for vector in results]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
