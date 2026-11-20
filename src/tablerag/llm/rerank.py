"""精排：DashScope gte-rerank REST；不可用时返回 None，由调用方保持 RRF 顺序。"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import httpx

from tablerag.config import RerankSettings
from tablerag.llm.keypool import KeyPool, classify_error

logger = logging.getLogger(__name__)


class RerankHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status_code = status


class Reranker:
    def __init__(self, settings: RerankSettings, keypool: KeyPool | None = None):
        self.settings = settings
        keys = [settings.api_key.strip()] if settings.api_key.strip() else []
        self.keypool = keypool or (KeyPool(keys) if keys else None)
        self.calls = 0
        self.failures = 0

    @property
    def enabled(self) -> bool:
        return self.settings.enabled and bool(self.settings.api_key or self.keypool)

    def _post(self, api_key: str, payload: dict) -> list[float] | None:
        headers = {"Authorization": f"Bearer {api_key}"}
        with httpx.Client(timeout=self.settings.timeout_s) as client:
            resp = client.post(self.settings.url, json=payload, headers=headers)
        if resp.status_code >= 400:
            raise RerankHTTPError(resp.status_code, resp.text)
        return resp.json()["output"]["results"]

    def score(self, query: str, documents: Sequence[str], top_n: int | None = None) -> list[float] | None:
        if not self.enabled or not documents:
            return None
        payload = {
            "model": self.settings.model,
            "input": {"query": query, "documents": [doc[:1200] for doc in documents]},
            "parameters": {"return_documents": False, "top_n": top_n or len(documents)},
        }
        retries = 0
        while True:
            api_key = self.keypool.current() if self.keypool else self.settings.api_key
            try:
                results = self._post(api_key, payload)
                self.calls += 1
                break
            except Exception as exc:  # noqa: BLE001 - 精排失败不能让整条链路挂掉
                kind = classify_error(exc)
                if kind == "rotate" and self.keypool and self.keypool.rotate(
                    reason=f"rerank: {exc}"[:120], mark_dead=True
                ):
                    continue
                if kind == "retry" and retries < 1:
                    retries += 1
                    time.sleep(2)
                    continue
                self.failures += 1
                logger.warning("rerank 调用失败，回退 RRF 顺序：%s", exc)
                return None
        scores = [0.0] * len(documents)
        for item in results or []:
            idx = int(item["index"])
            if 0 <= idx < len(scores):
                scores[idx] = float(item["relevance_score"])
        return scores

    @property
    def stats(self) -> dict[str, int]:
        return {"calls": self.calls, "failures": self.failures}
