"""OpenAI 兼容 LLM 客户端：普通对话 / JSON 输出 / Function Calling + token 计量。

支持多 key 轮换：额度耗尽（401/403/quota）时自动换下一个 key，429 退避重试。
评测会并发调用，因此计量与 key 切换都加锁。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from tablerag.config import LLMSettings
from tablerag.llm.keypool import KeyPool, KeysExhausted, classify_error

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出里抽第一个完整 JSON 对象；容忍 ```json 围栏与前后废话。"""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    if start < 0:
        raise LLMError(f"输出中没有 JSON 对象：{text[:200]}")
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise LLMError(f"JSON 解析失败：{exc}；原文：{text[:200]}") from exc
    if not isinstance(obj, dict):
        raise LLMError("JSON 顶层不是对象")
    return obj


@dataclass
class ChatClient:
    settings: LLMSettings
    keypool: KeyPool | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    last_raw: str = ""
    _client: Any = field(default=None, repr=False)
    _clients: dict[str, Any] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        keys = self.settings.resolved_keys()
        if not keys:
            raise LLMError("未配置 TR_LLM__API_KEY / TR_LLM__API_KEYS，请在 .env 中填写")
        if self.keypool is None:
            self.keypool = KeyPool(keys)
        if self._client is None:
            self._client = self._build_client(self.keypool.current())

    def _build_client(self, api_key: str) -> Any:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_s,
            max_retries=0,  # 重试由本类统一处理，才能感知换 key
        )
        self._clients[api_key] = client
        return client

    def _client_for(self, api_key: str) -> Any:
        with self._lock:
            if self._client is None:
                self._client = self._build_client(api_key)
            if api_key not in self._clients:
                self._clients[api_key] = self._build_client(api_key)
            return self._clients[api_key]

    def _record_usage(self, usage: Any) -> None:
        with self._lock:
            self.calls += 1
            if usage is None:
                return
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    @property
    def usage(self) -> dict[str, int]:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            }

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        assert self.keypool is not None
        retry_budget = max(1, self.settings.max_retries + 1)
        retries = 0
        while True:
            api_key = self.keypool.current()
            try:
                response = self._client_for(api_key).chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - 统一分类后决定换 key / 退避 / 失败
                kind = classify_error(exc)
                if kind == "rotate":
                    if not self.keypool.rotate(reason=str(exc)[:120], mark_dead=True):
                        raise LLMError(f"所有 api key 均不可用（额度或鉴权失败）：{exc}") from exc
                    continue
                if kind == "retry":
                    retries += 1
                    if retries > retry_budget:
                        raise LLMError(f"LLM 调用失败（重试 {retries - 1} 次后）：{exc}") from exc
                    time.sleep(min(2 ** retries, 20))
                    continue
                raise LLMError(f"LLM 调用失败：{exc}") from exc

        self._record_usage(getattr(response, "usage", None))
        message = response.choices[0].message
        self.last_raw = message.content or ""
        return message

    def text(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return (self.chat(messages, **kwargs).content or "").strip()

    def json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        content = self.text(messages, **kwargs)
        return extract_json(content)


__all__ = ["ChatClient", "LLMError", "KeysExhausted", "extract_json"]
