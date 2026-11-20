"""装配层：把配置、存储、缓存、检索、Agent 拼起来，供 API / CLI / eval 共用。

所有组件共享同一个 KeyPool：任何一个调用发现 key 额度耗尽，其他组件立刻跟着换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tablerag.agent.graph import RAGAgent, build_agent
from tablerag.config import Settings, get_settings
from tablerag.llm.client import ChatClient
from tablerag.llm.embeddings import Embedder
from tablerag.llm.keypool import KeyPool
from tablerag.llm.rerank import Reranker
from tablerag.retrieval.pipeline import HybridRetriever
from tablerag.storage.cache import build_cache
from tablerag.storage.db import Store


@dataclass
class AppContext:
    settings: Settings
    store: Store
    cache: Any
    retriever: HybridRetriever
    keypool: KeyPool | None = None
    _client: ChatClient | None = None
    _agent: RAGAgent | None = None
    _eval_agent: RAGAgent | None = None
    _config_lock: Any = field(default=None, repr=False)

    @property
    def client(self) -> ChatClient:
        if self._client is None:
            self._client = ChatClient(self.settings.llm, keypool=self.keypool)
        return self._client

    @property
    def agent(self) -> RAGAgent:
        if self._agent is None:
            self._agent = build_agent(
                self.settings, self.store, self.retriever, client=self.client,
                with_checkpointer=True,
            )
        return self._agent

    @property
    def eval_agent(self) -> RAGAgent:
        """评测用：不落 checkpoint（少一次 SQLite 写），可安全并发调用。"""
        if self._eval_agent is None:
            self._eval_agent = build_agent(
                self.settings, self.store, self.retriever, client=self.client,
                with_checkpointer=False,
            )
        return self._eval_agent


def build_context(settings: Settings | None = None, with_agent: bool = False) -> AppContext:
    settings = settings or get_settings()
    store = Store(settings.db_path)
    cache = build_cache(
        settings.cache.backend, store, settings.cache.url, settings.cache.ttl_s
    )
    keys = settings.llm.resolved_keys()
    keypool = KeyPool(keys) if keys else None
    retriever = HybridRetriever(
        settings, store,
        Embedder(settings.embedding, cache=cache, keypool=keypool),
        Reranker(settings.rerank, keypool=keypool),
    )
    ctx = AppContext(settings=settings, store=store, cache=cache, retriever=retriever,
                     keypool=keypool)
    if with_agent:
        ctx._agent = build_agent(settings, store, retriever, client=ctx.client,
                                 with_checkpointer=True)
    return ctx
