"""本地 trace：记录 LangGraph 节点流转、耗时、token。

需求文档允许 LangFuse 自托管卡住时降级为本地 JSON trace；本实现即该降级方案，
每条 trace 落 var/traces/<trace_id>.json，便于 bad case 复盘。
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class Tracer:
    def __init__(self, trace_dir: Path, trace_id: str | None = None, sink: str = "jsonl"):
        self.trace_dir = Path(trace_dir)
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.sink = sink
        self.started_at = time.time()
        self.spans: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def event(self, name: str, **attrs: Any) -> None:
        self.events.append(
            {"ts": round((time.time() - self.started_at) * 1000, 2), "name": name, **attrs}
        )

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = dict(attrs)
        start = time.time()
        try:
            yield payload
        finally:
            self.spans.append(
                {
                    "name": name,
                    "latency_ms": round((time.time() - start) * 1000, 2),
                    **payload,
                }
            )

    def finish(self, **summary: Any) -> dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "total_latency_ms": round((time.time() - self.started_at) * 1000, 2),
            "spans": self.spans,
            "events": self.events,
            "summary": summary,
        }
        if self.sink == "jsonl":
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.trace_dir / f"{self.trace_id}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
