from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    doc_id: str | None = None
    hyde: bool | None = None


class ChatRequest(AskRequest):
    thread_id: str = Field(min_length=1, max_length=128)


class CitationOut(BaseModel):
    doc_id: str
    table_id: str | None = None
    page: int | None = None
    row: int | None = None
    col: int | None = None
    text: str = ""


class AnswerOut(BaseModel):
    answer: str
    refused: bool
    refused_reason: str = ""
    citations: list[CitationOut] = Field(default_factory=list)
    used_compute: bool
    compute: dict[str, Any] | None = None
    route: str = ""
    trace_id: str = ""
    latency_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    hyde_rounds: int = 0

    @classmethod
    def from_answer(cls, answer) -> AnswerOut:
        return cls(
            answer=answer.answer,
            refused=answer.refused,
            refused_reason=answer.refused_reason,
            citations=[CitationOut(**c.model_dump()) for c in answer.citations],
            used_compute=answer.used_compute,
            compute=answer.compute.model_dump() if answer.compute else None,
            route=answer.route,
            trace_id=answer.trace_id,
            latency_ms=answer.latency_ms,
            token_usage=answer.token_usage,
            hyde_rounds=answer.hyde_rounds,
        )


class DocumentOut(BaseModel):
    doc_id: str
    source: str
    n_pages: int
    n_tables: int
    n_paragraphs: int
    method: str
    indexed_units: int = 0
