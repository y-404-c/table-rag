from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    doc_id: str | None
    history: list[dict[str, Any]]
    use_hyde: bool
    route: str
    route_reason: str
    query: str
    evidence: list[dict[str, Any]]
    sufficient: bool
    grade_reason: str
    hyde_rounds: int
    compute: dict[str, Any] | None
    compute_error: str
    answer: str
    refused: bool
    refused_reason: str
    citations: list[dict[str, Any]]
    steps: list[dict[str, Any]]
