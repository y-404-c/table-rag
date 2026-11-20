"""LangGraph 编排：路由 → 检索 → 证据判断 →（HyDE 重查 | 计算）→ 生成。

条件边的存在理由：
- 证据不足回退重查（HyDE，可开关、有 LoopGuard 上限）；
- 数值问题必须走 pandas 工具而不是让 LLM 心算。
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langgraph.graph import END, StateGraph

from tablerag.agent import prompts
from tablerag.agent.checkpointer import CheckpointerFactory
from tablerag.agent.state import AgentState
from tablerag.agent.tools import ToolError, dispatch, load_frame, tool_specs
from tablerag.config import Settings
from tablerag.domain.models import Answer, Citation, ComputeResult, ScoredUnit
from tablerag.llm.client import ChatClient, LLMError
from tablerag.observability.trace import Tracer
from tablerag.retrieval.pipeline import HybridRetriever
from tablerag.storage.db import Store

logger = logging.getLogger(__name__)

REFUSAL_TEMPLATE = "Insufficient evidence in the provided document to answer this question."
MAX_HISTORY_TURNS = 6


def _evidence_block(evidence: list[dict[str, Any]], limit: int = 10) -> str:
    lines = []
    for item in evidence[:limit]:
        text = item.get("text", "")[:700]
        meta = f"page={item.get('page')} table={item.get('table_id')} row={item.get('row')}"
        lines.append(f"[{item['unit_id']}] ({item.get('kind')}, {meta})\n{text}")
    return "\n\n".join(lines) if lines else "(no evidence retrieved)"


def _history_block(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = [f"Q: {turn['question']}\nA: {turn['answer']}" for turn in history[-MAX_HISTORY_TURNS:]]
    return "Previous turns:\n" + "\n".join(lines) + "\n\n"


class RAGAgent:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        retriever: HybridRetriever,
        client: ChatClient | None = None,
        checkpointer: Any | None = None,
    ):
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.client = client or ChatClient(settings.llm)
        self.checkpointer = checkpointer
        self.graph = self._build()

    # ---------- 图 ----------
    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("route", self._node_route)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("grade", self._node_grade)
        graph.add_node("rewrite", self._node_rewrite)
        graph.add_node("compute", self._node_compute)
        graph.add_node("generate", self._node_generate)
        graph.set_entry_point("route")
        graph.add_edge("route", "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade",
            self._after_grade,
            {"rewrite": "rewrite", "compute": "compute", "generate": "generate"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("compute", "generate")
        graph.add_edge("generate", END)
        return graph.compile(checkpointer=self.checkpointer)

    # ---------- 节点 ----------
    def _node_route(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        payload = f"{_history_block(state.get('history', []))}Question: {question}"
        try:
            result = self.client.json(
                [{"role": "system", "content": prompts.ROUTE_SYSTEM},
                 {"role": "user", "content": payload}],
                max_tokens=200,
            )
            route = str(result.get("route", "table_lookup"))
            if route not in {"table_calc", "table_lookup", "text_qa"}:
                route = "table_lookup"
            reason = str(result.get("reason", ""))
        except LLMError as exc:
            logger.warning("路由失败，退化为 table_lookup：%s", exc)
            route, reason = "table_lookup", f"route_fallback: {exc}"
        # 多轮：追问句（"那第二年呢"）本身缺少主语/指标名，检索时把最近两轮问题带上
        history = state.get("history", [])
        recent = " ".join(turn["question"] for turn in history[-2:]).strip()
        query = f"{recent} {question}".strip() if recent else question
        return {
            "route": route,
            "route_reason": reason,
            "query": query,
            "steps": self._step(state, "route", route=route, reason=reason),
        }

    def _node_retrieve(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query") or state["question"]
        candidates, trace = self.retriever.search(query, doc_id=state.get("doc_id"))
        evidence = [self._evidence_item(item) for item in candidates]
        return {
            "evidence": evidence,
            "steps": self._step(
                state, "retrieve", query=query, hits=len(evidence),
                bm25=trace.bm25_hits, vector=trace.vector_hits,
                fused=trace.fused, reranked=trace.reranked, rerank_used=trace.rerank_used,
                notes=trace.notes,
            ),
        }

    def _node_grade(self, state: AgentState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        if not evidence:
            return {
                "sufficient": False,
                "grade_reason": "no_evidence",
                "steps": self._step(state, "grade", sufficient=False, reason="no_evidence"),
            }
        payload = (
            f"Question: {state['question']}\n\nEvidence:\n{_evidence_block(evidence)}"
        )
        try:
            result = self.client.json(
                [{"role": "system", "content": prompts.GRADE_SYSTEM},
                 {"role": "user", "content": payload}],
                max_tokens=250,
            )
            sufficient = bool(result.get("sufficient", False))
            reason = str(result.get("reason", "")) + " " + str(result.get("missing", ""))
        except LLMError as exc:
            logger.warning("证据判断失败，按有证据处理：%s", exc)
            sufficient, reason = bool(evidence), f"grade_fallback: {exc}"
        return {
            "sufficient": sufficient,
            "grade_reason": reason.strip(),
            "steps": self._step(state, "grade", sufficient=sufficient, reason=reason.strip()),
        }

    def _after_grade(self, state: AgentState) -> str:
        if not state.get("sufficient", False):
            rounds = state.get("hyde_rounds", 0)
            if state.get("use_hyde") and rounds < self.settings.retrieval.max_hyde_rounds:
                return "rewrite"
            return "generate"
        if (
            self.settings.agent.compute_enabled
            and state.get("route") in {"table_calc", "table_lookup"}
        ):
            return "compute"
        return "generate"

    def _node_rewrite(self, state: AgentState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        payload = (
            f"Question: {state['question']}\n"
            f"Current query: {state.get('query', '')}\n"
            f"Evidence so far:\n{_evidence_block(evidence, limit=5)}"
        )
        try:
            result = self.client.json(
                [{"role": "system", "content": prompts.HYDE_SYSTEM},
                 {"role": "user", "content": payload}],
                max_tokens=300,
            )
            hypothesis = str(result.get("hypothesis", ""))
            keywords = str(result.get("query", ""))
            new_query = " ".join(part for part in (state["question"], keywords, hypothesis) if part)
        except LLMError as exc:
            logger.warning("HyDE 改写失败：%s", exc)
            hypothesis, new_query = "", state.get("query") or state["question"]
        rounds = state.get("hyde_rounds", 0) + 1
        return {
            "query": new_query,
            "hyde_rounds": rounds,
            "steps": self._step(state, "hyde_rewrite", round=rounds,
                                hypothesis=hypothesis[:300], query=new_query[:300]),
        }

    def _node_compute(self, state: AgentState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        table_ids = sorted({item["table_id"] for item in evidence if item.get("table_id")})
        if not table_ids:
            return {
                "compute": None,
                "compute_error": "no_table_in_evidence",
                "steps": self._step(state, "compute", ok=False, error="no_table_in_evidence"),
            }
        tables_hint = "\n".join(
            f"{tid}: header={self._table_header(tid)}" for tid in table_ids[:3]
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.COMPUTE_SYSTEM},
            {"role": "user", "content":
             f"{_history_block(state.get('history', []))}"
             f"Question: {state['question']}\n\nAvailable tables:\n{tables_hint}\n\n"
             f"Evidence rows:\n{_evidence_block(evidence, limit=8)}"},
        ]
        last_error = ""
        for _attempt in range(2):
            try:
                message = self.client.chat(messages, tools=tool_specs(), max_tokens=400)
            except LLMError as exc:
                last_error = f"llm_error: {exc}"
                break
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                break
            call = tool_calls[0]
            try:
                args = json.loads(call.function.arguments or "{}")
                result = dispatch(call.function.name, args, self.store)
                return {
                    "compute": result.model_dump(),
                    "compute_error": "",
                    "steps": self._step(state, "compute", ok=True,
                                        op=result.op, value=result.value,
                                        expression=result.expression),
                }
            except (ToolError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = str(exc)
                messages += [
                    {"role": "assistant", "content": "",
                     "tool_calls": [{"id": call.id, "type": "function",
                                     "function": {"name": call.function.name,
                                                  "arguments": call.function.arguments}}]},
                    {"role": "tool", "tool_call_id": call.id, "content": f"ERROR: {last_error}"},
                ]
            except Exception as exc:  # noqa: BLE001 - 工具内部异常不能打断整条链路
                logger.exception("计算工具异常")
                last_error = f"internal_error: {exc}"
                break
        return {
            "compute": None,
            "compute_error": last_error or "no_tool_call",
            "steps": self._step(state, "compute", ok=False,
                                error=last_error or "no_tool_call"),
        }

    def _node_generate(self, state: AgentState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        compute_raw = state.get("compute")
        compute = (
            compute_raw if isinstance(compute_raw, ComputeResult)
            else (ComputeResult(**compute_raw) if compute_raw else None)
        )

        if not state.get("sufficient", False):
            reason = state.get("grade_reason") or "evidence_insufficient"
            return {
                "answer": REFUSAL_TEMPLATE,
                "refused": True,
                "refused_reason": reason,
                "citations": [],
                "history": self._append_history(state, REFUSAL_TEMPLATE),
                "steps": self._step(state, "generate", refused=True, reason=reason),
            }

        # 设计要求：数值题必须由工具算出，工具失败时不允许模型心算
        if state.get("route") == "table_calc" and compute is None:
            reason = f"computation_required_but_tool_failed: {state.get('compute_error', '')}"
            message = ("Cannot produce a reliable numeric answer: the calculation tool could not "
                       "locate the required cells.")
            return {
                "answer": message,
                "refused": True,
                "refused_reason": reason,
                "citations": [],
                "history": self._append_history(state, message),
                "steps": self._step(state, "generate", refused=True, reason=reason),
            }

        compute_block = ""
        if compute is not None:
            compute_block = (
                "\nComputed result (authoritative, must be copied verbatim into the answer):\n"
                + compute.to_prompt()
            )
            for item in compute.inputs:
                compute_block += (
                    f"\n  from page {item.get('page')} table {item.get('table_id')} "
                    f"row {item.get('row')} col {item.get('col')} = {item.get('text')}"
                )

        payload = (
            f"{_history_block(state.get('history', []))}"
            f"Question: {state['question']}\n\nEvidence:\n{_evidence_block(evidence)}"
            f"{compute_block}"
        )
        try:
            result = self.client.json(
                [{"role": "system", "content": prompts.GENERATE_SYSTEM},
                 {"role": "user", "content": payload}],
                max_tokens=500,
            )
        except LLMError as exc:
            logger.warning("生成失败，降级为证据摘录：%s", exc)
            fallback = compute.display if compute else (evidence[0]["text"][:300] if evidence else "")
            return {
                "answer": fallback,
                "refused": False,
                "refused_reason": "",
                "citations": [self._citation(evidence[0]).model_dump()] if evidence else [],
                "history": self._append_history(state, fallback),
                "steps": self._step(state, "generate", degraded=str(exc)),
            }

        answer_text = str(result.get("answer", "")).strip()
        refused = bool(result.get("refused", False))
        if refused:
            answer_text = answer_text or REFUSAL_TEMPLATE
        citations = self._validate_citations(result.get("citations", []), evidence, compute)
        if compute is not None and not refused:
            answer_text = self._ensure_computed_value(answer_text, compute)
        return {
            "answer": answer_text,
            "refused": refused,
            "refused_reason": str(result.get("refused_reason", "")),
            "citations": [c.model_dump() for c in citations],
            "history": self._append_history(state, answer_text),
            "steps": self._step(state, "generate", refused=refused,
                                citations=[c.text[:80] for c in citations]),
        }

    # ---------- 对外 ----------
    def ask(
        self,
        question: str,
        doc_id: str | None = None,
        thread_id: str | None = None,
        hyde: bool | None = None,
    ) -> Answer:
        tracer = Tracer(self.settings.trace_dir, sink=self.settings.agent.trace_sink)
        started = time.time()
        before = dict(self.client.usage)
        payload: dict[str, Any] = {
            "question": question,
            "doc_id": doc_id,
            "use_hyde": self.settings.retrieval.hyde if hyde is None else hyde,
            "query": question,
            "evidence": [],
            "sufficient": False,
            "grade_reason": "",
            "hyde_rounds": 0,
            "compute": None,
            "compute_error": "",
            "answer": "",
            "refused": False,
            "refused_reason": "",
            "citations": [],
            "steps": [],
        }
        config = {"configurable": {"thread_id": thread_id or tracer.trace_id}}
        with tracer.span("graph"):
            final = self.graph.invoke(payload, config=config)
        after = self.client.usage
        usage = {key: after[key] - before.get(key, 0) for key in after}
        latency_ms = int((time.time() - started) * 1000)
        tracer.event("answer", route=final.get("route"), refused=final.get("refused"))
        tracer.finish(question=question, usage=usage, latency_ms=latency_ms,
                      steps=final.get("steps", []))
        raw_citations = final.get("citations", []) or []
        citations = [
            item if isinstance(item, Citation) else Citation(**item) for item in raw_citations
        ]
        compute_raw = final.get("compute")
        compute = (
            compute_raw if isinstance(compute_raw, ComputeResult)
            else (ComputeResult(**compute_raw) if compute_raw else None)
        )
        return Answer(
            answer=final.get("answer", ""),
            citations=citations,
            refused=bool(final.get("refused", False)),
            refused_reason=final.get("refused_reason", ""),
            used_compute=compute is not None,
            compute=compute,
            route=final.get("route", ""),
            trace_id=tracer.trace_id,
            latency_ms=latency_ms,
            token_usage=usage,
            hyde_rounds=int(final.get("hyde_rounds", 0)),
        )

    # ---------- 辅助 ----------
    def _evidence_item(self, scored: ScoredUnit) -> dict[str, Any]:
        unit = scored.unit
        return {
            "unit_id": unit.unit_id,
            "doc_id": unit.doc_id,
            "kind": unit.kind,
            "text": unit.text,
            "page": unit.page,
            "table_id": unit.table_id,
            "row": unit.row,
            "col": unit.col,
            "score": round(scored.score, 4),
            "bm25": scored.bm25_score,
            "vector": round(scored.vector_score, 4) if scored.vector_score is not None else None,
            "rerank": round(scored.rerank_score, 4) if scored.rerank_score is not None else None,
        }

    def _validate_citations(
        self, raw: Any, evidence: list[dict[str, Any]], compute: ComputeResult | None
    ) -> list[Citation]:
        allowed = {item["unit_id"]: item for item in evidence}
        citations: list[Citation] = []
        seen_cells: set[tuple] = set()

        # 计算结果的每个输入单元格都单独给出处：这才是 cell 级溯源
        if compute is not None:
            for record in compute.inputs:
                key = (record.get("table_id"), record.get("row"), record.get("col"))
                if key in seen_cells:
                    continue
                seen_cells.add(key)
                citations.append(
                    Citation(doc_id=evidence[0]["doc_id"] if evidence else "",
                             table_id=record.get("table_id"), page=record.get("page"),
                             row=record.get("row"), col=record.get("col"),
                             text=str(record.get("text", "")))
                )

        ids = raw if isinstance(raw, list) else []
        for item in ids:
            unit_id = str(item)
            unit = allowed.get(unit_id)
            if unit is None:
                continue
            citation = self._citation(unit)
            key = (citation.table_id, citation.row, citation.col)
            if key in seen_cells:
                continue
            seen_cells.add(key)
            citations.append(citation)
        return citations

    @staticmethod
    def _citation(item: dict[str, Any]) -> Citation:
        return Citation(
            doc_id=item.get("doc_id", ""),
            table_id=item.get("table_id"),
            page=item.get("page"),
            row=item.get("row"),
            col=item.get("col"),
            text=(item.get("text") or "")[:200],
        )

    @staticmethod
    def _ensure_computed_value(answer: str, compute: ComputeResult) -> str:
        """工具算出的数值必须出现在答案里；模型漏写就补上，避免"答对了但没出处"。

        只要答案里已有与计算值一致的数值（允许千分位/单位写法差异）就不动它，
        避免把"$1,234 million"强行改写成带表达式后缀，进而让 EM 变 0。
        """
        from tablerag.domain.text import numbers_match, parse_number

        if compute.value is not None:
            for token in re.findall(r"[-+]?\(?\$?\d[\d,]*(?:\.\d+)?\)?%?", answer):
                value = parse_number(token)
                if value is not None and numbers_match(value, compute.value):
                    return answer
        if compute.display and compute.display in answer:
            return answer
        suffix = f" ({compute.expression} = {compute.display})"
        return f"{answer}{suffix}" if answer else compute.display

    @staticmethod
    def _append_history(state: AgentState, answer: str) -> list[dict[str, Any]]:
        history = list(state.get("history", []))
        history.append({"question": state["question"], "answer": answer,
                        "route": state.get("route", "")})
        return history[-MAX_HISTORY_TURNS:]

    @staticmethod
    def _step(state: AgentState, name: str, **payload: Any) -> list[dict[str, Any]]:
        steps = list(state.get("steps", []))
        steps.append({"node": name, **payload})
        return steps

    def _table_header(self, table_id: str) -> str:
        table = self.store.get_table(table_id)
        if table is None:
            return "(missing)"
        try:
            return " | ".join(str(c) for c in load_frame(table).columns if c != "__row__")
        except Exception:  # noqa: BLE001
            return " | ".join(table.cells[0]) if table.cells else ""


def build_agent(settings: Settings, store: Store, retriever: HybridRetriever,
                client: ChatClient | None = None,
                with_checkpointer: bool = True) -> RAGAgent:
    checkpointer = CheckpointerFactory(settings).build() if with_checkpointer else None
    return RAGAgent(settings, store, retriever, client=client, checkpointer=checkpointer)
