from types import SimpleNamespace

from tablerag.agent.graph import REFUSAL_TEMPLATE, RAGAgent
from tablerag.config import LLMSettings, Settings
from tablerag.domain.models import ParsedDoc, ScoredUnit, TableGrid, Unit
from tablerag.ingest.dataset import build_units
from tablerag.retrieval.pipeline import RetrievalTrace
from tablerag.storage.db import Store


class FakeClient:
    """按 system prompt 返回固定 JSON，让图跑通但不产生真实调用。"""

    def __init__(self, route: str = "table_calc", sufficient: bool = False):
        self.route = route
        self.sufficient = sufficient
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.settings = SimpleNamespace(model="fake")

    def json(self, messages, **kwargs):
        system = messages[0]["content"]
        if "router" in system:
            return {"route": self.route, "reason": "fake"}
        if "gatekeeper" in system:
            return {"sufficient": self.sufficient, "reason": "fake", "missing": ""}
        return {"answer": "fake answer", "citations": [], "refused": False}

    def chat(self, messages, **kwargs):
        # 不返回 tool_calls -> compute 节点失败
        return SimpleNamespace(content="", tool_calls=[])

    def text(self, messages, **kwargs):
        return ""

    def begin_scope(self):
        pass

    def scope_usage(self):
        return dict(self.usage)


class FakeRetriever:
    def __init__(self, evidence: list[ScoredUnit] | None = None):
        self.evidence = evidence or []
        self.default_flags = None

    def search(self, query, doc_id=None, top_k=None, flags=None):
        return self.evidence, RetrievalTrace()


def _settings(tmp_path) -> Settings:
    settings = Settings(llm=LLMSettings(api_key="sk-fake"), _env_file=None)
    settings.var_dir = tmp_path
    settings.reports_dir = tmp_path / "reports"
    return settings


def test_refuses_without_evidence(tmp_path):
    agent = RAGAgent(_settings(tmp_path), Store(tmp_path / "a.db"), FakeRetriever(),
                     client=FakeClient(sufficient=False), checkpointer=None)
    answer = agent.ask("What is the revenue in 2019?")
    assert answer.refused
    assert answer.answer == REFUSAL_TEMPLATE
    assert answer.citations == []


def test_numeric_question_refuses_when_tool_fails(tmp_path):
    store = Store(tmp_path / "a.db")
    cells = [["", "2019"], ["Revenue", "100"]]
    doc = ParsedDoc(doc_id="doc", tables=[TableGrid(table_id="doc:t0", doc_id="doc",
                                                    cells=cells, n_rows=2, n_cols=2)])
    store.save_doc(doc, build_units(doc))
    unit = Unit(unit_id="doc:t0:r1", doc_id="doc", kind="table_row",
                text="row1 | 2019: 100", page=1, table_id="doc:t0", row=1)
    agent = RAGAgent(_settings(tmp_path), store,
                     FakeRetriever([ScoredUnit(unit=unit, rrf_score=1.0)]),
                     client=FakeClient(route="table_calc", sufficient=True),
                     checkpointer=None)
    answer = agent.ask("What is the revenue in 2019?", doc_id="doc")
    assert answer.refused
    assert "computation_required" in answer.refused_reason
    assert answer.used_compute is False


class _AnswerClient(FakeClient):
    """生成节点返回指定答案，其余走 FakeClient 的默认路由/闸门。"""

    def __init__(self, answer: str, route: str = "text_qa"):
        super().__init__(route=route, sufficient=True)
        self._answer = answer

    def json(self, messages, **kwargs):
        system = messages[0]["content"]
        if "router" in system:
            return {"route": self.route, "reason": "fake"}
        if "gatekeeper" in system:
            return {"sufficient": True, "reason": "fake", "missing": ""}
        return {"answer": self._answer, "citations": [], "refused": False}


def test_numeric_answer_gets_cell_coordinates_enriched(tmp_path):
    """非计算路径（text_qa）LLM 只给 unit_id → 行级出处；应确定性补全到具体单元格。"""
    store = Store(tmp_path / "a.db")
    cells = [["", "2019"], ["Revenue", "100"]]
    doc = ParsedDoc(doc_id="doc", tables=[TableGrid(table_id="doc:t0", doc_id="doc",
                                                    cells=cells, n_rows=2, n_cols=2)])
    store.save_doc(doc, build_units(doc))
    unit = Unit(unit_id="doc:t0:r1", doc_id="doc", kind="table_row",
                text="row1 | 2019: 100", page=1, table_id="doc:t0", row=1)
    agent = RAGAgent(_settings(tmp_path), store,
                     FakeRetriever([ScoredUnit(unit=unit, rrf_score=1.0)]),
                     client=_AnswerClient("The revenue was 100", route="text_qa"),
                     checkpointer=None)
    answer = agent.ask("What was the revenue?", doc_id="doc")
    assert not answer.refused
    cell_cites = [c for c in answer.citations if c.row is not None and c.col is not None]
    assert cell_cites, "数值答案必须能溯源到具体单元格（row+col）"
    assert any(c.table_id == "doc:t0" and c.row == 1 and c.col == 1 for c in cell_cites)


def test_refused_answer_does_not_enrich_cell_citations(tmp_path):
    """拒答答案不应拿拒答模板去误匹配单元格。"""
    store = Store(tmp_path / "a.db")
    cells = [["", "2019"], ["Revenue", "100"]]
    doc = ParsedDoc(doc_id="doc", tables=[TableGrid(table_id="doc:t0", doc_id="doc",
                                                    cells=cells, n_rows=2, n_cols=2)])
    store.save_doc(doc, build_units(doc))
    unit = Unit(unit_id="doc:t0:r1", doc_id="doc", kind="table_row",
                text="row1 | 2019: 100", page=1, table_id="doc:t0", row=1)
    agent = RAGAgent(_settings(tmp_path), store,
                     FakeRetriever([ScoredUnit(unit=unit, rrf_score=1.0)]),
                     client=FakeClient(route="text_qa", sufficient=False),
                     checkpointer=None)
    answer = agent.ask("What was the revenue?", doc_id="doc")
    assert answer.refused
    assert answer.citations == []
