"""FastAPI 服务：上传 PDF / 提问 / 多轮追问。"""

from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from tablerag.api.schemas import AnswerOut, AskRequest, ChatRequest, DocumentOut
from tablerag.ingest.dataset import build_units
from tablerag.ingest.flatten import flatten_pdf
from tablerag.ingest.parser import parse_pdf
from tablerag.service import AppContext, build_context

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ctx = build_context(with_agent=True)
    app.state.ctx.settings.ensure_dirs()
    yield


app = FastAPI(title="Table RAG Agent", version="0.1.0", lifespan=lifespan)


def _ctx() -> AppContext:
    return app.state.ctx


@app.get("/health")
def health() -> dict:
    ctx = _ctx()
    return {
        "status": "ok",
        "llm_model": ctx.settings.llm.model,
        "embedding_model": ctx.settings.embedding.model,
        "rerank_model": ctx.settings.rerank.model,
        "index": ctx.retriever.stats(),
        "cache": {"backend": ctx.cache.name, "stats": ctx.cache.stats()},
    }


@app.get("/documents")
def list_documents() -> list[dict]:
    return _ctx().store.list_docs()


@app.post("/documents", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    doc_id: str | None = None,
    fidelity: str = "structured",
) -> DocumentOut:
    ctx = _ctx()
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只接受文本型 PDF（不做 OCR）")
    doc_key = doc_id or f"upload:{uuid.uuid4().hex[:12]}"
    target = ctx.settings.var_dir / "uploads" / f"{doc_key.replace(':', '_')}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    if fidelity == "flatten":
        doc = flatten_pdf(target, doc_key)
    else:
        doc = parse_pdf(target, doc_key)
    if not doc.tables and not doc.paragraphs:
        raise HTTPException(status_code=422, detail="PDF 中没有解析到可索引内容")
    ctx.retriever.index_docs([doc])
    return DocumentOut(
        doc_id=doc.doc_id,
        source=Path(file.filename or target.name).name,
        n_pages=doc.n_pages,
        n_tables=len(doc.tables),
        n_paragraphs=len(doc.paragraphs),
        method=doc.method,
        indexed_units=len(build_units(doc)),
    )


@app.post("/ask", response_model=AnswerOut)
def ask(request: AskRequest) -> AnswerOut:
    answer = _ctx().agent.ask(request.question, doc_id=request.doc_id, hyde=request.hyde)
    return AnswerOut.from_answer(answer)


@app.post("/chat", response_model=AnswerOut)
def chat(request: ChatRequest) -> AnswerOut:
    answer = _ctx().agent.ask(
        request.question, doc_id=request.doc_id, thread_id=request.thread_id,
        hyde=request.hyde,
    )
    return AnswerOut.from_answer(answer)
