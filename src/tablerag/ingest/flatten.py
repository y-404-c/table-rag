"""拍平基线：pypdf 直接抽文本，不做任何结构恢复。

用途只有一个——作为对照，证明"表格拍平成纯文本"在 cell index accuracy 上会掉。
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from tablerag.domain.models import Paragraph, ParsedDoc, TableGrid
from tablerag.domain.text import normalize_cell


def flatten_pdf(path: Path | str, doc_id: str, chunk_chars: int = 600) -> ParsedDoc:
    reader = PdfReader(str(path))
    paragraphs: list[Paragraph] = []
    order = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        buffer: list[str] = []
        size = 0
        for line in text.splitlines():
            line = normalize_cell(line)
            if not line:
                continue
            buffer.append(line)
            size += len(line)
            if size >= chunk_chars:
                order += 1
                paragraphs.append(Paragraph(pid=f"{doc_id}:flat{order}",
                                            doc_id=doc_id, page=page_number,
                                            order=order, text=" ".join(buffer)))
                buffer, size = [], 0
        if buffer:
            order += 1
            paragraphs.append(Paragraph(pid=f"{doc_id}:flat{order}", doc_id=doc_id,
                                        page=page_number, order=order,
                                        text=" ".join(buffer)))
    # 拍平基线没有表格结构；保留一个空 TableGrid 仅为接口一致（n_cols=0 不会产生行单元）
    return ParsedDoc(
        doc_id=doc_id,
        source=str(path),
        method="flatten",
        n_pages=len(reader.pages),
        tables=[TableGrid(table_id=f"{doc_id}:flat", doc_id=doc_id, page=1, cells=[],
                          n_rows=0, n_cols=0, header_rows=0,
                          meta={"flattened": True})],
        paragraphs=paragraphs,
        meta={"parser": "pypdf-flatten", "chunk_chars": chunk_chars},
    )
