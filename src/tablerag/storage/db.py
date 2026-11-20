"""SQLite 仓储：文档、表格、段落、检索单元、缓存。"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from tablerag.domain.models import Paragraph, ParsedDoc, TableGrid, Unit

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT,
    method TEXT,
    n_pages INTEGER,
    meta TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS tables (
    table_id TEXT PRIMARY KEY,
    doc_id TEXT,
    title TEXT,
    page INTEGER,
    n_rows INTEGER,
    n_cols INTEGER,
    header_rows INTEGER,
    cells TEXT,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS paragraphs (
    pid TEXT PRIMARY KEY,
    doc_id TEXT,
    page INTEGER,
    ord INTEGER,
    text TEXT
);
CREATE TABLE IF NOT EXISTS units (
    unit_id TEXT PRIMARY KEY,
    doc_id TEXT,
    kind TEXT,
    text TEXT,
    page INTEGER,
    table_id TEXT,
    row INTEGER,
    col INTEGER,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT,
    expires_at REAL
);
CREATE INDEX IF NOT EXISTS idx_tables_doc ON tables(doc_id);
CREATE INDEX IF NOT EXISTS idx_paragraphs_doc ON paragraphs(doc_id);
CREATE INDEX IF NOT EXISTS idx_units_doc ON units(doc_id);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------- 文档 ----------
    def save_doc(self, doc: ParsedDoc, units: list[Unit]) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id=?", (doc.doc_id,))
            conn.execute("DELETE FROM tables WHERE doc_id=?", (doc.doc_id,))
            conn.execute("DELETE FROM paragraphs WHERE doc_id=?", (doc.doc_id,))
            conn.execute("DELETE FROM units WHERE doc_id=?", (doc.doc_id,))
            conn.execute(
                "INSERT INTO documents(doc_id, source, method, n_pages, meta, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (doc.doc_id, doc.source, doc.method, doc.n_pages,
                 json.dumps(doc.meta, ensure_ascii=False), time.time()),
            )
            conn.executemany(
                "INSERT INTO tables(table_id, doc_id, title, page, n_rows, n_cols,"
                " header_rows, cells, meta) VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (t.table_id, t.doc_id, t.title, t.page, t.n_rows, t.n_cols,
                     t.header_rows, json.dumps(t.cells, ensure_ascii=False),
                     json.dumps(t.meta, ensure_ascii=False))
                    for t in doc.tables
                ],
            )
            conn.executemany(
                "INSERT INTO paragraphs(pid, doc_id, page, ord, text) VALUES (?,?,?,?,?)",
                [(p.pid, p.doc_id, p.page, p.order, p.text) for p in doc.paragraphs],
            )
            conn.executemany(
                "INSERT OR REPLACE INTO units(unit_id, doc_id, kind, text, page, table_id,"
                " row, col, meta) VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (u.unit_id, u.doc_id, u.kind, u.text, u.page, u.table_id, u.row, u.col,
                     json.dumps(u.meta, ensure_ascii=False))
                    for u in units
                ],
            )

    def get_doc(self, doc_id: str) -> ParsedDoc | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
            if row is None:
                return None
            tables = [
                TableGrid(
                    table_id=r["table_id"],
                    doc_id=r["doc_id"],
                    title=r["title"] or "",
                    page=r["page"],
                    cells=json.loads(r["cells"]),
                    n_rows=r["n_rows"],
                    n_cols=r["n_cols"],
                    header_rows=r["header_rows"] or 1,
                    meta=json.loads(r["meta"] or "{}"),
                )
                for r in conn.execute(
                    "SELECT * FROM tables WHERE doc_id=? ORDER BY page, table_id", (doc_id,)
                )
            ]
            paragraphs = [
                Paragraph(pid=r["pid"], doc_id=r["doc_id"], page=r["page"],
                          order=r["ord"], text=r["text"])
                for r in conn.execute(
                    "SELECT * FROM paragraphs WHERE doc_id=? ORDER BY ord", (doc_id,)
                )
            ]
        return ParsedDoc(
            doc_id=row["doc_id"],
            source=row["source"] or "",
            method=row["method"] or "structured",
            n_pages=row["n_pages"] or 0,
            tables=tables,
            paragraphs=paragraphs,
            meta=json.loads(row["meta"] or "{}"),
        )

    def list_docs(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT d.doc_id, d.source, d.method, d.n_pages, d.created_at,"
                " (SELECT COUNT(*) FROM tables t WHERE t.doc_id=d.doc_id) AS n_tables,"
                " (SELECT COUNT(*) FROM paragraphs p WHERE p.doc_id=d.doc_id) AS n_paragraphs"
                " FROM documents d ORDER BY d.created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def has_doc(self, doc_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute(
                "SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone() is not None

    def delete_doc(self, doc_id: str) -> None:
        with self._conn() as conn:
            for table in ("documents", "tables", "paragraphs", "units"):
                conn.execute(f"DELETE FROM {table} WHERE doc_id=?", (doc_id,))

    def get_table(self, table_id: str) -> TableGrid | None:
        with self._conn() as conn:
            r = conn.execute("SELECT * FROM tables WHERE table_id=?", (table_id,)).fetchone()
        if r is None:
            return None
        return TableGrid(
            table_id=r["table_id"], doc_id=r["doc_id"], title=r["title"] or "",
            page=r["page"], cells=json.loads(r["cells"]), n_rows=r["n_rows"],
            n_cols=r["n_cols"], header_rows=r["header_rows"] or 1,
            meta=json.loads(r["meta"] or "{}"),
        )

    def get_units(self, doc_id: str | None = None) -> list[Unit]:
        with self._conn() as conn:
            if doc_id:
                rows = conn.execute(
                    "SELECT * FROM units WHERE doc_id=? ORDER BY unit_id", (doc_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM units ORDER BY unit_id").fetchall()
        return [
            Unit(
                unit_id=r["unit_id"], doc_id=r["doc_id"], kind=r["kind"], text=r["text"],
                page=r["page"], table_id=r["table_id"], row=r["row"], col=r["col"],
                meta=json.loads(r["meta"] or "{}"),
            )
            for r in rows
        ]

    def count_units(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM units").fetchone()[0])

    # ---------- 缓存表（SQLite 后端的存储介质） ----------
    def cache_get(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] and row["expires_at"] < time.time():
                conn.execute("DELETE FROM cache WHERE key=?", (key,))
                return None
            return row["value"]

    def cache_set(self, key: str, value: str, ttl_s: int) -> None:
        expires = time.time() + ttl_s if ttl_s > 0 else 0.0
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache(key, value, expires_at) VALUES (?,?,?)",
                (key, value, expires),
            )

    def cache_stats(self) -> dict[str, int]:
        with self._conn() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0])
            expired = int(
                conn.execute(
                    "SELECT COUNT(*) FROM cache WHERE expires_at>0 AND expires_at<?",
                    (time.time(),),
                ).fetchone()[0]
            )
        return {"entries": total, "expired": expired}
