from __future__ import annotations

import re
from pathlib import Path

from aar.storage.db import Database

CHUNK_SIZE = 700


class DocumentIndex:
    def __init__(self, db: Database) -> None:
        self.db = db

    def count(self) -> int:
        with self.db.lock:
            row = self.db.conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return int(row["n"] if row else 0)

    def index_directory(self, corpus_dir: Path) -> int:
        added = 0
        for path in sorted(corpus_dir.glob("*.txt")):
            added += self.index_text(path.stem, path.stem.replace("_", " "), path.read_text(encoding="utf-8"))
        return added

    def index_text(self, doc_id: str, title: str, text: str) -> int:
        chunks = _chunk_text(text)
        added = 0
        with self.db.transaction() as conn:
            for index, chunk in enumerate(chunks, start=1):
                chunk_id = f"c{index:03d}"
                citation = f"doc:{doc_id}#chunk:{chunk_id}"
                existing = conn.execute(
                    "SELECT 1 FROM documents WHERE doc_id = ? AND chunk_id = ?",
                    (doc_id, chunk_id),
                ).fetchone()
                if existing:
                    continue
                conn.execute(
                    """
                    INSERT INTO documents (doc_id, chunk_id, title, text, citation)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (doc_id, chunk_id, title, chunk, citation),
                )
                conn.execute(
                    """
                    INSERT INTO documents_fts (doc_id, chunk_id, title, text, citation)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (doc_id, chunk_id, title, chunk, citation),
                )
                added += 1
        return added

    def search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        terms = [t for t in re.findall(r"[A-Za-z0-9]{3,}", query) if t.lower() not in {"the", "and"}]
        if not terms:
            return []
        match = " OR ".join(terms)
        with self.db.lock:
            try:
                rows = self.db.conn.execute(
                    """
                    SELECT d.doc_id, d.chunk_id, d.title, d.text, d.citation
                    FROM documents_fts f
                    JOIN documents d ON d.doc_id = f.doc_id AND d.chunk_id = f.chunk_id
                    WHERE documents_fts MATCH ?
                    LIMIT ?
                    """,
                    (match, limit),
                ).fetchall()
            except Exception:
                rows = []
            if rows:
                return [dict(row) for row in rows]
            like_rows: list[dict[str, str]] = []
            for term in terms[:4]:
                found = self.db.conn.execute(
                    """
                    SELECT doc_id, chunk_id, title, text, citation
                    FROM documents
                    WHERE lower(text) LIKE ?
                    LIMIT ?
                    """,
                    (f"%{term.lower()}%", limit),
                ).fetchall()
                like_rows.extend(dict(row) for row in found)
        seen: set[str] = set()
        unique: list[dict[str, str]] = []
        for row in like_rows:
            if row["citation"] in seen:
                continue
            seen.add(row["citation"])
            unique.append(row)
            if len(unique) >= limit:
                break
        return unique


def _chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) > CHUNK_SIZE:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current.strip())
    return chunks or [text.strip()]
