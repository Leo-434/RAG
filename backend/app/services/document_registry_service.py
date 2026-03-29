"""SQLite-backed document metadata registry.

Replaces the in-memory dict (doc_registry) so document records survive
backend restarts. Uses the synchronous sqlite3 stdlib module — document
operations are infrequent and do not need async.
"""

import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.models.document import DocumentMeta, IngestStatus
from app.utils.logger import get_logger

log = get_logger(__name__)

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'uploading',
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    entity_count    INTEGER NOT NULL DEFAULT 0,
    relation_count  INTEGER NOT NULL DEFAULT 0,
    ingested_at     TEXT,
    error_msg       TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
"""


class DocumentRegistryService:
    """Persists DocumentMeta records in SQLite so they survive process restarts."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()          # serialise writes (reads are lock-free)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        log.info("document_registry.initialized", path=db_path)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as db:
                db.executescript(_DDL)

    # ── Public CRUD ──────────────────────────────────────────────────────────

    def create_doc(self, filename: str, file_size: int) -> str:
        """Insert a new document record (status=uploading) and return its doc_id."""
        doc_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self._lock:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO documents (doc_id, filename, file_size_bytes, status, created_at)"
                    " VALUES (?, ?, ?, 'uploading', ?)",
                    (doc_id, filename, file_size, now),
                )
        log.info("document_registry.created", doc_id=doc_id, filename=filename)
        return doc_id

    def get_doc(self, doc_id: str) -> Optional[DocumentMeta]:
        """Return metadata for a single document, or None if not found."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return self._row_to_meta(row) if row else None

    def list_docs(self) -> list[DocumentMeta]:
        """Return all documents sorted by creation time (newest first)."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_meta(r) for r in rows]

    def save_doc(self, meta: DocumentMeta) -> None:
        """Persist (UPDATE) a DocumentMeta that already exists in the DB."""
        ingested_at: Optional[str] = None
        if meta.ingested_at is not None:
            ingested_at = (
                meta.ingested_at.isoformat()
                if isinstance(meta.ingested_at, datetime)
                else str(meta.ingested_at)
            )
        with self._lock:
            with self._connect() as db:
                db.execute(
                    """
                    UPDATE documents
                    SET filename        = ?,
                        file_size_bytes = ?,
                        status          = ?,
                        chunk_count     = ?,
                        entity_count    = ?,
                        relation_count  = ?,
                        ingested_at     = ?,
                        error_msg       = ?
                    WHERE doc_id = ?
                    """,
                    (
                        meta.filename,
                        meta.file_size_bytes,
                        meta.status.value,
                        meta.chunk_count,
                        meta.entity_count,
                        meta.relation_count,
                        ingested_at,
                        meta.error_msg,
                        meta.doc_id,
                    ),
                )
        log.debug("document_registry.saved", doc_id=meta.doc_id, status=meta.status.value)

    def delete_doc(self, doc_id: str) -> bool:
        """Remove a document record. Returns True if a row was actually deleted."""
        with self._lock:
            with self._connect() as db:
                result = db.execute(
                    "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
                )
                deleted = result.rowcount > 0
        log.info("document_registry.deleted", doc_id=doc_id, deleted=deleted)
        return deleted

    # ── Conversion helper ─────────────────────────────────────────────────────

    @staticmethod
    def _row_to_meta(row: sqlite3.Row) -> DocumentMeta:
        ingested_at: Optional[datetime] = None
        if row["ingested_at"]:
            try:
                ingested_at = datetime.fromisoformat(row["ingested_at"])
            except ValueError:
                pass
        return DocumentMeta(
            doc_id=row["doc_id"],
            filename=row["filename"],
            file_size_bytes=row["file_size_bytes"],
            status=IngestStatus(row["status"]),
            chunk_count=row["chunk_count"] or 0,
            entity_count=row["entity_count"] or 0,
            relation_count=row["relation_count"] or 0,
            ingested_at=ingested_at,
            error_msg=row["error_msg"],
        )
