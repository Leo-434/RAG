"""SQLite-backed conversation and message persistence."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from app.models.conversation import Conversation, ConversationListItem, ConversationMessage
from app.models.query import RetrievalMode, SourceItem
from app.utils.logger import get_logger

log = get_logger(__name__)

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS conversations (
    id             TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    retrieval_mode TEXT NOT NULL DEFAULT 'hybrid',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    sources         TEXT NOT NULL DEFAULT '[]',
    latency_ms      INTEGER,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conv_updated  ON conversations(updated_at DESC);
"""


class ConversationService:
    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_DDL)
            await db.commit()
        log.info("conversation_service.db_initialized", path=self._db_path)

    # ── CRUD ────────────────────────────────────────────────────────────────

    async def create_conversation(
        self,
        title: Optional[str] = None,
        retrieval_mode: str = "hybrid",
    ) -> Conversation:
        now = datetime.now(timezone.utc).isoformat()
        conv_id = str(uuid.uuid4())
        title = (title or "新对话").strip() or "新对话"
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO conversations (id, title, retrieval_mode, created_at, updated_at) VALUES (?,?,?,?,?)",
                (conv_id, title, retrieval_mode, now, now),
            )
            await db.commit()
        log.info("conversation_service.created", id=conv_id, title=title)
        return Conversation(
            id=conv_id,
            title=title,
            retrieval_mode=RetrievalMode(retrieval_mode),
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            messages=[],
        )

    async def list_conversations(self) -> list[ConversationListItem]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT
                    c.id, c.title, c.retrieval_mode, c.created_at, c.updated_at,
                    COUNT(m.id) AS message_count,
                    MAX(CASE WHEN m.role = 'assistant' THEN m.content ELSE NULL END) AS last_answer
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """
            ) as cursor:
                rows = await cursor.fetchall()
        items = []
        for row in rows:
            preview = (row["last_answer"] or "")[:80]
            items.append(
                ConversationListItem(
                    id=row["id"],
                    title=row["title"],
                    retrieval_mode=RetrievalMode(row["retrieval_mode"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    message_count=row["message_count"],
                    preview=preview,
                )
            )
        return items

    async def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return None
            async with db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conv_id,),
            ) as cursor:
                msg_rows = await cursor.fetchall()

        messages = [
            ConversationMessage(
                id=r["id"],
                conversation_id=conv_id,
                role=r["role"],
                content=r["content"],
                sources=[SourceItem(**s) for s in json.loads(r["sources"] or "[]")],
                latency_ms=r["latency_ms"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in msg_rows
        ]
        return Conversation(
            id=row["id"],
            title=row["title"],
            retrieval_mode=RetrievalMode(row["retrieval_mode"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            messages=messages,
        )

    async def delete_conversation(self, conv_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            result = await db.execute(
                "DELETE FROM conversations WHERE id = ?", (conv_id,)
            )
            await db.commit()
            deleted = result.rowcount > 0
        log.info("conversation_service.deleted", id=conv_id, deleted=deleted)
        return deleted

    async def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        sources: Optional[list[SourceItem]] = None,
        latency_ms: Optional[int] = None,
    ) -> ConversationMessage:
        now = datetime.now(timezone.utc).isoformat()
        msg_id = str(uuid.uuid4())
        sources_list = sources or []
        sources_json = json.dumps([s.model_dump() for s in sources_list], ensure_ascii=False)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, sources, latency_ms, created_at) VALUES (?,?,?,?,?,?,?)",
                (msg_id, conv_id, role, content, sources_json, latency_ms, now),
            )
            await db.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conv_id),
            )
            await db.commit()
        return ConversationMessage(
            id=msg_id,
            conversation_id=conv_id,
            role=role,
            content=content,
            sources=sources_list,
            latency_ms=latency_ms,
            created_at=datetime.fromisoformat(now),
        )

    async def update_title(self, conv_id: str, title: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title[:60], conv_id),
            )
            await db.commit()
