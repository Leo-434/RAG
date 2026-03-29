"""Pydantic models for multi-turn conversation history."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.models.query import RetrievalMode, SourceItem


class ConversationMessage(BaseModel):
    id: str
    conversation_id: str
    role: str  # "user" | "assistant"
    content: str
    sources: list[SourceItem] = []
    latency_ms: Optional[int] = None
    created_at: datetime


class Conversation(BaseModel):
    id: str
    title: str
    retrieval_mode: RetrievalMode = RetrievalMode.hybrid
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessage] = []


class ConversationListItem(BaseModel):
    id: str
    title: str
    retrieval_mode: RetrievalMode = RetrievalMode.hybrid
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    preview: str = ""


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None
    retrieval_mode: RetrievalMode = RetrievalMode.hybrid


class ConversationListResponse(BaseModel):
    conversations: list[ConversationListItem]
    total: int
