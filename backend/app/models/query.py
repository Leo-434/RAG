from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    hybrid = "hybrid"
    vector_only = "vector_only"
    graph_only = "graph_only"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    retrieval_mode: RetrievalMode = RetrievalMode.hybrid
    top_k: int = Field(5, ge=1, le=20)


class SourceItem(BaseModel):
    source_type: Literal["vector", "graph"]
    content: Optional[str] = None
    score: Optional[float] = None
    filename: Optional[str] = None
    entity: Optional[str] = None
    relation: Optional[str] = None
    context: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    latency_ms: int
