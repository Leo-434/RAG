from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class IngestStatus(str, Enum):
    uploading = "uploading"
    ingesting = "ingesting"
    ready = "ready"
    failed = "failed"


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    file_size_bytes: int
    status: IngestStatus
    chunk_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    ingested_at: Optional[datetime] = None
    error_msg: Optional[str] = None


class DocumentResponse(BaseModel):
    doc_id: str
    status: IngestStatus
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentMeta]
    total: int


class DeleteResponse(BaseModel):
    doc_id: str
    message: str
