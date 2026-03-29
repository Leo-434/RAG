"""REST API for conversation history management."""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_conversation_service
from app.models.conversation import (
    Conversation,
    ConversationListResponse,
    CreateConversationRequest,
)
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    svc: ConversationService = Depends(get_conversation_service),
):
    """Return all conversations sorted by most recently updated."""
    items = await svc.list_conversations()
    return ConversationListResponse(conversations=items, total=len(items))


@router.post("", response_model=Conversation, status_code=201)
async def create_conversation(
    req: CreateConversationRequest,
    svc: ConversationService = Depends(get_conversation_service),
):
    """Create a new empty conversation."""
    return await svc.create_conversation(
        title=req.title,
        retrieval_mode=req.retrieval_mode.value,
    )


@router.get("/{conv_id}", response_model=Conversation)
async def get_conversation(
    conv_id: str,
    svc: ConversationService = Depends(get_conversation_service),
):
    """Get a conversation with all its messages."""
    conv = await svc.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.delete("/{conv_id}", status_code=204)
async def delete_conversation(
    conv_id: str,
    svc: ConversationService = Depends(get_conversation_service),
):
    """Permanently delete a conversation and all its messages."""
    deleted = await svc.delete_conversation(conv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
