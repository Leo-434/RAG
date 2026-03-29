import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.dependencies import get_answer_service, get_conversation_service
from app.models.query import QueryRequest, QueryResponse
from app.services.answer_service import AnswerService
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/api/query", tags=["query"])


async def _load_history(conv_id: str | None, conv_svc: ConversationService) -> list[dict]:
    """Load conversation history as plain dicts for answer_service."""
    if not conv_id:
        return []
    conv = await conv_svc.get_conversation(conv_id)
    if not conv:
        return []
    return [{"role": m.role, "content": m.content} for m in conv.messages]


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    answer_service: AnswerService = Depends(get_answer_service),
    conv_svc: ConversationService = Depends(get_conversation_service),
):
    """Non-streaming knowledge base Q&A."""
    start_ms = int(time.time() * 1000)
    history = await _load_history(request.conversation_id, conv_svc)
    response = await answer_service.answer(request, start_ms, history=history)

    # Persist user question + assistant answer to conversation
    if request.conversation_id:
        await conv_svc.add_message(request.conversation_id, "user", request.question)
        await conv_svc.add_message(
            request.conversation_id,
            "assistant",
            response.answer,
            sources=response.sources,
            latency_ms=response.latency_ms,
        )
        # Auto-set title from the first user question
        conv = await conv_svc.get_conversation(request.conversation_id)
        if conv and conv.title == "新对话" and len(conv.messages) <= 2:
            await conv_svc.update_title(request.conversation_id, request.question[:50])

    return response


@router.post("/stream")
async def query_stream(
    request: QueryRequest,
    answer_service: AnswerService = Depends(get_answer_service),
    conv_svc: ConversationService = Depends(get_conversation_service),
):
    """SSE streaming knowledge base Q&A."""
    history = await _load_history(request.conversation_id, conv_svc)
    return StreamingResponse(
        answer_service.stream_answer(request, history=history, conv_svc=conv_svc),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
