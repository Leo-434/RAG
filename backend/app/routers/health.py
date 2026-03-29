from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    neo4j_ok = request.app.state.graph_service.ping()
    chroma_ok = request.app.state.vector_service.ping()
    status = "ok" if (neo4j_ok and chroma_ok) else "degraded"
    return {
        "status": status,
        "neo4j": "ok" if neo4j_ok else "error",
        "chromadb": "ok" if chroma_ok else "error",
    }
