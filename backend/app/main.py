"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import conversations, documents, graph, health, query
from app.services.answer_service import AnswerService
from app.services.conversation_service import ConversationService
from app.services.document_registry_service import DocumentRegistryService
from app.services.entity_extraction_service import EntityExtractionService
from app.services.graph_service import GraphService
from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.services.ingestion_service import IngestionService
from app.services.vector_service import VectorService
from app.utils.logger import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()

    # Ensure data directories exist
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.CONVERSATIONS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    log.info("startup.begin")

    # Document metadata persistence (SQLite — survives restarts)
    doc_registry_service = DocumentRegistryService(settings.CONVERSATIONS_DB_PATH)
    app.state.doc_registry_service = doc_registry_service

    # Conversation persistence (SQLite — same DB file)
    conversation_service = ConversationService(settings.CONVERSATIONS_DB_PATH)
    await conversation_service.init_db()
    app.state.conversation_service = conversation_service

    # Initialize services
    graph_service = GraphService(settings)
    graph_service.initialize()
    app.state.graph_service = graph_service

    vector_service = VectorService(settings)
    app.state.vector_service = vector_service

    entity_extraction_service = EntityExtractionService(settings)

    hybrid_service = HybridRetrievalService(
        vector_service=vector_service,
        graph_service=graph_service,
        settings=settings,
    )
    app.state.hybrid_service = hybrid_service

    answer_service = AnswerService(
        settings=settings,
        hybrid_service=hybrid_service,
        graph_service=graph_service,
    )
    app.state.answer_service = answer_service

    ingestion_service = IngestionService(
        settings=settings,
        vector_service=vector_service,
        graph_service=graph_service,
        entity_extraction_service=entity_extraction_service,
        doc_registry=doc_registry_service,
    )
    app.state.ingestion_service = ingestion_service

    log.info("startup.complete")
    yield

    # Shutdown
    graph_service.close()
    log.info("shutdown.complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="GraphRAG Knowledge Base API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(graph.router)
    app.include_router(conversations.router)

    return app


app = create_app()
