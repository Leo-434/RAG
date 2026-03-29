"""FastAPI dependency helpers that pull singletons from app.state."""

from fastapi import Request

from app.config import Settings, get_settings  # noqa: F401 — re-export for convenience


def get_neo4j_driver(request: Request):
    return request.app.state.neo4j_driver


def get_graph_service(request: Request):
    return request.app.state.graph_service


def get_vector_service(request: Request):
    return request.app.state.vector_service


def get_hybrid_service(request: Request):
    return request.app.state.hybrid_service


def get_answer_service(request: Request):
    return request.app.state.answer_service


def get_ingestion_service(request: Request):
    return request.app.state.ingestion_service


def get_doc_registry_service(request: Request):
    """Return the persistent document registry service (SQLite-backed)."""
    return request.app.state.doc_registry_service


def get_conversation_service(request: Request):
    return request.app.state.conversation_service
