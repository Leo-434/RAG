from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_graph_service
from app.models.graph import (
    EntityListResponse,
    EntitySearchRequest,
    GraphNode,
    GraphStatsResponse,
    RelationshipListResponse,
    SubgraphResponse,
)
from app.services.graph_service import GraphService

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/entities", response_model=EntityListResponse)
async def list_entities(
    entity_type: Optional[str] = Query(None, description="Filter by type: PERSON/ORGANIZATION/PRODUCT/CONCEPT"),
    doc_id: Optional[str] = Query(None, description="Filter by document ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    graph: GraphService = Depends(get_graph_service),
):
    items, total = graph.get_entities(entity_type=entity_type, doc_id=doc_id, page=page, page_size=page_size)
    return EntityListResponse(entities=items, total=total, page=page, page_size=page_size)


@router.get("/entities/{name}/neighbors", response_model=SubgraphResponse)
async def get_neighbors(
    name: str,
    graph: GraphService = Depends(get_graph_service),
):
    return graph.get_neighbors(name)


@router.get("/relationships", response_model=RelationshipListResponse)
async def list_relationships(
    doc_id: Optional[str] = Query(None, description="Filter by document ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(200, ge=1, le=500),
    graph: GraphService = Depends(get_graph_service),
):
    rels, total = graph.get_relationships(doc_id=doc_id, page=page, page_size=page_size)
    return RelationshipListResponse(relationships=rels, total=total, page=page, page_size=page_size)


@router.get("/stats", response_model=GraphStatsResponse)
async def get_stats(graph: GraphService = Depends(get_graph_service)):
    stats = graph.get_stats()
    return GraphStatsResponse(**stats)


@router.post("/search", response_model=list[GraphNode])
async def search_entities(
    request: EntitySearchRequest,
    graph: GraphService = Depends(get_graph_service),
):
    results = graph.fulltext_search(request.keyword, limit=request.limit)
    return [
        GraphNode(
            name=r["name"],
            label=r.get("label", "CONCEPT"),
            properties={"score": r.get("score")},
            doc_id=r.get("doc_id"),
        )
        for r in results
    ]
