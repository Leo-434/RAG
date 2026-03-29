from typing import Optional

from pydantic import BaseModel


class GraphNode(BaseModel):
    name: str
    label: str
    properties: dict = {}
    doc_id: Optional[str] = None


class GraphRelationship(BaseModel):
    source: str
    target: str
    type: str
    properties: dict = {}


class SubgraphResponse(BaseModel):
    center: str
    nodes: list[GraphNode]
    relationships: list[GraphRelationship]


class EntityListResponse(BaseModel):
    entities: list[GraphNode]
    total: int
    page: int
    page_size: int


class GraphStatsResponse(BaseModel):
    node_count: int
    relation_count: int
    entity_types: dict[str, int]


class EntitySearchRequest(BaseModel):
    keyword: str
    limit: int = 10


class RelationshipListResponse(BaseModel):
    relationships: list[GraphRelationship]
    total: int
    page: int
    page_size: int
