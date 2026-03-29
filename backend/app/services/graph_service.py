"""Neo4j graph read/write service."""

import asyncio
from typing import Any

from neo4j import GraphDatabase

from app.config import Settings
from app.models.graph import GraphNode, GraphRelationship, SubgraphResponse
from app.utils.logger import get_logger

log = get_logger(__name__)

_ENTITY_LABELS = ["PERSON", "ORGANIZATION", "PRODUCT", "CONCEPT"]
_UNIQUE_CONSTRAINTS = [
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE"
    for label in _ENTITY_LABELS
]
_FULLTEXT_INDEX = """
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (n:PERSON|ORGANIZATION|PRODUCT|CONCEPT)
ON EACH [n.name, n.role, n.type, n.definition]
"""
_DOC_INDEX = """
CREATE INDEX entity_doc_id IF NOT EXISTS
FOR (n:PERSON) ON (n.doc_id)
"""


class GraphService:
    def __init__(self, settings: Settings):
        self._driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create constraints and indexes. Call once at startup."""
        with self._driver.session() as session:
            for constraint in _UNIQUE_CONSTRAINTS:
                session.run(constraint)
            session.run(_FULLTEXT_INDEX)
            # Doc index only on PERSON to avoid repeated index error; fulltext covers rest
            try:
                session.run(_DOC_INDEX)
            except Exception:
                pass
        log.info("graph_service.initialized")

    def close(self) -> None:
        self._driver.close()

    def ping(self) -> bool:
        try:
            info = self._driver.get_server_info()
            return bool(info.agent)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_graph_data(self, graph_data: dict, doc_id: str) -> tuple[int, int]:
        """
        Write nodes and edges from LangExtract output.
        Returns (node_count, edge_count).
        """
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        with self._driver.session() as session:
            # Write nodes
            for node in nodes:
                label = node.get("label", "CONCEPT").upper()
                if label not in _ENTITY_LABELS:
                    label = "CONCEPT"
                props = {k: v for k, v in node.items() if k not in ("label",)}
                props["doc_id"] = doc_id
                session.run(
                    f"MERGE (n:{label} {{name: $name}}) SET n += $props",
                    name=node.get("name", "Unknown"),
                    props=props,
                )

            # Write edges
            for edge in edges:
                rel_type = edge.get("type", "RELATED_TO").upper().replace(" ", "_")
                props = {k: v for k, v in edge.items() if k not in ("source", "target", "type")}
                props["doc_id"] = doc_id
                session.run(
                    """
                    MATCH (a {name: $src}), (b {name: $tgt})
                    MERGE (a)-[r:""" + rel_type + """]->(b)
                    SET r += $props
                    """,
                    src=edge.get("source", ""),
                    tgt=edge.get("target", ""),
                    props=props,
                )

        log.info("graph_service.write_done", nodes=len(nodes), edges=len(edges))
        return len(nodes), len(edges)

    def delete_doc(self, doc_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH (n {doc_id: $doc_id}) DETACH DELETE n",
                doc_id=doc_id,
            )
        log.info("graph_service.doc_deleted", doc_id=doc_id)

    # ------------------------------------------------------------------
    # Read / Retrieval
    # ------------------------------------------------------------------

    def fulltext_search(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search over entity names/properties."""
        results = []
        with self._driver.session() as session:
            try:
                records = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('entity_fulltext', $search_text)
                    YIELD node, score
                    RETURN node.name AS name, labels(node)[0] AS label,
                           score, node.doc_id AS doc_id
                    LIMIT $limit
                    """,
                    search_text=query,
                    limit=limit,
                )
                for r in records:
                    results.append({
                        "name": r["name"],
                        "label": r["label"],
                        "score": r["score"],
                        "doc_id": r["doc_id"],
                    })
            except Exception as e:
                log.warning("graph_service.fulltext_search_error", error=str(e))
        return results

    def get_neighbors(self, name: str, hops: int = 1) -> SubgraphResponse:
        """Return 1-hop subgraph around an entity."""
        nodes: list[GraphNode] = []
        rels: list[GraphRelationship] = []
        seen_nodes: set[str] = set()
        seen_rels: set[tuple] = set()

        with self._driver.session() as session:
            records = session.run(
                """
                MATCH (center {name: $name})-[r]-(neighbor)
                RETURN center, r, neighbor
                LIMIT 50
                """,
                name=name,
            )
            for record in records:
                for node in (record["center"], record["neighbor"]):
                    node_name = node["name"]
                    if node_name not in seen_nodes:
                        seen_nodes.add(node_name)
                        label = list(node.labels)[0] if node.labels else "CONCEPT"
                        nodes.append(GraphNode(
                            name=node_name,
                            label=label,
                            properties=dict(node),
                            doc_id=node.get("doc_id"),
                        ))

                rel = record["r"]
                key = (rel.start_node["name"], type(rel).__name__, rel.end_node["name"])
                if key not in seen_rels:
                    seen_rels.add(key)
                    rels.append(GraphRelationship(
                        source=rel.start_node["name"],
                        target=rel.end_node["name"],
                        type=rel.type,
                        properties=dict(rel),
                    ))

        return SubgraphResponse(center=name, nodes=nodes, relationships=rels)

    def get_entities(
        self,
        entity_type: str | None = None,
        doc_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[GraphNode], int]:
        """List entities with optional filters, return (items, total)."""
        where_clauses = []
        params: dict[str, Any] = {"skip": (page - 1) * page_size, "limit": page_size}

        label_filter = f":{entity_type.upper()}" if entity_type else ""
        if doc_id:
            where_clauses.append("n.doc_id = $doc_id")
            params["doc_id"] = doc_id

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._driver.session() as session:
            count_rec = session.run(
                f"MATCH (n{label_filter}) {where} RETURN count(n) AS total",
                **params,
            ).single()
            total = count_rec["total"] if count_rec else 0

            records = session.run(
                f"""
                MATCH (n{label_filter}) {where}
                RETURN n, labels(n)[0] AS label
                SKIP $skip LIMIT $limit
                """,
                **params,
            )
            items = []
            for r in records:
                node = r["n"]
                items.append(GraphNode(
                    name=node["name"],
                    label=r["label"],
                    properties=dict(node),
                    doc_id=node.get("doc_id"),
                ))

        return items, total

    def get_relationships(
        self,
        doc_id: str | None = None,
        page: int = 1,
        page_size: int = 200,
    ) -> tuple[list[GraphRelationship], int]:
        """List all relationships with optional doc_id filter, paginated."""
        where_clause = "WHERE r.doc_id = $doc_id" if doc_id else ""
        params: dict[str, Any] = {"skip": (page - 1) * page_size, "limit": page_size}
        if doc_id:
            params["doc_id"] = doc_id

        with self._driver.session() as session:
            count_rec = session.run(
                f"MATCH ()-[r]->() {where_clause} RETURN count(r) AS total",
                **params,
            ).single()
            total = count_rec["total"] if count_rec else 0

            records = session.run(
                f"""
                MATCH (a)-[r]->(b) {where_clause}
                RETURN a.name AS source, b.name AS target,
                       type(r) AS rel_type, properties(r) AS props
                SKIP $skip LIMIT $limit
                """,
                **params,
            )
            rels: list[GraphRelationship] = []
            for rec in records:
                rels.append(GraphRelationship(
                    source=rec["source"],
                    target=rec["target"],
                    type=rec["rel_type"],
                    properties=dict(rec["props"]),
                ))

        return rels, total

    def get_stats(self) -> dict:
        with self._driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            type_counts: dict[str, int] = {}
            for label in _ENTITY_LABELS:
                cnt = session.run(
                    f"MATCH (n:{label}) RETURN count(n) AS c"
                ).single()["c"]
                type_counts[label] = cnt

        return {"node_count": node_count, "relation_count": rel_count, "entity_types": type_counts}

    def entity_graph_context(self, entity_name: str) -> str:
        """Return a text summary of an entity's relationships (for Agent tool)."""
        with self._driver.session() as session:
            records = session.run(
                """
                MATCH (a {name: $name})-[r]-(b)
                RETURN a.name AS src, type(r) AS rel, b.name AS tgt
                LIMIT 20
                """,
                name=entity_name,
            )
            lines = []
            for rec in records:
                lines.append(f"{rec['src']} -[{rec['rel']}]-> {rec['tgt']}")
        return "\n".join(lines) if lines else f"No relationships found for '{entity_name}'."

    def investment_search(self, org_name: str) -> str:
        """Return investment relationships for an organization (for Agent tool)."""
        with self._driver.session() as session:
            records = session.run(
                """
                MATCH (inv)-[r:INVESTED_IN]->(org)
                WHERE org.name CONTAINS $name OR inv.name CONTAINS $name
                RETURN inv.name AS investor, org.name AS company,
                       r.amount AS amount, r.year AS year, r.period AS period
                LIMIT 20
                """,
                name=org_name,
            )
            lines = []
            for r in records:
                line = f"{r['investor']} invested in {r['company']}"
                parts = []
                if r["amount"]:
                    parts.append(f"amount: {r['amount']}")
                if r["year"]:
                    parts.append(f"year: {r['year']}")
                if r["period"]:
                    parts.append(f"period: {r['period']}")
                if parts:
                    line += f" ({', '.join(parts)})"
                lines.append(line)
        return "\n".join(lines) if lines else f"No investment data found for '{org_name}'."

    # ------------------------------------------------------------------
    # Async wrappers
    # ------------------------------------------------------------------

    async def async_write_graph_data(self, graph_data: dict, doc_id: str) -> tuple[int, int]:
        return await asyncio.to_thread(self.write_graph_data, graph_data, doc_id)

    async def async_fulltext_search(self, query: str, limit: int = 5) -> list[dict]:
        return await asyncio.to_thread(self.fulltext_search, query, limit)
