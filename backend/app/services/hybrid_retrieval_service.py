"""Hybrid retrieval: vector + graph fulltext, fused with RRF."""

import asyncio

from app.config import Settings
from app.models.query import SourceItem
from app.services.graph_service import GraphService
from app.services.vector_service import VectorService
from app.utils.rrf import reciprocal_rank_fusion
from app.utils.logger import get_logger

log = get_logger(__name__)


class HybridRetrievalService:
    def __init__(
        self,
        vector_service: VectorService,
        graph_service: GraphService,
        settings: Settings,
    ):
        self._vector = vector_service
        self._graph = graph_service
        self._rrf_k = settings.RRF_K

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[SourceItem]:
        """Retrieve relevant context given a query string."""

        if mode == "vector_only":
            return await self._vector.async_similarity_search(query, k=top_k)

        if mode == "graph_only":
            graph_raw = await self._graph.async_fulltext_search(query, limit=top_k)
            return [self._graph_result_to_source(r) for r in graph_raw]

        # hybrid: run both in parallel then RRF
        vector_task = self._vector.async_similarity_search(query, k=top_k)
        graph_task = self._graph.async_fulltext_search(query, limit=top_k)
        vector_results, graph_raw = await asyncio.gather(vector_task, graph_task)

        graph_results = [self._graph_result_to_source(r) for r in graph_raw]

        # RRF fusion
        fused = reciprocal_rank_fusion(
            ranked_lists=[vector_results, graph_results],
            key_fn=lambda s: s.content or s.entity or "",
            k=self._rrf_k,
        )

        top = [item for item, _ in fused[:top_k]]
        log.info("hybrid.retrieve_done", mode=mode, returned=len(top))
        return top

    @staticmethod
    def _graph_result_to_source(r: dict) -> SourceItem:
        return SourceItem(
            source_type="graph",
            entity=r.get("name"),
            score=r.get("score"),
            context=f"{r.get('label', '')}: {r.get('name', '')}",
        )
