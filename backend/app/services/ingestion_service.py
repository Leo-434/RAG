"""Document ingestion pipeline orchestrator."""

import asyncio
import time
from datetime import datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.models.document import DocumentMeta, IngestStatus
from app.services.document_registry_service import DocumentRegistryService
from app.services.entity_extraction_service import EntityExtractionService
from app.services.graph_service import GraphService
from app.services.vector_service import VectorService
from app.utils.logger import get_logger
from app.utils.mineru_parser import MinerUParser

log = get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        vector_service: VectorService,
        graph_service: GraphService,
        entity_extraction_service: EntityExtractionService,
        doc_registry: DocumentRegistryService,
    ):
        self._settings = settings
        self._vector = vector_service
        self._graph = graph_service
        self._extractor = entity_extraction_service
        self._registry = doc_registry
        self._parser = MinerUParser(
            api_key=settings.MINERU_API_KEY,
            base_url=settings.MINERU_BASE_URL,
            model_version=settings.MINERU_MODEL_VERSION,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )

    # ── Registry delegation ──────────────────────────────────────────────────

    def create_doc_entry(self, filename: str, file_size: int) -> str:
        """Create a document record (status=uploading) and return its doc_id."""
        return self._registry.create_doc(filename, file_size)

    def get_doc(self, doc_id: str) -> DocumentMeta | None:
        return self._registry.get_doc(doc_id)

    def list_docs(self) -> list[DocumentMeta]:
        return self._registry.list_docs()

    # ── Pipeline ─────────────────────────────────────────────────────────────

    async def run_pipeline(self, doc_id: str, file_path: Path) -> None:
        """
        Full ingestion pipeline (runs in background):
        MinerU parse → split → parallel(vector embed + entity extract → graph write) → done
        """
        meta = self._registry.get_doc(doc_id)
        if not meta:
            log.error("ingestion.doc_not_found", doc_id=doc_id)
            return

        meta.status = IngestStatus.ingesting
        self._registry.save_doc(meta)
        log.info("ingestion.start", doc_id=doc_id, file=str(file_path))

        try:
            # Step 1: Parse PDF with MinerU
            t0 = time.time()
            markdown = await asyncio.to_thread(self._parser.parse, file_path)
            log.info("ingestion.mineru_done", doc_id=doc_id, elapsed=round(time.time() - t0, 1))

            # Step 2: Split into chunks
            chunks = self._splitter.split_text(markdown)
            log.info("ingestion.chunks", doc_id=doc_id, count=len(chunks))

            # Step 3: Parallel embed + extract
            vector_task = self._vector.async_embed_and_store(
                chunks, doc_id=doc_id, filename=meta.filename
            )
            extract_task = self._extractor.async_extract(markdown)

            chunk_count, graph_data = await asyncio.gather(vector_task, extract_task)
            log.info("ingestion.parallel_done", doc_id=doc_id)

            # Step 4: Write graph
            node_count, edge_count = await self._graph.async_write_graph_data(graph_data, doc_id)

            # Step 5: Mark ready and persist
            meta.chunk_count = chunk_count
            meta.entity_count = node_count
            meta.relation_count = edge_count
            meta.status = IngestStatus.ready
            meta.ingested_at = datetime.utcnow()
            self._registry.save_doc(meta)
            log.info(
                "ingestion.complete",
                doc_id=doc_id,
                chunks=chunk_count,
                entities=node_count,
                relations=edge_count,
            )

        except Exception as e:
            log.error("ingestion.failed", doc_id=doc_id, error=str(e))
            meta.status = IngestStatus.failed
            meta.error_msg = str(e)
            self._registry.save_doc(meta)

    # ── Deletion ─────────────────────────────────────────────────────────────

    async def delete_doc(self, doc_id: str) -> bool:
        """Remove document from graph, vector store, disk, and registry."""
        if not self._registry.get_doc(doc_id):
            return False

        # 1. Remove from Neo4j
        try:
            await asyncio.to_thread(self._graph.delete_doc, doc_id)
        except Exception as e:
            log.warning("ingestion.delete_graph_error", doc_id=doc_id, error=str(e))

        # 2. Remove from ChromaDB
        try:
            await asyncio.to_thread(self._vector.delete_doc, doc_id)
        except Exception as e:
            log.warning("ingestion.delete_vector_error", doc_id=doc_id, error=str(e))

        # 3. Remove uploaded file
        upload_dir = Path(self._settings.UPLOAD_DIR)
        for f in upload_dir.glob(f"{doc_id}_*"):
            try:
                f.unlink()
            except Exception:
                pass

        # 4. Remove from registry (SQLite)
        self._registry.delete_doc(doc_id)
        log.info("ingestion.deleted", doc_id=doc_id)
        return True
