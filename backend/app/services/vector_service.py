"""ChromaDB vector store service."""

import asyncio

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import Settings
from app.models.query import SourceItem
from app.utils.logger import get_logger

log = get_logger(__name__)


class VectorService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL,
            check_embedding_ctx_length=False,  # DashScope requires raw strings, not token IDs
        )
        self._vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self._embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        log.info("vector_service.initialized")

    def ping(self) -> bool:
        """Health-check: confirm the vector store is reachable."""
        try:
            # Use public get() with limit=1 instead of private _collection.count()
            self._vectorstore.get(limit=1)
            return True
        except Exception:
            return False

    def embed_and_store(self, chunks: list[str], doc_id: str, filename: str) -> int:
        """Embed text chunks and persist in ChromaDB. Returns the number stored."""
        batch_size = self._settings.EMBED_BATCH_SIZE
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            metadatas = [
                {"doc_id": doc_id, "filename": filename, "chunk_index": i + j}
                for j, _ in enumerate(batch)
            ]
            self._vectorstore.add_texts(texts=batch, metadatas=metadatas)
            total += len(batch)
            log.debug("vector_service.batch_stored", batch_end=i + len(batch), total_chunks=len(chunks))
        log.info("vector_service.store_done", doc_id=doc_id, chunks=total)
        return total

    def similarity_search(self, query: str, k: int = 5) -> list[SourceItem]:
        """Return top-k semantically similar chunks for a query."""
        results = self._vectorstore.similarity_search_with_score(query, k=k)
        items = []
        for doc, score in results:
            # ChromaDB returns L2 distance (lower = more similar); convert to [0,1] similarity
            sim = 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score)
            items.append(SourceItem(
                source_type="vector",
                content=doc.page_content,
                score=round(sim, 4),
                filename=doc.metadata.get("filename"),
            ))
        return items

    def delete_doc(self, doc_id: str) -> None:
        """Delete all vectors belonging to a document using the public Chroma API."""
        # Step 1: resolve document IDs via metadata filter (public get())
        result = self._vectorstore.get(where={"doc_id": doc_id})
        ids: list[str] = result.get("ids", [])
        # Step 2: bulk-delete by IDs (public delete())
        if ids:
            self._vectorstore.delete(ids=ids)
        log.info("vector_service.doc_deleted", doc_id=doc_id, deleted_count=len(ids))

    # ------------------------------------------------------------------
    # Async wrappers
    # ------------------------------------------------------------------

    async def async_embed_and_store(self, chunks: list[str], doc_id: str, filename: str) -> int:
        return await asyncio.to_thread(self.embed_and_store, chunks, doc_id, filename)

    async def async_similarity_search(self, query: str, k: int = 5) -> list[SourceItem]:
        return await asyncio.to_thread(self.similarity_search, query, k)
