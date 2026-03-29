from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # DashScope
    DASHSCOPE_API_KEY: str
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "qwen3-max-2026-01-23"
    EMBEDDING_MODEL: str = "text-embedding-v4"

    # MinerU
    MINERU_API_KEY: str
    MINERU_BASE_URL: str = "https://mineru.net/api/v4"
    MINERU_MODEL_VERSION: str = "vlm"

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "document_chunks"

    # Ingestion params
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 60
    EMBED_BATCH_SIZE: int = 8
    LANGEXTRACT_MAX_WORKERS: int = 4
    LANGEXTRACT_MAX_CHAR_BUFFER: int = 3000

    # Retrieval params
    DEFAULT_TOP_K: int = 5
    RRF_K: int = 60

    # File upload
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_MB: int = 50

    # Conversations DB
    CONVERSATIONS_DB_PATH: str = "./data/conversations.db"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
