"""Central configuration, loaded from environment variables / .env (no secrets in code)."""

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "rag_documents"

    # Embeddings
    embedding_provider: str = "huggingface"  # huggingface | openai | hash
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # LLM
    llm_provider: str = "auto"  # auto | gemini | openai | ollama | extractive
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Chunking
    default_chunking_strategy: str = "fixed"
    chunk_size: int = 500
    chunk_overlap: int = 100
    semantic_distance_threshold: float = 0.35

    # Retrieval
    hybrid_alpha: float = 0.6
    fusion_method: str = "rrf"  # rrf | minmax
    top_k: int = 4
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_cosine_score: float = 0.20
    min_rerank_score: float = 0.05

    # API
    api_port: int = 8000
    max_upload_mb: int = 25
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
