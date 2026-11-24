"""App settings, loaded from environment variables or a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Claude
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"  # low | medium | high | xhigh | max
    max_tokens: int = 64000  # upper bound only; answers are streamed

    # Storage
    qdrant_path: Path = Path("data/qdrant")
    collection_name: str = "documents"

    # Models (run locally via FastEmbed, downloaded once into model_cache_dir)
    model_cache_dir: Path = Path("data/models")
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-12-v2"  # ~120 MB cross-encoder

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    candidates_per_search: int = 20  # results taken from each search before fusion
    rerank_candidates: int = 20  # fused results passed to the reranker
    top_k: int = 5  # chunks finally given to the LLM
    rrf_k: int = 60  # Reciprocal Rank Fusion constant


@lru_cache
def get_settings() -> Settings:
    return Settings()
