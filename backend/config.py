"""Central configuration.

Every tunable and secret comes from the environment (or a local .env file) —
nothing is hardcoded. Defaults are chosen for local development; docker-compose
overrides the service hosts.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Anthropic (all LLM reasoning calls) ---
    anthropic_api_key: str
    generation_model: str = "claude-opus-4-8"   # answer generation
    grading_model: str = "claude-haiku-4-5"     # relevance grading + groundedness checks

    # --- Embeddings (local, free, offline — Anthropic has no embeddings API) ---
    # A sentence-transformers model name; downloaded from HuggingFace on first
    # use, then served from the local cache with no network or key required.
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Vector store ---
    # If CHROMA_HOST is set (docker-compose), talk to a Chroma server over HTTP.
    # Otherwise fall back to an embedded on-disk instance (plain local dev).
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_dir: str = "./chroma_db"
    collection_name: str = "documents"

    # --- Postgres audit / timing log ---
    # Unset -> logging is disabled with a startup warning (app still works).
    database_url: str | None = None

    # --- Auth ---
    # Single shared API key checked via the X-API-Key header.
    # Unset -> auth disabled (local dev convenience).
    api_key: str | None = None

    # --- Retrieval ---
    top_k: int = 6            # chunks handed to the LLM after fusion
    rrf_candidates: int = 12  # candidates pulled from EACH retriever before fusion
    rrf_k: int = 60           # standard RRF damping constant

    # --- Chunking (~600 tokens with 15% overlap) ---
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 90

    # --- Input limits (validated before any model spend) ---
    max_query_chars: int = 2000
    max_upload_mb: int = 20
    max_pages: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()
