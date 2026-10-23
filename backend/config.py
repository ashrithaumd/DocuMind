"""Central configuration.

Every tunable and secret comes from the environment (or a local .env file) —
nothing is hardcoded. Defaults are chosen for local development; docker-compose
overrides the service hosts.
"""
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (all reasoning calls go through the gateway in llm.py) ---
    llm_provider: Literal["gemini", "anthropic"] = "gemini"
    gemini_api_key: str | None = None      # required when llm_provider=gemini
    anthropic_api_key: str | None = None   # required when llm_provider=anthropic
    # Empty -> the provider's default from PROVIDER_DEFAULT_MODELS below, so
    # switching LLM_PROVIDER alone is enough. Set either to pin a model.
    generation_model: str = ""   # answer generation
    grading_model: str = ""      # relevance grading + groundedness checks
    # Gemini thinking level for the two checker calls (a yes/no verdict needs
    # little reasoning). "minimal" is the lowest the Gemini 3 docs allow;
    # thinking can't be fully disabled. Empty = the model's default.
    # Generation stays at the model default. Ignored by the Anthropic provider.
    grading_thinking_level: Literal["", "minimal", "low", "medium", "high"] = "minimal"
    # Attempts per call on 429/overloaded errors (exponential backoff between).
    llm_max_attempts: int = 5
    # Seconds before a single Gemini request is abandoned and retried (counts
    # as one failed attempt). Free-tier requests occasionally hang for minutes.
    llm_request_timeout_s: float = 60.0

    # --- Embeddings (local, free, offline — no provider embeddings API) ---
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

    @model_validator(mode="after")
    def _fill_default_models(self) -> "Settings":
        generation, grading = PROVIDER_DEFAULT_MODELS[self.llm_provider]
        self.generation_model = self.generation_model or generation
        self.grading_model = self.grading_model or grading
        return self


# (generation, grading) per provider. Gemini defaults are free-tier models
# (Google AI Studio, no billing). Generation defaults to Flash-Lite, not
# Flash: gemini-3.6-flash allows only 20 requests/day on the free tier, too
# few for repeated baselines plus development traffic. Set
# GENERATION_MODEL=gemini-3.6-flash for demos.
PROVIDER_DEFAULT_MODELS = {
    "gemini": ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),
    "anthropic": ("claude-opus-4-8", "claude-haiku-4-5"),
}


@lru_cache
def get_settings() -> Settings:
    return Settings()
