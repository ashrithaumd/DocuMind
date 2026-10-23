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

    # --- Embeddings (embeddings.py) ---
    # gemini (default): gemini-embedding-2 via the Gemini API key, or Vertex
    # AI with EMBEDDING_USE_VERTEXAI=true. local: sentence-transformers on the
    # CPU, no key or network (CI, offline dev).
    embedding_provider: Literal["local", "gemini"] = "gemini"
    # Empty -> the provider's default from EMBEDDING_DEFAULT_MODELS below.
    embedding_model: str = ""
    # Vector size: Gemini's output_dimensionality (128-3072; Google
    # recommends 768/1536/3072), or the local model's fixed size. 0 = the
    # default from EMBEDDING_DEFAULT_DIMENSIONS. The chunks table's vector
    # column is created with this size (migration 0002).
    embedding_dimensions: int = 0
    embedding_batch_size: int = 100   # texts per Gemini embedding request
    embedding_use_vertexai: bool = False
    google_cloud_project: str | None = None   # Vertex only
    google_cloud_location: str = "us-central1"

    # --- Postgres: chunk store (pgvector + full-text) and query log ---
    # Required: the chunks live here. Migrations run at startup.
    database_url: str | None = None

    # --- Auth ---
    # Single shared API key checked via the X-API-Key header.
    # Unset -> auth disabled (local dev convenience).
    api_key: str | None = None
    # Return raw provider error text and per-request LLM usage (attempts per
    # model, every failed attempt's message) from /query. Off by default:
    # clients get provider-neutral errors and the details stay in the server
    # log. The eval harness's backend runs with it on to record quota ids.
    expose_llm_diagnostics: bool = False

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
        self.embedding_model = self.embedding_model or EMBEDDING_DEFAULT_MODELS[self.embedding_provider]
        if not self.embedding_dimensions:
            try:
                self.embedding_dimensions = EMBEDDING_DEFAULT_DIMENSIONS[self.embedding_model]
            except KeyError:
                raise ValueError(f"Set EMBEDDING_DIMENSIONS for embedding model "
                                 f"'{self.embedding_model}'") from None
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


EMBEDDING_DEFAULT_MODELS = {
    "local": "all-MiniLM-L6-v2",
    "gemini": "gemini-embedding-2",
}
EMBEDDING_DEFAULT_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,      # fixed by the model
    "gemini-embedding-2": 1536,   # reduced from its native 3072
}


@lru_cache
def get_settings() -> Settings:
    return Settings()
