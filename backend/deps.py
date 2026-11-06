"""Process-wide singletons shared by the API layer and the graph nodes.

lru_cache gives us lazy, exactly-once construction without a DI framework.
Two independent backends live here on purpose:
  - get_llm():      the LLM gateway (Gemini by default, Anthropic via
                    LLM_PROVIDER) — every reasoning call (grade/generate/verify)
  - get_embeddings(): the embedding provider (EMBEDDING_PROVIDER: local
                    sentence-transformers or Gemini). It is chosen
                    independently of LLM_PROVIDER, so switching LLM providers
                    never invalidates the vector index.
"""
from functools import lru_cache

from config import get_settings
from embeddings import EmbeddingProvider, build_embeddings
from ingestion.store import HybridStore
from llm import ClaudeLLM, GeminiLLM, build_llm


@lru_cache
def get_llm() -> GeminiLLM | ClaudeLLM:
    return build_llm(get_settings())


@lru_cache
def get_embeddings() -> EmbeddingProvider:
    return build_embeddings(get_settings())


@lru_cache
def get_store() -> HybridStore:
    return HybridStore(get_settings(), get_embeddings())
