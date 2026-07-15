"""Process-wide singletons shared by the API layer and the graph nodes.

lru_cache gives us lazy, exactly-once construction without a DI framework.
Two independent backends live here on purpose:
  - get_llm():      Anthropic — every reasoning call (grade/generate/verify)
  - get_embedder(): sentence-transformers — local, free, no key, no network
This split is the whole provider story: Anthropic has no embeddings API, so
embeddings run locally instead of pulling in a second paid provider.
"""
from functools import lru_cache

from anthropic import AsyncAnthropic

from config import get_settings
from ingestion.store import HybridStore
from llm import ClaudeLLM


@lru_cache
def get_llm() -> ClaudeLLM:
    return ClaudeLLM(AsyncAnthropic(api_key=get_settings().anthropic_api_key))


@lru_cache
def get_embedder():
    # Imported lazily: torch is heavy, and tests that stub the embedder
    # shouldn't pay the import cost.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


@lru_cache
def get_store() -> HybridStore:
    return HybridStore(get_settings(), get_embedder())
