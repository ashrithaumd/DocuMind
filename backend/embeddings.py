"""Embedding providers. The store embeds through this interface only, so the
vector index doesn't care where vectors come from:

  - LocalEmbeddings:  sentence-transformers (all-MiniLM-L6-v2, 384 dims) on
                      the CPU. No key, no network; CI and offline dev use it.
  - GeminiEmbeddings: gemini-embedding-2 through the google-genai SDK, at a
                      reduced output dimensionality (1536 by default). The
                      same SDK client serves the Gemini API (AI Studio key,
                      the default) and Vertex AI (EMBEDDING_USE_VERTEXAI=true
                      with GOOGLE_CLOUD_PROJECT/LOCATION and application
                      default credentials), so Vertex is a config switch.

build_embeddings() picks one from EMBEDDING_PROVIDER.

Every provider returns L2-normalized vectors (the store ranks by cosine) and
reports `model` and `dimension`, which the store records next to the vectors
so an index built with one model is never queried with another.

Gemini calls go through the LLM gateway's retry loop: same backoff, same
quota logging, and they count in track_usage() under the embedding model's
name. On the Gemini API a batch of texts is one batchEmbedContents request.
Each text must be its own Content: for gemini-embedding-2 the SDK merges a
plain list of strings into ONE multi-part content and returns a single
combined embedding. Vertex's embedContent takes one content per call, so the
Vertex path sends one text per request.
"""
import asyncio
import logging
import math
from typing import Protocol

from google import genai
from google.genai import types as genai_types

from llm import _BackoffLLM, gemini_retry_delay

logger = logging.getLogger(__name__)

# gemini-embedding-2 takes task instructions as text prefixes instead of a
# task_type field, and they must be used consistently. Queries are questions
# answered from the retrieved passages, hence "question answering"; chunks
# carry no real title, hence "title: none" (the documented no-title form).
GEMINI_QUERY_PREFIX = "task: question answering | query: "
GEMINI_DOCUMENT_PREFIX = "title: none | text: "


class EmbeddingProvider(Protocol):
    model: str
    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def _normalized(vector) -> list[float]:
    values = [float(x) for x in vector]
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm else values


class LocalEmbeddings:
    def __init__(self, model: str, encoder=None):
        """`encoder` is a loaded sentence-transformers model (or anything with
        encode() and get_embedding_dimension()); loaded from `model` if None."""
        if encoder is None:
            # Imported lazily: torch is heavy, and tests that stub the
            # encoder shouldn't pay the import cost.
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model)
        self.model = model
        self._encoder = encoder
        self.dimension = encoder.get_embedding_dimension()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """encode() is CPU-bound and synchronous, so it runs in a worker
        thread to keep the event loop free. MiniLM truncates inputs at 256
        word-pieces (a documented limitation for 600-token chunks)."""
        vectors = await asyncio.to_thread(self._encoder.encode, texts, normalize_embeddings=True)
        return [[float(x) for x in vector] for vector in vectors]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class GeminiEmbeddings(_BackoffLLM):
    def __init__(self, client: genai.Client, model: str, dimension: int, batch_size: int = 100,
                 **backoff):
        super().__init__(**backoff)
        self._client = client
        self.model = model
        self.dimension = dimension
        self._batch_size = 1 if getattr(client, "vertexai", False) else batch_size
        self._config = genai_types.EmbedContentConfig(output_dimensionality=dimension)

    def _retry_delay(self, exc: Exception) -> float | None:
        return gemini_retry_delay(exc)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start:start + self._batch_size]
            response = await self._call(
                self.model,
                lambda batch=batch: self._client.aio.models.embed_content(
                    model=self.model,
                    contents=[genai_types.Content(parts=[genai_types.Part(text=t)]) for t in batch],
                    config=self._config,
                ),
            )
            if len(response.embeddings) != len(batch):
                raise RuntimeError(f"{self.model} returned {len(response.embeddings)} embeddings "
                                   f"for {len(batch)} texts")
            for embedding in response.embeddings:
                if len(embedding.values) != self.dimension:
                    raise RuntimeError(f"{self.model} returned {len(embedding.values)} dims, "
                                       f"expected {self.dimension}")
                # gemini-embedding-2 normalizes reduced dimensions itself;
                # normalizing again is a no-op that keeps the guarantee local.
                vectors.append(_normalized(embedding.values))
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed([GEMINI_DOCUMENT_PREFIX + t for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([GEMINI_QUERY_PREFIX + text]))[0]


def build_embeddings(settings) -> EmbeddingProvider:
    """The configured embedding provider. Fails loudly if its credentials
    are missing."""
    if settings.embedding_provider == "local":
        return LocalEmbeddings(settings.embedding_model)
    if settings.embedding_use_vertexai:
        if not settings.google_cloud_project:
            raise RuntimeError("EMBEDDING_USE_VERTEXAI=true but GOOGLE_CLOUD_PROJECT is not set")
        client = genai.Client(vertexai=True, project=settings.google_cloud_project,
                              location=settings.google_cloud_location)
    else:
        if not settings.gemini_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=gemini but GEMINI_API_KEY is not set")
        client = genai.Client(api_key=settings.gemini_api_key)
    return GeminiEmbeddings(
        client,
        model=settings.embedding_model,
        dimension=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
        max_attempts=settings.llm_max_attempts,
        request_timeout_s=settings.llm_request_timeout_s,
    )
