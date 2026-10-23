"""Embedding providers against stubbed backends: the local provider
normalizes and reports its model's dimension; the Gemini provider adds the
task prefixes, asks for the configured dimensionality, batches, checks what
comes back, retries through the gateway loop (counted in track_usage), and
build_embeddings() picks the provider and fails loudly without credentials."""
import math
from types import SimpleNamespace

import pytest
from google.genai import types as genai_types

import llm
from config import Settings
from embeddings import (
    GEMINI_DOCUMENT_PREFIX,
    GEMINI_QUERY_PREFIX,
    GeminiEmbeddings,
    LocalEmbeddings,
    build_embeddings,
)
from llm import track_usage
from tests.unit.test_llm_gateway import _Scripted, gemini_error


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)


def _reply(*vectors):
    return genai_types.EmbedContentResponse(
        embeddings=[genai_types.ContentEmbedding(values=list(v)) for v in vectors]
    )


def _gemini(*results, dimension=4, batch_size=100):
    embed = _Scripted(*results)
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed)))
    return GeminiEmbeddings(client, model="gemini-embedding-2", dimension=dimension,
                            batch_size=batch_size), embed


def _texts(call) -> list[str]:
    """One text per Content: the SDK would merge plain strings into a
    single multi-part content (one combined embedding) for this model."""
    contents = call["contents"]
    assert all(isinstance(c, genai_types.Content) and len(c.parts) == 1 for c in contents)
    return [c.parts[0].text for c in contents]


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


# --- local -------------------------------------------------------------------------


class _Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True):
        self.calls.append((list(texts), normalize_embeddings))
        return [[1.0, 0.0, 0.0] for _ in texts]

    def get_embedding_dimension(self):
        return 3


async def test_local_provider_reports_dimension_and_asks_for_normalized_vectors():
    encoder = _Encoder()
    provider = LocalEmbeddings("all-MiniLM-L6-v2", encoder=encoder)

    vectors = await provider.embed_documents(["a", "b"])
    query = await provider.embed_query("q")

    assert (provider.model, provider.dimension) == ("all-MiniLM-L6-v2", 3)
    assert vectors == [[1.0, 0.0, 0.0]] * 2 and query == [1.0, 0.0, 0.0]
    assert encoder.calls == [(["a", "b"], True), (["q"], True)]


# --- gemini ------------------------------------------------------------------------


async def test_gemini_prefixes_documents_and_queries_and_sets_dimensionality():
    provider, embed = _gemini(_reply([3, 4, 0, 0]), _reply([0, 0, 0, 2]))

    docs = await provider.embed_documents(["chunk text"])
    query = await provider.embed_query("a question?")

    assert _texts(embed.calls[0]) == [GEMINI_DOCUMENT_PREFIX + "chunk text"]
    assert _texts(embed.calls[1]) == [GEMINI_QUERY_PREFIX + "a question?"]
    assert all(c["model"] == "gemini-embedding-2" for c in embed.calls)
    assert all(c["config"].output_dimensionality == 4 for c in embed.calls)
    # Returned vectors are unit length whatever the API sends.
    assert docs == [[0.6, 0.8, 0.0, 0.0]] and query == [0.0, 0.0, 0.0, 1.0]
    assert math.isclose(_norm(docs[0]), 1.0)


async def test_gemini_batches_texts_and_keeps_order():
    provider, embed = _gemini(
        _reply([1, 0], [2, 0]), _reply([0, 1], [0, 2]), _reply([1, 1]),
        dimension=2, batch_size=2,
    )

    vectors = await provider.embed_documents(["a", "b", "c", "d", "e"])

    assert [_texts(c) for c in embed.calls] == [
        [GEMINI_DOCUMENT_PREFIX + t for t in batch] for batch in (["a", "b"], ["c", "d"], ["e"])
    ]
    assert len(vectors) == 5
    assert vectors[2] == [0.0, 1.0]


async def test_gemini_rejects_wrong_count_or_dimension():
    provider, _ = _gemini(_reply([1, 0, 0, 0]))
    with pytest.raises(RuntimeError, match="1 embeddings for 2 texts"):
        await provider.embed_documents(["a", "b"])

    provider, _ = _gemini(_reply([1, 0]))
    with pytest.raises(RuntimeError, match="2 dims, expected 4"):
        await provider.embed_query("q")


async def test_gemini_retries_rate_limits_and_counts_every_attempt():
    provider, embed = _gemini(
        gemini_error(429, quota_id="EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier"),
        _reply([1, 0, 0, 0]),
    )

    with track_usage() as usage:
        await provider.embed_query("q")

    assert len(embed.calls) == 2
    assert usage["attempts_per_model"] == {"gemini-embedding-2": 2}
    assert "EmbedContentRequestsPerMinute" in usage["failures"][0]["error"]


async def test_gemini_daily_quota_is_not_retried():
    provider, embed = _gemini(gemini_error(429, quota_id="EmbedContentRequestsPerDayPerProjectPerModel"))

    with pytest.raises(Exception):
        await provider.embed_query("q")
    assert len(embed.calls) == 1


# --- selection -----------------------------------------------------------------------


async def test_vertex_sends_one_text_per_request():
    embed = _Scripted(_reply([1, 0]), _reply([0, 1]))
    client = SimpleNamespace(vertexai=True,
                             aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed)))
    provider = GeminiEmbeddings(client, model="gemini-embedding-2", dimension=2, batch_size=100)

    await provider.embed_documents(["a", "b"])

    assert [len(c["contents"]) for c in embed.calls] == [1, 1]


def test_build_embeddings_picks_gemini_with_defaults():
    settings = Settings(_env_file=None, embedding_provider="gemini", gemini_api_key="k")

    provider = build_embeddings(settings)

    assert isinstance(provider, GeminiEmbeddings)
    assert (provider.model, provider.dimension) == ("gemini-embedding-2", 1536)


def test_build_embeddings_fails_loudly_without_credentials():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        build_embeddings(Settings(_env_file=None, embedding_provider="gemini", gemini_api_key=None))
    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        build_embeddings(Settings(_env_file=None, embedding_provider="gemini",
                                  embedding_use_vertexai=True))


def test_local_is_the_default_provider():
    settings = Settings(_env_file=None)

    assert (settings.embedding_provider, settings.embedding_model) == ("local", "all-MiniLM-L6-v2")
