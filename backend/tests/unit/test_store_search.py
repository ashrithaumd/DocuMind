"""The store's search split: search() is embed_query() + search_with_vector(),
which is the top_k cut of fused_ranking(), the full fused candidate list
the benchmark scores gold-page ranks against."""
import pytest

from config import Settings
from ingestion.chunking import Chunk
from ingestion.store import HybridStore
from tests.helpers import FakeEmbedder


@pytest.fixture
async def store(tmp_path):
    settings = Settings(
        _env_file=None,
        chroma_host=None,
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="search_test",
        top_k=3,
    )
    store = HybridStore(settings, FakeEmbedder())
    await store.add_chunks([
        Chunk(chunk_id=f"d1-{i}", doc_id="d1", filename="a.pdf", page_number=i + 1,
              chunk_index=i, text=f"Pump {i} runs at {400 + i} RPM in module {i}.")
        for i in range(6)
    ])
    return store


async def test_search_is_top_k_of_the_fused_ranking(store):
    query = "Which pump runs at 402 RPM?"
    vector = await store.embed_query(query)

    ranking = await store.fused_ranking(query, vector, doc_ids=["d1"])
    results = await store.search(query, doc_ids=["d1"])

    assert len(ranking) > 3
    assert [c.chunk_id for c in results] == [c.chunk_id for c in ranking[:3]]
    assert [c.chunk_id for c in await store.search_with_vector(query, vector, ["d1"])] == \
        [c.chunk_id for c in results]


async def test_fused_ranking_keeps_both_leg_ranks(store):
    query = "Which pump runs at 402 RPM?"
    ranking = await store.fused_ranking(query, await store.embed_query(query))

    assert any(c.dense_rank is not None for c in ranking)
    assert any(c.sparse_rank is not None for c in ranking)
    scores = [c.score for c in ranking]
    assert scores == sorted(scores, reverse=True)
