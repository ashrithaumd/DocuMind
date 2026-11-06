"""The hybrid store on real Postgres + pgvector (fresh database per test, from
TEST_POSTGRES_URL; skipped without it): ingestion, both legs, RRF fusion,
doc scoping, filtered HNSW search with iterative scans, OR-semantics
full-text matching, and the startup checks against a mismatched provider."""
import sqlalchemy as sa
import pytest

from config import Settings
from db import session as db_session
from ingestion.chunking import Chunk
from ingestion.store import HybridStore
from tests.helpers import FakeEmbeddings


def _settings(url, **overrides) -> Settings:
    # FakeEmbeddings produces 32-dim vectors.
    return Settings(**{"_env_file": None, "database_url": url, "embedding_dimensions": 32,
                       "top_k": 3, **overrides})


def _store(url, embeddings=None, **overrides) -> HybridStore:
    settings = _settings(url, **overrides)
    assert db_session.init_db(settings)
    return HybridStore(settings, embeddings or FakeEmbeddings())


def _chunks(doc_id, texts, filename="a.pdf"):
    return [Chunk(chunk_id=f"{doc_id}-{i:04d}", doc_id=doc_id, filename=filename,
                  page_number=i + 1, chunk_index=i, text=t) for i, t in enumerate(texts)]


PUMPS = [f"Pump {i} runs at {400 + i} RPM in module {i}." for i in range(6)]


async def test_ingest_count_and_list_documents(pg_url):
    store = _store(pg_url)
    await store.add_chunks(_chunks("d1", PUMPS))
    await store.add_chunks(_chunks("d2", ["The HR handbook covers vacation."], "hr.docx"))

    assert store.count() == 7
    docs = store.list_documents()
    assert [(d["doc_id"], d["filename"], d["chunks"]) for d in docs] == [
        ("d2", "hr.docx", 1), ("d1", "a.pdf", 6)]  # newest first


async def test_search_is_top_k_of_the_fused_ranking_with_both_legs(pg_url):
    store = _store(pg_url)
    await store.add_chunks(_chunks("d1", PUMPS))
    query = "Which pump runs at 402 RPM?"
    vector = await store.embed_query(query)

    ranking = await store.fused_ranking(query, vector, doc_ids=["d1"])
    results = await store.search(query, doc_ids=["d1"])

    assert [c.chunk_id for c in results] == [c.chunk_id for c in ranking[:3]]
    assert any(c.dense_rank is not None for c in ranking)
    assert any(c.sparse_rank is not None for c in ranking)
    assert [c.score for c in ranking] == sorted((c.score for c in ranking), reverse=True)
    assert all(c.similarity is None or -1.0 <= c.similarity <= 1.0 for c in ranking)


async def test_sparse_leg_matches_any_shared_term_with_english_stemming(pg_url):
    store = _store(pg_url)
    await store.add_chunks(_chunks("d1", [
        "Cooling pumps were running all night.",
        "The greenhouse uses carbon dioxide flooding.",
    ]))

    # Only "pumping" (stemmed to "pump") is shared; the other words aren't.
    ranking = await store.fused_ranking("pumping schedule overview",
                                        await store.embed_query("x"))

    sparse = {c.chunk_id: c.sparse_rank for c in ranking if c.sparse_rank is not None}
    assert sparse == {"d1-0000": 1}


async def test_scoping_restricts_both_legs(pg_url):
    store = _store(pg_url)
    await store.add_chunks(_chunks("d1", PUMPS))
    await store.add_chunks(_chunks("d2", ["Pump maintenance happens monthly."]))

    ranking = await store.fused_ranking("pump", await store.embed_query("pump"), doc_ids=["d2"])

    assert {c.chunk_id for c in ranking} == {"d2-0000"}


async def test_filtered_hnsw_search_uses_iterative_scans(pg_url):
    """With the HNSW index forced, a filter that discards most candidates
    would leave the dense leg short; iterative scans keep it full. (The
    doc_id index is dropped here: for a filter this selective the planner
    rightly prefers it, which is not the case being tested.)"""
    store = _store(pg_url, rrf_candidates=3)
    await store.add_chunks(_chunks("big", [f"Filler passage number {i}." for i in range(200)]))
    await store.add_chunks(_chunks("small", ["Alpha.", "Beta.", "Gamma."]))
    engine = sa.create_engine(pg_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        conn.execute(sa.text("DROP INDEX ix_chunks_doc_id"))
        conn.execute(sa.text("ANALYZE chunks"))
        conn.execute(sa.text(f'ALTER DATABASE "{engine.url.database}" SET enable_seqscan = off'))
        conn.execute(sa.text(f'ALTER DATABASE "{engine.url.database}" SET hnsw.ef_search = 10'))
    engine.dispose()
    store._engine.dispose()  # new connections pick up the database settings
    vector = await store.embed_query("Filler passage number 7.")

    with store._engine.connect() as conn:
        plan = "\n".join(conn.execute(sa.text(
            "EXPLAIN SELECT chunk_id FROM chunks WHERE doc_id = 'small' "
            "ORDER BY embedding <=> CAST(:v AS vector) LIMIT 3"
        ), {"v": "[" + ",".join(map(str, vector)) + "]"}).scalars())
        assert "ix_chunks_embedding_hnsw" in plan

        # Control: the same filtered HNSW query without iterative scans
        # loses the small document's rows to the filter.
        conn.execute(sa.text("SET hnsw.iterative_scan = off"))
        without = conn.execute(sa.text(
            "SELECT chunk_id FROM chunks WHERE doc_id = 'small' "
            "ORDER BY embedding <=> CAST(:v AS vector) LIMIT 3"
        ), {"v": "[" + ",".join(map(str, vector)) + "]"}).scalars().all()
    assert len(without) < 3

    ranking = await store.fused_ranking("zzz", vector, doc_ids=["small"])

    assert sorted(c.chunk_id for c in ranking if c.dense_rank) == [
        "small-0000", "small-0001", "small-0002"]


async def test_store_refuses_a_different_dimension(pg_url):
    settings = _settings(pg_url, embedding_dimensions=16)
    assert db_session.init_db(settings)

    with pytest.raises(RuntimeError, match=r"vector\(16\).*vector\(32\)"):
        HybridStore(settings, FakeEmbeddings())


async def test_store_refuses_vectors_from_another_model(pg_url):
    store = _store(pg_url)
    await store.add_chunks(_chunks("d1", PUMPS))

    class OtherModel(FakeEmbeddings):
        model = "other-model"

    with pytest.raises(RuntimeError, match="fake-embeddings"):
        HybridStore(_settings(pg_url), OtherModel())


def test_store_needs_the_migrations(pg_url):
    with pytest.raises(RuntimeError, match="run the migrations"):
        HybridStore(_settings(pg_url), FakeEmbeddings())
