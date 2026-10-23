"""Hybrid chunk store on Postgres: pgvector (dense) + full-text search
(sparse) over the `chunks` table (migration 0002), fused with RRF.

Same interface and fusion as the Chroma + BM25 store it replaces, so the
pipeline and the benchmark don't change:

  - Dense leg: cosine distance on the HNSW index. Scoping by doc_id is a
    filter on an approximate index, which can return fewer than n rows
    once the filter discards candidates; iterative index scans
    (hnsw.iterative_scan = relaxed_order, pgvector 0.8) keep scanning until
    n rows pass the filter. relaxed_order may return rows slightly out of
    order, so a materialized CTE re-sorts them by exact distance.
  - Sparse leg: ts_rank_cd over the stored 'english' tsvector. The query is
    plainto_tsquery with its ANDs turned into ORs: like BM25, any shared
    term is a match. (With AND, a natural-language question rarely has
    every term in one chunk: the gold page of only 4 of the 14 eval
    questions matched.) Only chunks that match rank, like BM25's
    zero-score cut.

The store refuses to start if the vector column's size or the stored
embedding_model doesn't match the configured provider.
"""
import asyncio
import logging

import sqlalchemy as sa

from config import Settings
from embeddings import EmbeddingProvider
from ingestion.chunking import Chunk
from ingestion.store import RetrievedChunk, rrf_fuse

logger = logging.getLogger(__name__)

_DENSE = sa.text("""
    WITH candidates AS MATERIALIZED (
        SELECT chunk_id, text, filename, page_number,
               embedding <=> CAST(:query_vector AS vector) AS distance
        FROM chunks
        WHERE CAST(:doc_ids AS text[]) IS NULL OR doc_id = ANY(CAST(:doc_ids AS text[]))
        ORDER BY distance
        LIMIT :n
    )
    SELECT chunk_id, text, filename, page_number, distance
    FROM candidates
    ORDER BY distance + 0
""")

_SPARSE = sa.text("""
    SELECT chunk_id, text, filename, page_number, ts_rank_cd(tsv, query) AS score
    FROM chunks,
         CAST(replace(plainto_tsquery('english', :query)::text, '&', '|') AS tsquery) AS query
    WHERE tsv @@ query
      AND (CAST(:doc_ids AS text[]) IS NULL OR doc_id = ANY(CAST(:doc_ids AS text[])))
    ORDER BY score DESC, chunk_id
    LIMIT :n
""")


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class PgHybridStore:
    def __init__(self, settings: Settings, embeddings: EmbeddingProvider, engine: sa.Engine | None = None):
        if engine is None:
            if not settings.database_url:
                raise RuntimeError("The Postgres chunk store needs DATABASE_URL.")
            engine = sa.create_engine(settings.database_url, pool_pre_ping=True)
        self._settings = settings
        self._embeddings = embeddings
        self._engine = engine
        self._check_embedding_index()

    def _check_embedding_index(self) -> None:
        """Fail fast if the chunks table can't hold this provider's vectors."""
        with self._engine.connect() as conn:
            column = conn.execute(sa.text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'"
            )).scalar()
            if column is None:
                raise RuntimeError("No chunks table: run the migrations (alembic upgrade head) "
                                   "against DATABASE_URL first.")
            models = conn.execute(sa.text("SELECT DISTINCT embedding_model FROM chunks")).scalars().all()
        expected = f"vector({self._embeddings.dimension})"
        if column != expected:
            raise RuntimeError(
                f"The chunks table holds {column}, but embedding model '{self._embeddings.model}' "
                f"produces {expected}. It was built for a different embedding model: use a fresh "
                "database (or drop the chunks table and alembic_version) and re-upload your documents."
            )
        foreign = [m for m in models if m != self._embeddings.model]
        if foreign:
            raise RuntimeError(
                f"The chunks table holds vectors from {foreign}, but the configured embedding "
                f"model is '{self._embeddings.model}'. Vectors from different models aren't "
                "comparable: use a fresh database and re-upload your documents."
            )

    # --- index maintenance -------------------------------------------------

    def count(self) -> int:
        with self._engine.connect() as conn:
            return conn.execute(sa.text("SELECT count(*) FROM chunks")).scalar()

    def list_documents(self) -> list[dict]:
        """Distinct uploaded documents, newest first."""
        with self._engine.connect() as conn:
            rows = conn.execute(sa.text("""
                SELECT doc_id, min(filename) AS filename, max(uploaded_at) AS uploaded_at,
                       count(*) AS chunks
                FROM chunks GROUP BY doc_id ORDER BY max(uploaded_at) DESC, doc_id
            """)).mappings().all()
        return [
            {"doc_id": r["doc_id"], "filename": r["filename"],
             "uploaded_at": r["uploaded_at"].isoformat(), "chunks": r["chunks"]}
            for r in rows
        ]

    # --- ingestion ----------------------------------------------------------

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await self._embeddings.embed_documents([c.text for c in chunks])
        rows = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "filename": c.filename,
             "page_number": c.page_number, "chunk_index": c.chunk_index, "text": c.text,
             "embedding_model": self._embeddings.model, "embedding": _vector_literal(v)}
            for c, v in zip(chunks, vectors)
        ]

        def insert() -> None:
            # One transaction, one now(): a document's chunks share uploaded_at.
            with self._engine.begin() as conn:
                conn.execute(sa.text("""
                    INSERT INTO chunks (chunk_id, doc_id, filename, page_number, chunk_index,
                                        text, embedding_model, embedding)
                    VALUES (:chunk_id, :doc_id, :filename, :page_number, :chunk_index,
                            :text, :embedding_model, CAST(:embedding AS vector))
                """), rows)

        await asyncio.to_thread(insert)

    # --- retrieval ----------------------------------------------------------

    async def embed_query(self, query: str) -> list[float]:
        return await self._embeddings.embed_query(query)

    async def search(self, query: str, doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        """Hybrid search: top candidates from each leg, fused with RRF.
        `doc_ids` scopes both legs to those documents."""
        return await self.search_with_vector(query, await self.embed_query(query), doc_ids)

    async def search_with_vector(
        self, query: str, query_vector: list[float], doc_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """search() with the query already embedded, so the benchmark can
        time embedding and search separately."""
        return (await self.fused_ranking(query, query_vector, doc_ids))[: self._settings.top_k]

    async def fused_ranking(
        self, query: str, query_vector: list[float], doc_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Every candidate from either leg, fused and sorted, before the top_k
        cut, each keeping its rank in each leg."""
        return await asyncio.to_thread(self._fused_ranking, query, query_vector, doc_ids)

    def _fused_ranking(self, query: str, query_vector: list[float],
                       doc_ids: list[str] | None) -> list[RetrievedChunk]:
        params = {"doc_ids": list(doc_ids) if doc_ids else None, "n": self._settings.rrf_candidates}
        with self._engine.begin() as conn:
            conn.execute(sa.text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            dense_rows = conn.execute(
                _DENSE, {**params, "query_vector": _vector_literal(query_vector)}
            ).mappings().all()
            sparse_rows = conn.execute(_SPARSE, {**params, "query": query}).mappings().all()

        dense = {
            r["chunk_id"]: (rank, 1.0 - r["distance"], r["text"],
                            {"filename": r["filename"], "page_number": r["page_number"]})
            for rank, r in enumerate(dense_rows, start=1)
        }
        sparse = {
            r["chunk_id"]: (rank, r["text"],
                            {"filename": r["filename"], "page_number": r["page_number"]})
            for rank, r in enumerate(sparse_rows, start=1)
        }
        return rrf_fuse(dense, sparse, self._settings.rrf_k)
