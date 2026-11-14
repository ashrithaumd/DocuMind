"""Hybrid chunk store on Postgres: pgvector (dense) + full-text search
(sparse) over the `chunks` table (migration 0002), fused with Reciprocal
Rank Fusion.

Why hybrid: dense embeddings catch paraphrases ("remuneration" ~ "salary"),
full-text search catches exact terms embeddings dilute (part numbers,
acronyms, names). RRF combines the two rankings without having to calibrate
their incomparable score scales: only ranks matter.

  - Dense leg: cosine distance on the HNSW index. Scoping by doc_id is a
    filter on an approximate index, which can return fewer than n rows
    once the filter discards candidates; iterative index scans
    (hnsw.iterative_scan = relaxed_order, pgvector 0.8) keep scanning until
    n rows pass the filter. relaxed_order may return rows slightly out of
    order, so a materialized CTE re-sorts them by exact distance.
  - Sparse leg: ts_rank_cd over the stored 'english' tsvector. The query is
    plainto_tsquery with its ANDs turned into ORs: any shared term is a
    match. (With AND, a natural-language question rarely has every term in
    one chunk: the gold page of only 4 of the 14 eval questions matched.)
    Only chunks that match get a rank.

Phase 0 used Chroma for the dense leg and an in-memory BM25 for the sparse
one (see docs/PROGRESS.md for the before/after comparison).

The store refuses to start if the vector column's size doesn't match the
configured provider. (Whether stored vectors came from another model is
checked by the migrate step, which can see every org's chunks:
db.session._check_embedding_models.)

Access control is row-level security in Postgres (migration 0004), not
filtering here. `store.scoped(access)` returns a copy bound to one caller's
org, clearance and teams; every transaction it opens first sets those as
transaction-local settings (set_config(..., true)), so both retrieval legs,
listings, counts and writes run under the policy, and nothing outlives the
transaction on a pooled connection. An unscoped store sets nothing and sees
no rows. The store refuses to start on a connection that could bypass RLS.

The documents table is the source of truth for access. add_document writes
the documents row and its chunks in one transaction, the chunks copying org,
team and clearance from that row in SQL. Nothing else writes access.
"""
import asyncio
import copy
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass

import sqlalchemy as sa

from config import Settings
from db.access import WRITER_ROLES, Access, set_access  # noqa: F401  (Access re-exported)
from embeddings import EmbeddingProvider
from ingestion.chunking import Chunk

logger = logging.getLogger(__name__)

DOCUMENT_META = ("doc_type", "department", "version", "effective_date", "superseded_on")



@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    filename: str
    page_number: int
    score: float              # fused RRF score (used for the final ranking)
    dense_rank: int | None    # rank in the dense leg, None if it didn't place
    sparse_rank: int | None   # rank in the full-text leg, None if it didn't place
    similarity: float | None  # cosine similarity, only known for the dense leg


def rrf_fuse(
    dense: dict[str, tuple[int, float, str, dict]],
    sparse: dict[str, tuple[int, str, dict]],
    k: int,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score(c) = sum over legs of 1 / (k + rank).

    `dense` maps chunk_id -> (rank, similarity, text, meta), `sparse` maps
    chunk_id -> (rank, text, meta); meta has filename and page_number. Every
    candidate from either leg comes back, sorted by fused score, keeping its
    rank in each leg.

    Tie-break rule (equal fused scores, which happen whenever two chunks hold
    mirrored ranks, e.g. dense 1 + sparse 2 vs dense 2 + sparse 1): a chunk
    the dense leg found comes first, ordered by its dense rank; chunks only
    the sparse leg found follow, ordered by sparse rank. This is the order
    the Phase 0 store produced implicitly (stable sort over dense-first
    insertion), so run A and the Postgres runs break ties the same way.
    """
    fused: dict[str, RetrievedChunk] = {}
    for cid, (rank, similarity, text, meta) in dense.items():
        fused[cid] = RetrievedChunk(
            chunk_id=cid,
            text=text,
            filename=meta["filename"],
            page_number=meta["page_number"],
            score=1.0 / (k + rank),
            dense_rank=rank,
            sparse_rank=None,
            similarity=round(similarity, 4),
        )
    for cid, (rank, text, meta) in sparse.items():
        if cid in fused:
            fused[cid].score += 1.0 / (k + rank)
            fused[cid].sparse_rank = rank
        else:
            fused[cid] = RetrievedChunk(
                chunk_id=cid,
                text=text,
                filename=meta["filename"],
                page_number=meta["page_number"],
                score=1.0 / (k + rank),
                dense_rank=None,
                sparse_rank=rank,
                similarity=None,
            )
    return sorted(fused.values(), key=_fused_order)


def _fused_order(chunk: RetrievedChunk) -> tuple:
    """Sort key: fused score descending, then the tie-break rule in rrf_fuse."""
    if chunk.dense_rank is not None:
        return (-chunk.score, 0, chunk.dense_rank)
    return (-chunk.score, 1, chunk.sparse_rank)



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


class HybridStore:
    def __init__(self, settings: Settings, embeddings: EmbeddingProvider, engine: sa.Engine | None = None):
        if engine is None:
            if not settings.database_url:
                raise RuntimeError("The Postgres chunk store needs DATABASE_URL.")
            engine = sa.create_engine(settings.database_url, pool_pre_ping=True)
        self._settings = settings
        self._embeddings = embeddings
        self._engine = engine
        self._access = None
        self._check_embedding_index()
        self._check_role_cannot_bypass_rls()

    def scoped(self, access) -> "HybridStore":
        """A copy of this store (same engine) whose transactions run as `access`."""
        scoped = copy.copy(self)
        scoped._access = access
        return scoped

    def _require_writer(self) -> None:
        """Writes (the corpus loader) are for editors and admins only,
        whoever calls the store."""
        if self._access is None:
            raise RuntimeError("Writes need a scoped store (store.scoped(access)).")
        if getattr(self._access, "role", None) not in WRITER_ROLES:
            raise PermissionError("Only editors and admins can write documents.")

    @contextmanager
    def _begin(self):
        with self._engine.begin() as conn:
            if self._access is not None:
                set_access(conn, self._access)
            yield conn

    def _check_role_cannot_bypass_rls(self) -> None:
        """Refuse a connection the policies wouldn't apply to."""
        with self._engine.connect() as conn:
            row = conn.execute(sa.text("""
                SELECT current_user AS name, r.rolsuper, r.rolbypassrls,
                       pg_has_role(current_user, c.relowner, 'USAGE') AS owns
                FROM pg_roles r, pg_class c
                WHERE r.rolname = current_user AND c.oid = 'chunks'::regclass
            """)).mappings().one()
        if row["rolsuper"] or row["rolbypassrls"] or row["owns"]:
            raise RuntimeError(
                f"DATABASE_URL connects as '{row['name']}', which bypasses row-level security "
                "(superuser, BYPASSRLS or table owner). Point DATABASE_URL at the finpilot_app "
                "role and MIGRATION_DATABASE_URL at the owner."
            )

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
        expected = f"vector({self._embeddings.dimension})"
        if column != expected:
            raise RuntimeError(
                f"The chunks table holds {column}, but embedding model '{self._embeddings.model}' "
                f"produces {expected}. It was built for a different embedding model: use a fresh "
                "database (or drop the chunks table and alembic_version) and re-upload your documents."
            )

    # --- index maintenance -------------------------------------------------

    def count(self) -> int:
        """Chunks this store's caller can see."""
        with self._begin() as conn:
            return conn.execute(sa.text("SELECT count(*) FROM chunks")).scalar()

    def list_documents(self) -> list[dict]:
        """Documents the caller can see (the documents table, under RLS),
        newest first."""
        with self._begin() as conn:
            rows = conn.execute(sa.text("""
                SELECT d.doc_id, d.filename, d.uploaded_at, d.team_id, d.clearance,
                       count(c.chunk_id) AS chunks
                FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id
                GROUP BY d.doc_id ORDER BY d.uploaded_at DESC, d.doc_id
            """)).mappings().all()
        return [
            {"doc_id": r["doc_id"], "filename": r["filename"],
             "uploaded_at": r["uploaded_at"].isoformat(), "chunks": r["chunks"],
             "team_id": r["team_id"], "clearance": r["clearance"]}
            for r in rows
        ]

    # --- users (migration 0003) ---------------------------------------------

    def get_user(self, firebase_uid: str) -> dict | None:
        """The user row for a Firebase uid, with its team ids; None if unknown.
        No access context: the user tables have no RLS (the app role can only
        read them)."""
        with self._engine.connect() as conn:
            row = conn.execute(sa.text("""
                SELECT u.user_id, u.firebase_uid, u.org_id, u.role, u.clearance, u.active,
                       u.customer_id,
                       coalesce(array_agg(m.team_id ORDER BY m.team_id)
                                FILTER (WHERE m.team_id IS NOT NULL), '{}') AS team_ids
                FROM users u LEFT JOIN team_memberships m ON m.user_id = u.user_id
                WHERE u.firebase_uid = :uid
                GROUP BY u.user_id
            """), {"uid": firebase_uid}).mappings().first()
        return dict(row) if row else None

    # --- ingestion ----------------------------------------------------------

    async def add_document(self, chunks: list[Chunk], *, team_id: str | None,
                           clearance: int, uploaded_by: str | None, tables=(),
                           meta: dict | None = None, replace: bool = False) -> None:
        """Store one document: its documents row (the caller's org, `team_id`
        or org-wide when None, `clearance`) and its chunks, in one
        transaction. The chunks copy org, team and clearance from that row in
        SQL, so they can't differ from it. RLS refuses a team the caller isn't
        in or a clearance above theirs. `tables` (ingestion.tables.DocTable)
        go in the same transaction and copy access the same way. `meta` is
        the manifest's doc_type, department, version, effective_date and
        superseded_on. `replace` first deletes a document with the same id
        (the corpus loader's stable ids), in the same transaction."""
        if not chunks:
            raise ValueError("add_document needs at least one chunk")
        self._require_writer()
        doc_id, filename = chunks[0].doc_id, chunks[0].filename
        if any(c.doc_id != doc_id for c in chunks):
            raise ValueError("add_document takes the chunks of one document")
        vectors = await self._embeddings.embed_documents([c.text for c in chunks])
        rows = [
            {"chunk_id": c.chunk_id, "doc_id": doc_id, "page_number": c.page_number,
             "chunk_index": c.chunk_index, "text": c.text, "section": c.section,
             "summary": c.summary, "keywords": c.keywords, "questions": c.questions,
             "embedding_model": self._embeddings.model, "embedding": _vector_literal(v)}
            for c, v in zip(chunks, vectors)
        ]

        def insert() -> None:
            # One transaction, one now(): the document and its chunks share
            # uploaded_at, and a failure leaves neither behind.
            with self._begin() as conn:
                if replace:  # tables and their rows go with the document and chunks (cascade)
                    conn.execute(sa.text("DELETE FROM chunks WHERE doc_id = :d"), {"d": doc_id})
                    conn.execute(sa.text("DELETE FROM documents WHERE doc_id = :d"), {"d": doc_id})
                conn.execute(sa.text("""
                    INSERT INTO documents (doc_id, filename, org_id, team_id, clearance, uploaded_by,
                                           doc_type, department, version, effective_date, superseded_on)
                    VALUES (:doc_id, :filename, :org_id, :team_id, :clearance, :uploaded_by,
                            :doc_type, :department, :version, :effective_date, :superseded_on)
                """), {"doc_id": doc_id, "filename": filename, "org_id": self._access.org_id,
                       "team_id": team_id, "clearance": clearance, "uploaded_by": uploaded_by,
                       **{k: (meta or {}).get(k) for k in DOCUMENT_META}})
                inserted = conn.execute(sa.text("""
                    INSERT INTO chunks (chunk_id, doc_id, filename, page_number, chunk_index,
                                        text, section, summary, keywords, questions,
                                        embedding_model, embedding, org_id, team_id, clearance)
                    SELECT :chunk_id, d.doc_id, d.filename, :page_number, :chunk_index,
                           :text, :section, :summary, :keywords, :questions,
                           :embedding_model, CAST(:embedding AS vector),
                           d.org_id, d.team_id, d.clearance
                    FROM documents d WHERE d.doc_id = :doc_id
                """), rows).rowcount
                if inserted != len(rows):  # the documents row must be visible right here
                    raise RuntimeError(f"Inserted {inserted} of {len(rows)} chunks for {doc_id}")
                for table in (t for t in tables if t.rows):
                    conn.execute(sa.text("""
                        INSERT INTO doc_tables (table_id, doc_id, section, columns, org_id, team_id, clearance)
                        SELECT :table_id, d.doc_id, :section, CAST(:columns AS jsonb),
                               d.org_id, d.team_id, d.clearance
                        FROM documents d WHERE d.doc_id = :doc_id
                    """), {"table_id": table.table_id, "doc_id": doc_id, "section": table.section,
                           "columns": json.dumps(table.columns)})
                    conn.execute(sa.text("""
                        INSERT INTO doc_table_rows (table_id, row_index, chunk_id, cells,
                                                    org_id, team_id, clearance)
                        SELECT t.table_id, :row_index, :chunk_id, CAST(:cells AS jsonb),
                               t.org_id, t.team_id, t.clearance
                        FROM doc_tables t WHERE t.table_id = :table_id
                    """), [{"table_id": table.table_id, "row_index": i, "chunk_id": chunk_id,
                            "cells": json.dumps(cells)} for i, (chunk_id, cells) in enumerate(table.rows)])

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
        # Both legs run under the RLS policy: the access check is a filter on
        # each row's own columns, which iterative scans handle like doc_id.
        with self._begin() as conn:
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
