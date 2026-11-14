"""Hybrid chunk store on Postgres: pgvector (dense) + full-text search
(sparse) + the hypothetical-question index (migration 0012: one vector per
chunk for the questions it answers) over the `chunks` table, fused with
Reciprocal Rank Fusion. RETRIEVAL_MODE=dense uses the chunk vectors alone.
After fusion, optionally: a cross-encoder rerank (RERANK, ingestion/rerank.py)
and parent-section expansion (PARENT_EXPANSION: each hit replaced by its
top-level section's text around it, up to PARENT_MAX_TOKENS). Metadata
filters (Filters: department, doc_type, the version in force on a date,
today by default) apply to every leg in SQL.

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
from dataclasses import dataclass, field, replace
from datetime import date

import sqlalchemy as sa

from config import Settings
from db.access import WRITER_ROLES, Access, set_access  # noqa: F401  (Access re-exported)
from embeddings import EmbeddingProvider
from ingestion.chunking import Chunk, tokens
from ingestion.rerank import Reranker

logger = logging.getLogger(__name__)

DOCUMENT_META = ("doc_type", "department", "version", "effective_date", "superseded_on")



@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str                 # the chunk's text, or its parent section's (parent_expansion)
    filename: str
    page_number: int
    score: float              # fused RRF score (used for the final ranking)
    dense_rank: int | None    # rank in the dense leg, None if it didn't place
    sparse_rank: int | None   # rank in the full-text leg, None if it didn't place
    similarity: float | None  # cosine similarity, only known for the dense leg
    question_rank: int | None = None  # rank in the hypothetical-question leg
    rerank_score: float | None = None  # cross-encoder score, when reranked
    doc_id: str = ""
    section: str = ""
    chunk_index: int = 0
    # The chunks whose text `text` holds: [chunk_id], or the parent section's
    # chunks around it when expanded.
    context_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Filters:
    """Metadata filters, applied in SQL to every retrieval leg. By default
    only versions in force today: effective on or before `as_of` and not
    superseded by then (a document without dates is always in force)."""
    department: str | None = None
    doc_type: str | None = None
    as_of: date | None = None   # None: today
    all_versions: bool = False  # True: superseded and not-yet-effective versions too

    def params(self) -> dict:
        return {"department": self.department, "doc_type": self.doc_type,
                "as_of": None if self.all_versions else (self.as_of or date.today())}


def rrf_fuse(
    dense: dict[str, tuple[int, float, str, dict]],
    sparse: dict[str, tuple[int, str, dict]],
    k: int,
    questions: dict[str, tuple[int, str, dict]] | None = None,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score(c) = sum over legs of 1 / (k + rank).

    `dense` maps chunk_id -> (rank, similarity, text, meta); `sparse` and
    `questions` (the hypothetical-question leg) map chunk_id -> (rank, text,
    meta); meta has filename and page_number (and doc_id, section,
    chunk_index from the store). Every candidate from any leg comes back,
    sorted by fused score, keeping its rank in each leg.

    Tie-break rule (equal fused scores, which happen whenever two chunks hold
    mirrored ranks, e.g. dense 1 + sparse 2 vs dense 2 + sparse 1): a chunk
    the dense leg found comes first, ordered by its dense rank; then chunks
    the sparse leg found, by sparse rank; then question-leg-only chunks.
    The first two are the order the Phase 0 store produced implicitly
    (stable sort over dense-first insertion), so run A and the Postgres runs
    break ties the same way.
    """
    fused: dict[str, RetrievedChunk] = {}
    for cid, (rank, similarity, text, meta) in dense.items():
        fused[cid] = RetrievedChunk(chunk_id=cid, text=text, score=1.0 / (k + rank), dense_rank=rank,
                                    sparse_rank=None, similarity=round(similarity, 4), **meta)
    for leg, rows in (("sparse_rank", sparse), ("question_rank", questions or {})):
        for cid, (rank, text, meta) in rows.items():
            if cid not in fused:
                fused[cid] = RetrievedChunk(chunk_id=cid, text=text, score=0.0, dense_rank=None,
                                            sparse_rank=None, similarity=None, **meta)
            fused[cid].score += 1.0 / (k + rank)
            setattr(fused[cid], leg, rank)
    return sorted(fused.values(), key=_fused_order)


def _fused_order(chunk: RetrievedChunk) -> tuple:
    """Sort key: fused score descending, then the tie-break rule in rrf_fuse."""
    if chunk.dense_rank is not None:
        return (-chunk.score, 0, chunk.dense_rank)
    if chunk.sparse_rank is not None:
        return (-chunk.score, 1, chunk.sparse_rank)
    return (-chunk.score, 2, chunk.question_rank)


def parent_of(section: str) -> str:
    """The top-level section a chunk belongs to: title > heading (a sheet or
    an API operation is its own)."""
    return " > ".join(section.split(" > ")[:2])


def expand_to_parents(hits: list[RetrievedChunk], rows: list[dict], max_tokens: int) -> list[RetrievedChunk]:
    """Give each hit its parent section's text: the chunks of the same document
    and top-level section (`rows`: chunk_id, doc_id, section, text, in
    chunk_index order), grown outward from the hit while under `max_tokens`
    (always contiguous, always including the hit). A second hit in a parent
    already given is dropped: its text is in the context once. A hit with no
    section (a page-chunked upload) stays as it is."""
    from ingestion.chunking import tokens

    parents: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        parents.setdefault((row["doc_id"], parent_of(row["section"])), []).append(row)
    out, given = [], set()
    for hit in hits:
        key = (hit.doc_id, parent_of(hit.section))
        members = parents.get(key, []) if hit.section else []
        at = next((i for i, m in enumerate(members) if m["chunk_id"] == hit.chunk_id), None)
        if at is None:
            out.append(replace(hit, context_chunk_ids=[hit.chunk_id]))
            continue
        if key in given:
            continue
        given.add(key)
        lo = hi = at
        grew = True
        while grew:
            grew = False
            for j in (hi + 1, lo - 1):  # after, then before
                if 0 <= j < len(members) and \
                        tokens(_joined(members[min(lo, j):max(hi, j) + 1])) <= max_tokens:
                    lo, hi, grew = min(lo, j), max(hi, j), True
        window = members[lo:hi + 1]
        out.append(replace(hit, text=_joined(window), context_chunk_ids=[m["chunk_id"] for m in window]))
    return out


def _joined(window: list[dict]) -> str:
    """Chunk texts in order, each section's path line kept once."""
    parts, previous = [], None
    for m in window:
        text = m["text"]
        if m["section"] == previous and text.startswith(m["section"] + "\n\n"):
            text = text[len(m["section"]) + 2:]
        parts.append(text)
        previous = m["section"]
    return "\n\n".join(parts)


# Every leg: the caller's doc_ids, then the metadata filters (Filters.params)
# as a subquery on documents, which RLS also covers.
_WHERE = """
    (CAST(:doc_ids AS text[]) IS NULL OR doc_id = ANY(CAST(:doc_ids AS text[])))
    AND (CAST(:department AS text) IS NULL AND CAST(:doc_type AS text) IS NULL
         AND CAST(:as_of AS date) IS NULL
         OR doc_id IN (
            SELECT d.doc_id FROM documents d
            WHERE (CAST(:department AS text) IS NULL OR d.department = CAST(:department AS text))
              AND (CAST(:doc_type AS text) IS NULL OR d.doc_type = CAST(:doc_type AS text))
              AND (CAST(:as_of AS date) IS NULL
                   OR (coalesce(d.effective_date <= CAST(:as_of AS date), true)
                       AND coalesce(d.superseded_on > CAST(:as_of AS date), true)))))
"""
_COLUMNS = "chunk_id, doc_id, chunk_index, section, text, filename, page_number"


def _nearest(column: str) -> sa.TextClause:
    return sa.text(f"""
        WITH candidates AS MATERIALIZED (
            SELECT {_COLUMNS}, {column} <=> CAST(:query_vector AS vector) AS distance
            FROM chunks
            WHERE {column} IS NOT NULL AND {_WHERE}
            ORDER BY distance
            LIMIT :n
        )
        SELECT * FROM candidates
        ORDER BY distance + 0
    """)


_DENSE = _nearest("embedding")
_QUESTIONS = _nearest("question_embedding")

_SPARSE = sa.text(f"""
    SELECT {_COLUMNS}, ts_rank_cd(tsv, query) AS score
    FROM chunks,
         CAST(replace(plainto_tsquery('english', :query)::text, '&', '|') AS tsquery) AS query
    WHERE tsv @@ query AND {_WHERE}
    ORDER BY score DESC, chunk_id
    LIMIT :n
""")


def _meta(row) -> dict:
    return {k: row[k] for k in ("filename", "page_number", "doc_id", "section", "chunk_index")}


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class HybridStore:
    def __init__(self, settings: Settings, embeddings: EmbeddingProvider, engine: sa.Engine | None = None,
                 reranker: Reranker | None = None):
        if engine is None:
            if not settings.database_url:
                raise RuntimeError("The Postgres chunk store needs DATABASE_URL.")
            engine = sa.create_engine(settings.database_url, pool_pre_ping=True)
        self._settings = settings
        self._embeddings = embeddings
        self._engine = engine
        self._reranker = reranker or Reranker(settings.rerank_model)  # loads on first use
        self._access = None
        self._check_embedding_index()
        self._check_role_cannot_bypass_rls()

    def scoped(self, access) -> "HybridStore":
        """A copy of this store (same engine) whose transactions run as `access`."""
        scoped = copy.copy(self)
        scoped._access = access
        return scoped

    def configured(self, **overrides) -> "HybridStore":
        """A copy with retrieval settings overridden (retrieval_mode, rerank,
        parent_expansion, top_k...), so modes can be compared on one store."""
        configured = copy.copy(self)
        configured._settings = self._settings.model_copy(update=overrides)
        return configured

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

    def _stored_vectors(self, doc_id: str) -> dict[str, str]:
        """Text -> its stored vector (pgvector's text form): chunk texts and
        joined question lists, for this document and the current embedding
        model, as far as RLS shows it."""
        with self._begin() as conn:
            return dict(conn.execute(sa.text("""
                SELECT text, embedding::text FROM chunks WHERE doc_id = :d AND embedding_model = :m
                UNION ALL
                SELECT array_to_string(questions, chr(10)), question_embedding::text FROM chunks
                WHERE doc_id = :d AND embedding_model = :m AND question_embedding IS NOT NULL
            """), {"d": doc_id, "m": self._embeddings.model}).all())

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
        # Each chunk's hypothetical questions, joined, get one vector of their
        # own (the question index). A replaced document keeps the vectors of
        # text that didn't change (same model); only new text is embedded.
        asked = {c.chunk_id: "\n".join(c.questions) for c in chunks if c.questions}
        vectors = await asyncio.to_thread(self._stored_vectors, doc_id) if replace else {}
        todo = list(dict.fromkeys(t for t in [c.text for c in chunks] + list(asked.values())
                                  if t not in vectors))
        if todo:
            fresh = await self._embeddings.embed_documents(todo)
            vectors |= {t: _vector_literal(v) for t, v in zip(todo, fresh)}
        rows = [
            {"chunk_id": c.chunk_id, "doc_id": doc_id, "page_number": c.page_number,
             "chunk_index": c.chunk_index, "text": c.text, "section": c.section,
             "summary": c.summary, "keywords": c.keywords, "questions": c.questions,
             "embedding_model": self._embeddings.model, "embedding": vectors[c.text],
             "question_embedding": vectors.get(asked.get(c.chunk_id))}
            for c in chunks
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
                                        embedding_model, embedding, question_embedding,
                                        org_id, team_id, clearance)
                    SELECT :chunk_id, d.doc_id, d.filename, :page_number, :chunk_index,
                           :text, :section, :summary, :keywords, :questions,
                           :embedding_model, CAST(:embedding AS vector),
                           CAST(:question_embedding AS vector),
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

    async def search(self, query: str, doc_ids: list[str] | None = None,
                     filters: Filters | None = None) -> list[RetrievedChunk]:
        """Retrieve the top_k chunks for `query`: the legs of RETRIEVAL_MODE
        fused with RRF, optionally reranked (RERANK), optionally expanded to
        their parent sections (PARENT_EXPANSION). `doc_ids` scopes every leg
        to those documents; `filters` defaults to the versions in force today."""
        return await self.search_with_vector(query, await self.embed_query(query), doc_ids, filters)

    async def search_with_vector(
        self, query: str, query_vector: list[float], doc_ids: list[str] | None = None,
        filters: Filters | None = None,
    ) -> list[RetrievedChunk]:
        """search() with the query already embedded, so the benchmark can
        time embedding and search separately."""
        return await asyncio.to_thread(self._search, query, query_vector, doc_ids, filters)

    async def fused_ranking(
        self, query: str, query_vector: list[float], doc_ids: list[str] | None = None,
        filters: Filters | None = None,
    ) -> list[RetrievedChunk]:
        """Every candidate from the legs, fused and sorted, before reranking
        and the top_k cut, each keeping its rank in each leg."""
        return await asyncio.to_thread(self._fused_ranking, query, query_vector, doc_ids, filters)

    def _search(self, query: str, query_vector: list[float], doc_ids: list[str] | None,
                filters: Filters | None) -> list[RetrievedChunk]:
        ranked = self._fused_ranking(query, query_vector, doc_ids, filters)
        if self._settings.rerank and ranked:
            head = ranked[: self._settings.rerank_candidates]
            for chunk, score in zip(head, self._reranker.scores(query, [c.text for c in head])):
                chunk.rerank_score = round(score, 4)
            ranked = sorted(head, key=lambda c: -c.rerank_score)
        hits = ranked[: self._settings.top_k]
        if self._settings.parent_expansion and hits:
            return self._expanded(hits)
        return [replace(h, context_chunk_ids=[h.chunk_id]) for h in hits]

    def _expanded(self, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
        with self._begin() as conn:
            rows = conn.execute(sa.text(
                "SELECT chunk_id, doc_id, section, text FROM chunks "
                "WHERE doc_id = ANY(:d) AND section <> '' ORDER BY doc_id, chunk_index"),
                {"d": sorted({h.doc_id for h in hits})}).mappings().all()
        return expand_to_parents(hits, rows, self._settings.parent_max_tokens)

    def _fused_ranking(self, query: str, query_vector: list[float], doc_ids: list[str] | None,
                       filters: Filters | None) -> list[RetrievedChunk]:
        params = {"doc_ids": list(doc_ids) if doc_ids else None, "n": self._settings.rrf_candidates,
                  **(filters or Filters()).params()}
        vector = {**params, "query_vector": _vector_literal(query_vector)}
        hybrid = self._settings.retrieval_mode == "hybrid"
        # Every leg runs under the RLS policy: the access check is a filter on
        # each row's own columns, which iterative scans handle like doc_id.
        with self._begin() as conn:
            conn.execute(sa.text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            dense_rows = conn.execute(_DENSE, vector).mappings().all()
            sparse_rows = conn.execute(_SPARSE, {**params, "query": query}).mappings().all() if hybrid else []
            question_rows = conn.execute(_QUESTIONS, vector).mappings().all() if hybrid else []

        dense = {r["chunk_id"]: (rank, 1.0 - r["distance"], r["text"], _meta(r))
                 for rank, r in enumerate(dense_rows, start=1)}
        sparse = {r["chunk_id"]: (rank, r["text"], _meta(r)) for rank, r in enumerate(sparse_rows, start=1)}
        questions = {r["chunk_id"]: (rank, r["text"], _meta(r)) for rank, r in enumerate(question_rows, start=1)}
        return rrf_fuse(dense, sparse, self._settings.rrf_k, questions)
