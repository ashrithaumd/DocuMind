"""chunks: pgvector embeddings + Postgres full-text search.

One row per chunk, replacing Chroma (dense leg) and the in-memory BM25
(sparse leg):

  - embedding vector(N), N = EMBEDDING_DIMENSIONS at migration time (384 for
    the local MiniLM provider, 1536 for gemini-embedding-2), with an HNSW
    index for cosine distance (pgvector defaults m=16, ef_construction=64).
    HNSW needs a fixed dimension, which is why the size comes from config.
  - embedding_model records which model produced each vector; the store
    refuses to start if it, or the column's dimension, doesn't match the
    configured provider (vectors from two models are never comparable).
  - tsv is a stored generated tsvector with the 'english' configuration
    (stemming + English stop words), with a GIN index.

Postgres only: on SQLite (the log-DB stand-in in unit tests) this revision
does nothing, since the chunk index needs pgvector anyway.

Revision ID: 0002
Revises: 0001
"""
from alembic import context, op

from config import get_settings

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    settings = context.config.attributes.get("settings") or get_settings()
    dimensions = int(settings.embedding_dimensions)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(f"""
        CREATE TABLE chunks (
            chunk_id        text PRIMARY KEY,
            doc_id          text NOT NULL,
            filename        text NOT NULL,
            page_number     integer NOT NULL,
            chunk_index     integer NOT NULL,
            text            text NOT NULL,
            uploaded_at     timestamptz NOT NULL DEFAULT now(),
            embedding_model text NOT NULL,
            embedding       vector({dimensions}) NOT NULL,
            tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        )
    """)
    op.execute("CREATE INDEX ix_chunks_doc_id ON chunks (doc_id)")
    op.execute("CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
               "USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")


def downgrade() -> None:
    raise NotImplementedError("forward-only")
