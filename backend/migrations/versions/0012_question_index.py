"""The hypothetical-question index.

chunks.question_embedding: one vector per chunk for its LLM-written
hypothetical questions (joined), searched as a third retrieval leg beside
the chunk vectors and full-text search. NULL until the chunk has questions.
Same type as chunks.embedding (read from the table, so the two can't
differ), same model (chunks.embedding_model), HNSW cosine index. It lives
on the chunk row, so the chunks policy covers it.

documents (department, doc_type, effective_date, superseded_on) gets an
index for the metadata filters' subquery.

Revision ID: 0012
Revises: 0011
"""
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    vector_type = bind.exec_driver_sql(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'").scalar()
    op.execute(f"ALTER TABLE chunks ADD COLUMN question_embedding {vector_type}")
    op.execute("CREATE INDEX ix_chunks_question_embedding_hnsw ON chunks "
               "USING hnsw (question_embedding vector_cosine_ops)")
    op.execute("CREATE INDEX ix_documents_filters ON documents "
               "(department, doc_type, effective_date, superseded_on)")


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP INDEX ix_documents_filters")
    op.execute("ALTER TABLE chunks DROP COLUMN question_embedding")
