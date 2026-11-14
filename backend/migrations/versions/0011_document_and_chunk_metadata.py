"""Metadata for retrieval.

documents: doc_type, department, version, effective_date, superseded_on,
from the corpus manifest (data/bank/manifest.yaml), never from the text, so
Phase 4 can filter "as of" a date in SQL.

chunks: section (the heading path the chunk starts in, for parent-section
joins), and summary, keywords and hypothetical questions written by one
LLM call per few chunks (bank/metadata.py). Empty until generated; the
chunks are usable without them.

Revision ID: 0011
Revises: 0010
"""
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("""
        ALTER TABLE documents
            ADD COLUMN doc_type       text,
            ADD COLUMN department     text,
            ADD COLUMN version        text,
            ADD COLUMN effective_date date,
            ADD COLUMN superseded_on  date
    """)
    op.execute("""
        ALTER TABLE chunks
            ADD COLUMN section   text NOT NULL DEFAULT '',
            ADD COLUMN summary   text NOT NULL DEFAULT '',
            ADD COLUMN keywords  text[] NOT NULL DEFAULT '{}',
            ADD COLUMN questions text[] NOT NULL DEFAULT '{}'
    """)


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE chunks DROP COLUMN section, DROP COLUMN summary, "
               "DROP COLUMN keywords, DROP COLUMN questions")
    op.execute("ALTER TABLE documents DROP COLUMN doc_type, DROP COLUMN department, "
               "DROP COLUMN version, DROP COLUMN effective_date, DROP COLUMN superseded_on")
