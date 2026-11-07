"""The documents table becomes the write path for access.

  - clearance on chunks and documents: NOT NULL with no default. 0003 gave
    both DEFAULT 0, so a write that forgot clearance silently got the lowest
    level. Now it fails: a document's clearance is always chosen, and its
    chunks copy it from the documents row (ingestion/store.py, add_document).
  - chunks.doc_id references documents: a chunk can't exist without its
    documents row, the source of its access. NOT VALID: chunks from before
    0003 have no documents row (and no org, so RLS hides them); new and
    updated chunks are checked.

Revision ID: 0006
Revises: 0005
"""
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    for table in ("chunks", "documents"):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN clearance DROP DEFAULT, "
                   f"ALTER COLUMN clearance SET NOT NULL")
    op.execute("ALTER TABLE chunks ADD CONSTRAINT fk_chunks_document FOREIGN KEY (doc_id) "
               "REFERENCES documents (doc_id) NOT VALID")


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE chunks DROP CONSTRAINT fk_chunks_document")
    for table in ("chunks", "documents"):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN clearance SET DEFAULT 0")
