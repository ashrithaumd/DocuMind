"""Structured data pulled out of unstructured sources.

doc_tables, doc_table_rows: tables found in documents, one row per table
row, each linked to the chunk that holds it (chunk_id), so a calculation
can cite its source. They carry the document's org, team and clearance,
copied from the documents row when written, under the same tenant_access
policy as chunks: a staff-only table is as invisible to a customer as the
staff-only text around it. Deleting a document deletes its tables; deleting
a chunk deletes the rows it holds.

receipt_images: synthetic receipt images rendered from a customer's own
transactions, with the fields one vision call extracted and the check of
each field against the transaction. A receipt belongs to a customer, so it
sits under the customer_owner policy (migration 0009), not the documents'.
txn_id is not a foreign key: reloading the synthetic data replaces the
transactions, and the extraction (the cached vision call) must survive it.

Revision ID: 0010
Revises: 0009
"""
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

APP_ROLE = "finpilot_app"
TENANT_POLICY = """
    org_id = nullif(current_setting('app.org_id', true), '')
    AND clearance <= nullif(current_setting('app.clearance', true), '')::integer
    AND (team_id IS NULL
         OR team_id = ANY (nullif(current_setting('app.team_ids', true), '')::text[]))
"""
CUSTOMER_POLICY = "customer_id = nullif(current_setting('app.customer_id', true), '')"

TABLES = """
CREATE TABLE doc_tables (
    table_id  text PRIMARY KEY,
    doc_id    text NOT NULL REFERENCES documents ON DELETE CASCADE,
    section   text NOT NULL,
    columns   jsonb NOT NULL,
    org_id    text NOT NULL,
    team_id   text,
    clearance integer NOT NULL
);
CREATE INDEX ix_doc_tables_doc ON doc_tables (doc_id);

CREATE TABLE doc_table_rows (
    table_id  text NOT NULL REFERENCES doc_tables ON DELETE CASCADE,
    row_index integer NOT NULL,
    chunk_id  text NOT NULL REFERENCES chunks ON DELETE CASCADE,
    cells     jsonb NOT NULL,
    org_id    text NOT NULL,
    team_id   text,
    clearance integer NOT NULL,
    PRIMARY KEY (table_id, row_index)
);
CREATE INDEX ix_doc_table_rows_chunk ON doc_table_rows (chunk_id);

CREATE TABLE receipt_images (
    image_id     text PRIMARY KEY,
    customer_id  text NOT NULL REFERENCES customers,
    txn_id       text NOT NULL,
    path         text NOT NULL,
    model        text,
    fields       jsonb,
    matches      jsonb,
    extracted_at timestamptz
);
CREATE INDEX ix_receipt_images_customer ON receipt_images (customer_id);
"""


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(TABLES)
    for table in ("doc_tables", "doc_table_rows"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY, FORCE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY tenant_access ON {table} USING ({TENANT_POLICY}) WITH CHECK ({TENANT_POLICY})")
    op.execute("ALTER TABLE receipt_images ENABLE ROW LEVEL SECURITY, FORCE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY customer_owner ON receipt_images USING ({CUSTOMER_POLICY}) "
               f"WITH CHECK ({CUSTOMER_POLICY})")
    # The store writes documents as the app role (like chunks); receipts are read-only to it.
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON doc_tables, doc_table_rows TO {APP_ROLE}")
    op.execute(f"GRANT SELECT ON receipt_images TO {APP_ROLE}")


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE receipt_images, doc_table_rows, doc_tables")
