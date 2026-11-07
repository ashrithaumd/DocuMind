"""Query and feedback logs record who asked, under RLS by org.

  - query_logs and feedback get user_id and org_id (the caller's, from the
    principal). Rows from before this revision have neither: no org, so
    nobody sees them under RLS.
  - On Postgres, ENABLE + FORCE ROW LEVEL SECURITY on both, with one policy
    for every command: org_id = app.org_id (set per transaction, like the
    chunks policy). Admins read their own org's logs; nobody reads
    another org's, and a write into another org is refused.

The columns are added on SQLite too (the log-table stand-in in unit tests
must match the models); RLS is Postgres only.

Revision ID: 0007
Revises: 0006
"""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

LOG_TABLES = ("query_logs", "feedback")
POLICY = "org_id = nullif(current_setting('app.org_id', true), '')"


def upgrade() -> None:
    for table in LOG_TABLES:
        with op.batch_alter_table(table) as batch:
            batch.add_column(sa.Column("user_id", sa.Text(), nullable=True))
            batch.add_column(sa.Column("org_id", sa.Text(), nullable=True))
            batch.create_index(f"ix_{table}_org_id", ["org_id"])
    if op.get_bind().dialect.name != "postgresql":
        return
    for table in LOG_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY, FORCE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY org_access ON {table} USING ({POLICY}) WITH CHECK ({POLICY})")


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        for table in LOG_TABLES:
            op.execute(f"DROP POLICY org_access ON {table}")
            op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY, DISABLE ROW LEVEL SECURITY")
    for table in LOG_TABLES:
        with op.batch_alter_table(table) as batch:
            batch.drop_index(f"ix_{table}_org_id")
            batch.drop_column("org_id")
            batch.drop_column("user_id")
