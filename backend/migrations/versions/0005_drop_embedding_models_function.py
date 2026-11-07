"""Drop chunk_embedding_models(), the SECURITY DEFINER hole from 0004.

The embedding-model check moved to the one-shot migrate step
(db.session._check_embedding_models), which runs as the owner. The app role
no longer needs a way to read every org's chunks, so it loses it.

Revision ID: 0005
Revises: 0004
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP FUNCTION chunk_embedding_models()")


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("""
        CREATE FUNCTION chunk_embedding_models() RETURNS SETOF text
        LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, public
        AS $$ SELECT DISTINCT embedding_model FROM public.chunks $$
    """)
    op.execute("REVOKE ALL ON FUNCTION chunk_embedding_models() FROM PUBLIC")
    op.execute("GRANT EXECUTE ON FUNCTION chunk_embedding_models() TO documind_app")
