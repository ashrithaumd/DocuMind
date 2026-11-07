"""The documind_app role, and row-level security on chunks and documents.

Migrations run as the owner (MIGRATION_DATABASE_URL); the app connects as
documind_app (DATABASE_URL): not the owner, not a superuser, no BYPASSRLS.
The store refuses to start on a connection that could bypass the policies.

Policy (the same on both tables, for every command, so reads and writes are
checked): a row is visible, and may be written, only when
  - org_id equals app.org_id,
  - clearance <= app.clearance, and
  - team_id is NULL (org-wide) or one of app.team_ids.
The store sets those three settings per transaction (set_config(..., true))
from the principal loaded from Postgres on that request. Unset settings read
as NULL, or '' on a pooled connection that once had them set: nullif() maps
both to NULL, so with no context no row matches (and nothing errors).

Each row is checked on its own columns, with no join, so the HNSW
iterative-scan search and the full-text search simply run under it.

FORCE ROW LEVEL SECURITY applies the policies to the table owner too. Only a
superuser or a BYPASSRLS role skips them.

chunk_embedding_models() is the one deliberate hole: the store's startup
check must see every chunk's embedding model, across all orgs, to refuse a
mismatched provider. It returns model names only.

Grants (least privilege): chunks and documents read/write (under the policy);
the user tables read-only, so the app can load a principal but never change
one; the log tables insert/read (feedback also update). RLS on the log tables
comes with the admin-only diagnostics commit.

The role is cluster-wide, so downgrade revokes its grants but keeps the role
(other databases on the server may use it). Its password is set at startup
from DATABASE_URL (db.session.migrate), not here.

Revision ID: 0004
Revises: 0003
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

APP_ROLE = "documind_app"
RLS_TABLES = ("chunks", "documents")

POLICY = """
    org_id = nullif(current_setting('app.org_id', true), '')
    AND clearance <= nullif(current_setting('app.clearance', true), '')::integer
    AND (team_id IS NULL
         OR team_id = ANY (nullif(current_setting('app.team_ids', true), '')::text[]))
"""


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(f"""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                CREATE ROLE {APP_ROLE} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
            END IF;
        END $$
    """)
    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON chunks, documents TO {APP_ROLE}")
    op.execute(f"GRANT SELECT ON orgs, teams, users, team_memberships TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT ON query_logs TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE ON feedback TO {APP_ROLE}")
    op.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {APP_ROLE}")

    for table in RLS_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY, FORCE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY tenant_access ON {table} USING ({POLICY}) WITH CHECK ({POLICY})")

    # shortcut: runs as the function's owner, which skips RLS only if that
    # owner is a superuser (true for the Docker Postgres). On a managed
    # Postgres with a non-superuser owner (Phase 9), give the owner BYPASSRLS
    # or move the model check to the migration step.
    op.execute("""
        CREATE FUNCTION chunk_embedding_models() RETURNS SETOF text
        LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, public
        AS $$ SELECT DISTINCT embedding_model FROM public.chunks $$
    """)
    op.execute("REVOKE ALL ON FUNCTION chunk_embedding_models() FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION chunk_embedding_models() TO {APP_ROLE}")


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP FUNCTION chunk_embedding_models()")
    for table in RLS_TABLES:
        op.execute(f"DROP POLICY tenant_access ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY, DISABLE ROW LEVEL SECURITY")
    op.execute(f"REVOKE ALL ON chunks, documents, orgs, teams, users, team_memberships, "
               f"query_logs, feedback FROM {APP_ROLE}")
    op.execute(f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {APP_ROLE}")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {APP_ROLE}")
