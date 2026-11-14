"""The app role becomes finpilot_app (the project was renamed from DocuMind).

finpilot_app gets exactly the table grants documind_app had (0004; 0005
dropped its one function grant), then documind_app loses them in this
database. Not ALTER ROLE ... RENAME: roles
are cluster-wide, so the second database migrated on a server (every test
database, for one) would find finpilot_app already there and 0004 would have
created documind_app again. Creating if missing and moving the grants works
on every database. documind_app itself is kept, like 0004's downgrade keeps
it: other databases on the server may still use it.

Revision ID: 0008
Revises: 0007
"""
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

OLD, NEW = "documind_app", "finpilot_app"


def _grant(role: str) -> None:
    op.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON chunks, documents TO {role}")
    op.execute(f"GRANT SELECT ON orgs, teams, users, team_memberships TO {role}")
    op.execute(f"GRANT SELECT, INSERT ON query_logs TO {role}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE ON feedback TO {role}")
    op.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}")


def _revoke(role: str) -> None:
    op.execute(f"REVOKE ALL ON chunks, documents, orgs, teams, users, team_memberships, "
               f"query_logs, feedback FROM {role}")
    op.execute(f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {role}")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {role}")


def _ensure_role(role: str) -> None:
    op.execute(f"""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') THEN
                CREATE ROLE {role} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
            END IF;
        END $$
    """)


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    _ensure_role(NEW)
    _grant(NEW)
    _revoke(OLD)


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    _ensure_role(OLD)
    _grant(OLD)
    _revoke(NEW)
