"""Orgs, teams, users and documents, plus access columns on chunks.

Firebase only identifies the user; org, teams, role and clearance live here.
`documents` is the source of truth for a document's access (org, team,
clearance); the same three values are copied onto every chunk so a row-level
policy can check a chunk on its own columns, with no join.

  - Composite foreign keys (team_id, org_id) make a team, a membership or a
    document in a different org than its team impossible, not just unlikely.
  - clearance is an integer level; higher sees more. role is
    viewer/editor/admin.
  - Existing chunks get org_id NULL: they belong to no org, so once RLS lands
    (a later revision) no one can see them. Until then nothing filters on it.
  - chunks.doc_id is not a foreign key: chunks from before this revision have
    no documents row.

Postgres only, like 0002 (the SQLite stand-in has no chunks table).

Revision ID: 0003
Revises: 0002
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("""
        CREATE TABLE orgs (
            org_id text PRIMARY KEY,
            name   text NOT NULL
        )
    """)
    op.execute("""
        CREATE TABLE teams (
            team_id text PRIMARY KEY,
            org_id  text NOT NULL REFERENCES orgs,
            name    text NOT NULL,
            UNIQUE (team_id, org_id)
        )
    """)
    op.execute("""
        CREATE TABLE users (
            user_id      text PRIMARY KEY DEFAULT gen_random_uuid()::text,
            firebase_uid text NOT NULL UNIQUE,
            org_id       text REFERENCES orgs,
            role         text NOT NULL DEFAULT 'viewer'
                         CHECK (role IN ('viewer', 'editor', 'admin')),
            clearance    integer NOT NULL DEFAULT 0,
            active       boolean NOT NULL DEFAULT true,
            UNIQUE (user_id, org_id)
        )
    """)
    op.execute("""
        CREATE TABLE team_memberships (
            user_id text NOT NULL,
            team_id text NOT NULL,
            org_id  text NOT NULL,
            PRIMARY KEY (user_id, team_id),
            FOREIGN KEY (user_id, org_id) REFERENCES users (user_id, org_id),
            FOREIGN KEY (team_id, org_id) REFERENCES teams (team_id, org_id)
        )
    """)
    op.execute("""
        CREATE TABLE documents (
            doc_id      text PRIMARY KEY,
            filename    text NOT NULL,
            org_id      text NOT NULL REFERENCES orgs,
            team_id     text,
            clearance   integer NOT NULL DEFAULT 0,
            uploaded_by text REFERENCES users,
            uploaded_at timestamptz NOT NULL DEFAULT now(),
            FOREIGN KEY (team_id, org_id) REFERENCES teams (team_id, org_id)
        )
    """)
    op.execute("""
        ALTER TABLE chunks
            ADD COLUMN org_id    text REFERENCES orgs,
            ADD COLUMN team_id   text,
            ADD COLUMN clearance integer NOT NULL DEFAULT 0,
            ADD FOREIGN KEY (team_id, org_id) REFERENCES teams (team_id, org_id)
    """)
    op.execute("CREATE INDEX ix_chunks_org_id ON chunks (org_id)")


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    # Dropping the columns drops their index and foreign keys with them.
    op.execute("ALTER TABLE chunks DROP COLUMN org_id, DROP COLUMN team_id, DROP COLUMN clearance")
    for table in ("documents", "team_memberships", "users", "teams", "orgs"):
        op.execute(f"DROP TABLE {table}")
