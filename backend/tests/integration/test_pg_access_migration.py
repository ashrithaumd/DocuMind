"""Migration 0003 on real Postgres: access tables and chunk columns appear on
upgrade, old chunks end up in no org, org/team mismatches are impossible, and
downgrade restores the 0002 schema without touching chunk data."""
import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.exc import IntegrityError

from config import Settings
from db.session import ALEMBIC_INI

NEW_TABLES = {"orgs", "teams", "users", "team_memberships", "documents"}
VEC = "[" + ",".join(["0.1"] * 384) + "]"


@pytest.fixture
def at():
    """Run alembic on an engine to a given revision."""
    def run(engine, revision, downgrade=False):
        settings = Settings(_env_file=None, database_url=engine.url.render_as_string(hide_password=False),
                            embedding_provider="local")
        config = Config(str(ALEMBIC_INI))
        with engine.begin() as conn:
            config.attributes["connection"] = conn
            config.attributes["settings"] = settings
            (command.downgrade if downgrade else command.upgrade)(config, revision)
    return run


def _tables(engine):
    return set(sa.inspect(engine).get_table_names())


def _chunk_columns(engine):
    return {c["name"] for c in sa.inspect(engine).get_columns("chunks")}


def _insert_chunk(conn):
    conn.execute(sa.text(
        "INSERT INTO chunks (chunk_id, doc_id, filename, page_number, chunk_index, text, "
        "embedding_model, embedding) VALUES ('d-0', 'd', 'f.pdf', 1, 0, 'x', 'm', :v)"
    ), {"v": VEC})


def test_upgrade_adds_tables_and_leaves_old_chunks_in_no_org(pg_url, at):
    engine = sa.create_engine(pg_url)
    at(engine, "0002")
    with engine.begin() as conn:
        _insert_chunk(conn)
    assert not NEW_TABLES & _tables(engine)

    at(engine, "head")

    assert NEW_TABLES <= _tables(engine)
    with engine.connect() as conn:
        row = conn.execute(sa.text("SELECT org_id, team_id, clearance FROM chunks")).one()
    assert tuple(row) == (None, None, 0)


def test_a_team_cannot_cross_orgs(pg_url, at):
    engine = sa.create_engine(pg_url)
    at(engine, "head")
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO orgs VALUES ('a', 'A'), ('b', 'B')"))
        conn.execute(sa.text("INSERT INTO teams VALUES ('ta', 'a', 'Team A')"))
        conn.execute(sa.text("INSERT INTO users (user_id, firebase_uid, org_id) VALUES ('ub', 'f1', 'b')"))

    # A document of org b on org a's team; a member of org b on org a's team
    # (under either org id); a chunk likewise.
    for sql in (
        "INSERT INTO documents (doc_id, filename, org_id, team_id) VALUES ('d', 'f', 'b', 'ta')",
        "INSERT INTO team_memberships VALUES ('ub', 'ta', 'b')",
        "INSERT INTO team_memberships VALUES ('ub', 'ta', 'a')",
    ):
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(sa.text(sql))
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO chunks (chunk_id, doc_id, filename, page_number, chunk_index, text, "
            "embedding_model, embedding, org_id, team_id) "
            "VALUES ('c', 'd', 'f', 1, 0, 'x', 'm', :v, 'b', 'ta')"), {"v": VEC})


def test_user_defaults_and_role_check(pg_url, at):
    engine = sa.create_engine(pg_url)
    at(engine, "head")
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO users (firebase_uid) VALUES ('f1')"))
        row = conn.execute(sa.text("SELECT user_id, org_id, role, clearance, active FROM users")).one()
    assert row.user_id and (row.org_id, row.role, row.clearance, row.active) == (None, "viewer", 0, True)

    # An unknown role, then a duplicate firebase_uid.
    for sql in ("INSERT INTO users (firebase_uid, role) VALUES ('f2', 'owner')",
                "INSERT INTO users (firebase_uid) VALUES ('f1')"):
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(sa.text(sql))


def test_downgrade_restores_0002_and_keeps_chunks(pg_url, at):
    engine = sa.create_engine(pg_url)
    at(engine, "head")
    with engine.begin() as conn:
        _insert_chunk(conn)

    at(engine, "0002", downgrade=True)

    assert not NEW_TABLES & _tables(engine)
    assert _chunk_columns(engine).isdisjoint({"org_id", "team_id", "clearance"})
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT count(*) FROM chunks")).scalar() == 1
    at(engine, "head")  # and it upgrades again
    assert NEW_TABLES <= _tables(engine)
