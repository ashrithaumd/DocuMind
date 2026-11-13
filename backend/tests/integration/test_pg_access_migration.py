"""Migrations 0003-0006 on real Postgres: access tables and chunk columns
appear on upgrade, old chunks end up in no org, org/team mismatches are
impossible, clearance has no default, a chunk needs its documents row, and
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
NOT_NULL_VIOLATION, FOREIGN_KEY_VIOLATION = "23502", "23503"

CHUNK_COLUMNS = ("chunk_id, doc_id, filename, page_number, chunk_index, text, "
                 "embedding_model, embedding")


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


@pytest.fixture
def head(pg_url, at):
    """An engine (as the owner) on a database at head, with orgs a and b and
    org a's team ta."""
    engine = sa.create_engine(pg_url)
    at(engine, "head")
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO orgs VALUES ('a', 'A'), ('b', 'B')"))
        conn.execute(sa.text("INSERT INTO teams VALUES ('ta', 'a', 'Team A')"))
    yield engine
    engine.dispose()


def _tables(engine):
    return set(sa.inspect(engine).get_table_names())


def _chunk_columns(engine):
    return {c["name"] for c in sa.inspect(engine).get_columns("chunks")}


def _refused(engine, sql, params=None) -> IntegrityError:
    with pytest.raises(IntegrityError) as err, engine.begin() as conn:
        conn.execute(sa.text(sql), {"v": VEC, **(params or {})})
    return err.value


def test_upgrade_adds_tables_and_leaves_old_chunks_in_no_org(pg_url, at):
    engine = sa.create_engine(pg_url)
    at(engine, "0002")
    with engine.begin() as conn:
        conn.execute(sa.text(f"INSERT INTO chunks ({CHUNK_COLUMNS}) "
                             "VALUES ('d-0', 'd', 'f.pdf', 1, 0, 'x', 'm', :v)"), {"v": VEC})
    assert not NEW_TABLES & _tables(engine)

    at(engine, "head")

    assert NEW_TABLES <= _tables(engine)
    with engine.connect() as conn:
        row = conn.execute(sa.text("SELECT org_id, team_id, clearance FROM chunks")).one()
    # No org: invisible under RLS. (Its 0 came from 0003's default, since dropped.)
    assert tuple(row) == (None, None, 0)


def test_a_team_cannot_cross_orgs(head):
    with head.begin() as conn:
        conn.execute(sa.text("INSERT INTO users (user_id, firebase_uid, org_id) VALUES ('ub', 'f1', 'b')"))
        conn.execute(sa.text("INSERT INTO documents (doc_id, filename, org_id, clearance) "
                             "VALUES ('d', 'f', 'a', 0)"))

    # A document of org b on org a's team; a member of org b on org a's team
    # (under either org id); a chunk of org b on org a's team. Each one is
    # refused by the (team_id, org_id) foreign key, nothing else.
    for sql in (
        "INSERT INTO documents (doc_id, filename, org_id, team_id, clearance) VALUES ('e', 'f', 'b', 'ta', 0)",
        "INSERT INTO team_memberships VALUES ('ub', 'ta', 'b')",
        "INSERT INTO team_memberships VALUES ('ub', 'ta', 'a')",
        f"INSERT INTO chunks ({CHUNK_COLUMNS}, org_id, team_id, clearance) "
        "VALUES ('c', 'd', 'f', 1, 0, 'x', 'm', :v, 'b', 'ta', 0)",
    ):
        error = _refused(head, sql)
        assert error.orig.pgcode == FOREIGN_KEY_VIOLATION
        assert "team_id" in str(error.orig) or "user_id" in str(error.orig), sql


def test_user_defaults_and_role_check(head):
    with head.begin() as conn:
        conn.execute(sa.text("INSERT INTO users (firebase_uid) VALUES ('f1')"))
        row = conn.execute(sa.text("SELECT user_id, org_id, role, clearance, active FROM users")).one()
    assert row.user_id and (row.org_id, row.role, row.clearance, row.active) == (None, "viewer", 0, True)

    # An unknown role, then a duplicate firebase_uid.
    for sql in ("INSERT INTO users (firebase_uid, role) VALUES ('f2', 'owner')",
                "INSERT INTO users (firebase_uid) VALUES ('f1')"):
        _refused(head, sql)


# --- 0006: the documents table is the write path ------------------------------

def test_a_chunk_without_its_documents_row_is_refused(head):
    error = _refused(head, f"INSERT INTO chunks ({CHUNK_COLUMNS}, org_id, clearance) "
                           "VALUES ('x-0', 'no-such-doc', 'f', 1, 0, 'x', 'm', :v, 'a', 0)")
    assert error.orig.pgcode == FOREIGN_KEY_VIOLATION
    assert "fk_chunks_document" in str(error.orig)


def test_clearance_has_no_default_on_documents_or_chunks(head):
    with head.begin() as conn:
        conn.execute(sa.text("INSERT INTO documents (doc_id, filename, org_id, clearance) "
                             "VALUES ('d', 'f', 'a', 0)"))

    for sql in (
        "INSERT INTO documents (doc_id, filename, org_id) VALUES ('e', 'f', 'a')",
        f"INSERT INTO chunks ({CHUNK_COLUMNS}, org_id) VALUES ('d-0', 'd', 'f', 1, 0, 'x', 'm', :v, 'a')",
    ):
        error = _refused(head, sql)
        assert error.orig.pgcode == NOT_NULL_VIOLATION
        assert '"clearance"' in str(error.orig), sql


def test_downgrade_restores_0002_and_keeps_chunks(head, at):
    with head.begin() as conn:
        conn.execute(sa.text("INSERT INTO documents (doc_id, filename, org_id, clearance) "
                             "VALUES ('d', 'f.pdf', 'a', 0)"))
        conn.execute(sa.text(f"INSERT INTO chunks ({CHUNK_COLUMNS}, org_id, clearance) "
                             "VALUES ('d-0', 'd', 'f.pdf', 1, 0, 'x', 'm', :v, 'a', 0)"), {"v": VEC})

    at(head, "0002", downgrade=True)

    assert not NEW_TABLES & _tables(head)
    assert _chunk_columns(head).isdisjoint({"org_id", "team_id", "clearance"})
    with head.connect() as conn:
        assert conn.execute(sa.text("SELECT count(*) FROM chunks")).scalar() == 1
    at(head, "head")  # and it upgrades again
    assert NEW_TABLES <= _tables(head)


def _can_read_chunks(engine, role) -> bool:
    with engine.connect() as conn:
        return conn.execute(sa.text("SELECT has_table_privilege(:r, 'chunks', 'SELECT')"),
                            {"r": role}).scalar()


def test_0008_moves_the_grants_to_finpilot_app_and_back(pg_url, at):
    engine = sa.create_engine(pg_url)
    at(engine, "head")
    assert _can_read_chunks(engine, "finpilot_app")
    assert not _can_read_chunks(engine, "documind_app")

    at(engine, "0007", downgrade=True)
    assert _can_read_chunks(engine, "documind_app")
    assert not _can_read_chunks(engine, "finpilot_app")
    engine.dispose()
