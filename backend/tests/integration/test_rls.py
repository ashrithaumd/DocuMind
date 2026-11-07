"""Row-level security on chunks and documents (migration 0004), on real
Postgres, with the app connected as documind_app.

The corpus, seeded by the owner (5 documents, every chunk mentions pumps so
both retrieval legs match all of them):
    a_public   org A, org-wide,  clearance 0   visible to the viewer
    a_team1    org A, team t1,   clearance 0   visible (the viewer is in t1)
    a_team2    org A, team t2,   clearance 0   hidden: another team
    a_secret   org A, org-wide,  clearance 2   hidden: higher clearance
    b_public   org B, org-wide,  clearance 0   hidden: another org
The viewer: org A, team t1, clearance 1, role viewer, signed in explicitly
(not the client fixture's default admin).
"""
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, ProgrammingError

import deps
from ingestion.store import Access
from tests.helpers import FakeEmbeddings, _fake_vector

VIEWER_UID = "uid-viewer"
VIEWER = Access(org_id="A", clearance=1, team_ids=("t1",))
VISIBLE = {"a_public", "a_team1"}
HIDDEN = {"a_team2": "another team", "a_secret": "higher clearance", "b_public": "another org"}
CORPUS = {  # doc_id: (org, team, clearance)
    "a_public": ("A", None, 0),
    "a_team1": ("A", "t1", 0),
    "a_team2": ("A", "t2", 0),
    "a_secret": ("A", None, 2),
    "b_public": ("B", None, 0),
}


@pytest.fixture
def owner(pg_url):
    engine = sa.create_engine(pg_url)
    yield engine
    engine.dispose()


@pytest.fixture
def corpus(client, owner):
    """The five documents above, plus the viewer's user row. Needs `client`
    first: its startup runs the migrations."""
    vectors = [_fake_vector(f"{d} pump") for d in CORPUS]
    with owner.begin() as conn:
        conn.execute(sa.text("INSERT INTO orgs VALUES ('A', 'Org A'), ('B', 'Org B')"))
        conn.execute(sa.text("INSERT INTO teams VALUES ('t1', 'A', 'One'), ('t2', 'A', 'Two')"))
        for (doc, (org, team, clearance)), vector in zip(CORPUS.items(), vectors):
            conn.execute(sa.text("""
                INSERT INTO documents (doc_id, filename, org_id, team_id, clearance)
                VALUES (:doc, :doc || '.txt', :org, :team, :clearance)
            """), {"doc": doc, "org": org, "team": team, "clearance": clearance})
            conn.execute(sa.text("""
                INSERT INTO chunks (chunk_id, doc_id, filename, page_number, chunk_index, text,
                                    embedding_model, embedding, org_id, team_id, clearance)
                VALUES (:doc || '-0000', :doc, :doc || '.txt', 1, 0,
                        'The ' || :doc || ' pump runs at 400 RPM.',
                        'fake-embeddings', CAST(:v AS vector), :org, :team, :clearance)
            """), {"doc": doc, "org": org, "team": team, "clearance": clearance,
                   "v": "[" + ",".join(map(str, vector)) + "]"})
        user_id = conn.execute(sa.text(
            "INSERT INTO users (firebase_uid, org_id, role, clearance) "
            "VALUES (:uid, 'A', 'viewer', 1) RETURNING user_id"), {"uid": VIEWER_UID}).scalar()
        conn.execute(sa.text("INSERT INTO team_memberships VALUES (:u, 't1', 'A')"), {"u": user_id})


AS_VIEWER = {"Authorization": f"Bearer token:{VIEWER_UID}"}


def _docs(chunks) -> set[str]:
    return {c.chunk_id.rsplit("-", 1)[0] for c in chunks}


# --- the store, as the viewer -------------------------------------------------

async def test_both_retrieval_legs_return_only_visible_rows(corpus):
    store = deps.get_store().scoped(VIEWER)
    ranking = await store.fused_ranking("pump RPM", await store.embed_query("pump"))

    assert _docs(c for c in ranking if c.dense_rank is not None) == VISIBLE  # HNSW leg
    assert _docs(c for c in ranking if c.sparse_rank is not None) == VISIBLE  # full-text leg


@pytest.mark.parametrize("doc", HIDDEN)
async def test_asking_for_a_hidden_document_by_id_finds_nothing(corpus, doc):
    store = deps.get_store().scoped(VIEWER)
    assert await store.fused_ranking("pump", await store.embed_query("pump"), doc_ids=[doc]) == []


def test_listing_and_count_see_only_visible_rows(corpus):
    store = deps.get_store().scoped(VIEWER)
    assert {d["doc_id"] for d in store.list_documents()} == VISIBLE
    assert store.count() == len(VISIBLE)


async def test_no_access_context_sees_zero_rows(corpus):
    store = deps.get_store()  # unscoped: no settings at all
    assert store.count() == 0
    assert store.list_documents() == []
    assert await store.fused_ranking("pump", await store.embed_query("pump")) == []


# --- the API, signed in as the viewer -----------------------------------------

def test_documents_endpoint_lists_only_visible_documents(client, corpus):
    response = client.get("/documents", headers=AS_VIEWER)
    assert response.status_code == 200
    assert {d["doc_id"] for d in response.json()["documents"]} == VISIBLE


@pytest.mark.parametrize("doc", HIDDEN)
def test_query_retrieves_nothing_from_a_hidden_document(client, corpus, doc):
    body = client.post("/query", headers=AS_VIEWER,
                       json={"question": "What RPM does the pump run at?", "doc_ids": [doc]}).json()
    assert body["chunks"] == [], HIDDEN[doc]


def test_query_across_all_visible_documents(client, corpus):
    body = client.post("/query", headers=AS_VIEWER,
                       json={"question": "pump RPM", "doc_ids": list(CORPUS)}).json()
    assert {c["chunk_id"].rsplit("-", 1)[0] for c in body["chunks"]} == VISIBLE


def test_viewers_cannot_upload(client, corpus):
    response = client.post("/upload", headers=AS_VIEWER, data={"clearance": "0"},
                           files={"file": ("new.txt", b"Fresh pump notes.", "text/plain")})
    assert response.status_code == 403


# --- documind_app can't get around the policy ---------------------------------

@pytest.fixture
def app_conn(corpus, app_url):
    engine = sa.create_engine(app_url)
    with engine.connect() as conn:
        yield conn
    engine.dispose()


def test_app_role_is_not_superuser_and_cannot_bypass_rls(app_conn):
    row = app_conn.execute(sa.text(
        "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")).one()
    assert tuple(row) == (False, False)
    app_conn.rollback()
    for table in ("chunks", "documents"):
        forced = app_conn.execute(sa.text(
            "SELECT relrowsecurity AND relforcerowsecurity FROM pg_class WHERE oid = CAST(:t AS regclass)"),
            {"t": table}).scalar()
        assert forced, table


@pytest.mark.parametrize("statement", [
    "SET row_security = off; SELECT count(*) FROM chunks",  # errors instead of bypassing
    "ALTER TABLE chunks DISABLE ROW LEVEL SECURITY",
    "ALTER TABLE chunks NO FORCE ROW LEVEL SECURITY",
    "DROP POLICY tenant_access ON chunks",
    "ALTER TABLE documents DISABLE ROW LEVEL SECURITY",
    "SET ROLE {owner}",  # the migration owner
    "UPDATE users SET role = 'admin', clearance = 99",  # can't promote itself
    "INSERT INTO team_memberships SELECT user_id, 't2', 'A' FROM users",
])
def test_app_role_cannot_escape_the_policy(app_conn, pg_url, statement):
    owner = sa.engine.make_url(pg_url).username
    with pytest.raises((ProgrammingError, DBAPIError)):
        app_conn.exec_driver_sql(statement.format(owner=owner))
    app_conn.rollback()


def test_owner_tables_are_unreadable_without_context_even_by_name(app_conn):
    # No set_config in this transaction: zero rows, not an error.
    assert app_conn.execute(sa.text("SELECT count(*) FROM chunks")).scalar() == 0
    assert app_conn.execute(sa.text("SELECT count(*) FROM documents")).scalar() == 0


@pytest.mark.parametrize("org,team,clearance,why", [
    ("B", None, 0, "another org"),
    ("A", "t2", 0, "a team the viewer isn't in"),
    ("A", None, 2, "a clearance above the viewer's"),
])
def test_writes_outside_the_viewers_access_are_refused(app_conn, org, team, clearance, why):
    from ingestion.store import set_access

    set_access(app_conn, VIEWER)
    with pytest.raises(ProgrammingError, match="row-level security"):
        app_conn.execute(sa.text("""
            INSERT INTO chunks (chunk_id, doc_id, filename, page_number, chunk_index, text,
                                embedding_model, embedding, org_id, team_id, clearance)
            SELECT 'x-0000', 'a_public', 'x.txt', 1, 0, 'x', 'fake-embeddings', embedding,
                   :org, :team, :clearance FROM chunks LIMIT 1
        """), {"org": org, "team": team, "clearance": clearance})
    app_conn.rollback()

    set_access(app_conn, VIEWER)  # moving a visible row out of reach is refused too
    with pytest.raises(ProgrammingError, match="row-level security"):
        app_conn.execute(sa.text(
            "UPDATE chunks SET org_id = :org, team_id = :team, clearance = :clearance "
            "WHERE doc_id = 'a_public'"), {"org": org, "team": team, "clearance": clearance})
    app_conn.rollback()


# --- the connection pool ------------------------------------------------------

def test_access_context_does_not_leak_through_the_pool(corpus, app_url):
    from config import get_settings
    from ingestion.store import HybridStore

    engine = sa.create_engine(app_url, pool_size=1, max_overflow=0)
    store = HybridStore(get_settings(), deps.get_embeddings(), engine)
    with engine.connect() as conn:
        pid = conn.execute(sa.text("SELECT pg_backend_pid()")).scalar()

    assert store.scoped(VIEWER).count() == len(VISIBLE)
    # Even a transaction that fails part-way must not leave its context behind.
    with pytest.raises(DBAPIError):
        with store.scoped(VIEWER)._begin() as conn:
            conn.execute(sa.text("SELECT 1 / 0"))

    with engine.connect() as conn:  # the same pooled connection, no context set
        assert conn.execute(sa.text("SELECT pg_backend_pid()")).scalar() == pid
        settings = conn.execute(sa.text(
            "SELECT current_setting('app.org_id', true), current_setting('app.clearance', true), "
            "current_setting('app.team_ids', true)")).one()
        assert all(not value for value in settings)  # '' or NULL: reset at transaction end
        assert conn.execute(sa.text("SELECT count(*) FROM chunks")).scalar() == 0
    assert store.count() == 0
    engine.dispose()


def test_two_callers_on_one_connection_each_see_their_own_rows(corpus, app_url, owner):
    from config import get_settings
    from ingestion.store import HybridStore

    engine = sa.create_engine(app_url, pool_size=1, max_overflow=0)
    store = HybridStore(get_settings(), deps.get_embeddings(), engine)
    org_b = Access(org_id="B", clearance=0)

    for _ in range(2):  # interleaved on the single pooled connection
        assert {d["doc_id"] for d in store.scoped(VIEWER).list_documents()} == VISIBLE
        assert {d["doc_id"] for d in store.scoped(org_b).list_documents()} == {"b_public"}
    engine.dispose()


# --- the app role's password ---------------------------------------------------

def test_app_role_password_with_quotes_and_percent_signs_is_set_safely(pg_url, app_url):
    from config import Settings
    from db import session as db_session

    tricky = "it's 100%: a:b@c"
    url = sa.engine.make_url(app_url).set(password=tricky).render_as_string(hide_password=False)
    try:
        db_session.migrate_database(Settings(_env_file=None, database_url=url,
                                             migration_database_url=pg_url, embedding_dimensions=32))
        engine = sa.create_engine(url)
        with engine.connect() as conn:
            assert conn.execute(sa.text("SELECT current_user")).scalar() == "documind_app"
        engine.dispose()
    finally:  # restore the password the rest of the suite uses
        db_session.migrate_database(Settings(_env_file=None, database_url=app_url,
                                             migration_database_url=pg_url, embedding_dimensions=32))


def test_bench_runs_under_rls_as_its_own_org(pg_url, app_url, monkeypatch):
    import sys
    from pathlib import Path

    from config import Settings
    from db import session as db_session

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
    import bench_retrieval

    monkeypatch.setattr("embeddings.build_embeddings", lambda settings: FakeEmbeddings())
    store = bench_retrieval.build_store(Settings(_env_file=None, database_url=app_url,
                                                 migration_database_url=pg_url,
                                                 embedding_dimensions=32))
    assert store._access.org_id == bench_retrieval.BENCH_ORG
    assert store.count() == 0
