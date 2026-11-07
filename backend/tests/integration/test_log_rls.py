"""The query and feedback logs (migration 0007): who asked is recorded, RLS
keeps each org's logs to that org, /admin/* is admin-only and shows only the
admin's own org, and LLM diagnostics go to admins only.

Users, all signed in explicitly (never the client fixture's default admin):
    viewer_a   org A, viewer
    admin_a    org A, admin
    admin_b    org B, admin
"""
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import ProgrammingError

from db.access import Access, set_access


def _as(uid):
    return {"Authorization": f"Bearer token:{uid}"}


@pytest.fixture
def owner(pg_url):
    engine = sa.create_engine(pg_url)
    yield engine
    engine.dispose()


@pytest.fixture
def users(client, owner):
    """Needs `client` first: its startup runs the migrations. Also gives each
    org one document, so queries go through the whole pipeline."""
    ids = {}
    with owner.begin() as conn:
        conn.execute(sa.text("INSERT INTO orgs VALUES ('A', 'Org A'), ('B', 'Org B')"))
        for uid, org, role in (("viewer_a", "A", "viewer"), ("admin_a", "A", "admin"),
                               ("admin_b", "B", "admin")):
            ids[uid] = conn.execute(sa.text(
                "INSERT INTO users (firebase_uid, org_id, role, clearance) "
                "VALUES (:uid, :org, :role, 0) RETURNING user_id"),
                {"uid": uid, "org": org, "role": role}).scalar()
    for uid in ("admin_a", "admin_b"):
        response = client.post("/upload", headers=_as(uid), data={"clearance": "0"},
                               files={"file": ("pumps.txt", b"Cooling pumps run at 400 RPM.",
                                               "text/plain")})
        assert response.status_code == 200, response.text
    return ids


def _ask(client, uid, question="What RPM do the cooling pumps run at?"):
    response = client.post("/query", headers=_as(uid), json={"question": question})
    assert response.status_code == 200, response.text
    return response.json()


def _rows(owner, table):
    with owner.connect() as conn:
        return [dict(r) for r in conn.execute(sa.text(
            f"SELECT id, user_id, org_id FROM {table} ORDER BY id")).mappings()]


# --- what gets recorded ---------------------------------------------------------

def test_query_and_feedback_logs_record_user_and_org(client, users, owner):
    query_id = _ask(client, "viewer_a")["query_id"]
    response = client.post("/feedback", headers=_as("viewer_a"),
                           json={"query_id": query_id, "thumbs_up": False, "comment": "meh"})
    assert response.status_code == 200

    assert _rows(owner, "query_logs") == [{"id": query_id, "user_id": users["viewer_a"], "org_id": "A"}]
    assert [(r["user_id"], r["org_id"]) for r in _rows(owner, "feedback")] == [(users["viewer_a"], "A")]


# --- /admin/* -------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/admin/logs", "/admin/stats"])
def test_admin_routes_refuse_a_viewer(client, users, path):
    assert client.get(path, headers=_as("viewer_a")).status_code == 403


def test_an_admin_sees_only_their_own_orgs_logs(client, users):
    a_ids = {_ask(client, "viewer_a")["query_id"], _ask(client, "admin_a")["query_id"]}
    b_id = _ask(client, "admin_b")["query_id"]

    logs_a = client.get("/admin/logs", headers=_as("admin_a")).json()["logs"]
    logs_b = client.get("/admin/logs", headers=_as("admin_b")).json()["logs"]

    assert {log["id"] for log in logs_a} == a_ids
    assert {log["org_id"] for log in logs_a} == {"A"}
    assert [log["id"] for log in logs_b] == [b_id]


def test_admin_stats_count_only_their_own_org(client, users):
    query_id = _ask(client, "viewer_a")["query_id"]
    _ask(client, "viewer_a")
    client.post("/feedback", headers=_as("viewer_a"), json={"query_id": query_id, "thumbs_up": False})

    stats_a = client.get("/admin/stats", headers=_as("admin_a")).json()
    stats_b = client.get("/admin/stats", headers=_as("admin_b")).json()

    assert stats_a["window"] == 2 and stats_a["feedback"]["total"] == 1
    assert stats_b["window"] == 0 and stats_b["feedback"]["total"] == 0
    assert stats_b["feedback"]["recent_negative"] == []  # A's questions don't leak via B's stats


def test_another_orgs_admin_cannot_give_or_overwrite_feedback(client, users, owner):
    query_id = _ask(client, "viewer_a")["query_id"]
    client.post("/feedback", headers=_as("viewer_a"), json={"query_id": query_id, "thumbs_up": True})

    response = client.post("/feedback", headers=_as("admin_b"),
                           json={"query_id": query_id, "thumbs_up": False, "comment": "sabotage"})

    assert response.status_code == 404  # org A's query doesn't exist for org B
    with owner.connect() as conn:
        assert conn.execute(sa.text("SELECT thumbs_up, comment FROM feedback")).one() == (True, None)


# --- the database itself, as documind_app ---------------------------------------

@pytest.fixture
def app_conn(users, app_url):
    engine = sa.create_engine(app_url)
    with engine.connect() as conn:
        yield conn
    engine.dispose()


def test_without_context_the_app_role_sees_no_logs(client, users, app_conn):
    _ask(client, "viewer_a")
    for table in ("query_logs", "feedback"):
        assert app_conn.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar() == 0
    forced = app_conn.execute(sa.text(
        "SELECT bool_and(relrowsecurity AND relforcerowsecurity) FROM pg_class "
        "WHERE relname IN ('query_logs', 'feedback')")).scalar()
    assert forced


def test_writing_a_log_row_into_another_org_is_refused(users, app_conn):
    set_access(app_conn, Access(org_id="A", clearance=0))
    with pytest.raises(ProgrammingError, match="row-level security"):
        app_conn.execute(sa.text(
            "INSERT INTO query_logs (question, outcome, regenerated, org_id) "
            "VALUES ('q', 'answered', false, 'B')"))
    app_conn.rollback()


@pytest.mark.parametrize("statement", [
    "ALTER TABLE query_logs DISABLE ROW LEVEL SECURITY",
    "DROP POLICY org_access ON feedback",
    "SET row_security = off; SELECT count(*) FROM query_logs",
])
def test_app_role_cannot_escape_the_log_policy(app_conn, statement):
    with pytest.raises(sa.exc.DBAPIError):
        app_conn.exec_driver_sql(statement)
    app_conn.rollback()


# --- LLM diagnostics: admins only -------------------------------------------------

def test_llm_diagnostics_go_to_admins_only(client, users):
    assert "llm_usage" not in _ask(client, "viewer_a")
    assert set(_ask(client, "admin_a")["llm_usage"]) == {
        "attempts_per_model", "failures", "calls", "tokens_per_role"}
