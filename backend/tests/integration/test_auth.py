"""Deny by default: every endpoint except /me and /health needs a valid token
(401 otherwise) whose user has a row, an org, and active=true (403
otherwise). The user row is read on every request, so a change applies on the
very next one. Tokens come from FakeTokenChecker: no Firebase, no network."""
import pytest
import sqlalchemy as sa

from auth import get_admin_principal, get_principal
from tests.conftest import TEST_ORG, TEST_UID

# One request per protected endpoint; the body only has to get past auth.
PROTECTED = [
    ("get", "/documents", {}),
    ("post", "/query", {"json": {"question": "hi"}}),
    ("post", "/query/stream", {"json": {"question": "hi"}}),
    ("post", "/upload", {"files": {"file": ("a.txt", b"hello", "text/plain")}}),
    ("post", "/feedback", {"json": {"query_id": 1, "thumbs_up": True}}),
    ("get", "/admin/logs", {}),
    ("get", "/admin/stats", {}),
]


@pytest.fixture
def sql(pg_url):
    engine = sa.create_engine(pg_url)

    def run(statement, **params):
        with engine.begin() as conn:
            conn.execute(sa.text(statement), params)

    yield run
    engine.dispose()


def _as(uid):
    return {"Authorization": f"Bearer token:{uid}"}


def _call(client, method, path, kwargs, headers):
    return getattr(client, method)(path, headers=headers, **kwargs)


def test_every_route_but_me_and_health_requires_a_principal():
    from main import app

    open_routes = {"/me", "/health", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    for route in app.routes:
        if route.path in open_routes:
            continue
        calls = {d.call for d in route.dependant.dependencies}
        # Admin routes also check revocation with Firebase; the rest don't.
        expected = get_admin_principal if route.path.startswith("/admin/") else get_principal
        assert calls & {get_principal, get_admin_principal} == {expected}, route.path
    assert {p for _, p, _ in PROTECTED} == {r.path for r in app.routes} - open_routes


@pytest.mark.parametrize("method,path,kwargs", PROTECTED)
def test_no_or_bad_token_is_401(client, method, path, kwargs):
    for headers in ({"Authorization": ""}, {"Authorization": "Bearer not-a-real-token"}):
        response = _call(client, method, path, kwargs, headers)
        assert response.status_code == 401, (headers, response.text)
        assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("method,path,kwargs", PROTECTED)
def test_unknown_user_no_org_and_inactive_are_403(client, sql, method, path, kwargs):
    sql("INSERT INTO users (firebase_uid) VALUES ('no-org')")
    sql("INSERT INTO users (firebase_uid, org_id, active) VALUES ('off', :org, false)", org=TEST_ORG)

    for uid, reason in (("stranger", "unknown_user"), ("no-org", "no_org"), ("off", "inactive")):
        response = _call(client, method, path, kwargs, _as(uid))
        assert response.status_code == 403, (uid, response.text)
        assert reason in response.json()["detail"]


def test_set_up_user_gets_through(client):
    assert client.get("/documents").status_code == 200


def test_deactivation_applies_on_the_very_next_request(client, sql):
    assert client.get("/documents").status_code == 200
    sql("UPDATE users SET active = false WHERE firebase_uid = :uid", uid=TEST_UID)
    assert client.get("/documents").status_code == 403
    sql("UPDATE users SET active = true WHERE firebase_uid = :uid", uid=TEST_UID)
    assert client.get("/documents").status_code == 200


def test_me_answers_any_valid_token(client, sql):
    sql("INSERT INTO users (firebase_uid) VALUES ('no-org')")
    sql("INSERT INTO users (firebase_uid, org_id, active) VALUES ('off', :org, false)", org=TEST_ORG)

    for uid, reason in (("stranger", "unknown_user"), ("no-org", "no_org"), ("off", "inactive")):
        response = client.get("/me", headers=_as(uid))
        assert response.status_code == 200
        assert response.json() == {"firebase_uid": uid, "set_up": False, "reason": reason}


def test_me_for_a_set_up_user_lists_org_role_clearance_and_teams(client, sql):
    sql("INSERT INTO teams VALUES ('t2', :org, 'Two'), ('t1', :org, 'One')", org=TEST_ORG)
    sql("INSERT INTO team_memberships SELECT user_id, t, :org FROM users, "
        "unnest(ARRAY['t2', 't1']) t WHERE firebase_uid = :uid", org=TEST_ORG, uid=TEST_UID)

    body = client.get("/me").json()

    assert body.pop("user_id")
    assert body == {"firebase_uid": TEST_UID, "set_up": True, "reason": None, "org_id": TEST_ORG,
                    "role": "admin", "clearance": 0, "team_ids": ["t1", "t2"]}


def test_me_still_needs_a_valid_token(client):
    assert client.get("/me", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_health_stays_open(client):
    assert client.get("/health", headers={"Authorization": ""}).status_code == 200


# --- revoked tokens (admin routes) -------------------------------------------

ADMIN_ROUTES = ["/admin/logs", "/admin/stats"]


@pytest.mark.parametrize("path", ADMIN_ROUTES)
def test_revoked_token_is_refused_on_the_next_admin_request(client, token_checker, path):
    token = f"token:{TEST_UID}"
    assert client.get(path).status_code == 200

    token_checker.revoke(token)

    response = client.get(path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_revoked_token_still_works_off_admin_routes_until_it_expires(client, token_checker):
    # Firebase's revocation check is a network call, so only admin routes make
    # it. Elsewhere the token works until it expires (at most an hour), and
    # users.active=false is the immediate off switch.
    token_checker.revoke(f"token:{TEST_UID}")
    assert client.get("/documents").status_code == 200
    assert client.get("/admin/stats").status_code == 401


def test_only_admin_routes_ask_firebase_about_revocation(client, token_checker):
    client.get("/documents")
    client.get("/me")
    assert token_checker.revocation_checks == 0
    client.get("/admin/logs")
    assert token_checker.revocation_checks == 1


@pytest.mark.parametrize("path", ADMIN_ROUTES)
def test_deactivated_user_is_refused_on_the_next_admin_request(client, sql, token_checker, path):
    assert client.get(path).status_code == 200

    sql("UPDATE users SET active = false WHERE firebase_uid = :uid", uid=TEST_UID)

    response = client.get(path)
    assert response.status_code == 403
    assert "inactive" in response.json()["detail"]
    assert not token_checker.revoked  # refused on the flag alone, token still valid
