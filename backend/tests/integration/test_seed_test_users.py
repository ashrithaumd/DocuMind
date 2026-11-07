"""scripts/seed_test_users.py on real Postgres, as the owner: every test user
lands with its org, role, clearance and teams, re-seeding changes nothing,
and each one signs in through the API with the fake token checker."""
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "kb" / "tools"))

pytestmark = pytest.mark.skipif(not (REPO / "kb" / "test-users.yaml").exists(),
                                reason="needs the Phase 3 corpus (kb/)")

import seed_test_users as s  # noqa: E402
from manifest import load_manifest  # noqa: E402


def _rows(engine):
    with engine.connect() as conn:
        users = conn.execute(sa.text(
            "SELECT firebase_uid, org_id, role, clearance, active FROM users "
            "WHERE firebase_uid LIKE 'test:%' ORDER BY 1")).all()
        teams = conn.execute(sa.text(
            "SELECT u.firebase_uid, m.team_id FROM team_memberships m JOIN users u USING (user_id) "
            "WHERE u.firebase_uid LIKE 'test:%' ORDER BY 1, 2")).all()
    return [tuple(r) for r in users], [tuple(r) for r in teams]


def test_seeding_is_complete_and_idempotent(client, pg_url):
    manifest = load_manifest()
    users = [u for u in s.load_users(manifest=manifest) if not u.email]
    engine = sa.create_engine(pg_url)

    first_ids = s.seed(pg_url, users, manifest)
    first = _rows(engine)
    second_ids = s.seed(pg_url, users, manifest)  # again: same rows, no duplicates

    assert first_ids == second_ids and _rows(engine) == first
    rows, teams = first
    assert len(rows) == len(users) == 29
    expected = sorted((u.fake_uid, u.org, u.role, u.clearance, True) for u in users)
    assert rows == expected
    assert teams == sorted((u.fake_uid, t) for u in users for t in u.teams)
    engine.dispose()


def test_seeded_users_sign_in_with_their_access(client, pg_url):
    manifest = load_manifest()
    users = [u for u in s.load_users(manifest=manifest) if not u.email]
    s.seed(pg_url, users, manifest)

    for user in users:
        me = client.get("/me", headers={"Authorization": f"Bearer token:{user.fake_uid}"}).json()
        assert me["set_up"] is True, user.id
        assert (me["org_id"], me["role"], me["clearance"], tuple(me["team_ids"])) == (
            user.org, user.role, user.clearance, tuple(sorted(user.teams))), user.id


def test_a_changed_entry_is_updated_and_memberships_replaced(client, pg_url):
    manifest = load_manifest()
    user = next(u for u in s.load_users(manifest=manifest) if u.id == "aurora-platform-c0")
    s.seed(pg_url, [user], manifest)
    moved = s.TestUser(id=user.id, org="aurora", role="editor", clearance=2, teams=("firmware",))

    s.seed(pg_url, [moved], manifest)

    me = client.get("/me", headers={"Authorization": f"Bearer token:{user.fake_uid}"}).json()
    assert (me["role"], me["clearance"], me["team_ids"]) == ("editor", 2, ["firmware"])
