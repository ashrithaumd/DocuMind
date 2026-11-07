"""kb/test-users.yaml and scripts/seed_test_users.py without a database: the
file is valid against the manifest, it covers every access boundary the
Phase 3 tests need, and bad entries are rejected."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "kb" / "tools"))

pytestmark = pytest.mark.skipif(not (REPO / "kb" / "test-users.yaml").exists(),
                                reason="needs the Phase 3 corpus (kb/)")

import seed_test_users as s  # noqa: E402
from manifest import load_manifest  # noqa: E402


@pytest.fixture(scope="module")
def users():
    return s.load_users()


def test_the_real_file_is_valid(users):
    assert len(users) == 32
    assert sum(1 for u in users if u.email) == 3


def test_it_covers_every_access_boundary(users):
    manifest = load_manifest()
    test_users = [u for u in users if not u.email]
    for org in manifest.orgs:
        mine = [u for u in test_users if u.org == org]
        # an org-wide viewer at every clearance
        assert {u.clearance for u in mine if u.role == "viewer" and not u.teams} == {0, 1, 2, 3}, org
        # every team at the lowest and the highest clearance
        for team in manifest.teams(org):
            levels = {u.clearance for u in mine if u.teams == (team,)}
            assert {0, 3} <= levels, (org, team)
        assert any(u.role == "editor" for u in mine) and any(u.role == "admin" for u in mine)
    by_id = {u.id: u for u in users}
    assert (by_id["bluefin-finance-c2"].teams, by_id["bluefin-finance-c2"].clearance) == (("finance",), 2)
    assert (by_id["demo-bluefin-ml"].teams, by_id["demo-bluefin-ml"].clearance) == (("ml",), 3)


def test_demo_users_have_test_emails_and_env_password_names(users):
    demo = {u.email: u.password_var for u in users if u.email}
    assert demo == {
        "aurora-viewer@documind.test": "DEMO_PASSWORD_AURORA_VIEWER",
        "bluefin-finance@documind.test": "DEMO_PASSWORD_BLUEFIN_FINANCE",
        "bluefin-ml@documind.test": "DEMO_PASSWORD_BLUEFIN_ML",
    }


def test_fake_uids_match_the_fake_token_checker(users):
    user = next(u for u in users if u.id == "aurora-viewer-c0")
    assert user.fake_uid == "test:aurora-viewer-c0"  # signs in with "token:test:aurora-viewer-c0"


@pytest.mark.parametrize("entry,problem", [
    ("{id: x, org: aurora, role: viewer, clearance: 0, teams: [finance]}", "not teams of aurora"),
    ("{id: x, org: nowhere, role: viewer, clearance: 0, teams: []}", "unknown org"),
    ("{id: x, org: aurora, role: owner, clearance: 0, teams: []}", "role must be"),
    ("{id: x, org: aurora, role: viewer, clearance: 4, teams: []}", "clearance must be"),
    ("{id: x, org: aurora, role: viewer, clearance: 0, teams: [], email: a@gmail.com}", "documind.test"),
    ("{id: x, org: aurora, role: viewer, clearance: 0}", "needs id, org, role"),
])
def test_bad_entries_are_rejected(tmp_path, entry, problem):
    path = tmp_path / "users.yaml"
    path.write_text(f"users:\n  - {entry}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=problem):
        s.load_users(path)


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "users.yaml"
    line = "  - {id: x, org: aurora, role: viewer, clearance: 0, teams: []}\n"
    path.write_text("users:\n" + line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        s.load_users(path)
