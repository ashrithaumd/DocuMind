"""scripts/setup_eval_admin.py on real Postgres, as the owner: creates the
eval org and an active admin, is idempotent, and repairs an existing row
(promoted, moved into the eval org, reactivated). The Firebase lookup is not
exercised here (scripts/live_firebase_check.py covers it live)."""
import subprocess
import sys
from pathlib import Path

import sqlalchemy as sa

from config import Settings
from db import session as db_session

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from setup_eval_admin import ensure_eval_admin  # noqa: E402


def _users(engine):
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(sa.text(
            "SELECT firebase_uid, org_id, role, clearance, active FROM users"))]


def test_creates_the_eval_admin_and_is_idempotent(pg_url):
    db_session.migrate_database(Settings(_env_file=None, database_url=pg_url))
    engine = sa.create_engine(pg_url)

    first = ensure_eval_admin(pg_url, "uid-eval", clearance=1)
    second = ensure_eval_admin(pg_url, "uid-eval", clearance=1)

    assert first == second
    assert _users(engine) == [("uid-eval", "eval", "admin", 1, True)]
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT count(*) FROM orgs WHERE org_id = 'eval'")).scalar() == 1
    engine.dispose()


def test_repairs_an_existing_row(pg_url):
    db_session.migrate_database(Settings(_env_file=None, database_url=pg_url))
    engine = sa.create_engine(pg_url)
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO orgs VALUES ('other', 'Other')"))
        conn.execute(sa.text("INSERT INTO users (firebase_uid, org_id, role, active) "
                             "VALUES ('uid-eval', 'other', 'viewer', false)"))

    ensure_eval_admin(pg_url, "uid-eval")

    assert _users(engine) == [("uid-eval", "eval", "admin", 0, True)]
    engine.dispose()


def test_refuses_a_non_test_account(tmp_path):
    env = {"PATH": "", "EVAL_ADMIN_EMAIL": "someone@gmail.com", "SYSTEMROOT": ""}
    import os

    result = subprocess.run([sys.executable, str(SCRIPTS / "setup_eval_admin.py"),
                             "--database-url", "postgresql://unused"],
                            env={**os.environ, **env}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 2
    assert "@documind.test" in result.stderr
