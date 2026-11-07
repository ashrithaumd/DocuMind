"""Owner credentials stay out of the serving process: migrations run as their
own one-shot step (migrate.py), and the backend refuses to start if
MIGRATION_DATABASE_URL is in its environment."""
import subprocess
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa

import config

BACKEND = Path(__file__).resolve().parents[2]


def test_backend_refuses_to_start_with_the_owner_url(app_env, pg_url, monkeypatch):
    from fastapi.testclient import TestClient

    from main import app

    monkeypatch.setenv("MIGRATION_DATABASE_URL", pg_url)
    config.get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="MIGRATION_DATABASE_URL is set"):
        with TestClient(app):
            pass


def _run_migrate(env_overrides):
    import os

    env = {**os.environ, "PYTHONPATH": str(BACKEND), "GEMINI_API_KEY": "test-key-not-real",
           "EMBEDDING_PROVIDER": "local", "EMBEDDING_MODEL": "", "EMBEDDING_DIMENSIONS": "32",
           **env_overrides}
    return subprocess.run([sys.executable, "migrate.py"], cwd=BACKEND, env=env,
                          capture_output=True, text=True, timeout=120)


def test_migrate_step_migrates_and_lets_the_app_role_log_in(pg_url, app_url):
    result = _run_migrate({"MIGRATION_DATABASE_URL": pg_url, "DATABASE_URL": app_url})
    assert result.returncode == 0, result.stderr

    engine = sa.create_engine(app_url)  # the password it set works
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT current_user")).scalar() == "documind_app"
        assert conn.execute(sa.text("SELECT to_regclass('chunks') IS NOT NULL")).scalar()
    engine.dispose()


def test_migrate_step_needs_both_urls(app_url):
    result = _run_migrate({"MIGRATION_DATABASE_URL": "", "DATABASE_URL": app_url})
    assert result.returncode == 2
    assert "MIGRATION_DATABASE_URL" in result.stderr


def test_backend_checks_firebase_config_at_startup(client, token_checker):
    assert token_checker.ready_checks == 1


def test_backend_refuses_to_start_without_firebase_config(app_env, token_checker, monkeypatch):
    from fastapi.testclient import TestClient

    from main import app

    def not_ready():
        raise RuntimeError("Firebase credentials not found: set GOOGLE_APPLICATION_CREDENTIALS")

    monkeypatch.setattr(token_checker, "check_ready", not_ready)
    with pytest.raises(RuntimeError, match="GOOGLE_APPLICATION_CREDENTIALS"):
        with TestClient(app):
            pass
