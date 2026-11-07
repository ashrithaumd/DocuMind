"""scripts/phase3_db.py without Docker: passwords are generated into .env
once and never changed or shown, and the URLs quote any password safely."""
import sys
from pathlib import Path

import sqlalchemy as sa

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import phase3_db  # noqa: E402


def test_init_adds_missing_passwords_once_and_keeps_the_rest(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=keep-me\r\nPOSTGRES_PASSWORD=also-kept\r\n", encoding="utf-8")

    assert phase3_db.ensure_secrets(env) == ["PHASE3_POSTGRES_PASSWORD", "PHASE3_APP_DB_PASSWORD"]
    first = phase3_db.read_env(env)
    assert phase3_db.ensure_secrets(env) == []  # second run: nothing added, nothing changed
    second = phase3_db.read_env(env)

    assert first == second
    assert first["GEMINI_API_KEY"] == "keep-me" and first["POSTGRES_PASSWORD"] == "also-kept"
    assert len(first["PHASE3_POSTGRES_PASSWORD"]) >= 40
    assert first["PHASE3_POSTGRES_PASSWORD"] != first["PHASE3_APP_DB_PASSWORD"]
    raw = env.read_bytes().decode("utf-8")
    assert raw.count("\r\n") == raw.count("\n")  # every line still CRLF, none mixed


def test_init_fills_only_what_is_missing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("PHASE3_POSTGRES_PASSWORD=existing", encoding="utf-8")  # no final newline
    assert phase3_db.ensure_secrets(env) == ["PHASE3_APP_DB_PASSWORD"]
    values = phase3_db.read_env(env)
    assert values["PHASE3_POSTGRES_PASSWORD"] == "existing" and values["PHASE3_APP_DB_PASSWORD"]


def test_urls_quote_awkward_passwords_and_use_the_port():
    env = {"PHASE3_POSTGRES_PASSWORD": "p@ss:w/rd%", "PHASE3_APP_DB_PASSWORD": "app pw#1",
           "PHASE3_DB_PORT": "5999"}
    built = phase3_db.urls(env)
    owner = sa.engine.make_url(built["MIGRATION_DATABASE_URL"])
    app = sa.engine.make_url(built["DATABASE_URL"])
    assert (owner.username, owner.password, owner.port) == ("documind", "p@ss:w/rd%", 5999)
    assert (app.username, app.password, app.host) == ("documind_app", "app pw#1", "localhost")
