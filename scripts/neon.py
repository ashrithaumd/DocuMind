"""Set up FinPilot on Neon, or run a backend command against it. Prints no URL.

    backend/.venv/Scripts/python scripts/neon.py setup
    backend/.venv/Scripts/python scripts/neon.py run -- python -m bank.corpus load

Reads backend/.env: NEON_MIGRATION_DATABASE_URL (the direct owner URL) and
NEON_POOLED_DATABASE_URL (the pooled host). `setup` first writes
NEON_APP_DATABASE_URL there if it is missing: finpilot_app on the pooled
host, with a fresh random password. Then it runs the migrate step, which
migrates the schema, creates finpilot_app and sets that password.

`run` maps the Neon URLs onto the names the code reads
(MIGRATION_DATABASE_URL = owner, DATABASE_URL = finpilot_app) and runs the
command in backend/. Neon's postgresql+psycopg:// scheme is handled by
config.sqlalchemy_url.
"""
import secrets
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
ENV = BACKEND / ".env"


def _values() -> dict:
    from dotenv import dotenv_values

    return dotenv_values(ENV)


def ensure_app_url() -> None:
    """Add NEON_APP_DATABASE_URL to backend/.env unless it is there."""
    from sqlalchemy import make_url

    values = _values()
    if values.get("NEON_APP_DATABASE_URL"):
        return
    app = make_url(values["NEON_POOLED_DATABASE_URL"]).set(
        username="finpilot_app", password=secrets.token_urlsafe(32))
    with ENV.open("a", encoding="utf-8") as f:
        f.write(f"\nNEON_APP_DATABASE_URL={app.render_as_string(hide_password=False)}\n")
    print("Added NEON_APP_DATABASE_URL to backend/.env (finpilot_app, new random password).")


def run(command: list[str]) -> int:
    import os

    values = _values()
    missing = [k for k in ("NEON_MIGRATION_DATABASE_URL", "NEON_APP_DATABASE_URL") if not values.get(k)]
    if missing:
        print(f"backend/.env lacks {', '.join(missing)} (run `neon.py setup` first).", file=sys.stderr)
        return 2
    env = dict(os.environ, MIGRATION_DATABASE_URL=values["NEON_MIGRATION_DATABASE_URL"],
               DATABASE_URL=values["NEON_APP_DATABASE_URL"])
    if command[0] == "python":  # this interpreter (the venv), found the same way on every OS
        command = [sys.executable, *command[1:]]
    return subprocess.call(command, env=env, cwd=BACKEND)


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["setup"]:
        ensure_app_url()
        return run(["python", "migrate.py"])
    if args[:1] == ["run"] and len(args) > 1:
        return run(args[2:] if args[1] == "--" else args[1:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
