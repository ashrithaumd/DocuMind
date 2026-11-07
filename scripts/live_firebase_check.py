"""Live Firebase auth check: real sign-in, real token checks, real revocation.

Not part of the test suite (that uses a fake token checker and no network).
Run it after auth changes, after the frontend login lands, and against the
deployed service:

    backend/.venv/Scripts/python scripts/live_firebase_check.py
        Starts a throwaway pgvector container and a local backend on :8011,
        checks, then removes both.
    ... live_firebase_check.py --api https://... --database-url postgresql+psycopg2://...
        Checks a running backend; the database URL is where the test user
        gets seeded (the backend's own database, as the owner role: the
        app role can't write the user tables).

Refuses to run unless EVAL_ADMIN_EMAIL ends in @documind.test: step 5 revokes
that account's sessions, so it must never be a real person's.

Everything comes from the repo-root .env: EVAL_ADMIN_EMAIL,
EVAL_ADMIN_PASSWORD, VITE_FIREBASE_API_KEY, FIREBASE_PROJECT_ID and
GOOGLE_APPLICATION_CREDENTIALS (a key file outside the repo). The password
and tokens are never printed.

Steps:
  1. sign in through the Firebase REST API, and use the token immediately
     (a backend whose clock lags Google's must still accept it),
  2. GET /me -> set_up=false, unknown_user (when the user isn't seeded yet),
  3. seed the user as an admin in org 'eval',
  4. GET /me and GET /admin/stats -> 200,
  5. revoke the user's refresh tokens (Admin SDK), then the same token:
     /admin/stats -> 401 (revocation checked) and /documents -> 200 (not
     checked there, by design); a fresh sign-in used immediately -> 200.
Side effect: step 5 really revokes the test account's sessions.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[1]
LOCAL_API = "http://127.0.0.1:8011"
LOCAL_PG_PORT = 5440
LOCAL_DB = f"postgresql+psycopg2://documind:documind@localhost:{LOCAL_PG_PORT}/documind"
# The local backend connects as documind_app (RLS applies); this throwaway
# password is set on the role by the migration step at startup.
LOCAL_APP_DB = f"postgresql+psycopg2://documind_app:livecheck@localhost:{LOCAL_PG_PORT}/documind"
TEST_EMAIL_DOMAIN = "@documind.test"
CONTAINER = "documind-livecheck-fb"
LOG = Path(tempfile.gettempdir()) / "documind_live_backend.log"

ENV = dotenv_values(REPO / ".env")


def setting(name: str) -> str:
    return ENV.get(name) or os.environ.get(name) or ""


def step(msg):
    print(f"\n== {msg}", flush=True)


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def sign_in() -> tuple[str, str]:
    """(uid, id_token). On failure prints only Firebase's error code."""
    r = httpx.post(
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
        params={"key": setting("VITE_FIREBASE_API_KEY")},
        json={"email": setting("EVAL_ADMIN_EMAIL"), "password": setting("EVAL_ADMIN_PASSWORD"),
              "returnSecureToken": True},
        timeout=20,
    )
    if r.status_code != 200:
        code = r.json().get("error", {}).get("message", f"HTTP {r.status_code}")
        raise SystemExit(f"  FAIL sign-in refused by Firebase: {code}")
    body = r.json()
    return body["localId"], body["idToken"]


def call(api, path, token):
    r = httpx.get(api + path, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    try:
        body = r.json()
    except ValueError:
        body = r.text[:200]
    return r.status_code, body


def preflight():
    step("Preflight")
    for name in ("EVAL_ADMIN_EMAIL", "VITE_FIREBASE_API_KEY", "FIREBASE_PROJECT_ID"):
        check(bool(setting(name)), f"{name} set")
    # Step 5 revokes this account's sessions: only ever a test account.
    check(setting("EVAL_ADMIN_EMAIL").lower().endswith(TEST_EMAIL_DOMAIN),
          f"EVAL_ADMIN_EMAIL is a {TEST_EMAIL_DOMAIN} test account")
    check(bool(setting("EVAL_ADMIN_PASSWORD")), "EVAL_ADMIN_PASSWORD set (value not shown)")
    key = Path(setting("GOOGLE_APPLICATION_CREDENTIALS"))
    check(key.is_file(), f"key file exists: {key}")
    check(not key.resolve().is_relative_to(REPO), "key file is outside the repo")


def start_local_stack() -> subprocess.Popen:
    step("Start throwaway Postgres + backend on :8011")
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--rm", "--name", CONTAINER,
                    "-e", "POSTGRES_USER=documind", "-e", "POSTGRES_PASSWORD=documind",
                    "-e", "POSTGRES_DB=documind", "-p", f"{LOCAL_PG_PORT}:5432",
                    "pgvector/pgvector:0.8.7-pg16"], check=True, capture_output=True)
    for _ in range(30):
        if subprocess.run(["docker", "exec", CONTAINER, "pg_isready", "-U", "documind", "-q"]).returncode == 0:
            break
        time.sleep(1)
    time.sleep(2)  # the image restarts once after init
    env = {**os.environ,
           "DATABASE_URL": LOCAL_APP_DB,
           "FIREBASE_PROJECT_ID": setting("FIREBASE_PROJECT_ID"),
           "GOOGLE_APPLICATION_CREDENTIALS": setting("GOOGLE_APPLICATION_CREDENTIALS"),
           "EMBEDDING_PROVIDER": "gemini"}  # no torch; startup makes no embedding call
    env.pop("MIGRATION_DATABASE_URL", None)
    # The migrate step, as the owner; the backend itself never gets that URL.
    subprocess.run([sys.executable, "migrate.py"], cwd=REPO / "backend", check=True,
                   env={**env, "MIGRATION_DATABASE_URL": LOCAL_DB}, capture_output=True)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", "8011"],
                            cwd=REPO / "backend", env=env, stdout=LOG.open("w"),
                            stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            if httpx.get(LOCAL_API + "/health", timeout=2).status_code == 200:
                check(True, "backend healthy")
                return proc
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            break
        time.sleep(1)
    proc.kill()
    raise SystemExit(f"  FAIL backend didn't start; see {LOG}")


def seed_admin(database_url, uid):
    import sqlalchemy as sa

    engine = sa.create_engine(database_url)
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO orgs VALUES ('eval', 'Eval') ON CONFLICT DO NOTHING"))
        conn.execute(sa.text(
            "INSERT INTO users (firebase_uid, org_id, role, clearance) VALUES (:uid, 'eval', 'admin', 0) "
            "ON CONFLICT (firebase_uid) DO UPDATE SET org_id = 'eval', role = 'admin', active = true"
        ), {"uid": uid})
    engine.dispose()


def revoke(uid):
    import firebase_admin
    from firebase_admin import auth, credentials

    app = firebase_admin.initialize_app(
        credentials.Certificate(setting("GOOGLE_APPLICATION_CREDENTIALS")),
        {"projectId": setting("FIREBASE_PROJECT_ID")}, name="livecheck")
    auth.revoke_refresh_tokens(uid, app=app)


def run_checks(api, database_url):
    step("1. Sign in through the Firebase REST API and use the token immediately")
    uid, token = sign_in()
    status, body = call(api, "/me", token)
    check(True, f"signed in as {setting('EVAL_ADMIN_EMAIL')} (uid {uid}; token not shown)")

    step("2. GET /me before the user is seeded (same request as step 1)")
    print(f"  {status} {body}")
    check(status == 200, "fresh token accepted right after sign-in")
    if body.get("set_up"):
        print("  (user already set up in this database; skipping the unknown_user check)")
    else:
        check(body.get("reason") == "unknown_user", "not set up: unknown_user")

    step("3. Seed the user as an admin in org 'eval'")
    seed_admin(database_url, uid)
    check(True, "seeded")

    step("4. GET /me and GET /admin/stats as the seeded admin")
    status, body = call(api, "/me", token)
    print(f"  {status} {body}")
    check(status == 200 and body.get("set_up") is True and body.get("role") == "admin", "/me: set up, admin")
    status, body = call(api, "/admin/stats", token)
    print(f"  {status} {body}")
    check(status == 200, "/admin/stats: 200 (real revoked-token check passed)")

    step("5. Revoke the user's tokens in Firebase")
    revoke(uid)
    time.sleep(2)  # Firebase compares revocation in whole seconds
    status, body = call(api, "/admin/stats", token)
    print(f"  {status} {body}")
    check(status == 401, "/admin/stats with the revoked token: 401")
    status, _ = call(api, "/documents", token)
    check(status == 200, "/documents with the revoked token: 200 (no revocation check off admin routes, by design)")
    _, fresh = sign_in()
    status, body = call(api, "/admin/stats", fresh)  # immediately: no wait after sign-in
    print(f"  {status} {body if status != 200 else '(stats)'}")
    check(status == 200, "fresh sign-in used immediately -> /admin/stats: 200")

    print("\nALL CHECKS PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api", help="Backend to check (default: start a local one)")
    parser.add_argument("--database-url", help="Where to seed the test user (required with --api)")
    args = parser.parse_args()
    if args.api and not args.database_url:
        parser.error("--api needs --database-url")

    preflight()
    if args.api:
        run_checks(args.api.rstrip("/"), args.database_url)
        return
    proc = start_local_stack()
    try:
        run_checks(LOCAL_API, LOCAL_DB)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(["docker", "stop", CONTAINER], capture_output=True)


if __name__ == "__main__":
    main()
