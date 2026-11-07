"""Create the eval org and make EVAL_ADMIN_EMAIL an admin in it. Idempotent.

Runs as the owner role, like the migrate step: the app role (documind_app)
can only read the user tables, so it can't create an admin. Also how the
first admin of a fresh database is created.

    backend/.venv/Scripts/python scripts/setup_eval_admin.py --database-url OWNER_URL [--clearance 0]

OWNER_URL defaults to MIGRATION_DATABASE_URL from the environment (never
from .env: the backend refuses to start with it there). The Firebase uid of
EVAL_ADMIN_EMAIL is looked up with the Admin SDK (FIREBASE_PROJECT_ID and
GOOGLE_APPLICATION_CREDENTIALS from the repo-root .env). Only
@documind.test accounts are accepted: this is the eval account, never a
real person's. Prints no secret.
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL_ORG = "eval"
TEST_EMAIL_DOMAIN = "@documind.test"


def ensure_eval_admin(owner_url: str, firebase_uid: str, clearance: int = 0) -> str:
    """Upsert org 'eval' and an active admin for `firebase_uid` in it, at
    `clearance`. Returns the user's id. Running it again changes nothing."""
    import sqlalchemy as sa

    engine = sa.create_engine(owner_url)
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("INSERT INTO orgs VALUES (:org, 'Eval') ON CONFLICT DO NOTHING"),
                         {"org": EVAL_ORG})
            return conn.execute(sa.text("""
                INSERT INTO users (firebase_uid, org_id, role, clearance, active)
                VALUES (:uid, :org, 'admin', :clearance, true)
                ON CONFLICT (firebase_uid) DO UPDATE
                    SET org_id = EXCLUDED.org_id, role = 'admin',
                        clearance = EXCLUDED.clearance, active = true
                RETURNING user_id
            """), {"uid": firebase_uid, "org": EVAL_ORG, "clearance": clearance}).scalar()
    finally:
        engine.dispose()


def firebase_uid_for(email: str, project: str, key_path: str) -> str:
    import firebase_admin
    from firebase_admin import auth, credentials

    try:
        app = firebase_admin.get_app("setup-eval-admin")
    except ValueError:
        app = firebase_admin.initialize_app(credentials.Certificate(key_path),
                                            {"projectId": project}, name="setup-eval-admin")
    return auth.get_user_by_email(email, app=app).uid


def main() -> int:
    from dotenv import dotenv_values

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--database-url", default=os.environ.get("MIGRATION_DATABASE_URL"),
                        help="Owner Postgres URL (default: MIGRATION_DATABASE_URL)")
    parser.add_argument("--clearance", type=int, default=0, help="The eval admin's clearance")
    args = parser.parse_args()

    env = {**dotenv_values(REPO / ".env"), **{k: v for k, v in os.environ.items() if v}}
    email = env.get("EVAL_ADMIN_EMAIL") or ""
    if not args.database_url:
        print("Pass --database-url (the owner) or set MIGRATION_DATABASE_URL.", file=sys.stderr)
        return 2
    if not email.lower().endswith(TEST_EMAIL_DOMAIN):
        print(f"EVAL_ADMIN_EMAIL must be a {TEST_EMAIL_DOMAIN} account.", file=sys.stderr)
        return 2
    key_path = env.get("GOOGLE_APPLICATION_CREDENTIALS") or ""
    if not key_path or Path(key_path).resolve().is_relative_to(REPO):
        print("GOOGLE_APPLICATION_CREDENTIALS must point to a key file outside the repo.",
              file=sys.stderr)
        return 2

    uid = firebase_uid_for(email, env.get("FIREBASE_PROJECT_ID") or "", key_path)
    user_id = ensure_eval_admin(args.database_url, uid, args.clearance)
    print(f"{email} (uid {uid}) is an admin in org '{EVAL_ORG}' (user {user_id}, "
          f"clearance {args.clearance}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
