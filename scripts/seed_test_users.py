"""Seed the test users in kb/test-users.yaml, as the owner. Idempotent.

    backend/.venv/Scripts/python scripts/phase3_db.py run -- python scripts/seed_test_users.py seed [--with-demo]
    backend/.venv/Scripts/python scripts/seed_test_users.py create-demo-firebase-users

seed
    Upserts every test user (Firebase uid "test:<id>", for the fake token
    checker) with its org, role, clearance and team memberships, creating
    the orgs and teams from kb/manifest.yaml first. With --with-demo, also
    the demo users, whose uids are looked up in Firebase by email (create
    them first). Runs as the owner (MIGRATION_DATABASE_URL or
    --database-url): the app role can only read the user tables. Re-running
    leaves the same rows: memberships are replaced, never duplicated.

create-demo-firebase-users
    Creates the demo accounts in Firebase (email/password) if they don't
    exist. Each password is generated into the repo-root .env
    (DEMO_PASSWORD_<NAME>, e.g. DEMO_PASSWORD_AURORA_VIEWER) and never
    printed; an existing account gets the password from .env. Needs
    FIREBASE_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS in .env. Only
    @documind.test accounts.
"""
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "kb" / "tools"))
sys.path.insert(0, str(REPO / "scripts"))
from manifest import Manifest, load_manifest  # noqa: E402

USERS_FILE = REPO / "kb" / "test-users.yaml"
ROLES = {"viewer", "editor", "admin"}
TEST_EMAIL_DOMAIN = "@documind.test"


@dataclass(frozen=True)
class TestUser:
    id: str
    org: str
    role: str
    clearance: int
    teams: tuple[str, ...]
    email: str | None = None

    @property
    def fake_uid(self) -> str:
        return f"test:{self.id}"

    @property
    def password_var(self) -> str:
        return "DEMO_PASSWORD_" + self.email.split("@")[0].upper().replace("-", "_")


def load_users(path: Path = USERS_FILE, manifest: Manifest | None = None) -> list[TestUser]:
    """Parse and validate against the manifest. Raises ValueError listing every problem."""
    manifest = manifest or load_manifest()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    problems, users, seen_ids, seen_emails = [], [], set(), set()
    for i, u in enumerate(raw.get("users") or [], start=1):
        where = f"user #{i} ({u.get('id', '?') if isinstance(u, dict) else u!r})"
        allowed = {"id", "org", "role", "clearance", "teams", "email"}
        if not isinstance(u, dict) or not {"id", "org", "role", "clearance", "teams"} <= set(u) <= allowed:
            problems.append(f"{where}: needs id, org, role, clearance, teams (and optionally email)")
            continue
        if u["id"] in seen_ids:
            problems.append(f"{where}: duplicate id")
        seen_ids.add(u["id"])
        if u["org"] not in manifest.orgs:
            problems.append(f"{where}: unknown org {u['org']!r}")
        else:
            bad = [t for t in u["teams"] or [] if t not in manifest.teams(u["org"])]
            if bad:
                problems.append(f"{where}: {bad} are not teams of {u['org']}")
        if u["role"] not in ROLES:
            problems.append(f"{where}: role must be one of {sorted(ROLES)}")
        if not isinstance(u["clearance"], int) or isinstance(u["clearance"], bool) or not 0 <= u["clearance"] <= 3:
            problems.append(f"{where}: clearance must be an integer 0-3")
        email = u.get("email")
        if email is not None:
            if not str(email).lower().endswith(TEST_EMAIL_DOMAIN):
                problems.append(f"{where}: demo emails must be {TEST_EMAIL_DOMAIN} accounts")
            if email in seen_emails:
                problems.append(f"{where}: duplicate email")
            seen_emails.add(email)
        users.append(TestUser(id=u["id"], org=u["org"], role=u["role"], clearance=u["clearance"],
                              teams=tuple(u["teams"] or ()), email=email))
    if problems:
        raise ValueError("\n".join(problems))
    return users


def seed(owner_url: str, users: list[TestUser], manifest: Manifest,
         uid_for=lambda user: user.fake_uid) -> dict[str, str]:
    """Upsert users and memberships; returns {test-user id: users.user_id}."""
    import sqlalchemy as sa

    from load_corpus import _upsert_orgs

    engine = sa.create_engine(owner_url)
    ids: dict[str, str] = {}
    try:
        with engine.begin() as conn:
            _upsert_orgs(conn, manifest)
            for user in users:
                user_id = conn.execute(sa.text("""
                    INSERT INTO users (firebase_uid, org_id, role, clearance, active)
                    VALUES (:uid, :org, :role, :clearance, true)
                    ON CONFLICT (firebase_uid) DO UPDATE SET org_id = EXCLUDED.org_id,
                        role = EXCLUDED.role, clearance = EXCLUDED.clearance, active = true
                    RETURNING user_id
                """), {"uid": uid_for(user), "org": user.org, "role": user.role,
                       "clearance": user.clearance}).scalar()
                conn.execute(sa.text("DELETE FROM team_memberships WHERE user_id = :u"), {"u": user_id})
                for team in user.teams:
                    conn.execute(sa.text("INSERT INTO team_memberships VALUES (:u, :t, :org)"),
                                 {"u": user_id, "t": team, "org": user.org})
                ids[user.id] = user_id
    finally:
        engine.dispose()
    return ids


# --- real Firebase demo accounts ---------------------------------------------------

def _firebase_app(env: dict[str, str]):
    import firebase_admin
    from firebase_admin import credentials

    key = env.get("GOOGLE_APPLICATION_CREDENTIALS") or ""
    if not key or Path(key).resolve().is_relative_to(REPO):
        raise SystemExit("GOOGLE_APPLICATION_CREDENTIALS must point to a key file outside the repo.")
    try:
        return firebase_admin.get_app("seed-test-users")
    except ValueError:
        return firebase_admin.initialize_app(credentials.Certificate(key),
                                             {"projectId": env.get("FIREBASE_PROJECT_ID")},
                                             name="seed-test-users")


def create_demo_firebase_users(users: list[TestUser]) -> None:
    from firebase_admin import auth

    from phase3_db import ENV_FILE, ensure_secrets, read_env

    demo = [u for u in users if u.email]
    added = ensure_secrets(ENV_FILE, tuple(u.password_var for u in demo),
                           header="# Firebase demo users (scripts/seed_test_users.py)")
    env = {**read_env(), **{k: v for k, v in os.environ.items() if v}}
    app = _firebase_app(env)
    for user in demo:
        password = env[user.password_var]
        try:
            existing = auth.get_user_by_email(user.email, app=app)
            auth.update_user(existing.uid, password=password, app=app)
            print(f"{user.email}: existed; password set from .env ({user.password_var})")
        except auth.UserNotFoundError:
            auth.create_user(email=user.email, password=password, email_verified=True, app=app)
            print(f"{user.email}: created; password in .env ({user.password_var})")
    if added:
        print(f"Generated {', '.join(added)} in .env (values not shown).")


def firebase_uids(users: list[TestUser]) -> dict[str, str]:
    from firebase_admin import auth

    from phase3_db import read_env

    env = {**read_env(), **{k: v for k, v in os.environ.items() if v}}
    app = _firebase_app(env)
    return {u.id: auth.get_user_by_email(u.email, app=app).uid for u in users if u.email}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seed")
    s.add_argument("--database-url", default=os.environ.get("MIGRATION_DATABASE_URL"))
    s.add_argument("--with-demo", action="store_true", help="Also seed the Firebase demo users")
    sub.add_parser("create-demo-firebase-users")
    args = parser.parse_args()

    manifest = load_manifest()
    users = load_users(manifest=manifest)
    if args.command == "create-demo-firebase-users":
        create_demo_firebase_users(users)
        return 0
    if not args.database_url:
        print("Pass --database-url (the owner) or set MIGRATION_DATABASE_URL.", file=sys.stderr)
        return 2
    test_users = [u for u in users if not u.email]
    ids = seed(args.database_url, test_users, manifest)
    print(f"Seeded {len(ids)} test users (fake-checker uids test:<id>).")
    if args.with_demo:
        demo = [u for u in users if u.email]
        uids = firebase_uids(demo)
        seed(args.database_url, demo, manifest, uid_for=lambda u: uids[u.id])
        for u in demo:
            print(f"Seeded demo user {u.email}: {u.org}, {u.role}, clearance {u.clearance}, "
                  f"teams {list(u.teams) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
