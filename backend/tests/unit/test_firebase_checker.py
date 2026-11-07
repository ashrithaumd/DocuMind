"""FirebaseTokenChecker maps Firebase's verdicts without touching the network:
verify_id_token is replaced, so only our wrapping is under test."""
import pytest
from pydantic import ValidationError
from firebase_admin import auth as fb_auth

from auth import FirebaseTokenChecker, InvalidToken
from config import Settings


@pytest.fixture
def checker():
    return FirebaseTokenChecker(Settings(_env_file=None, firebase_project_id="demo-project"))


def test_valid_token_returns_the_uid(checker, monkeypatch):
    seen = {}

    def verify(token, app, check_revoked=False, clock_skew_seconds=0):
        seen.update(token=token, project=app.project_id, check_revoked=check_revoked,
                    clock_skew_seconds=clock_skew_seconds)
        return {"uid": "u1"}

    monkeypatch.setattr(fb_auth, "verify_id_token", verify)
    assert checker.verify("tok") == "u1"
    assert seen == {"token": "tok", "project": "demo-project", "check_revoked": False,
                    "clock_skew_seconds": 5}
    assert checker.verify("tok", check_revoked=True) == "u1"
    assert seen["check_revoked"] is True


@pytest.mark.parametrize("error", [
    fb_auth.InvalidIdTokenError("bad"),
    fb_auth.ExpiredIdTokenError("old", cause=None),
    fb_auth.RevokedIdTokenError("revoked"),
    fb_auth.UserDisabledError("disabled"),  # not an InvalidIdTokenError subclass
])
def test_bad_tokens_become_invalid_token(checker, monkeypatch, error):
    def verify(token, app, check_revoked=False, clock_skew_seconds=0):
        raise error

    monkeypatch.setattr(fb_auth, "verify_id_token", verify)
    with pytest.raises(InvalidToken):
        checker.verify("tok")


def test_server_side_failures_are_not_reported_as_bad_tokens(checker, monkeypatch):
    # e.g. no project id configured: a 500, never a misleading 401.
    def verify(token, app, check_revoked=False, clock_skew_seconds=0):
        raise ValueError("A project ID is required")

    monkeypatch.setattr(fb_auth, "verify_id_token", verify)
    with pytest.raises(ValueError):
        checker.verify("tok")


def test_clock_skew_follows_the_setting(monkeypatch):
    # A fresh FirebaseTokenChecker reuses the named app; only the skew differs.
    seen = {}
    monkeypatch.setattr(fb_auth, "verify_id_token",
                        lambda token, app, check_revoked=False, clock_skew_seconds=0:
                        seen.update(skew=clock_skew_seconds) or {"uid": "u1"})
    FirebaseTokenChecker(Settings(_env_file=None, firebase_clock_skew_seconds=0)).verify("tok")
    assert seen["skew"] == 0


@pytest.mark.parametrize("value", [-1, 61])
def test_clock_skew_outside_firebases_0_to_60_is_rejected(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, firebase_clock_skew_seconds=value)


# --- startup check: fail fast instead of a 500 on the first sign-in -----------

def test_check_ready_needs_a_project_id():
    checker = FirebaseTokenChecker(Settings(_env_file=None, firebase_project_id=None))
    checker._project_id = None
    with pytest.raises(RuntimeError, match="FIREBASE_PROJECT_ID is not set"):
        checker.check_ready()


def test_check_ready_needs_credentials(checker, monkeypatch):
    # firebase-admin loads credentials when its auth client is created, so a
    # missing key breaks even plain token checks (seen in docker-compose).
    from google.auth.exceptions import DefaultCredentialsError

    def missing():
        raise DefaultCredentialsError("Your default credentials were not found.")

    monkeypatch.setattr(checker._app.credential, "get_credential", missing)
    with pytest.raises(RuntimeError, match="GOOGLE_APPLICATION_CREDENTIALS"):
        checker.check_ready()


def test_check_ready_passes_with_project_and_credentials(checker, monkeypatch):
    monkeypatch.setattr(checker._app.credential, "get_credential", lambda: object())
    checker.check_ready()
