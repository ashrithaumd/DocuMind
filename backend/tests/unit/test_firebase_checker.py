"""FirebaseTokenChecker maps Firebase's verdicts without touching the network:
verify_id_token is replaced, so only our wrapping is under test."""
import pytest
from firebase_admin import auth as fb_auth

from auth import FirebaseTokenChecker, InvalidToken
from config import Settings


@pytest.fixture
def checker():
    return FirebaseTokenChecker(Settings(_env_file=None, firebase_project_id="demo-project"))


def test_valid_token_returns_the_uid(checker, monkeypatch):
    seen = {}

    def verify(token, app, check_revoked=False):
        seen.update(token=token, project=app.project_id, check_revoked=check_revoked)
        return {"uid": "u1"}

    monkeypatch.setattr(fb_auth, "verify_id_token", verify)
    assert checker.verify("tok") == "u1"
    assert seen == {"token": "tok", "project": "demo-project", "check_revoked": False}
    assert checker.verify("tok", check_revoked=True) == "u1"
    assert seen["check_revoked"] is True


@pytest.mark.parametrize("error", [
    fb_auth.InvalidIdTokenError("bad"),
    fb_auth.ExpiredIdTokenError("old", cause=None),
    fb_auth.RevokedIdTokenError("revoked"),
    fb_auth.UserDisabledError("disabled"),  # not an InvalidIdTokenError subclass
])
def test_bad_tokens_become_invalid_token(checker, monkeypatch, error):
    def verify(token, app, check_revoked=False):
        raise error

    monkeypatch.setattr(fb_auth, "verify_id_token", verify)
    with pytest.raises(InvalidToken):
        checker.verify("tok")


def test_server_side_failures_are_not_reported_as_bad_tokens(checker, monkeypatch):
    # e.g. no project id configured: a 500, never a misleading 401.
    def verify(token, app, check_revoked=False):
        raise ValueError("A project ID is required")

    monkeypatch.setattr(fb_auth, "verify_id_token", verify)
    with pytest.raises(ValueError):
        checker.verify("tok")
