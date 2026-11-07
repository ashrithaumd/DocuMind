"""The eval harness's Firebase sign-in (eval/firebase_auth.py), against a fake
Firebase and a fake API: no network. Signs in once, sends the token on
every call, refreshes before expiry (runs last over an hour), retries once
on a 401, and never puts the password or a token in an error or repr."""
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from firebase_auth import REFRESH_MARGIN_S, FirebaseAuth, FirebaseSignInError  # noqa: E402

PASSWORD = "s3cret-pw"


class FakeFirebase:
    """Issues id-1, id-2, ... and remembers what the API saw."""

    def __init__(self, sign_in_status=200):
        self.issued = 0
        self.sign_in_status = sign_in_status
        self.api_tokens: list[str] = []
        self.reject_next_api_call = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "accounts:signInWithPassword" in url:
            if self.sign_in_status != 200:
                return httpx.Response(400, json={"error": {"message": "INVALID_LOGIN_CREDENTIALS"}})
            self.issued += 1
            return httpx.Response(200, json={"idToken": f"id-{self.issued}",
                                             "refreshToken": "refresh", "expiresIn": "3600"})
        if "securetoken" in url:
            assert b"grant_type=refresh_token" in request.content
            self.issued += 1
            return httpx.Response(200, json={"id_token": f"id-{self.issued}",
                                             "refresh_token": "refresh", "expires_in": "3600"})
        self.api_tokens.append(request.headers.get("Authorization"))
        if self.reject_next_api_call:
            self.reject_next_api_call = False
            return httpx.Response(401)
        return httpx.Response(200, json={"ok": True})


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def world():
    firebase, clock = FakeFirebase(), Clock()
    transport = httpx.MockTransport(firebase.handler)
    auth = FirebaseAuth("web-key", "eval-admin@documind.test", PASSWORD,
                        http=httpx.Client(transport=transport), clock=clock)
    api = httpx.Client(transport=transport, auth=auth, base_url="http://api")
    return firebase, clock, auth, api


def test_signs_in_once_and_sends_the_token_on_every_call(world):
    firebase, _, auth, api = world
    for _ in range(3):
        assert api.get("/me").status_code == 200
    assert firebase.api_tokens == ["Bearer id-1"] * 3
    assert (auth.sign_ins, auth.refreshes) == (1, 0)


def test_refreshes_before_the_token_expires(world):
    firebase, clock, auth, api = world
    api.get("/me")
    clock.now += 3600 - REFRESH_MARGIN_S - 1  # still comfortably valid
    api.get("/me")
    clock.now += 2  # inside the margin: refresh, not a new sign-in
    api.get("/me")
    clock.now += 3 * 3600  # hours later, as in a long run
    api.get("/me")

    assert firebase.api_tokens == ["Bearer id-1", "Bearer id-1", "Bearer id-2", "Bearer id-3"]
    assert (auth.sign_ins, auth.refreshes) == (1, 2)


def test_a_401_refreshes_and_retries_once(world):
    firebase, _, auth, api = world
    api.get("/me")
    firebase.reject_next_api_call = True

    assert api.get("/me").status_code == 200
    assert firebase.api_tokens[-2:] == ["Bearer id-1", "Bearer id-2"]
    assert auth.refreshes == 1


def test_refused_sign_in_reports_only_firebases_code():
    firebase = FakeFirebase(sign_in_status=400)
    auth = FirebaseAuth("web-key", "eval-admin@documind.test", PASSWORD,
                        http=httpx.Client(transport=httpx.MockTransport(firebase.handler)))
    with pytest.raises(FirebaseSignInError) as err:
        auth.token()
    assert "INVALID_LOGIN_CREDENTIALS" in str(err.value)
    assert PASSWORD not in str(err.value)


def test_repr_holds_no_password_or_token(world):
    _, _, auth, api = world
    api.get("/me")
    assert PASSWORD not in repr(auth) and "id-1" not in repr(auth)
