"""Signs the eval harness in as the eval admin, for every API call.

FirebaseAuth is an httpx.Auth: it signs in through the Firebase Auth REST API
(email + password), sends the ID token as a Bearer header on every request,
and refreshes it with the refresh token before it expires (ID tokens last an
hour; eval runs can last longer). A 401 triggers one refresh and retry, in
case the token was revoked or the clocks disagree. The password and tokens
are never printed, logged or put in an exception message.
"""
import time

import httpx

SIGN_IN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
REFRESH_URL = "https://securetoken.googleapis.com/v1/token"
REFRESH_MARGIN_S = 300  # refresh this long before the token expires


class FirebaseSignInError(RuntimeError):
    """Firebase refused the sign-in or refresh. The message holds only
    Firebase's error code (e.g. INVALID_LOGIN_CREDENTIALS)."""


class FirebaseAuth(httpx.Auth):
    def __init__(self, web_api_key: str, email: str, password: str,
                 http: httpx.Client | None = None, clock=time.time):
        self._key = web_api_key
        self._email = email
        self._password = password
        self._http = http or httpx.Client(timeout=20)
        self._clock = clock
        self._id_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at = 0.0
        self.sign_ins = 0
        self.refreshes = 0

    def __repr__(self) -> str:  # never the password or a token
        return f"FirebaseAuth(email={self._email!r})"

    def _post(self, url: str, **kwargs) -> dict:
        response = self._http.post(url, params={"key": self._key}, **kwargs)
        if response.status_code != 200:
            try:
                code = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                code = f"HTTP {response.status_code}"
            raise FirebaseSignInError(f"Firebase refused the request: {code}")
        return response.json()

    def _sign_in(self) -> None:
        body = self._post(SIGN_IN_URL, json={"email": self._email, "password": self._password,
                                             "returnSecureToken": True})
        self.sign_ins += 1
        self._store(body["idToken"], body["refreshToken"], body["expiresIn"])

    def _refresh(self) -> None:
        try:
            body = self._post(REFRESH_URL, data={"grant_type": "refresh_token",
                                                 "refresh_token": self._refresh_token})
        except FirebaseSignInError:
            self._sign_in()  # e.g. the refresh token was revoked: start over
            return
        self.refreshes += 1
        self._store(body["id_token"], body["refresh_token"], body["expires_in"])

    def _store(self, id_token: str, refresh_token: str, expires_in) -> None:
        self._id_token, self._refresh_token = id_token, refresh_token
        self._expires_at = self._clock() + int(expires_in)

    def token(self) -> str:
        if self._id_token is None:
            self._sign_in()
        elif self._clock() >= self._expires_at - REFRESH_MARGIN_S:
            self._refresh()
        return self._id_token

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self.token()}"
        response = yield request
        if response.status_code == 401:
            self._refresh()
            request.headers["Authorization"] = f"Bearer {self._id_token}"
            yield request
