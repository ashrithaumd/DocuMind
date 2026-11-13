"""Who is calling: Firebase ID token -> user row -> Principal.

Firebase only identifies the user (its uid). Org, teams, role, clearance and
the `active` flag live in Postgres (migration 0003) and are read on every
request, so disabling a user or moving them between orgs takes effect on
their next request, not at their next token refresh.

Deny by default: every endpoint depends on `get_principal`, which answers
  - 401 without a valid bearer token,
  - 403 for a valid token whose user has no row, no org, or active=false.
`GET /me` is the one exception: any valid token gets an answer saying
whether the user is set up (so the frontend can show "not set up yet").
`/health` stays open.

Admin routes use `get_admin_principal`, which also asks Firebase whether the
token was revoked (or the Firebase account disabled), so a revoked admin
session is refused on its very next request. Other routes skip that check:
it's a network call per request. There, a revoked token keeps working until
it expires (at most an hour); setting users.active=false is the immediate
off switch everywhere.

The token check sits behind the TokenChecker protocol: production uses
FirebaseTokenChecker, tests use tests.helpers.FakeTokenChecker (no
emulator, no network).
"""
import asyncio
from dataclasses import dataclass
from typing import Protocol

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import deps
from config import Settings


class InvalidToken(Exception):
    """The token is malformed, expired, revoked, or not for this project."""


class TokenChecker(Protocol):
    def check_ready(self) -> None:
        """Raise RuntimeError, with what to fix, if tokens can't be checked."""
        ...

    def verify(self, token: str, check_revoked: bool = False) -> str:
        """Return the Firebase uid the token was issued to, or raise InvalidToken.
        check_revoked also refuses revoked tokens and disabled accounts."""
        ...


class FirebaseTokenChecker:
    APP_NAME = "finpilot"

    def __init__(self, settings: Settings):
        import firebase_admin

        self._project_id = settings.firebase_project_id
        try:
            self._app = firebase_admin.get_app(self.APP_NAME)
        except ValueError:
            # Credentials come from GOOGLE_APPLICATION_CREDENTIALS (a key file
            # kept outside the repo; in docker-compose, mounted read-only).
            # firebase-admin loads them when its auth client is created, so
            # even a plain signature check needs them: check_ready() at
            # startup fails fast instead of a 500 on the first request.
            options = {"projectId": settings.firebase_project_id} if settings.firebase_project_id else None
            self._app = firebase_admin.initialize_app(options=options, name=self.APP_NAME)
        # Without it, a token used the moment it's issued fails on any server
        # whose clock is slightly behind Google's (see config.py).
        self._clock_skew = settings.firebase_clock_skew_seconds

    def check_ready(self) -> None:
        """Fail at startup, with what to fix, if Firebase config is missing."""
        if not self._project_id:
            raise RuntimeError("FIREBASE_PROJECT_ID is not set: no sign-in token can be checked.")
        from google.auth.exceptions import DefaultCredentialsError

        try:
            self._app.credential.get_credential()
        except DefaultCredentialsError as exc:
            # The message names the missing variable or file, never a secret.
            raise RuntimeError(
                "Firebase credentials not found: set GOOGLE_APPLICATION_CREDENTIALS to the "
                "service account key file (outside the repo; docker-compose mounts it "
                f"read-only). Details: {exc}") from exc

    def verify(self, token: str, check_revoked: bool = False) -> str:
        from firebase_admin import auth

        try:
            # check_revoked looks the user up in Firebase, which needs the
            # service account key (GOOGLE_APPLICATION_CREDENTIALS).
            claims = auth.verify_id_token(token, app=self._app, check_revoked=check_revoked,
                                          clock_skew_seconds=self._clock_skew)
        except (auth.InvalidIdTokenError, auth.UserDisabledError) as exc:
            # Expired and revoked tokens are InvalidIdTokenError subclasses;
            # UserDisabledError (check_revoked only) is not. Anything else (no
            # project id or key configured, Firebase unreachable) is a server
            # problem, not a bad token, and surfaces as a 500.
            raise InvalidToken(str(exc)) from exc
        return claims["uid"]


@dataclass(frozen=True)
class Principal:
    user_id: str
    firebase_uid: str
    org_id: str
    role: str
    clearance: int
    team_ids: tuple[str, ...]
    customer_id: str | None = None  # set for bank customers; None for staff


_bearer = HTTPBearer(auto_error=False)


async def _identify(credentials: HTTPAuthorizationCredentials | None,
                    check_revoked: bool = False) -> tuple[str, dict | None]:
    """The token's Firebase uid and that user's row (None if unknown); 401 on a bad token."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Sign in required.",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        uid = await asyncio.to_thread(deps.get_token_checker().verify, credentials.credentials,
                                      check_revoked)
    except InvalidToken as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token.",
                            headers={"WWW-Authenticate": "Bearer"}) from exc
    return uid, await asyncio.to_thread(deps.get_store().get_user, uid)


def _not_set_up(user: dict | None) -> str | None:
    """Why this user may not use the API, or None if they may."""
    if user is None:
        return "unknown_user"
    if not user["active"]:
        return "inactive"
    if user["org_id"] is None:
        return "no_org"
    return None


async def get_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    return await _principal(credentials, check_revoked=False)


async def get_admin_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    """For /admin/*: get_principal plus Firebase's revoked-token check, and
    role admin (403 otherwise). What an admin reads is still limited to
    their own org, by RLS."""
    principal = await _principal(credentials, check_revoked=True)
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Admins only.")
    return principal


async def _principal(credentials, check_revoked: bool) -> Principal:
    _, user = await _identify(credentials, check_revoked)
    reason = _not_set_up(user)
    if reason:
        raise HTTPException(status_code=403, detail=f"Account not set up ({reason}).")
    return Principal(
        user_id=user["user_id"], firebase_uid=user["firebase_uid"], org_id=user["org_id"],
        role=user["role"], clearance=user["clearance"], team_ids=tuple(user["team_ids"]),
        customer_id=user.get("customer_id"),
    )


async def get_me(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    """Body of GET /me: any valid token gets an answer, set up or not."""
    uid, user = await _identify(credentials)
    reason = _not_set_up(user)
    me = {"firebase_uid": uid, "set_up": reason is None, "reason": reason}
    if reason is None:
        me |= {k: user.get(k) for k in ("user_id", "org_id", "role", "clearance", "customer_id")}
        me["team_ids"] = list(user["team_ids"])
    return me
