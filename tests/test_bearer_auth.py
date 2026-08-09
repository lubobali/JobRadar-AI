"""The bearer-token guard on the reachable-from-anywhere deployment."""

from __future__ import annotations

import asyncio
from typing import Any

import bearer_auth
import pytest
from pytest import MonkeyPatch

TOKEN = "s3cret-token-value"


class TestConfiguredToken:
    def test_unset_means_the_server_runs_open(self, monkeypatch: MonkeyPatch) -> None:
        # Correct on Databricks Apps, where the platform authenticates every
        # request before it reaches the process.
        monkeypatch.delenv(bearer_auth.ENV_VAR, raising=False)
        assert bearer_auth.configured_token() is None

    def test_blank_is_treated_as_unset(self, monkeypatch: MonkeyPatch) -> None:
        # An empty env var is a deployment mistake, not a request for an empty
        # password. Treating "" as a valid token would accept every request
        # that sends no credential at all.
        monkeypatch.setenv(bearer_auth.ENV_VAR, "   ")
        assert bearer_auth.configured_token() is None

    def test_a_real_value_is_returned_stripped(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv(bearer_auth.ENV_VAR, f"  {TOKEN}\n")
        assert bearer_auth.configured_token() == TOKEN


class TestTokenIsValid:
    def test_accepts_the_bearer_scheme(self) -> None:
        assert bearer_auth.token_is_valid(f"Bearer {TOKEN}", TOKEN)

    def test_scheme_is_case_insensitive(self) -> None:
        assert bearer_auth.token_is_valid(f"bearer {TOKEN}", TOKEN)

    def test_accepts_a_bare_token(self) -> None:
        # Some clients send the scheme, some do not. Rejecting the bare form
        # produces a 401 that looks like a wrong token rather than a wrong
        # format, which is a long way to walk for a missing word.
        assert bearer_auth.token_is_valid(TOKEN, TOKEN)

    def test_rejects_a_wrong_token(self) -> None:
        assert not bearer_auth.token_is_valid(f"Bearer {TOKEN}x", TOKEN)

    def test_rejects_a_missing_header(self) -> None:
        assert not bearer_auth.token_is_valid(None, TOKEN)

    def test_rejects_an_empty_header(self) -> None:
        assert not bearer_auth.token_is_valid("   ", TOKEN)

    def test_rejects_the_scheme_with_no_credential(self) -> None:
        assert not bearer_auth.token_is_valid("Bearer ", TOKEN)


class Recorder:
    """A minimal ASGI app that records whether it was reached."""

    def __init__(self) -> None:
        self.reached = False

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:  # noqa: ANN401
        self.reached = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def call(
    app: Any,  # noqa: ANN401
    path: str = "/mcp",
    authorization: str | None = None,
    scope_type: str = "http",
) -> list[dict]:
    """Drive one ASGI request through the wrapped app and collect the messages."""
    headers = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))

    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(app({"type": scope_type, "path": path, "headers": headers}, receive, send))
    return sent


class TestWrap:
    def test_a_valid_token_reaches_the_app(self) -> None:
        inner = Recorder()
        call(bearer_auth.wrap(inner, TOKEN), authorization=f"Bearer {TOKEN}")
        assert inner.reached

    def test_no_token_is_rejected_with_401(self) -> None:
        inner = Recorder()
        sent = call(bearer_auth.wrap(inner, TOKEN))
        assert not inner.reached
        assert sent[0]["status"] == 401

    def test_a_wrong_token_is_rejected(self) -> None:
        inner = Recorder()
        sent = call(bearer_auth.wrap(inner, TOKEN), authorization="Bearer nope")
        assert not inner.reached
        assert sent[0]["status"] == 401

    def test_the_401_names_the_scheme(self) -> None:
        # RFC 7235 requires it, and it is what lets a client fix itself rather
        # than guess at what was wrong.
        sent = call(bearer_auth.wrap(Recorder(), TOKEN))
        headers = {key.lower(): value for key, value in sent[0]["headers"]}
        assert b"Bearer" in headers[b"www-authenticate"]

    def test_the_401_body_is_json(self) -> None:
        # The caller is an MCP client that parses every response as JSON. An
        # HTML error page makes it fail while parsing, hiding a 401 behind a
        # complaint about an unexpected '<'.
        sent = call(bearer_auth.wrap(Recorder(), TOKEN))
        headers = {key.lower(): value for key, value in sent[0]["headers"]}
        assert headers[b"content-type"] == b"application/json"
        assert b'"error"' in sent[1]["body"]

    @pytest.mark.parametrize("path", sorted(bearer_auth.PUBLIC_PATHS))
    def test_public_paths_need_no_token(self, path: str) -> None:
        # The landing page explains what this server is, which is only useful
        # if a human can read it. Neither it nor /status costs an upstream API
        # call or reveals a secret value.
        inner = Recorder()
        call(bearer_auth.wrap(inner, TOKEN), path=path)
        assert inner.reached

    def test_the_mcp_path_is_not_public(self) -> None:
        inner = Recorder()
        call(bearer_auth.wrap(inner, TOKEN), path="/mcp")
        assert not inner.reached

    def test_a_path_that_merely_starts_like_a_public_one_is_guarded(self) -> None:
        # "/statuses" must not inherit "/status"'s exemption. An exact-match
        # set rather than a prefix test is what makes that true.
        inner = Recorder()
        call(bearer_auth.wrap(inner, TOKEN), path="/statuses")
        assert not inner.reached

    def test_non_http_scopes_pass_through(self) -> None:
        # Lifespan startup and shutdown are not requests and carry no headers;
        # rejecting them would stop the server booting.
        inner = Recorder()
        call(bearer_auth.wrap(inner, TOKEN), scope_type="lifespan")
        assert inner.reached
