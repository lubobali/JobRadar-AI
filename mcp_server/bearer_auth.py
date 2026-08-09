"""Bearer-token authentication for the MCP server.

Ported unchanged from SkyCast-AI except for the environment variable name.
There, it was a workaround discovered late; here it is the plan from the start,
because the reason has not changed.

**The Databricks AI Gateway cannot authenticate to a Databricks App.** Proven
on the previous project by working through every method its own form offers:

  Dynamic Client Registration   The gateway offers it, and it is the MCP
                                standard. The workspace OIDC server refuses:
                                "Authorization server ... does not support DCR.
                                The server metadata does not include a
                                registration_endpoint."

  Bearer token / PAT            A personal access token is not an OAuth token.
                                Requests carrying one get a 302 to
                                /oidc/oauth2/v2.0/authorize - the app's auth
                                proxy rejects it before the request ever
                                reaches this process.

  OAuth M2M                     Needs a service principal with an OAuth secret.
                                Creating one is a workspace-admin operation,
                                and this is a shared bootcamp workspace where
                                the account is in `users`, not `admins`:
                                "...ServicePrincipals is only accessible by
                                admins."

  OAuth U2M                     Needs the client's redirect URI registered
                                against the app's OAuth client, which is
                                Databricks-managed and not editable.

So JobRadar splits the two deployments by role rather than duplicating one.
The **Databricks App is the frontend** (capstone requirement 4, which wants a
frontend anyway) and the **MCP server runs on jobradar.lubot.ai** behind this
token (requirement 5). One deployment each, doing what it is good at, instead
of the same code running twice.

The token is read from the environment and compared in constant time. Set
SKYCAST_BEARER_TOKEN to switch this on; leave it unset and the server runs
open, which is correct on Databricks Apps where the platform authenticates
every request before it arrives.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# Spelled out rather than imported from starlette, so this module keeps no
# dependency on the framework it happens to be wrapping.
Scope = dict
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

ENV_VAR = "JOBRADAR_BEARER_TOKEN"

# Reachable without a token. The landing page explains what this server is,
# which is only useful if a human can actually read it, and /status carries no
# weather data and no secret value - only which source the contact string came
# from. Everything that costs an upstream API call is behind the token.
PUBLIC_PATHS = frozenset({"/", "/status", "/healthz", "/favicon.ico"})


def configured_token() -> str | None:
    """The expected token, or None if this deployment runs open."""
    token = (os.environ.get(ENV_VAR) or "").strip()
    return token or None


def _presented(header: str | None) -> str:
    """Pull the credential out of an Authorization header.

    Accepts "Bearer <token>" and a bare token. Some clients send the scheme,
    some do not, and rejecting the bare form produces a 401 that looks like a
    wrong token rather than a wrong format.
    """
    value = (header or "").strip()
    if not value:
        return ""
    scheme, _, rest = value.partition(" ")
    if scheme.lower() == "bearer" and rest.strip():
        return rest.strip()
    return value


def token_is_valid(header: str | None, expected: str) -> bool:
    """Constant-time comparison of the presented credential.

    hmac.compare_digest rather than ==, so the comparison does not return early
    on the first wrong character. The practical risk here is small; the cost of
    doing it properly is one function call.
    """
    presented = _presented(header)
    if not presented:
        return False
    return hmac.compare_digest(presented, expected)


def wrap(app: ASGIApp, expected_token: str) -> ASGIApp:
    """Wrap an ASGI app so every non-public path requires the token.

    Written as raw ASGI rather than a Starlette BaseHTTPMiddleware because the
    MCP transport is a streaming one: BaseHTTPMiddleware buffers the response
    body, which breaks streamable HTTP in a way that only shows up under a real
    client, not under a test that reads the whole body anyway.
    """

    async def guarded(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in PUBLIC_PATHS:
            await app(scope, receive, send)
            return

        headers = {key.decode("latin-1").lower(): value for key, value in scope.get("headers", [])}
        presented = headers.get("authorization")
        if token_is_valid(presented.decode("latin-1") if presented else None, expected_token):
            await app(scope, receive, send)
            return

        logger.warning("Rejected an unauthenticated request to %s", scope.get("path"))
        body = b'{"error":"Unauthorized. Send Authorization: Bearer <token>."}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    # Naming the scheme is what lets a client fix itself rather
                    # than guess. Required by RFC 7235 for a 401 anyway.
                    (b"www-authenticate", b'Bearer realm="jobradar"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return guarded


__all__ = ["ENV_VAR", "PUBLIC_PATHS", "configured_token", "token_is_valid", "wrap"]
