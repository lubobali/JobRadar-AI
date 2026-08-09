"""Shared plumbing for every source fetcher."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

TIMEOUT = httpx.Timeout(10.0, connect=5.0)
"""A board that stalls must not stall the whole run; Phase 4 fans out over dozens of them."""

USER_AGENT = "aws-job-streamer/0.1 (+https://github.com/lubobali/aws-job-streamer)"
"""Identify ourselves honestly to the public APIs we poll."""


class FetchError(Exception):
    """A source could not be read.

    Raised rather than returning `[]` so that a broken board is never silently
    indistinguishable from a board with no open roles.
    """


def utcnow() -> datetime:
    """Return the current UTC time. Injected as `now` so tests never touch the clock."""
    return datetime.now(UTC)


def build_client() -> httpx.Client:
    """Return an HTTP client configured the same way for every source."""
    return httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
