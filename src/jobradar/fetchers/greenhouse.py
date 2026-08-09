"""Greenhouse — public job board API.

    GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true

No auth, no approval: a company's board token is all it takes. There is no server-side
search, so we fetch a board whole and filter downstream.

Parsing is pure and IO is a thin shell around it, so the field mapping — the part that
actually breaks — is tested against a recorded board with no HTTP in the way.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx

from jobradar.fetchers.base import FetchError, build_client, utcnow
from jobradar.html_text import to_plain_text
from jobradar.models import Job

SOURCE = "greenhouse"
BOARD_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"

_LOCATION_TYPE = "Location Type"
_REMOTE_LOCATION_TYPE = "Remote"


def fetch_jobs(
    board_token: str,
    *,
    client: httpx.Client | None = None,
    now: Callable[[], datetime] = utcnow,
) -> list[Job]:
    """Fetch and normalize every open posting on one Greenhouse board.

    Raises FetchError if the board cannot be read or does not return the documented shape.
    """
    owned = client is None
    http = client or build_client()
    try:
        response = http.get(BOARD_URL.format(token=board_token), params={"content": "true"})
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise FetchError(
            f"greenhouse board {board_token!r} returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"greenhouse board {board_token!r} is unreachable: {exc}") from exc
    except ValueError as exc:
        raise FetchError(f"greenhouse board {board_token!r} returned non-JSON") from exc
    finally:
        if owned:
            http.close()

    return parse_board(payload, fetched_at=now())


def parse_board(payload: dict[str, Any], *, fetched_at: datetime) -> list[Job]:
    """Normalize a raw board payload into Jobs. Pure — no clock, no network."""
    try:
        raw_jobs = payload["jobs"]
        return [_parse_job(raw, fetched_at=fetched_at) for raw in raw_jobs]
    except (KeyError, TypeError) as exc:
        raise FetchError(f"unexpected greenhouse payload shape: {exc}") from exc


def _parse_job(raw: dict[str, Any], *, fetched_at: datetime) -> Job:
    location = raw["location"]["name"]
    metadata = raw.get("metadata") or []
    return Job(
        source=SOURCE,
        source_id=str(raw["id"]),  # arrives as a JSON int; the dedup key must be type-stable
        company=raw["company_name"],
        title=raw["title"],
        url=raw["absolute_url"],
        location=location,
        remote=_is_remote(location, metadata),
        # Greenhouse exposes no salary field; it is prose inside `content`, if present at all.
        salary=None,
        description=to_plain_text(raw.get("content") or ""),
        posted_at=_parse_timestamp(raw.get("first_published")),
        fetched_at=fetched_at,
    )


def _is_remote(location: str, metadata: list[dict[str, Any]]) -> bool:
    """Report whether a posting can be worked remotely, believing either signal.

    Both signals are unreliable alone, and real boards contain each without the other: a
    posting tagged Location Type=Remote whose location reads "Sydney, Australia", and an
    On-Site-tagged posting open to "Remote-Friendly, United States". Taking the OR trades a
    glance at a false positive for never missing a real remote role.

    >>> _is_remote("Sydney, Australia", [{"name": "Location Type", "value": "Remote"}])
    True
    >>> _is_remote("Remote-Friendly, United States", [])
    True
    >>> _is_remote("San Francisco, CA | Seattle, WA", [])
    False
    """
    tagged_remote = any(
        item.get("name") == _LOCATION_TYPE and item.get("value") == _REMOTE_LOCATION_TYPE
        for item in metadata
    )
    return tagged_remote or "remote" in location.lower()


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse a Greenhouse ISO-8601 timestamp, keeping its offset.

    `first_published` is the field that matters and `updated_at` is a decoy: boards bulk-
    refresh it, so it reports the last sync rather than the day the role went live.
    Returning None beats guessing — a wrong date silently defeats the ghost-job age filter.

    >>> _parse_timestamp("2026-07-10T16:15:41-04:00").isoformat()
    '2026-07-10T16:15:41-04:00'
    >>> _parse_timestamp(None) is None
    True
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
