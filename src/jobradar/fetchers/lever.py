"""Lever — public job board API.

    GET https://api.lever.co/v0/postings/{slug}?mode=json

No auth. Like Greenhouse there is no server-side search, so we fetch a board whole.

Three things differ from Greenhouse and each one costs a bug if missed:
  * the payload is a bare JSON list, not an object with a "jobs" key;
  * `createdAt` is epoch MILLIseconds, not seconds;
  * the posting body is split across `descriptionPlain`, `lists` and `additionalPlain` —
    the requirements live in `lists`, so the intro alone is not the posting.

The upside: Lever states `workplaceType` outright, so remote needs no guessing.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from jobradar.fetchers.base import FetchError, build_client, utcnow
from jobradar.html_text import to_plain_text
from jobradar.models import Job

SOURCE = "lever"
BOARD_URL = "https://api.lever.co/v0/postings/{slug}"

_REMOTE_WORKPLACE_TYPE = "remote"
_MILLIS_PER_SECOND = 1000


def fetch_jobs(
    board_slug: str,
    *,
    company: str | None = None,
    client: httpx.Client | None = None,
    now: Callable[[], datetime] = utcnow,
) -> list[Job]:
    """Fetch and normalize every open posting on one Lever board.

    `company` is a parameter because Lever's API never names the company — only the slug
    identifies it. It defaults to the slug so a caller can stay terse.

    Raises FetchError if the board cannot be read or does not return the documented shape.
    """
    owned = client is None
    http = client or build_client()
    try:
        response = http.get(BOARD_URL.format(slug=board_slug), params={"mode": "json"})
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise FetchError(
            f"lever board {board_slug!r} returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"lever board {board_slug!r} is unreachable: {exc}") from exc
    except ValueError as exc:
        raise FetchError(f"lever board {board_slug!r} returned non-JSON") from exc
    finally:
        if owned:
            http.close()

    if not isinstance(payload, list):
        # Decision Log #8: an error object is not an empty board. Lever answers an unknown slug
        # with {"ok": false, "error": ...}; treating that as zero jobs would hide a dead board.
        raise FetchError(
            f"lever board {board_slug!r} returned {type(payload).__name__}, not a list"
        )

    return parse_board(payload, company=company or board_slug, fetched_at=now())


def parse_board(payload: list[dict[str, Any]], *, company: str, fetched_at: datetime) -> list[Job]:
    """Normalize a raw Lever payload into Jobs. Pure — no clock, no network."""
    try:
        return [_parse_posting(raw, company=company, fetched_at=fetched_at) for raw in payload]
    except (KeyError, TypeError) as exc:
        raise FetchError(f"unexpected lever payload shape: {exc}") from exc


def _parse_posting(raw: dict[str, Any], *, company: str, fetched_at: datetime) -> Job:
    categories = raw.get("categories") or {}
    return Job(
        source=SOURCE,
        source_id=str(raw["id"]),
        company=company,
        title=raw["text"],
        url=raw["hostedUrl"],
        location=_join_locations(categories),
        remote=raw.get("workplaceType") == _REMOTE_WORKPLACE_TYPE,
        salary=None,  # Lever exposes no salary field.
        description=_full_description(raw),
        posted_at=_parse_created_at(raw.get("createdAt")),
        fetched_at=fetched_at,
    )


def _join_locations(categories: dict[str, Any]) -> str | None:
    """Return every location a posting is open to, not just the headline one.

    `location` names one office while `allLocations` carries the full set. Keeping only the
    first would hide a role that is also open in a city Lubo can actually work from.

    >>> _join_locations({"location": "Stockholm", "allLocations": ["Stockholm", "London"]})
    'Stockholm, London'
    >>> _join_locations({"location": "New York, NY", "allLocations": ["New York, NY"]})
    'New York, NY'
    >>> _join_locations({}) is None
    True
    """
    all_locations = categories.get("allLocations") or []
    if all_locations:
        return ", ".join(all_locations)
    return categories.get("location")


def _full_description(raw: dict[str, Any]) -> str:
    """Return the whole posting, not just its intro.

    Lever splits a posting across keys: `descriptionPlain` is the marketing intro, `lists` holds
    the sections that actually state the requirements ("What You'll Do", "Who You Are"), and
    `additionalPlain` closes it out. On a real Spotify posting the intro was 991 chars and the
    lists another 3407 — so scoring the intro alone reads the pitch and never the qualifications.

    Section names vary by company ("Who You Are" rather than "Requirements"), so we concatenate
    everything rather than trying to recognise which section matters.
    """
    parts = [raw.get("descriptionPlain") or ""]
    for section in raw.get("lists") or []:
        parts.append(section.get("text") or "")
        parts.append(to_plain_text(section.get("content") or ""))
    parts.append(raw.get("additionalPlain") or "")
    return " ".join(part for part in parts if part).strip()


def _parse_created_at(value: int | None) -> datetime | None:
    """Parse Lever's `createdAt`, which is epoch MILLIseconds.

    Read as seconds it lands in 1970, every job looks 56 years old, and the Phase 2 ghost-age
    filter silently discards the entire board.

    >>> _parse_created_at(1782821535482).isoformat()
    '2026-06-30T12:12:15.482000+00:00'
    >>> _parse_created_at(None) is None
    True
    """
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value / _MILLIS_PER_SECOND, UTC)
    except (OverflowError, OSError, TypeError, ValueError):
        return None
