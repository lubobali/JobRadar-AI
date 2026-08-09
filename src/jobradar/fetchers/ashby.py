"""Ashby — public job board API.

    GET https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true

No auth. No server-side search, so we fetch a board whole.

The richest source we have. It states `workplaceType` (Remote/Hybrid/OnSite) outright, publishes
a clean ISO-8601 `publishedAt`, ships a complete `descriptionPlain`, and — uniquely — carries the
employer's own salary range. `includeCompensation=true` is required to get it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx

from jobradar.fetchers.base import FetchError, build_client, utcnow
from jobradar.html_text import to_plain_text
from jobradar.models import Job

SOURCE = "ashby"
BOARD_URL = "https://api.ashbyhq.com/posting-api/job-board/{slug}"


def fetch_jobs(
    board_slug: str,
    *,
    company: str | None = None,
    client: httpx.Client | None = None,
    now: Callable[[], datetime] = utcnow,
) -> list[Job]:
    """Fetch and normalize every listed posting on one Ashby board.

    `company` defaults to the slug: like Lever, Ashby's board API never names the company.

    Raises FetchError if the board cannot be read or does not return the documented shape.
    """
    owned = client is None
    http = client or build_client()
    try:
        response = http.get(
            BOARD_URL.format(slug=board_slug), params={"includeCompensation": "true"}
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise FetchError(
            f"ashby board {board_slug!r} returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"ashby board {board_slug!r} is unreachable: {exc}") from exc
    except ValueError as exc:
        # An unknown slug answers with the bare text "Not Found", not JSON.
        raise FetchError(f"ashby board {board_slug!r} returned non-JSON") from exc
    finally:
        if owned:
            http.close()

    return parse_board(payload, company=company or board_slug, fetched_at=now())


def parse_board(payload: dict[str, Any], *, company: str, fetched_at: datetime) -> list[Job]:
    """Normalize a raw Ashby payload into Jobs. Pure — no clock, no network.

    Unlisted postings are dropped: `isListed=False` means the employer pulled it from the public
    board, so surfacing it would send Lubo to a role he cannot apply to.
    """
    try:
        raw_jobs = payload["jobs"]
        return [
            _parse_job(raw, company=company, fetched_at=fetched_at)
            for raw in raw_jobs
            if raw.get("isListed", True)
        ]
    except (KeyError, TypeError) as exc:
        raise FetchError(f"unexpected ashby payload shape: {exc}") from exc


def _parse_job(raw: dict[str, Any], *, company: str, fetched_at: datetime) -> Job:
    return Job(
        source=SOURCE,
        source_id=str(raw["id"]),
        company=company,
        title=raw["title"],
        url=raw["jobUrl"],
        location=_join_locations(raw),
        remote=_is_remote(raw),
        salary=_salary(raw),
        description=to_plain_text(raw.get("descriptionPlain") or ""),
        posted_at=_parse_published_at(raw.get("publishedAt")),
        fetched_at=fetched_at,
    )


def _is_remote(raw: dict[str, Any]) -> bool:
    """Report whether a posting is genuinely remote — from `workplaceType`, NOT `isRemote`.

    `isRemote` is unreliable: measured on the live Ramp board it was True for 119 of 126 jobs,
    including 101 that are actually Hybrid and even OnSite roles (Ashby sets it True whenever any
    location option is remote-friendly). Trusting it flagged a Hybrid-NYC role as remote and put
    it in the top tier. `workplaceType` states the real arrangement: "Remote", "Hybrid", "OnSite".

    >>> _is_remote({"workplaceType": "Remote", "isRemote": True})
    True
    >>> _is_remote({"workplaceType": "Hybrid", "isRemote": True})
    False
    >>> _is_remote({"isRemote": True})
    False
    """
    return (raw.get("workplaceType") or "").lower() == "remote"


def _join_locations(raw: dict[str, Any]) -> str | None:
    """Return the primary location plus every secondary one.

    A role listed at "New York, NY (HQ)" with a secondary of "Remote (US)" is remote-eligible;
    keeping only the primary would read as New-York-only and hide it.

    >>> _join_locations({"location": "New York, NY (HQ)",
    ...                  "secondaryLocations": [{"location": "Remote (US)"}]})
    'New York, NY (HQ), Remote (US)'
    >>> _join_locations({"location": "London", "secondaryLocations": []})
    'London'
    >>> _join_locations({}) is None
    True
    """
    primary = raw.get("location")
    secondary = [s.get("location") for s in raw.get("secondaryLocations") or []]
    parts = [p for p in [primary, *secondary] if p]
    return ", ".join(parts) if parts else None


def _salary(raw: dict[str, Any]) -> str | None:
    """Return the employer's own salary range, or None.

    This is real employer data, not an estimate — Ashby is the only source so far that carries
    it. When `shouldDisplayCompensationOnJobPostings` is False the employer opted out and no tier
    summary exists, so reading the field simply yields None; no special case is needed.

    >>> _salary({"compensation": {"scrapeableCompensationSalarySummary": "$151K - $231K"}})
    '$151K - $231K'
    >>> _salary({"compensation": {}}) is None
    True
    >>> _salary({}) is None
    True
    """
    compensation = raw.get("compensation") or {}
    return compensation.get("scrapeableCompensationSalarySummary") or None


def _parse_published_at(value: str | None) -> datetime | None:
    """Parse Ashby's `publishedAt` (clean ISO-8601 with an offset).

    >>> _parse_published_at("2026-07-07T20:47:09.753+00:00").isoformat()
    '2026-07-07T20:47:09.753000+00:00'
    >>> _parse_published_at(None) is None
    True
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
