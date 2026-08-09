"""Breezy HR — public job board API.

    GET https://{company}.breezy.hr/json

No auth: a company's Breezy subdomain is all it takes, and it returns a JSON array of open
positions. Added for Dave Asprey's portfolio (The Asprey Group runs one central Breezy board
for all its brands), but it works for any company on Breezy.

DISCOVERY-STYLE, like Adzuna: the `/json` list is rich metadata (title, department, employment
type, location, salary, remote flag) but carries NO job description — that lives only in the
position page's HTML. Rather than scrape it, we score on the metadata and the reader gets the
full posting on click-through. So `description` here is SYNTHESIZED from the real fields, and
honestly labelled as such, so the scorer has the discipline signal (title + department) it needs.

Parsing is pure and IO is a thin shell around it, so the field mapping — the part that actually
breaks — is tested against a recorded board with no HTTP in the way.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx

from jobradar.fetchers.base import FetchError, build_client, utcnow
from jobradar.models import Job

SOURCE = "breezy"
BOARD_URL = "https://{company}.breezy.hr/json"


def fetch_jobs(
    company_slug: str,
    *,
    company: str | None = None,
    client: httpx.Client | None = None,
    now: Callable[[], datetime] = utcnow,
) -> list[Job]:
    """Fetch and normalize every open posting on one Breezy board.

    `company_slug` is the subdomain (`the-asprey-group`); `company` is the display name to show
    in the digest (falls back to the board's own name, then the slug). Raises FetchError if the
    board cannot be read or does not return the documented shape.
    """
    owned = client is None
    http = client or build_client()
    try:
        response = http.get(BOARD_URL.format(company=company_slug))
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise FetchError(
            f"breezy board {company_slug!r} returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"breezy board {company_slug!r} is unreachable: {exc}") from exc
    except ValueError as exc:
        raise FetchError(f"breezy board {company_slug!r} returned non-JSON") from exc
    finally:
        if owned:
            http.close()

    return parse_board(payload, company_slug=company_slug, company_name=company, fetched_at=now())


def parse_board(
    payload: list[dict[str, Any]],
    *,
    company_slug: str,
    company_name: str | None = None,
    fetched_at: datetime,
) -> list[Job]:
    """Normalize a raw Breezy board payload (a JSON list) into Jobs. Pure — no clock, no network.

    An empty board is `[]`, which is a real answer (no open roles), not an error.
    """
    if not isinstance(payload, list):
        raise FetchError(f"unexpected breezy payload shape: expected a list, got {type(payload)}")
    try:
        return [
            _parse_job(
                raw, company_slug=company_slug, company_name=company_name, fetched_at=fetched_at
            )
            for raw in payload
        ]
    except (KeyError, TypeError) as exc:
        raise FetchError(f"unexpected breezy payload shape: {exc}") from exc


def _parse_job(
    raw: dict[str, Any],
    *,
    company_slug: str,
    company_name: str | None,
    fetched_at: datetime,
) -> Job:
    location = raw.get("location") or {}
    location_name = location.get("name")
    company = company_name or (raw.get("company") or {}).get("name") or company_slug
    salary = raw.get("salary")  # employer-entered string, or absent
    return Job(
        source=SOURCE,
        source_id=str(raw["id"]),
        company=company,
        title=raw["name"],
        url=raw["url"],
        location=location_name,
        remote=bool(location.get("is_remote")) or "remote" in (location_name or "").lower(),
        salary=salary,
        # Breezy's list API has no description; synthesize one from the real metadata so the
        # scorer can judge discipline (title + department), and say so plainly. Full JD on click.
        description=_synthesise_description(raw, company=company, location=location_name),
        posted_at=_parse_timestamp(raw.get("published_date")),
        fetched_at=fetched_at,
    )


def _synthesise_description(raw: dict[str, Any], *, company: str, location: str | None) -> str:
    """Build a short, honest description from the list metadata (Breezy exposes no JD here).

    >>> _synthesise_description({"name": "DE"}, company="Acme", location=None)
    'DE at Acme. (Metadata only; full JD on the posting.)'
    """
    bits = [f"{raw['name']} at {company}."]
    department = (raw.get("department") or "").strip()
    if department:
        bits.append(f"Team/department: {department}.")
    kind = ((raw.get("type") or {}).get("name") or "").strip()
    if kind:
        bits.append(f"Employment type: {kind}.")
    if location:
        bits.append(f"Location: {location}.")
    salary = (raw.get("salary") or "").strip()
    if salary:
        bits.append(f"Listed salary: {salary}.")
    bits.append("(Metadata only; full JD on the posting.)")
    return " ".join(bits)


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse Breezy's `published_date` (ISO-8601, UTC 'Z'), keeping the offset.

    >>> _parse_timestamp("2026-07-23T17:35:10.884Z").isoformat()
    '2026-07-23T17:35:10.884000+00:00'
    >>> _parse_timestamp(None) is None
    True
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
