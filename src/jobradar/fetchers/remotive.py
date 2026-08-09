"""Remotive — a remote-jobs board with a real keyword search.

    GET https://remotive.com/api/remote-jobs?search=<phrase>

No auth. Every posting is remote, so this feeds the workable digest directly — the point of the
"catch more, miss nothing" phase: a tight location filter needs a wide remote funnel.

Everything below was measured against the live API:
  * The payload is an object with a `jobs` array, alongside attribution keys (`0-legal-notice`,
    `00-warning`). Read `jobs` only. Remotive's ToS asks for a link back — honoured in the digest.
  * **`candidate_required_location` region-locks most jobs.** Measured across data/AI/backend
    searches: 52 "Brazil", plus "Mexico", "Uruguay", "Europe"-only, etc. Only postings naming
    USA / Americas / North America / Worldwide are workable for a US candidate. This field becomes
    the job's `location`, so the prefilter (extended for remote regions) drops the rest.
  * `publication_date` is a naive ISO string ("2026-07-16T10:10:51") — stamped UTC here.
  * `salary` is a free-text employer string ("$80k - $100k", "$90 - $150 /hour") — real, not a
    prediction, so it is kept as-is and never flagged estimated.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from jobradar.fetchers.base import FetchError, build_client, utcnow
from jobradar.html_text import to_plain_text
from jobradar.models import Job

SOURCE = "remotive"
SEARCH_URL = "https://remotive.com/api/remote-jobs"


def fetch_jobs(
    search: str,
    *,
    client: httpx.Client | None = None,
    now: Callable[[], datetime] = utcnow,
) -> list[Job]:
    """Search Remotive for `search` and return normalized remote Jobs.

    Raises FetchError if the board cannot be read or does not return the documented shape.
    """
    owned = client is None
    http = client or build_client()
    try:
        response = http.get(SEARCH_URL, params={"search": search})
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise FetchError(
            f"remotive search {search!r} returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"remotive is unreachable: {exc}") from exc
    except ValueError as exc:
        raise FetchError(f"remotive search {search!r} returned non-JSON") from exc
    finally:
        if owned:
            http.close()

    return parse_search(payload, fetched_at=now())


def parse_search(payload: dict[str, Any], *, fetched_at: datetime) -> list[Job]:
    """Normalize a raw Remotive payload into Jobs. Pure — no clock, no network.

    Only the `jobs` array is read; the attribution keys alongside it are ignored.
    """
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        # An error or an unexpected shape is not an empty search — fail loudly (Decision Log #8).
        raise FetchError(f"unexpected remotive payload: keys={sorted(payload)[:5]}")
    try:
        return [_parse_job(raw, fetched_at=fetched_at) for raw in jobs]
    except (KeyError, TypeError) as exc:
        raise FetchError(f"unexpected remotive job shape: {exc}") from exc


def _parse_job(raw: dict[str, Any], *, fetched_at: datetime) -> Job:
    return Job(
        source=SOURCE,
        source_id=str(raw["id"]),
        company=raw.get("company_name") or "",
        title=raw["title"],
        url=raw["url"],
        # The candidate-eligibility region IS the location; the prefilter decides US-workability.
        location=raw.get("candidate_required_location") or None,
        remote=True,
        salary=raw.get("salary") or None,
        salary_is_estimated=False,
        description=to_plain_text(raw.get("description") or ""),
        posted_at=_parse_published(raw.get("publication_date")),
        fetched_at=fetched_at,
    )


def _parse_published(value: str | None) -> datetime | None:
    """Parse Remotive's naive `publication_date` and stamp it UTC.

    >>> _parse_published("2026-07-16T10:10:51").isoformat()
    '2026-07-16T10:10:51+00:00'
    >>> _parse_published(None) is None
    True
    >>> _parse_published("not a date") is None
    True
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
