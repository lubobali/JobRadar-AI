"""USAJobs — the U.S. federal government's official jobs API.

    GET https://data.usajobs.gov/api/search?Keyword=..&LocationName=..&Radius=..

Requires a free API key (`Authorization-Key` header) plus a `User-Agent` (your registered email).
Every posting is a federal role, so US citizenship is the norm — which is Lubo's MOAT, not a
barrier — and no other source covers this category.

Like Adzuna, it is pointed at his WORKABLE scopes (his metros + remote), because most federal jobs
are onsite at a facility and would be filtered from the digest. Measured against the live API:
  * Fully-remote federal roles (`RemoteIndicator`) are rare, but a metro search surfaces both his
    local jobs and the "Anywhere in the U.S. (remote job)" ones.
  * `PublicationStartDate` is naive with 4 fractional digits ("2025-10-01T00:00:00.0000").
  * `PositionRemuneration` is a real employer range (min/max + interval), never a prediction.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from jobradar.fetchers.base import FetchError, build_client, utcnow
from jobradar.html_text import to_plain_text
from jobradar.models import Job

SOURCE = "usajobs"
SEARCH_URL = "https://data.usajobs.gov/api/search"
_HOST = "data.usajobs.gov"


@dataclass(frozen=True, slots=True)
class UsaJobsCredentials:
    """The API key + the registered email USAJobs requires as the User-Agent.

    The key is kept out of `repr` so it cannot leak into a log line or a traceback.
    """

    email: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls) -> UsaJobsCredentials:
        key = os.environ.get("USAJOBS_API_KEY")
        email = os.environ.get("USAJOBS_EMAIL")
        if not key or not email:
            raise FetchError(
                "usajobs credentials missing: set USAJOBS_API_KEY and USAJOBS_EMAIL "
                "(free key from https://developer.usajobs.gov/apirequest/)"
            )
        return cls(email=email, api_key=key)


def fetch_jobs(  # noqa: PLR0913 — each arg is a real search dimension or an injected seam
    keyword: str,
    *,
    credentials: UsaJobsCredentials | None = None,
    location_name: str | None = None,
    radius: int | None = None,
    remote_only: bool = False,
    max_results: int = 25,
    client: httpx.Client | None = None,
    now: Callable[[], datetime] = utcnow,
) -> list[Job]:
    """Search USAJobs for `keyword`, optionally scoped to a metro or to remote-only.

    Raises FetchError if the API cannot be read or does not return the documented shape.
    """
    creds = credentials or UsaJobsCredentials.from_env()
    owned = client is None
    http = client or build_client()
    try:
        params: dict[str, str | int] = {"Keyword": keyword, "ResultsPerPage": max_results}
        if location_name:
            params["LocationName"] = location_name
        if radius is not None:
            params["Radius"] = radius
        if remote_only:
            params["RemoteIndicator"] = "True"
        response = http.get(
            SEARCH_URL,
            params=params,
            headers={
                "Authorization-Key": creds.api_key,
                "User-Agent": creds.email,
                "Host": _HOST,
            },
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise FetchError(
            f"usajobs search {keyword!r} returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"usajobs is unreachable: {exc}") from exc
    except ValueError as exc:
        raise FetchError(f"usajobs search {keyword!r} returned non-JSON") from exc
    finally:
        if owned:
            http.close()

    return parse_search(payload, fetched_at=now())


def parse_search(payload: dict[str, Any], *, fetched_at: datetime) -> list[Job]:
    """Normalize a raw USAJobs payload into Jobs. Pure — no clock, no network."""
    result = payload.get("SearchResult")
    if not isinstance(result, dict):
        raise FetchError(f"unexpected usajobs payload: keys={sorted(payload)[:5]}")
    items = result.get("SearchResultItems")
    if not isinstance(items, list):
        raise FetchError("usajobs payload has no SearchResultItems list")
    try:
        return [_parse_item(item, fetched_at=fetched_at) for item in items]
    except (KeyError, TypeError) as exc:
        raise FetchError(f"unexpected usajobs item shape: {exc}") from exc


def _parse_item(item: dict[str, Any], *, fetched_at: datetime) -> Job:
    md = item["MatchedObjectDescriptor"]
    details = md.get("UserArea", {}).get("Details", {})
    return Job(
        source=SOURCE,
        source_id=str(md["PositionID"]),
        company=md.get("OrganizationName") or "",
        title=md["PositionTitle"],
        url=md.get("PositionURI") or "",
        location=md.get("PositionLocationDisplay") or None,
        remote=bool(details.get("RemoteIndicator")),
        salary=_salary(md.get("PositionRemuneration")),
        salary_is_estimated=False,  # a real federal pay range, never a guess
        description=to_plain_text(details.get("JobSummary") or ""),
        posted_at=_parse_published(md.get("PublicationStartDate")),
        fetched_at=fetched_at,
    )


def _salary(remuneration: list[dict[str, Any]] | None) -> str | None:
    """Format the federal pay range, or None.

    >>> _salary([{"MinimumRange": "74584", "MaximumRange": "156755", "RateIntervalCode": "PA"}])
    '$74,584 - $156,755/yr'
    >>> _salary([{"MinimumRange": "50", "MaximumRange": "50", "RateIntervalCode": "PH"}])
    '$50/hr'
    >>> _salary([]) is None
    True
    """
    if not remuneration:
        return None
    row = remuneration[0]
    low, high = row.get("MinimumRange"), row.get("MaximumRange")
    if not low and not high:
        return None
    try:
        low_f, high_f = float(low or high), float(high or low)
    except (TypeError, ValueError):
        return None
    interval = {"PA": "/yr", "PH": "/hr", "PD": "/day"}.get(row.get("RateIntervalCode", ""), "")
    if low_f == high_f:
        return f"${low_f:,.0f}{interval}"
    return f"${low_f:,.0f} - ${high_f:,.0f}{interval}"


def _parse_published(value: str | None) -> datetime | None:
    """Parse USAJobs' naive `PublicationStartDate` (4 fractional digits) and stamp it UTC.

    >>> _parse_published("2025-10-01T00:00:00.0000").isoformat()
    '2025-10-01T00:00:00+00:00'
    >>> _parse_published(None) is None
    True
    >>> _parse_published("nonsense") is None
    True
    """
    if not value:
        return None
    for candidate in (value, value.split(".")[0]):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
    return None
