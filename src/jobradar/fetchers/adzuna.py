"""Adzuna — aggregator with real keyword search.

    GET https://api.adzuna.com/v1/api/jobs/us/search/{page}?app_id=..&app_key=..&what_phrase=..

The only source that can answer "who is hiring data engineers in Chicago?" without being handed
a company list first. That makes it our DISCOVERY tool: it finds companies not on the watchlist,
and their ATS board then gets watched properly. It is deliberately NOT the source of truth —
descriptions are truncated at 500 chars and two thirds of its salaries are invented.

Everything below was measured against the live API:
  * `what` is FUZZY — "data engineer" returns 91829 hits, matching "data" OR "engineer" loosely
    and surfacing "Manager of ETL Development" first. `what_phrase` returns 29378. We use the
    phrase; the fuzzy one would flood the digest.
  * `salary_is_predicted` is "1" for 34/50 live results — Adzuna guessed, always as a min==max
    point estimate ($119,026-$119,026). Never present that as fact.
  * `results_per_page` silently caps at 50: ask for 100, get 50, no error.
  * There is NO remote signal — every job is geocoded to a physical city.
  * The API intermittently answers 503 with a CloudFront HTML page.
  * `full_time=1` / `permanent=1` look useful but are traps: they cut 29378 -> 8556 / 412 because
    most postings are simply untagged, so filtering on them silently discards real jobs.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from jobradar.fetchers.base import FetchError, build_client, utcnow
from jobradar.html_text import to_plain_text
from jobradar.models import Job

SOURCE = "adzuna"
SEARCH_URL = "https://api.adzuna.com/v1/api/jobs/us/search/{page}"

_LAND_AD_PATH = re.compile(r"^/land/ad/(\d+)")

_PAGE_SIZE = 50
"""Hard cap. results_per_page=100 silently returns 50 — asking for more is a lie, not an error."""

_PREDICTED = "1"
"""salary_is_predicted: "1" means Adzuna invented the number."""


@dataclass(frozen=True, slots=True)
class AdzunaCredentials:
    """Adzuna's app_id + app_key.

    `app_key` is the secret and is kept out of `repr` so it cannot leak into a log line or a
    traceback. (`app_id` is not secret — Adzuna publishes it in every redirect url it returns.)
    """

    app_id: str
    app_key: str = field(repr=False)

    @classmethod
    def from_env(cls) -> AdzunaCredentials:
        """Load credentials from the environment, failing loudly if absent.

        Missing credentials must raise rather than yield zero jobs: a silent empty result is
        indistinguishable from "nothing was posted today", and would go unnoticed for weeks.
        """
        app_id = os.environ.get("ADZUNA_APP_ID")
        app_key = os.environ.get("ADZUNA_APP_KEY")
        if not app_id or not app_key:
            raise FetchError(
                "adzuna credentials missing: set ADZUNA_APP_ID and ADZUNA_APP_KEY "
                "(see .env.example; keys from https://developer.adzuna.com/admin/access_details)"
            )
        return cls(app_id=app_id, app_key=app_key)


def fetch_jobs(  # noqa: PLR0913 — each extra arg is a real Adzuna search dimension or a seam
    phrase: str,
    *,
    credentials: AdzunaCredentials | None = None,
    where: str | None = None,
    distance: int | None = None,
    max_days_old: int | None = None,
    max_results: int = _PAGE_SIZE,
    client: httpx.Client | None = None,
    now: Callable[[], datetime] = utcnow,
) -> list[Job]:
    """Search Adzuna for `phrase` and return normalized Jobs, freshest first.

    `phrase` is sent as `what_phrase` (an exact phrase), never as the fuzzy `what`.
    `max_days_old` is what makes a 15-minute poll cheap: ask only for what is new.
    """
    creds = credentials or AdzunaCredentials.from_env()
    owned = client is None
    http = client or build_client()
    jobs: list[Job] = []
    try:
        page = 1
        while len(jobs) < max_results:
            payload = _search_page(
                http,
                creds,
                phrase,
                page=page,
                where=where,
                distance=distance,
                max_days_old=max_days_old,
            )
            found = parse_search(payload, fetched_at=now())
            jobs.extend(found)
            if len(found) < _PAGE_SIZE:
                break  # a short page is the last page
            page += 1
        return jobs[:max_results]
    finally:
        if owned:
            http.close()


def _search_page(  # noqa: PLR0913 — mirrors the API's own query parameters
    http: httpx.Client,
    creds: AdzunaCredentials,
    phrase: str,
    *,
    page: int,
    where: str | None,
    distance: int | None,
    max_days_old: int | None,
) -> dict[str, Any]:
    params: dict[str, str | int] = {
        "app_id": creds.app_id,
        "app_key": creds.app_key,
        "what_phrase": phrase,
        # Fixed, never "just what's left". Adzuna's offset is (page-1) * results_per_page, so
        # shrinking the last page re-reads earlier results: page 2 at size 10 returns items
        # 11-20, which page 1 already gave us. Caught live — 60 jobs, only 50 unique ids.
        "results_per_page": _PAGE_SIZE,
        "sort_by": "date",  # freshest first — being early is the entire point
    }
    if where:
        params["where"] = where
    if distance is not None:
        params["distance"] = distance
    if max_days_old is not None:
        params["max_days_old"] = max_days_old

    try:
        response = http.get(SEARCH_URL.format(page=page), params=params)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise FetchError(
            f"adzuna search {phrase!r} returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"adzuna is unreachable: {exc}") from exc
    except ValueError as exc:
        # Adzuna intermittently answers with a CloudFront HTML page instead of JSON.
        raise FetchError(f"adzuna search {phrase!r} returned non-JSON") from exc


def parse_search(payload: dict[str, Any], *, fetched_at: datetime) -> list[Job]:
    """Normalize a raw Adzuna payload into Jobs. Pure — no clock, no network."""
    results = payload.get("results")
    if not isinstance(results, list):
        # Decision Log #8: validate the body. Adzuna reports auth failures in a 200-shaped
        # object with no `results` key; that is not an empty search.
        raise FetchError(f"unexpected adzuna payload: keys={sorted(payload)[:5]}")
    try:
        return [_parse_result(raw, fetched_at=fetched_at) for raw in results]
    except (KeyError, TypeError) as exc:
        raise FetchError(f"unexpected adzuna result shape: {exc}") from exc


def _parse_result(raw: dict[str, Any], *, fetched_at: datetime) -> Job:
    location = (raw.get("location") or {}).get("display_name")
    description = to_plain_text(raw.get("description") or "")
    salary, estimated = _salary(raw)
    return Job(
        source=SOURCE,
        source_id=str(raw["id"]),
        company=(raw.get("company") or {}).get("display_name") or "",
        title=raw["title"],
        # Normalized to the shape that opens. Decision Log #12: Adzuna's /land/ad/{id} redirect
        # 403s (Cloudflare bot-block), but its /details/{id} twin for the same id returns 200.
        url=_normalise_apply_url(raw["redirect_url"]),
        location=location,
        remote=_is_remote(raw["title"], location, description),
        salary=salary,
        salary_is_estimated=estimated,
        description=description,
        posted_at=_parse_created(raw.get("created")),
        fetched_at=fetched_at,
    )


def _normalise_apply_url(url: str) -> str:
    """Return an Adzuna apply url in the shape that actually opens.

    Adzuna hands back two url shapes for the same posting. `/details/{id}` returns HTTP 200;
    `/land/ad/{id}?se=...` returns 403 (Cloudflare bot-block) even with a browser User-Agent, so a
    digest built from the raw url ships dead links. Both carry the same numeric id, so a
    `/land/ad/` url is rewritten to its `/details/` twin. The `utm_*` params are kept — the
    stripped url also 403s — while the per-fetch `se=`/`v=` land-tokens are dropped. Any other
    shape is returned unchanged.

    >>> _normalise_apply_url("https://www.adzuna.com/details/123?utm_medium=api&utm_source=abc")
    'https://www.adzuna.com/details/123?utm_medium=api&utm_source=abc'
    >>> _normalise_apply_url("https://www.adzuna.com/land/ad/123?se=TOKEN&utm_medium=api&utm_source=abc&v=X")
    'https://www.adzuna.com/details/123?utm_medium=api&utm_source=abc'
    >>> _normalise_apply_url("https://example.com/whatever")
    'https://example.com/whatever'
    """
    parsed = urlparse(url)
    match = _LAND_AD_PATH.match(parsed.path)
    if not match:
        return url

    job_id = match.group(1)
    utm = urlencode([(k, v) for k, v in parse_qsl(parsed.query) if k.startswith("utm_")])
    return urlunparse((parsed.scheme, parsed.netloc, f"/details/{job_id}", "", utm, ""))


def _salary(raw: dict[str, Any]) -> tuple[str | None, bool]:
    """Return (salary, is_estimated).

    A predicted salary is kept rather than discarded — it is still a signal — but it is flagged,
    and rendered as a single number because Adzuna's guesses are min==max point estimates.
    Formatting "$119,026 - $119,026" would dress a guess up as a real range.

    >>> _salary({"salary_min": 85389.0, "salary_max": 116975.0, "salary_is_predicted": "0"})
    ('$85,389 - $116,975', False)
    >>> _salary({"salary_min": 119026.0, "salary_max": 119026.0, "salary_is_predicted": "1"})
    ('$119,026', True)
    >>> _salary({})
    (None, False)
    """
    low, high = raw.get("salary_min"), raw.get("salary_max")
    if low is None and high is None:
        return None, False

    estimated = raw.get("salary_is_predicted") == _PREDICTED
    low = low if low is not None else high
    high = high if high is not None else low
    if low == high:
        return f"${low:,.0f}", estimated
    return f"${low:,.0f} - ${high:,.0f}", estimated


def _is_remote(title: str, location: str | None, description: str) -> bool:
    """Best-effort remote detection — Adzuna gives us no flag to read.

    Measured over 50 live results: no remote/workplace field exists, and "remote" appeared in
    0/50 titles and 0/50 locations because every job is geocoded to a physical city. Only the
    text can hint, and the description is truncated at 500 chars, so this misses far more than
    it finds. Adzuna is for discovery and local search; trust Ashby/Lever for remote.

    >>> _is_remote("Data Engineer (Remote)", "Chicago, Cook County", "")
    True
    >>> _is_remote("Data Engineer", "Chicago, Cook County", "Fully remote role")
    True
    >>> _is_remote("Data Engineer", "Chicago, Cook County", "Onsite in the Loop")
    False
    """
    haystack = " ".join(filter(None, [title, location, description])).lower()
    return "remote" in haystack


def _parse_created(value: str | None) -> datetime | None:
    """Parse Adzuna's `created` ("2026-07-15T08:33:45Z").

    >>> _parse_created("2026-07-15T08:33:45Z").isoformat()
    '2026-07-15T08:33:45+00:00'
    >>> _parse_created(None) is None
    True
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
