"""Workday — enterprise ATS, reached through the JSON API behind its public career sites.

    POST https://{tenant}.{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs   (search)
    GET  https://{tenant}.{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}{path}  (detail)

Why this module exists: probing 20 real employers found only 4 on Greenhouse/Lever/Ashby. Every
large one — Capital One, Citigroup, J.P. Morgan, Humana, GDIT, ManTech — is on an enterprise ATS.
Workday is where the fintech and gov-contractor roles actually live, so skipping it means missing
most of the market.

robots.txt permits this: `/wday/cxs/` is not disallowed and the career site is explicitly
`Allow`ed (only `/talentcommunity/` and `/refreshFacet/` are barred).

Workday makes us work for the data, and the shape of this module is a consequence:
  * search is thin — no description, no salary, no real date;
  * `postedOn` is prose ("Posted 8 Days Ago"); the real date is `startDate` on the detail call;
  * `limit` caps at 20 and one GDIT query reports 1064 hits, so hydrating everything would cost
    54 search calls plus 1064 detail calls. search() and hydrate() are therefore separate, and
    the caller prefilters cheap stubs before paying for detail.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from jobradar.fetchers.base import FetchError, build_client, utcnow
from jobradar.html_text import to_plain_text
from jobradar.models import Job

SOURCE = "workday"

_HOSTS = ("wd1", "wd2", "wd3", "wd5")
"""Workday shards tenants across numbered hosts; which one a company is on is not derivable."""

_PAGE_SIZE = 20
"""Hard API cap. limit=50 is rejected with an error object, not a truncated page."""

_SITEMAP_LINE = re.compile(r"^Sitemap:\s*https?://[^/]+/([^/]+)/", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True, slots=True)
class WorkdayBoard:
    """Everything needed to address one company's Workday board."""

    tenant: str
    site: str
    host: str

    @property
    def api_root(self) -> str:
        return f"https://{self.host}/wday/cxs/{self.tenant}/{self.site}"

    @property
    def site_root(self) -> str:
        return f"https://{self.host}/{self.site}"


@dataclass(frozen=True, slots=True)
class JobStub:
    """A search hit. Cheap: no description, no date, no salary — hydrate() fills those in."""

    title: str
    external_path: str
    location: str
    req_id: str | None


def discover_board(tenant: str, *, client: httpx.Client | None = None) -> WorkdayBoard | None:
    """Find a tenant's board by reading its robots.txt, or None if it has no Workday site.

    The `{site}` segment is unguessable — "External", "External_Career_Site" and
    "NVIDIAExternalCareerSite" are all real, and a wrong guess returns HTTP 422 rather than a
    404 that would tell us we were close. robots.txt names it outright on the Sitemap line, so
    we read the file that exists to be read instead of brute-forcing.
    """
    owned = client is None
    http = client or build_client()
    try:
        for host_shard in _HOSTS:
            host = f"{tenant}.{host_shard}.myworkdayjobs.com"
            try:
                response = http.get(f"https://{host}/robots.txt")
            except httpx.HTTPError:
                continue
            if response.status_code != httpx.codes.OK:
                continue
            match = _SITEMAP_LINE.search(response.text)
            if match:
                return WorkdayBoard(tenant=tenant, site=match.group(1), host=host)
        return None
    finally:
        if owned:
            http.close()


def search(
    board: WorkdayBoard,
    query: str,
    *,
    max_results: int = 100,
    client: httpx.Client | None = None,
) -> list[JobStub]:
    """Return up to `max_results` search hits as cheap stubs.

    Workday's `searchText` is FUZZY — a query for "data engineer" also returns anything loosely
    related, and one GDIT query reports 1064 hits. Treat the count as "this board is worth
    watching", never as a job count, and filter properly downstream.
    """
    owned = client is None
    http = client or build_client()
    stubs: list[JobStub] = []
    try:
        while len(stubs) < max_results:
            page = _search_page(http, board, query, offset=len(stubs))
            if not page:
                break
            stubs.extend(page)
        return stubs[:max_results]
    finally:
        if owned:
            http.close()


def _search_page(
    http: httpx.Client, board: WorkdayBoard, query: str, *, offset: int
) -> list[JobStub]:
    try:
        response = http.post(
            f"{board.api_root}/jobs",
            json={"appliedFacets": {}, "limit": _PAGE_SIZE, "offset": offset, "searchText": query},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise FetchError(
            f"workday board {board.tenant!r} returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"workday board {board.tenant!r} is unreachable: {exc}") from exc
    except ValueError as exc:
        raise FetchError(f"workday board {board.tenant!r} returned non-JSON") from exc

    return parse_search(payload)


def parse_search(payload: dict[str, Any]) -> list[JobStub]:
    """Normalize a raw search payload into stubs, dropping any posting we cannot act on.

    Pure — no clock, no network.

    Real GDIT data contains bare stubs: 1 of 20 live postings was `{"bulletFields": ["RQ222505"]}`
    with no title and no externalPath — nothing to display and nothing to fetch. Raising on it
    would throw away all 647 jobs on the board every 15 minutes over one bad record, so an
    unusable posting is skipped while the rest survive.
    """
    postings = payload.get("jobPostings")
    if not isinstance(postings, list):
        # Decision Log #8: validate the body. A rejected query answers 200-shaped error objects
        # with no jobPostings key; that is not an empty board.
        raise FetchError(f"unexpected workday search payload: keys={sorted(payload)[:5]}")
    return [stub for p in postings if (stub := _parse_stub(p)) is not None]


def _parse_stub(raw: dict[str, Any]) -> JobStub | None:
    """Return a JobStub, or None if the posting lacks the fields that make it actionable."""
    title, path = raw.get("title"), raw.get("externalPath")
    if not title or not path:
        return None
    return JobStub(
        title=title,
        external_path=path,
        location=raw.get("locationsText") or "",
        req_id=next(iter(raw.get("bulletFields") or []), None),
    )


def hydrate(
    board: WorkdayBoard,
    stub: JobStub,
    *,
    company: str,
    fetched_at: datetime,
    client: httpx.Client | None = None,
) -> Job:
    """Fetch one job's detail and normalize it. One HTTP call — spend these deliberately."""
    owned = client is None
    http = client or build_client()
    try:
        response = http.get(
            f"{board.api_root}{stub.external_path}", headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise FetchError(
            f"workday job {stub.external_path!r} returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"workday job {stub.external_path!r} is unreachable: {exc}") from exc
    except ValueError as exc:
        raise FetchError(f"workday job {stub.external_path!r} returned non-JSON") from exc
    finally:
        if owned:
            http.close()

    return parse_detail(payload, board=board, company=company, fetched_at=fetched_at)


def parse_detail(
    payload: dict[str, Any], *, board: WorkdayBoard, company: str, fetched_at: datetime
) -> Job:
    """Normalize a raw detail payload into a Job. Pure — no clock, no network."""
    try:
        info = payload["jobPostingInfo"]
    except (KeyError, TypeError) as exc:
        raise FetchError(f"unexpected workday detail payload: {exc}") from exc

    location = info.get("location") or ""
    return Job(
        source=SOURCE,
        source_id=_source_id(board.tenant, info),
        company=company,
        title=info["title"],
        url=info.get("externalUrl") or f"{board.site_root}{info.get('externalPath', '')}",
        location=location or None,
        remote="remote" in location.lower(),
        salary=None,  # Workday exposes no salary field.
        description=to_plain_text(info.get("jobDescription") or ""),
        posted_at=_parse_start_date(info.get("startDate")),
        fetched_at=fetched_at,
    )


def _source_id(tenant: str, info: dict[str, Any]) -> str:
    """Return a source id namespaced by tenant.

    Workday requisition ids are unique only WITHIN a tenant: RQ216675 exists at GDIT and could
    equally exist at Humana. Since job_id is hash(source + source_id) and `source` is "workday"
    for every tenant, a bare req id would collide two different jobs into one id and silently
    hide one of them forever.

    >>> _source_id("gdit", {"jobReqId": "RQ216675"})
    'gdit:RQ216675'
    """
    req_id = info.get("jobReqId") or info.get("jobPostingId") or info.get("id")
    return f"{tenant}:{req_id}"


def _parse_start_date(value: str | None) -> datetime | None:
    """Parse Workday's `startDate` ("2026-07-07"), the only real date it gives us.

    The search response's `postedOn` is prose — "Posted 8 Days Ago" — which is unparseable,
    imprecise, and drifts every time the board refreshes. `startDate` is a date with no time or
    zone, so it is anchored to UTC midnight; a few hours of slop is irrelevant to a ghost-job
    age filter measured in days.

    >>> _parse_start_date("2026-07-07").isoformat()
    '2026-07-07T00:00:00+00:00'
    >>> _parse_start_date(None) is None
    True
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError:
        return None


def fetch_jobs(  # noqa: PLR0913 — every extra arg is an injected seam (clock, client, filter)
    board: WorkdayBoard,
    query: str,
    *,
    company: str | None = None,
    max_results: int = 20,
    keep: Callable[[JobStub], bool] | None = None,
    now: Callable[[], datetime] = utcnow,
    client: httpx.Client | None = None,
) -> list[Job]:
    """Search a board and hydrate the hits, skipping any job whose detail call fails.

    `keep` filters stubs BEFORE the expensive detail calls — that is the entire point of the
    two-stage design. `max_results` bounds the search; one dead job must not lose the board, so
    a failed detail call drops that job and the rest survive.
    """
    owned = client is None
    http = client or build_client()
    try:
        stubs = search(board, query, max_results=max_results, client=http)
        wanted = [stub for stub in stubs if keep is None or keep(stub)]
        fetched_at = now()
        jobs = []
        for stub in wanted:
            try:
                jobs.append(
                    hydrate(
                        board,
                        stub,
                        company=company or board.tenant,
                        fetched_at=fetched_at,
                        client=http,
                    )
                )
            except FetchError:
                continue
        return jobs
    finally:
        if owned:
            http.close()
