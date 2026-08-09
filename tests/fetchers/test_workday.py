"""The Workday fetcher, tested against real recorded GDIT responses.

Fixtures: workday_search.json, workday_job_detail.json, workday_robots.txt — all real.

Workday is the odd one out and the most valuable. It is where enterprise fintech and gov
contractors live (invisible to Greenhouse/Lever/Ashby), it has real server-side search, and
it makes us work for the data:
  * the search response is thin — no description, no salary, no date;
  * `postedOn` is a human sentence ("Posted 8 Days Ago"), so the real date needs a detail call;
  * `limit` caps at 20, and one GDIT query reports 1064 hits — hydrating everything would be
    54 search calls plus 1064 detail calls, so search and hydrate must be separable;
  * req ids are unique only WITHIN a tenant, so source_id must be namespaced.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from jobradar.fetchers import workday
from jobradar.fetchers.base import FetchError

FIXTURES = Path(__file__).parent.parent / "fixtures"
SEARCH = FIXTURES / "workday_search.json"
DETAIL = FIXTURES / "workday_job_detail.json"
ROBOTS = FIXTURES / "workday_robots.txt"

FETCHED_AT = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
BOARD = workday.WorkdayBoard(
    tenant="gdit", site="External_Career_Site", host="gdit.wd5.myworkdayjobs.com"
)


@pytest.fixture
def search_payload() -> dict[str, Any]:
    return json.loads(SEARCH.read_text())


@pytest.fixture
def detail_payload() -> dict[str, Any]:
    return json.loads(DETAIL.read_text())


class TestParseSearch:
    def test_returns_a_stub_per_posting(self, search_payload: dict[str, Any]) -> None:
        assert len(workday.parse_search(search_payload)) == 4

    def test_maps_the_stub_fields(self, search_payload: dict[str, Any]) -> None:
        stub = workday.parse_search(search_payload)[0]

        assert stub.title == "Data Engineer"
        assert stub.external_path == "/job/USA-DC-Washington/Data-Engineer_RQ216675"
        assert stub.location == "USA DC Washington"
        assert stub.req_id == "RQ216675"

    def test_empty_search_yields_no_stubs(self) -> None:
        assert workday.parse_search({"total": 0, "jobPostings": []}) == []

    def test_missing_bullet_fields_leaves_req_id_none(self) -> None:
        payload = {
            "total": 1,
            "jobPostings": [
                {
                    "title": "DE",
                    "externalPath": "/job/x",
                    "locationsText": "Remote",
                    "bulletFields": [],
                }
            ],
        }
        assert workday.parse_search(payload)[0].req_id is None

    def test_malformed_payload_raises_fetch_error(self) -> None:
        with pytest.raises(FetchError):
            workday.parse_search({"errorCode": "HTTP_422"})

    def test_one_unusable_posting_does_not_lose_the_whole_board(
        self, search_payload: dict[str, Any]
    ) -> None:
        """Real GDIT data: 1 of 20 live postings is a bare stub with no title and no path.

        {"bulletFields": ["RQ222505"]} — nothing to show and nothing to fetch. Letting it raise
        would drop all 647 jobs on the board every 15 minutes because of one bad record.
        """
        search_payload["jobPostings"].insert(1, {"bulletFields": ["RQ222505"]})

        stubs = workday.parse_search(search_payload)

        assert len(stubs) == 4  # the four real postings survive; the stub is dropped
        assert all(stub.title and stub.external_path for stub in stubs)

    def test_a_posting_without_an_external_path_is_dropped(self) -> None:
        """Without externalPath there is no detail call to make and no link to send."""
        payload = {"total": 1, "jobPostings": [{"title": "Data Engineer", "bulletFields": []}]}

        assert workday.parse_search(payload) == []


class TestParseDetail:
    def test_maps_the_core_fields(self, detail_payload: dict[str, Any]) -> None:
        job = workday.parse_detail(
            detail_payload, board=BOARD, company="GDIT", fetched_at=FETCHED_AT
        )

        assert job.source == "workday"
        assert job.company == "GDIT"
        assert job.title == "Data Engineer"
        assert job.location == "USA DC Washington"
        assert job.url == (
            "https://gdit.wd5.myworkdayjobs.com/External_Career_Site"
            "/job/USA-DC-Washington/Data-Engineer_RQ216675"
        )
        assert job.fetched_at == FETCHED_AT

    def test_description_is_plain_text_from_the_html(self, detail_payload: dict[str, Any]) -> None:
        job = workday.parse_detail(
            detail_payload, board=BOARD, company="GDIT", fetched_at=FETCHED_AT
        )

        assert len(job.description) > 2000
        assert "<p>" not in job.description
        assert "Type of Requisition" in job.description

    def test_salary_is_none_because_workday_does_not_expose_it(
        self, detail_payload: dict[str, Any]
    ) -> None:
        job = workday.parse_detail(
            detail_payload, board=BOARD, company="GDIT", fetched_at=FETCHED_AT
        )
        assert job.salary is None


class TestSourceIdNamespacing:
    """Workday req ids are unique only WITHIN a tenant.

    RQ216675 at GDIT and RQ216675 at Humana are different jobs. Since job_id is
    hash(source + source_id) and `source` is "workday" for every tenant, an un-namespaced
    req id would collide the two into one id — silently hiding one of the jobs forever.
    """

    def test_source_id_includes_the_tenant(self, detail_payload: dict[str, Any]) -> None:
        job = workday.parse_detail(
            detail_payload, board=BOARD, company="GDIT", fetched_at=FETCHED_AT
        )
        assert job.source_id == "gdit:RQ216675"

    def test_same_req_id_at_two_tenants_does_not_collide(
        self, detail_payload: dict[str, Any]
    ) -> None:
        humana = workday.WorkdayBoard(
            tenant="humana", site="Humana_External_Career_Site", host="humana.wd5.myworkdayjobs.com"
        )

        at_gdit = workday.parse_detail(
            detail_payload, board=BOARD, company="GDIT", fetched_at=FETCHED_AT
        )
        at_humana = workday.parse_detail(
            detail_payload, board=humana, company="Humana", fetched_at=FETCHED_AT
        )

        assert at_gdit.job_id != at_humana.job_id


class TestPostedAt:
    """`startDate` is a real date; `postedOn` is a human sentence we deliberately ignore."""

    def test_reads_start_date(self, detail_payload: dict[str, Any]) -> None:
        job = workday.parse_detail(
            detail_payload, board=BOARD, company="GDIT", fetched_at=FETCHED_AT
        )
        assert job.posted_at == datetime(2026, 7, 7, tzinfo=UTC)

    def test_is_timezone_aware(self, detail_payload: dict[str, Any]) -> None:
        job = workday.parse_detail(
            detail_payload, board=BOARD, company="GDIT", fetched_at=FETCHED_AT
        )
        assert job.posted_at is not None
        assert job.posted_at.tzinfo is not None

    def test_ignores_the_human_readable_posted_on(self, detail_payload: dict[str, Any]) -> None:
        """ "Posted 8 Days Ago" is unparseable prose and drifts with every refresh."""
        assert "Days Ago" in detail_payload["jobPostingInfo"]["postedOn"]

        job = workday.parse_detail(
            detail_payload, board=BOARD, company="GDIT", fetched_at=FETCHED_AT
        )

        assert job.posted_at == datetime(2026, 7, 7, tzinfo=UTC)

    def test_missing_start_date_is_none_rather_than_a_guess(
        self, detail_payload: dict[str, Any]
    ) -> None:
        del detail_payload["jobPostingInfo"]["startDate"]

        job = workday.parse_detail(
            detail_payload, board=BOARD, company="GDIT", fetched_at=FETCHED_AT
        )

        assert job.posted_at is None


class TestRemoteDetection:
    def test_location_naming_remote_is_remote(self, detail_payload: dict[str, Any]) -> None:
        detail_payload["jobPostingInfo"]["location"] = "USA Remote"

        job = workday.parse_detail(
            detail_payload, board=BOARD, company="GDIT", fetched_at=FETCHED_AT
        )

        assert job.remote is True

    def test_physical_location_is_not_remote(self, detail_payload: dict[str, Any]) -> None:
        job = workday.parse_detail(
            detail_payload, board=BOARD, company="GDIT", fetched_at=FETCHED_AT
        )
        assert job.remote is False


class TestDiscoverBoard:
    """The site name is unguessable — a wrong guess returns 422, not 404.

    robots.txt names it on its Sitemap line, so we read robots.txt as intended rather than
    brute-forcing site names.
    """

    def test_reads_the_site_name_from_the_sitemap_line(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__regex=r"https://gdit\.wd[123]\.myworkdayjobs\.com/robots\.txt").mock(
            return_value=httpx.Response(404)
        )
        respx_mock.get("https://gdit.wd5.myworkdayjobs.com/robots.txt").mock(
            return_value=httpx.Response(200, text=ROBOTS.read_text())
        )

        board = workday.discover_board("gdit")

        assert board == workday.WorkdayBoard(
            tenant="gdit", site="External_Career_Site", host="gdit.wd5.myworkdayjobs.com"
        )

    def test_tries_each_workday_host_until_one_answers(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__regex=r"https://gdit\.wd[123]\.myworkdayjobs\.com/robots\.txt").mock(
            return_value=httpx.Response(404)
        )
        respx_mock.get("https://gdit.wd5.myworkdayjobs.com/robots.txt").mock(
            return_value=httpx.Response(200, text=ROBOTS.read_text())
        )

        board = workday.discover_board("gdit")

        assert board is not None
        assert board.host == "gdit.wd5.myworkdayjobs.com"

    def test_unknown_tenant_returns_none(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__regex=r".*robots\.txt").mock(return_value=httpx.Response(404))

        assert workday.discover_board("nosuchtenant") is None

    def test_robots_without_a_sitemap_line_returns_none(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__regex=r".*robots\.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nDisallow: /")
        )

        assert workday.discover_board("gdit") is None


class TestSearch:
    def test_posts_the_query_to_the_documented_endpoint(
        self, respx_mock: respx.MockRouter, search_payload: dict[str, Any]
    ) -> None:
        route = respx_mock.post(
            "https://gdit.wd5.myworkdayjobs.com/wday/cxs/gdit/External_Career_Site/jobs"
        ).mock(return_value=httpx.Response(200, json=search_payload))

        workday.search(BOARD, "data engineer")

        assert route.called
        assert json.loads(route.calls[0].request.content)["searchText"] == "data engineer"

    def test_never_requests_more_than_the_api_cap_of_20(
        self, respx_mock: respx.MockRouter, search_payload: dict[str, Any]
    ) -> None:
        """limit=50 is rejected by Workday with an error object, so the page size is fixed."""
        route = respx_mock.post(url__regex=r".*/jobs").mock(
            return_value=httpx.Response(200, json=search_payload)
        )

        workday.search(BOARD, "data engineer", max_results=100)

        assert all(json.loads(c.request.content)["limit"] <= 20 for c in route.calls)

    def test_stops_once_max_results_is_reached(
        self, respx_mock: respx.MockRouter, search_payload: dict[str, Any]
    ) -> None:
        respx_mock.post(url__regex=r".*/jobs").mock(
            return_value=httpx.Response(200, json=search_payload)
        )

        assert len(workday.search(BOARD, "data engineer", max_results=2)) == 2

    def test_http_error_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(url__regex=r".*/jobs").mock(return_value=httpx.Response(422))

        with pytest.raises(FetchError, match="gdit"):
            workday.search(BOARD, "data engineer")


class TestFetchJobs:
    def test_searches_then_hydrates_only_the_stubs_it_keeps(
        self,
        respx_mock: respx.MockRouter,
        search_payload: dict[str, Any],
        detail_payload: dict[str, Any],
    ) -> None:
        """The whole point of the two-stage design: 1064 hits must not mean 1064 detail calls."""
        respx_mock.post(url__regex=r".*/jobs").mock(
            return_value=httpx.Response(200, json=search_payload)
        )
        detail = respx_mock.get(url__regex=r".*/job/.*").mock(
            return_value=httpx.Response(200, json=detail_payload)
        )

        jobs = workday.fetch_jobs(BOARD, "data engineer", company="GDIT", max_results=2)

        assert len(jobs) == 2
        assert detail.call_count == 2

    def test_a_single_dead_detail_call_does_not_lose_the_whole_board(
        self,
        respx_mock: respx.MockRouter,
        search_payload: dict[str, Any],
        detail_payload: dict[str, Any],
    ) -> None:
        respx_mock.post(url__regex=r".*/jobs").mock(
            return_value=httpx.Response(200, json=search_payload)
        )
        respx_mock.get(url__regex=r".*Data-Engineer_RQ216675").mock(
            return_value=httpx.Response(500)
        )
        respx_mock.get(url__regex=r".*/job/.*").mock(
            return_value=httpx.Response(200, json=detail_payload)
        )

        jobs = workday.fetch_jobs(BOARD, "data engineer", company="GDIT", max_results=4)

        assert len(jobs) == 3  # the dead one is skipped, the rest survive
