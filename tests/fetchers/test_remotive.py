"""The Remotive fetcher, tested against a real recorded search response.

Fixture: tests/fixtures/remotive_jobs.json — four real postings chosen to cover the eligibility
cases the live probe surfaced: USA, Worldwide (both workable), Brazil (region-locked, not workable),
and "Americas, Europe, Israel" (workable via Americas, but Israel would trip the foreign check).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from jobradar.fetchers import remotive
from jobradar.fetchers.base import FetchError
from jobradar.prefilter import is_us_eligible

FIXTURE = Path(__file__).parent.parent / "fixtures" / "remotive_jobs.json"
FETCHED_AT = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)

USA, WORLDWIDE, BRAZIL, AMERICAS = 0, 1, 2, 3


@pytest.fixture
def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def jobs(payload: dict[str, Any]) -> list[Any]:
    return remotive.parse_search(payload, fetched_at=FETCHED_AT)


class TestParseSearch:
    def test_reads_the_jobs_array_and_ignores_attribution_keys(self, jobs: list[Any]) -> None:
        """The payload carries `0-legal-notice`/`job-count` beside `jobs`; only `jobs` is read."""
        assert len(jobs) == 4

    def test_a_missing_jobs_array_raises_rather_than_reading_empty(self) -> None:
        with pytest.raises(FetchError):
            remotive.parse_search({"0-legal-notice": "x"}, fetched_at=FETCHED_AT)

    def test_maps_the_core_fields(self, jobs: list[Any]) -> None:
        job = jobs[WORLDWIDE]

        assert job.source == "remotive"
        assert job.source_id == "2091068"
        assert job.company == "garden3d"
        assert job.title == "Head of Marketing & Communications"
        assert job.url.startswith("https://remotive.com/remote-jobs/")
        assert job.fetched_at == FETCHED_AT


class TestRemoteAndLocation:
    def test_every_job_is_remote(self, jobs: list[Any]) -> None:
        """Every Remotive posting is remote — that is the whole point of the source."""
        assert all(job.remote is True for job in jobs)

    def test_location_is_the_candidate_required_region(self, jobs: list[Any]) -> None:
        assert jobs[USA].location == "USA"
        assert jobs[BRAZIL].location == "Brazil"
        assert jobs[AMERICAS].location == "Americas, Europe, Israel"


class TestUsEligibilityOfTheRegion:
    """The load-bearing case: a remote job is only useful if a US worker can actually hold it."""

    def test_us_and_worldwide_and_americas_are_kept(self, jobs: list[Any]) -> None:
        assert is_us_eligible(jobs[USA].location) is True
        assert is_us_eligible(jobs[WORLDWIDE].location) is True
        # Americas includes the US, so it wins even though "Israel"/"Europe" also appear.
        assert is_us_eligible(jobs[AMERICAS].location) is True

    def test_a_region_locked_job_is_dropped(self, jobs: list[Any]) -> None:
        """Most Remotive jobs are region-locked (52 "Brazil" in the probe), so not workable."""
        assert is_us_eligible(jobs[BRAZIL].location) is False


class TestSalary:
    def test_reads_the_employer_string_and_never_flags_it_estimated(self, jobs: list[Any]) -> None:
        assert jobs[WORLDWIDE].salary == "$150k - $230k"
        assert jobs[WORLDWIDE].salary_is_estimated is False

    def test_blank_salary_is_none(self, jobs: list[Any]) -> None:
        assert jobs[USA].salary is None


class TestPostedAt:
    def test_naive_date_is_stamped_utc(self, jobs: list[Any]) -> None:
        assert jobs[AMERICAS].posted_at == datetime(2026, 7, 16, 10, 10, 51, tzinfo=UTC)

    def test_all_are_timezone_aware(self, jobs: list[Any]) -> None:
        assert all(job.posted_at.tzinfo is not None for job in jobs)


class TestDescription:
    def test_is_plain_text(self, jobs: list[Any]) -> None:
        description = jobs[BRAZIL].description
        assert "<p>" not in description
        assert "<div>" not in description


class TestFetchJobs:
    def test_requests_the_search_endpoint(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        route = respx_mock.get(
            "https://remotive.com/api/remote-jobs", params={"search": "data engineer"}
        ).mock(return_value=httpx.Response(200, json=payload))

        remotive.fetch_jobs("data engineer")

        assert route.called

    def test_returns_normalized_jobs(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        respx_mock.get(url__startswith="https://remotive.com").mock(
            return_value=httpx.Response(200, json=payload)
        )

        jobs = remotive.fetch_jobs("data engineer", now=lambda: FETCHED_AT)

        assert len(jobs) == 4
        assert all(job.remote for job in jobs)

    @pytest.mark.parametrize("status", [429, 500, 503])
    def test_http_error_raises_fetch_error(
        self, respx_mock: respx.MockRouter, status: int
    ) -> None:
        respx_mock.get(url__startswith="https://remotive.com").mock(
            return_value=httpx.Response(status)
        )

        with pytest.raises(FetchError):
            remotive.fetch_jobs("data engineer")

    def test_non_json_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__startswith="https://remotive.com").mock(
            return_value=httpx.Response(200, text="<html>down for maintenance</html>")
        )

        with pytest.raises(FetchError):
            remotive.fetch_jobs("data engineer")
