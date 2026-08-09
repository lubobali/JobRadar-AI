"""The Ashby fetcher, tested against a real recorded board response.

Fixture: tests/fixtures/ashby_jobs.json — four real Ramp postings from api.ashbyhq.com.
Ashby is the richest source so far: it states workplaceType outright, publishes a clean ISO
timestamp, ships a complete descriptionPlain, and — uniquely — carries REAL employer salary.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from jobradar.fetchers import ashby
from jobradar.fetchers.base import FetchError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "ashby_jobs.json"
FETCHED_AT = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
COMPANY = "Ramp"

REMOTE_WITH_SALARY = 0  # "Remote (US)", $151K - $231K
ONSITE_NO_SALARY = 1  # London, shouldDisplayCompensation=False
REMOTE_SECONDARY = 2  # "New York, NY (HQ)" + secondary "Remote (US)"
ONSITE_NO_SALARY_2 = 3


@pytest.fixture
def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def jobs(payload: dict[str, Any]) -> list[Any]:
    return ashby.parse_board(payload, company=COMPANY, fetched_at=FETCHED_AT)


class TestParseBoard:
    def test_parses_every_job(self, jobs: list[Any]) -> None:
        assert len(jobs) == 4

    def test_empty_board_yields_no_jobs(self) -> None:
        empty = {"jobs": [], "apiVersion": "1"}
        assert ashby.parse_board(empty, company=COMPANY, fetched_at=FETCHED_AT) == []

    def test_maps_the_core_fields(self, jobs: list[Any]) -> None:
        job = jobs[REMOTE_WITH_SALARY]

        assert job.source == "ashby"
        assert job.source_id == "03e2d4e1-73ad-4f09-a058-2eb9ce34c2bc"
        assert job.company == "Ramp"
        assert job.title == "Technical Consultant, Mid-Market"
        assert job.url == "https://jobs.ashbyhq.com/ramp/03e2d4e1-73ad-4f09-a058-2eb9ce34c2bc"
        assert job.fetched_at == FETCHED_AT

    def test_company_comes_from_the_caller(self, jobs: list[Any]) -> None:
        """Like Lever, Ashby's board API never names the company."""
        assert all(job.company == "Ramp" for job in jobs)


class TestSalary:
    """Ashby is the FIRST source carrying real employer-stated salary.

    Greenhouse and Lever expose no salary field at all; Adzuna's is usually a prediction.
    Ashby's comes from the employer, so it can be trusted and shown.
    """

    def test_reads_the_employer_stated_range(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE_WITH_SALARY].salary == "$151K - $231K"

    def test_is_none_when_the_employer_opted_out_of_display(self, jobs: list[Any]) -> None:
        """shouldDisplayCompensationOnJobPostings=False -> no tier summary exists to read."""
        assert jobs[ONSITE_NO_SALARY].salary is None

    def test_missing_compensation_object_is_none_not_a_crash(self) -> None:
        raw = _minimal_raw_job()
        del raw["compensation"]

        job = ashby.parse_board({"jobs": [raw]}, company=COMPANY, fetched_at=FETCHED_AT)[0]

        assert job.salary is None


class TestPostedAt:
    def test_reads_published_at(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE_WITH_SALARY].posted_at == datetime.fromisoformat(
            "2026-07-07T20:47:09.753+00:00"
        )

    def test_is_timezone_aware(self, jobs: list[Any]) -> None:
        assert all(job.posted_at.tzinfo is not None for job in jobs)

    def test_missing_published_at_is_none_rather_than_a_guess(self) -> None:
        raw = _minimal_raw_job()
        del raw["publishedAt"]

        job = ashby.parse_board({"jobs": [raw]}, company=COMPANY, fetched_at=FETCHED_AT)[0]

        assert job.posted_at is None


class TestRemoteDetection:
    """Remote comes from `workplaceType`, NOT `isRemote`.

    `isRemote` is unreliable — on the live Ramp board it was True for 119 of 126 jobs, including
    101 Hybrid and several OnSite roles. Trusting it flagged a Hybrid-NYC job as remote and put it
    in the top tier (the bug Lubo caught opening the digest). `workplaceType` is authoritative.
    """

    def test_remote_workplace_type_is_remote(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE_WITH_SALARY].remote is True

    def test_onsite_is_not_remote(self, jobs: list[Any]) -> None:
        assert jobs[ONSITE_NO_SALARY].remote is False

    def test_a_hybrid_job_is_not_remote_even_when_isremote_is_true(self, jobs: list[Any]) -> None:
        """The exact bug: fixture job 2 is workplaceType=Hybrid at "New York, NY (HQ)" but Ashby
        reports isRemote=True. It must NOT be flagged remote."""
        job = jobs[REMOTE_SECONDARY]

        assert "New York, NY (HQ)" in job.location
        assert job.remote is False

    def test_missing_workplace_type_defaults_to_not_remote(self) -> None:
        raw = _minimal_raw_job()
        raw.pop("workplaceType", None)

        job = ashby.parse_board({"jobs": [raw]}, company=COMPANY, fetched_at=FETCHED_AT)[0]

        assert job.remote is False


class TestLocation:
    def test_secondary_locations_are_kept(self, jobs: list[Any]) -> None:
        """A role open in NY *and* remote must not read as NY-only."""
        assert jobs[REMOTE_SECONDARY].location == "New York, NY (HQ), Remote (US)"

    def test_single_location_is_unchanged(self, jobs: list[Any]) -> None:
        assert jobs[ONSITE_NO_SALARY].location == "London"


class TestDescription:
    def test_is_complete_not_just_an_intro(self, jobs: list[Any]) -> None:
        assert len(jobs[REMOTE_WITH_SALARY].description) > 2000

    def test_is_plain_text(self, jobs: list[Any]) -> None:
        description = jobs[REMOTE_WITH_SALARY].description

        assert "<p>" not in description
        assert "<h1>" not in description

    def test_normalizes_non_breaking_spaces(self, jobs: list[Any]) -> None:
        """Ashby's descriptionPlain is littered with \\xa0."""
        assert "\xa0" not in jobs[REMOTE_WITH_SALARY].description


class TestUnlistedJobs:
    def test_unlisted_jobs_are_excluded(self) -> None:
        """isListed=False means the employer pulled it from the public board."""
        listed = _minimal_raw_job()
        unlisted = _minimal_raw_job() | {"id": "unlisted-1", "isListed": False}

        jobs = ashby.parse_board(
            {"jobs": [listed, unlisted]}, company=COMPANY, fetched_at=FETCHED_AT
        )

        assert len(jobs) == 1
        assert jobs[0].source_id != "unlisted-1"


class TestJobId:
    def test_is_stable_across_two_fetches(self, payload: dict[str, Any]) -> None:
        first = ashby.parse_board(payload, company=COMPANY, fetched_at=FETCHED_AT)
        later = ashby.parse_board(
            payload, company=COMPANY, fetched_at=datetime(2026, 7, 15, 12, 15, tzinfo=UTC)
        )
        assert [j.job_id for j in first] == [j.job_id for j in later]

    def test_is_unique_per_posting(self, jobs: list[Any]) -> None:
        assert len({job.job_id for job in jobs}) == len(jobs)


class TestFetchJobs:
    def test_requests_the_documented_endpoint_with_compensation(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        route = respx_mock.get(
            "https://api.ashbyhq.com/posting-api/job-board/ramp",
            params={"includeCompensation": "true"},
        ).mock(return_value=httpx.Response(200, json=payload))

        ashby.fetch_jobs("ramp", company=COMPANY)

        assert route.called

    def test_returns_normalized_jobs(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        respx_mock.get(url__startswith="https://api.ashbyhq.com").mock(
            return_value=httpx.Response(200, json=payload)
        )

        jobs = ashby.fetch_jobs("ramp", company=COMPANY)

        assert len(jobs) == 4
        assert jobs[0].salary == "$151K - $231K"

    def test_company_defaults_to_the_slug(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        respx_mock.get(url__startswith="https://api.ashbyhq.com").mock(
            return_value=httpx.Response(200, json=payload)
        )

        assert ashby.fetch_jobs("ramp")[0].company == "ramp"

    def test_stamps_fetched_at_from_the_injected_clock(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        respx_mock.get(url__startswith="https://api.ashbyhq.com").mock(
            return_value=httpx.Response(200, json=payload)
        )

        jobs = ashby.fetch_jobs("ramp", now=lambda: FETCHED_AT)

        assert all(job.fetched_at == FETCHED_AT for job in jobs)

    @pytest.mark.parametrize("status", [404, 429, 500])
    def test_http_error_raises_fetch_error_naming_the_board(
        self, respx_mock: respx.MockRouter, status: int
    ) -> None:
        respx_mock.get(url__startswith="https://api.ashbyhq.com").mock(
            return_value=httpx.Response(status, text="Not Found")
        )

        with pytest.raises(FetchError, match="nosuchboard"):
            ashby.fetch_jobs("nosuchboard")

    def test_non_json_body_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        """A real unknown Ashby slug answers with the bare text "Not Found"."""
        respx_mock.get(url__startswith="https://api.ashbyhq.com").mock(
            return_value=httpx.Response(200, text="Not Found")
        )

        with pytest.raises(FetchError):
            ashby.fetch_jobs("nosuchboard")

    def test_network_failure_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__startswith="https://api.ashbyhq.com").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with pytest.raises(FetchError):
            ashby.fetch_jobs("ramp")


def _minimal_raw_job() -> dict[str, Any]:
    return {
        "id": "03e2d4e1-73ad-4f09-a058-2eb9ce34c2bc",
        "title": "Data Engineer",
        "location": "Remote (US)",
        "secondaryLocations": [],
        "publishedAt": "2026-07-07T20:47:09.753+00:00",
        "isListed": True,
        "isRemote": True,
        "jobUrl": "https://jobs.ashbyhq.com/acme/03e2d4e1",
        "descriptionPlain": "Build pipelines",
        "compensation": {"scrapeableCompensationSalarySummary": "$151K - $231K"},
    }
