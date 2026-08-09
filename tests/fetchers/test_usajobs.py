"""The USAJobs fetcher, tested against a real recorded federal-search response.

Fixture: tests/fixtures/usajobs_jobs.json — three real postings (one remote "Anywhere in the U.S.",
two onsite/multi-location), pruned to the fields the fetcher reads (live UserArea.Details is huge).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from jobradar.fetchers import usajobs
from jobradar.fetchers.base import FetchError
from jobradar.fetchers.usajobs import UsaJobsCredentials

FIXTURE = Path(__file__).parent.parent / "fixtures" / "usajobs_jobs.json"
FETCHED_AT = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
CREDS = UsaJobsCredentials(email="me@example.com", api_key="test-key")

REMOTE, MULTI, FAA = 0, 1, 2


@pytest.fixture
def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def jobs(payload: dict[str, Any]) -> list[Any]:
    return usajobs.parse_search(payload, fetched_at=FETCHED_AT)


class TestParseSearch:
    def test_parses_every_item(self, jobs: list[Any]) -> None:
        assert len(jobs) == 3

    def test_a_missing_result_raises(self) -> None:
        with pytest.raises(FetchError):
            usajobs.parse_search({"nope": 1}, fetched_at=FETCHED_AT)

    def test_maps_core_fields(self, jobs: list[Any]) -> None:
        job = jobs[REMOTE]

        assert job.source == "usajobs"
        assert job.source_id == "PHMSA.PSRG-2026-0018"
        assert job.company == "Pipeline and Hazardous Materials Safety Administration"
        assert job.title.startswith("General Engineer")
        assert job.url.startswith("https://www.usajobs.gov")
        assert job.fetched_at == FETCHED_AT


class TestRemote:
    def test_remote_indicator_sets_the_flag(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE].remote is True

    def test_onsite_jobs_are_not_remote(self, jobs: list[Any]) -> None:
        assert jobs[MULTI].remote is False
        assert jobs[FAA].remote is False

    def test_location_is_the_display_string(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE].location == "Anywhere in the U.S. (remote job)"


class TestSalary:
    def test_formats_the_federal_pay_range_per_year(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE].salary == "$106,437 - $158,322/yr"

    def test_never_flags_the_range_estimated(self, jobs: list[Any]) -> None:
        assert all(job.salary_is_estimated is False for job in jobs)


class TestPostedAt:
    def test_naive_4_fraction_date_is_stamped_utc(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE].posted_at == datetime(2026, 7, 6, tzinfo=UTC)


class TestCredentials:
    def test_missing_credentials_raise_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("USAJOBS_API_KEY", raising=False)
        monkeypatch.delenv("USAJOBS_EMAIL", raising=False)

        with pytest.raises(FetchError, match="usajobs credentials"):
            UsaJobsCredentials.from_env()

    def test_the_api_key_is_kept_out_of_repr(self) -> None:
        assert "test-key" not in repr(CREDS)


class TestFetchJobs:
    def test_sends_the_auth_key_and_user_agent(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        route = respx_mock.get("https://data.usajobs.gov/api/search").mock(
            return_value=httpx.Response(200, json=payload)
        )

        usajobs.fetch_jobs("data engineer", credentials=CREDS)

        request = route.calls.last.request
        assert request.headers["Authorization-Key"] == "test-key"
        assert request.headers["User-Agent"] == "me@example.com"

    def test_passes_location_and_remote_filters(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        route = respx_mock.get("https://data.usajobs.gov/api/search").mock(
            return_value=httpx.Response(200, json=payload)
        )

        usajobs.fetch_jobs(
            "data engineer", credentials=CREDS, location_name="Tampa, Florida", radius=60
        )

        url = str(route.calls.last.request.url)
        assert "LocationName=Tampa" in url
        assert "Radius=60" in url

    def test_returns_normalized_jobs(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        respx_mock.get(url__startswith="https://data.usajobs.gov").mock(
            return_value=httpx.Response(200, json=payload)
        )

        jobs = usajobs.fetch_jobs("data engineer", credentials=CREDS, now=lambda: FETCHED_AT)

        assert len(jobs) == 3

    @pytest.mark.parametrize("status", [401, 429, 500])
    def test_http_error_raises_fetch_error(
        self, respx_mock: respx.MockRouter, status: int
    ) -> None:
        respx_mock.get(url__startswith="https://data.usajobs.gov").mock(
            return_value=httpx.Response(status)
        )

        with pytest.raises(FetchError):
            usajobs.fetch_jobs("data engineer", credentials=CREDS)
