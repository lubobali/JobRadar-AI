"""The Greenhouse fetcher, tested against a real recorded board response.

Fixture: tests/fixtures/greenhouse_jobs.json — four real postings from
boards-api.greenhouse.io, chosen because each breaks a different naive assumption.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from jobradar.fetchers import greenhouse
from jobradar.fetchers.base import FetchError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "greenhouse_jobs.json"
FETCHED_AT = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)

REMOTE_BY_METADATA = 0  # location says "Sydney, Australia"; metadata says Remote
ONSITE_MULTI_CITY = 1  # "San Francisco, CA | New York City, NY | Seattle, WA"
REMOTE_BY_LOCATION = 2  # metadata says On-Site; location says "Remote-Friendly, United States"
NO_METADATA = 3  # metadata absent entirely


@pytest.fixture
def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def jobs(payload: dict[str, Any]) -> list[Any]:
    return greenhouse.parse_board(payload, fetched_at=FETCHED_AT)


class TestParseBoard:
    def test_parses_every_job(self, jobs: list[Any]) -> None:
        assert len(jobs) == 4

    def test_empty_board_yields_no_jobs(self) -> None:
        empty = {"jobs": [], "meta": {"total": 0}}
        assert greenhouse.parse_board(empty, fetched_at=FETCHED_AT) == []

    def test_maps_the_core_fields(self, jobs: list[Any]) -> None:
        job = jobs[REMOTE_BY_METADATA]

        assert job.source == "greenhouse"
        assert job.source_id == "5311686008"
        assert job.company == "Anthropic"
        assert job.title == "Country Lead, Data Center Security"
        assert job.url == "https://job-boards.greenhouse.io/anthropic/jobs/5311686008"
        assert job.location == "Sydney, Australia"
        assert job.fetched_at == FETCHED_AT

    def test_source_id_is_a_string_even_though_the_api_sends_an_int(self, jobs: list[Any]) -> None:
        """Greenhouse sends `id` as a JSON number; the dedup key must not vary by type."""
        assert all(isinstance(job.source_id, str) for job in jobs)

    def test_description_is_plain_text(self, jobs: list[Any]) -> None:
        description = jobs[REMOTE_BY_METADATA].description

        assert description.startswith("About Anthropic")
        assert "<" not in description
        assert "&nbsp;" not in description

    def test_salary_is_none_because_the_api_does_not_expose_it(self, jobs: list[Any]) -> None:
        """Greenhouse has no salary field — it lives in the description prose, if at all.

        Guessing one would poison the Phase 2 below-$-floor skip rule.
        """
        assert all(job.salary is None for job in jobs)


class TestPostedAt:
    """posted_at must track first_published, never updated_at.

    Every job on the recorded board shares one updated_at (a bulk board refresh) while
    first_published spans eight months. Reading updated_at would date a seven-month-old
    evergreen posting as "today" and blind the Phase 2 ghost-job filter.
    """

    def test_uses_first_published(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE_BY_METADATA].posted_at == datetime.fromisoformat(
            "2026-07-10T16:15:41-04:00"
        )

    def test_does_not_use_updated_at(self, jobs: list[Any], payload: dict[str, Any]) -> None:
        bulk_refresh = datetime.fromisoformat(payload["jobs"][REMOTE_BY_METADATA]["updated_at"])
        assert jobs[REMOTE_BY_METADATA].posted_at != bulk_refresh

    def test_preserves_a_stale_posting_s_true_age(self, jobs: list[Any]) -> None:
        stale = jobs[REMOTE_BY_LOCATION]
        assert stale.posted_at == datetime.fromisoformat("2025-12-11T12:27:23-05:00")
        assert (FETCHED_AT - stale.posted_at).days > 200

    def test_is_timezone_aware(self, jobs: list[Any]) -> None:
        assert all(job.posted_at.tzinfo is not None for job in jobs)

    def test_missing_first_published_is_none_rather_than_a_guess(self) -> None:
        raw = _minimal_raw_job()
        del raw["first_published"]

        jobs = greenhouse.parse_board({"jobs": [raw], "meta": {"total": 1}}, fetched_at=FETCHED_AT)

        assert jobs[0].posted_at is None


class TestRemoteDetection:
    """Neither signal alone is sufficient, so remote is the OR of both.

    A missed remote role is invisible forever; a false positive costs one glance. The
    fixture contains a real example of each signal firing without the other.
    """

    def test_trusts_metadata_when_the_location_name_hides_it(self, jobs: list[Any]) -> None:
        job = jobs[REMOTE_BY_METADATA]
        assert "remote" not in job.location.lower()
        assert job.remote is True

    def test_trusts_the_location_name_when_metadata_disagrees(self, jobs: list[Any]) -> None:
        job = jobs[REMOTE_BY_LOCATION]
        assert "Remote-Friendly" in job.location
        assert job.remote is True

    def test_multi_city_onsite_is_not_remote(self, jobs: list[Any]) -> None:
        assert jobs[ONSITE_MULTI_CITY].remote is False

    def test_absent_metadata_falls_back_to_the_location_name(self, jobs: list[Any]) -> None:
        assert jobs[NO_METADATA].remote is False


class TestJobId:
    def test_is_stable_across_two_fetches_of_the_same_board(self, payload: dict[str, Any]) -> None:
        first = greenhouse.parse_board(payload, fetched_at=FETCHED_AT)
        later = greenhouse.parse_board(
            payload, fetched_at=datetime(2026, 7, 15, 12, 15, tzinfo=UTC)
        )
        assert [j.job_id for j in first] == [j.job_id for j in later]

    def test_is_unique_per_posting(self, jobs: list[Any]) -> None:
        assert len({job.job_id for job in jobs}) == len(jobs)


class TestFetchJobs:
    def test_requests_the_documented_endpoint_with_full_content(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        route = respx_mock.get(
            "https://boards-api.greenhouse.io/v1/boards/anthropic/jobs",
            params={"content": "true"},
        ).mock(return_value=httpx.Response(200, json=payload))

        greenhouse.fetch_jobs("anthropic")

        assert route.called

    def test_returns_normalized_jobs(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        respx_mock.get(url__startswith="https://boards-api.greenhouse.io").mock(
            return_value=httpx.Response(200, json=payload)
        )

        jobs = greenhouse.fetch_jobs("anthropic")

        assert len(jobs) == 4
        assert jobs[0].company == "Anthropic"

    def test_stamps_fetched_at_from_the_injected_clock(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        respx_mock.get(url__startswith="https://boards-api.greenhouse.io").mock(
            return_value=httpx.Response(200, json=payload)
        )

        jobs = greenhouse.fetch_jobs("anthropic", now=lambda: FETCHED_AT)

        assert all(job.fetched_at == FETCHED_AT for job in jobs)

    @pytest.mark.parametrize("status", [404, 429, 500, 503])
    def test_http_error_raises_fetch_error_naming_the_board(
        self, respx_mock: respx.MockRouter, status: int
    ) -> None:
        """One dead board must be diagnosable, and must not be mistaken for zero jobs."""
        respx_mock.get(url__startswith="https://boards-api.greenhouse.io").mock(
            return_value=httpx.Response(status)
        )

        with pytest.raises(FetchError, match="nosuchboard"):
            greenhouse.fetch_jobs("nosuchboard")

    def test_network_failure_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__startswith="https://boards-api.greenhouse.io").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with pytest.raises(FetchError):
            greenhouse.fetch_jobs("anthropic")

    def test_malformed_payload_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__startswith="https://boards-api.greenhouse.io").mock(
            return_value=httpx.Response(200, text="<html>maintenance</html>")
        )

        with pytest.raises(FetchError):
            greenhouse.fetch_jobs("anthropic")


def _minimal_raw_job() -> dict[str, Any]:
    return {
        "id": 1,
        "title": "Data Engineer",
        "company_name": "Acme",
        "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1",
        "location": {"name": "Remote - US"},
        "first_published": "2026-07-01T10:00:00-04:00",
        "updated_at": "2026-07-14T18:35:00-04:00",
        "content": "&lt;p&gt;Build pipelines&lt;/p&gt;",
        "metadata": [],
    }
