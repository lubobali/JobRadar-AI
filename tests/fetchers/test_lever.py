"""The Lever fetcher, tested against a real recorded board response.

Fixture: tests/fixtures/lever_postings.json — four real postings from api.lever.co.
Lever differs from Greenhouse in three ways that each cost a bug if missed: the payload is a
bare list, `createdAt` is epoch MILLIseconds, and the posting body is split across several keys.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from jobradar.fetchers import lever
from jobradar.fetchers.base import FetchError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "lever_postings.json"
FETCHED_AT = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
COMPANY = "Spotify"

REMOTE = 0  # workplaceType: remote
ONSITE = 1  # workplaceType: onsite
HYBRID = 2  # workplaceType: hybrid
MULTI_LOCATION = 3  # allLocations: ["Stockholm", "London"]


@pytest.fixture
def payload() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def jobs(payload: list[dict[str, Any]]) -> list[Any]:
    return lever.parse_board(payload, company=COMPANY, fetched_at=FETCHED_AT)


class TestParseBoard:
    def test_parses_every_posting(self, jobs: list[Any]) -> None:
        assert len(jobs) == 4

    def test_empty_board_yields_no_jobs(self) -> None:
        assert lever.parse_board([], company=COMPANY, fetched_at=FETCHED_AT) == []

    def test_maps_the_core_fields(self, jobs: list[Any]) -> None:
        job = jobs[REMOTE]

        assert job.source == "lever"
        assert job.source_id == "b4ad9572-e20f-4185-a284-99d9740d04f0"
        assert job.title == "Android Engineer I - Subscriptions"
        assert job.url == "https://jobs.lever.co/spotify/b4ad9572-e20f-4185-a284-99d9740d04f0"
        assert job.location == "London, Stockholm"  # open to both; see TestLocation
        assert job.fetched_at == FETCHED_AT

    def test_company_comes_from_the_caller(self, jobs: list[Any]) -> None:
        """Lever's API never names the company — only the board slug identifies it."""
        assert all(job.company == "Spotify" for job in jobs)

    def test_salary_is_none_because_the_api_does_not_expose_it(self, jobs: list[Any]) -> None:
        assert all(job.salary is None for job in jobs)


class TestPostedAt:
    def test_reads_epoch_milliseconds_not_seconds(self, jobs: list[Any]) -> None:
        """createdAt is in MILLIseconds. Parsed as seconds every job dates to 1970-01-01,
        looks 56 years old, and the Phase 2 ghost-age filter silently drops the whole board.
        """
        assert jobs[REMOTE].posted_at == datetime(2026, 6, 30, 12, 12, 15, 482000, tzinfo=UTC)

    def test_is_not_in_1970(self, jobs: list[Any]) -> None:
        assert all(job.posted_at.year >= 2020 for job in jobs)

    def test_is_timezone_aware(self, jobs: list[Any]) -> None:
        assert all(job.posted_at.tzinfo is not None for job in jobs)

    def test_missing_created_at_is_none_rather_than_a_guess(self) -> None:
        raw = _minimal_raw_posting()
        del raw["createdAt"]

        assert lever.parse_board([raw], company=COMPANY, fetched_at=FETCHED_AT)[0].posted_at is None


class TestDescription:
    """The posting body is split across keys, and `descriptionPlain` is only the intro.

    On a real Spotify posting descriptionPlain is 991 chars while `lists` holds 3407 more —
    "What You'll Do" and "Who You Are", i.e. the actual requirements. Scoring on the intro alone
    would judge fit from marketing copy and never read the qualifications.
    """

    def test_includes_the_intro(self, jobs: list[Any]) -> None:
        assert "Support what you love" in jobs[HYBRID].description

    def test_includes_the_list_sections_that_hold_the_requirements(self, jobs: list[Any]) -> None:
        description = jobs[HYBRID].description

        assert "What You'll Do" in description
        assert "Who You Are" in description

    def test_is_longer_than_the_intro_alone(
        self, jobs: list[Any], payload: list[dict[str, Any]]
    ) -> None:
        intro = payload[HYBRID]["descriptionPlain"]
        assert len(jobs[HYBRID].description) > len(intro) * 2

    def test_is_plain_text_not_html(self, jobs: list[Any]) -> None:
        description = jobs[HYBRID].description

        assert "<li>" not in description
        assert "<div>" not in description
        assert "&nbsp;" not in description

    def test_survives_a_posting_with_no_lists(self) -> None:
        raw = _minimal_raw_posting()
        raw["lists"] = []

        job = lever.parse_board([raw], company=COMPANY, fetched_at=FETCHED_AT)[0]

        assert "Build pipelines" in job.description


class TestRemoteDetection:
    """Lever states workplaceType outright — no guessing from the location string."""

    def test_remote_is_remote(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE].remote is True

    def test_onsite_is_not_remote(self, jobs: list[Any]) -> None:
        assert jobs[ONSITE].remote is False

    def test_hybrid_is_not_remote(self, jobs: list[Any]) -> None:
        """Hybrid means office attendance, so it is not remote — but it is not excluded either."""
        assert jobs[HYBRID].remote is False

    def test_missing_workplace_type_is_not_remote(self) -> None:
        raw = _minimal_raw_posting()
        del raw["workplaceType"]

        assert lever.parse_board([raw], company=COMPANY, fetched_at=FETCHED_AT)[0].remote is False


class TestLocation:
    def test_multi_location_postings_keep_every_location(self, jobs: list[Any]) -> None:
        """allLocations carries the full list; `location` alone would hide Chicago-eligibility."""
        assert jobs[MULTI_LOCATION].location == "Stockholm, London"

    def test_single_location_is_unchanged(self, jobs: list[Any]) -> None:
        assert jobs[ONSITE].location == "New York, NY"


class TestJobId:
    def test_is_stable_across_two_fetches(self, payload: list[dict[str, Any]]) -> None:
        first = lever.parse_board(payload, company=COMPANY, fetched_at=FETCHED_AT)
        later = lever.parse_board(
            payload, company=COMPANY, fetched_at=datetime(2026, 7, 15, 12, 15, tzinfo=UTC)
        )
        assert [j.job_id for j in first] == [j.job_id for j in later]

    def test_is_unique_per_posting(self, jobs: list[Any]) -> None:
        assert len({job.job_id for job in jobs}) == len(jobs)


class TestFetchJobs:
    def test_requests_the_documented_endpoint(
        self, respx_mock: respx.MockRouter, payload: list[dict[str, Any]]
    ) -> None:
        route = respx_mock.get(
            "https://api.lever.co/v0/postings/spotify", params={"mode": "json"}
        ).mock(return_value=httpx.Response(200, json=payload))

        lever.fetch_jobs("spotify", company=COMPANY)

        assert route.called

    def test_returns_normalized_jobs(
        self, respx_mock: respx.MockRouter, payload: list[dict[str, Any]]
    ) -> None:
        respx_mock.get(url__startswith="https://api.lever.co").mock(
            return_value=httpx.Response(200, json=payload)
        )

        jobs = lever.fetch_jobs("spotify", company=COMPANY)

        assert len(jobs) == 4
        assert jobs[0].company == "Spotify"

    def test_company_defaults_to_the_slug_when_not_given(
        self, respx_mock: respx.MockRouter, payload: list[dict[str, Any]]
    ) -> None:
        respx_mock.get(url__startswith="https://api.lever.co").mock(
            return_value=httpx.Response(200, json=payload)
        )

        assert lever.fetch_jobs("spotify")[0].company == "spotify"

    def test_stamps_fetched_at_from_the_injected_clock(
        self, respx_mock: respx.MockRouter, payload: list[dict[str, Any]]
    ) -> None:
        respx_mock.get(url__startswith="https://api.lever.co").mock(
            return_value=httpx.Response(200, json=payload)
        )

        jobs = lever.fetch_jobs("spotify", company=COMPANY, now=lambda: FETCHED_AT)

        assert all(job.fetched_at == FETCHED_AT for job in jobs)

    @pytest.mark.parametrize("status", [404, 429, 500])
    def test_http_error_raises_fetch_error_naming_the_board(
        self, respx_mock: respx.MockRouter, status: int
    ) -> None:
        respx_mock.get(url__startswith="https://api.lever.co").mock(
            return_value=httpx.Response(status, json={"ok": False, "error": "Document not found"})
        )

        with pytest.raises(FetchError, match="nosuchboard"):
            lever.fetch_jobs("nosuchboard")

    def test_error_object_instead_of_a_list_raises_fetch_error(
        self, respx_mock: respx.MockRouter
    ) -> None:
        """Decision Log #8: validate the content, never the status code.

        Lever answers an unknown slug with 200 + {"ok": false} on some paths; a bare error
        object must never be mistaken for a board with no jobs.
        """
        respx_mock.get(url__startswith="https://api.lever.co").mock(
            return_value=httpx.Response(200, json={"ok": False, "error": "Document not found"})
        )

        with pytest.raises(FetchError):
            lever.fetch_jobs("nosuchboard")

    def test_network_failure_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__startswith="https://api.lever.co").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with pytest.raises(FetchError):
            lever.fetch_jobs("spotify")


def _minimal_raw_posting() -> dict[str, Any]:
    return {
        "id": "b4ad9572-e20f-4185-a284-99d9740d04f0",
        "text": "Data Engineer",
        "hostedUrl": "https://jobs.lever.co/acme/b4ad9572",
        "categories": {"location": "Remote", "allLocations": ["Remote"]},
        "workplaceType": "remote",
        "createdAt": 1782821535482,
        "descriptionPlain": "Build pipelines",
        "lists": [{"text": "Who You Are", "content": "<li>5 years Python</li>"}],
        "additionalPlain": "Equal opportunity employer",
    }
