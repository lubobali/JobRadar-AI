"""The Breezy fetcher, tested against a recorded board shape.

Fixture: tests/fixtures/breezy_board.json — the exact shape the live `/json` endpoint returns
(probed against the-asprey-group.breezy.hr and breezy-hr.breezy.hr). Key facts measured live:
  * the payload is a JSON LIST of positions, not an object;
  * an open-role-free board is `[]` (a real answer, not an error);
  * there is NO description field in the list — only rich metadata;
  * `location.is_remote` is the remote signal; `published_date` is ISO-8601 UTC ('Z').
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from jobradar.fetchers import breezy
from jobradar.fetchers.base import FetchError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "breezy_board.json"
FETCHED_AT = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
SLUG = "the-asprey-group"

REMOTE_ENG = 0  # Senior Data Engineer, is_remote=true, has salary + department
ONSITE = 1  # Biohacker Technician, onsite LA, no salary, no department


@pytest.fixture
def payload() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def jobs(payload: list[dict[str, Any]]) -> list[Any]:
    return breezy.parse_board(payload, company_slug=SLUG, company_name="The Asprey Group",
                              fetched_at=FETCHED_AT)


class TestParseBoard:
    def test_parses_every_position(self, jobs: list[Any]) -> None:
        assert len(jobs) == 2

    def test_an_empty_board_is_no_jobs_not_an_error(self) -> None:
        assert breezy.parse_board([], company_slug=SLUG, fetched_at=FETCHED_AT) == []

    def test_maps_the_core_fields(self, jobs: list[Any]) -> None:
        job = jobs[REMOTE_ENG]

        assert job.source == "breezy"
        assert job.source_id == "abc123def456"
        assert job.title == "Senior Data Engineer"
        assert job.url.endswith("/p/abc123def456-senior-data-engineer")
        assert job.location == "Remote, United States"
        assert job.fetched_at == FETCHED_AT

    def test_display_company_name_overrides_the_boards_own(self, jobs: list[Any]) -> None:
        """The passed-in name is the reliable label; the board's own can be an ATS account name."""
        assert jobs[REMOTE_ENG].company == "The Asprey Group"

    def test_falls_back_to_the_boards_company_when_no_name_given(
        self, payload: list[dict[str, Any]]
    ) -> None:
        jobs = breezy.parse_board(payload, company_slug=SLUG, fetched_at=FETCHED_AT)

        assert jobs[ONSITE].company == "Upgrade Labs"

    def test_posted_at_reads_published_date_utc(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE_ENG].posted_at == datetime(2026, 7, 25, 14, 0, tzinfo=UTC)
        assert jobs[REMOTE_ENG].posted_at.tzinfo is not None

    def test_a_non_list_payload_raises_fetch_error(self) -> None:
        """The live API can only fail into HTML/an object; a dict is not a board."""
        with pytest.raises(FetchError):
            breezy.parse_board({"error": "nope"}, company_slug=SLUG, fetched_at=FETCHED_AT)  # type: ignore[arg-type]

    def test_a_position_missing_a_required_field_raises(self) -> None:
        with pytest.raises(FetchError):
            breezy.parse_board([{"id": "x"}], company_slug=SLUG, fetched_at=FETCHED_AT)


class TestRemote:
    def test_is_remote_flag_is_believed(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE_ENG].remote is True

    def test_an_onsite_role_is_not_remote(self, jobs: list[Any]) -> None:
        assert jobs[ONSITE].remote is False

    def test_remote_in_the_location_name_is_a_fallback_signal(self) -> None:
        raw = _minimal_raw() | {"location": {"is_remote": False, "name": "Remote (US)"}}

        job = breezy.parse_board([raw], company_slug=SLUG, fetched_at=FETCHED_AT)[0]

        assert job.remote is True


class TestSalary:
    def test_keeps_the_employer_stated_salary_as_a_fact(self, jobs: list[Any]) -> None:
        assert jobs[REMOTE_ENG].salary == "$140,000 – $170,000 / year"  # noqa: RUF001 — real Breezy en-dash
        assert jobs[REMOTE_ENG].salary_is_estimated is False

    def test_missing_salary_is_none(self, jobs: list[Any]) -> None:
        assert jobs[ONSITE].salary is None


class TestDescription:
    """Breezy's list has no JD, so we synthesize an honest one from the metadata."""

    def test_names_the_role_and_company(self, jobs: list[Any]) -> None:
        desc = jobs[REMOTE_ENG].description
        assert "Senior Data Engineer" in desc
        assert "The Asprey Group" in desc

    def test_carries_the_department_so_the_scorer_can_judge_discipline(
        self, jobs: list[Any]
    ) -> None:
        """Department is the strongest engineering signal when there is no JD."""
        assert "Engineering" in jobs[REMOTE_ENG].description

    def test_is_labelled_as_metadata_not_a_real_jd(self, jobs: list[Any]) -> None:
        """Never let the scorer (or a reader) mistake a synthesized blurb for the full posting."""
        assert "metadata" in jobs[REMOTE_ENG].description.lower()


class TestFetchJobs:
    def test_hits_the_boards_json_endpoint(
        self, respx_mock: respx.MockRouter, payload: list[dict[str, Any]]
    ) -> None:
        route = respx_mock.get("https://the-asprey-group.breezy.hr/json").mock(
            return_value=httpx.Response(200, json=payload)
        )

        breezy.fetch_jobs(SLUG, company="The Asprey Group")

        assert route.called

    def test_passes_the_display_company_through(
        self, respx_mock: respx.MockRouter, payload: list[dict[str, Any]]
    ) -> None:
        respx_mock.get(url__startswith="https://the-asprey-group.breezy.hr").mock(
            return_value=httpx.Response(200, json=payload)
        )

        jobs = breezy.fetch_jobs(SLUG, company="The Asprey Group")

        assert all(j.company == "The Asprey Group" for j in jobs)

    def test_an_empty_board_returns_no_jobs(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__startswith="https://the-asprey-group.breezy.hr").mock(
            return_value=httpx.Response(200, json=[])
        )

        assert breezy.fetch_jobs(SLUG) == []

    def test_http_error_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__startswith="https://the-asprey-group.breezy.hr").mock(
            return_value=httpx.Response(404)
        )

        with pytest.raises(FetchError):
            breezy.fetch_jobs(SLUG)

    def test_a_200_html_body_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__startswith="https://the-asprey-group.breezy.hr").mock(
            return_value=httpx.Response(200, text="<!DOCTYPE html>")
        )

        with pytest.raises(FetchError):
            breezy.fetch_jobs(SLUG)

    def test_network_failure_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__startswith="https://the-asprey-group.breezy.hr").mock(
            side_effect=httpx.ConnectError("refused")
        )

        with pytest.raises(FetchError):
            breezy.fetch_jobs(SLUG)

    def test_stamps_fetched_at_from_the_injected_clock(
        self, respx_mock: respx.MockRouter, payload: list[dict[str, Any]]
    ) -> None:
        respx_mock.get(url__startswith="https://the-asprey-group.breezy.hr").mock(
            return_value=httpx.Response(200, json=payload)
        )

        jobs = breezy.fetch_jobs(SLUG, now=lambda: FETCHED_AT)

        assert all(job.fetched_at == FETCHED_AT for job in jobs)


def _minimal_raw() -> dict[str, Any]:
    return {
        "id": "min-1",
        "name": "Data Engineer",
        "url": "https://the-asprey-group.breezy.hr/p/min-1-data-engineer",
        "published_date": "2026-07-25T14:00:00.000Z",
        "type": {"id": "full-time", "name": "Full-Time"},
        "location": {"is_remote": False, "name": "Tampa, FL"},
        "department": "Engineering",
        "salary": None,
        "company": {"name": "The Asprey Group"},
    }
