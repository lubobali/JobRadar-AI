"""The Adzuna fetcher, tested against a real recorded search response.

Fixture: tests/fixtures/adzuna_search.json — four real results from api.adzuna.com.

Adzuna is the odd source: the only one with real keyword search, and the only one that will
hand back a salary it invented. Everything here is measured against the live API, not assumed:
  * `what` is FUZZY (91829 hits) — `what_phrase` is precise (29378);
  * `salary_is_predicted` is "1" for 34/50 live results, always as a min==max point estimate;
  * descriptions are truncated at exactly 500 chars;
  * there is NO remote signal at all — every job is geocoded to a physical city;
  * `results_per_page` silently caps at 50;
  * the API intermittently answers 503 with an HTML body.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from jobradar.fetchers import adzuna
from jobradar.fetchers.base import FetchError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "adzuna_search.json"
FETCHED_AT = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
CREDS = adzuna.AdzunaCredentials(app_id="testid", app_key="testkey")

REAL_SALARY = 0  # salary_is_predicted="0", $85,389-$116,975
PREDICTED_SALARY = 1  # salary_is_predicted="1", min==max
LAND_AD_URL = 2  # redirect_url shape /land/ad/{id}?se=...
DETAILS_URL = 3


@pytest.fixture
def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def jobs(payload: dict[str, Any]) -> list[Any]:
    return adzuna.parse_search(payload, fetched_at=FETCHED_AT)


class TestParseSearch:
    def test_parses_every_result(self, jobs: list[Any]) -> None:
        assert len(jobs) == 4

    def test_empty_search_yields_no_jobs(self) -> None:
        assert adzuna.parse_search({"count": 0, "results": []}, fetched_at=FETCHED_AT) == []

    def test_maps_the_core_fields(self, jobs: list[Any]) -> None:
        job = jobs[REAL_SALARY]

        assert job.source == "adzuna"
        assert job.source_id == "5801147013"
        assert job.company == "SSG"
        assert job.title == "Senior Software Developer"
        assert job.location == "State Farm, Arlington County"
        assert job.fetched_at == FETCHED_AT

    def test_posted_at_reads_created(self, jobs: list[Any]) -> None:
        assert jobs[REAL_SALARY].posted_at == datetime(2026, 7, 15, 8, 33, 45, tzinfo=UTC)

    def test_posted_at_is_timezone_aware(self, jobs: list[Any]) -> None:
        assert all(job.posted_at.tzinfo is not None for job in jobs)

    def test_missing_created_is_none_rather_than_a_guess(self) -> None:
        raw = _minimal_raw_result()
        del raw["created"]

        job = adzuna.parse_search({"results": [raw]}, fetched_at=FETCHED_AT)[0]

        assert job.posted_at is None

    def test_malformed_payload_raises_fetch_error(self) -> None:
        with pytest.raises(FetchError):
            adzuna.parse_search({"exception": "AUTH_FAIL"}, fetched_at=FETCHED_AT)


class TestSalary:
    """Adzuna invents a salary for ~2/3 of results and presents it exactly like a real one.

    Measured live: 34/50 had salary_is_predicted="1", every one a min==max point estimate
    (e.g. $119,026-$119,026). Phase 2 must never treat a guess as fact — hence the flag.
    """

    def test_employer_stated_salary_is_kept_and_trusted(self, jobs: list[Any]) -> None:
        job = jobs[REAL_SALARY]

        assert job.salary == "$85,389 - $116,975"
        assert job.salary_is_estimated is False

    def test_predicted_salary_is_kept_but_flagged(self, jobs: list[Any]) -> None:
        """Kept, not dropped: it is still a signal — as long as it is never shown as fact."""
        job = jobs[PREDICTED_SALARY]

        assert job.salary == "$119,026"
        assert job.salary_is_estimated is True

    def test_a_predicted_point_estimate_is_not_rendered_as_a_fake_range(
        self, jobs: list[Any]
    ) -> None:
        """min==max is not a range. "$119,026 - $119,026" would look like real data."""
        assert " - " not in jobs[PREDICTED_SALARY].salary

    def test_missing_salary_is_none(self) -> None:
        raw = _minimal_raw_result()
        del raw["salary_min"]
        del raw["salary_max"]

        job = adzuna.parse_search({"results": [raw]}, fetched_at=FETCHED_AT)[0]

        assert job.salary is None
        assert job.salary_is_estimated is False


class TestUrl:
    """Normalize the apply URL to the shape that actually opens.

    Decision Log #12, measured live: Adzuna returns two url shapes for the same job. The
    `/details/{id}` shape returns HTTP 200; the `/land/ad/{id}?se=...` shape returns 403
    (Cloudflare bot-block) even with a browser User-Agent. Both carry the same numeric id, so a
    `/land/ad/` url is rewritten to the working `/details/` url — otherwise the digest ships dead
    links. The utm_* params are kept (the stripped url also 403s); the per-fetch `se=`/`v=`
    land-tokens are dropped. The url carries app_id (public, Adzuna's own attribution) but never
    app_key.
    """

    def test_keeps_a_working_details_url_as_is(self, jobs: list[Any]) -> None:
        assert jobs[DETAILS_URL].url.startswith("https://www.adzuna.com/details/5801171816")
        assert "utm_medium=api" in jobs[DETAILS_URL].url

    def test_rewrites_the_blocked_land_ad_shape_to_details(self, jobs: list[Any]) -> None:
        """The /land/ad/ shape 403s; rewrite it to the /details/ twin, which opens."""
        url = jobs[LAND_AD_URL].url

        assert url.startswith("https://www.adzuna.com/details/5801264268")
        assert "/land/ad/" not in url
        assert "se=" not in url  # the token that 403s and changes per fetch is gone

    def test_a_rewritten_url_keeps_the_utm_attribution(self, jobs: list[Any]) -> None:
        """utm_* is required for the /details/ url to resolve; keep it."""
        assert "utm_medium=api" in jobs[LAND_AD_URL].url

    def test_job_id_ignores_the_volatile_url(self, payload: dict[str, Any]) -> None:
        first = adzuna.parse_search(payload, fetched_at=FETCHED_AT)
        payload["results"][LAND_AD_URL]["redirect_url"] = (
            "https://www.adzuna.com/land/ad/5801264268?se=TOTALLY_DIFFERENT_TOKEN"
        )
        later = adzuna.parse_search(payload, fetched_at=FETCHED_AT)

        assert first[LAND_AD_URL].job_id == later[LAND_AD_URL].job_id


class TestRemote:
    """Adzuna has NO remote field, and geocodes every job to a physical city.

    Measured over 50 live results: "remote" appeared in 0/50 titles and 0/50 location names.
    Only the (500-char-truncated) description ever says so, which is why this is best-effort:
    it misses far more than it finds. Adzuna is for discovery and local search — trust
    Ashby/Lever for remote.
    """

    def test_adzunas_location_field_hides_genuinely_remote_jobs(self, jobs: list[Any]) -> None:
        """A real case from the fixture, and the reason we read the text at all.

        This VA role's description says "Location: Remote", yet Adzuna geocoded it to
        "State Farm, Arlington County". Trusting the location field alone would file a remote
        job as Virginia-only and bury it.
        """
        job = jobs[REAL_SALARY]

        assert job.location == "State Farm, Arlington County"
        assert "Location: Remote" in job.description
        assert job.remote is True

    def test_a_job_with_no_remote_mention_anywhere_is_not_remote(self, jobs: list[Any]) -> None:
        assert jobs[PREDICTED_SALARY].remote is False

    def test_remote_in_the_title_is_detected(self) -> None:
        raw = _minimal_raw_result() | {"title": "Data Engineer (Remote)"}

        job = adzuna.parse_search({"results": [raw]}, fetched_at=FETCHED_AT)[0]

        assert job.remote is True


class TestDescription:
    def test_is_plain_text(self, jobs: list[Any]) -> None:
        assert "<p>" not in jobs[REAL_SALARY].description

    def test_truncation_is_left_visible_rather_than_hidden(self, jobs: list[Any]) -> None:
        """Adzuna truncates at 500 chars. Phase 2 must know it is scoring a teaser, not a JD."""
        assert jobs[REAL_SALARY].description.endswith("…")


class TestFetchJobs:
    def test_uses_what_phrase_for_precision_not_the_fuzzy_what(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        """Measured: what=data engineer -> 91829 hits (matches "data" OR "engineer" loosely);
        what_phrase=data engineer -> 29378. `what` would flood the digest with noise.
        """
        route = respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(200, json=payload)
        )

        adzuna.fetch_jobs("data engineer", credentials=CREDS)

        params = route.calls[0].request.url.params
        assert params["what_phrase"] == "data engineer"
        assert "what" not in params

    def test_sorts_by_date_so_the_freshest_jobs_arrive_first(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        """Being early is the whole point; relevance ordering buries today's postings."""
        route = respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(200, json=payload)
        )

        adzuna.fetch_jobs("data engineer", credentials=CREDS)

        assert route.calls[0].request.url.params["sort_by"] == "date"

    def test_sends_credentials(self, respx_mock: respx.MockRouter, payload: dict[str, Any]) -> None:
        route = respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(200, json=payload)
        )

        adzuna.fetch_jobs("data engineer", credentials=CREDS)

        params = route.calls[0].request.url.params
        assert params["app_id"] == "testid"
        assert params["app_key"] == "testkey"

    def test_passes_location_and_freshness_filters(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        route = respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(200, json=payload)
        )

        adzuna.fetch_jobs(
            "data engineer", credentials=CREDS, where="Chicago", distance=50, max_days_old=1
        )

        params = route.calls[0].request.url.params
        assert params["where"] == "Chicago"
        assert params["distance"] == "50"
        assert params["max_days_old"] == "1"

    def test_never_requests_more_than_the_silent_page_cap_of_50(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        """results_per_page=100 silently returns 50 with no error, so asking is a lie."""
        route = respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(200, json=payload)
        )

        adzuna.fetch_jobs("data engineer", credentials=CREDS, max_results=500)

        assert all(int(c.request.url.params["results_per_page"]) <= 50 for c in route.calls)

    def test_page_size_stays_constant_across_pages(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        """Adzuna's offset is (page-1) * results_per_page, so the page size must not shrink.

        Sizing the last page to "just what's left" (min(50, remaining)) re-reads earlier
        results: page 2 with results_per_page=10 returns items 11-20, which page 1 already
        returned. Caught live — a 60-job fetch yielded only 50 unique ids.
        """
        route = respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(200, json=payload)
        )

        adzuna.fetch_jobs("data engineer", credentials=CREDS, max_results=6)

        sizes = {c.request.url.params["results_per_page"] for c in route.calls}
        assert len(sizes) == 1, f"page size changed across pages: {sizes}"

    def test_trims_to_max_results_after_fetching_whole_pages(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        """Whole pages are fetched, then trimmed — the page size is never bent to fit."""
        respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(200, json=_full_page(payload))
        )

        assert len(adzuna.fetch_jobs("data engineer", credentials=CREDS, max_results=6)) == 6

    def test_paginates_until_max_results(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        route = respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(200, json=_full_page(payload))
        )

        jobs = adzuna.fetch_jobs("data engineer", credentials=CREDS, max_results=60)

        assert len(jobs) == 60
        assert route.call_count == 2
        assert route.calls[0].request.url.path.endswith("/search/1")
        assert route.calls[1].request.url.path.endswith("/search/2")

    def test_stops_paginating_on_a_short_page(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(200, json={"count": 4, "results": payload["results"][:1]})
        )

        jobs = adzuna.fetch_jobs("data engineer", credentials=CREDS, max_results=100)

        assert len(jobs) == 1

    def test_transient_503_html_raises_fetch_error_not_a_json_crash(
        self, respx_mock: respx.MockRouter
    ) -> None:
        """Measured live: Adzuna intermittently answers 503 with a CloudFront HTML page."""
        respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(503, text="<!DOCTYPE html><html>Uh oh</html>")
        )

        with pytest.raises(FetchError):
            adzuna.fetch_jobs("data engineer", credentials=CREDS)

    def test_200_with_html_body_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(200, text="<!DOCTYPE html>")
        )

        with pytest.raises(FetchError):
            adzuna.fetch_jobs("data engineer", credentials=CREDS)

    def test_network_failure_raises_fetch_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            side_effect=httpx.ConnectError("refused")
        )

        with pytest.raises(FetchError):
            adzuna.fetch_jobs("data engineer", credentials=CREDS)

    def test_stamps_fetched_at_from_the_injected_clock(
        self, respx_mock: respx.MockRouter, payload: dict[str, Any]
    ) -> None:
        respx_mock.get(url__startswith="https://api.adzuna.com").mock(
            return_value=httpx.Response(200, json=payload)
        )

        jobs = adzuna.fetch_jobs("data engineer", credentials=CREDS, now=lambda: FETCHED_AT)

        assert all(job.fetched_at == FETCHED_AT for job in jobs)


class TestCredentialsFromEnv:
    def test_reads_the_documented_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ADZUNA_APP_ID", "abc")
        monkeypatch.setenv("ADZUNA_APP_KEY", "xyz")

        assert adzuna.AdzunaCredentials.from_env() == adzuna.AdzunaCredentials("abc", "xyz")

    def test_missing_credentials_fail_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A silent empty result would look like "no jobs today" forever."""
        monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
        monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)

        with pytest.raises(FetchError, match="ADZUNA_APP_ID"):
            adzuna.AdzunaCredentials.from_env()

    def test_credentials_are_not_exposed_by_repr(self) -> None:
        """The key must not leak into logs or a traceback."""
        assert "testkey" not in repr(CREDS)


def _full_page(payload: dict[str, Any]) -> dict[str, Any]:
    """Grow the 4-job fixture into a FULL 50-result page, with distinct ids.

    A short page means "last page", so pagination can only be tested against a full one.
    """
    template = payload["results"][0]
    results = [template | {"id": f"job-{n}"} for n in range(50)]
    return {"count": 29378, "results": results}


def _minimal_raw_result() -> dict[str, Any]:
    return {
        "id": "5801147013",
        "title": "Data Engineer",
        "company": {"display_name": "Acme"},
        "location": {"display_name": "Chicago, Cook County", "area": ["US", "Illinois"]},
        "created": "2026-07-15T08:33:45Z",
        "redirect_url": "https://www.adzuna.com/details/5801147013?utm_medium=api",
        "description": "Build pipelines",
        "salary_min": 119026.0,
        "salary_max": 119026.0,
        "salary_is_predicted": "1",
    }
