"""The normalized job schema every fetcher must produce."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from jobradar.models import Job, make_job_id

GREENHOUSE_URL = "https://boards.greenhouse.io/acme/jobs/4001"
SOURCE_ID = "4001"


def a_job(**overrides: object) -> Job:
    """Build a Job with sensible defaults, overriding only what a test cares about."""
    defaults = {
        "source": "greenhouse",
        "source_id": SOURCE_ID,
        "company": "Acme",
        "title": "Data Engineer",
        "url": GREENHOUSE_URL,
        "fetched_at": datetime(2026, 7, 15, 12, 0, tzinfo=UTC),
    }
    return Job(**(defaults | overrides))  # type: ignore[arg-type]


class TestMakeJobId:
    def test_is_stable_across_calls(self) -> None:
        assert make_job_id("greenhouse", SOURCE_ID) == make_job_id("greenhouse", SOURCE_ID)

    def test_differs_by_source(self) -> None:
        assert make_job_id("greenhouse", SOURCE_ID) != make_job_id("lever", SOURCE_ID)

    def test_differs_by_source_id(self) -> None:
        assert make_job_id("greenhouse", SOURCE_ID) != make_job_id("greenhouse", SOURCE_ID + "0")

    def test_source_and_id_boundary_cannot_be_forged(self) -> None:
        """Naive concatenation would collide these two; a delimiter must prevent it."""
        assert make_job_id("green", "house/x") != make_job_id("greenhouse", "/x")

    def test_is_a_sha256_hex_digest(self) -> None:
        assert len(make_job_id("greenhouse", SOURCE_ID)) == 64


class TestJob:
    def test_job_id_derives_from_source_and_source_id(self) -> None:
        assert a_job().job_id == make_job_id("greenhouse", SOURCE_ID)

    def test_same_posting_fetched_twice_dedupes_to_one_id(self) -> None:
        """The dedup guarantee: volatile fields must not change the id."""
        first = a_job(fetched_at=datetime(2026, 7, 15, 12, 0, tzinfo=UTC))
        second = a_job(fetched_at=datetime(2026, 7, 15, 12, 15, tzinfo=UTC), salary="$200k")
        assert first.job_id == second.job_id

    def test_job_id_survives_a_url_that_changes_every_fetch(self) -> None:
        """Decision Log #1: Adzuna's redirect_url carries a per-request token.

        Hashing the url would mint a new id on every poll and re-email the same job forever.
        """
        first = a_job(url="https://adzuna.com/land/ad/5798462542?se=wvXDz_p_8RG")
        second = a_job(url="https://adzuna.com/land/ad/5798462542?se=Giua0Pp_8RG")
        assert first.job_id == second.job_id

    def test_is_immutable(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            a_job().title = "Staff Data Engineer"  # type: ignore[misc]

    def test_scoring_fields_are_absent_until_phase_2(self) -> None:
        """Phase 1 ingests only; fit_score/skip_flags are not invented early."""
        assert not hasattr(a_job(), "fit_score")

    def test_optional_fields_default_to_empty(self) -> None:
        job = a_job()
        assert job.location is None
        assert job.salary is None
        assert job.posted_at is None
        assert job.remote is False
        assert job.description == ""


class TestSalaryIsEstimated:
    """Some sources guess a salary and present it like a fact.

    Adzuna predicts one for 34 of 50 live results (min==max point estimates). Ashby's comes
    from the employer. A digest that shows both identically would be lying, so the flag travels
    with the number rather than the number being silently dropped.
    """

    def test_defaults_to_false_so_a_salary_is_trusted_unless_flagged(self) -> None:
        assert a_job(salary="$151K - $231K").salary_is_estimated is False

    def test_can_mark_a_guessed_salary(self) -> None:
        assert a_job(salary="$119,026", salary_is_estimated=True).salary_is_estimated is True

    def test_does_not_change_the_dedup_key(self) -> None:
        """It is metadata about the posting, not identity."""
        assert a_job(salary_is_estimated=True).job_id == a_job(salary_is_estimated=False).job_id
