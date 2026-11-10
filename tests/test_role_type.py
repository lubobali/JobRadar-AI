"""Role type, work mode and week: the three columns the gold demand table groups by.

Titles marked REAL come from the recorded fixtures or from the 651-job sample quoted in
`prefilter.py`. The rest are plain textbook titles for buckets the fixtures do not cover.

Titles are a weak signal for MATCHING (see prefilter.py). For counting demand in aggregate they
are fine, as long as the traps below land in `other` and not in a data bucket.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from jobradar.role_type import ROLE_TYPES, classify_role, week_start, work_mode


class TestRoleTypes:
    def test_the_allowed_set_is_fixed(self) -> None:
        assert set(ROLE_TYPES) == {
            "ml_ai_engineer",
            "data_engineer",
            "analytics_engineer",
            "data_scientist",
            "data_analyst",
            "software_engineer",
            "other",
        }


class TestClassifyRole:
    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            # REAL
            ("Senior Data Engineer", "data_engineer"),
            ("Cloud Data Engineer", "data_engineer"),
            ("Google Cloud Platform Data Engineer", "data_engineer"),
            ("Data Engineer Senior Principal (Hybrid)", "data_engineer"),
            ("Data Engineer - TS/SCI with Polygraph", "data_engineer"),
            ("Manager of ETL Development", "data_engineer"),
            ("Applied AI Architect", "ml_ai_engineer"),
            ("ML Systems Engineer", "ml_ai_engineer"),
            ("Analytics Engineer", "analytics_engineer"),
            ("Senior Software Developer", "software_engineer"),
            ("Staff Software Engineer, Product (Belo Horizonte)", "software_engineer"),
            ("Software Engineer III", "software_engineer"),
            # Textbook
            ("Machine Learning Engineer", "ml_ai_engineer"),
            ("Senior LLM Engineer", "ml_ai_engineer"),
            ("Data Platform Engineer", "data_engineer"),
            ("Data Scientist", "data_scientist"),
            ("Data Analyst", "data_analyst"),
            ("Business Intelligence Analyst", "data_analyst"),
        ],
    )
    def test_buckets(self, title: str, expected: str) -> None:
        assert classify_role(title) == expected

    @pytest.mark.parametrize(
        "title",
        [
            # REAL traps: a data word or "analyst" that is not a data job.
            "Country Lead, Data Center Security",
            "Data Center Electrical Engineer",
            "Analyst, Licensing Finance – FP&A",  # noqa: RUF001 - real en dash
            # REAL non-data jobs.
            "Account Executive",
            "Biohacker Technician",
            "Head of Marketing & Communications",
            "Assistant Account Payable",
        ],
    )
    def test_traps_and_unrelated_land_in_other(self, title: str) -> None:
        assert classify_role(title) == "other"

    @pytest.mark.parametrize("title", [None, "", "   "])
    def test_missing_title_is_other(self, title: str | None) -> None:
        assert classify_role(title) == "other"

    def test_case_does_not_matter(self) -> None:
        assert classify_role("SENIOR DATA ENGINEER") == "data_engineer"
        assert classify_role("senior data engineer") == "data_engineer"

    @pytest.mark.parametrize(
        "title",
        [
            "Distinguished Technologist, Distributed Systems & Agentic Query Optimization",
            "Android Engineer I - Subscriptions",
            "IT Specialist (APPSW)",
            "General Engineer (Inspector) (Control Room Management) - DIRECT HIRE",
        ],
    )
    def test_always_returns_an_allowed_type(self, title: str) -> None:
        assert classify_role(title) in ROLE_TYPES


class TestWorkMode:
    def test_remote(self) -> None:
        assert work_mode(True) == "remote"

    def test_false_is_not_remote_not_onsite(self) -> None:
        # FALSE is the column default, so it means "not marked remote", never "onsite".
        assert work_mode(False) == "not_remote"

    def test_missing_is_not_remote(self) -> None:
        assert work_mode(None) == "not_remote"


class TestWeekStart:
    def test_midweek_rolls_back_to_monday(self) -> None:
        wed = datetime(2026, 8, 12, 15, tzinfo=UTC)
        assert week_start(wed, None) == date(2026, 8, 10)

    def test_monday_is_its_own_week(self) -> None:
        mon = datetime(2026, 8, 10, 0, 0, tzinfo=UTC)
        assert week_start(mon, None) == date(2026, 8, 10)

    def test_sunday_belongs_to_the_week_before(self) -> None:
        sun = datetime(2026, 8, 16, 23, 59, tzinfo=UTC)
        assert week_start(sun, None) == date(2026, 8, 10)

    def test_posted_at_wins_over_fetched_at(self) -> None:
        posted = datetime(2026, 8, 3, tzinfo=UTC)
        fetched = datetime(2026, 8, 12, tzinfo=UTC)
        assert week_start(posted, fetched) == date(2026, 8, 3)

    def test_falls_back_to_fetched_at(self) -> None:
        fetched = datetime(2026, 8, 12, tzinfo=UTC)
        assert week_start(None, fetched) == date(2026, 8, 10)

    def test_no_date_at_all_has_no_week(self) -> None:
        assert week_start(None, None) is None

    def test_weeks_are_cut_in_utc(self) -> None:
        # Sunday 23:00 in Chicago (UTC-5) is Monday 04:00 UTC: the NEXT week.
        chicago = timezone(timedelta(hours=-5))
        sun_night = datetime(2026, 8, 9, 23, tzinfo=chicago)
        assert week_start(sun_night, None) == date(2026, 8, 10)
