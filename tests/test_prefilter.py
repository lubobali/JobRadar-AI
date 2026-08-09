"""The prefilter — a cheap, deterministic volume cut before the LLM.

Every case below is a REAL location string taken from a live board (97 distinct strings across
651 jobs from Greenhouse, Lever, Ashby, Workday and Adzuna). None are invented.

The prefilter's mistakes are invisible: a job it drops is never scored, never emailed, and
never missed-in-a-way-anyone-notices. So it only ever drops what it can PROVE is unusable.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from jobradar.models import Job
from jobradar.prefilter import is_us_eligible, keep_worth_scoring


def a_job(location: str | None, source_id: str = "1", title: str = "Data Engineer") -> Job:
    return Job(
        source="greenhouse",
        source_id=source_id,
        company="Acme",
        title=title,
        url="https://x.io/j/1",
        location=location,
        fetched_at=datetime(2026, 7, 16, tzinfo=UTC),
    )


class TestUsEligibleKeeps:
    """Real strings that MUST survive."""

    @pytest.mark.parametrize(
        "location",
        [
            "San Francisco, CA",
            "New York City, NY",
            "Washington, DC",
            "Chicago, IL",
            "Miami, FL",
            "Boston, MA; New York City, NY; Washington, DC",
            "San Francisco, CA | New York City, NY | Seattle, WA",
            "New York, NY (HQ), San Francisco, CA, Remote (US)",
            "Remote-Friendly, United States",
            "New York, NY, United States of America (Home Mix)",
            "USA DC Washington",  # workday
            "USA NC Fort Bragg",  # workday
            "Remote (US), San Francisco, CA, New York, NY (HQ)",
        ],
    )
    def test_us_locations_are_kept(self, location: str) -> None:
        assert is_us_eligible(location) is True

    @pytest.mark.parametrize(
        "location",
        [
            # A job open in BOTH places is open to Lubo. Rejecting on the foreign half would
            # silently discard a role he could actually take.
            "London, UK; Ontario, CAN; Remote-Friendly, United States; San Francisco, CA",
            "New York, NY (HQ), Remote (Canada), Remote (US), Miami, FL",
            "Remote (US), Remote (Canada), San Francisco, CA, New York, NY (HQ), Toronto, ON",
            "New York, NY, London",
            "Washington D.C., London",
            "New York, NY, Stockholm",
        ],
    )
    def test_a_us_signal_beats_a_foreign_one(self, location: str) -> None:
        assert is_us_eligible(location) is True

    @pytest.mark.parametrize(
        "location",
        [
            "Chicago, Cook County",  # adzuna: no state, no country
            "The Gap, Chicago",  # adzuna
            "Illinois Medical District, Chicago",  # adzuna
            "State Farm, Arlington County",  # adzuna
            "Boston, Suffolk County",  # adzuna
            "Eagan, Dakota County",  # adzuna
        ],
    )
    def test_unrecognised_locations_are_kept(self, location: str) -> None:
        """Adzuna names a city and county but never a state or country.

        Requiring positive proof of US-ness would silently delete EVERY Adzuna job. Unknown
        means unknown — pass it to the LLM, which reads the actual posting.
        """
        assert is_us_eligible(location) is True

    def test_a_missing_location_is_kept(self) -> None:
        assert is_us_eligible(None) is True

    def test_an_empty_location_is_kept(self) -> None:
        assert is_us_eligible("") is True


class TestUsEligibleDrops:
    """Real strings that are genuinely unusable — Lubo is US-based and needs no visa."""

    @pytest.mark.parametrize(
        "location",
        [
            "London, UK",
            "Sydney, Australia",
            "Tokyo, Japan",
            "Bangalore, India",
            "Munich, Germany",
            "Paris, France",
            "Dublin, IE",
            "Seoul, South Korea",
            "Zürich, CH",
            "Ontario, CAN",
            "Toronto, ON",
            "Remote (Canada)",
            "Remote (Buenos Aires, Argentina)",
            "DEU Wiesbaden - Wiesbaden Army Airfield (APC180)",  # workday
        ],
    )
    def test_foreign_locations_with_a_country_are_dropped(self, location: str) -> None:
        assert is_us_eligible(location) is False

    @pytest.mark.parametrize(
        "location",
        [
            "London",
            "Berlin",
            "Milan",
            "Mumbai",
            "Jakarta",
            "Istanbul",
            "Nairobi",
            "Bogotá",
            "Brussels",
            "Dubai",
            "Stockholm",
            "Sydney",
            "Toronto",
            "Singapore",
            "Seoul",
            "São Paulo",
            "Taiwan",
            "London, Stockholm",
            "Stockholm, London",  # the exact string Lever emits for Spotify's Analytics Engineer II
            "Singapore, Seoul",
        ],
    )
    def test_bare_foreign_cities_are_dropped(self, location: str) -> None:
        """Half the foreign postings name no country at all — just the city.

        "Stockholm, London" is the real Spotify posting Lubo caught in the throwaway test digest:
        two bare foreign cities, no country. The shipped pipeline's prefilter drops it; that manual
        digest simply did not route through the prefilter.
        """
        assert is_us_eligible(location) is False


class TestRemoteRegions:
    """Remotive's `candidate_required_location` is a region, not a city. A US-inclusive scope is
    workable; a region-locked one is not. Added when Remotive (Phase B1) came online."""

    @pytest.mark.parametrize(
        "location",
        [
            "Worldwide",
            "Anywhere",
            "Americas",
            "North America",
            "Northern America",
            "USA, Canada, USA timezones",
            "Americas, Europe, Israel",  # Americas includes the US -> keep despite Israel/Europe
            "Northern America, LATAM, Europe, APAC",
        ],
    )
    def test_us_inclusive_remote_scopes_are_kept(self, location: str) -> None:
        assert is_us_eligible(location) is True

    @pytest.mark.parametrize(
        "location",
        ["Brazil", "Mexico", "Uruguay", "LATAM, Europe", "Europe, UK, Germany, France"],
    )
    def test_region_locked_non_us_scopes_are_dropped(self, location: str) -> None:
        assert is_us_eligible(location) is False


class TestFalsePositiveTraps:
    """US cities that share a name with a foreign one. The US signal must win."""

    @pytest.mark.parametrize(
        "location",
        ["Paris, TX", "London, OH", "Berlin, NH", "Toronto, OH", "Milan, MI"],
    )
    def test_us_namesakes_are_kept(self, location: str) -> None:
        assert is_us_eligible(location) is True

    def test_canada_ca_does_not_read_as_california(self) -> None:
        """ "CAN" must not match the state code "CA" — word boundaries matter."""
        assert is_us_eligible("Ontario, CAN") is False

    def test_workday_canada_country_prefix_is_dropped(self) -> None:
        """Snowflake/Workday format "CA-Ontario-Toronto" is Canada — the leading CA is the country
        code, not California. It leaked a Toronto job into a real digest until this was handled."""
        assert is_us_eligible("CA-Ontario-Toronto") is False
        assert is_us_eligible("CA-British Columbia-Vancouver") is False

    def test_workday_us_country_prefix_is_kept(self) -> None:
        """The mirror format "US-CA-Menlo Park" IS genuinely California — must not be dropped."""
        assert is_us_eligible("US-CA-Menlo Park") is True
        assert is_us_eligible("US-NY-New York") is True

    def test_california_still_reads_as_california(self) -> None:
        assert is_us_eligible("San Francisco, CA") is True


class TestKeepWorthScoring:
    def test_keeps_us_jobs_and_drops_foreign_ones(self) -> None:
        jobs = [
            a_job("San Francisco, CA", "1"),
            a_job("London, UK", "2"),
            a_job("Chicago, Cook County", "3"),
            a_job("Sydney, Australia", "4"),
        ]

        assert [j.source_id for j in keep_worth_scoring(jobs)] == ["1", "3"]

    def test_does_not_filter_on_title(self) -> None:
        """Measured on 651 real jobs: a title deny-list would save 32 LLM calls (6.7%) while
        risking silent false negatives, and a title ALLOW-list would hide 160 real matches
        (every "Applied AI Architect", "ML Systems", "Analytics Engineer"). The LLM reads the
        JD body and judges; the prefilter only cuts geography.
        """
        jobs = [
            a_job("San Francisco, CA", "1", title="Data Center Electrical Engineer"),
            a_job("San Francisco, CA", "2", title="Software Engineer III"),
            a_job("San Francisco, CA", "3", title="Account Executive"),
        ]

        assert len(keep_worth_scoring(jobs)) == 3

    def test_preserves_order(self) -> None:
        jobs = [a_job("Chicago, IL", "3"), a_job("Miami, FL", "1"), a_job("Seattle, WA", "2")]

        assert [j.source_id for j in keep_worth_scoring(jobs)] == ["3", "1", "2"]

    def test_an_empty_batch_is_empty(self) -> None:
        assert keep_worth_scoring([]) == []
