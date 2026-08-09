"""Ranking jobs by Lubo's stated location preference.

His order, verbatim:
  1. Remote 100%
  2. Hybrid in Sarasota / Tampa Bay FL   (he would relocate immediately)
  3. On-site in Sarasota County FL       (he would relocate immediately)
  4. Hybrid anywhere, office <= ~2x/month
  5. Chicago hybrid — acceptable as a temporary 6-12 month bridge, then he relocates

This RANKS. It must never drop, because he would take a Chicago hybrid or a "2 days a month"
role anywhere — filtering on location would hide exactly the jobs he would accept.

Per LUBO'S RULES the tier is arithmetic, so it lives in Python. The LLM's job (Phase 2) is to
read a JD and report facts — workplace type, office days per month — never to rank.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from jobradar.location_rank import (
    Tier,
    Workplace,
    location_tier,
    mentions_target_metro,
    rank_by_location,
)
from jobradar.models import Job


def a_job(location: str | None, *, remote: bool = False, source_id: str = "1") -> Job:
    return Job(
        source="greenhouse",
        source_id=source_id,
        company="Acme",
        title="Data Engineer",
        url="https://x.io/j/1",
        location=location,
        remote=remote,
        fetched_at=datetime(2026, 7, 16, tzinfo=UTC),
    )


class TestTier1RemoteIsBest:
    def test_a_remote_us_job_is_tier_1(self) -> None:
        assert location_tier(a_job("Remote (US)", remote=True)) is Tier.REMOTE_US

    def test_remote_wins_even_when_an_office_city_is_listed(self) -> None:
        """ "New York, NY (HQ), Remote (US)" is a remote job with an HQ, not a New York job."""
        job = a_job("New York, NY (HQ), San Francisco, CA, Remote (US)", remote=True)

        assert location_tier(job) is Tier.REMOTE_US

    def test_remote_beats_the_target_metro(self) -> None:
        """Remote is his #1. A remote job is better than a Sarasota office job."""
        remote = location_tier(a_job("Remote (US)", remote=True))
        sarasota = location_tier(a_job("Sarasota, FL"))

        assert remote.value < sarasota.value


class TestTier2And3TargetMetro:
    @pytest.mark.parametrize(
        "location",
        ["Venice, FL", "Sarasota, FL", "North Port, FL", "Nokomis, FL", "Osprey, FL"],
    )
    def test_sarasota_county_onsite_is_tier_3(self, location: str) -> None:
        assert location_tier(a_job(location)) is Tier.TARGET_METRO_ONSITE

    @pytest.mark.parametrize(
        "location",
        ["Tampa, FL", "St. Petersburg, FL", "Clearwater, FL", "Bradenton, FL", "Brandon, FL"],
    )
    def test_tampa_bay_is_the_target_metro(self, location: str) -> None:
        """He would move to Venice and commute within the bay area."""
        assert location_tier(a_job(location)).value <= Tier.TARGET_METRO_ONSITE.value

    def test_hybrid_in_the_target_metro_is_tier_2(self) -> None:
        job = a_job("Tampa, FL")

        assert location_tier(job, workplace=Workplace.HYBRID) is Tier.TARGET_METRO_HYBRID

    def test_target_metro_beats_chicago(self) -> None:
        assert location_tier(a_job("Venice, FL")).value < location_tier(a_job("Chicago, IL")).value


class TestTier4HybridWithRareTravel:
    def test_hybrid_anywhere_with_two_office_days_a_month_is_tier_4(self) -> None:
        """He would fly in from Venice for this."""
        job = a_job("Denver, CO")

        tier = location_tier(job, workplace=Workplace.HYBRID, office_days_per_month=2)

        assert tier is Tier.HYBRID_RARE_TRAVEL

    def test_a_normal_hybrid_far_from_home_is_not_tier_4(self) -> None:
        """3 days a WEEK in Denver is a relocation he does not want."""
        job = a_job("Denver, CO")

        tier = location_tier(job, workplace=Workplace.HYBRID, office_days_per_month=12)

        assert tier.value > Tier.HYBRID_RARE_TRAVEL.value

    def test_rare_travel_beats_chicago_hybrid(self) -> None:
        rare = location_tier(
            a_job("Denver, CO"), workplace=Workplace.HYBRID, office_days_per_month=1
        )
        chicago = location_tier(a_job("Chicago, IL"), workplace=Workplace.HYBRID)

        assert rare.value < chicago.value


class TestTier5ChicagoBridge:
    @pytest.mark.parametrize(
        "location",
        [
            "Chicago, IL",
            "Evanston, IL",
            "Naperville, IL",
            "Lemont, IL",  # Argonne National Laboratory — Chicago metro, ~25mi, commutable
            "Joliet, IL",
            "Schaumburg, IL",
            "Buffalo Grove, IL",  # SPR AI Platform role interim site
            "DuPage County, IL",
        ],
    )
    def test_chicagoland_is_ranked_not_dropped(self, location: str) -> None:
        """A temporary bridge he would accept — so it must still surface, and NOT sink to OTHER_US
        (an Argonne role in Lemont did exactly that until Chicagoland was broadened)."""
        assert location_tier(a_job(location)) is Tier.CURRENT_BASE

    def test_chicago_beats_an_unrelated_us_city(self) -> None:
        """He already lives there; Austin on-site would mean an unwanted move."""
        chicago = location_tier(a_job("Chicago, IL"))
        austin = location_tier(a_job("Austin, TX"))

        assert chicago.value < austin.value


class TestOtherUs:
    def test_an_unrelated_us_onsite_job_ranks_last_but_still_ranks(self) -> None:
        assert location_tier(a_job("Austin, TX")) is Tier.OTHER_US

    def test_an_unknown_location_is_not_thrown_away(self) -> None:
        """Adzuna gives "Chicago, Cook County" — no state. It must still rank."""
        assert location_tier(a_job(None)) is Tier.OTHER_US

    def test_adzunas_county_format_still_finds_chicago(self) -> None:
        assert location_tier(a_job("Chicago, Cook County")) is Tier.CURRENT_BASE


class TestMentionsTargetMetro:
    """Used by the digest to flag a company sitting in his target area, even for a remote role."""

    @pytest.mark.parametrize(
        "text",
        ["company in Tampa", "Sarasota, FL", "Venice", "St. Petersburg", "Bradenton, Florida"],
    )
    def test_detects_the_target_metro(self, text: str) -> None:
        assert mentions_target_metro(text) is True

    @pytest.mark.parametrize("text", ["Chicago, IL", "Austin, TX", "Remote (US)", "", "London, UK"])
    def test_ignores_everywhere_else(self, text: str) -> None:
        assert mentions_target_metro(text) is False

    def test_handles_none(self) -> None:
        assert mentions_target_metro(None) is False


class TestRankByLocation:
    def test_orders_by_preference(self) -> None:
        jobs = [
            a_job("Austin, TX", source_id="other"),
            a_job("Chicago, IL", source_id="chicago"),
            a_job("Remote (US)", remote=True, source_id="remote"),
            a_job("Venice, FL", source_id="venice"),
        ]

        ranked = rank_by_location(jobs)

        assert [j.source_id for j in ranked] == ["remote", "venice", "chicago", "other"]

    def test_never_drops_a_job(self) -> None:
        """The whole point: ranking, not filtering."""
        jobs = [a_job("Austin, TX"), a_job("Chicago, IL"), a_job("Remote (US)", remote=True)]

        assert len(rank_by_location(jobs)) == len(jobs)

    def test_is_stable_within_a_tier(self) -> None:
        jobs = [a_job("Tampa, FL", source_id="1"), a_job("Sarasota, FL", source_id="2")]

        assert [j.source_id for j in rank_by_location(jobs)] == ["1", "2"]

    def test_an_empty_batch_is_empty(self) -> None:
        assert rank_by_location([]) == []
