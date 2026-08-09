"""fit.py — the Python that turns the LLM's facts into the ranked digest.

This is the arithmetic half of Phase 2 (LUBO'S RULES: the LLM writes prose and reports facts;
Python does every number and every decision). It takes a ScoredJob — score, plus the facts the
LLM read off the posting (workplace, office days, years required) — and:

  1. decides skips in Python (years >= wall from the reported number; azure/discipline from the
     LLM's semantic flags), marking rather than deleting;
  2. fills the location tier using the extracted workplace facts, so tiers 2 and 4 — which no job
     API exposes — finally have values;
  3. ranks: fit score first, location as the tiebreaker, freshness last.
"""

from __future__ import annotations

from datetime import UTC, datetime

from jobradar.fit import RankedJob, Status, for_digest, rank, rank_one
from jobradar.location_rank import Tier
from jobradar.models import Job
from jobradar.scoring import ScoredJob

PROFILE = {"skip_flags": {"years_required_above": 8}}


def a_scored(  # noqa: PLR0913 — a builder; every field is an independent knob a test may set
    *,
    score: int = 70,
    location: str = "Remote (US)",
    remote: bool = True,
    workplace: str | None = "remote",
    office_days_per_month: int | None = None,
    years_required: int | None = None,
    work_authorization: str | None = None,
    skip_flags: tuple[str, ...] = (),
    source_id: str = "1",
    company: str = "Acme",
    title: str = "Data Engineer",
    posted_at: datetime | None = None,
) -> ScoredJob:
    job = Job(
        source="greenhouse",
        source_id=source_id,
        company=company,
        title=title,
        url="https://x.io/j/1",
        location=location,
        remote=remote,
        posted_at=posted_at,
        fetched_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    return ScoredJob(
        job=job,
        score=score,
        reason="reason",
        skip_flags=skip_flags,
        workplace=workplace,
        office_days_per_month=office_days_per_month,
        years_required=years_required,
        work_authorization=work_authorization,
    )


class TestSkipYears:
    """The years decision is Python's, computed from the number the LLM reported.

    Calibrated on Lubo's real applications: he applies to 1-3, 4-8, 5+ and 6+ year roles. Only
    8+ is his wall. Skipping a "5+ years" role would have hidden most of his own targets.
    """

    def test_a_role_at_the_wall_is_skipped(self) -> None:
        result = rank_one(a_scored(years_required=8), profile=PROFILE)

        assert result.status is Status.SKIPPED
        assert "8" in (result.skip_reason or "")

    def test_a_role_above_the_wall_is_skipped(self) -> None:
        assert rank_one(a_scored(years_required=12), profile=PROFILE).status is Status.SKIPPED

    def test_six_years_is_not_skipped(self) -> None:
        assert rank_one(a_scored(years_required=6), profile=PROFILE).status is Status.RANKED

    def test_five_plus_is_not_skipped(self) -> None:
        """The exact bug the gold-set caught — five-plus is not a wall."""
        assert rank_one(a_scored(years_required=5), profile=PROFILE).status is Status.RANKED

    def test_unknown_years_is_not_skipped(self) -> None:
        assert rank_one(a_scored(years_required=None), profile=PROFILE).status is Status.RANKED


class TestSkipFlags:
    """Non-numeric skips are semantic judgements, so they come from the LLM's flags."""

    def test_azure_mandatory_is_skipped(self) -> None:
        result = rank_one(a_scored(skip_flags=("azure_mandatory",)), profile=PROFILE)

        assert result.status is Status.SKIPPED
        assert "azure" in (result.skip_reason or "").lower()

    def test_wrong_discipline_does_not_hard_skip(self) -> None:
        """Discipline is what the score already measures, so hard-skipping on it is redundant and
        wrongly hides a low-but-wanted role (4C, the Node/TS sports job Lubo targets, scores ~42
        and carries this flag). A wrong-discipline job scores low and sinks off the top-N by
        itself — it must stay ranked and inspectable, not vanish.
        """
        result = rank_one(a_scored(score=42, skip_flags=("wrong_discipline",)), profile=PROFILE)

        assert result.status is Status.RANKED

    def test_a_clean_job_is_ranked(self) -> None:
        assert rank_one(a_scored(), profile=PROFILE).status is Status.RANKED

    def test_an_unrecognised_flag_does_not_skip(self) -> None:
        """Only the known killers skip; a stray flag must not silently drop a job."""
        assert rank_one(a_scored(skip_flags=("something_new",)), profile=PROFILE).status is (
            Status.RANKED
        )


class TestSkipWorkAuthorization:
    """Skip a job ONLY when it requires an authorization he cannot satisfy — the residual foreign
    role the geography prefilter let through (e.g. a bare "Remote" post whose body says "must be
    authorized to work in the UK"). Foreign LOCATION never skips here; the requirement does. And
    US citizenship / clearance is his MOAT — those roles must stay ranked, never skipped.
    """

    def test_foreign_authorization_requirement_is_skipped(self) -> None:
        result = rank_one(a_scored(work_authorization="foreign_required"), profile=PROFILE)

        assert result.status is Status.SKIPPED
        assert "authoriz" in (result.skip_reason or "").lower()

    def test_us_ok_is_not_skipped(self) -> None:
        assert rank_one(a_scored(work_authorization="us_ok"), profile=PROFILE).status is (
            Status.RANKED
        )

    def test_us_citizen_or_clearance_is_never_skipped(self) -> None:
        """His moat, not a barrier: he is a US citizen and clearance-eligible."""
        result = rank_one(a_scored(work_authorization="us_citizen_or_clearance"), profile=PROFILE)

        assert result.status is Status.RANKED

    def test_unknown_authorization_is_not_skipped(self) -> None:
        """Unknown means keep — the same asymmetry as the geography prefilter."""
        assert rank_one(a_scored(work_authorization="unknown"), profile=PROFILE).status is (
            Status.RANKED
        )

    def test_missing_authorization_is_not_skipped(self) -> None:
        assert rank_one(a_scored(work_authorization=None), profile=PROFILE).status is Status.RANKED


class TestLocationTierUsesExtractedFacts:
    """The LLM's workplace facts fill tiers 2 and 4, which no job API exposes."""

    def test_hybrid_in_the_target_metro_is_tier_2(self) -> None:
        scored = a_scored(location="Tampa, FL", remote=False, workplace="hybrid")

        assert rank_one(scored, profile=PROFILE).location_tier is Tier.TARGET_METRO_HYBRID

    def test_hybrid_with_rare_travel_is_tier_4(self) -> None:
        scored = a_scored(
            location="Denver, CO", remote=False, workplace="hybrid", office_days_per_month=2
        )

        assert rank_one(scored, profile=PROFILE).location_tier is Tier.HYBRID_RARE_TRAVEL

    def test_a_remote_job_is_tier_1(self) -> None:
        assert rank_one(a_scored(workplace="remote"), profile=PROFILE).location_tier is (
            Tier.REMOTE_US
        )

    def test_an_unknown_workplace_string_does_not_crash(self) -> None:
        result = rank_one(a_scored(workplace=None), profile=PROFILE)

        assert result.location_tier is Tier.REMOTE_US  # falls back to the job's remote flag


class TestRankOrder:
    def test_higher_fit_score_ranks_first(self) -> None:
        jobs = [a_scored(score=60, source_id="lo"), a_scored(score=90, source_id="hi")]

        assert [r.scored.job.source_id for r in rank(jobs, profile=PROFILE)] == ["hi", "lo"]

    def test_location_breaks_a_score_tie(self) -> None:
        """Two equally-good fits: the better location floats up."""
        remote = a_scored(score=80, source_id="remote", workplace="remote")
        austin = a_scored(
            score=80, source_id="austin", location="Austin, TX", remote=False, workplace="onsite"
        )

        ranked = rank([austin, remote], profile=PROFILE)

        assert [r.scored.job.source_id for r in ranked] == ["remote", "austin"]

    def test_a_strong_match_outranks_a_better_located_weak_match(self) -> None:
        """Fit is primary: a great Chicago job beats a mediocre remote one."""
        weak_remote = a_scored(score=45, source_id="weak", workplace="remote")
        strong_chicago = a_scored(
            score=92, source_id="strong", location="Chicago, IL", remote=False, workplace="onsite"
        )

        ranked = rank([weak_remote, strong_chicago], profile=PROFILE)

        assert ranked[0].scored.job.source_id == "strong"

    def test_an_unworkable_location_sinks_below_a_lower_scored_workable_job(self) -> None:
        """A 92 onsite in a city he won't relocate to (SF/Austin = OTHER_US) must NOT outrank a 72
        remote role he can actually take. It stays ranked/visible, just below every workable job."""
        austin_strong = a_scored(
            score=92, source_id="austin", location="Austin, TX", remote=False, workplace="onsite"
        )
        remote_weaker = a_scored(score=72, source_id="remote", workplace="remote")

        ranked = rank([austin_strong, remote_weaker], profile=PROFILE)

        assert [r.scored.job.source_id for r in ranked] == ["remote", "austin"]

    def test_it_is_workability_not_remoteness_that_demotes(self) -> None:
        """Chicago onsite (a bridge he'd take = workable) still beats a higher-scored OTHER_US."""
        chicago = a_scored(
            score=60, source_id="chi", location="Chicago, IL", remote=False, workplace="onsite"
        )
        austin = a_scored(
            score=92, source_id="austin", location="Austin, TX", remote=False, workplace="onsite"
        )

        ranked = rank([austin, chicago], profile=PROFILE)

        assert [r.scored.job.source_id for r in ranked] == ["chi", "austin"]

    def test_freshness_breaks_a_score_and_location_tie(self) -> None:
        older = a_scored(score=80, source_id="old", posted_at=datetime(2026, 7, 1, tzinfo=UTC))
        newer = a_scored(score=80, source_id="new", posted_at=datetime(2026, 7, 15, tzinfo=UTC))

        ranked = rank([older, newer], profile=PROFILE)

        assert [r.scored.job.source_id for r in ranked] == ["new", "old"]

    def test_a_missing_posted_at_sorts_after_a_dated_one(self) -> None:
        dated = a_scored(score=80, source_id="dated", posted_at=datetime(2026, 7, 15, tzinfo=UTC))
        undated = a_scored(score=80, source_id="undated", posted_at=None)

        ranked = rank([undated, dated], profile=PROFILE)

        assert [r.scored.job.source_id for r in ranked] == ["dated", "undated"]


class TestRankNeverDrops:
    def test_every_job_comes_back_including_skipped(self) -> None:
        """Skips MARK, they do not delete — the same discipline as the location ranker."""
        jobs = [
            a_scored(score=90, source_id="good"),
            a_scored(score=90, source_id="azure", skip_flags=("azure_mandatory",)),
            a_scored(score=90, source_id="senior", years_required=10),
        ]

        ranked = rank(jobs, profile=PROFILE)

        assert len(ranked) == 3
        assert {r.scored.job.source_id for r in ranked} == {"good", "azure", "senior"}

    def test_skipped_jobs_sort_below_every_ranked_job(self) -> None:
        """A skipped 90 must not sit above a ranked 50 — status dominates the order."""
        jobs = [
            a_scored(score=90, source_id="skipped", years_required=10),
            a_scored(score=50, source_id="kept"),
        ]

        ranked = rank(jobs, profile=PROFILE)

        assert [r.scored.job.source_id for r in ranked] == ["kept", "skipped"]

    def test_an_empty_batch_is_empty(self) -> None:
        assert rank([], profile=PROFILE) == []


class TestForDigest:
    def test_excludes_skipped_jobs(self) -> None:
        jobs = [
            a_scored(score=90, source_id="good"),
            a_scored(score=95, source_id="azure", skip_flags=("azure_mandatory",)),
        ]

        digest = for_digest(rank(jobs, profile=PROFILE))

        assert [r.scored.job.source_id for r in digest] == ["good"]

    def test_takes_only_the_top_n(self) -> None:
        # Distinct companies so the per-company cap doesn't trim before the limit does.
        jobs = [a_scored(score=90 - i, source_id=str(i), company=f"Co{i}") for i in range(15)]

        digest = for_digest(rank(jobs, profile=PROFILE), limit=10)

        assert len(digest) == 10
        assert [r.scored.job.source_id for r in digest] == [str(i) for i in range(10)]

    def test_returns_a_list_of_ranked_jobs(self) -> None:
        digest = for_digest(rank([a_scored()], profile=PROFILE))

        assert isinstance(digest[0], RankedJob)

    def test_an_all_skipped_batch_yields_an_empty_digest(self) -> None:
        jobs = [a_scored(years_required=10, source_id=str(i)) for i in range(3)]

        assert for_digest(rank(jobs, profile=PROFILE)) == []


class TestDigestScoreFloor:
    """The digest emails only genuinely-strong matches. A weak-but-ranked job stays in the full
    ranking (inspectable), but never lands in Lubo's inbox — "nothing great today" beats five
    mediocre ones. The floor is 65 by default.
    """

    def test_a_job_below_the_floor_is_not_emailed(self) -> None:
        digest = for_digest(rank([a_scored(score=64)], profile=PROFILE))

        assert digest == []

    def test_a_job_at_the_floor_is_emailed(self) -> None:
        digest = for_digest(rank([a_scored(score=65)], profile=PROFILE))

        assert len(digest) == 1

    def test_the_floor_is_configurable(self) -> None:
        jobs = [a_scored(score=70, source_id="a"), a_scored(score=80, source_id="b")]

        digest = for_digest(rank(jobs, profile=PROFILE), min_score=75)

        assert [r.scored.job.source_id for r in digest] == ["b"]

    def test_a_weak_job_stays_in_the_full_ranking_even_when_floored_out(self) -> None:
        """Floored from the email, NOT dropped from the ranking — the anti-silent-drop rule."""
        ranked = rank([a_scored(score=40)], profile=PROFILE)

        assert len(ranked) == 1
        assert ranked[0].status is Status.RANKED
        assert for_digest(ranked) == []


class TestPerCompanyCap:
    """No single employer floods the digest. The first real run emailed 6 Astronomer roles of 8;
    the cap keeps at most 2 per company so the digest stays varied. It keeps the BEST ones (the
    ranking is already best-first) and only trims within a company — never across."""

    def test_caps_a_flooding_company_to_two(self) -> None:
        # Distinct titles so the duplicate-role collapse doesn't merge them before the cap acts.
        jobs = [
            a_scored(score=90 - i, company="Astronomer", title=f"Role {i}", source_id=str(i))
            for i in range(6)
        ]

        digest = for_digest(rank(jobs, profile=PROFILE))

        assert len(digest) == 2

    def test_keeps_the_two_highest_scored_of_that_company(self) -> None:
        jobs = [
            a_scored(score=95, company="Astronomer", title="A", source_id="best"),
            a_scored(score=90, company="Astronomer", title="B", source_id="mid"),
            a_scored(score=80, company="Astronomer", title="C", source_id="low"),
        ]

        digest = for_digest(rank(jobs, profile=PROFILE))

        assert [r.scored.job.source_id for r in digest] == ["best", "mid"]

    def test_does_not_cap_across_different_companies(self) -> None:
        jobs = [
            a_scored(score=90, company="Modelyst", source_id="a"),
            a_scored(score=88, company="Foodsmart", source_id="b"),
            a_scored(score=86, company="Mercury", source_id="c"),
        ]

        digest = for_digest(rank(jobs, profile=PROFILE))

        assert len(digest) == 3

    def test_the_cap_is_configurable(self) -> None:
        jobs = [
            a_scored(score=90 - i, company="Astronomer", title=f"Role {i}", source_id=str(i))
            for i in range(5)
        ]

        digest = for_digest(rank(jobs, profile=PROFILE), per_company=3)

        assert len(digest) == 3

    def test_an_empty_company_is_not_capped(self) -> None:
        """Adzuna sometimes has no company name; those must not all collapse to one bucket."""
        jobs = [a_scored(score=90 - i, company="", source_id=str(i)) for i in range(4)]

        digest = for_digest(rank(jobs, profile=PROFILE))

        assert len(digest) == 4


class TestDigestWorkableOnly:
    """The email must contain only locations he can actually take — he should never open a link to
    find out it is onsite in a city he won't move to. OTHER_US (onsite/hybrid elsewhere) is kept in
    the full ranking but not emailed; remote / target-metro / Chicago all email normally."""

    def test_an_onsite_elsewhere_job_is_not_emailed(self) -> None:
        sf = a_scored(score=92, location="San Francisco, CA", remote=False, workplace="onsite")

        assert for_digest(rank([sf], profile=PROFILE)) == []

    def test_a_remote_job_is_emailed(self) -> None:
        remote = a_scored(score=92, workplace="remote")

        assert len(for_digest(rank([remote], profile=PROFILE))) == 1

    def test_a_chicago_bridge_job_is_emailed(self) -> None:
        chi = a_scored(score=80, location="Chicago, IL", remote=False, workplace="onsite")

        assert len(for_digest(rank([chi], profile=PROFILE))) == 1

    def test_workable_jobs_win_the_slots_over_unworkable_ones(self) -> None:
        """A batch with both: only the workable ones reach the inbox, even if scored lower."""
        sf = a_scored(score=95, location="San Francisco, CA", remote=False, workplace="onsite",
                      source_id="sf")
        remote = a_scored(score=70, source_id="remote", workplace="remote")

        digest = for_digest(rank([sf, remote], profile=PROFILE))

        assert [r.scored.job.source_id for r in digest] == ["remote"]

    def test_it_can_be_turned_off(self) -> None:
        sf = a_scored(score=92, location="San Francisco, CA", remote=False, workplace="onsite")

        assert len(for_digest(rank([sf], profile=PROFILE), workable_only=False)) == 1


class TestCollapseDuplicateRoles:
    """A company can post the same role twice under different IDs (Lithic did: two 'Senior
    Software Engineer, Data Platform', both 95). Layer-1 dedup keys on source+id, so both slip
    through as 'new'. The digest shows a role once — same company + title collapses to the best."""

    def test_same_company_same_title_collapses_to_one(self) -> None:
        jobs = [
            a_scored(score=95, company="Lithic", title="Senior SWE, Data Platform", source_id="a"),
            a_scored(score=95, company="Lithic", title="Senior SWE, Data Platform", source_id="b"),
        ]

        digest = for_digest(rank(jobs, profile=PROFILE))

        assert len(digest) == 1

    def test_it_keeps_the_higher_scored_of_the_duplicates(self) -> None:
        jobs = [
            a_scored(score=88, company="Lithic", title="Data Platform Eng", source_id="lower"),
            a_scored(score=95, company="Lithic", title="Data Platform Eng", source_id="higher"),
        ]

        digest = for_digest(rank(jobs, profile=PROFILE))

        assert [r.scored.job.source_id for r in digest] == ["higher"]

    def test_title_match_ignores_case_and_whitespace(self) -> None:
        jobs = [
            a_scored(score=90, company="Lithic", title="Data Engineer", source_id="a"),
            a_scored(score=88, company="Lithic", title="  data  engineer ", source_id="b"),
        ]

        assert len(for_digest(rank(jobs, profile=PROFILE))) == 1

    def test_same_title_different_company_is_kept(self) -> None:
        jobs = [
            a_scored(score=90, company="Lithic", title="Data Engineer", source_id="a"),
            a_scored(score=88, company="Mercury", title="Data Engineer", source_id="b"),
        ]

        assert len(for_digest(rank(jobs, profile=PROFILE))) == 2

    def test_different_titles_same_company_are_kept(self) -> None:
        jobs = [
            a_scored(score=90, company="Lithic", title="Data Engineer", source_id="a"),
            a_scored(score=88, company="Lithic", title="ML Ops Engineer", source_id="b"),
        ]

        assert len(for_digest(rank(jobs, profile=PROFILE))) == 2
