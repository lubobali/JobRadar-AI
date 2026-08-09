"""Rank jobs by Lubo's stated location preference.

His order, verbatim (2026-07-16):
  1. Remote 100%
  2. Hybrid in Sarasota / Tampa Bay FL   -- he would relocate immediately
  3. On-site in Sarasota County FL       -- he would relocate immediately
  4. Hybrid anywhere, office <= ~2x/month  -- he would fly in from Venice
  5. Chicago hybrid                      -- a temporary 6-12 month bridge, then he relocates

This module RANKS. It must never drop a job, and that is not a technicality: he would take a
Chicago hybrid, or a "two days a month" role in Denver. Filtering on location would hide
precisely the jobs he would accept, so the prefilter stays US-wide and preference only decides
what floats to the top.

Per LUBO'S RULES, ranking is arithmetic and therefore lives in Python. The LLM's job (Phase 2)
is to read a JD and report facts — is it hybrid, how many office days a month — never to rank.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum, IntEnum

from jobradar.models import Job


class Tier(IntEnum):
    """Preference rank. Lower is better; the numbers are his stated order."""

    REMOTE_US = 1
    TARGET_METRO_HYBRID = 2
    TARGET_METRO_ONSITE = 3
    HYBRID_RARE_TRAVEL = 4
    CURRENT_BASE = 5
    OTHER_US = 6


class Workplace(Enum):
    """How much office attendance a posting demands. Phase 2's LLM reads this off the JD."""

    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


MAX_RARE_TRAVEL_DAYS_PER_MONTH = 2
"""Above this, an out-of-state hybrid is a relocation he does not want."""

# Sarasota County + the wider Tampa Bay metro. He is moving to Venice, FL and would commute
# within the bay.
_TARGET_METRO = re.compile(
    r"\b(?:Venice|Sarasota|North Port|Osprey|Nokomis|Laurel|Englewood|Siesta Key"
    r"|Tampa|St\.? ?Petersburg|Saint Petersburg|Clearwater|Bradenton|Palmetto"
    r"|Brandon|Riverview|Largo|Lakeland)\b",
    re.IGNORECASE,
)

# Where he lives today. Matched loosely so Adzuna's "Chicago, Cook County" still lands here.
_CURRENT_BASE = re.compile(
    # Chicago + Chicagoland: his current base, a bridge he would take. Broadened 2026-07-17 when an
    # Argonne National Laboratory role in Lemont, IL sank to OTHER_US and got wrongly demoted as
    # unworkable — Lemont is Chicago metro (~25mi, commutable). Curated suburbs + collar counties;
    # ambiguous bare names (Aurora, Lake County) are left out to avoid matching other states.
    r"\b(?:Chicago|Chicagoland|Evanston|Oak Park|Naperville|Schaumburg|Lemont|Joliet|Elgin"
    r"|Cicero|Skokie|Arlington Heights|Des Plaines|Orland Park|Tinley Park|Bolingbrook"
    r"|Downers Grove|Elmhurst|Lombard|Wheaton|Oak Brook|Hinsdale|Palatine|Buffalo Grove"
    r"|Cook County|DuPage County|Will County|McHenry County)\b",
    re.IGNORECASE,
)

_REMOTE_TEXT = re.compile(r"\bremote\b", re.IGNORECASE)


def location_tier(
    job: Job,
    *,
    workplace: Workplace = Workplace.UNKNOWN,
    office_days_per_month: int | None = None,
) -> Tier:
    """Return how well `job` fits his location preference. Lower is better.

    `workplace` and `office_days_per_month` are facts Phase 2's LLM extracts from the JD. Until
    then they default to unknown, and the ranking falls back to the location string plus the
    source's own remote flag.

    >>> from datetime import UTC, datetime
    >>> def j(loc, remote=False):
    ...     return Job(source="s", source_id="1", company="c", title="t", url="u",
    ...                location=loc, remote=remote, fetched_at=datetime(2026, 7, 16, tzinfo=UTC))
    >>> location_tier(j("Remote (US)", remote=True)).name
    'REMOTE_US'
    >>> location_tier(j("Venice, FL")).name
    'TARGET_METRO_ONSITE'
    >>> location_tier(j("Chicago, IL")).name
    'CURRENT_BASE'
    >>> location_tier(j("Austin, TX")).name
    'OTHER_US'
    """
    if _is_remote(job, workplace):
        return Tier.REMOTE_US

    location = job.location or ""

    if _TARGET_METRO.search(location):
        # He would move for either, but hybrid beats a full office week.
        return (
            Tier.TARGET_METRO_HYBRID if workplace is Workplace.HYBRID else Tier.TARGET_METRO_ONSITE
        )

    if _is_rare_travel_hybrid(workplace, office_days_per_month):
        return Tier.HYBRID_RARE_TRAVEL

    if _CURRENT_BASE.search(location):
        # Ranked, never dropped: a bridge he would take to gain experience before relocating.
        return Tier.CURRENT_BASE

    return Tier.OTHER_US


def _is_remote(job: Job, workplace: Workplace) -> bool:
    """Report whether a posting is genuinely remote.

    An extracted workplace type wins when we have one. Otherwise fall back to the source's flag
    or the word "remote" in the location — Ashby and Lever state it outright, Greenhouse hides
    it in metadata, and Adzuna never says at all.
    """
    if workplace is Workplace.REMOTE:
        return True
    if workplace in (Workplace.HYBRID, Workplace.ONSITE):
        return False
    return job.remote or bool(_REMOTE_TEXT.search(job.location or ""))


def _is_rare_travel_hybrid(workplace: Workplace, office_days_per_month: int | None) -> bool:
    """Report whether a hybrid role asks for the office rarely enough to fly in for.

    Needs a stated number. An unqualified "hybrid" means a normal office week until a JD says
    otherwise, and assuming otherwise would rank an unwanted relocation above his home city.

    >>> _is_rare_travel_hybrid(Workplace.HYBRID, 2)
    True
    >>> _is_rare_travel_hybrid(Workplace.HYBRID, 12)
    False
    >>> _is_rare_travel_hybrid(Workplace.HYBRID, None)
    False
    """
    if workplace is not Workplace.HYBRID or office_days_per_month is None:
        return False
    return office_days_per_month <= MAX_RARE_TRAVEL_DAYS_PER_MONTH


def mentions_target_metro(text: str | None) -> bool:
    """Report whether `text` names somewhere in his target metro (Sarasota / Tampa Bay).

    The digest uses this to flag a company that sits in his target area even when the role is
    remote — a remote job at a Tampa company is worth surfacing.

    >>> mentions_target_metro("Remote only; company in Tampa")
    True
    >>> mentions_target_metro("Chicago, IL")
    False
    >>> mentions_target_metro(None)
    False
    """
    return bool(text) and bool(_TARGET_METRO.search(text))


def rank_by_location(jobs: Sequence[Job]) -> list[Job]:
    """Return every job, best-fitting location first. Stable within a tier.

    Returns ALL of them — this ranks, it never filters.
    """
    return sorted(jobs, key=location_tier)
