"""A cheap, deterministic cut before anything expensive runs.

This module's mistakes are INVISIBLE. A job it drops is never scored, never emailed, and never
noticed — there is no error and no empty result to investigate. So it drops only what it can
prove is unusable, and geography is the only thing it can prove.

It deliberately does NOT filter on title. Both directions were measured against 651 real jobs:
  * a title ALLOW-list ("data" in title) hides 160 real matches — every "Applied AI Architect",
    "ML Systems Engineer", "Analytics Engineer" — and still lets through "Data Center Electrical
    Engineer", which is a building job. That is the exact filter that put a data-centre role in
    Lubo's first digest.
  * a title DENY-list saves only 32 of 480 LLM calls (6.7%), and after the dedup gate that is
    1-3 calls a day. It buys nothing and risks silently dropping a role whose title hides it —
    the "Software Engineer III" that is really a data platform job.
Titles are a weak signal (PLAN.md Phase 2). The LLM reads the JD body and judges. This just
stops us paying to read jobs on the wrong continent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from jobradar.models import Job

_US_SIGNAL = re.compile(
    # Postal codes for the 50 states + DC, as whole words. "CAN" must not match "CA", which is
    # why every alternative is anchored with \b.
    r"\b(?:A[LKZR]|C[AOT]|D[CE]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]"
    r"|O[HKR]|PA|RI|S[CD]|T[NX]|UT|V[AT]|W[AIVY])\b"
    r"|\bUSA?\b|\bU\.S\.A?\.?|United States|\bD\.C\."
    # Remote-scope regions that INCLUDE the US (added for Remotive's candidate_required_location,
    # which is a region not a city). A US worker qualifies for a "Worldwide"/"Americas" remote
    # role; "Brazil"/"Europe"-only ones carry no US scope and fall through to the foreign check.
    r"|Worldwide|\bAnywhere\b|North(?:ern)? America|\bAmericas\b",
    re.IGNORECASE,
)
"""Positive proof a posting is open somewhere in the US — including a US-inclusive remote scope."""

_CANADA_PREFIX = re.compile(r"^\s*CA-", re.IGNORECASE)
"""Workday's country-first format: "CA-Ontario-Toronto" is CANADA, not California. The "CA" here
is the country code, and it wrongly triggered the California US-signal, so a Toronto job leaked
into a digest. "US-CA-Menlo Park" (US first) and "San Francisco, CA" (comma) are unaffected."""

_FOREIGN_SIGNAL = re.compile(
    # Countries and territories, as whole words.
    r"\b(?:UK|United Kingdom|England|Scotland|Ireland|IE|Australia|Japan|India|Germany|DEU"
    r"|France|Spain|Italy|Netherlands|Belgium|Sweden|SE|Norway|Denmark|Finland|Poland"
    r"|Switzerland|CH|Austria|Portugal|Greece|Turkey|Israel|UAE|Kenya|Nigeria|Egypt"
    r"|Singapore|Taiwan|China|Hong Kong|South Korea|Korea|Indonesia|Malaysia|Thailand"
    r"|Vietnam|Philippines|Brazil|Argentina|Chile|Colombia|Mexico|Canada|CAN|ON|BC|QC"
    r"|Uruguay|Peru|Ecuador|Venezuela|Bolivia|Paraguay"
    # Region locks (Remotive's candidate_required_location). US-inclusive scopes win FIRST in
    # is_us_eligible, so "Americas, Europe" stays kept; only a scope with no US region hits these.
    r"|Europe|EMEA|APAC|LATAM|EU)\b"
    # Foreign cities. Half the foreign postings on real boards name no country at all — just
    # "Berlin", "Milan", "Jakarta". A country-only list would let every one of them through.
    r"|London|Berlin|Paris|Milan|Rome|Munich|Frankfurt|Hamburg|Dublin|Amsterdam|Brussels"
    r"|Stockholm|Gothenburg|Copenhagen|Oslo|Helsinki|Warsaw|Prague|Vienna|Lisbon|Madrid"
    r"|Barcelona|Z[uü]rich|Geneva|Istanbul|Tel Aviv|Dubai|Abu Dhabi|Nairobi|Lagos|Cairo"
    r"|Mumbai|Bengaluru|Bangalore|Hyderabad|Delhi|Chennai|Pune|Jakarta|Manila|Bangkok"
    r"|Singapore|Seoul|Tokyo|Osaka|Shanghai|Beijing|Shenzhen|Sydney|Melbourne|Brisbane"
    r"|Perth|Auckland|Toronto|Vancouver|Montreal|Ottawa|Calgary|S[aã]o Paulo|Rio de Janeiro"
    r"|Buenos Aires|Bogot[aá]|Santiago|Lima|Mexico City|Guadalajara",
    re.IGNORECASE,
)
"""Evidence a posting is somewhere Lubo cannot work. Only consulted when no US signal exists."""


def is_us_eligible(location: str | None) -> bool:
    """Report whether a posting might be workable from the US.

    Deliberately asymmetric, because the two mistakes are not equal. Dropping a real US job
    makes it invisible forever; keeping a foreign one costs one LLM call and one glance. So a
    posting is dropped ONLY when it names somewhere foreign and nowhere American.

    A US signal always wins. Real boards list one job across several countries, and
    "London, UK; Ontario, CAN; Remote-Friendly, United States; San Francisco, CA" is a job Lubo
    can take. It also settles the namesakes: "Paris, TX" is Texas, "Paris, France" is not.

    Unknown means keep. Adzuna names a city and county and never a state or country, so
    demanding positive proof of US-ness would silently delete every Adzuna job we have.

    >>> is_us_eligible("San Francisco, CA")
    True
    >>> is_us_eligible("Sydney, Australia")
    False
    >>> is_us_eligible("London, UK; Remote-Friendly, United States; San Francisco, CA")
    True
    >>> is_us_eligible("Chicago, Cook County")     # adzuna: no state, no country
    True
    >>> is_us_eligible("Paris, TX")
    True
    >>> is_us_eligible("Ontario, CAN")             # CAN is not California
    False
    >>> is_us_eligible("CA-Ontario-Toronto")       # Workday: CA = Canada country code, not Calif.
    False
    >>> is_us_eligible("US-CA-Menlo Park")          # US first -> genuinely California
    True
    >>> is_us_eligible(None)
    True
    """
    if not location:
        return True
    if _CANADA_PREFIX.match(location):
        return False
    if _US_SIGNAL.search(location):
        return True
    return not _FOREIGN_SIGNAL.search(location)


def keep_worth_scoring(jobs: Sequence[Job]) -> list[Job]:
    """Return the jobs worth paying an LLM to read, in their original order."""
    return [job for job in jobs if is_us_eligible(job.location)]
