"""The one internal job schema that every source normalizes into.

Phase 1 fills the ingest fields only. The scoring fields from PLAN.md's data model
(fit_score, fit_reason, skip_flags, status, drafted_answer) arrive in Phase 2 and are
deliberately absent until there is code that sets them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

_ID_DELIMITER = "\x00"
"""Separates the parts so ("green", "house/x") cannot collide with ("greenhouse", "/x")."""


def make_job_id(source: str, source_id: str) -> str:
    """Return the stable dedup key for a posting: a SHA-256 of its source and the source's own id.

    Keyed on `source_id` rather than the url (PLAN.md Decision Log #1). Adzuna's `redirect_url`
    carries a per-request token, so the same posting arrives at a different url every fetch —
    hashing it would mint a fresh id on every 15-minute poll and re-email the same job forever.
    Every source exposes a stable id of its own; that is what makes "never see the same job
    twice" actually hold.

    >>> make_job_id("greenhouse", "5311686008") == make_job_id("greenhouse", "5311686008")
    True
    >>> make_job_id("greenhouse", "5311686008") == make_job_id("lever", "5311686008")
    False
    >>> len(make_job_id("greenhouse", "5311686008"))
    64
    """
    return hashlib.sha256(f"{source}{_ID_DELIMITER}{source_id}".encode()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class Job:
    """A single posting, normalized across every source.

    Frozen because a fetched posting is a fact: enriching it later (Phase 2 scoring)
    produces a new record rather than mutating this one.
    """

    source: str
    source_id: str
    """The id the source itself assigns. Stable across fetches — unlike the url."""
    company: str
    title: str
    url: str
    location: str | None = None
    remote: bool = False
    salary: str | None = None
    salary_is_estimated: bool = False
    """True when the SOURCE guessed the number rather than the employer stating it.

    Adzuna predicts a salary for ~2/3 of results and presents it identically to a real one
    (a min==max point estimate). Ashby's comes from the employer. Dropping a guess entirely
    would lose information; showing it unlabelled would be a lie — so the flag travels with it.
    """
    description: str = ""
    posted_at: datetime | None = None
    fetched_at: datetime | None = None

    @property
    def job_id(self) -> str:
        """The dedup key, derived rather than stored so it can never drift from the source id.

        >>> job = Job(source="lever", source_id="66acb66f", company="Acme", title="DE",
        ...           url="https://jobs.lever.co/acme/66acb66f")
        >>> job.job_id == make_job_id("lever", "66acb66f")
        True
        """
        return make_job_id(self.source, self.source_id)
