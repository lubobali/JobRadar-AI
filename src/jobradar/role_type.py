"""Role type, work mode and week: the three columns the gold demand table groups by.

Titles are a weak signal for MATCHING a job to a person (see prefilter.py), which is why the
pipeline lets an LLM read the description instead. For COUNTING demand in aggregate a title is
good enough, as long as the known traps ("Data Center ...", a finance "Analyst") stay out of the
data buckets. A title that fits nothing lands in `other`, so every posting is counted once.

Pure functions, no Spark: the notebook wraps them as UDFs, and the tests run them directly.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta

ROLE_TYPES: tuple[str, ...] = (
    "ml_ai_engineer",
    "data_engineer",
    "analytics_engineer",
    "data_scientist",
    "data_analyst",
    "software_engineer",
    "other",
)
"""Every value `classify_role` can return. The gold checks reject anything else."""

WORK_MODES: tuple[str, ...] = ("remote", "not_remote")
"""Every value `work_mode` can return. No hybrid: the source has only a remote boolean."""

_NOT_A_DATA_JOB = re.compile(r"\bdata\s+cent(?:er|re)s?\b", re.IGNORECASE)
"""A data centre is a building. Checked first so "Data Center Electrical Engineer" is `other`."""

# Ordered: the first match wins. ML/AI goes before data engineering so "ML Data Engineer" counts
# as ML; analytics engineering before data engineering for the same reason.
_BUCKETS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ml_ai_engineer",
        re.compile(
            r"\b(?:machine\s+learning|ml|mlops|ai|genai|llm|deep\s+learning|nlp"
            r"|computer\s+vision)\b",
            re.IGNORECASE,
        ),
    ),
    ("analytics_engineer", re.compile(r"\banalytics\s+engineer", re.IGNORECASE)),
    (
        "data_engineer",
        re.compile(
            r"\bdata\s+(?:engineer|platform|infrastructure|pipeline|architect|warehouse)"
            r"|\betl\b|\bbig\s+data\b",
            re.IGNORECASE,
        ),
    ),
    ("data_scientist", re.compile(r"\bdata\s+scien", re.IGNORECASE)),
    (
        "data_analyst",
        re.compile(
            r"\b(?:data|business\s+intelligence|bi|analytics|reporting)\s+analyst"
            r"|\bbi\s+(?:developer|engineer)",
            re.IGNORECASE,
        ),
    ),
    (
        "software_engineer",
        re.compile(
            r"\bsoftware\s+(?:engineer|developer)"
            r"|\b(?:back[\s-]?end|front[\s-]?end|full[\s-]?stack|android|ios|mobile)"
            r"\s+(?:engineer|developer)",
            re.IGNORECASE,
        ),
    ),
)


def classify_role(title: str | None) -> str:
    """Put a job title in one of `ROLE_TYPES`. Unknown or missing titles are `other`."""
    if not title or not title.strip():
        return "other"
    if _NOT_A_DATA_JOB.search(title):
        return "other"
    for role, pattern in _BUCKETS:
        if pattern.search(title):
            return role
    return "other"


def work_mode(remote: bool | None) -> str:
    """`remote` or `not_remote`. FALSE is the column default, so it never means "onsite"."""
    return "remote" if remote else "not_remote"


def week_start(posted_at: datetime | None, fetched_at: datetime | None) -> date | None:
    """The Monday (UTC) of the week a posting belongs to, or None if it has no date at all.

    Uses `posted_at` and falls back to `fetched_at`, because some boards never send a posted
    date and dropping those rows would hide real demand. A naive datetime is taken as UTC:
    the notebook sets the Spark session time zone to UTC, so that is what a UDF receives.
    """
    when = posted_at or fetched_at
    if when is None:
        return None
    if when.tzinfo is not None:
        when = when.astimezone(UTC)
    day = when.date()
    return day - timedelta(days=day.weekday())
