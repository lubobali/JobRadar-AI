"""The parts of the ingest pipeline that do not need Spark.

Split out on purpose. `notebooks/ingest_jobs.py` owns the Spark orchestration -
parallelize, DataFrame, foreachPartition - and everything here is plain Python
that runs and is tested without a cluster. A rule about which duplicate wins is
a rule whether or not there is a SparkContext.

**Why source specs exist.** `watchlist.all_sources()` returns no-argument
closures, which is a good shape for a thread pool and an impossible one for
Spark: a closure cannot be pickled, so it cannot be shipped to an executor.
A `SourceSpec` is a frozen dataclass of plain strings, which can. The executor
rebuilds the fetcher from it on arrival.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from jobradar import html_text, watchlist
from jobradar.fetchers import adzuna, ashby, breezy, greenhouse, lever, remotive, usajobs, workday
from jobradar.models import Job

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Which duplicate wins
# ---------------------------------------------------------------------------

SOURCE_PRIORITY: dict[str, int] = {
    # Applicant tracking systems. The company publishes here itself, so the
    # description is the full text and the id is stable.
    "greenhouse": 0,
    "ashby": 0,
    "lever": 0,
    "breezy": 0,
    "workday": 1,
    # Official, complete, but its own format.
    "usajobs": 2,
    # Aggregators. They relay someone else's posting and truncate the
    # description, which is the entire unstructured pipeline - so they lose
    # every tie they are in.
    "remotive": 3,
    "adzuna": 4,
}
"""Lower wins. Used only when two sources carry the same real-world job."""

DEFAULT_PRIORITY = 9
"""An unrecognized source sorts last rather than crashing the run. A new
fetcher that nobody added here still ingests; it just never beats a known one
in a tie."""


def source_priority(source: str) -> int:
    return SOURCE_PRIORITY.get(source, DEFAULT_PRIORITY)


def prefer(left: dict, right: dict) -> dict:
    """Pick the better of two rows describing the same real-world job.

    Priority first, because an ATS row carries the whole description and an
    aggregator's is truncated. Length breaks a tie between two rows of equal
    priority, since the longer description is the more complete one. Both are
    deterministic, which matters: a run that picks a different winner each time
    re-embeds the same job forever.
    """
    left_rank = (source_priority(left["source"]), -len(left.get("description") or ""))
    right_rank = (source_priority(right["source"]), -len(right.get("description") or ""))
    return left if left_rank <= right_rank else right


# ---------------------------------------------------------------------------
# Source specs - what gets shipped to an executor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One fetch, described in strings so it can cross a network boundary.

    `kind` selects the fetcher; the rest are its arguments. Frozen and made of
    primitives, so pickling it is trivial and an executor can rebuild the call
    without needing the closure that built it.
    """

    kind: str
    slug: str = ""
    company: str = ""
    query: str = ""
    where: str = ""
    # Workday alone needs three values to address a board (tenant, site, host)
    # rather than one slug. Carried as plain strings so the spec stays a bag of
    # primitives and pickles without ceremony.
    tenant: str = ""
    site: str = ""
    host: str = ""

    @property
    def label(self) -> str:
        """What shows up in logs when this one fails."""
        detail = self.company or self.slug or self.query
        return f"{self.kind}:{detail}" if detail else self.kind


def source_specs() -> list[SourceSpec]:
    """Every fetch a full run performs, as serializable descriptors.

    Built from the same watchlist the AWS pipeline used, so the 103
    probe-verified boards and the tuned search terms come across unchanged.
    """
    specs: list[SourceSpec] = [
        SourceSpec(kind=board.source, slug=board.slug, company=board.company)
        for board in watchlist.WATCHLIST
    ]
    specs += [
        SourceSpec(kind="remotive", query=search) for search in watchlist.REMOTIVE_SEARCHES
    ]
    specs += [
        SourceSpec(kind="adzuna", query=query.phrase, where=query.where or "")
        for query in watchlist.ADZUNA_QUERIES
    ]
    specs += [
        SourceSpec(kind="usajobs", query=query.keyword, where=query.location_name or "")
        for query in watchlist.USAJOBS_QUERIES
    ]
    specs += [
        SourceSpec(kind="breezy", slug=slug, company=company)
        for slug, company in watchlist._BREEZY_BOARDS
    ]
    specs += [
        SourceSpec(
            kind="workday",
            company=company,
            query=term,
            tenant=board.tenant,
            site=board.site,
            host=board.host,
        )
        for board, company in watchlist._WORKDAY_BOARDS
        for term in watchlist.WORKDAY_SEARCHES
    ]
    return specs


def _workday(spec: SourceSpec) -> list[Job]:
    """Workday needs a board object built from three parts, so it gets a name."""
    board = workday.WorkdayBoard(spec.tenant, spec.site, spec.host)
    # max_results is small deliberately: each hit costs a second hydration call,
    # and the yield is a handful of clearance roles.
    return workday.fetch_jobs(board, spec.query, company=spec.company or None, max_results=10)


DISPATCH: dict[str, Any] = {
    "greenhouse": lambda s: greenhouse.fetch_jobs(s.slug),
    "lever": lambda s: lever.fetch_jobs(s.slug, company=s.company or None),
    "ashby": lambda s: ashby.fetch_jobs(s.slug, company=s.company or None),
    "breezy": lambda s: breezy.fetch_jobs(s.slug, company=s.company or None),
    "remotive": lambda s: remotive.fetch_jobs(s.query),
    "adzuna": lambda s: adzuna.fetch_jobs(s.query, where=s.where or None),
    "usajobs": lambda s: usajobs.fetch_jobs(s.query, location_name=s.where or None),
    "workday": _workday,
}
"""kind -> how to run it. A table rather than an if-chain, so adding a source is
one line here plus one in source_specs(), and so an unregistered kind is a
lookup miss with a clear message rather than a silent fall-through returning
nothing."""


def fetch_spec(spec: SourceSpec) -> tuple[list[Job], str | None]:
    """Run one spec. Returns its jobs and, if it failed, why.

    Never raises. This runs on an executor across a hundred-odd sources, and
    one dead board must not sink the run - the whole point of fanning out is
    that failures are independent. The error travels back as data so the driver
    can report which sources came up empty, rather than a silent short count.
    """
    handler = DISPATCH.get(spec.kind)
    if handler is None:
        return [], f"{spec.label}: no fetcher registered for kind {spec.kind!r}"

    try:
        return handler(spec), None
    except Exception as exc:
        logger.warning("%s failed: %s", spec.label, exc)
        return [], f"{spec.label}: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Normalizing one job into one row
# ---------------------------------------------------------------------------

ROW_FIELDS = (
    "id", "source", "source_id", "company", "title", "url", "location",
    "remote", "salary", "salary_is_estimated", "description", "posted_at",
    "fetched_at", "content_hash", "cross_source_key",
)
"""The DataFrame's columns, in order. Declared here rather than inferred,
because an inferred schema silently changes shape when a source starts
returning a field it did not before."""


def to_row(job: Job) -> dict[str, Any]:
    """Flatten one Job into the row the DataFrame carries.

    HTML is stripped here rather than at query time. The description is the
    input to the whole unstructured pipeline, and storing the markup would mean
    every consumer - the chunker, the LLM scorer, the UI - either strips it
    again or accidentally embeds `<div>`.
    """
    # Imported here rather than at module scope so this module does not import
    # repository, which imports psycopg2, which an executor need not have.
    from jobradar.repository import content_hash, cross_source_key  # noqa: PLC0415

    description = html_text.to_plain_text(job.description or "")

    return {
        # Derived on Job rather than stored, so it can never drift from the
        # source id it is built out of.
        "id": job.job_id,
        "source": job.source,
        "source_id": job.source_id,
        "company": job.company,
        "title": job.title,
        "url": job.url,
        "location": job.location,
        "remote": bool(job.remote),
        "salary": job.salary,
        "salary_is_estimated": bool(job.salary_is_estimated),
        "description": description,
        "posted_at": job.posted_at,
        "fetched_at": job.fetched_at or datetime.now(UTC),
        "content_hash": content_hash(description),
        "cross_source_key": cross_source_key(job.company, job.title, job.location),
    }


_EPOCH = datetime.min.replace(tzinfo=UTC)
"""Sorts before every real timestamp. Timezone-aware on purpose: the fetchers
return aware datetimes, and comparing one to a naive datetime.min raises
"can't compare offset-naive and offset-aware datetimes" - which only shows up
on the rows where fetched_at happens to be missing."""


def _fetched_at(row: dict) -> datetime:
    return row.get("fetched_at") or _EPOCH


def deduplicate(rows: list[dict]) -> list[dict]:
    """Both dedup rounds, in plain Python.

    The Spark job does this with window functions over the whole DataFrame;
    this is the same rule, small enough to read and to test, and it is what
    the tests assert against. Two implementations of one rule would be a
    problem - so the Spark version calls `source_priority` and `prefer` from
    here rather than re-deciding.

    Round 1: one row per `id`, keeping the most recently fetched. The same
    board polled twice in one run returns the same posting twice.

    Round 2: one row per `cross_source_key`, keeping the better source. This is
    the Caterpillar role that arrives from Greenhouse and again from Adzuna
    under a different id, which round 1 cannot see is one job.
    """
    by_id: dict[str, dict] = {}
    for row in rows:
        existing = by_id.get(row["id"])
        if existing is None or _fetched_at(row) > _fetched_at(existing):
            by_id[row["id"]] = row

    by_job: dict[str, dict] = {}
    for row in by_id.values():
        key = row["cross_source_key"]
        existing = by_job.get(key)
        by_job[key] = row if existing is None else prefer(existing, row)

    return list(by_job.values())


__all__ = [
    "DEFAULT_PRIORITY",
    "ROW_FIELDS",
    "SOURCE_PRIORITY",
    "SourceSpec",
    "deduplicate",
    "fetch_spec",
    "prefer",
    "source_priority",
    "source_specs",
    "to_row",
]
