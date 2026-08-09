"""Every SQL statement JobRadar issues, in one place.

Descended from SkyIndex-AI's repository, which ran this search against Lakebase
in production. The retrieval SQL is kept nearly intact because each awkward
clause in it was paid for once already; the comments say what each one cost.

Split into three sections:

    reads       what the Search tab and the agent's read tools call
    writes      saving, applying, notes, contacts - the surface requirement 5
                is actually about
    plumbing    ingest, embedding bookkeeping, schema checks

**Reads and writes share these functions.** The UI buttons and the agent's MCP
tools both call in here, rather than each having its own SQL. Two paths to the
same table is how "save" starts meaning two different things.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

from psycopg2.extras import execute_values

from jobradar import lakebase

logger = logging.getLogger(__name__)

JOBS_TABLE = "job_postings"
EMBEDDINGS_TABLE = "job_embeddings"
SCORES_TABLE = "job_scores"

EMBEDDING_DIM = 384
"""Tied to sentence-transformers/all-MiniLM-L6-v2 and to schema.sql.
verify_schema() checks the two agree."""

APPLICATION_STATUSES = (
    "interested",
    "applied",
    "screening",
    "interviewing",
    "offer",
    "rejected",
    "withdrawn",
)
"""Closed set, enforced here, in validation, and by a CHECK constraint. Three
places on purpose: the agent writes to this table, and a model asked to "mark
it as in progress" will cheerfully invent a status that no query filters on
ever again."""

CANDIDATE_OVERFETCH = 10
MIN_CANDIDATES = 100
"""Pull more rows out of the index than the caller asked for, because the two
dedup rounds below collapse many of them. Asking the index for exactly top_k
and then deduplicating returns fewer than top_k results."""

EF_SEARCH = int(os.environ.get("HNSW_EF_SEARCH", "400"))
"""How many candidates the HNSW index explores before answering.

**pgvector defaults this to 40**, and that default silently caps the whole
query: a LIMIT of 3000 still gets answered out of a pool of about forty, so
asking for 300 results returned 35. Nothing errors, the rows that come back are
genuinely the nearest of the ones examined, and the ones never examined are
simply invisible.

400 is measured, not guessed. Benchmarking the same index on the previous
project moved recall from 0.18 at ef_search=40 to 0.80 at 400, for a latency
cost small enough not to notice on a corpus this size. It has to be set per
transaction, because it is a session GUC and the pool hands out a different
connection each time."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_vector_literal(values: Sequence[float]) -> str:
    """Render an embedding as the string form pgvector parses.

    Width is checked here rather than left to Postgres. A dimension mismatch
    reported by the driver names neither the model nor the column, and arrives
    halfway through a batch insert.
    """
    if len(values) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding has {len(values)} dimensions, expected {EMBEDDING_DIM}. "
            "The model and schema.sql disagree."
        )
    return "[" + ",".join(f"{float(value):.7g}" for value in values) + "]"


def content_hash(text: str) -> str:
    """Fingerprint the exact text that gets embedded.

    A board can edit a posting in place under the same id, so "have we embedded
    this job" is a question about this revision of it, not about the id.
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^a-z0-9 ]+")


def cross_source_key(company: str, title: str, location: str | None) -> str:
    """The second dedup key: one value per real-world job, across sources.

    `make_job_id` cannot see that a Caterpillar posting on Greenhouse and the
    same posting relayed by Adzuna are one job - they carry different source
    ids. This hashes what actually identifies the role.

    Deliberately coarse. Punctuation stripped, case folded, whitespace
    collapsed, because "Sr. Data Engineer" and "Sr Data Engineer" are the same
    job advertised by two systems with different house style. Location is
    included, since the same title at the same company in two cities is two
    jobs.
    """
    parts = [company or "", title or "", location or ""]
    normalized = " ".join(
        _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", part.lower())).strip() for part in parts
    )
    # md5 because this is a dedup key, not a credential. Collisions here merge
    # two job postings; nothing is being protected.
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _as_row(job: Any) -> dict:  # noqa: ANN401 - a Job dataclass or a plain dict
    """Accept either a Job dataclass or a plain dict.

    The Spark ingest hands over dicts; the fetchers hand over Job instances.
    Both arrive here rather than each caller remembering to convert.
    """
    row = job.as_row() if hasattr(job, "as_row") else dict(job)
    row.setdefault("description", "")
    row["content_hash"] = row.get("content_hash") or content_hash(row["description"])
    row["cross_source_key"] = row.get("cross_source_key") or cross_source_key(
        row.get("company", ""), row.get("title", ""), row.get("location")
    )
    return row


# ---------------------------------------------------------------------------
# Plumbing - ingest and embedding bookkeeping
# ---------------------------------------------------------------------------

_JOB_COLUMNS = (
    "id", "source", "source_id", "company", "title", "url", "location",
    "remote", "salary", "salary_is_estimated", "description", "posted_at",
    "fetched_at", "content_hash", "cross_source_key",
)


def upsert_jobs(jobs: Iterable[Any]) -> int:
    """Insert or refresh job postings. Returns the number written.

    On conflict the row is refreshed rather than ignored, because a board edits
    postings in place - a salary appears, a description is rewritten. Ignoring
    the conflict would freeze whatever version happened to arrive first.

    `fetched_at` is deliberately NOT refreshed on conflict: it records when we
    first saw the posting, which is what "posted 3 days ago" is derived from
    when a board omits `posted_at`.
    """
    rows = [_as_row(job) for job in jobs]
    if not rows:
        return 0

    values = [tuple(row.get(column) for column in _JOB_COLUMNS) for row in rows]
    updates = ", ".join(
        f"{column} = EXCLUDED.{column}"
        for column in _JOB_COLUMNS
        if column not in ("id", "fetched_at")
    )
    sql = f"""
        INSERT INTO {JOBS_TABLE} ({", ".join(_JOB_COLUMNS)})
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET {updates}
    """

    with lakebase.get_connection() as conn, conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=200)
        return cur.rowcount


def fetch_unembedded_jobs(model_name: str, limit: int = 500) -> list[dict]:
    """Jobs with no current vectors, oldest first.

    An anti-join on three columns, not one. Matching on `job_id` alone would
    call an edited posting done; adding `content_hash` catches the edit, and
    `model_name` catches a change of embedding model. "Pending" is derived from
    the data rather than tracked in a cursor, so a half-finished run resumes
    correctly with no state to reconcile.
    """
    sql = f"""
        SELECT j.id, j.title, j.company, j.description, j.content_hash
        FROM {JOBS_TABLE} j
        WHERE j.description <> ''
          AND NOT EXISTS (
              SELECT 1 FROM {EMBEDDINGS_TABLE} e
              WHERE e.job_id = j.id
                AND e.content_hash = j.content_hash
                AND e.model_name = %s
          )
        ORDER BY j.fetched_at
        LIMIT %s
    """
    return lakebase.run_query(sql, (model_name, limit))


def replace_job_embeddings(job_id: str, chunks: Sequence[dict], model_name: str) -> int:
    """Replace every vector for one job, atomically.

    Delete-then-insert rather than upserting on (job_id, chunk_index). A
    shorter revision of a posting produces fewer chunks, and an upsert would
    leave the surplus tail from the previous revision in place - stale text
    that still scores and still gets returned.
    """
    if not chunks:
        return 0

    values = [
        (
            f"{job_id}:{chunk['chunk_index']}",
            job_id,
            chunk["chunk_index"],
            chunk["chunk_text"],
            to_vector_literal(chunk["embedding"]),
            model_name,
            chunk["content_hash"],
        )
        for chunk in chunks
    ]

    with lakebase.get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"DELETE FROM {EMBEDDINGS_TABLE} WHERE job_id = %s", (job_id,))
        execute_values(
            cur,
            f"""
            INSERT INTO {EMBEDDINGS_TABLE}
                (id, job_id, chunk_index, chunk_text, embedding, model_name, content_hash)
            VALUES %s
            """,
            values,
            # The cast is here, on the first insert. The pattern this was built
            # from stores double precision[] and asks the operator to run an
            # UPDATE ... ::vector by hand afterwards, which leaves the table
            # unqueryable and the HNSW index unusable in between - and is a
            # manual step that can simply be forgotten.
            template="(%s, %s, %s, %s, %s::vector, %s, %s)",
            page_size=100,
        )
        return len(values)


def upsert_scores(rows: Iterable[dict]) -> int:
    """Write LLM fit scores. Only the top few hundred jobs ever get one."""
    values = [
        (r["job_id"], r["user_id"], r["fit_score"], r.get("reason"), r.get("model_name"))
        for r in rows
    ]
    if not values:
        return 0
    with lakebase.get_connection() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO {SCORES_TABLE} (job_id, user_id, fit_score, reason, model_name)
            VALUES %s
            ON CONFLICT (job_id, user_id) DO UPDATE SET
                fit_score = EXCLUDED.fit_score,
                reason = EXCLUDED.reason,
                model_name = EXCLUDED.model_name,
                scored_at = now()
            """,
            values,
            page_size=200,
        )
        return len(values)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _filter_clauses(
    *,
    source: str | None,
    remote_only: bool,
    posted_within_days: int | None,
    min_score: int | None,
    user_id: int | None,
) -> tuple[list[str], list[Any]]:
    """The filters the Search tab exposes, shared by both list paths."""
    clauses: list[str] = []
    params: list[Any] = []
    if source:
        clauses.append("j.source = %s")
        params.append(source)
    if remote_only:
        clauses.append("j.remote IS TRUE")
    if posted_within_days:
        clauses.append(
            "COALESCE(j.posted_at, j.fetched_at) >= now() - make_interval(days => %s)"
        )
        params.append(posted_within_days)
    if min_score is not None and user_id is not None:
        clauses.append("s.fit_score >= %s")
        params.append(min_score)
    return clauses, params


def list_jobs(  # noqa: PLR0913 - each argument is one filter the Search tab exposes
    *,
    user_id: int,
    limit: int = 25,
    offset: int = 0,
    source: str | None = None,
    remote_only: bool = False,
    posted_within_days: int | None = None,
    min_score: int | None = None,
) -> list[dict]:
    """The Search tab with an empty search box: every job, best score first.

    This is the default view, so it must not depend on anything being embedded
    or scored. The join to scores is a LEFT JOIN and unscored jobs sort last
    rather than vanishing - a job fetched ten minutes ago has no score yet and
    is still a job.
    """
    clauses, params = _filter_clauses(
        source=source,
        remote_only=remote_only,
        posted_within_days=posted_within_days,
        min_score=min_score,
        user_id=user_id,
    )
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    sql = f"""
        SELECT j.id, j.source, j.company, j.title, j.url, j.location,
               j.remote, j.salary, j.posted_at, j.fetched_at,
               s.fit_score, s.reason,
               (sj.job_id IS NOT NULL) AS saved,
               a.status AS application_status
        FROM {JOBS_TABLE} j
        LEFT JOIN {SCORES_TABLE} s ON s.job_id = j.id AND s.user_id = %s
        LEFT JOIN saved_jobs sj ON sj.job_id = j.id AND sj.user_id = %s
        LEFT JOIN applications a ON a.job_id = j.id AND a.user_id = %s
        {where}
        ORDER BY s.fit_score DESC NULLS LAST,
                 COALESCE(j.posted_at, j.fetched_at) DESC
        LIMIT %s OFFSET %s
    """
    return lakebase.run_query(sql, (user_id, user_id, user_id, *params, limit, offset))


def count_jobs(
    *,
    user_id: int,
    source: str | None = None,
    remote_only: bool = False,
    posted_within_days: int | None = None,
    min_score: int | None = None,
) -> int:
    """How many rows the current filters match, for the "1,247 jobs" line."""
    clauses, params = _filter_clauses(
        source=source,
        remote_only=remote_only,
        posted_within_days=posted_within_days,
        min_score=min_score,
        user_id=user_id,
    )
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT count(*) AS total
        FROM {JOBS_TABLE} j
        LEFT JOIN {SCORES_TABLE} s ON s.job_id = j.id AND s.user_id = %s
        {where}
    """
    row = lakebase.run_query_one(sql, (user_id, *params))
    return int(row["total"]) if row else 0


def search(  # noqa: PLR0913 - the query plus the same filters as list_jobs
    embedding: Sequence[float],
    *,
    user_id: int,
    top_k: int = 25,
    source: str | None = None,
    remote_only: bool = False,
    posted_within_days: int | None = None,
) -> list[dict]:
    """Rank jobs by cosine similarity to a query embedding.

    ORDER BY is the bare distance operator ascending, never "1 - (...) DESC".
    Only the bare form can be answered by the HNSW index; wrapping it produces
    something the planner cannot match, and the query silently degrades to a
    full scan that still returns correct results - the worst kind of
    regression, because nothing looks broken.

    Runs `SET LOCAL hnsw.ef_search` in the same transaction as the query. It is
    a session GUC and the pool hands out a different connection each time, so
    setting it anywhere else would apply to some queries and not others -
    which is worse than not setting it at all, because the results would be
    inconsistent rather than uniformly poor.
    """
    vector = to_vector_literal(embedding)
    candidate_limit = max(top_k * CANDIDATE_OVERFETCH, MIN_CANDIDATES)

    clauses, filter_params = _filter_clauses(
        source=source,
        remote_only=remote_only,
        posted_within_days=posted_within_days,
        min_score=None,
        user_id=None,
    )
    filter_where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    sql = f"""
        WITH candidates AS (
            SELECT e.job_id,
                   e.chunk_index,
                   e.chunk_text,
                   j.cross_source_key,
                   e.embedding <=> %s::vector AS distance
            FROM {EMBEDDINGS_TABLE} e
            JOIN {JOBS_TABLE} j ON j.id = e.job_id
            {filter_where}
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        ),
        -- Two rounds of collapsing, because the corpus repeats itself twice
        -- over.
        --
        -- 1. One result per JOB. A long description splits into several chunks
        --    that all score well on the same query, and five paragraphs of one
        --    posting is a worse answer than five different jobs.
        per_job AS (
            SELECT DISTINCT ON (job_id) *
            FROM candidates
            ORDER BY job_id, distance
        ),
        -- 2. One result per real-world JOB. The same role published on
        --    Greenhouse and relayed by Adzuna has two different ids, so it
        --    survives round 1 untouched.
        --
        --    Keyed on cross_source_key rather than on the chunk text. Hashing
        --    the text was the first attempt, inherited from a corpus of
        --    weather alerts where identical wording really did mean the same
        --    alert reissued per county. Job postings are not like that: every
        --    role at one company shares a boilerplate paragraph - "At X, we
        --    are passionate about..." - so forty DIFFERENT jobs collapsed into
        --    one and a top_k of 300 returned 31 rows. Same text is not the
        --    same job.
        best AS (
            SELECT DISTINCT ON (cross_source_key) *
            FROM per_job
            ORDER BY cross_source_key, distance
        )
        SELECT j.id, j.source, j.company, j.title, j.url, j.location,
               j.remote, j.salary, j.posted_at, j.fetched_at,
               b.chunk_text AS matched_text,
               1 - b.distance AS similarity,
               s.fit_score, s.reason,
               (sj.job_id IS NOT NULL) AS saved,
               a.status AS application_status
        FROM best b
        JOIN {JOBS_TABLE} j ON j.id = b.job_id
        LEFT JOIN {SCORES_TABLE} s ON s.job_id = j.id AND s.user_id = %s
        LEFT JOIN saved_jobs sj ON sj.job_id = j.id AND sj.user_id = %s
        LEFT JOIN applications a ON a.job_id = j.id AND a.user_id = %s
        ORDER BY b.distance
        LIMIT %s
    """
    # Order matters and is easy to get subtly wrong: the filters live inside
    # the CTE, so their parameters land between the two vector literals and the
    # candidate limit, not at the end. Wrong order here produces a query that
    # runs, returns rows, and applies the wrong filter.
    params = (
        vector,             # the distance expression in the SELECT list
        vector,             # the same distance in ORDER BY, inside the CTE
        *filter_params,     # source / remote / posted_within, inside the CTE
        candidate_limit,    # the CTE's LIMIT
        user_id,            # scores join
        user_id,            # saved_jobs join
        user_id,            # applications join
        top_k,              # the outer LIMIT
    )
    with lakebase.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL hnsw.ef_search = %s", (EF_SEARCH,))
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def get_job(job_id: str, *, user_id: int) -> dict | None:
    """One posting in full, for the detail page."""
    sql = f"""
        SELECT j.*, s.fit_score, s.reason,
               (sj.job_id IS NOT NULL) AS saved,
               a.id AS application_id, a.status AS application_status
        FROM {JOBS_TABLE} j
        LEFT JOIN {SCORES_TABLE} s ON s.job_id = j.id AND s.user_id = %s
        LEFT JOIN saved_jobs sj ON sj.job_id = j.id AND sj.user_id = %s
        LEFT JOIN applications a ON a.job_id = j.id AND a.user_id = %s
        WHERE j.id = %s
    """
    return lakebase.run_query_one(sql, (user_id, user_id, user_id, job_id))


def get_user(email: str) -> dict | None:
    return lakebase.run_query_one("SELECT * FROM users WHERE email = %s", (email,))


def get_profile(user_id: int) -> dict | None:
    """Profile plus skills, which is what both the matcher and the agent want."""
    profile = lakebase.run_query_one("SELECT * FROM profiles WHERE user_id = %s", (user_id,))
    if profile is None:
        return None
    profile["skills"] = [
        row["skill"]
        for row in lakebase.run_query(
            "SELECT skill FROM skills WHERE user_id = %s ORDER BY skill", (user_id,)
        )
    ]
    return profile


def list_saved(user_id: int, limit: int = 100) -> list[dict]:
    sql = f"""
        SELECT j.id, j.source, j.company, j.title, j.url, j.location, j.remote,
               j.salary, j.posted_at, sj.note, sj.saved_at,
               s.fit_score, s.reason, a.status AS application_status
        FROM saved_jobs sj
        JOIN {JOBS_TABLE} j ON j.id = sj.job_id
        LEFT JOIN {SCORES_TABLE} s ON s.job_id = j.id AND s.user_id = sj.user_id
        LEFT JOIN applications a ON a.job_id = j.id AND a.user_id = sj.user_id
        WHERE sj.user_id = %s
        ORDER BY sj.saved_at DESC
        LIMIT %s
    """
    return lakebase.run_query(sql, (user_id, limit))


# An application nobody has touched in this many days is stale by default. Two
# weeks is roughly when "they will get back to me" turns into "they are not
# going to". It is a default and not a rule; the caller can say otherwise.
DEFAULT_STALE_DAYS = 14

# Statuses where silence means nothing is coming and there is nothing to chase.
CLOSED_STATUSES = ("offer", "rejected", "withdrawn")


def list_applications(
    user_id: int,
    status: str | None = None,
    stale_days: int | None = None,
) -> list[dict]:
    """The Applied tab. Notes come back as a JSON array, newest first.

    `stale_days` narrows to applications untouched for that many days and still
    open - the ones worth chasing. A rejection that has sat for a year is not
    stale, it is finished, so CLOSED_STATUSES are excluded.
    """
    clauses = []
    params: list[Any] = [user_id]
    if status:
        clauses.append("AND a.status = %s")
        params.append(status)
    if stale_days is not None:
        clauses.append(
            "AND a.updated_at < now() - make_interval(days => %s) "
            f"AND a.status <> ALL(ARRAY[{','.join(['%s'] * len(CLOSED_STATUSES))}])"
        )
        params.append(stale_days)
        params.extend(CLOSED_STATUSES)
    clause = " ".join(clauses)

    sql = f"""
        SELECT a.id, a.status, a.applied_at, a.updated_at, a.follow_up_on,
               EXTRACT(DAY FROM now() - a.updated_at)::int AS days_since_update,
               j.id AS job_id, j.company, j.title, j.url, j.location, j.source,
               COALESCE(
                   (SELECT json_agg(json_build_object('note', n.note,
                                                      'created_at', n.created_at)
                                    ORDER BY n.created_at DESC)
                    FROM interview_notes n WHERE n.application_id = a.id),
                   '[]'::json
               ) AS notes
        FROM applications a
        JOIN {JOBS_TABLE} j ON j.id = a.job_id
        WHERE a.user_id = %s {clause}
        ORDER BY a.updated_at DESC
    """
    rows = lakebase.run_query(sql, tuple(params))
    for row in rows:
        if isinstance(row.get("notes"), str):
            row["notes"] = json.loads(row["notes"])
    return rows


# ---------------------------------------------------------------------------
# Writes
#
# The surface requirement 5 is actually about. Three rules hold across all of
# them:
#
#   Every write returns the row it wrote. The agent has to be able to say what
#   it changed, and "done" is not a report.
#
#   Nothing here deletes a job posting or an application. Unsaving is the only
#   removal, and it removes a bookmark.
#
#   Status is validated against APPLICATION_STATUSES before it reaches SQL, so
#   the failure is a sentence rather than a constraint violation.
# ---------------------------------------------------------------------------


def save_job(user_id: int, job_id: str, note: str | None = None) -> dict:
    """Bookmark a job. Saving twice updates the note rather than failing."""
    sql = """
        INSERT INTO saved_jobs (user_id, job_id, note)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, job_id) DO UPDATE SET
            note = COALESCE(EXCLUDED.note, saved_jobs.note),
            saved_at = now()
        RETURNING user_id, job_id, note, saved_at
    """
    return lakebase.run_query_one(sql, (user_id, job_id, note))


def unsave_job(user_id: int, job_id: str) -> bool:
    """Remove a bookmark. The only delete in this module, and it removes a
    bookmark rather than anything that took work to produce."""
    return (
        lakebase.run_write(
            "DELETE FROM saved_jobs WHERE user_id = %s AND job_id = %s", (user_id, job_id)
        )
        > 0
    )


def log_application(
    user_id: int, job_id: str, status: str = "applied", note: str | None = None
) -> dict:
    """Record that a job was applied to.

    Idempotent on (user_id, job_id): asking twice is a status change, not a
    second application. Without that, the agent creates duplicates simply by
    being asked the same thing two different ways.
    """
    if status not in APPLICATION_STATUSES:
        raise ValueError(
            f"Unknown status {status!r}. Expected one of {', '.join(APPLICATION_STATUSES)}."
        )

    sql = """
        INSERT INTO applications (user_id, job_id, status)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, job_id) DO UPDATE SET
            status = EXCLUDED.status,
            updated_at = now()
        RETURNING id, user_id, job_id, status, applied_at, updated_at
    """
    with lakebase.get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (user_id, job_id, status))
        application = dict(cur.fetchone())
        if note:
            cur.execute(
                "INSERT INTO interview_notes (application_id, note) VALUES (%s, %s)",
                (application["id"], note),
            )
    return application


def update_application_status(user_id: int, application_id: int, status: str) -> dict | None:
    """Move an application along. Returns None if it is not this user's."""
    if status not in APPLICATION_STATUSES:
        raise ValueError(
            f"Unknown status {status!r}. Expected one of {', '.join(APPLICATION_STATUSES)}."
        )
    sql = """
        UPDATE applications
        SET status = %s, updated_at = now()
        WHERE id = %s AND user_id = %s
        RETURNING id, job_id, status, updated_at
    """
    return lakebase.run_query_one(sql, (status, application_id, user_id))


def add_interview_note(user_id: int, application_id: int, note: str) -> dict | None:
    """Attach a note to an application.

    The user_id is checked in the SELECT rather than trusted from the caller,
    because the agent passes an application id it read from a previous tool
    result and nothing else stops it passing someone else's.
    """
    owner = lakebase.run_query_one(
        "SELECT id FROM applications WHERE id = %s AND user_id = %s",
        (application_id, user_id),
    )
    if owner is None:
        return None
    return lakebase.run_query_one(
        """
        INSERT INTO interview_notes (application_id, note)
        VALUES (%s, %s)
        RETURNING id, application_id, note, created_at
        """,
        (application_id, note),
    )


def set_follow_up(user_id: int, application_id: int, follow_up_on: date | None) -> dict | None:
    """Set or clear the date to chase this application.

    Ownership is checked in the SELECT for the same reason as add_interview_note:
    the id came from a previous tool result, and nothing else stops the agent
    passing someone else's.

    Passing None clears the date, which is what "I heard back, drop the reminder"
    has to mean. It does NOT touch updated_at - setting a reminder is not
    activity on the application, and counting it as such would make an
    application look fresh precisely when it needed chasing.
    """
    owner = lakebase.run_query_one(
        "SELECT id FROM applications WHERE id = %s AND user_id = %s",
        (application_id, user_id),
    )
    if owner is None:
        return None
    return lakebase.run_query_one(
        """
        UPDATE applications
           SET follow_up_on = %s
         WHERE id = %s
        RETURNING id, status, follow_up_on, applied_at, updated_at
        """,
        (follow_up_on, application_id),
    )


def job_for_drafting(user_id: int, job_id: str) -> dict | None:
    """Everything needed to write about one posting: the job and the profile.

    One call rather than two, because a draft written from the description
    without the profile is a generic cover letter, and every one of those reads
    like a generic cover letter.
    """
    job = get_job(job_id, user_id=user_id)
    if job is None:
        return None
    return {"job": job, "profile": get_profile(user_id)}


def add_contact(
    user_id: int, company: str, name: str, role: str | None = None, notes: str | None = None
) -> dict:
    return lakebase.run_query_one(
        """
        INSERT INTO contacts (user_id, company, name, role, notes)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id, company, name, role, notes, created_at
        """,
        (user_id, company, name, role, notes),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def verify_schema() -> None:
    """Fail loudly if the stored vector width disagrees with the model.

    Checked at startup rather than discovered as a driver type error partway
    through the first batch, which names neither the model nor the column.
    """
    row = lakebase.run_query_one(
        """
        SELECT a.atttypmod AS dimensions
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = %s AND a.attname = 'embedding'
        """,
        (EMBEDDINGS_TABLE,),
    )
    if row is None:
        raise RuntimeError(
            f"{EMBEDDINGS_TABLE} has no embedding column. Run lakebase.apply_schema()."
        )
    stored = int(row["dimensions"])
    if stored != EMBEDDING_DIM:
        raise RuntimeError(
            f"{EMBEDDINGS_TABLE}.embedding is vector({stored}) but the model "
            f"produces {EMBEDDING_DIM}. Change schema.sql and re-embed."
        )


def stats() -> dict:
    """Row counts, for /status and the ingest logs."""
    sql = f"""
        SELECT (SELECT count(*) FROM {JOBS_TABLE})        AS jobs,
               (SELECT count(*) FROM {EMBEDDINGS_TABLE})  AS embeddings,
               (SELECT count(*) FROM {SCORES_TABLE})      AS scores,
               (SELECT count(*) FROM saved_jobs)          AS saved,
               (SELECT count(*) FROM applications)        AS applications
    """
    return lakebase.run_query_one(sql) or {}


__all__ = [
    "APPLICATION_STATUSES",
    "EMBEDDING_DIM",
    "add_contact",
    "add_interview_note",
    "content_hash",
    "count_jobs",
    "cross_source_key",
    "fetch_unembedded_jobs",
    "get_job",
    "get_profile",
    "get_user",
    "list_applications",
    "list_jobs",
    "list_saved",
    "log_application",
    "replace_job_embeddings",
    "save_job",
    "search",
    "stats",
    "to_vector_literal",
    "unsave_job",
    "update_application_status",
    "upsert_jobs",
    "upsert_scores",
    "verify_schema",
]
