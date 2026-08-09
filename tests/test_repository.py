"""The repository, against a fake cursor.

These prove the shape of what gets sent: the right columns, the right conflict
targets, the right parameter order. They cannot prove the SQL parses - a fake
cursor records a string, it does not read it. That is what the `live` tests are
for, and on the previous project every unit test passed while the first real
query died on a syntax error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from jobradar import repository

Captured = tuple[list, dict]
"""What the `captured` fixture yields: every (sql, params) pair the module
sent, and a dict the test stages return values in."""


class FakeCursor:
    """Records statements and returns whatever the test queued."""

    def __init__(self, results: list | None = None) -> None:
        self.statements: list[tuple[str, object]] = []
        self.results = list(results or [])
        self.rowcount = 0

    def execute(self, sql: str, params: object = None) -> None:
        self.statements.append((sql, params))
        self.rowcount = 1

    def fetchone(self) -> dict | None:
        return self.results.pop(0) if self.results else None

    def fetchall(self) -> list:
        return self.results

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> FakeCursor:
        return self._cursor

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture
def captured(monkeypatch: MonkeyPatch) -> Captured:
    """Capture every SQL string and parameter tuple the module sends."""
    calls: list[tuple[str, object]] = []
    state: dict = {"rows": [], "one": None}

    def run_query(sql: str, params: object = None) -> list:
        calls.append((sql, params))
        return state["rows"]

    def run_query_one(sql: str, params: object = None) -> dict | None:
        calls.append((sql, params))
        return state["one"]

    def run_write(sql: str, params: object = None) -> int:
        calls.append((sql, params))
        return state.get("written", 1)

    monkeypatch.setattr(repository.lakebase, "run_query", run_query)
    monkeypatch.setattr(repository.lakebase, "run_query_one", run_query_one)
    monkeypatch.setattr(repository.lakebase, "run_write", run_write)
    return calls, state


class TestVectorLiteral:
    def test_renders_the_pgvector_form(self) -> None:
        assert repository.to_vector_literal([0.5] * 384).startswith("[0.5,")

    def test_rejects_the_wrong_width(self) -> None:
        # Caught here rather than by the driver halfway through a batch, where
        # the error names neither the model nor the column.
        with pytest.raises(ValueError, match="384"):
            repository.to_vector_literal([0.1, 0.2, 0.3])

    def test_the_message_says_what_disagrees(self) -> None:
        with pytest.raises(ValueError, match=r"schema\.sql"):
            repository.to_vector_literal([0.1])


class TestCrossSourceKey:
    """The second dedup level. make_job_id cannot see that a Greenhouse
    posting and its Adzuna copy are one job."""

    def test_same_job_from_two_sources_collides(self) -> None:
        args = ("Caterpillar", "Senior Data Engineer", "Chicago, IL")
        greenhouse = repository.cross_source_key(*args)
        adzuna = repository.cross_source_key(*args)
        assert greenhouse == adzuna

    def test_house_style_does_not_matter(self) -> None:
        # Boards write the same title differently. "Sr." and "Sr" are one job.
        assert repository.cross_source_key(
            "Caterpillar", "Sr. Data Engineer", "Chicago, IL"
        ) == repository.cross_source_key("Caterpillar", "Sr Data Engineer", "chicago il")

    def test_case_and_spacing_do_not_matter(self) -> None:
        assert repository.cross_source_key(
            "CATERPILLAR", "  Data   Engineer ", "Chicago, IL"
        ) == repository.cross_source_key("caterpillar", "Data Engineer", "Chicago IL")

    def test_a_different_city_is_a_different_job(self) -> None:
        # Same title, same company, two offices. Two jobs.
        assert repository.cross_source_key(
            "Caterpillar", "Data Engineer", "Chicago, IL"
        ) != repository.cross_source_key("Caterpillar", "Data Engineer", "Peoria, IL")

    def test_a_different_company_is_a_different_job(self) -> None:
        assert repository.cross_source_key(
            "Caterpillar", "Data Engineer", "Chicago, IL"
        ) != repository.cross_source_key("Foodsmart", "Data Engineer", "Chicago, IL")

    def test_a_missing_location_does_not_crash(self) -> None:
        assert repository.cross_source_key("Caterpillar", "Data Engineer", None)


class TestUpsertJobs:
    def _job(self, **overrides: object) -> dict:
        job = {
            "id": "abc123",
            "source": "greenhouse",
            "source_id": "5311686008",
            "company": "Caterpillar",
            "title": "Senior Data Engineer",
            "url": "https://example.test/j/1",
            "location": "Chicago, IL",
            "remote": False,
            "salary": None,
            "salary_is_estimated": False,
            "description": "Build data pipelines.",
            "posted_at": None,
            "fetched_at": None,
        }
        job.update(overrides)
        return job

    def test_nothing_to_write_makes_no_call(self, monkeypatch: MonkeyPatch) -> None:
        called = []
        monkeypatch.setattr(repository.lakebase, "get_connection", lambda: called.append(1))
        assert repository.upsert_jobs([]) == 0
        assert called == []

    def test_derives_content_hash_and_cross_source_key(self) -> None:
        row = repository._as_row(self._job())
        assert row["content_hash"] == repository.content_hash("Build data pipelines.")
        assert row["cross_source_key"] == repository.cross_source_key(
            "Caterpillar", "Senior Data Engineer", "Chicago, IL"
        )

    def test_an_explicit_hash_is_respected(self) -> None:
        row = repository._as_row(self._job(content_hash="given"))
        assert row["content_hash"] == "given"

    def test_a_missing_description_is_empty_not_none(self) -> None:
        job = self._job()
        del job["description"]
        assert repository._as_row(job)["description"] == ""

    def test_fetched_at_is_not_refreshed_on_conflict(self, monkeypatch: MonkeyPatch) -> None:
        # It records when the posting was FIRST seen, which is what "posted 3
        # days ago" falls back to when a board omits posted_at. Refreshing it
        # would make every job look brand new on every poll.
        cursor = FakeCursor()
        monkeypatch.setattr(
            repository.lakebase, "get_connection", lambda: FakeConnection(cursor)
        )
        captured_sql: list[str] = []
        monkeypatch.setattr(
            repository, "execute_values",
            lambda cur, sql, values, **kw: captured_sql.append(sql),
        )
        repository.upsert_jobs([self._job()])
        sql = captured_sql[0]
        assert "ON CONFLICT (id) DO UPDATE" in sql
        assert "fetched_at = EXCLUDED.fetched_at" not in sql
        assert "description = EXCLUDED.description" in sql


class TestFetchUnembedded:
    def test_anti_join_covers_hash_and_model_not_just_id(self, captured: Captured) -> None:
        # On id alone an edited posting counts as done; on id + hash a change
        # of embedding model goes unnoticed. All three, or the incremental run
        # is silently wrong.
        calls, _ = captured
        repository.fetch_unembedded_jobs("all-MiniLM-L6-v2")
        sql = calls[0][0]
        assert "e.job_id = j.id" in sql
        assert "e.content_hash = j.content_hash" in sql
        assert "e.model_name = %s" in sql

    def test_skips_jobs_with_no_description(self, captured: Captured) -> None:
        calls, _ = captured
        repository.fetch_unembedded_jobs("m")
        assert "j.description <> ''" in calls[0][0]


class TestSearch:
    def test_orders_by_the_bare_distance_operator(self, captured: Captured) -> None:
        # Only the bare form can be answered by the HNSW index. Wrapping it as
        # "1 - (...) DESC" still returns correct rows, from a full scan, which
        # is the worst kind of regression because nothing looks broken.
        calls, _ = captured
        repository.search([0.1] * 384, user_id=1)
        sql = calls[0][0]
        assert "ORDER BY e.embedding <=> %s::vector" in sql

    def test_deduplicates_per_job_then_per_real_world_job(self, captured: Captured) -> None:
        calls, _ = captured
        repository.search([0.1] * 384, user_id=1)
        sql = calls[0][0]
        assert "DISTINCT ON (job_id)" in sql
        assert "DISTINCT ON (cross_source_key)" in sql

    def test_it_does_not_deduplicate_on_the_chunk_text(self, captured: Captured) -> None:
        """The bug this replaced, kept as a test so it cannot come back.

        Hashing the chunk text was inherited from a corpus of weather alerts,
        where identical wording did mean the same alert reissued per county.
        Job postings are not like that: every role at one company shares a
        boilerplate paragraph, so forty DIFFERENT jobs collapsed into one and a
        top_k of 300 returned 31 rows. Same text is not the same job.
        """
        calls, _ = captured
        repository.search([0.1] * 384, user_id=1)
        sql = calls[0][0]
        assert "chunk_key" not in sql
        assert "regexp_replace" not in sql

    def test_overfetches_candidates_before_deduplicating(self, captured: Captured) -> None:
        calls, _ = captured
        repository.search([0.1] * 384, user_id=1, top_k=5)
        params = calls[0][1]
        assert repository.MIN_CANDIDATES in params

    def test_parameter_order_puts_filters_inside_the_cte(self, captured: Captured) -> None:
        # The filters live in the CTE, so their parameters sit between the two
        # vector literals and the candidate limit. Wrong order produces a query
        # that runs and filters on the wrong thing.
        calls, _ = captured
        repository.search([0.1] * 384, user_id=7, top_k=5, source="greenhouse")
        params = calls[0][1]
        assert params[2] == "greenhouse"
        assert params[3] == repository.MIN_CANDIDATES
        assert params[-1] == 5

    def test_no_filter_means_no_where_clause_in_the_cte(self, captured: Captured) -> None:
        calls, _ = captured
        repository.search([0.1] * 384, user_id=1)
        assert "j.source = %s" not in calls[0][0]

    def test_a_bad_embedding_never_reaches_sql(self, captured: Captured) -> None:
        calls, _ = captured
        with pytest.raises(ValueError):
            repository.search([0.1, 0.2], user_id=1)
        assert calls == []


class TestListJobs:
    def test_unscored_jobs_still_appear(self, captured: Captured) -> None:
        # The default view must not depend on anything being scored yet. A job
        # fetched ten minutes ago has no score and is still a job.
        calls, _ = captured
        repository.list_jobs(user_id=1)
        sql = calls[0][0]
        assert "LEFT JOIN job_scores" in sql
        assert "NULLS LAST" in sql

    def test_filters_are_applied(self, captured: Captured) -> None:
        calls, _ = captured
        repository.list_jobs(user_id=1, source="lever", remote_only=True, posted_within_days=7)
        sql, params = calls[0]
        assert "j.source = %s" in sql
        assert "j.remote IS TRUE" in sql
        assert "make_interval(days => %s)" in sql
        assert "lever" in params and 7 in params

    def test_reports_whether_the_user_saved_or_applied(self, captured: Captured) -> None:
        calls, _ = captured
        repository.list_jobs(user_id=1)
        sql = calls[0][0]
        assert "saved" in sql
        assert "application_status" in sql


class TestWrites:
    def test_saving_twice_updates_rather_than_failing(self, captured: Captured) -> None:
        calls, state = captured
        state["one"] = {"user_id": 1, "job_id": "j", "note": "n"}
        repository.save_job(1, "j", "n")
        assert "ON CONFLICT (user_id, job_id) DO UPDATE" in calls[0][0]

    def test_saving_without_a_note_keeps_the_existing_one(self, captured: Captured) -> None:
        calls, state = captured
        state["one"] = {}
        repository.save_job(1, "j")
        assert "COALESCE(EXCLUDED.note, saved_jobs.note)" in calls[0][0]

    def test_every_write_returns_the_row_it_wrote(self, captured: Captured) -> None:
        # The agent has to be able to say what it changed. "Done" is not a
        # report.
        calls, state = captured
        state["one"] = {}
        repository.save_job(1, "j")
        repository.update_application_status(1, 5, "screening")
        repository.add_contact(1, "Caterpillar", "Jane")
        for sql, _ in calls:
            assert "RETURNING" in sql

    def test_applying_twice_is_a_status_change_not_a_duplicate(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Without this the agent creates duplicates simply by being asked the
        # same thing two different ways.
        cursor = FakeCursor(results=[{"id": 1, "status": "applied"}])
        monkeypatch.setattr(
            repository.lakebase, "get_connection", lambda: FakeConnection(cursor)
        )
        repository.log_application(1, "job-1")
        sql = cursor.statements[0][0]
        assert "ON CONFLICT (user_id, job_id) DO UPDATE" in sql

    @pytest.mark.parametrize("status", repository.APPLICATION_STATUSES)
    def test_every_valid_status_is_accepted(self, monkeypatch: MonkeyPatch, status: str) -> None:
        cursor = FakeCursor(results=[{"id": 1}])
        monkeypatch.setattr(
            repository.lakebase, "get_connection", lambda: FakeConnection(cursor)
        )
        repository.log_application(1, "job-1", status)

    def test_an_invented_status_is_refused_before_sql(self, monkeypatch: MonkeyPatch) -> None:
        # The agent writes here. Told to "mark it in progress" a model will
        # happily invent a status that no query ever filters on again.
        called = []
        monkeypatch.setattr(
            repository.lakebase, "get_connection", lambda: called.append(1)
        )
        with pytest.raises(ValueError, match="in progress"):
            repository.log_application(1, "job-1", "in progress")
        assert called == []

    def test_the_refusal_lists_the_valid_statuses(self, captured: Captured) -> None:
        with pytest.raises(ValueError, match="interviewing"):
            repository.update_application_status(1, 5, "nonsense")

    def test_a_note_cannot_be_attached_to_someone_elses_application(
        self, captured: Captured
    ) -> None:
        # The agent passes an application id it read from an earlier tool
        # result, and nothing else stops it passing one that is not the user's.
        calls, state = captured
        state["one"] = None
        assert repository.add_interview_note(1, 999, "hello") is None
        assert len(calls) == 1
        assert "WHERE id = %s AND user_id = %s" in calls[0][0]

    def test_nothing_deletes_a_job_an_application_or_a_note(self) -> None:
        """The write surface is additive. Read the source and prove it.

        Requirement 5 wants an agent that writes, and an agent that writes is
        an agent that can be talked into destroying something. The guarantee is
        not "the prompt says not to" - it is that no function here can.

        Exactly two DELETEs exist: one removes a bookmark, one replaces a job's
        vectors atomically on re-embed. Neither destroys anything a person
        produced.
        """
        text = Path(repository.__file__).read_text(encoding="utf-8")
        deletes = [line.strip() for line in text.splitlines() if "DELETE FROM" in line]

        assert len(deletes) == 2, f"unexpected DELETE statements: {deletes}"
        assert any("saved_jobs" in line for line in deletes)
        assert any("{EMBEDDINGS_TABLE}" in line for line in deletes)

        for table in ("job_postings", "applications", "interview_notes", "contacts", "users"):
            assert not any(table in line for line in deletes), f"something deletes {table}"
