"""Every repository function, against a real Lakebase.

Skipped unless LAKEBASE_URL is set, so the fast suite stays offline:

    LAKEBASE_URL='postgresql://...' pytest -m live

**Why this file exists.** The rest of the suite mocks the cursor, which proves
the shape of what gets sent - right columns, right conflict target, right
parameter order - and cannot prove the SQL parses. A fake cursor records a
string; it does not read it.

On the previous project every unit test passed while the first live query died
with `syntax error at or near "ozone"`, because a `\\n` inside a Python string
had broken a SQL comment. Nothing in a mocked suite can catch that. So every
statement this module issues gets executed here at least once.

Each test seeds its own user and job under a random id and deletes them
afterwards, so a live run leaves the database as it found it and two runs
never collide.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import psycopg2
import pytest

from jobradar import lakebase, repository

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("LAKEBASE_URL"),
        reason="LAKEBASE_URL is not set; skipping the live Lakebase tests",
    ),
]


@pytest.fixture
def seeded() -> Iterator[dict]:
    """A real user, profile, and job posting, written through the module itself."""
    email = f"live-{uuid.uuid4().hex[:8]}@example.test"
    job_id = f"live-{uuid.uuid4().hex[:8]}"

    user = lakebase.run_query_one(
        "INSERT INTO users (email) VALUES (%s) RETURNING id", (email,)
    )
    user_id = user["id"]
    lakebase.run_write(
        "INSERT INTO profiles (user_id, headline, resume_text) VALUES (%s, %s, %s)",
        (user_id, "Senior Data & AI Platform Engineer", "Spark, Python, Postgres, AWS."),
    )
    repository.upsert_jobs(
        [
            {
                "id": job_id,
                "source": "greenhouse",
                "source_id": job_id,
                "company": "Caterpillar",
                "title": "Senior Data Engineer",
                "url": "https://example.test/1",
                "location": "Chicago, IL",
                "remote": False,
                "salary": None,
                "salary_is_estimated": False,
                "description": "Build and operate Spark data pipelines on AWS.",
                "posted_at": None,
                "fetched_at": None,
            }
        ]
    )

    yield {"user_id": user_id, "job_id": job_id}

    # ON DELETE CASCADE takes the profile, saved rows, applications, notes and
    # contacts with the user, and the embeddings and scores with the job.
    lakebase.run_write("DELETE FROM users WHERE id = %s", (user_id,))
    lakebase.run_write("DELETE FROM job_postings WHERE id = %s", (job_id,))


class TestSchema:
    def test_the_stored_vector_width_matches_the_model(self) -> None:
        repository.verify_schema()

    def test_every_table_exists(self) -> None:
        rows = lakebase.run_query(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'jobradar'
            """
        )
        found = {row["table_name"] for row in rows}
        assert {
            "users", "profiles", "skills", "job_postings", "job_embeddings",
            "job_scores", "saved_jobs", "applications", "interview_notes",
            "contacts",
        } <= found

    def test_the_hnsw_index_is_built_for_cosine(self) -> None:
        # An index built for a different distance function is silently ignored
        # by the planner, which looks exactly like the index not helping.
        row = lakebase.run_query_one(
            "SELECT indexdef FROM pg_indexes WHERE indexname = %s",
            ("idx_job_embeddings_hnsw",),
        )
        assert row is not None
        assert "hnsw" in row["indexdef"]
        assert "vector_cosine_ops" in row["indexdef"]


class TestReads:
    def test_list_jobs_returns_the_seeded_job(self, seeded: dict) -> None:
        rows = repository.list_jobs(user_id=seeded["user_id"], limit=200)
        assert any(row["id"] == seeded["job_id"] for row in rows)

    def test_count_matches_the_filters(self, seeded: dict) -> None:
        assert repository.count_jobs(user_id=seeded["user_id"]) >= 1
        assert (
            repository.count_jobs(user_id=seeded["user_id"], source="not-a-real-source") == 0
        )

    def test_every_filter_parses(self, seeded: dict) -> None:
        # The point is that each clause is valid SQL, not that it matches.
        repository.list_jobs(
            user_id=seeded["user_id"],
            source="greenhouse",
            remote_only=True,
            posted_within_days=7,
            min_score=70,
        )

    def test_search_runs_on_an_empty_index(self, seeded: dict) -> None:
        # A freshly deployed database has no vectors. That is a normal state
        # and must return zero results, not fail.
        assert repository.search([0.1] * 384, user_id=seeded["user_id"]) == []

    def test_search_with_filters_parses(self, seeded: dict) -> None:
        # This is the query with the two CTEs, the whitespace-stripping md5,
        # and eight positional parameters in a specific order. If any of that
        # is wrong, it is wrong here and nowhere else.
        repository.search(
            [0.1] * 384,
            user_id=seeded["user_id"],
            top_k=5,
            source="greenhouse",
            remote_only=False,
            posted_within_days=30,
        )

    def test_get_job(self, seeded: dict) -> None:
        job = repository.get_job(seeded["job_id"], user_id=seeded["user_id"])
        assert job["title"] == "Senior Data Engineer"

    def test_get_job_missing_returns_none(self, seeded: dict) -> None:
        assert repository.get_job("no-such-job", user_id=seeded["user_id"]) is None

    def test_get_profile_includes_skills(self, seeded: dict) -> None:
        lakebase.run_write(
            "INSERT INTO skills (user_id, skill) VALUES (%s, %s)",
            (seeded["user_id"], "Spark"),
        )
        profile = repository.get_profile(seeded["user_id"])
        assert "Spark" in profile["skills"]

    def test_stats(self) -> None:
        stats = repository.stats()
        assert set(stats) == {"jobs", "embeddings", "scores", "saved", "applications"}


class TestWrites:
    def test_save_then_list(self, seeded: dict) -> None:
        repository.save_job(seeded["user_id"], seeded["job_id"], "looks good")
        saved = repository.list_saved(seeded["user_id"])
        assert saved[0]["job_id"] == seeded["job_id"]
        assert saved[0]["note"] == "looks good"

    def test_saving_twice_does_not_duplicate(self, seeded: dict) -> None:
        repository.save_job(seeded["user_id"], seeded["job_id"], "first")
        repository.save_job(seeded["user_id"], seeded["job_id"], "second")
        assert len(repository.list_saved(seeded["user_id"])) == 1

    def test_unsave(self, seeded: dict) -> None:
        repository.save_job(seeded["user_id"], seeded["job_id"])
        assert repository.unsave_job(seeded["user_id"], seeded["job_id"]) is True
        assert repository.list_saved(seeded["user_id"]) == []

    def test_log_application_then_read_it_back(self, seeded: dict) -> None:
        app = repository.log_application(
            seeded["user_id"], seeded["job_id"], "applied", "applied via careers page"
        )
        assert app["status"] == "applied"
        rows = repository.list_applications(seeded["user_id"])
        assert rows[0]["job_id"] == seeded["job_id"]
        assert rows[0]["notes"][0]["note"] == "applied via careers page"

    def test_applying_twice_updates_rather_than_duplicating(self, seeded: dict) -> None:
        # The constraint that stops the agent creating two applications by
        # being asked the same thing two different ways.
        first = repository.log_application(seeded["user_id"], seeded["job_id"], "interested")
        second = repository.log_application(seeded["user_id"], seeded["job_id"], "applied")
        assert first["id"] == second["id"]
        assert len(repository.list_applications(seeded["user_id"])) == 1

    def test_status_moves_through_the_pipeline(self, seeded: dict) -> None:
        app = repository.log_application(seeded["user_id"], seeded["job_id"])
        for status in ("screening", "interviewing", "offer"):
            updated = repository.update_application_status(
                seeded["user_id"], app["id"], status
            )
            assert updated["status"] == status

    def test_the_check_constraint_backs_up_the_python_validation(
        self, seeded: dict
    ) -> None:
        # Python refuses an invented status before SQL sees it. This proves the
        # database would refuse it too, which is what protects the table from
        # anything that does not go through this module.
        with pytest.raises(psycopg2.errors.CheckViolation):
            lakebase.run_write(
                "INSERT INTO applications (user_id, job_id, status) VALUES (%s, %s, %s)",
                (seeded["user_id"], seeded["job_id"], "in progress"),
            )

    def test_a_note_on_someone_elses_application_is_refused(self, seeded: dict) -> None:
        app = repository.log_application(seeded["user_id"], seeded["job_id"])
        stranger = lakebase.run_query_one(
            "INSERT INTO users (email) VALUES (%s) RETURNING id",
            (f"stranger-{uuid.uuid4().hex[:8]}@example.test",),
        )
        try:
            assert repository.add_interview_note(stranger["id"], app["id"], "hi") is None
        finally:
            lakebase.run_write("DELETE FROM users WHERE id = %s", (stranger["id"],))

    def test_add_contact(self, seeded: dict) -> None:
        contact = repository.add_contact(
            seeded["user_id"], "Caterpillar", "Jane", "Recruiter", "met at a meetup"
        )
        assert contact["name"] == "Jane"


class TestEmbeddings:
    def test_a_new_job_is_pending(self, seeded: dict) -> None:
        pending = repository.fetch_unembedded_jobs("all-MiniLM-L6-v2", limit=500)
        assert any(row["id"] == seeded["job_id"] for row in pending)

    def test_write_then_search_finds_it(self, seeded: dict) -> None:
        # The whole vector path end to end: a real vector(384) written with the
        # inline ::vector cast, then retrieved through the two-CTE query.
        vector = [0.0] * 384
        vector[0] = 1.0
        repository.replace_job_embeddings(
            seeded["job_id"],
            [
                {
                    "chunk_index": 0,
                    "chunk_text": "Build and operate Spark data pipelines on AWS.",
                    "embedding": vector,
                    "content_hash": repository.content_hash("x"),
                }
            ],
            model_name="all-MiniLM-L6-v2",
        )
        hits = repository.search(vector, user_id=seeded["user_id"], top_k=5)
        assert any(hit["id"] == seeded["job_id"] for hit in hits)
        assert hits[0]["similarity"] > 0.99

    def test_re_embedding_replaces_rather_than_appending(self, seeded: dict) -> None:
        # Delete-then-insert, not upsert on (job_id, chunk_index). A shorter
        # revision produces fewer chunks, and an upsert would leave the surplus
        # tail of the previous revision in place - stale text that still scores.
        vector = [0.0] * 384
        vector[0] = 1.0
        chunks = [
            {
                "chunk_index": index,
                "chunk_text": f"chunk {index}",
                "embedding": vector,
                "content_hash": "h1",
            }
            for index in range(3)
        ]
        repository.replace_job_embeddings(seeded["job_id"], chunks, "all-MiniLM-L6-v2")
        repository.replace_job_embeddings(seeded["job_id"], chunks[:1], "all-MiniLM-L6-v2")

        row = lakebase.run_query_one(
            "SELECT count(*) AS total FROM job_embeddings WHERE job_id = %s",
            (seeded["job_id"],),
        )
        assert row["total"] == 1

    def test_scores_upsert(self, seeded: dict) -> None:
        repository.upsert_scores(
            [
                {
                    "job_id": seeded["job_id"],
                    "user_id": seeded["user_id"],
                    "fit_score": 87,
                    "reason": "Spark and AWS both present",
                    "model_name": "claude-haiku-4-5",
                }
            ]
        )
        job = repository.get_job(seeded["job_id"], user_id=seeded["user_id"])
        assert job["fit_score"] == 87

    def test_a_scored_job_sorts_above_an_unscored_one(self, seeded: dict) -> None:
        repository.upsert_scores(
            [
                {
                    "job_id": seeded["job_id"],
                    "user_id": seeded["user_id"],
                    "fit_score": 99,
                    "reason": "top match",
                    "model_name": "test",
                }
            ]
        )
        rows = repository.list_jobs(user_id=seeded["user_id"], limit=5)
        assert rows[0]["id"] == seeded["job_id"]
