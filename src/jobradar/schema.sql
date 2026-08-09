-- JobRadar-AI schema for Lakebase (Databricks-managed Postgres + pgvector).
--
-- Idempotent by construction: every statement is IF NOT EXISTS, so applying
-- this file to an existing database is a no-op rather than an error. It runs
-- as one script through lakebase.apply_schema().
--
-- The vector width (384) is tied to sentence-transformers/all-MiniLM-L6-v2.
-- Changing the embedding model means changing this number and re-embedding
-- everything. repository.verify_schema() checks the two agree at startup, so a
-- mismatch surfaces immediately instead of as an insert failure halfway
-- through a batch.

CREATE EXTENSION IF NOT EXISTS vector;


-- ===========================================================================
-- Who the jobs are for
-- ===========================================================================

CREATE TABLE IF NOT EXISTS users (
    id          BIGSERIAL PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- resume_text is the second unstructured input, alongside the job
-- descriptions. It gets embedded too, and matching is the cosine distance
-- between the two - which is why a profile update legitimately changes every
-- score, and why job rows are never deleted for scoring low.
CREATE TABLE IF NOT EXISTS profiles (
    user_id        BIGINT PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
    headline       TEXT,
    summary        TEXT,
    resume_text    TEXT,
    target_titles  TEXT[] NOT NULL DEFAULT '{}',
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS skills (
    id       BIGSERIAL PRIMARY KEY,
    user_id  BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    skill    TEXT NOT NULL,
    level    TEXT,
    UNIQUE (user_id, skill)
);


-- ===========================================================================
-- The jobs themselves
-- ===========================================================================

-- id is make_job_id(source, source_id): sha256 of the source and the id that
-- source assigns. Keyed on the source's own id and never on the url, because
-- Adzuna's redirect_url carries a per-request token - hashing it would mint a
-- fresh id every poll and the same posting would arrive forever.
CREATE TABLE IF NOT EXISTS job_postings (
    id                TEXT PRIMARY KEY,
    source            TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    company           TEXT NOT NULL,
    title             TEXT NOT NULL,
    url               TEXT NOT NULL,
    location          TEXT,
    remote            BOOLEAN NOT NULL DEFAULT FALSE,
    salary            TEXT,
    salary_is_estimated BOOLEAN NOT NULL DEFAULT FALSE,
    description       TEXT NOT NULL DEFAULT '',
    posted_at         TIMESTAMPTZ,
    fetched_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- sha256 of description. A board can edit a posting in place, so "have we
    -- embedded this job" is a question about this revision of it, not the id.
    content_hash      TEXT NOT NULL,
    -- md5 of normalized company + title + location. The SECOND dedup level:
    -- the same Caterpillar role arrives from Greenhouse AND from Adzuna under
    -- different source ids, so make_job_id cannot see they are one job. On a
    -- collision the ATS row wins, because aggregators truncate the description
    -- and the description is the whole unstructured pipeline.
    cross_source_key  TEXT NOT NULL,
    UNIQUE (source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_job_postings_cross_source
    ON job_postings (cross_source_key);
CREATE INDEX IF NOT EXISTS idx_job_postings_posted_at
    ON job_postings (posted_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_job_postings_source
    ON job_postings (source);
CREATE INDEX IF NOT EXISTS idx_job_postings_remote
    ON job_postings (remote);


-- ---------------------------------------------------------------------------
-- The unstructured half. Requirement 3.
--
-- chunk_text is stored beside its vector rather than recomputed from the
-- description at query time. Recomputing would make the retrieved passage
-- depend on the chunker's CURRENT settings, so tuning CHUNK_SIZE later would
-- silently change what already-stored vectors claim to represent.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_embeddings (
    id            TEXT PRIMARY KEY,
    job_id        TEXT NOT NULL REFERENCES job_postings (id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    chunk_text    TEXT NOT NULL,
    embedding     VECTOR(384) NOT NULL,
    model_name    TEXT NOT NULL,
    -- Carried from the revision this vector was produced from, so a re-run can
    -- tell "already embedded" from "the posting has been edited since".
    content_hash  TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, chunk_index)
);

-- HNSW rather than IVFFlat: it builds on an empty table, which a fresh deploy
-- is, and it does not need rebuilding as rows accumulate.
--
-- vector_cosine_ops matches the <=> operator used at query time. An index
-- built for a different distance function is silently ignored by the planner,
-- which looks exactly like the index not helping.
CREATE INDEX IF NOT EXISTS idx_job_embeddings_hnsw
    ON job_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_job_embeddings_job_id
    ON job_embeddings (job_id);


-- The expensive step, so only the semantically-nearest few hundred jobs get a
-- row here. Per user, because the score is a judgement about a particular
-- resume and stops meaning anything without one.
CREATE TABLE IF NOT EXISTS job_scores (
    job_id     TEXT NOT NULL REFERENCES job_postings (id) ON DELETE CASCADE,
    user_id    BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    fit_score  INTEGER NOT NULL CHECK (fit_score BETWEEN 0 AND 100),
    reason     TEXT,
    model_name TEXT,
    scored_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_job_scores_user_score
    ON job_scores (user_id, fit_score DESC);


-- ===========================================================================
-- What the user does about them. Everything below here is the WRITE surface,
-- reachable from the UI buttons and from the agent's write tools - the same
-- functions, so the two can never drift.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS saved_jobs (
    user_id   BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    job_id    TEXT NOT NULL REFERENCES job_postings (id) ON DELETE CASCADE,
    note      TEXT,
    saved_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, job_id)
);


-- status is a closed set enforced here as well as in validation. The agent
-- writes to this table, and a model asked to "mark it as in progress" will
-- happily invent a status that no query filters on ever again.
CREATE TABLE IF NOT EXISTS applications (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    job_id      TEXT NOT NULL REFERENCES job_postings (id) ON DELETE CASCADE,
    status      TEXT NOT NULL DEFAULT 'applied'
                CHECK (status IN ('interested', 'applied', 'screening',
                                  'interviewing', 'offer', 'rejected',
                                  'withdrawn')),
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- One application per job per user. Asking twice is a status change, not a
    -- second application, and without this the agent can create duplicates by
    -- being asked the same thing in two different ways.
    UNIQUE (user_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_applications_user_status
    ON applications (user_id, status);


CREATE TABLE IF NOT EXISTS interview_notes (
    id              BIGSERIAL PRIMARY KEY,
    application_id  BIGINT NOT NULL
                    REFERENCES applications (id) ON DELETE CASCADE,
    note            TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_interview_notes_application
    ON interview_notes (application_id, created_at DESC);


CREATE TABLE IF NOT EXISTS contacts (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    company    TEXT NOT NULL,
    name       TEXT NOT NULL,
    role       TEXT,
    notes      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_contacts_user_company
    ON contacts (user_id, company);
