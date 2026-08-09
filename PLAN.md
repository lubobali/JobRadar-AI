# JobRadar-AI — Capstone Build Plan

**Project:** Databricks AI Bootcamp capstone, option 5 (AI Job Hunting Copilot)
**Repo:** https://github.com/lubobali/JobRadar-AI
**Local:** `/Users/lu/15_week_bootcamp/AI Bootcamp/JobRadar-AI/`
**Target:** 100/100
**Written:** 2026-08-08, build starts 2026-08-09

---

## The five requirements, and what satisfies each

Every capstone must have all five. This is the whole grading surface, so it is
the first thing in this document.

| # | Requirement | What satisfies it here |
|---|---|---|
| 1 | **A data pipeline in Spark** | `notebooks/ingest_jobs.py` — fetch from 8 job APIs, normalize into one schema in a Spark DataFrame, dedup, rank, write to Lakebase |
| 2 | **Third-party API integration** | 8 of them: Greenhouse, Ashby, Lever, Adzuna, Remotive, USAJobs, Workday, Breezy |
| 3 | **Unstructured data processing** | Job descriptions. HTML to text, chunked, embedded, stored as `vector(384)` in Lakebase pgvector, searched by cosine distance |
| 4 | **Databricks App with frontend** | Flask app: semantic search over jobs, match scores, save a job, log an application, move it through the pipeline |
| 5 | **AI agent with read/write on your database** | MCP server with read tools *and* write tools, driven by an Agent Bricks agent. The write half is the part that matters |

---

## Where everything comes from

This is not a green field. Three shipped projects supply most of it.

### From `aws-job-streamer` (~1,500 lines, zero AWS coupling)

Audited 2026-08-08: 4,144 lines of source, 498 tests, Terraform, CI.

| File | Lines | Ports as |
|---|---|---|
| `fetchers/*.py` (8 files) | ~1,100 | The API layer. Every one exposes `SOURCE` and `fetch_jobs(...)` — one uniform interface |
| `models.py` | 92 | The `Job` dataclass and `make_job_id`, the stable dedup key |
| `fit.py` | 242 | Match scoring |
| `prefilter.py` | 114 | Cheap rejects before anything expensive runs |
| `watchlist.py` | 344 | 103 company boards |
| `location_rank.py` | 175 | Workable-location ranking |
| `html_text.py` | 34 | HTML to text, the front of the unstructured pipeline |

**Left behind on purpose:** `dedup`, `digest`, `runner`, `pipeline`,
`lambda_handler` — the only five modules that import boto3. Databricks replaces
what they do.

**`aws-job-streamer` is not touched.** It stays public and untouched, because it
is the live resume artifact proving AWS/Terraform. This is a separate repo that
reuses its modules.

### From `SkyIndex-AI` (Day 2, 100/100)

| File | Ports as |
|---|---|
| `lakebase.py` | Connection + pooling. Straight copy |
| `embeddings.py` | Chunking and embedding. Point it at job descriptions |
| `repository.py` | The pgvector search SQL, including the dedup CTE that took four attempts |
| `schema.sql` | The DDL pattern: idempotent, HNSW index, `content_hash` on both tables |
| `app.py` + `templates/` | The Flask frontend skeleton — most of requirement 4 |
| `setup_secrets.py` | Secret scope + service principal READ ACL |

### From `SkyCast-AI` (Day 3, 100/100)

| File | Ports as |
|---|---|
| `weather_mcp_server.py` | The MCP server shape: thin `@mcp.tool` functions, nothing raises |
| `bearer_auth.py` | Auth for the public MCP instance |
| `http_client.py` | Pacing, bounded retries, readable errors |
| `validation.py` | Cleaning tool arguments |
| `agent/system_prompt.md` | The guardrail patterns |

---

## Hard-won lessons that change the plan

Do not rediscover these.

**Host the MCP server publicly from the start.** The Databricks AI Gateway
cannot authenticate to a Databricks App. Not DCR (no `registration_endpoint`),
not a PAT (Apps take OAuth only), not OAuth M2M (service principals are
admin-only and this account is in `users`), not OAuth U2M (redirect URI is
Databricks-managed). Cost 90 minutes on Day 3. The MCP server goes on
`jobradar.lubot.ai` behind a bearer token on day one, and the *frontend* is the
Databricks App.

**FastMCP discards `Returns:`.** It lifts `Args:` into the JSON schema and drops
`Returns:` entirely. Anything the agent must know goes above the `Args:` line.

**Never serve anything meaningful on `/healthz`.** Databricks Apps intercepts it.
Use `/status`.

**Use `claude-haiku-4-5`.** `gpt-5-6-sol` refuses function tools with
reasoning_effort; `claude-sonnet-5` is rate-limited in this workspace.

**This is Zach's shared workspace, not the Free Edition account.** Not an admin.
Cannot `CREATE CONNECTION` in `main.default` — use `bootcamp_students.lubo_*`.
PATs *do* work here.

**Write tools need a confirmation story.** Requirement 5 is read *and write*.
An agent that can write is an agent that can write the wrong thing, so every
write tool returns what it changed, and destructive operations are not exposed
at all.

---

## Data model

Lakebase (Databricks Postgres), reusing the existing instance, in its own
database. Tables follow Zach's suggested set for option 5.

```
users               id, email, created_at
profiles            user_id, headline, summary, resume_text, target_titles[]
skills              user_id, skill, level
job_postings        id, source, source_id, company, title, url, location,
                    remote, salary, description, posted_at, fetched_at,
                    content_hash
job_embeddings      job_id, chunk_index, chunk_text, embedding vector(384),
                    model_name, content_hash          <- unstructured (req 3)
job_scores          job_id, user_id, fit_score, reason, scored_at
saved_jobs          user_id, job_id, saved_at, note
applications        id, user_id, job_id, status, applied_at, updated_at
interview_notes     application_id, note, created_at
contacts            id, user_id, company, name, role, notes
```

`saved_jobs`, `applications`, `interview_notes`, `contacts` are the **write**
surface. Everything else the agent only reads.

---

## Phases

### Phase 0 — De-risk (20 min, before any code)

- [ ] 0.1 Confirm the existing Lakebase instance is reachable and a second
      database can be created in it
- [ ] 0.2 Confirm Spark is available (serverless notebook, this workspace)
- [ ] 0.3 Confirm an App slot is free
- [ ] 0.4 `dig jobradar.lubot.ai` — add the A record early, it needs propagation

### Phase 1 — Scaffold and port (60 min)

- [ ] 1.1 Repo structure (below)
- [ ] 1.2 Copy the 8 fetchers + `models.py` + `fit.py` + `prefilter.py` +
      `watchlist.py` + `location_rank.py` + `html_text.py` from `aws-job-streamer`,
      with their tests
- [ ] 1.3 Copy `lakebase.py`, `embeddings.py`, `repository.py`, `setup_secrets.py`
      from SkyIndex-AI
- [ ] 1.4 Copy `http_client.py`, `bearer_auth.py`, `validation.py` from SkyCast-AI
- [ ] 1.5 Full test suite green before writing one new line

### Phase 2 — Schema and storage (45 min)

- [ ] 2.1 `schema.sql` — the nine tables, `CREATE EXTENSION vector` first,
      HNSW index on `job_embeddings`, FKs `ON DELETE CASCADE`
- [ ] 2.2 `repository.py` — adapt SkyIndex's search to `job_embeddings`, plus
      the write functions for saved jobs, applications, notes, contacts
- [ ] 2.3 Seed one user and one profile from his real resume (gitignored;
      `profile.example.json` in the repo)
- [ ] 2.4 Tests against a fake cursor, then **against the real Lakebase** —
      SkyIndex proved a fake cursor records SQL, it does not parse it

### Phase 3 — Spark ingest pipeline (90 min) — **requirement 1**

- [ ] 3.1 `notebooks/ingest_jobs.py`: fetch from all 8 sources concurrently
- [ ] 3.2 Normalize into one Spark DataFrame with an explicit `StructType` —
      not inferred, so a source changing shape fails loudly
- [ ] 3.3 Dedup in Spark on `make_job_id`, keep the freshest row per id
- [ ] 3.4 `html_text` + prefilter as Spark UDFs
- [ ] 3.5 Write to `job_postings` via psycopg2 `execute_values`, **not**
      `spark.write.jdbc` (SkyIndex: JDBC cannot express `ON CONFLICT` or write
      pgvector types)
- [ ] 3.6 Log rows in, rows kept, rows written, per source

### Phase 4 — Unstructured processing (60 min) — **requirement 3**

- [ ] 4.1 `notebooks/embed_jobs.py`: chunk descriptions, embed with
      `all-MiniLM-L6-v2`, write `vector(384)` inline via `%s::vector`
- [ ] 4.2 Incremental: anti-join on `job_id` AND `content_hash` AND `model_name`
- [ ] 4.3 Semantic match — embed the profile, rank jobs by cosine distance
- [ ] 4.4 LLM scoring on the top N only (cost control, same as `aws-job-streamer`)

### Phase 5 — MCP server (90 min) — **requirement 5**

Read tools:
- [ ] `search_jobs(query, top_k, filters)` — semantic search
- [ ] `get_job(job_id)` — full posting
- [ ] `list_applications(status)` — the pipeline
- [ ] `get_profile()` — skills and target roles

Write tools:
- [ ] `save_job(job_id, note)`
- [ ] `log_application(job_id, status, note)`
- [ ] `update_application_status(application_id, status)`
- [ ] `add_interview_note(application_id, note)`
- [ ] `add_contact(company, name, role, notes)`

- [ ] 5.1 Every write returns what it changed, so the agent can report it
- [ ] 5.2 No delete tools at all
- [ ] 5.3 Status is a closed set, validated: `interested`, `applied`,
      `screening`, `interviewing`, `offer`, `rejected`, `withdrawn`
- [ ] 5.4 Deploy to `jobradar.lubot.ai` behind a bearer token, systemd, own user
- [ ] 5.5 Register in AI Gateway, build the agent, write the system prompt

### Phase 6 — Databricks App frontend (75 min) — **requirement 4**

- [ ] 6.1 Flask app from SkyIndex's `app.py` + templates
- [ ] 6.2 Pages: search jobs, job detail, saved jobs, application pipeline
- [ ] 6.3 Deploy as a Databricks App, secret scope + service principal ACL
- [ ] 6.4 Screenshot everything the moment it works

### Phase 7 — Docs and submit (60 min)

- [ ] 7.1 README: architecture diagram, the five requirements mapped to files,
      setup, data model
- [ ] 7.2 **Traceability table** — the thing that earned 100 three times
- [ ] 7.3 "Deliberate deviations" section
- [ ] 7.4 `agent/system_prompt.md` + `agent/agent_config.md` with transcripts
- [ ] 7.5 Full suite green, ruff clean
- [ ] 7.6 URLs at the top of the README, in the zip
- [ ] 7.7 Zip and submit

---

## Structure

```
JobRadar-AI/
├── src/jobradar/
│   ├── fetchers/            8 job APIs, from aws-job-streamer
│   ├── models.py            the Job schema and its stable id
│   ├── fit.py               match scoring
│   ├── prefilter.py         cheap rejects
│   ├── watchlist.py         103 company boards
│   ├── location_rank.py
│   ├── html_text.py         HTML -> text
│   ├── lakebase.py          from SkyIndex-AI
│   ├── embeddings.py        from SkyIndex-AI
│   ├── repository.py        reads + WRITES
│   └── schema.sql
├── mcp_server/              requirement 5
│   ├── jobs_mcp_server.py
│   ├── bearer_auth.py
│   ├── http_client.py
│   ├── validation.py
│   ├── app.yaml
│   └── requirements.txt
├── app/                     requirement 4, the Databricks App
│   ├── app.py
│   ├── templates/
│   ├── app.yaml
│   └── requirements.txt
├── notebooks/
│   ├── ingest_jobs.py       requirement 1, Spark
│   └── embed_jobs.py        requirement 3
├── agent/
│   ├── system_prompt.md
│   └── agent_config.md
├── tests/
├── screenshots/
├── PLAN.md
└── README.md
```

---

## Rules

1. **Screenshot the moment a thing works.** Apps in this workspace do not live
   forever.
2. **Nothing deploys before its tests are green.**
3. **No secrets in git.** His real resume and profile are gitignored;
   `profile.example.json` is what ships.
4. **Test on real Lakebase, not just mocks.** A fake cursor records SQL, it does
   not parse it.
5. **`aws-job-streamer` is not touched.**
