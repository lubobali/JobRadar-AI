# JobRadar-AI — Capstone Build Plan

**Project:** Databricks AI Bootcamp capstone, option 5 (AI Job Hunting Copilot)
**Repo:** https://github.com/lubobali/JobRadar-AI
**Local:** `/Users/lu/15_week_bootcamp/AI Bootcamp/JobRadar-AI/`
**Target:** 100/100
**Written:** 2026-08-08 · revised 2026-08-09 with the UI decisions

---

## 1. The five requirements

Every capstone must have all five. This is the whole grading surface, so it is
the first thing in the document.

| # | Requirement | What satisfies it | Phase |
|---|---|---|---|
| 1 | **A data pipeline in Spark** | `notebooks/ingest_jobs.py` — 8 job APIs → one Spark DataFrame → dedup → Lakebase | 3 |
| 2 | **Third-party API integration** | 8 of them, already written and tested | 1, 3 |
| 3 | **Unstructured data processing** | Job descriptions: HTML → text → chunk → embed → `vector(384)` → cosine search | 4 |
| 4 | **Databricks App with frontend** | Flask app, three tabs, filters, working buttons | 6 |
| 5 | **AI agent with read/write on your DB** | MCP server, 4 read tools + 5 write tools, Agent Bricks agent | 5 |

---

## 2. Source repos

Three shipped projects supply most of this. **Nothing in them is modified** —
files are copied out.

| Repo | Where | Why it matters |
|---|---|---|
| `aws-job-streamer` | EC2 `ubuntu@18.191.209.36:~/aws-job-streamer`, GitHub `lubobali/aws-job-streamer` | The domain logic. 4,144 lines, 498 tests. **Do not touch it** — it is the live resume artifact for AWS/Terraform |
| `SkyIndex-AI` | `/Users/lu/15_week_bootcamp/AI Bootcamp/SkyIndex-AI/` | Lakebase, pgvector, embeddings, the Flask frontend |
| `SkyCast-AI` | `/Users/lu/15_week_bootcamp/AI Bootcamp/SkyCast-AI/` | The MCP server shape, bearer auth, HTTP client |

### Copy manifest

Every file, where it comes from, and which phase needs it.

| Destination | From | Source path | Phase | Change needed |
|---|---|---|---|---|
| `src/jobradar/fetchers/adzuna.py` | job-streamer | `src/aws_job_streamer/fetchers/adzuna.py` | 1 | none |
| `src/jobradar/fetchers/ashby.py` | job-streamer | same dir | 1 | none |
| `src/jobradar/fetchers/breezy.py` | job-streamer | same dir | 1 | none |
| `src/jobradar/fetchers/greenhouse.py` | job-streamer | same dir | 1 | none |
| `src/jobradar/fetchers/lever.py` | job-streamer | same dir | 1 | none |
| `src/jobradar/fetchers/remotive.py` | job-streamer | same dir | 1 | none |
| `src/jobradar/fetchers/usajobs.py` | job-streamer | same dir | 1 | none |
| `src/jobradar/fetchers/workday.py` | job-streamer | same dir | 1 | none |
| `src/jobradar/fetchers/base.py` | job-streamer | same dir | 1 | none |
| `src/jobradar/models.py` | job-streamer | `src/aws_job_streamer/models.py` | 1 | add `content_hash` |
| `src/jobradar/html_text.py` | job-streamer | `src/aws_job_streamer/html_text.py` | 1 | none |
| `src/jobradar/prefilter.py` | job-streamer | `src/aws_job_streamer/prefilter.py` | 1 | none |
| `src/jobradar/fit.py` | job-streamer | `src/aws_job_streamer/fit.py` | 1 | none |
| `src/jobradar/location_rank.py` | job-streamer | `src/aws_job_streamer/location_rank.py` | 1 | none |
| `src/jobradar/watchlist.py` | job-streamer | `src/aws_job_streamer/watchlist.py` | 1 | none |
| `src/jobradar/scoring.py` | job-streamer | `src/aws_job_streamer/scoring.py` | 4 | **swap provider**: Bedrock/OpenRouter → Databricks `claude-haiku-4-5` |
| `tests/fetchers/*`, `tests/fixtures/*` | job-streamer | `tests/` | 1 | none |
| `src/jobradar/lakebase.py` | SkyIndex-AI | `lakebase.py` | 2 | scope/key names |
| `src/jobradar/embeddings.py` | SkyIndex-AI | `embeddings.py` | 4 | none |
| `src/jobradar/repository.py` | SkyIndex-AI | `repository.py` | 2 | new tables + **write** functions |
| `src/jobradar/schema.sql` | SkyIndex-AI | `schema.sql` | 2 | nine tables instead of two |
| `setup_secrets.py` | SkyIndex-AI | `setup_secrets.py` | 2 | keys for Adzuna + USAJobs |
| `app/app.py`, `app/templates/` | SkyIndex-AI | `app.py`, `templates/` | 6 | three tabs, job cards |
| `mcp_server/jobs_mcp_server.py` | SkyCast-AI | `weather_mcp_server.py` | 5 | job tools, incl. writes |
| `mcp_server/bearer_auth.py` | SkyCast-AI | `bearer_auth.py` | 5 | none |
| `mcp_server/validation.py` | SkyCast-AI | `validation.py` | 5 | job-shaped arguments |
| `src/jobradar/http_client.py` | SkyCast-AI | `http_client.py` | 1 | none |
| `agent/system_prompt.md` | SkyCast-AI | `agent/system_prompt.md` | 5 | rewritten, same patterns |
| `scripts/smoke_test.py` | SkyCast-AI | `scripts/smoke_test.py` | 5 | job tool cases |

**Deliberately left behind** (Databricks replaces them): `pipeline.py`,
`runner.py`, `lambda_handler.py`, `dedup.py` (Spark does it), `digest.py` and
`daily_summary.py` (the frontend replaces the email), all of `infra/`.

---

## 3. How the data flows

Decided 2026-08-09. **Store wide, score narrow, filter at display time.**

```
8 sources                       ~2,000 postings fetched      Phase 3
   ↓  prefilter.py              drop killers: Azure-mandatory,
                                8+ years, wrong discipline
   ↓                            ~1,200 written to job_postings
   ↓  html_text.py              HTML → clean text
   ↓  embeddings.py             chunk + embed, local model, no API cost   Phase 4
   ↓  repository.search()       rank vs the resume embedding
   ↓  top ~200
   ↓  scoring.py                LLM score ONLY these  ← the expensive step
   ↓
   job_scores                   fit_score + reason
   ↓
   UI: all 1,200 listed, sorted by score, filter defaults to 70+   Phase 6
```

**Why not store only the >70s.** Two reasons. Semantic search needs a corpus —
keep only high scorers and "show me anything mentioning Databricks" returns
nothing. And scores are relative to the resume *as it is today*; update the
profile and a 62 becomes an 81, but only if the row still exists.

**Nothing hits an API when the user types.** Ingest is scheduled; the UI reads
what is already stored.

**Dedup, two levels:**

1. *Same source, same job* — `make_job_id` = SHA-256 of `source + source_id`.
   Already solved in `models.py`. Keyed on the source's own id, never the URL,
   because Adzuna's URL carries a per-request token that changes every fetch.
2. *Same job, two sources* — a Caterpillar role on Greenhouse also arrives via
   Adzuna. **New work.** Second key: `md5(normalized company + title + location)`.
   On collision keep the ATS row and drop the aggregator, because the ATS has the
   full description and the aggregator truncates it.

---

## 4. Data model

Lakebase, reusing the existing instance, in **its own database**. Table names
follow Zach's suggested set for option 5.

```
users               id, email, created_at
profiles            user_id, headline, summary, resume_text, target_titles[]
skills              user_id, skill, level
job_postings        id, source, source_id, company, title, url, location,
                    remote, salary, description, posted_at, fetched_at,
                    content_hash, cross_source_key
job_embeddings      job_id, chunk_index, chunk_text, embedding vector(384),
                    model_name, content_hash            ← requirement 3
job_scores          job_id, user_id, fit_score, reason, scored_at
saved_jobs          user_id, job_id, saved_at, note     ← WRITE
applications        id, user_id, job_id, status, applied_at, updated_at  ← WRITE
interview_notes     application_id, note, created_at    ← WRITE
contacts            id, user_id, company, name, role, notes  ← WRITE
```

`status` is a closed set: `interested`, `applied`, `screening`, `interviewing`,
`offer`, `rejected`, `withdrawn`.

---

## 5. The UI

Decided 2026-08-09.

```
┌──────────────────────────────────────────────────────────┐
│  [ Search ]  [ Saved ]  [ Applied ]                      │
│                                                          │
│   ┌────────────────────────────────────────────┐         │
│   │  Search jobs...                        🔍  │         │
│   └────────────────────────────────────────────┘         │
│   source: all ▾  remote ☐  last 7 days ▾  score 70+ ▾    │
│                                                          │
│   1,247 jobs                                             │
│                                                          │
│   87%  Senior Data Engineer                              │
│        Caterpillar · Chicago, IL · greenhouse · 2d ago   │
│        [ Save ]  [ Log applied ]                         │
│   ────────────────────────────────────────────────────   │
│   81%  Data Platform Engineer                            │
│        Foodsmart · Remote (US) · ashby · 4d ago          │
│        [ Save ]  [ Log applied ]                         │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  💬 Ask JobRadar...                                      │
└──────────────────────────────────────────────────────────┘
```

| Tab | What it is |
|---|---|
| **Search** | **the list of every stored job**, ranked by score, 25 per page. The search box re-ranks it semantically; it does not fetch anything |
| **Saved** | the same cards, filtered to saved, with your note |
| **Applied** | one row per application: company, title, status dropdown, last updated, notes |

Click a job title → `/job/<id>`, the full description.

**Two boxes, different jobs.** The search box filters the list. The chat bar at
the bottom is the agent, and it is where reads and writes meet: *"save the second
one"*, *"log that I applied to Caterpillar"*, *"what did I apply to last week?"*
The buttons and the chat call the same write tools, which is the point.

**Not called "pipeline".** That is data-engineer jargon for a list of job
applications. It is called **Applied**.

---

## 6. Lessons that change the plan

Do not rediscover these.

**Host the MCP server publicly from day one.** The Databricks AI Gateway cannot
authenticate to a Databricks App: no DCR (`registration_endpoint` absent), PATs
rejected (Apps take OAuth only), OAuth M2M needs a service principal
(admin-only, and this account is in `users`), OAuth U2M needs a redirect URI on
a Databricks-managed client. Cost 90 minutes on Day 3. So: **MCP server on
`jobradar.lubot.ai`, frontend as the Databricks App.**

**FastMCP discards `Returns:`.** It lifts `Args:` into the JSON schema and drops
the rest. Everything the agent must know goes *above* the `Args:` line.

**Never serve anything meaningful on `/healthz`** — Databricks Apps intercepts
it. Use `/status`.

**Use `claude-haiku-4-5`.** `gpt-5-6-sol` refuses function tools with
reasoning_effort; `claude-sonnet-5` is rate-limited here.

**This is Zach's shared workspace.** Not an admin. Cannot `CREATE CONNECTION` in
`main.default` — use `bootcamp_students.lubo_jobradar`. PATs *do* work here.

**A fake cursor records SQL, it does not parse it.** Unit tests passed while the
first live SkyIndex query died on a syntax error. Test against real Lakebase.

**Write tools need a safety story.** Every write returns what it changed. Status
is validated against the closed set. **No delete tools exist.** The agent drafts
and records; it never applies on your behalf.

---

## 7. Phases

### Phase 0 — De-risk (20 min, before any code)

- [ ] 0.1 Lakebase reachable, and a second **database** can be created in the
      existing instance
- [ ] 0.2 Spark available in a serverless notebook in this workspace
- [ ] 0.3 An App slot is free
- [ ] 0.4 **DNS**: `jobradar.lubot.ai` A → `178.156.214.8` (needs propagation,
      do it first)
- [ ] 0.5 Adzuna + USAJobs API keys located

### Phase 1 — Scaffold and port (60 min)

Nothing new is written. Everything green before moving on.

- [ ] 1.1 Repo structure (§8), `.gitignore`, `pytest.ini`, venv
- [ ] 1.2 **From `aws-job-streamer`**, via `scp` from EC2: the 8 fetchers +
      `base.py`, `models.py`, `html_text.py`, `prefilter.py`, `fit.py`,
      `location_rank.py`, `watchlist.py` — and their tests and fixtures
- [ ] 1.3 **From SkyCast-AI**: `http_client.py`
- [ ] 1.4 Rename the package `aws_job_streamer` → `jobradar` throughout
- [ ] 1.5 Full suite green (should be ~400 of the original 498)

### Phase 2 — Schema and storage (45 min)

- [ ] 2.1 **From SkyIndex-AI**: `lakebase.py`, `setup_secrets.py`,
      `schema.sql` as the pattern
- [ ] 2.2 Write the nine tables. `CREATE EXTENSION vector` first, HNSW index on
      `job_embeddings`, FKs `ON DELETE CASCADE`, idempotent DDL
- [ ] 2.3 **From SkyIndex-AI**: `repository.py`. Adapt the search to
      `job_embeddings`, keep the dedup CTE
- [ ] 2.4 **New**: the write functions — `save_job`, `log_application`,
      `update_application_status`, `add_interview_note`, `add_contact`
- [ ] 2.5 Seed one user + profile from the real resume (gitignored;
      `profile.example.json` ships)
- [ ] 2.6 Tests: fake cursor **and** real Lakebase

### Phase 3 — Spark ingest (90 min) — requirement 1

- [ ] 3.1 `notebooks/ingest_jobs.py`. Fetch all 8 sources concurrently
      (**reuses** the Phase 1 fetchers unchanged)
- [ ] 3.2 Normalize into one Spark DataFrame with an explicit `StructType`, not
      inferred, so a source changing shape fails loudly
- [ ] 3.3 Dedup level 1 in Spark on `make_job_id`, keep the freshest per id
- [ ] 3.4 Dedup level 2 on `cross_source_key`, prefer ATS over aggregator
- [ ] 3.5 `html_text` and `prefilter` as Spark UDFs
- [ ] 3.6 Write via psycopg2 `execute_values`, **not** `spark.write.jdbc` —
      JDBC cannot express `ON CONFLICT` or write pgvector types
- [ ] 3.7 Log rows in / kept / written, per source
- [ ] 3.8 Schedule it as a Databricks Job

### Phase 4 — Unstructured processing (60 min) — requirement 3

- [ ] 4.1 **From SkyIndex-AI**: `embeddings.py`
- [ ] 4.2 `notebooks/embed_jobs.py` — chunk descriptions, embed, write
      `vector(384)` inline with `%s::vector` on the first insert
- [ ] 4.3 Incremental: anti-join on `job_id` AND `content_hash` AND `model_name`
- [ ] 4.4 Embed the resume; rank jobs by cosine distance to it
- [ ] 4.5 **From `aws-job-streamer`**: `scoring.py`, provider swapped to
      Databricks `claude-haiku-4-5`. Score the top ~200 only
- [ ] 4.6 Keep its rule: **the LLM writes prose, Python computes every number**
- [ ] 4.7 Keep its fencing: a job description is untrusted input

### Phase 5 — MCP server and agent (90 min) — requirement 5

- [ ] 5.1 **From SkyCast-AI**: `weather_mcp_server.py` as the shape,
      `bearer_auth.py`, `validation.py`, `scripts/smoke_test.py`
- [ ] 5.2 Read tools: `search_jobs`, `get_job`, `list_applications`, `get_profile`
- [ ] 5.3 Write tools: `save_job`, `log_application`,
      `update_application_status`, `add_interview_note`, `add_contact`
- [ ] 5.4 Every write returns what it changed. Status validated. No deletes
- [ ] 5.5 Deploy to `jobradar.lubot.ai`: own dir, own user, own port, systemd,
      nginx + certbot, bearer token generated **on the server**
- [ ] 5.6 Register in AI Gateway under `bootcamp_students.lubo_jobradar`
- [ ] 5.7 Agent Bricks Supervisor agent, `claude-haiku-4-5`,
      `agent/system_prompt.md`
- [ ] 5.8 Demo transcripts into `agent/agent_config.md`

### Phase 6 — Databricks App frontend (2 hrs) — requirement 4

- [ ] 6.1 **From SkyIndex-AI**: `app.py` + `templates/` + CSS
- [ ] 6.2 Search tab: the list, filters, 25 per page
- [ ] 6.3 `/job/<id>` detail page
- [ ] 6.4 Saved tab
- [ ] 6.5 Applied tab: table + status dropdown + notes
- [ ] 6.6 Chat bar → the agent's serving endpoint.
      **Risk: this is the one new integration.** Fallback if it fights back:
      the bar links out to the agent Playground. Requirement 5 is still met
- [ ] 6.7 Deploy as a Databricks App, secret scope + service principal READ ACL
- [ ] 6.8 Screenshot everything the moment it works

### Phase 7 — Docs and submit (60 min)

- [ ] 7.1 README: architecture diagram, the five requirements, setup, data model
- [ ] 7.2 **Traceability table** — the thing that earned 100 three times
- [ ] 7.3 "Deliberate deviations" section
- [ ] 7.4 `agent/system_prompt.md` + `agent/agent_config.md` with transcripts
- [ ] 7.5 Full suite green, ruff clean
- [ ] 7.6 URLs at the top of the README, inside the zip
- [ ] 7.7 Zip and submit

**Total: roughly 8 hours.**

---

## 8. Structure

```
JobRadar-AI/
├── src/jobradar/
│   ├── fetchers/            8 APIs          ← aws-job-streamer
│   ├── models.py            Job + stable id ← aws-job-streamer
│   ├── html_text.py         HTML → text     ← aws-job-streamer
│   ├── prefilter.py         cheap rejects   ← aws-job-streamer
│   ├── fit.py               match scoring   ← aws-job-streamer
│   ├── location_rank.py                     ← aws-job-streamer
│   ├── watchlist.py         103 boards      ← aws-job-streamer
│   ├── scoring.py           LLM boundary    ← aws-job-streamer (provider swap)
│   ├── http_client.py       retries/pacing  ← SkyCast-AI
│   ├── lakebase.py          connection      ← SkyIndex-AI
│   ├── embeddings.py        chunk + embed   ← SkyIndex-AI
│   ├── repository.py        reads + WRITES  ← SkyIndex-AI + new
│   └── schema.sql                           ← SkyIndex-AI pattern
├── notebooks/
│   ├── ingest_jobs.py       requirement 1, Spark        NEW
│   └── embed_jobs.py        requirement 3               NEW
├── mcp_server/              requirement 5   ← SkyCast-AI
│   ├── jobs_mcp_server.py
│   ├── bearer_auth.py
│   ├── validation.py
│   ├── app.yaml
│   └── requirements.txt
├── app/                     requirement 4   ← SkyIndex-AI
│   ├── app.py
│   ├── templates/
│   ├── app.yaml
│   └── requirements.txt
├── agent/
│   ├── system_prompt.md
│   └── agent_config.md
├── scripts/smoke_test.py                    ← SkyCast-AI
├── tests/                                   ← aws-job-streamer + new
├── screenshots/
├── setup_secrets.py                         ← SkyIndex-AI
├── PLAN.md
└── README.md
```

---

## 9. Rules

1. **Screenshot the moment a thing works.** Apps here do not live forever.
2. **Nothing deploys before its tests are green.**
3. **No secrets in git.** The real resume and profile are gitignored;
   `profile.example.json` ships.
4. **Test against real Lakebase, not only mocks.**
5. **`aws-job-streamer` is never modified.** Copy out, never edit in place.
6. **The agent never deletes and never applies on your behalf.**
