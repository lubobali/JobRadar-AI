# JobRadar-AI — capstone submission

**Lubo Bali** · Databricks AI Bootcamp capstone, **option 5: AI Job Hunting
Copilot** · 2026-08-09

---

## Live links

| What | Where |
|---|---|
| **Databricks App** (requirement 4) | https://jobradar-ai-1352785079224954.aws.databricksapps.com |
| **GitHub repo** | https://github.com/lubobali/JobRadar-AI |
| **MCP server** (requirement 5) | https://jobradar.lubot.ai/mcp · [tool list + status](https://jobradar.lubot.ai/status) |
| **Agent** | `jobradar-agent`, Databricks Agent Bricks (workspace `bootcamp_students`) |
| **MCP service in Unity Catalog** | `bootcamp_students.lubo_jobradar.jobradar_mcp` |
| **Database** | Lakebase, schema `jobradar`, 11 tables |
| **Delta tables** | `bootcamp_students.lubo_jobradar.jobradar_{applications,saved_jobs,job_postings,analytics_daily}`, CDF enabled |

The App requires a Databricks login. `https://jobradar.lubot.ai/status` is
public and returns the live row counts:

```json
{"status":"ok","server":"jobradar","tools":11,
 "counts":{"jobs":5718,"embeddings":60860,"scores":300,"saved":3,"applications":2}}
```

---

## The six required elements

| # | Requirement | Implementation | Proof it ran |
|---|---|---|---|
| 1 | A data pipeline in Spark | [`notebooks/ingest_jobs.py`](notebooks/ingest_jobs.py) — 129 source specs fanned out as a UDF over a DataFrame on serverless compute, exploded, deduplicated twice, written to Lakebase | **5,718 postings.** Run output in [screenshot 01](screenshots/01-req1-spark-ingest-run.png) shows both dedup passes as separate numbers: `fetched 8314 → after_id_dedup 8222 → after_cross_source_dedup 8145` |
| 2 | Third-party API integration | [`src/jobradar/fetchers/`](src/jobradar/fetchers/) — 8 clients: Greenhouse, Ashby, Lever, Workday, Remotive, Breezy, Adzuna, USAJobs | 6 sources producing data: greenhouse 3,040 · ashby 2,335 · **adzuna 177** · lever 97 · workday 50 · remotive 19 |
| 3 | Processing of unstructured data | Job descriptions: HTML → text → contextual chunks → `vector(384)` ([`embeddings.py`](src/jobradar/embeddings.py)), **and** an LLM reading each description against the resume ([`scoring.py`](src/jobradar/scoring.py)) | **60,860+ vectors**; **300 LLM-scored**, scores spanning 8–92, mean 42 |
| 4 | A Databricks App with a frontend | [`app/`](app/) — Flask + Jinja. Four tabs: ranked Search with filters, Saved, Applied, and Ask (a full conversation with the agent). Job pages carry three Draft buttons | Live at the App URL. [Screenshots 03, 04, 05, 11, 12, 13](screenshots/) |
| 5 | An AI agent that reads **and writes** | [`mcp_server/jobs_mcp_server.py`](mcp_server/jobs_mcp_server.py) — **11 tools, 5 read and 6 write** — plus [`agent/system_prompt.md`](agent/system_prompt.md) | Transcripts with tool calls in [`agent/agent_config.md`](agent/agent_config.md). [Screenshots 06–10](screenshots/) |
| 6 | **Change Data Feed → Delta analytics table** | [`notebooks/cdf_analytics.py`](notebooks/cdf_analytics.py) — Lakebase mirrored into Delta with `delta.enableChangeDataFeed = true`, `table_changes()` read back, aggregated into `jobradar_analytics_daily`, published to Lakebase for the App | 4 Delta tables, CDF confirmed on each; change rows `applications 2 · saved_jobs 3 · job_postings 5,718`; **Insights** tab in the App |

## Option 5's own capability list

The job-hunting option names six things the agent should do. All six work.

| Capability | Where | Evidence |
|---|---|---|
| Search and rank postings against the profile | `search_jobs`, `fit_score` | [08](screenshots/08-req5-agent-search-answer.png) |
| Explain why a posting is or is not a match | `fit_reason` + `get_profile` | [05](screenshots/05-req5-app-agent-multiturn-save.png) |
| Save to a pipeline stage | `save_job`, `log_application`, `update_application_status` | [09](screenshots/09-req5-agent-log-application-write.png) |
| Draft a cover-letter snippet or resume bullet | `draft_application_text` + three buttons on the job page | [11](screenshots/11-req4-draft-cover-letter.png), [12](screenshots/12-req4-draft-resume-bullets.png), [13](screenshots/13-req4-draft-outreach.png) |
| Track interview notes and follow-up dates | `add_interview_note`, `set_follow_up` | verified against the live database |
| Surface stale applications | `list_applications(stale_days=…)` | verified against the live database |

Required tables — users, profiles, skills, job postings, applications, saved
jobs, interview notes, contacts — are all present, plus `job_embeddings`,
`job_scores` and `analytics_daily`. Eleven in total:
[`src/jobradar/schema.sql`](src/jobradar/schema.sql),
[screenshot 02](screenshots/02-lakebase-schema-10-tables.png) (taken at ten,
before the analytics table).

---

## Where to look first

1. **[`README.md`](README.md)** — architecture diagram, the six problems that
   cost real debugging time, and the deliberate deviations.
2. **[`screenshots/README.md`](screenshots/README.md)** — every image, and what
   to look at in it.
3. **[`agent/agent_config.md`](agent/agent_config.md)** — the agent's
   configuration and four transcripts, including the one where a write lands
   and is read back out.

## Running the tests

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements-dev.txt
./venv/bin/pip install -e .
./venv/bin/python -m pytest -q
```

**842 tests, no credentials or network needed**, in under two seconds. A further
26 hit a real Lakebase and skip themselves when `LAKEBASE_URL` is unset. Ruff
clean across `src`, `tests`, `scripts`, `mcp_server`, `app` and `notebooks`.

## Two things this submission does not hide

**USAJobs returns no data.** Its key answers 401 to a direct curl with the
documented headers — the key is dead, not missing. The client is written and
tested and reports the failure as data, so the run completes. Adzuna, the other
keyed source, works and contributes 177 postings.

**Drafting runs in the App, not on the MCP host.** It needs a Databricks
identity to reach a Foundation Model, and this workspace disables personal
access tokens, so the outside host running the MCP server cannot hold one. The
MCP tool ships anyway, uses the same module and prompt, and reports itself
unavailable rather than pretending.

Both are explained at more length under "Deliberate deviations" in the README.
