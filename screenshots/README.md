# Screenshots

Evidence that each capstone requirement runs, in the order a reader should look
at them. Every number in these images is from the real system, not a fixture.

---

### 1. `01-req1-spark-ingest-run.png` — requirements 1 and 2

The Spark ingest notebook's final report cell, on serverless compute, with the
run's own output underneath:

```
sources 129 · sources_failed 16 · fetched 8314 · after_id_dedup 8222
after_cross_source_dedup 8145 · after_prefilter 5539 · written 5539
```

Both deduplication passes are visible as separate numbers: 8,314 fetched become
8,222 after `job_id` (the same posting fetched twice), then 8,145 after
`cross_source_key` (the same job listed on two boards).

> Taken on an earlier run, when Adzuna's credentials were not reaching the Spark
> executors — hence 16 failed sources and 5,539 written. Adzuna works now and the
> corpus is 5,718 across six sources; see screenshot 14. The cause is written up
> under "The parts that were not obvious" in the main README.

### 2. `02-lakebase-schema-10-tables.png` — the data model

`schema.sql` applied to Lakebase, and the ten tables it created read back from
`information_schema`. This is also why the schema is called `jobradar` rather
than being its own database: `CREATE DATABASE` is refused on Lakebase.

### 3. `03-req4-app-search-ranked.png` — requirement 4

The Databricks App's Search tab. Every stored job ranked by fit score, filters
for source, remote, recency and score, and working Save / Log applied buttons on
each card. Each score carries the reason the LLM gave for it.

> Taken before the agent moved to its own **Ask** tab, so the chat appears here
> as an overlay at the bottom. Screenshots 4 and 5 show the current layout.

### 4. `04-req4-app-agent-numbered-search.png` — requirements 4 and 5

The **Ask** tab. Ten roles, numbered, with company, location, salary, and the
fit score where one exists. The numbering is a prompt rule, not a formatting
accident — without it "the second one" in the next screenshot has nothing to
count.

### 5. `05-req5-app-agent-multiturn-save.png` — requirement 5, the write path

Three turns in one image, and the most complete piece of evidence here:

- **"Am I a fit for the second one?"** — resolved positionally to the Snowflake
  role, answered against `get_profile` and `get_job` together. It leads with
  *"Not a strong fit. The role scored 28"* and explains that the job is about
  selling the platform rather than building on it.
- **"save it"** — no title, no company, no id in the message. Answered
  **"Saved the Senior Data Platform Architect role at Snowflake."** The `saved`
  count on `/status` went 2 → 3.
- **"What have I applied to?"** — reads the application logged earlier back out
  of the database.

`save it` is the hard turn: it needs the tool results from three turns earlier
still in context. An earlier version of the page replayed only the visible text
and the agent wrote a reconstructed id, which the foreign key refused. See
[`../agent/agent_config.md`](../agent/agent_config.md).

### 6. `06-req5-mcp-service-9-tools.png` — requirement 5, the wiring

The MCP server registered as a Unity Catalog MCP Service,
`bootcamp_students.lubo_jobradar.jobradar_mcp`, with **all tools selected**.

> The image shows 9, which is what existed when it was taken. There are 11 now —
> `draft_application_text` and `set_follow_up` were added later. "Automatically
> include tools added to this server in the future" is ticked in the same
> screenshot, which is why the agent picked them up without re-registering.
The `search_jobs` description is readable here — the tool descriptions are
written for the agent, and say what the tool is for and what to do when it
fails.

### 7. `07-req5-agent-search-tool-call.png` — requirement 5, expanded

The same question in the Agent Bricks playground, with the tool call expanded:
the arguments the agent chose (`remote_only: true`, `top_k: 10`, and a query
expanded into vocabulary the postings actually use) and the raw JSON that came
back, `fit_score` and `fit_reason` included.

### 8. `08-req5-agent-search-answer.png` — requirement 5, the read

What the agent made of that output. Note the Snowflake role at #6 with its score
of **28** and the reason quoted — cosine similarity ranked that same job
**first**, at 0.742. The gap between the two is the argument for having both a
retrieval layer and a ranking layer.

### 9. `09-req5-agent-log-application-write.png` — requirement 5, the write

**"Log that I applied to the Sardine one"** → `log_application` → *"Logged as
applied, application 2."* The reply names the row that was written rather than
saying "done", which is what lets a wrong write be caught.

Nothing was sent to Sardine. No tool in this server contacts an employer.

### 10. `10-req5-agent-parallel-read-back.png` — requirement 5, read after write

**"What have I applied to so far, and what does the Sardine one actually want?"**
The agent recognised a compound question and ran `list_applications` and
`get_job` **in parallel**. The application from screenshot 9 is really in the
database — this is a fresh read, not the agent remembering what it said.

### 11–13. Drafting — `11-req4-draft-cover-letter.png`, `12-…-resume-bullets.png`, `13-…-outreach.png`

The option-5 capability "draft a tailored cover-letter snippet or resume
bullet", on the job page: three buttons, three forms, all written from the
stored posting and the stored profile.

The same job in all three, so the differences are the point:

- **Cover letter** — one paragraph, naming the specific overlap: *"I've shipped
  production AI systems end-to-end — LLM gateways, agentic workflows, RAG
  retrieval with pgvector — and I'm fluent with the exact stack you use."*
- **Resume bullets** — three one-liners, each starting with a verb and carrying
  a number where the profile gives one.
- **Outreach** — a short message written to be read on a phone, ending
  *"Let's talk."*

Everything in them is from the profile. Nothing is invented, which is the rule
the prompt spends most of its words on: a cover letter that claims a year of
something is worse than no cover letter, because the interview finds out.

These run in the **App** rather than through the MCP server. Drafting needs a
Databricks identity to reach a Foundation Model, this workspace disables
personal access tokens, and the MCP host is outside Databricks — so it cannot
hold one. A Databricks App authenticates natively. Same module, same prompt,
same untrusted-description fence; only the identity differs.

### 14. `14-req6-insights-cdf-dashboard.png` — requirement 6

The App's **Insights** tab. Every number on it was computed in Delta from a
Change Data Feed and published back to Lakebase; the page itself calculates
nothing. Each panel names the change type it came from — `CDF insert`,
`CDF update_postimage`, `CDF delete`, or the Delta mirror's current state — so
no figure on the page is ambiguous about its provenance.

Visible here:

- **Corpus by source**, six sources including **adzuna 177** — the API that
  produced nothing until its credentials reached the Spark executors.
- **Postings ingested**, the same six from `CDF insert` rather than from a
  `COUNT(*)`. The two agreeing is the point: one is current state, the other is
  reconstructed from the change feed.
- **Status transitions**, reading *"nothing recorded yet"*.

That last panel is the honest one, and worth not skipping past. At the time of
this run every row in the feed was an `insert` — the mirrors had just been
created, so there was nothing for an application to transition *from*. The panel
says so rather than rendering a zero, because "no transitions have happened" and
"transitions are broken" would otherwise look identical.

Transitions appear on the next run after any status change: the MERGE produces
`update_preimage` / `update_postimage` pairs, and the postimage rows are what
that panel counts. The code path is the same one already producing the insert
counts beside it.
