# JobRadar-AI — agent configuration and transcripts

How the Databricks Agent Bricks agent is wired up, and what it actually does
when asked. The prompt itself is in [`system_prompt.md`](system_prompt.md).

This is the evidence for **capstone requirement 5: an AI agent that can read
from and write to your database.**

---

## Configuration

| | |
|---|---|
| **Name** | `jobradar-agent` |
| **Type** | Agent Bricks — Supervisor Agent |
| **Model** | `claude-haiku-4-5` |
| **Instructions** | the full text of [`system_prompt.md`](system_prompt.md) |
| **Tools** | `bootcamp_students.lubo_jobradar.jobradar_mcp` (UC MCP Service) |

### The MCP connection

| | |
|---|---|
| URL | `https://jobradar.lubot.ai/mcp` |
| Transport | streamable HTTP |
| Auth | bearer token, generated on the host, stored in a UC connection |
| UC object | `bootcamp_students.lubo_jobradar.jobradar_mcp` |

`python_exec` is attached to every Supervisor Agent by Databricks and cannot be
removed. It is unused here; every answer below comes from an MCP tool call.

### Why the MCP server is not a Databricks App

The Databricks AI Gateway cannot authenticate to a Databricks App. Tested
against all four methods its own connection form offers:

| Method | Result |
|---|---|
| DCR | the workspace OIDC server publishes no `registration_endpoint` |
| PAT | Apps accept OAuth only; the token gets a 302 to `/authorize` |
| OAuth M2M | requires a service principal, which is admin-only in this workspace |
| OAuth U2M | requires a redirect URI on a Databricks-managed client |

So the MCP server runs on a hardened host at `jobradar.lubot.ai` behind nginx
and Let's Encrypt, and the Databricks App is the human frontend rather than the
agent's transport. Deploy script: [`../scripts/deploy_mcp.sh`](../scripts/deploy_mcp.sh).

---

## Transcripts

Three exchanges, run in order against the live database on 2026-08-09. Together
they exercise a read, a write, and a read-back that proves the write persisted.

### 1. Semantic search — a read

> **Find me remote roles where I'd be building streaming data pipelines**

**Tool call:** `search_jobs`

```json
{
  "query": "building streaming data pipelines with technologies like Kafka, Spark Streaming, Flink, or real-time data processing",
  "remote_only": true,
  "top_k": 10
}
```

Ten results. The top three, abridged:

| Job | Fit | Note |
|---|---|---|
| Data Engineer - Onboarding, **Sardine** — $150K-$205K | **72** | "Strong core match on data engineering, ML productionization, fraud domain, and fintech; GCP stack is a real gap, and 3 years is well below the 8+ ask" |
| Senior Data Engineer, **Gemini** | not scored | batch and real-time workloads, Spark/Flink/Beam |
| Senior Data Platform Architect, **Snowflake** | **28** | "a pre-sales Solutions Architect role... the core job is customer-facing architecture and deal closure, not hands-on platform engineering" |

**What this shows.**

The user never said "Kafka" or "Flink". The agent expanded a plain-English
request into the vocabulary the postings actually use, which is what the vector
index needs — several of these jobs match on meaning rather than on any word in
the question.

The Snowflake result is the more interesting one. It is a job that *looks* like a
perfect match: the words on the page are the words on the resume. Cosine
similarity ranked it **first**, at 0.742. The LLM scorer gave it **28** and
explained why, and the agent surfaced the score with its reason attached rather
than presenting the job as a strong match. That gap between 1st by similarity
and 28 by judgement is the argument for having both layers.

Two product-manager roles appear lower in the list. They are semantically close
— they are about streaming platforms — and neither is scored, so nothing claimed
they were a fit. Real behaviour, not a curated demo.

The agent also left the unscored jobs unscored rather than inventing numbers for
them, which is the prompt's "a null field is absent, not empty" rule holding.

---

### 2. Logging an application — a write

> **Log that I applied to the Sardine one**

**Tool call:** `log_application`

> Logged as applied, application 2.

**What this shows.**

Two rules under test.

*Positional reference.* "The Sardine one" is not an ID. The agent resolved it
from the previous result rather than searching again or asking. This is the case
the prompt calls out specifically, because getting it wrong on a read is
invisible while getting it wrong on a write records an application to a job the
user never applied to.

*Reporting the row it wrote.* The answer is "application 2", not "done" — the
identifier the tool returned, so the user can catch an error. The prompt asks
for this explicitly.

Worth being precise about what did not happen: **nothing was sent to Sardine.**
No tool in this server contacts an employer. `log_application` records an
application the user has already submitted. The prompt states this first and
repeats it, because "apply to that one" is genuinely ambiguous in English and an
agent that implies it applied leaves someone believing an application exists
that does not.

---

### 3. Read-back and detail — two tools in one turn

> **What have I applied to so far, and what does the Sardine one actually want?**

**Tool calls:** `list_applications` and `get_job`, run in parallel.

> **Your applications**
> You've applied to **1 role**:
> **Data Engineer - Onboarding** at **Sardine** — applied today
> North America · $150K - $205K
>
> **What Sardine wants**
> **The role:** Senior Data/ML Engineer owning the data and ML foundation for
> compliance decisions (onboarding, KYC, fraud detection).
>
> **Core responsibilities:** streaming pipelines (Pub/Sub, Apache Beam on
> Dataflow, Flink) and batch pipelines (Python, Airflow, Spark on Dataproc);
> Chronon feature definitions computed by Flink and Spark with windows from one
> hour to 300 days under sub-second latency; training pipelines on Vertex AI and
> Kubeflow; KYC/AML identity risk signals; a BigQuery warehouse layer; entity
> resolution across customers, devices, emails, phones and cards; field-level
> encryption and PII handling.
>
> **Requirements:** 8+ years, deep Python and SQL, fluency in Spark/Beam/Flink
> with streaming semantics, GCP strongly preferred, ML engineering, high-volume
> low-latency serving, fraud/risk/payments/KYC domain.
>
> **Fit score: 72** — Strong match on data engineering, ML productionization,
> fraud domain, Python/SQL/Spark/streaming fundamentals. GCP stack is a gap
> (you're AWS-focused), and 3 years experience is below their 8+ ask.

**What this shows.**

The write from the previous turn is really in the database — this is a fresh
`list_applications` call, not the agent remembering what it said.

The agent recognised a compound question and issued both calls in parallel
rather than in sequence. It also correctly used `get_job` for the description
instead of reusing the search summary, which deliberately omits it.

And the fit score arrives with its reasoning intact, including the two things
working against the application. An agent that reported "72, strong match" and
stopped would be less useful than one that says the GCP stack is a gap.

---

## Design notes on the prompt

Fuller reasoning is at the bottom of [`system_prompt.md`](system_prompt.md).
The three decisions that most shaped the behaviour above:

**Writes are separated from actions.** The single instruction whose failure has
consequences outside the database is "you never apply to anything", so it is
stated first, repeated, and given its own ambiguity rule.

**Errors carry a type, and the prompt says what to do with each.**
`bad_request` means fix it and retry once, `not_found` means get fresh IDs,
`internal_error` means say the tool is unavailable and substitute nothing. The
alternative is an agent that retries a malformed call forever or, worse, answers
from memory when a tool fails.

**Job descriptions are treated as untrusted input.** This corpus contains a real
posting that asks the reader to mention an invented internal product in their
cover letter — a filter for whether a human read it, and something an LLM will
cheerfully comply with. The agent is told to report what a description says and
never to follow what it asks.
