# Databricks notebook source
# MAGIC %md
# MAGIC # JobRadar-AI — Spark ingest
# MAGIC
# MAGIC **Capstone requirement 1: a data pipeline in Spark.** Also requirement 2,
# MAGIC since the fan-out below is 129 calls across 8 third-party job APIs.
# MAGIC
# MAGIC ```
# MAGIC 129 source specs
# MAGIC    │  createDataFrame    one row per source
# MAGIC    ▼
# MAGIC  fetch UDF               129 HTTP calls, on executors, in parallel.
# MAGIC    │                     A dead board returns an error, never raises.
# MAGIC    ▼
# MAGIC  explode                 one row per posting, explicit StructType
# MAGIC    │
# MAGIC    ├─ dedup 1            window over id, keep the freshest
# MAGIC    ├─ dedup 2            window over cross_source_key, ATS beats aggregator
# MAGIC    ├─ prefilter          drop the killers
# MAGIC    ▼
# MAGIC  foreachPartition        psycopg2 execute_values, one connection per
# MAGIC                          partition, ON CONFLICT DO UPDATE
# MAGIC ```
# MAGIC
# MAGIC **Why the fetch runs on executors.** `watchlist.all_sources()` returns
# MAGIC closures, and a closure cannot be pickled. `ingest.source_specs()` returns
# MAGIC frozen dataclasses of strings instead, which ship fine — so this is a real
# MAGIC distributed fan-out and not Spark used as a write buffer.
# MAGIC
# MAGIC **Why the DataFrame API and not `sparkContext.parallelize`.** Serverless
# MAGIC compute has no `SparkContext`:
# MAGIC
# MAGIC ```
# MAGIC [JVM_ATTRIBUTE_NOT_SUPPORTED] SparkContext is not supported on serverless
# MAGIC compute. If you require direct access to the SparkContext, switch to
# MAGIC Dedicated access mode.
# MAGIC ```
# MAGIC
# MAGIC So the RDD API is out, and with it `addPyFile` — which is why the package
# MAGIC is `%pip install`ed from git rather than shipped as a zip. A UDF over a
# MAGIC DataFrame of specs distributes exactly the same way.
# MAGIC
# MAGIC **Why not `spark.write.jdbc`.** It cannot express `ON CONFLICT DO UPDATE`,
# MAGIC and it cannot write pgvector types. Both matter here: boards edit postings
# MAGIC in place, and the embeddings table is `vector(384)`.
# MAGIC
# MAGIC **Every rule this notebook applies lives in `jobradar.ingest`,** which has
# MAGIC no Spark import and 34 tests. A rule that only exists in a notebook cell is
# MAGIC a rule nothing can test.

# COMMAND ----------

# DBTITLE 1,Dependencies
#
# The package is installed rather than added to sys.path, because serverless has
# no addPyFile and the executors need it too - a notebook-scoped %pip install is
# the one mechanism that reaches both.
# MAGIC %pip install -q httpx psycopg2-binary git+https://github.com/lubobali/JobRadar-AI.git

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config
dbutils.widgets.text("git_ref", "main", "Branch or commit to run")
dbutils.widgets.text("user_email", "lubobali23@gmail.com", "Whose profile to score against")
dbutils.widgets.text("secret_scope", "lubo-jobradar", "Secret scope")
dbutils.widgets.dropdown("dry_run", "false", ["true", "false"], "Fetch but do not write")

GIT_REF = dbutils.widgets.get("git_ref")
USER_EMAIL = dbutils.widgets.get("user_email")
SECRET_SCOPE = dbutils.widgets.get("secret_scope")
DRY_RUN = dbutils.widgets.get("dry_run") == "true"

# COMMAND ----------

# DBTITLE 1,Which build is running
#
# Printed so a run in the logs can be tied to a commit. The install above pulls
# the branch named by the widget.

import jobradar

print(f"jobradar {jobradar.__file__}")
commit = GIT_REF

# COMMAND ----------

# DBTITLE 1,Credentials
#
# Adzuna and USAJobs are the only two sources that need keys. The other six are
# open. Read from the secret scope and pushed into the environment, because the
# fetchers read them from there - the same code path a local run uses.

import os

from databricks.sdk import WorkspaceClient

_client = WorkspaceClient()


def _secret(key: str) -> str | None:
    """One secret, or None. Missing is not fatal: the two sources that need a
    key report an error and the other six still run."""
    try:
        import base64
        return base64.b64decode(
            _client.secrets.get_secret(scope=SECRET_SCOPE, key=key).value
        ).decode("utf-8").strip()
    except Exception as exc:
        print(f"  {key}: not available ({type(exc).__name__})")
        return None


for env_var, secret_key in [
    ("LAKEBASE_URL", "lakebase-url"),
    ("ADZUNA_APP_ID", "adzuna-app-id"),
    ("ADZUNA_APP_KEY", "adzuna-app-key"),
    ("USAJOBS_EMAIL", "usajobs-email"),
    ("USAJOBS_API_KEY", "usajobs-api-key"),
]:
    value = _secret(secret_key)
    if value:
        os.environ[env_var] = value

LAKEBASE_URL = os.environ.get("LAKEBASE_URL")
assert LAKEBASE_URL, "No Lakebase URL. Run setup_secrets.py first."
print("credentials loaded")

# COMMAND ----------

# DBTITLE 1,The row schema
#
# Declared, never inferred. An inferred schema takes its shape from whatever
# happened to arrive, so a source that starts returning nulls for a column
# silently changes that column's type, and the failure surfaces much later as a
# write error naming neither the source nor the field.

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

JOB_SCHEMA = StructType([
    StructField("id", StringType(), nullable=False),
    StructField("source", StringType(), nullable=False),
    StructField("source_id", StringType(), nullable=False),
    StructField("company", StringType(), nullable=False),
    StructField("title", StringType(), nullable=False),
    StructField("url", StringType(), nullable=False),
    StructField("location", StringType()),
    StructField("remote", BooleanType(), nullable=False),
    StructField("salary", StringType()),
    StructField("salary_is_estimated", BooleanType(), nullable=False),
    StructField("description", StringType(), nullable=False),
    StructField("posted_at", TimestampType()),
    StructField("fetched_at", TimestampType(), nullable=False),
    StructField("content_hash", StringType(), nullable=False),
    StructField("cross_source_key", StringType(), nullable=False),
])

FETCH_RESULT = StructType([
    StructField("rows", ArrayType(JOB_SCHEMA)),
    # The error travels back as DATA, not as an exception. A raise inside a UDF
    # fails the whole stage, and the fan-out exists precisely so one dead board
    # cannot do that. It also means a short count always has a name attached.
    StructField("error", StringType()),
])

# COMMAND ----------

# DBTITLE 1,Fan out across every source
#
# One row per source, one UDF call per row, executed across the cluster. These
# are HTTP calls to 129 different hosts, so the work is entirely I/O-bound and
# the only thing that matters is that they do not queue behind one another.

from dataclasses import asdict

from jobradar import ingest

specs = ingest.source_specs()
print(f"{len(specs)} sources")

SPEC_FIELDS = ["kind", "slug", "company", "query", "where", "tenant", "site", "host"]
specs_df = spark.createDataFrame([asdict(spec) for spec in specs]).repartition(32)


# The credentials have to travel INSIDE the closure.
#
# The cell above put them in os.environ, which is a driver-side process. An
# executor is a different Python process on a different machine and inherits
# none of it, so every fetcher that reads os.environ found nothing there and
# reported a missing key - while the same spec fetched fine from a terminal.
#
# The symptom is quiet, which is what makes it worth the comment: the run
# succeeds, the numbers look plausible, and the only trace is that the sources
# needing a key are in the failed list every single time. It reads like a bad
# key rather than a key that never arrived.
#
# A closure is pickled and shipped with the UDF, so this dict does arrive.
_CREDENTIALS = {
    name: os.environ[name]
    for name in ("ADZUNA_APP_ID", "ADZUNA_APP_KEY", "USAJOBS_EMAIL", "USAJOBS_API_KEY")
    if os.environ.get(name)
}
print(f"shipping {len(_CREDENTIALS)} credentials to the executors")


@F.udf(returnType=FETCH_RESULT)
def fetch(kind, slug, company, query, where, tenant, site, host):
    # Imported inside the UDF because this runs on an executor, which has the
    # package from the %pip install rather than from the driver's namespace.
    import os as executor_os

    executor_os.environ.update(_CREDENTIALS)

    from jobradar import ingest

    spec = ingest.SourceSpec(
        kind=kind or "",
        slug=slug or "",
        company=company or "",
        query=query or "",
        where=where or "",
        tenant=tenant or "",
        site=site or "",
        host=host or "",
    )
    jobs, error = ingest.fetch_spec(spec)
    return {"rows": [ingest.to_row(job) for job in jobs], "error": error}


fetched = specs_df.withColumn("result", fetch(*[F.col(f) for f in SPEC_FIELDS]))

# Collected in ONE pass, deliberately.
#
# `.cache()` is PERSIST TABLE underneath, and serverless refuses it:
#
#   [NOT_SUPPORTED_WITH_SERVERLESS] PERSIST TABLE is not supported on
#   serverless compute
#
# Without a cache, `fetched` is a lazy plan whose source is 129 HTTP calls, and
# every action re-runs all of them. There are four actions below - the error
# list, two counts and the write - so an uncached version would hammer every job
# board four times and could get four different answers.
#
# So the fan-out is materialised exactly once, here. The HTTP work stays
# distributed across the executors; only the assembled rows come back.
collected = fetched.select("result").collect()

errors = [row["result"]["error"] for row in collected if row["result"]["error"]]
job_rows = [job for row in collected for job in row["result"]["rows"]]

raw = spark.createDataFrame(job_rows, schema=JOB_SCHEMA)

print(f"fetched : {raw.count()} postings")
print(f"failed  : {len(errors)} sources")
ERRORS_SHOWN = 15
for error in errors[:ERRORS_SHOWN]:
    print(f"   {error}")
if len(errors) > ERRORS_SHOWN:
    print(f"   ... and {len(errors) - ERRORS_SHOWN} more")

display(raw.groupBy("source").count().orderBy(F.desc("count")))

# COMMAND ----------

# DBTITLE 1,Dedup round 1 — one row per posting
#
# The same board polled twice in a run returns the same posting twice. Keyed on
# id, which is sha256(source, source_id) - never on the url, because Adzuna's
# carries a per-request token and changes on every fetch.

from pyspark.sql import Window

freshest = Window.partitionBy("id").orderBy(F.col("fetched_at").desc())

deduped_by_id = (
    raw.withColumn("_rank", F.row_number().over(freshest))
    .filter(F.col("_rank") == 1)
    .drop("_rank")
)

print(f"after dedup on id: {deduped_by_id.count()} (from {raw.count()})")

# COMMAND ----------

# DBTITLE 1,Dedup round 2 — one row per real-world job
#
# The case round 1 cannot see: a Caterpillar role published on Greenhouse and
# relayed by Adzuna has two different source ids, so two different primary keys,
# and is one job.
#
# The winner is decided by ingest.SOURCE_PRIORITY, not by anything defined here,
# so the Spark path and the plain-Python path can never disagree. An ATS row
# wins because it carries the whole description; an aggregator truncates it, and
# the description is the entire unstructured pipeline.

priority_map = F.create_map(
    *[x for kv in ingest.SOURCE_PRIORITY.items() for x in (F.lit(kv[0]), F.lit(kv[1]))]
)

best = Window.partitionBy("cross_source_key").orderBy(
    F.coalesce(priority_map[F.col("source")], F.lit(ingest.DEFAULT_PRIORITY)).asc(),
    F.length("description").desc(),   # the tiebreak within one priority level
    F.col("id").asc(),                # and a stable last resort, so reruns agree
)

deduped = (
    deduped_by_id.withColumn("_rank", F.row_number().over(best))
    .filter(F.col("_rank") == 1)
    .drop("_rank")
)

collapsed = deduped_by_id.count() - deduped.count()
print(f"after cross-source dedup: {deduped.count()} ({collapsed} duplicates collapsed)")

# COMMAND ----------

# DBTITLE 1,Prefilter — drop what could never be worked from the US
#
# prefilter.is_us_eligible, ported unchanged from aws-job-streamer. It reads a
# location string and decides whether the role is plausibly workable from the
# US, which catches the "Remote (EMEA)" and "CA-Toronto" postings that the
# aggregators return in bulk.
#
# Applied here, before embedding, because embedding a posting nobody can take
# costs the same as embedding one they can. The sharper judgements - years of
# experience, wrong discipline, Azure-mandatory - live in fit.py and run later
# against a scored job, where they have the LLM's reading to work with.

@F.udf(returnType=BooleanType())
def us_eligible(location: str) -> bool:
    # Imported inside the UDF: this runs on an executor, which gets the package
    # from the %pip install rather than from the driver's namespace.
    from jobradar import prefilter

    return prefilter.is_us_eligible(location)


kept = deduped.filter(us_eligible(F.col("location")))
dropped = deduped.count() - kept.count()
print(f"after prefilter: {kept.count()} ({dropped} not US-workable)")

# COMMAND ----------

# DBTITLE 1,Write to Lakebase
#
# foreachPartition, one psycopg2 connection per partition, execute_values in
# batches. Deliberately not spark.write.jdbc: it cannot express ON CONFLICT DO
# UPDATE, and boards edit postings in place - a salary appears, a description is
# rewritten - so ignoring the conflict would freeze whichever version arrived
# first.

FINAL_COLUMNS = list(ingest.ROW_FIELDS)
to_write = kept.select(*FINAL_COLUMNS)

if DRY_RUN:
    print("dry run: nothing written")
    display(to_write.limit(20))
else:
    url = LAKEBASE_URL

    def write_partition(partition):
        import psycopg2
        from psycopg2.extras import execute_values

        batch = [tuple(row[column] for column in FINAL_COLUMNS) for row in partition]
        if not batch:
            return

        updates = ", ".join(
            f"{column} = EXCLUDED.{column}"
            for column in FINAL_COLUMNS
            if column not in ("id", "fetched_at")
        )
        sql = f"""
            INSERT INTO job_postings ({", ".join(FINAL_COLUMNS)})
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET {updates}
        """
        conn = psycopg2.connect(url, options="-c search_path=jobradar,public")
        try:
            with conn, conn.cursor() as cur:
                execute_values(cur, sql, batch, page_size=200)
        finally:
            conn.close()

    # Repartitioned before writing: 129 partitions means 129 Postgres
    # connections, and Lakebase has a connection limit well under that.
    to_write.repartition(8).foreachPartition(write_partition)
    print(f"wrote {to_write.count()} rows")

# COMMAND ----------

# DBTITLE 1,Report
from jobradar import repository

summary = {
    "sources": len(specs),
    "sources_failed": len(errors),
    "fetched": raw.count(),
    "after_id_dedup": deduped_by_id.count(),
    "after_cross_source_dedup": deduped.count(),
    "after_prefilter": kept.count(),
    "written": 0 if DRY_RUN else kept.count(),
    "commit": commit,
}

if not DRY_RUN:
    summary["table_totals"] = repository.stats()

for key, value in summary.items():
    print(f"  {key:<26} {value}")

dbutils.notebook.exit(str(summary))
