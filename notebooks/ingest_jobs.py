# Databricks notebook source
# MAGIC %md
# MAGIC # JobRadar-AI — Spark ingest
# MAGIC
# MAGIC **Capstone requirement 1: a data pipeline in Spark.** Also requirement 2,
# MAGIC since the fan-out below is 129 calls across 8 third-party job APIs.
# MAGIC
# MAGIC ```
# MAGIC 129 source specs
# MAGIC    │  parallelize        one partition per source, so 129 HTTP calls
# MAGIC    │                     happen on executors rather than on the driver
# MAGIC    ▼
# MAGIC  flatMap(fetch)          a dead board returns an error, never raises
# MAGIC    │
# MAGIC    ▼
# MAGIC  DataFrame               explicit StructType, never inferred
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
# MAGIC **Why not `spark.write.jdbc`.** It cannot express `ON CONFLICT DO UPDATE`,
# MAGIC and it cannot write pgvector types. Both matter here: boards edit postings
# MAGIC in place, and the embeddings table is `vector(384)`.
# MAGIC
# MAGIC **Every rule this notebook applies lives in `jobradar.ingest`,** which has
# MAGIC no Spark import and 34 tests. A rule that only exists in a notebook cell is
# MAGIC a rule nothing can test.

# COMMAND ----------

# DBTITLE 1,Dependencies
# MAGIC %pip install -q httpx psycopg2-binary

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

# DBTITLE 1,Get the code onto the driver AND the executors
#
# The driver imports jobradar from a clone. The executors need it too, because
# the fetch runs there - so the package is zipped and shipped with addPyFile.
# Without that, flatMap fails on every executor with ModuleNotFoundError, which
# reads like a broken cluster rather than a missing dependency.

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

CHECKOUT = Path("/tmp/jobradar-src")
if CHECKOUT.exists():
    shutil.rmtree(CHECKOUT)
subprocess.run(
    ["git", "clone", "-q", "--depth", "1", "--branch", GIT_REF,
     "https://github.com/lubobali/JobRadar-AI.git", str(CHECKOUT)],
    check=True,
)
sys.path.insert(0, str(CHECKOUT / "src"))

PACKAGE_ZIP = Path("/tmp/jobradar.zip")
if PACKAGE_ZIP.exists():
    PACKAGE_ZIP.unlink()
with zipfile.ZipFile(PACKAGE_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in (CHECKOUT / "src" / "jobradar").rglob("*.py"):
        archive.write(path, path.relative_to(CHECKOUT / "src"))

spark.sparkContext.addPyFile(str(PACKAGE_ZIP))

commit = subprocess.run(
    ["git", "-C", str(CHECKOUT), "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()
print(f"running {GIT_REF} @ {commit}")

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
    except Exception as exc:  # noqa: BLE001
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

# DBTITLE 1,Fan out across every source
#
# One partition per source. Not a performance flourish: these are HTTP calls to
# 129 different hosts, so the work is entirely I/O-bound and the only thing that
# matters is that they do not queue behind each other. One partition each means
# a slow board delays nothing but itself.

from jobradar import ingest

specs = ingest.source_specs()
print(f"{len(specs)} sources")


def fetch_partition(spec):
    """Runs on an executor. Returns rows plus, if it failed, one error row.

    Errors travel back as data rather than as exceptions, because a raise here
    kills the partition and a short count with no explanation is worse than a
    named failure.
    """
    from jobradar import ingest as executor_ingest

    jobs, error = executor_ingest.fetch_spec(spec)
    rows = [executor_ingest.to_row(job) for job in jobs]
    return [("ok", row) for row in rows] + ([("error", error)] if error else [])


fetched = (
    spark.sparkContext
    .parallelize(specs, numSlices=len(specs))
    .flatMap(fetch_partition)
    .collect()
)

rows = [payload for kind, payload in fetched if kind == "ok"]
errors = [payload for kind, payload in fetched if kind == "error"]

print(f"fetched : {len(rows)} postings")
print(f"failed  : {len(errors)} sources")
for error in errors[:15]:
    print(f"   {error}")
if len(errors) > 15:
    print(f"   ... and {len(errors) - 15} more")

# COMMAND ----------

# DBTITLE 1,Into a DataFrame, with a declared schema
#
# StructType written out rather than inferred. An inferred schema takes its
# shape from whatever happened to arrive, so a source that starts returning
# nulls for a column silently changes the type of that column, and the failure
# surfaces later as a write error naming neither the source nor the field.

from pyspark.sql import functions as F
from pyspark.sql.types import (
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

raw = spark.createDataFrame(rows, schema=JOB_SCHEMA)
raw.cache()
print(f"rows: {raw.count()}")
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
    # from addPyFile rather than from the driver's sys.path.
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
    "fetched": len(rows),
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
