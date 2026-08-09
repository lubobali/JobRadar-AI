# Databricks notebook source
# MAGIC %md
# MAGIC # JobRadar-AI — Change Data Feed into a Delta analytics table
# MAGIC
# MAGIC **Capstone requirement 6: CDF on Delta, feeding an analytics table.**
# MAGIC
# MAGIC ```
# MAGIC Lakebase (Postgres)          the OLTP store. The app and the agent write here.
# MAGIC    │                         applications · saved_jobs · job_postings
# MAGIC    │  snapshot + MERGE
# MAGIC    ▼
# MAGIC Delta mirrors                delta.enableChangeDataFeed = true
# MAGIC    │                         jobradar_applications · jobradar_saved_jobs
# MAGIC    │                         jobradar_job_postings
# MAGIC    │  table_changes()
# MAGIC    ▼
# MAGIC jobradar_analytics_daily     one Delta table, day by metric by dimension
# MAGIC    │
# MAGIC    │  publish
# MAGIC    ▼
# MAGIC Lakebase.analytics_daily     so the App can render it over the connection
# MAGIC                              it already has
# MAGIC ```
# MAGIC
# MAGIC **Why a mirror and not CDF on the source.** Change Data Feed is a Delta
# MAGIC feature and Lakebase is Postgres, so there is no CDF to turn on there. The
# MAGIC mirror is where change history becomes queryable: a MERGE against it turns
# MAGIC "the current state of Postgres" into rows in a change feed, and those rows
# MAGIC are the only place the *transitions* exist. Postgres holds one status per
# MAGIC application; the feed holds every status it has ever had.
# MAGIC
# MAGIC **Why the results go back to Lakebase.** The Databricks App renders these
# MAGIC numbers, and it already has a Postgres connection and no Delta one. Adding
# MAGIC a SQL warehouse dependency to a page showing six numbers would be a poor
# MAGIC trade. Delta computes; Postgres publishes.
# MAGIC
# MAGIC **Re-runnable.** The MERGE is idempotent: a run with no upstream change
# MAGIC produces no CDF rows and the analytics come out identical. Two runs with a
# MAGIC status change between them produce `update_preimage` / `update_postimage`
# MAGIC pairs, which is where the transition counts come from.

# COMMAND ----------

# DBTITLE 1,Dependencies
# MAGIC %pip install -q psycopg2-binary git+https://github.com/lubobali/JobRadar-AI.git

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config
dbutils.widgets.text("secret_scope", "lubo-jobradar", "Secret scope")
dbutils.widgets.text("catalog", "bootcamp_students", "Unity Catalog")
dbutils.widgets.text("schema", "lubo_jobradar", "Schema")
dbutils.widgets.dropdown("publish", "true", ["true", "false"], "Write results back to Lakebase")

SECRET_SCOPE = dbutils.widgets.get("secret_scope")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
PUBLISH = dbutils.widgets.get("publish") == "true"

FQN = f"{CATALOG}.{SCHEMA}"
print(f"Delta target: {FQN}")

# COMMAND ----------

# DBTITLE 1,Credentials
import os

from databricks.sdk import WorkspaceClient

_client = WorkspaceClient()


def _secret(key: str) -> str | None:
    try:
        import base64

        return base64.b64decode(
            _client.secrets.get_secret(scope=SECRET_SCOPE, key=key).value
        ).decode("utf-8").strip()
    except Exception as exc:
        print(f"  {key}: not available ({type(exc).__name__})")
        return None


url = _secret("lakebase-url")
if url:
    os.environ["LAKEBASE_URL"] = url

LAKEBASE_URL = os.environ.get("LAKEBASE_URL")
assert LAKEBASE_URL, "No Lakebase URL."
print("credentials loaded")

# COMMAND ----------

# DBTITLE 1,Pull the operational tables out of Lakebase
#
# Read once into pandas and hand to Spark. These are small - hundreds of rows,
# not millions - and a JDBC read would need a driver on the cluster to buy
# parallelism this does not need.

import pandas as pd
import psycopg2
import psycopg2.extras

MIRRORS = {
    "applications": """
        SELECT id, user_id, job_id, status,
               applied_at, updated_at, follow_up_on
          FROM jobradar.applications
    """,
    "saved_jobs": """
        SELECT user_id, job_id, note, saved_at
          FROM jobradar.saved_jobs
    """,
    "job_postings": """
        SELECT id, source, company, title, location, remote,
               posted_at, fetched_at
          FROM jobradar.job_postings
    """,
}

frames: dict[str, pd.DataFrame] = {}
with psycopg2.connect(LAKEBASE_URL) as conn:
    for name, sql in MIRRORS.items():
        frames[name] = pd.read_sql_query(sql, conn)
        print(f"  {name:14} {len(frames[name])} rows")

# COMMAND ----------

# DBTITLE 1,Create the Delta mirrors with Change Data Feed enabled
#
# enableChangeDataFeed has to be set at or before the first write. Turning it on
# later means the feed starts from that moment, so every change before it is
# gone - and the table looks correct while quietly having no history.

from delta.tables import DeltaTable
from pyspark.sql import functions as F

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQN}")

KEYS = {
    "applications": ["id"],
    "saved_jobs": ["user_id", "job_id"],
    "job_postings": ["id"],
}

for name, frame in frames.items():
    table = f"{FQN}.jobradar_{name}"
    incoming = spark.createDataFrame(frame)

    if not spark.catalog.tableExists(table):
        (
            incoming.write.format("delta")
            .option("delta.enableChangeDataFeed", "true")
            .saveAsTable(table)
        )
        print(f"  created {table} with CDF")
        continue

    # MERGE rather than overwrite. An overwrite rewrites every row, so the
    # change feed would report the whole table as changed on every run and the
    # transition counts below would be meaningless.
    condition = " AND ".join(f"t.{k} = s.{k}" for k in KEYS[name])
    (
        DeltaTable.forName(spark, table)
        .alias("t")
        .merge(incoming.alias("s"), condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(f"  merged into {table}")

# COMMAND ----------

# DBTITLE 1,Prove CDF is actually on
for name in MIRRORS:
    table = f"{FQN}.jobradar_{name}"
    props = spark.sql(f"SHOW TBLPROPERTIES {table}").collect()
    enabled = {r["key"]: r["value"] for r in props}.get("delta.enableChangeDataFeed")
    print(f"  {table:60} enableChangeDataFeed = {enabled}")

display(spark.sql(f"DESCRIBE HISTORY {FQN}.jobradar_applications"))

# COMMAND ----------

# DBTITLE 1,Read the change feed
#
# This is the part Postgres cannot answer. Lakebase holds ONE status per
# application - the current one. The feed holds every status it has ever had,
# as update_preimage / update_postimage pairs, so "how many applications moved
# from applied to screening, and when" is a question only this table can answer.

changes = {}
for name in MIRRORS:
    table = f"{FQN}.jobradar_{name}"
    try:
        reader = spark.read.format("delta")
        reader = reader.option("readChangeFeed", "true")
        reader = reader.option("startingVersion", 0)
        changes[name] = reader.table(table)
        print(f"  {name:14} {changes[name].count()} change rows")
    except Exception as exc:
        # A table created in THIS run has no feed before its first version, which
        # is not a failure - it is a first run.
        print(f"  {name:14} no feed yet ({type(exc).__name__})")
        changes[name] = None

if changes.get("applications") is not None:
    display(
        changes["applications"]
        .select("_change_type", "_commit_version", "_commit_timestamp", "id", "status")
        .orderBy("_commit_version", "id")
    )

# COMMAND ----------

# DBTITLE 1,Build the analytics table
#
# Long format - day, metric, dimension, value - rather than one column per
# metric. Adding a metric then costs a row rather than a schema migration, and
# the App renders any of them without knowing their names in advance.

from pyspark.sql import Row
from pyspark.sql.types import (
    DateType,
    LongType,
    StringType,
    StructField,
    StructType,
)

ANALYTICS_SCHEMA = StructType([
    StructField("day", DateType(), nullable=False),
    StructField("metric", StringType(), nullable=False),
    StructField("dimension", StringType(), nullable=True),
    StructField("value", LongType(), nullable=False),
])

rows: list[Row] = []


def add(frame, metric: str, day_col, dimension_col=None) -> None:
    """Aggregate one grouping into the long format and collect it."""
    grouping = [F.to_date(day_col).alias("day")]
    grouping.append(
        (F.col(dimension_col).cast("string") if dimension_col else F.lit(None).cast("string"))
        .alias("dimension")
    )
    agg = frame.groupBy(*grouping).count()
    for r in agg.collect():
        if r["day"] is None:
            continue
        rows.append(Row(day=r["day"], metric=metric, dimension=r["dimension"], value=r["count"]))


# --- from the change feed: what MOVED -------------------------------------
app_changes = changes.get("applications")
if app_changes is not None:
    # An insert is a new application. update_postimage is a status change; the
    # preimage row is dropped so each transition counts once.
    add(
        app_changes.filter(F.col("_change_type") == "insert"),
        "applications_created",
        "_commit_timestamp",
        "status",
    )
    add(
        app_changes.filter(F.col("_change_type") == "update_postimage"),
        "status_transitions",
        "_commit_timestamp",
        "status",
    )

saved_changes = changes.get("saved_jobs")
if saved_changes is not None:
    add(
        saved_changes.filter(F.col("_change_type") == "insert"),
        "jobs_saved",
        "_commit_timestamp",
    )
    add(
        saved_changes.filter(F.col("_change_type") == "delete"),
        "jobs_unsaved",
        "_commit_timestamp",
    )

job_changes = changes.get("job_postings")
if job_changes is not None:
    add(
        job_changes.filter(F.col("_change_type") == "insert"),
        "postings_ingested",
        "_commit_timestamp",
        "source",
    )

# --- from current state: what IS -------------------------------------------
add(spark.table(f"{FQN}.jobradar_applications"), "pipeline_now", F.current_date(), "status")
add(spark.table(f"{FQN}.jobradar_job_postings"), "corpus_now", F.current_date(), "source")

analytics = spark.createDataFrame(rows, schema=ANALYTICS_SCHEMA) if rows else spark.createDataFrame(
    [], schema=ANALYTICS_SCHEMA
)

ANALYTICS_TABLE = f"{FQN}.jobradar_analytics_daily"
(
    analytics.write.format("delta")
    .option("delta.enableChangeDataFeed", "true")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(ANALYTICS_TABLE)
)

print(f"{analytics.count()} analytics rows written to {ANALYTICS_TABLE}")
display(analytics.orderBy("metric", "day", "dimension"))

# COMMAND ----------

# DBTITLE 1,Publish to Lakebase so the App can render it
#
# The App has a Postgres connection and no Delta one. Rather than give a page
# that shows six numbers a SQL warehouse dependency, the computed rows are
# written back and read over the connection that already exists.

if PUBLISH:
    published = [
        (r["day"], r["metric"], r["dimension"], int(r["value"]))
        for r in analytics.collect()
    ]

    with psycopg2.connect(LAKEBASE_URL) as conn, conn.cursor() as cur:
        cur.execute("SET search_path = jobradar, public")
        # Replaced wholesale: this table is a published artefact of the run
        # above, not an accumulating log. The log is the Delta change feed.
        cur.execute("TRUNCATE analytics_daily")
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO analytics_daily (day, metric, dimension, value)
            VALUES %s
            ON CONFLICT (day, metric, dimension) DO UPDATE
               SET value = EXCLUDED.value, computed_at = now()
            """,
            published,
        )
        conn.commit()
        cur.execute("SELECT count(*) FROM analytics_daily")
        print(f"published {cur.fetchone()[0]} rows to Lakebase.analytics_daily")
else:
    print("publish=false, Lakebase not written")

# COMMAND ----------

# DBTITLE 1,Report
summary = {
    "delta_tables": [f"jobradar_{n}" for n in MIRRORS] + ["jobradar_analytics_daily"],
    "cdf_enabled": True,
    "change_rows": {n: (c.count() if c is not None else 0) for n, c in changes.items()},
    "analytics_rows": analytics.count(),
    "published_to_lakebase": PUBLISH,
}
for key, value in summary.items():
    print(f"  {key:24} {value}")

dbutils.notebook.exit(str(summary))
