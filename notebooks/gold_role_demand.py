# Databricks notebook source
# MAGIC %md
# MAGIC # JobRadar-AI — gold: weekly demand by role type and remote
# MAGIC
# MAGIC ```
# MAGIC jobradar_job_postings                 the Delta copy of the postings (silver), read as is
# MAGIC    │  classify_role(title) · work_mode(remote) · week_start(posted_at, fetched_at)
# MAGIC    ▼
# MAGIC group by week_start, role_type, work_mode
# MAGIC    │  check_gold(): never empty · unique grain · allowed values · no zero counts
# MAGIC    │                sum(postings) + no_week = source rows
# MAGIC    ▼
# MAGIC jobradar_gold_role_demand_weekly      written ONLY if every check passes
# MAGIC ```
# MAGIC
# MAGIC **Role type comes from the title**, via ordered regex buckets in
# MAGIC `jobradar.role_type`. Good enough for counting demand in aggregate; not used
# MAGIC for matching a job to a person.
# MAGIC
# MAGIC **Remote is `remote` / `not_remote`.** The source column is a boolean that
# MAGIC defaults to FALSE, so FALSE means "not marked remote", never "onsite". No hybrid.
# MAGIC
# MAGIC **The week** is the UTC Monday of `posted_at`, or of `fetched_at` when a board
# MAGIC sends no posted date. Ingest already removed cross-source duplicates, so one
# MAGIC source row is one posting.
# MAGIC
# MAGIC **Re-runnable.** The table is rebuilt in full each run. If any check fails,
# MAGIC the notebook raises before the write and the previous table stays as it was.

# COMMAND ----------

# DBTITLE 1,Dependencies
# The executors run the UDFs, so they need the package too: installed, not added to sys.path.
# Pinned to the feature branch until it is merged; switch back to main after the merge.
# MAGIC %pip install -q git+https://github.com/lubobali/JobRadar-AI.git@feature/gold-role-demand

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config
dbutils.widgets.text("catalog", "bootcamp_students", "Unity Catalog")
dbutils.widgets.text("schema", "lubo_jobradar", "Schema")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQN = f"{CATALOG}.{SCHEMA}"

SOURCE_TABLE = f"{FQN}.jobradar_job_postings"
GOLD_TABLE = f"{FQN}.jobradar_gold_role_demand_weekly"

# week_start() takes a naive datetime as UTC, and a Python UDF receives timestamps in the
# session time zone. Pinning the session to UTC makes those two agree.
spark.conf.set("spark.sql.session.timeZone", "UTC")

print(f"source: {SOURCE_TABLE}")
print(f"gold:   {GOLD_TABLE}")

# COMMAND ----------

# DBTITLE 1,Label every posting
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, StringType

from jobradar.gold_checks import GRAIN, check_gold
from jobradar.role_type import classify_role, week_start, work_mode

classify_role_udf = F.udf(classify_role, StringType())
work_mode_udf = F.udf(work_mode, StringType())
week_start_udf = F.udf(week_start, DateType())

source = spark.table(SOURCE_TABLE)
source_rows = source.count()

labeled = source.select(
    "id",
    "company",
    week_start_udf("posted_at", "fetched_at").alias("week_start"),
    classify_role_udf("title").alias("role_type"),
    work_mode_udf("remote").alias("work_mode"),
    F.col("posted_at").isNull().alias("used_fetched_at"),
).cache()

no_week = labeled.filter(F.col("week_start").isNull()).count()
print(f"source rows: {source_rows}   no week: {no_week}")

# COMMAND ----------

# DBTITLE 1,Group into the gold grain
weekly = (
    labeled.filter(F.col("week_start").isNotNull())
    .groupBy(*GRAIN)
    .agg(
        F.count("id").alias("postings"),
        F.countDistinct("company").alias("companies"),
    )
)

per_week = Window.partitionBy("week_start")
gold = (
    weekly.withColumn("share_of_week", F.col("postings") / F.sum("postings").over(per_week))
    # A week is complete once its Sunday is over (UTC). The current week is partial.
    .withColumn("week_complete", F.date_add("week_start", 7) <= F.current_date())
    .withColumn("computed_at", F.current_timestamp())
    .orderBy("week_start", "role_type", "work_mode")
).cache()

display(gold)

# COMMAND ----------

# DBTITLE 1,Gold checks - nothing is written unless all pass
rows = [r.asDict() for r in gold.select(*GRAIN, "postings", "companies").collect()]
failures = check_gold(rows, source_rows=source_rows, no_week=no_week)

if failures:
    for f in failures:
        print(f"  FAIL {f}")
    raise AssertionError(f"{len(failures)} gold check(s) failed; {GOLD_TABLE} not written")

print(f"all gold checks passed ({len(rows)} rows)")

# COMMAND ----------

# DBTITLE 1,Write
(
    gold.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_TABLE)
)
print(f"wrote {GOLD_TABLE}")

# COMMAND ----------

# DBTITLE 1,Report
total = sum(r["postings"] for r in rows)
other = sum(r["postings"] for r in rows if r["role_type"] == "other")
fell_back = labeled.filter(F.col("used_fetched_at")).count()

summary = {
    "gold_table": GOLD_TABLE,
    "gold_rows": len(rows),
    "weeks": len({r["week_start"] for r in rows}),
    "source_rows": source_rows,
    "postings": total,
    "no_week": no_week,
    "other_share": round(other / total, 3) if total else None,
    "used_fetched_at": fell_back,
}
for key, value in summary.items():
    print(f"  {key:16} {value}")

dbutils.notebook.exit(str(summary))
