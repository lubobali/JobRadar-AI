# Databricks notebook source
# MAGIC %md
# MAGIC # JobRadar-AI — embed the descriptions
# MAGIC
# MAGIC **Capstone requirement 3: unstructured data processing.**
# MAGIC
# MAGIC Job descriptions are the unstructured half of this project: 2–10KB of
# MAGIC prose each, written by a different person at every company, in HTML that
# MAGIC eight different APIs format eight different ways.
# MAGIC
# MAGIC ```
# MAGIC job_postings.description       already stripped to plain text at ingest
# MAGIC    │
# MAGIC    ▼
# MAGIC chunk_job                      800-char windows, word-aligned, overlapping,
# MAGIC    │                           each prefixed with "Title at Company. Location."
# MAGIC    ▼
# MAGIC all-MiniLM-L6-v2               384 dimensions, max_seq_length 256
# MAGIC    │
# MAGIC    ▼
# MAGIC job_embeddings.embedding       vector(384), HNSW index, cosine
# MAGIC ```
# MAGIC
# MAGIC **Incremental by construction.** "Pending" is derived from the data, not
# MAGIC tracked in a cursor: a job is pending when no vector exists for its id AND
# MAGIC its current `content_hash` AND this model. So an edited posting is
# MAGIC re-embedded, a changed model re-embeds everything, and a run that dies
# MAGIC halfway resumes exactly where it stopped with no state to reconcile.
# MAGIC
# MAGIC **Why this is not a Spark job.** Embedding is GPU/CPU-bound work against a
# MAGIC single in-process model. Distributing it would load a copy of the weights
# MAGIC on every executor to run the same batched matrix multiply, which is slower
# MAGIC and more fragile than doing it in one place. The Spark job is `ingest_jobs`,
# MAGIC where the work is 129 independent HTTP calls and distribution actually
# MAGIC buys something.

# COMMAND ----------

# DBTITLE 1,Dependencies
# MAGIC %pip install -q sentence-transformers psycopg2-binary git+https://github.com/lubobali/JobRadar-AI.git

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config
dbutils.widgets.text("secret_scope", "lubo-jobradar", "Secret scope")
dbutils.widgets.text("batch_limit", "10000", "Max jobs per run")
dbutils.widgets.text("encode_batch_size", "64", "Texts per encode call")
dbutils.widgets.dropdown("reset", "false", ["true", "false"], "Delete all vectors first")

SECRET_SCOPE = dbutils.widgets.get("secret_scope")
BATCH_LIMIT = int(dbutils.widgets.get("batch_limit"))
ENCODE_BATCH_SIZE = int(dbutils.widgets.get("encode_batch_size"))

# Set this after changing CHUNK_SIZE, CHUNK_OVERLAP, or the context header.
# content_hash fingerprints the DESCRIPTION, not the chunking config, so a
# chunker change leaves every stored vector stale in a way the incremental
# anti-join cannot see: the postings did not change, so they look done.
RESET = dbutils.widgets.get("reset") == "true"

# COMMAND ----------

# DBTITLE 1,Credentials and cache
import base64
import logging
import os
import time

from databricks.sdk import WorkspaceClient

os.environ["LAKEBASE_URL"] = (
    base64.b64decode(
        WorkspaceClient().secrets.get_secret(scope=SECRET_SCOPE, key="lakebase-url").value
    )
    .decode("utf-8")
    .strip()
)

# The model cache has to live somewhere writable. Without this, the download
# fails on a read-only home directory with a permissions error that reads like
# a broken cluster.
os.environ.setdefault("HF_HOME", "/tmp/.cache/huggingface")

# sentence-transformers logs an HTTP request per config file it resolves -
# about twenty lines - plus progress bars. In a scheduled run that buries the
# only output that matters.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for noisy in ("httpx", "urllib3", "sentence_transformers", "huggingface_hub", "transformers"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("embed")

# COMMAND ----------

# DBTITLE 1,Check the model and the column agree before doing any work
from jobradar import embeddings, repository

MODEL_NAME = embeddings.EMBEDDING_MODEL

logger.info("model      : %s (%s-dim)", MODEL_NAME, embeddings.EMBEDDING_DIM)
logger.info("chunking   : size=%s overlap=%s", embeddings.CHUNK_SIZE, embeddings.CHUNK_OVERLAP)

# Fails now rather than as a driver type error partway through the first batch,
# where the message names neither the model nor the column.
repository.verify_schema()
logger.info("schema     : verified")
logger.info("before     : %s", repository.stats())

# COMMAND ----------

# DBTITLE 1,Chunk, embed, write
#
# One job at a time rather than one enormous batch. A job's vectors are replaced
# atomically, so a failure midway leaves earlier jobs correctly embedded and
# later ones simply still pending - and the next run picks up exactly there,
# because pending is derived from the data rather than from a cursor.

started = time.monotonic()

if RESET:
    logger.info("reset      : clearing %s existing vectors", repository.stats()["embeddings"])
    from jobradar import lakebase

    lakebase.run_write("DELETE FROM job_embeddings")

pending = repository.fetch_unembedded_jobs(model_name=MODEL_NAME, limit=BATCH_LIMIT)
logger.info("pending    : %s jobs", len(pending))

jobs_written = 0
chunks_written = 0
skipped = 0

for position, job in enumerate(pending, start=1):
    chunks = embeddings.chunk_job(
        title=job["title"],
        company=job["company"],
        location=job.get("location"),
        description=job["description"],
        content_hash=job["content_hash"],
    )
    if not chunks:
        # A posting with no description after HTML stripping. A vector of an
        # empty string carries no signal but would still take a slot in every
        # result list.
        skipped += 1
        continue

    vectors = embeddings.embed_texts(
        [chunk["chunk_text"] for chunk in chunks], batch_size=ENCODE_BATCH_SIZE
    )
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk["embedding"] = vector

    chunks_written += repository.replace_job_embeddings(job["id"], chunks, MODEL_NAME)
    jobs_written += 1

    if position % 250 == 0 or position == len(pending):
        elapsed = time.monotonic() - started
        rate = position / elapsed if elapsed else 0
        logger.info(
            "  %s/%s jobs, %s chunks, %.1f jobs/s",
            position, len(pending), chunks_written, rate,
        )

elapsed = time.monotonic() - started

# COMMAND ----------

# DBTITLE 1,Report
logger.info("-" * 60)
logger.info("jobs embedded    : %s", jobs_written)
logger.info("chunks written   : %s", chunks_written)
logger.info("skipped (no text): %s", skipped)
logger.info("elapsed          : %.1fs", elapsed)
logger.info("after            : %s", repository.stats())

remaining = repository.fetch_unembedded_jobs(model_name=MODEL_NAME, limit=1)
if remaining:
    logger.info(
        "NOTE: jobs remain pending - this run hit the %s batch limit. Run again.",
        BATCH_LIMIT,
    )
else:
    logger.info("every job is embedded at its current revision")

summary = {
    "jobs_embedded": jobs_written,
    "chunks_written": chunks_written,
    "skipped": skipped,
    "elapsed_seconds": round(elapsed, 1),
    "totals": repository.stats(),
}
dbutils.notebook.exit(str(summary))
