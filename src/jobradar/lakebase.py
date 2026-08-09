"""Lakebase (Databricks-managed Postgres + pgvector) connection helper.

Ported unchanged from SkyIndex-AI apart from the secret scope and key names.
That project ran this exact code against Lakebase in production, so the parts
that look over-careful - the whitespace strip on a pasted URL, the pool reset -
are each there because something went wrong once.

Resolution order for the connection URL:

1. ``LAKEBASE_URL`` environment variable  - local development.
2. ``PG*`` variables injected by an attached app resource - deployed app.
3. Databricks secret scope/key            - deployed app with no resource.

Checking the environment first is what keeps local runs and deployed runs on a
single code path. Reading only from the secret scope would make the project
impossible to run, or test, outside a Databricks runtime. In production the
secret is still the only mechanism used: the URL is never committed, never
logged, and never placed in app.yaml.

Connections are pooled. Each request borrows and returns a connection instead
of paying a fresh TLS handshake to Postgres every time.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Secret scopes are workspace-wide. In a shared workspace a generic name like
# "database" may already exist and be owned by someone else, in which case
# writing to it fails with PermissionDenied. Hence the project-specific prefix.
_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "lubo-jobradar")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

# Lakebase here is a SHARED database. This account cannot CREATE DATABASE
# ("permission denied to create database"), and the instance already holds
# SkyIndex-AI's tables in public. So JobRadar gets its own SCHEMA instead, and
# every connection is pinned to it.
#
# Pinned on the connection rather than by qualifying table names in SQL,
# because a search_path set once is impossible to forget later - a single
# unqualified CREATE TABLE in a migration would otherwise land in public and
# collide with the other project.
_SCHEMA = os.environ.get("LAKEBASE_SCHEMA", "jobradar")

_POOL_MIN = int(os.environ.get("LAKEBASE_POOL_MIN", "1"))
_POOL_MAX = int(os.environ.get("LAKEBASE_POOL_MAX", "8"))

_pool: pg_pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


class LakebaseConfigError(RuntimeError):
    """Raised when no Lakebase connection URL can be resolved."""


def _clean_url(value: str) -> str:
    """Strip whitespace from inside a URL-form connection string.

    Pasting a long URL into a masked prompt can introduce line-wrap whitespace
    mid-string. The result still looks right at a glance but yields a hostname
    containing spaces, which fails DNS resolution with "Name or service not
    known" - an error that reads like a network fault and sends the whole
    investigation in the wrong direction.

    Only URL-form strings are touched. libpq also accepts a keyword/value DSN
    ("host=... port=..."), where spaces are meaningful and must be preserved.
    """
    stripped = (value or "").strip()
    if stripped.startswith(("postgresql://", "postgres://")):
        return "".join(stripped.split())
    return stripped


def _url_from_app_resource() -> str | None:
    """Build a connection URL from the credentials Databricks Apps injects.

    When a Lakebase instance is attached to an app as a database resource, the
    platform provisions the route and injects the standard libpq variables.
    """
    host = os.environ.get("PGHOST", "").strip()
    if not host:
        return None

    user = os.environ.get("PGUSER", "").strip()
    password = os.environ.get("PGPASSWORD", "").strip()
    database = os.environ.get("PGDATABASE", "databricks_postgres").strip()
    port = os.environ.get("PGPORT", "5432").strip()
    sslmode = os.environ.get("PGSSLMODE", "require").strip()

    credentials = quote(user, safe="")
    if password:
        credentials += ":" + quote(password, safe="")

    return f"postgresql://{credentials}@{host}:{port}/{database}?sslmode={sslmode}"


def _url_from_secret() -> str | None:
    """Read the connection URL from the Databricks secret scope.

    Returns None rather than raising when the SDK is absent or the secret is
    missing, so get_url() can produce one message describing every path it
    tried instead of an SDK stack trace.
    """
    try:
        # Deferred on purpose: local runs and CI have no databricks-sdk, and a
        # missing one should skip this lookup rather than stop the module loading.
        from databricks.sdk import WorkspaceClient  # noqa: PLC0415
    except ImportError:
        logger.debug("databricks-sdk not installed; skipping secret lookup")
        return None

    try:
        secret = WorkspaceClient().secrets.get_secret(scope=_SCOPE, key=_KEY)
    except Exception:
        logger.debug("Could not read secret %s/%s", _SCOPE, _KEY, exc_info=True)
        return None

    return _clean_url(base64.b64decode(secret.value).decode("utf-8"))


def get_url() -> str:
    """Resolve the Lakebase connection URL."""
    env_url = _clean_url(os.environ.get("LAKEBASE_URL", ""))
    if env_url:
        return env_url

    resource_url = (_url_from_app_resource() or "").strip()
    if resource_url:
        return resource_url

    secret_url = (_url_from_secret() or "").strip()
    if secret_url:
        return secret_url

    raise LakebaseConfigError(
        "No Lakebase connection URL found. Set LAKEBASE_URL for local "
        "development, attach the Lakebase instance to the app as a database "
        f"resource, or store the URL in the secret scope '{_SCOPE}' under key "
        f"'{_KEY}'."
    )


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    """Lazily build the connection pool. Safe to call from multiple threads."""
    # One pool per process is the entire point of a pool, so it is module
    # state by design rather than by accident.
    global _pool  # noqa: PLW0603
    if _pool is None:
        with _pool_lock:
            if _pool is None:  # re-check inside the lock
                _pool = pg_pool.ThreadedConnectionPool(
                    _POOL_MIN,
                    _POOL_MAX,
                    dsn=get_url(),
                    cursor_factory=RealDictCursor,
                    # public stays on the path so the vector type, which is
                    # installed there, still resolves.
                    options=f"-c search_path={_SCHEMA},public",
                )
                logger.info("Lakebase pool ready (max=%s, schema=%s)", _POOL_MAX, _SCHEMA)
    return _pool


def reset_pool() -> None:
    """Close and discard the pool. Used by tests and after credential changes."""
    global _pool  # noqa: PLW0603
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.closeall()
            except Exception:
                logger.warning("Error closing connection pool", exc_info=True)
            _pool = None


@contextmanager
def get_connection() -> Iterator[Any]:
    """Yield a pooled connection, committing on success, rolling back on error.

    Owning commit/rollback here means no caller can leave a half-applied
    transaction behind, and a connection is always returned to the pool.
    """
    connection_pool = _get_pool()
    conn = connection_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        connection_pool.putconn(conn)


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query and return rows as a list of dicts."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def run_query_one(sql: str, params: tuple | dict | None = None) -> dict | None:
    """Run a read query expected to return at most one row."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE and return the affected row count."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def healthcheck() -> bool:
    """Return True when Lakebase answers a trivial query."""
    try:
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            return cur.fetchone() is not None
    except Exception:
        logger.warning("Lakebase healthcheck failed", exc_info=True)
        return False


def apply_schema(schema_path: str | None = None) -> None:
    """Execute schema.sql against Lakebase (idempotent - see the file header)."""
    path = Path(schema_path) if schema_path else Path(__file__).resolve().parent / "schema.sql"
    ddl = path.read_text(encoding="utf-8")
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(ddl)
    logger.info("Applied schema from %s", path)


__all__ = [
    "LakebaseConfigError",
    "apply_schema",
    "get_connection",
    "get_url",
    "healthcheck",
    "reset_pool",
    "run_query",
    "run_query_one",
    "run_write",
]
