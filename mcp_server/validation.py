"""Cleaning what the agent passes in.

An MCP tool's caller is a language model, which makes the input a different
shape of untrusted than a web form. It will not attempt an injection, but it
will confidently pass `top_k="ten"`, or `status="in progress"`, or an
application id it half-remembers from three messages ago - and it does this
with no signal that it has guessed.

**Write tools raise the stakes.** A bad read returns nothing useful. A bad write
puts a row in the database that no query will ever match again, and the user
finds out weeks later when their pipeline is missing an application. So the
argument that decides *what gets written* is checked hardest: `status` is
validated against a closed set here, in the repository, and by a CHECK
constraint, because three cheap checks are worth less than one wrong row.

Errors are worded for the model to read and act on, because that is literally
what happens to them - the tool returns the message and the agent either fixes
its call or relays it to the user.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from jobradar.drafting import KINDS as DRAFT_KINDS
from jobradar.repository import APPLICATION_STATUSES

MAX_TOP_K = 50
"""Ceiling on a search. Fifty results is already more than any agent will read
out, and the cost of a bigger number is paid by the vector index."""

DEFAULT_TOP_K = 10
MAX_NOTE_CHARS = 4000
# Five years. Past this, "stale" stops meaning anything and the query is just
# scanning the whole table.
MAX_STALE_DAYS = 1825
MAX_QUERY_CHARS = 500
"""Longer than the model's own 256-token window, so nothing useful is lost by
capping here - but short enough that a runaway prompt cannot be pasted in as a
"search"."""


class BadArgument(ValueError):  # noqa: N818 - it IS the argument, not an "error"
    """A tool argument could not be used. The message is written for the caller.

    Named for what it is rather than BadArgumentError, because the name appears
    in nothing a user sees - it is caught and turned into a sentence - and
    "bad argument" reads better in the one place it does appear, the source.
    """


def clean_query(value: Any) -> str:
    """A search query. Must be text with something in it."""
    if not isinstance(value, str):
        raise BadArgument(
            f"query must be text, got {type(value).__name__}. "
            "Describe the job in plain English, for example 'remote senior data "
            "engineer working with Spark'."
        )
    cleaned = " ".join(value.split())
    if not cleaned:
        raise BadArgument(
            "query must not be empty. Describe the job in plain English, or call "
            "list_jobs to browse everything."
        )
    return cleaned[:MAX_QUERY_CHARS]


def clean_top_k(value: Any, default: int = DEFAULT_TOP_K) -> int:
    """How many results to return. Clamped rather than rejected.

    `top_k=500` is a reasonable thing for a model to try and an unreasonable
    thing to fail on. Clamping answers the question; rejecting starts a retry
    loop that ends with the agent asking for 10 anyway.
    """
    if value is None or value == "":
        return default
    try:
        top_k = int(value)
    except (TypeError, ValueError):
        raise BadArgument(f"top_k must be a whole number, got {value!r}.") from None
    return max(1, min(top_k, MAX_TOP_K))


def clean_job_id(value: Any) -> str:
    """A job id, as returned by a previous tool call.

    Not validated for shape beyond being non-empty text. The ids are sha256
    hexdigests, but checking that here would reject a future id format for no
    benefit - a wrong id is caught by the foreign key, and that error is
    accurate where a format guess would not be.
    """
    if not isinstance(value, str) or not value.strip():
        raise BadArgument(
            "job_id must be the id string from a previous search or list result."
        )
    return value.strip()


def clean_status(value: Any, default: str = "applied") -> str:
    """An application status, validated against the closed set.

    The single most important check in this file. A model told to "mark it as in
    progress" will invent exactly that, and an invented status is a row that no
    filter, no dashboard, and no future query will ever match again. The error
    lists the real options so the agent can correct itself in one turn instead
    of guessing again.
    """
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        raise BadArgument(f"status must be text, got {type(value).__name__}.")

    status = value.strip().lower()
    if status not in APPLICATION_STATUSES:
        raise BadArgument(
            f"{value!r} is not a valid status. Use one of: "
            f"{', '.join(APPLICATION_STATUSES)}."
        )
    return status


def clean_application_id(value: Any) -> int:
    """An application id, as returned by list_applications or log_application.

    Coerced to int here so a model passing "12" instead of 12 succeeds rather
    than producing a type error from the driver, which names a column and not
    the mistake.
    """
    try:
        application_id = int(value)
    except (TypeError, ValueError):
        raise BadArgument(
            f"application_id must be the number from a previous result, got {value!r}."
        ) from None
    if application_id <= 0:
        raise BadArgument(f"application_id must be positive, got {application_id}.")
    return application_id


def clean_note(value: Any, required: bool = False) -> str | None:
    """A free-text note. The only argument a user's own words reach directly."""
    if value is None or value == "":
        if required:
            raise BadArgument("note must not be empty.")
        return None
    if not isinstance(value, str):
        raise BadArgument(f"note must be text, got {type(value).__name__}.")
    cleaned = value.strip()
    if not cleaned:
        if required:
            raise BadArgument("note must not be empty.")
        return None
    return cleaned[:MAX_NOTE_CHARS]


def clean_text(value: Any, field: str, required: bool = True) -> str | None:
    """A short required string, like a company or a contact's name."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise BadArgument(f"{field} must not be empty.")
        return None
    if not isinstance(value, str):
        raise BadArgument(f"{field} must be text, got {type(value).__name__}.")
    return value.strip()[:200]


def clean_source(value: Any) -> str | None:
    """Restrict a search to one job board."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise BadArgument(f"source must be text, got {type(value).__name__}.")
    return value.strip().lower()


def clean_days(value: Any, maximum: int = 365) -> int | None:
    """"Posted within the last N days". None means no limit."""
    if value is None or value == "":
        return None
    try:
        days = int(value)
    except (TypeError, ValueError):
        raise BadArgument(f"posted_within_days must be a whole number, got {value!r}.") from None
    return max(1, min(days, maximum))


def clean_min_score(value: Any) -> int | None:
    """A fit-score floor, 0-100. None means unscored jobs are included too."""
    if value is None or value == "":
        return None
    try:
        score = int(value)
    except (TypeError, ValueError):
        raise BadArgument(f"min_score must be a whole number 0-100, got {value!r}.") from None
    return max(0, min(score, 100))


def clean_stale_days(value: Any) -> int | None:
    """"Not touched in N days". None means no staleness filter at all.

    Floored at 1 rather than 0: "stale for 0 days" is every open application,
    which is a filter that does nothing while looking like it did something.
    """
    if value is None or value == "":
        return None
    try:
        days = int(value)
    except (TypeError, ValueError):
        raise BadArgument(f"stale_days must be a whole number, got {value!r}.") from None
    return max(1, min(days, MAX_STALE_DAYS))


def clean_date(value: Any) -> date | None:
    """A calendar date as YYYY-MM-DD. None clears the field.

    Only ISO. The agent is told in the tool docstring to resolve "next Tuesday"
    itself, because a date parser that accepts prose is a date parser that
    silently guesses a year - and this value ends up in a reminder the user
    will act on.
    """
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise BadArgument(f"follow_up_on must be a date as YYYY-MM-DD, got {value!r}.")
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise BadArgument(
            f"follow_up_on must be a date as YYYY-MM-DD, got {value!r}."
        ) from None


def clean_draft_kind(value: Any) -> str:
    """Which kind of text to draft. Invalid values name the valid ones."""
    if value is None or value == "":
        return "cover_letter"
    wanted = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if wanted not in DRAFT_KINDS:
        raise BadArgument(
            f"kind must be one of {', '.join(sorted(DRAFT_KINDS))}, got {value!r}."
        )
    return wanted


def clean_flag(value: Any) -> bool:
    """A boolean an agent may send as a string.

    "false" is the case worth handling. Python considers a non-empty string
    truthy, so bool("false") is True - which would silently turn "remote only:
    no" into "remote only: yes".
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "y")
    return bool(value)


__all__ = [
    "DEFAULT_TOP_K",
    "MAX_NOTE_CHARS",
    "MAX_QUERY_CHARS",
    "MAX_TOP_K",
    "BadArgument",
    "clean_application_id",
    "clean_days",
    "clean_flag",
    "clean_job_id",
    "clean_min_score",
    "clean_note",
    "clean_query",
    "clean_source",
    "clean_status",
    "clean_text",
    "clean_top_k",
]
