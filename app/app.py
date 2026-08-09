"""JobRadar-AI - the frontend. Capstone requirement 4.

Three tabs and a detail page:

    /            Search   every stored job, ranked by fit score. The search box
                          re-ranks it semantically; it does not fetch anything.
    /saved       Saved    the same cards, filtered to bookmarked.
    /applied     Applied  one row per application, with its status and notes.
    /job/<id>              the full posting.

**The buttons and the agent call the same functions.** Every write on this page
goes through `jobradar.repository`, which is exactly what the MCP tools call.
Two paths to the same table is how "save" starts meaning two different things,
so there is only one.

**Nothing here fetches from a job board.** The corpus is refreshed by a
scheduled Spark job. This page reads what is already stored, which is what makes
it fast and what makes the same search twice give the same answer.
"""

from __future__ import annotations

import logging
import os
from http import HTTPStatus
from typing import Any

from flask import Flask, abort, jsonify, render_template, request

from jobradar import lakebase, matching, repository

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("jobradar-app")

app = Flask(__name__)

USER_EMAIL = os.environ.get("JOBRADAR_USER_EMAIL", "data@lubobali.com")
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", "25"))

# The agent's serving endpoint, for the chat bar. Optional on purpose: without
# it the bar explains where the agent lives, and every button on the page still
# works. The frontend is not allowed to depend on the agent being reachable.
AGENT_ENDPOINT = os.environ.get("JOBRADAR_AGENT_ENDPOINT", "")

_user_id: int | None = None


def user_id() -> int:
    """The owner's id, looked up once and cached."""
    global _user_id  # noqa: PLW0603 - one owner per process
    if _user_id is None:
        user = repository.get_user(USER_EMAIL)
        if user is None:
            raise RuntimeError(f"No user {USER_EMAIL!r}. Run scripts/seed_profile.py.")
        _user_id = int(user["id"])
    return _user_id


@app.errorhandler(Exception)
def handle_exception(err: Exception):  # noqa: ANN201
    """API paths get JSON, pages get a page.

    The fetch() calls parse every response as JSON, so a Flask HTML error page
    would make the frontend fail while parsing - hiding the real error behind a
    SyntaxError about an unexpected '<'.
    """
    status = getattr(err, "code", 500)
    if not isinstance(status, int):
        status = 500
    # A 404 is a user typing a stale job id, not a fault. Logging a traceback
    # for it buries the ones that matter.
    if status != HTTPStatus.NOT_FOUND:
        logger.exception("Unhandled error")
    if request.path.startswith("/api/"):
        return jsonify({"error": str(err)}), status
    return render_template("error.html", status=status, message=str(err)), status


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _filters() -> dict[str, Any]:
    """Read the filter bar off the query string, coerced and defaulted."""

    def as_int(name: str) -> int | None:
        raw = request.args.get(name, "").strip()
        try:
            return int(raw) if raw else None
        except ValueError:
            return None

    return {
        "query": request.args.get("q", "").strip(),
        "source": request.args.get("source", "").strip() or None,
        "remote_only": request.args.get("remote") == "on",
        "posted_within_days": as_int("days"),
        "min_score": as_int("score"),
        "page": max(1, as_int("page") or 1),
    }


@app.route("/")
def search_page():  # noqa: ANN201
    """The Search tab: the list of every stored job.

    An empty search box is the normal state, not a prompt. The list is already
    there, best fit first. Typing re-ranks it semantically.
    """
    selected = _filters()
    offset = (selected["page"] - 1) * PAGE_SIZE

    if selected["query"]:
        # Semantic search returns a ranked pool rather than a page. Page four
        # of a similarity ranking is noise, so pagination is switched off.
        vector = matching.embeddings.embed_query(selected["query"])
        jobs = repository.search(
            vector,
            user_id=user_id(),
            top_k=PAGE_SIZE * 2,
            source=selected["source"],
            remote_only=selected["remote_only"],
            posted_within_days=selected["posted_within_days"],
        )
        total, pages = len(jobs), 1
    else:
        jobs = repository.list_jobs(
            user_id=user_id(),
            limit=PAGE_SIZE,
            offset=offset,
            source=selected["source"],
            remote_only=selected["remote_only"],
            posted_within_days=selected["posted_within_days"],
            min_score=selected["min_score"],
        )
        total = repository.count_jobs(
            user_id=user_id(),
            source=selected["source"],
            remote_only=selected["remote_only"],
            posted_within_days=selected["posted_within_days"],
            min_score=selected["min_score"],
        )
        pages = max(1, -(-total // PAGE_SIZE))

    return render_template(
        "search.html",
        tab="search",
        jobs=jobs,
        total=total,
        pages=pages,
        sources=_sources(),
        agent_ready=bool(AGENT_ENDPOINT),
        **selected,
    )


@app.route("/saved")
def saved_page():  # noqa: ANN201
    jobs = repository.list_saved(user_id())
    return render_template(
        "saved.html", tab="saved", jobs=jobs, total=len(jobs),
        agent_ready=bool(AGENT_ENDPOINT),
    )


@app.route("/applied")
def applied_page():  # noqa: ANN201
    status = request.args.get("status", "").strip() or None
    applications = repository.list_applications(user_id(), status=status)
    return render_template(
        "applied.html",
        tab="applied",
        applications=applications,
        statuses=repository.APPLICATION_STATUSES,
        current_status=status,
        total=len(applications),
        agent_ready=bool(AGENT_ENDPOINT),
    )


@app.route("/chat")
def chat_page():  # noqa: ANN201
    """The agent, on its own page.

    It started as a bar fixed to the bottom of every page, which put a growing
    reply on top of the results it was talking about. A conversation and a
    ranked list are both trying to be the main thing on the screen; they each
    need their own.
    """
    return render_template("chat.html", tab="chat", agent_ready=bool(AGENT_ENDPOINT))


@app.route("/job/<job_id>")
def job_page(job_id: str):  # noqa: ANN201
    job = repository.get_job(job_id, user_id=user_id())
    if job is None:
        abort(404)
    return render_template(
        "job.html",
        tab="search",
        job=job,
        statuses=repository.APPLICATION_STATUSES,
        agent_ready=bool(AGENT_ENDPOINT),
    )


def _sources() -> list[str]:
    """Which boards are actually represented, for the filter dropdown.

    Read from the data rather than hardcoded, so a source that stops returning
    results disappears from the filter instead of offering an empty option.
    """
    return [
        row["source"]
        for row in lakebase.run_query("SELECT DISTINCT source FROM job_postings ORDER BY source")
    ]


# ---------------------------------------------------------------------------
# Writes
#
# Every one calls the same repository function the matching MCP tool calls. The
# buttons and the agent are two front doors onto one implementation.
# ---------------------------------------------------------------------------


@app.route("/api/save", methods=["POST"])
def api_save():  # noqa: ANN201
    body = request.get_json(silent=True) or {}
    job_id = (body.get("job_id") or "").strip()
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    if body.get("unsave"):
        repository.unsave_job(user_id(), job_id)
        return jsonify({"saved": False})

    repository.save_job(user_id(), job_id, (body.get("note") or "").strip() or None)
    return jsonify({"saved": True})


@app.route("/api/apply", methods=["POST"])
def api_apply():  # noqa: ANN201
    body = request.get_json(silent=True) or {}
    job_id = (body.get("job_id") or "").strip()
    status = (body.get("status") or "applied").strip()
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400
    if status not in repository.APPLICATION_STATUSES:
        return jsonify({"error": f"unknown status {status!r}"}), 400

    application = repository.log_application(
        user_id(), job_id, status, (body.get("note") or "").strip() or None
    )
    return jsonify({"application_id": application["id"], "status": application["status"]})


@app.route("/api/status", methods=["POST"])
def api_status():  # noqa: ANN201
    body = request.get_json(silent=True) or {}
    try:
        application_id = int(body.get("application_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "application_id is required"}), 400

    status = (body.get("status") or "").strip()
    if status not in repository.APPLICATION_STATUSES:
        return jsonify({"error": f"unknown status {status!r}"}), 400

    updated = repository.update_application_status(user_id(), application_id, status)
    if updated is None:
        return jsonify({"error": "no such application"}), 404
    return jsonify({"status": updated["status"]})


@app.route("/api/note", methods=["POST"])
def api_note():  # noqa: ANN201
    body = request.get_json(silent=True) or {}
    try:
        application_id = int(body.get("application_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "application_id is required"}), 400

    note = (body.get("note") or "").strip()
    if not note:
        return jsonify({"error": "note must not be empty"}), 400

    stored = repository.add_interview_note(user_id(), application_id, note)
    if stored is None:
        return jsonify({"error": "no such application"}), 404
    return jsonify({"added": True})


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


@app.route("/api/chat", methods=["POST"])
def api_chat():  # noqa: ANN201
    """Relay a message to the Agent Bricks agent.

    Returns 501 with a usable message when no endpoint is configured, rather
    than failing obscurely. Every button on the page works either way, because
    the agent is an additional way in and not the only one.
    """
    if not AGENT_ENDPOINT:
        return jsonify(
            {
                "error": (
                    "No agent endpoint configured on this deployment. The agent "
                    "runs in Databricks; set JOBRADAR_AGENT_ENDPOINT to wire it "
                    "into this page."
                )
            }
        ), 501

    body = request.get_json(silent=True) or {}
    try:
        turns = _conversation(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        payload = _call_agent(turns)
    except Exception as exc:
        logger.warning("Agent call failed: %s", exc)
        return jsonify({"error": _agent_error(exc)}), 502

    # The raw output items go back to the client so it can send them again on
    # the next turn. See _conversation for why the visible text is not enough.
    output = payload.get("output") if isinstance(payload, dict) else None
    return jsonify(
        {
            "reply": _extract_reply(payload),
            "items": output if isinstance(output, list) else [],
        }
    )


# The whole conversation is sent on every turn, because the endpoint keeps no
# state between calls. Sending only the newest message is what makes an agent
# that works in the Databricks playground fail here: it has no idea it ever
# listed anything.
#
# And the conversation is not just the visible text. A Responses envelope
# carries the agent's tool CALLS and their RESULTS as items alongside the
# message, and those items are where the job ids live. Replaying only the prose
# produced exactly one bug, which is worth stating plainly because it looked
# like an agent problem and was not:
#
#   "save it" -> the agent had its own summary, no job id anywhere in context,
#   and wrote a plausible-looking id it had reconstructed. Postgres refused it:
#
#     insert or update on table "saved_jobs" violates foreign key constraint
#     "saved_jobs_job_id_fkey"
#
# The foreign key is why that became an error instead of a saved row pointing
# at nothing. The fix is to replay the items, so the id it reads is the id the
# tool returned.
MAX_EXCHANGES = 8
_ROLES = ("user", "assistant")

# A role this app must never forward from a request body. The system prompt is
# what tells the agent it cannot apply to anything on the user's behalf; the
# page has no business being able to add to it or replace it.
_FORBIDDEN_ROLES = ("system", "developer", "tool")


def _conversation(body: dict) -> list[dict]:
    """The items to send, from any of the three request shapes.

    `items` is the full replay, including tool calls - what the chat page
    sends. `messages` is a plain conversation, and `message` is a single turn;
    both stay supported so this is usable from a script.

    Raises ValueError with a message meant for a person, since every one of
    these is something the caller can fix.
    """
    raw = body.get("items")
    if raw is not None:
        return _replay(raw)

    raw = body.get("messages")
    if raw is None:
        message = (body.get("message") or "").strip()
        if not message:
            raise ValueError("message must not be empty")
        return [{"role": "user", "content": message}]

    if not isinstance(raw, list) or not raw:
        raise ValueError("messages must be a non-empty list")

    turns: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("every message must be an object with role and content")
        role = item.get("role")
        raw_content = item.get("content")
        content = raw_content.strip() if isinstance(raw_content, str) else ""
        if role not in _ROLES:
            raise ValueError(f"role must be one of {', '.join(_ROLES)}")
        if not content:
            raise ValueError("every message must have content")
        turns.append({"role": role, "content": content})

    if turns[-1]["role"] != "user":
        raise ValueError("the last message must be from the user")
    return _trim(turns)


def _replay(raw: object) -> list[dict]:
    """Validate a full item replay - messages plus tool calls and results."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("items must be a non-empty list")

    items: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("every item must be an object")
        role = item.get("role")
        if role in _FORBIDDEN_ROLES:
            raise ValueError(f"role must be one of {', '.join(_ROLES)}")
        items.append(item)

    last = items[-1]
    if last.get("role") != "user":
        raise ValueError("the last item must be from the user")
    return _trim(items)


def _trim(items: list[dict]) -> list[dict]:
    """Keep the last MAX_EXCHANGES exchanges, cut only at a user message.

    A long session eventually exceeds the context window, and the failure is a
    400 on a message that looks perfectly fine. Trimming to a flat item count
    would fix that and cause a worse one: a tool RESULT whose matching tool CALL
    fell off the front is a malformed conversation, and the endpoint rejects the
    whole request. Cutting only at a user message keeps every call with its
    result.
    """
    starts = [i for i, item in enumerate(items) if item.get("role") == "user"]
    if len(starts) <= MAX_EXCHANGES:
        return items
    return items[starts[-MAX_EXCHANGES] :]


# The HTTP status is the whole diagnosis, and an exception type name is none of
# it: "The agent is unavailable (HTTPError)" sent me to a notebook to find out
# what any one of these four words would have told me directly. The status is
# not sensitive - it is the response line, not the response - so it is shown.
# The body still is not, because it can echo a request header back.
_AGENT_ERRORS = {
    403: (
        "The app is not allowed to query the agent. Grant its service principal "
        "CAN QUERY on the serving endpoint."
    ),
    404: "No serving endpoint by that name. Check JOBRADAR_AGENT_ENDPOINT.",
    400: "The agent rejected the request shape. Its task type may have changed.",
    429: "The agent is rate limited. Try again in a moment.",
}


def _agent_error(exc: Exception) -> str:
    """A message that names the actual problem, without echoing the response."""
    status = next((int(word) for word in str(exc).split() if word.isdigit()), None)
    if status in _AGENT_ERRORS:
        return f"{_AGENT_ERRORS[status]} (HTTP {status})"
    if status:
        return f"The agent returned HTTP {status}."
    return f"The agent is unreachable ({type(exc).__name__})."


# An Agent Bricks agent is served as task type "Agent (Responses)", which takes
# the OpenAI Responses shape - {"input": [...]} - not the chat-completions
# {"messages": [...]} every other Databricks endpoint takes. Sending the wrong
# one is a 400 that reads like an auth or availability problem.
#
# Both are sent rather than one, in that order, because the task type is a
# property of how the agent was published and can change without this code
# knowing. Trying the second shape only on a 4xx costs one wasted request in the
# rare case and never silently returns the wrong thing.
_AGENT_REQUEST_KEYS = ("input", "messages")

# A 4xx is the endpoint rejecting this body; the other shape may be accepted.
# A 5xx is the endpoint itself failing, and a different body will not help.
_HTTP_BAD_REQUEST = 400
_HTTP_SERVER_ERROR = 500


def _call_agent(turns: list[dict[str, str]]) -> Any:  # noqa: ANN401 - whatever it sent
    """POST the conversation to the serving endpoint, trying each shape in turn.

    Raises the last error if every shape was rejected, so the caller reports a
    real failure rather than an empty reply.
    """
    import requests  # noqa: PLC0415
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    client = WorkspaceClient()
    url = (
        f"{(client.config.host or '').rstrip('/')}"
        f"/serving-endpoints/{AGENT_ENDPOINT}/invocations"
    )
    headers = {**(client.config.authenticate() or {}), "Content-Type": "application/json"}

    last: Exception | None = None
    for key in _AGENT_REQUEST_KEYS:
        body = {key: turns}
        response = requests.post(url, headers=headers, json=body, timeout=120)
        if response.status_code < _HTTP_BAD_REQUEST:
            return response.json()
        last = requests.HTTPError(
            f"{response.status_code} from the agent endpoint: {response.text[:300]}"
        )
        if response.status_code >= _HTTP_SERVER_ERROR:
            break
    raise last or RuntimeError("no request shape was attempted")


def _text_of(item: Any) -> str | None:  # noqa: ANN401 - one node of an unknown envelope
    """The text carried by one item of a response list, or None.

    Handles the three ways an item carries text: as a bare string, as a
    `content`/`text`/`message` string, or - the Responses API shape - as a list
    of typed blocks of which only some are text. A tool-call block has no text
    and must not be mistaken for an empty reply.
    """
    if isinstance(item, str):
        return item or None
    if not isinstance(item, dict):
        return None

    for key in ("content", "text", "message", "output_text"):
        found = item.get(key)
        if isinstance(found, str) and found:
            return found
        if isinstance(found, dict):
            nested = _text_of(found)
            if nested:
                return nested
        if isinstance(found, list):
            blocks = [t for t in (_text_of(block) for block in found) if t]
            if blocks:
                return "\n".join(blocks)
    return None


def _extract_reply(payload: Any) -> str:  # noqa: ANN401 - whatever the endpoint sent
    """Pull the text out of whatever shape the endpoint returned.

    Serving endpoints differ in how they wrap a reply, and a page that only
    understands one shape breaks on a model swap. This understands
    chat-completions (`choices`), Responses (`output`), and the plain forms.

    Lists are scanned from the END BACKWARDS, not indexed at [-1]: a Responses
    envelope interleaves reasoning and tool-call items with the message, and the
    final item is often a tool call carrying no text at all. The last item that
    actually has text is the reply.
    """
    if isinstance(payload, dict):
        for key in ("output", "choices", "messages", "predictions"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in reversed(value):
                    text = _text_of(item)
                    if text:
                        return text
        for key in ("content", "text", "answer", "result"):
            found = payload.get(key)
            if isinstance(found, str) and found:
                return found
    elif isinstance(payload, str) and payload:
        return payload

    # Nothing recognisable. Showing the raw envelope beats showing nothing,
    # because it is the only clue about a shape this does not yet handle.
    return str(payload)[:2000]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def _status_payload() -> tuple[dict, int]:
    try:
        return {"status": "ok", "counts": repository.stats()}, 200
    except Exception as exc:
        return {"status": "degraded", "database": type(exc).__name__}, 503


@app.route("/healthz")
def healthz():  # noqa: ANN201
    payload, status_code = _status_payload()
    return jsonify(payload), status_code


@app.route("/status")
def status():  # noqa: ANN201
    """The same payload on a path no platform will claim.

    Databricks Apps intercepts /healthz for its own probing, so a page asking
    it for counts gets the platform's answer and shows "unavailable" on a
    perfectly healthy app. Learned twice already; applied up front here.
    """
    payload, status_code = _status_payload()
    return jsonify(payload), status_code


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("DATABRICKS_APP_PORT") or os.environ.get("PORT") or 8000),
        debug=os.environ.get("FLASK_DEBUG", "").lower() == "true",
    )
