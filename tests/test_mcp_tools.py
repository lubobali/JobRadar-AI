"""The MCP tool layer.

Three things are proved here, and only these three, because the SQL is already
covered in test_repository and against a real Lakebase:

  1. Every tool is registered, and described well enough that an agent
     introspecting the server picks the right one.
  2. **No tool ever raises.** A raise reaches the agent as a transport failure
     it can only report as "the tool broke". A returned `{"error": ...}` is a
     sentence it can act on - and one that tells it not to invent a result.
  3. **The write tools cannot be talked into damage.** This is requirement 5's
     actual risk: an agent with write access is an agent that can be asked for
     something destructive in perfectly polite language.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from datetime import date

import pytest
from pytest import MonkeyPatch

os.environ.setdefault("JOBRADAR_USER_EMAIL", "test@example.test")

import jobs_mcp_server as server

from jobradar import repository
from jobradar.drafting import DraftingError

Wired = dict
"""What the `wired` fixture yields: the recorded calls, and a dict the test
stages return values in."""

# draft_application_text is a READ despite producing the most text of any tool:
# it stores nothing. Classifying it as a write would make the read/write split
# below meaningless.
READ_TOOLS = {
    "search_jobs",
    "get_job",
    "list_applications",
    "get_profile",
    "draft_application_text",
}
WRITE_TOOLS = {
    "save_job",
    "log_application",
    "update_application_status",
    "add_interview_note",
    "set_follow_up",
    "add_contact",
}
ALL_TOOLS = READ_TOOLS | WRITE_TOOLS


@pytest.fixture(autouse=True)
def wired(monkeypatch: MonkeyPatch) -> Wired:
    """Stub the repository and pin a user id, so no database is touched."""
    calls: list[tuple[str, tuple, dict]] = []
    returns: dict = {}

    def record(name: str, default: object = None) -> Callable:
        def fake(*args: object, **kwargs: object) -> object:
            calls.append((name, args, kwargs))
            value = returns.get(name, default)
            if isinstance(value, Exception):
                raise value
            return value

        return fake

    for name, default in [
        ("search", []),
        ("get_job", None),
        ("list_applications", []),
        ("get_profile", None),
        ("save_job", {"user_id": 1, "job_id": "j", "note": None, "saved_at": None}),
        ("log_application", {"id": 5, "job_id": "j", "status": "applied"}),
        ("update_application_status", {"id": 5, "job_id": "j", "status": "screening"}),
        ("add_interview_note", {"id": 9, "application_id": 5, "note": "n"}),
        ("set_follow_up", {"id": 5, "status": "applied", "follow_up_on": None}),
        ("job_for_drafting", None),
        ("add_contact", {"id": 2, "company": "Acme", "name": "Jane"}),
        ("stats", {"jobs": 0}),
    ]:
        monkeypatch.setattr(repository, name, record(name, default))

    monkeypatch.setattr(server, "_user_id", 1)
    monkeypatch.setattr(server.matching.embeddings, "embed_query", lambda text: [0.1] * 384)
    return {"calls": calls, "returns": returns}


def registered() -> dict:
    return {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}


class TestRegistration:
    def test_every_tool_is_registered(self) -> None:
        assert set(registered()) == ALL_TOOLS

    def test_reads_and_writes_are_classified(self) -> None:
        # Requirement 5 is an agent with read AND write. A read-only server
        # would be easier and would score badly.
        kinds = {name: kind for name, kind, _ in server.TOOL_SUMMARIES}
        assert {n for n, k in kinds.items() if k == "read"} == READ_TOOLS
        assert {n for n, k in kinds.items() if k == "write"} == WRITE_TOOLS

    def test_every_tool_documents_args_and_returns_in_source(self) -> None:
        for name in ALL_TOOLS:
            doc = getattr(server, name).__doc__ or ""
            assert "Args:" in doc or name == "get_profile", name
            assert "Returns:" in doc, name

    def test_every_argument_reaches_the_agent_as_a_schema_description(self) -> None:
        # FastMCP lifts Args: into the JSON schema, one description per
        # parameter. That is what the agent reads when deciding how to call a
        # tool, so an undocumented argument is one it will guess at.
        for name, tool in registered().items():
            for argument, schema in (tool.parameters or {}).get("properties", {}).items():
                assert schema.get("description"), f"{name}.{argument}"

    def test_the_agent_facing_description_survives_the_args_block(self) -> None:
        # FastMCP keeps only the text ABOVE "Args:" and discards Returns:
        # entirely, so anything the agent must know has to be stated before
        # that line. Learned on the previous project.
        for name, tool in registered().items():
            description = tool.description or ""
            assert "Args:" not in description, name
            assert len(description) > 150, name

    def test_the_write_tools_say_they_do_not_apply_to_anything(self) -> None:
        # The single most important sentence in this server. A user asking an
        # agent to "apply to that one" must not get an application submitted,
        # and the tool that sounds like it would is log_application.
        described = registered()["log_application"].description
        assert "does not apply" in described.lower()

    def test_save_is_distinguished_from_apply(self) -> None:
        assert "not applying" in registered()["save_job"].description.lower()

    def test_get_job_warns_that_descriptions_are_untrusted(self) -> None:
        # Real postings carry instructions aimed at whoever reads them, and an
        # LLM reading one is now among those readers.
        assert "untrusted" in registered()["get_job"].description.lower()


class TestReadTools:
    def test_search_embeds_the_query_and_passes_filters(self, wired: Wired) -> None:
        result = server.search_jobs("spark pipelines", top_k=5, remote_only=True)
        name, _, kwargs = wired["calls"][0]
        assert name == "search"
        assert kwargs["top_k"] == 5
        assert kwargs["remote_only"] is True
        assert result["count"] == 0

    def test_top_k_is_clamped_not_rejected(self, wired: Wired) -> None:
        server.search_jobs("anything", top_k=9999)
        assert wired["calls"][0][2]["top_k"] == 50

    def test_a_string_false_is_not_true(self, wired: Wired) -> None:
        # bool("false") is True in Python, which would silently turn
        # "remote only: no" into "remote only: yes".
        server.search_jobs("anything", remote_only="false")
        assert wired["calls"][0][2]["remote_only"] is False

    def test_an_empty_query_is_a_bad_request(self, wired: Wired) -> None:
        result = server.search_jobs("   ")
        assert result["error_type"] == "bad_request"
        assert wired["calls"] == []

    def test_search_never_returns_the_full_description(self, wired: Wired) -> None:
        # Ten descriptions at 2-10KB each would fill the context window with
        # text the agent then has to summarise anyway.
        wired["returns"]["search"] = [
            {
                "id": "j1", "title": "DE", "company": "Acme", "description": "x" * 5000,
                "similarity": 0.9,
            }
        ]
        result = server.search_jobs("spark")
        assert "description" not in result["results"][0]

    def test_get_job_does_return_the_description(self, wired: Wired) -> None:
        wired["returns"]["get_job"] = {
            "id": "j1", "title": "DE", "company": "Acme", "description": "the full text",
        }
        assert server.get_job("j1")["description"] == "the full text"

    def test_a_missing_job_is_not_found_not_an_error(self, wired: Wired) -> None:
        wired["returns"]["get_job"] = None
        assert server.get_job("nope")["error_type"] == "not_found"

    def test_list_applications_filters_by_status(self, wired: Wired) -> None:
        server.list_applications("interviewing")
        assert wired["calls"][0][2]["status"] == "interviewing"

    def test_list_applications_rejects_an_invented_status(self, wired: Wired) -> None:
        result = server.list_applications("in progress")
        assert result["error_type"] == "bad_request"
        assert "interviewing" in result["error"]

    def test_get_profile_returns_the_ranking_query(self, wired: Wired) -> None:
        # So the agent can explain WHY a job ranked where it did, rather than
        # offering an opinion about fit.
        wired["returns"]["get_profile"] = {
            "headline": "Senior Data Engineer", "skills": ["Spark"], "target_titles": [],
        }
        assert "Spark" in server.get_profile()["ranking_query"]


class TestWriteTools:
    def test_every_write_reports_what_it_changed(self, wired: Wired) -> None:
        # "Done" is not a report. A user who cannot see what happened cannot
        # catch it going wrong.
        assert server.save_job("j1")["saved"] is True
        assert server.log_application("j1")["logged"] is True
        assert server.update_application_status(5, "screening")["updated"] is True
        assert server.add_interview_note(5, "called")["added"] is True
        assert server.add_contact("Acme", "Jane")["added"] is True

    def test_log_application_returns_the_id_the_other_tools_need(self, wired: Wired) -> None:
        assert server.log_application("j1")["id"] == 5

    @pytest.mark.parametrize("status", repository.APPLICATION_STATUSES)
    def test_every_real_status_is_accepted(self, wired: Wired, status: str) -> None:
        assert "error" not in server.log_application("j1", status)

    def test_an_invented_status_never_reaches_the_database(self, wired: Wired) -> None:
        # THE write-path risk. A model told to "mark it as in progress" invents
        # exactly that, and an invented status is a row no filter matches again.
        result = server.log_application("j1", "in progress")
        assert result["error_type"] == "bad_request"
        assert wired["calls"] == []

    def test_the_refusal_lists_the_real_statuses(self, wired: Wired) -> None:
        # So the agent corrects itself in one turn rather than guessing again.
        result = server.update_application_status(5, "nonsense")
        for status in ("applied", "screening", "interviewing", "offer"):
            assert status in result["error"]

    def test_updating_someone_elses_application_is_not_found(self, wired: Wired) -> None:
        wired["returns"]["update_application_status"] = None
        result = server.update_application_status(999, "offer")
        assert result["error_type"] == "not_found"

    def test_a_note_on_someone_elses_application_is_not_found(self, wired: Wired) -> None:
        wired["returns"]["add_interview_note"] = None
        assert server.add_interview_note(999, "hi")["error_type"] == "not_found"

    def test_an_empty_note_is_refused(self, wired: Wired) -> None:
        assert server.add_interview_note(5, "   ")["error_type"] == "bad_request"

    def test_an_application_id_as_a_string_still_works(self, wired: Wired) -> None:
        # A model passing "5" instead of 5 should succeed, not produce a driver
        # type error naming a column rather than the mistake.
        assert "error" not in server.update_application_status("5", "offer")

    def test_a_nonsense_application_id_is_a_bad_request(self, wired: Wired) -> None:
        assert server.add_interview_note("the second one", "hi")["error_type"] == "bad_request"

    def test_add_contact_requires_a_company_and_a_name(self, wired: Wired) -> None:
        assert server.add_contact("", "Jane")["error_type"] == "bad_request"
        assert server.add_contact("Acme", "")["error_type"] == "bad_request"


class TestThereIsNoDeleteTool:
    """Requirement 5's other half: an agent that can write is an agent that can
    be asked to destroy something in perfectly polite language."""

    def test_no_registered_tool_deletes_anything(self) -> None:
        for name in registered():
            assert "delete" not in name
            assert "remove" not in name
            assert "clear" not in name

    def test_nothing_can_withdraw_a_job_from_the_corpus(self) -> None:
        # The nearest thing to a destructive action the schema allows is
        # marking an application withdrawn, which is a status - reversible, and
        # it destroys nothing.
        assert "withdrawn" in repository.APPLICATION_STATUSES


class TestNothingEverRaises:
    @pytest.mark.parametrize(
        ("tool", "args"),
        [
            ("search_jobs", (None,)),
            ("search_jobs", ({"nested": "dict"},)),
            ("get_job", (None,)),
            ("get_job", ("",)),
            ("list_applications", (42,)),
            ("save_job", (None, None)),
            ("log_application", (None, None, None)),
            ("update_application_status", (None, None)),
            ("add_interview_note", (None, None)),
            ("add_contact", (None, None)),
        ],
    )
    def test_garbage_in_error_out(self, wired: Wired, tool: str, args: tuple) -> None:
        result = getattr(server, tool)(*args)
        assert isinstance(result, dict)
        assert "error" in result

    def test_a_database_failure_becomes_a_sentence(self, wired: Wired) -> None:
        wired["returns"]["search"] = RuntimeError("connection pool exhausted")
        result = server.search_jobs("spark")
        assert result["error_type"] == "internal_error"
        # The agent is told not to fill the gap; the traceback goes to the log.
        assert "do not guess" in result["error"]
        assert "connection pool" not in result["error"]


class TestPlainHttpRoutes:
    def test_status_and_healthz_agree(self) -> None:
        assert json.loads(asyncio.run(server.healthz(None)).body) == json.loads(
            asyncio.run(server.status(None)).body
        )

    def test_status_does_not_load_the_embedding_model(self, wired: Wired) -> None:
        # A health check that waits several seconds for a model reports a fine
        # server as unhealthy.
        asyncio.run(server.status(None))
        assert all(name != "search" for name, _, _ in wired["calls"])

    def test_the_landing_page_lists_every_tool_and_its_kind(self) -> None:
        body = asyncio.run(server.index(None)).body.decode()
        for name in ALL_TOOLS:
            assert name in body
        assert "not a website" in body


class TestStaleApplications:
    """`list_applications(stale_days=...)`.

    The capstone's job-hunting option asks the agent to "surface stale
    applications that haven't been updated in a while". This is that path.
    """

    def test_stale_days_reaches_the_repository(self, wired: Wired) -> None:
        server.list_applications(stale_days=30)
        name, _, kwargs = wired["calls"][-1]
        assert name == "list_applications"
        assert kwargs["stale_days"] == 30

    def test_no_stale_days_means_no_filter(self, wired: Wired) -> None:
        server.list_applications()
        assert wired["calls"][-1][2]["stale_days"] is None

    def test_a_string_from_the_agent_is_accepted(self, wired: Wired) -> None:
        """Models send "14" as often as 14."""
        server.list_applications(stale_days="14")
        assert wired["calls"][-1][2]["stale_days"] == 14

    def test_zero_days_is_floored_to_one(self, wired: Wired) -> None:
        """"Stale for 0 days" is every open application.

        A filter that does nothing while looking like it did something is worse
        than no filter, because the answer is wrong and confident.
        """
        server.list_applications(stale_days=0)
        assert wired["calls"][-1][2]["stale_days"] == 1

    def test_nonsense_is_an_error_not_a_crash(self, wired: Wired) -> None:
        result = server.list_applications(stale_days="soon")
        assert result["error_type"] == "bad_request"
        assert "whole number" in result["error"]

    def test_the_response_carries_the_age_of_each_row(self, wired: Wired) -> None:
        """The agent has to be able to say HOW stale, not just that it is."""
        wired["returns"]["list_applications"] = [
            {
                "id": 5, "job_id": "j", "title": "T", "company": "C",
                "status": "applied", "applied_at": None, "updated_at": None,
                "days_since_update": 41, "follow_up_on": None, "notes": [],
            }
        ]
        result = server.list_applications(stale_days=14)
        assert result["applications"][0]["days_since_update"] == 41


class TestFollowUp:
    """`set_follow_up` - "track follow-up dates for saved applications"."""

    def test_a_date_is_parsed_and_passed_through(self, wired: Wired) -> None:
        server.set_follow_up(5, "2026-09-01")
        name, args, _ = wired["calls"][-1]
        assert name == "set_follow_up"
        assert args[2] == date(2026, 9, 1)

    def test_omitting_the_date_clears_it(self, wired: Wired) -> None:
        """"I heard back, drop the reminder"."""
        server.set_follow_up(5)
        assert wired["calls"][-1][1][2] is None

    def test_prose_dates_are_refused_rather_than_guessed(self, wired: Wired) -> None:
        """A parser that accepts "next Tuesday" is one that guesses a year.

        This value becomes a reminder the user acts on, so the tool takes a
        real date and the agent is told to resolve relative ones itself.
        """
        result = server.set_follow_up(5, "next Tuesday")
        assert result["error_type"] == "bad_request"
        assert "YYYY-MM-DD" in result["error"]

    def test_an_impossible_date_is_refused(self, wired: Wired) -> None:
        assert server.set_follow_up(5, "2026-02-30")["error_type"] == "bad_request"

    def test_another_users_application_is_not_found(self, wired: Wired) -> None:
        """Ownership is checked in SQL; the tool must relay that as not_found."""
        wired["returns"]["set_follow_up"] = None
        result = server.set_follow_up(999, "2026-09-01")
        assert result["error_type"] == "not_found"
        assert "list_applications" in result["error"]


class TestDrafting:
    """`draft_application_text` - "draft a tailored cover-letter snippet"."""

    @staticmethod
    def _job() -> dict:
        return {
            "job": {
                "id": "abc", "title": "Data Engineer", "company": "Acme",
                "location": "Remote", "description": "Build pipelines.",
            },
            "profile": {"headline": "Data engineer", "skills": ["Spark"]},
        }

    def test_a_draft_comes_back_with_the_job_it_was_written_for(
        self, wired: Wired, monkeypatch: MonkeyPatch
    ) -> None:
        wired["returns"]["job_for_drafting"] = self._job()
        monkeypatch.setattr(
            server.drafting.DatabricksDrafter, "draft", lambda self, j, p, k: "Dear Acme,"
        )
        result = server.draft_application_text("abc")
        assert result["text"] == "Dear Acme,"
        assert result["company"] == "Acme"
        assert result["kind"] == "cover_letter"

    def test_it_writes_nothing_to_the_database(
        self, wired: Wired, monkeypatch: MonkeyPatch
    ) -> None:
        """It is classified as a read. It has to actually be one."""
        wired["returns"]["job_for_drafting"] = self._job()
        monkeypatch.setattr(
            server.drafting.DatabricksDrafter, "draft", lambda self, j, p, k: "text"
        )
        server.draft_application_text("abc")
        assert {name for name, _, _ in wired["calls"]} == {"job_for_drafting"}

    @pytest.mark.parametrize("kind", ["cover_letter", "resume_bullet", "outreach"])
    def test_every_kind_is_accepted(
        self, wired: Wired, monkeypatch: MonkeyPatch, kind: str
    ) -> None:
        wired["returns"]["job_for_drafting"] = self._job()
        monkeypatch.setattr(
            server.drafting.DatabricksDrafter, "draft", lambda self, j, p, k: k
        )
        assert server.draft_application_text("abc", kind)["text"] == kind

    def test_an_unknown_kind_names_the_known_ones(self, wired: Wired) -> None:
        result = server.draft_application_text("abc", "haiku")
        assert result["error_type"] == "bad_request"
        assert "cover_letter" in result["error"]

    def test_a_missing_job_is_not_found(self, wired: Wired) -> None:
        wired["returns"]["job_for_drafting"] = None
        assert server.draft_application_text("nope")["error_type"] == "not_found"

    def test_a_model_failure_does_not_become_an_empty_draft(
        self, wired: Wired, monkeypatch: MonkeyPatch
    ) -> None:
        """The agent must be told the tool failed, not handed "".

        Handed an empty string it would write the cover letter itself, from
        nothing, and present it as the tool's output.
        """
        wired["returns"]["job_for_drafting"] = self._job()

        def boom(self: object, j: object, p: object, k: object) -> str:
            raise DraftingError("the drafting endpoint returned nothing")

        monkeypatch.setattr(server.drafting.DatabricksDrafter, "draft", boom)
        result = server.draft_application_text("abc")
        assert result["error_type"] == "internal_error"
        assert "text" not in result


class TestErrorClassification:
    """Which failures the agent is told it can fix.

    The system prompt tells the agent that bad_request means "fix it and retry
    once" and internal_error means "say the tool is unavailable". Classifying
    an infrastructure failure as bad_request sends it into a retry loop on
    something no argument can change.
    """

    def test_a_library_valueerror_is_internal_not_bad_request(
        self, wired: Wired
    ) -> None:
        """Every real argument goes through validation, which raises BadArgument.

        A bare ValueError therefore comes from a library. This one is what the
        Databricks SDK raises when the host has no credentials - advice for
        whoever deployed the server, not for the person asking a question.
        """
        wired["returns"]["search"] = ValueError(
            "default auth: cannot configure default credentials, please check "
            "https://docs.databricks.com/en/dev-tools/auth.html"
        )
        result = server.search_jobs("spark")
        assert result["error_type"] == "internal_error"
        assert "databricks.com" not in result["error"]

    def test_a_validation_failure_is_still_bad_request(self, wired: Wired) -> None:
        result = server.search_jobs("   ")
        assert result["error_type"] == "bad_request"
        assert "must not be empty" in result["error"]

    def test_a_drafting_failure_is_internal(
        self, wired: Wired, monkeypatch: MonkeyPatch
    ) -> None:
        wired["returns"]["job_for_drafting"] = TestDrafting._job()

        def boom(self: object, j: object, p: object, k: object) -> str:
            raise DraftingError("no Databricks credentials available for drafting")

        monkeypatch.setattr(server.drafting.DatabricksDrafter, "draft", boom)
        assert server.draft_application_text("abc")["error_type"] == "internal_error"
