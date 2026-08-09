"""The chat relay between the Databricks App and the Agent Bricks agent.

The relay is the one place in the frontend that talks to something it does not
control, and the endpoint's contract is not fixed: an Agent Bricks agent is
published as task type "Agent (Responses)" and takes the OpenAI Responses shape,
while every other Databricks serving endpoint takes chat-completions. These
tests pin both directions - what gets sent, and what gets read back out.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from flask.testing import FlaskClient

sys.path.insert(0, "app")

app_module = importlib.import_module("app")


# ---------------------------------------------------------------------------
# Reading the reply out of whatever the endpoint returned
# ---------------------------------------------------------------------------


class TestExtractReply:
    """`_extract_reply` against every envelope this could meet."""

    def test_responses_envelope_with_typed_blocks(self) -> None:
        """The Responses shape: content is a list of typed blocks, not a string."""
        payload = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Found 10 remote roles."}],
                }
            ]
        }
        assert app_module._extract_reply(payload) == "Found 10 remote roles."

    def test_chat_completions_envelope(self) -> None:
        payload = {"choices": [{"message": {"role": "assistant", "content": "Logged."}}]}
        assert app_module._extract_reply(payload) == "Logged."

    def test_trailing_tool_call_is_not_mistaken_for_the_reply(self) -> None:
        """The last item is often a tool call with no text.

        Indexing at [-1] returns the tool call, and the user sees a stringified
        function call instead of an answer. The scan runs backwards to the last
        item that actually carries text.
        """
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "the real answer"}],
                },
                {"type": "function_call", "name": "search_jobs", "arguments": "{}"},
            ]
        }
        assert app_module._extract_reply(payload) == "the real answer"

    def test_multiple_text_blocks_are_joined(self) -> None:
        payload = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "first"},
                        {"type": "output_text", "text": "second"},
                    ]
                }
            ]
        }
        assert app_module._extract_reply(payload) == "first\nsecond"

    def test_reasoning_item_before_the_message_is_skipped(self) -> None:
        """A reasoning item carries no text and must not shadow the message."""
        payload = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
                {"type": "reasoning", "summary": []},
            ]
        }
        assert app_module._extract_reply(payload) == "answer"

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"content": "flat content"}, "flat content"),
            ({"text": "flat text"}, "flat text"),
            ({"answer": "flat answer"}, "flat answer"),
            ({"messages": ["a bare string"]}, "a bare string"),
            ("a bare payload", "a bare payload"),
        ],
    )
    def test_plain_shapes(self, payload: object, expected: str) -> None:
        assert app_module._extract_reply(payload) == expected

    def test_unrecognised_shape_shows_the_envelope(self) -> None:
        """Showing the raw envelope beats showing nothing.

        It is the only clue about a shape this does not handle yet, and an empty
        chat bubble is indistinguishable from a hang.
        """
        result = app_module._extract_reply({"surprise": {"nested": 1}})
        assert "surprise" in result

    def test_long_envelope_is_truncated(self) -> None:
        result = app_module._extract_reply({"surprise": "x" * 5000})
        assert len(result) <= 2000

    def test_empty_strings_do_not_count_as_a_reply(self) -> None:
        """An empty string is not text; the scan should keep looking."""
        payload = {"output": [{"content": ""}, {"content": "real"}]}
        assert app_module._extract_reply(payload) == "real"


# ---------------------------------------------------------------------------
# What gets sent
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, payload: object = None, text: str = "") -> None:
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self) -> object:
        return self._payload


class TestCallAgent:
    """`_call_agent` and the two request shapes."""

    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, responses: list[_FakeResponse]) -> list[dict]:
        """Install fake `requests` and `databricks.sdk`, return the sent bodies."""
        sent: list[dict] = []
        queue = list(responses)

        def fake_post(url: str, headers: dict, json: dict, timeout: float) -> _FakeResponse:
            sent.append(json)
            return queue.pop(0)

        fake_requests = type(
            "requests", (), {"post": staticmethod(fake_post), "HTTPError": RuntimeError}
        )
        fake_config = type("cfg", (), {"host": "https://example.databricks.com"})()
        fake_client = type(
            "WorkspaceClient",
            (),
            {"config": fake_config, "__init__": lambda self: None},
        )
        fake_config.authenticate = lambda: {"Authorization": "Bearer x"}  # type: ignore[attr-defined]
        fake_sdk = type("sdk", (), {"WorkspaceClient": fake_client})

        monkeypatch.setitem(sys.modules, "requests", fake_requests)
        monkeypatch.setitem(sys.modules, "databricks.sdk", fake_sdk)
        monkeypatch.setattr(app_module, "AGENT_ENDPOINT", "mas-test-endpoint")
        return sent

    def test_responses_shape_is_tried_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An Agent Bricks agent takes `input`, so that shape goes first."""
        sent = self._patch(monkeypatch, [_FakeResponse(200, {"output": []})])
        app_module._call_agent("hello")
        assert list(sent[0]) == ["input"]
        assert sent[0]["input"] == [{"role": "user", "content": "hello"}]

    def test_falls_back_to_chat_completions_on_a_4xx(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 400 means the wrong shape, which the other shape may fix."""
        sent = self._patch(
            monkeypatch,
            [_FakeResponse(400, text="unexpected field"), _FakeResponse(200, {"choices": []})],
        )
        app_module._call_agent("hello")
        assert [next(iter(body)) for body in sent] == ["input", "messages"]

    def test_a_5xx_stops_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The endpoint itself is unhappy; a different body will not help."""
        sent = self._patch(monkeypatch, [_FakeResponse(503, text="overloaded")])
        with pytest.raises(Exception, match="503"):
            app_module._call_agent("hello")
        assert len(sent) == 1

    def test_every_shape_rejected_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two 4xx responses must raise, not return an empty reply."""
        self._patch(monkeypatch, [_FakeResponse(400, text="no"), _FakeResponse(422, text="no")])
        with pytest.raises(Exception, match="422"):
            app_module._call_agent("hello")


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> FlaskClient:
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


class TestChatRoute:
    def test_no_endpoint_configured_returns_501_and_says_why(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The page must work without the agent, and explain rather than fail.

        The frontend is not allowed to depend on the agent being reachable;
        every button still works when this returns 501.
        """
        monkeypatch.setattr(app_module, "AGENT_ENDPOINT", "")
        response = client.post("/api/chat", json={"message": "hi"})
        assert response.status_code == 501
        assert "JOBRADAR_AGENT_ENDPOINT" in response.get_json()["error"]

    def test_empty_message_is_rejected(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_module, "AGENT_ENDPOINT", "mas-test-endpoint")
        response = client.post("/api/chat", json={"message": "   "})
        assert response.status_code == 400

    def test_missing_body_is_rejected(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_module, "AGENT_ENDPOINT", "mas-test-endpoint")
        response = client.post("/api/chat", json={})
        assert response.status_code == 400

    def test_a_failing_endpoint_becomes_502_without_leaking_internals(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The user gets a type name, not a stack trace or a token."""
        monkeypatch.setattr(app_module, "AGENT_ENDPOINT", "mas-test-endpoint")

        def boom(message: str) -> object:
            raise ValueError("Bearer supersecrettoken leaked in here")

        monkeypatch.setattr(app_module, "_call_agent", boom)
        response = client.post("/api/chat", json={"message": "hi"})
        assert response.status_code == 502
        assert "supersecrettoken" not in response.get_data(as_text=True)
        assert "ValueError" in response.get_json()["error"]

    def test_a_successful_call_returns_the_reply(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_module, "AGENT_ENDPOINT", "mas-test-endpoint")
        monkeypatch.setattr(
            app_module,
            "_call_agent",
            lambda message: {
                "output": [{"content": [{"type": "output_text", "text": f"echo: {message}"}]}]
            },
        )
        response = client.post("/api/chat", json={"message": "find me spark jobs"})
        assert response.status_code == 200
        assert response.get_json()["reply"] == "echo: find me spark jobs"
