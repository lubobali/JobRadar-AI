"""Turning ATS description HTML into plain text.

Greenhouse (and Lever/Ashby) hand back HTML with the markup itself escaped, so the
real-world input is nastier than it looks. Cases here are taken from the recorded
fixture, not invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobradar.html_text import to_plain_text

FIXTURE = Path(__file__).parent / "fixtures" / "greenhouse_jobs.json"


class TestToPlainText:
    def test_strips_tags(self) -> None:
        assert to_plain_text("<p>Data Engineer</p>") == "Data Engineer"

    def test_unescapes_greenhouse_style_escaped_markup(self) -> None:
        assert to_plain_text("&lt;h2&gt;&lt;strong&gt;About&lt;/strong&gt;&lt;/h2&gt;") == "About"

    def test_decodes_entities_that_survive_tag_stripping(self) -> None:
        """The double-escape trap: &amp;nbsp; unescapes to &nbsp;, which is still an entity."""
        assert to_plain_text("&lt;p&gt;A&amp;nbsp;B&lt;/p&gt;") == "A B"

    def test_collapses_whitespace_runs(self) -> None:
        assert to_plain_text("<p>A</p>\n\n\n   <p>B</p>") == "A B"

    def test_tags_become_a_separator_not_a_join(self) -> None:
        """<li>a</li><li>b</li> must not read as 'ab'."""
        assert to_plain_text("<li>Python</li><li>SQL</li>") == "Python SQL"

    def test_plain_text_passes_through(self) -> None:
        assert to_plain_text("Data Engineer") == "Data Engineer"

    @pytest.mark.parametrize("empty", ["", "   ", "<p></p>"])
    def test_empty_input_yields_empty_string(self, empty: str) -> None:
        assert to_plain_text(empty) == ""

    def test_handles_a_real_recorded_description(self) -> None:
        raw = json.loads(FIXTURE.read_text())["jobs"][0]["content"]
        text = to_plain_text(raw)

        assert text.startswith("About Anthropic")
        assert "<" not in text and ">" not in text
        assert "&nbsp;" not in text and "&amp;" not in text
        assert "  " not in text
