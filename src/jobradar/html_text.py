"""Convert ATS description HTML into the plain text the scorer will read.

Structure (lists, headings) is deliberately flattened: Phase 2 feeds this to an LLM and
matches keywords against it, and neither needs the markup. Readability of a *rendered*
description is not a goal here.
"""

from __future__ import annotations

import html
import re

_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def to_plain_text(raw_html: str) -> str:
    """Return `raw_html` as collapsed, entity-free plain text.

    The unescape happens twice by necessity. Greenhouse escapes the markup itself, so one
    pass turns `&lt;p&gt;` into a real `<p>` tag; any entity in the original *text* (a
    `&nbsp;` arrives as `&amp;nbsp;`) only becomes an entity after that pass, and needs a
    second one once the tags are gone.

    >>> to_plain_text("&lt;p&gt;Data&amp;nbsp;Engineer&lt;/p&gt;")
    'Data Engineer'
    >>> to_plain_text("<li>Python</li><li>SQL</li>")
    'Python SQL'
    >>> to_plain_text("")
    ''
    """
    markup = html.unescape(raw_html)
    text = _TAG.sub(" ", markup)
    return _WHITESPACE.sub(" ", html.unescape(text)).strip()
