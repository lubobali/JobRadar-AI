"""Turning a profile into something the model can match jobs against.

This module exists because of one measured constraint. `all-MiniLM-L6-v2` has a
`max_seq_length` of **256 word-pieces**. A 5,000-character resume is roughly a
thousand of them, so `embed_query(resume_text)` silently embeds the first
quarter and discards the rest - with no error, no warning, and a perfectly
plausible-looking result. Matching would appear to work while ignoring most of
a career.

So the query is built rather than pasted. Two ways, and both are here because
they answer different questions:

`profile_query_text`
    One short, dense string - headline, target titles, skills - that fits
    inside the budget. Used to rank the whole corpus, because one vector
    against 33,000 is one query.

`profile_query_chunks`
    The resume chunked the same way job descriptions are, each chunk embedded
    separately. Used where recall matters more than cost: a resume covers
    several distinct areas, and one averaged vector sits in the middle of all
    of them, close to nothing in particular.

Neither is "correct". The first is what the nightly ranking uses; the second is
what a "find me anything touching Kafka" search wants.
"""

from __future__ import annotations

import logging

from jobradar import embeddings

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 900
"""About 180 word-pieces, leaving headroom under the model's 256. Measured
rather than guessed: an 800-character chunk tokenizes to 162."""

MAX_SKILLS = 25
"""Enough to describe a career, few enough that the list does not crowd out the
headline. A skills list long enough to fill the window on its own turns every
query into the same query."""


def profile_query_text(profile: dict) -> str:
    """Build one dense query string from a profile.

    Ordered deliberately: headline first, then target titles, then skills, then
    whatever summary space is left for. Truncation takes from the end, so the
    thing most likely to be cut is the least specific.

    The resume body is used last and only as filler. It is prose written for a
    human reader - "collaborated with stakeholders to deliver" - and prose like
    that is close to every job description ever written, which makes it noise
    in a similarity search rather than signal.
    """
    parts: list[str] = []

    headline = (profile.get("headline") or "").strip()
    if headline:
        parts.append(headline)

    targets = [str(t).strip() for t in (profile.get("target_titles") or []) if str(t).strip()]
    if targets:
        parts.append("Target roles: " + ", ".join(targets))

    skills = [str(s).strip() for s in (profile.get("skills") or []) if str(s).strip()]
    if skills:
        parts.append("Skills: " + ", ".join(skills[:MAX_SKILLS]))

    summary = (profile.get("summary") or "").strip()
    if summary:
        parts.append(summary)

    query = ". ".join(parts)

    if not query:
        # A profile with nothing in it cannot rank anything, and returning an
        # empty string would embed to a vector that is equally close to
        # everything - which looks like ranking, and is not.
        raise ValueError(
            "Profile has no headline, target titles, skills, or summary. "
            "There is nothing to match jobs against."
        )

    if len(query) > MAX_QUERY_CHARS:
        cut = query[:MAX_QUERY_CHARS].rsplit(" ", 1)[0]
        logger.info(
            "Profile query truncated from %s to %s characters", len(query), len(cut)
        )
        query = cut

    return query


def profile_query_chunks(profile: dict) -> list[str]:
    """The resume as several queries rather than one.

    A resume covers distinct areas - data engineering, AI, payments - and a
    single averaged vector sits between them, close to none of them in
    particular. Chunking keeps each area sharp, at the cost of one search per
    chunk.

    Falls back to the dense query when there is no resume body, so a caller
    never has to check first.
    """
    resume = (profile.get("resume_text") or "").strip()
    if not resume:
        return [profile_query_text(profile)]

    header = profile_query_text(profile)[:200]
    return [
        header + "\n\n" + chunk
        for chunk in embeddings.chunk_text(resume, chunk_size=MAX_QUERY_CHARS)
    ]


def embed_profile(profile: dict) -> list[float]:
    """One vector for a profile, for ranking the whole corpus."""
    return embeddings.embed_query(profile_query_text(profile))


__all__ = [
    "MAX_QUERY_CHARS",
    "MAX_SKILLS",
    "embed_profile",
    "profile_query_chunks",
    "profile_query_text",
]
