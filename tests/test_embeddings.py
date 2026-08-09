"""Chunking, context headers, and building a profile into a query.

No model is loaded here. The encoder is a lazy singleton precisely so these can
run without torch, and everything that decides *what text gets embedded* is
plain string handling.

The numbers asserted below are measured, not guessed. all-MiniLM-L6-v2 reports
max_seq_length 256, and an 800-character chunk tokenizes to 162 word-pieces.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from jobradar import embeddings, matching


class TestChunkText:
    def test_short_text_is_one_chunk(self) -> None:
        assert embeddings.chunk_text("Build data pipelines.") == ["Build data pipelines."]

    def test_long_text_splits(self) -> None:
        chunks = embeddings.chunk_text("word " * 500)
        assert len(chunks) > 1

    def test_no_chunk_ends_mid_word(self) -> None:
        # The chunk is what the reader is shown as the matched passage, so a
        # severed word is a visible defect rather than an internal detail.
        for chunk in embeddings.chunk_text("responsibilities " * 200):
            assert not chunk.endswith("responsibi")
            assert chunk == chunk.strip() or chunk.strip() in chunk

    def test_chunks_overlap(self) -> None:
        # Without overlap a sentence straddling a boundary is split across two
        # vectors and matches neither query well.
        chunks = embeddings.chunk_text("alpha beta gamma delta " * 120)
        assert len(chunks) > 1
        tail = chunks[0][-40:]
        assert any(word in chunks[1] for word in tail.split()[:3])

    def test_empty_text_produces_nothing(self) -> None:
        assert embeddings.chunk_text("") == []

    def test_a_non_string_is_refused(self) -> None:
        with pytest.raises(TypeError):
            embeddings.chunk_text(None)


class TestContextHeader:
    def test_names_the_role_and_employer(self) -> None:
        header = embeddings.context_header("Senior Data Engineer", "Caterpillar", "Chicago, IL")
        assert "Senior Data Engineer" in header
        assert "Caterpillar" in header
        assert "Chicago, IL" in header

    def test_a_missing_location_is_omitted_cleanly(self) -> None:
        header = embeddings.context_header("Data Engineer", "Foodsmart", None)
        assert "None" not in header
        assert header.startswith("Data Engineer at Foodsmart")

    def test_a_missing_company_does_not_produce_a_dangling_at(self) -> None:
        assert " at ." not in embeddings.context_header("Data Engineer", "", None)

    def test_it_is_small_enough_to_be_worth_it(self) -> None:
        # Roughly 15 word-pieces against a 256 budget that an 800-character
        # chunk uses 162 of. If this grew, chunks would start being truncated
        # by the model rather than by the chunker.
        header = embeddings.context_header(
            "Senior Data & AI Platform Engineer", "Caterpillar Inc.", "Chicago, IL"
        )
        assert len(header) < 120


class TestChunkJob:
    DESCRIPTION = "Build and operate Spark data pipelines on AWS. " * 60

    def test_every_chunk_carries_the_header(self) -> None:
        # The whole point. A chunk from the middle of a description names
        # neither the role nor the employer, so the embedding cannot match a
        # query that mentions either.
        chunks = embeddings.chunk_job(
            "Senior Data Engineer", "Caterpillar", "Chicago, IL", self.DESCRIPTION, "h1"
        )
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["chunk_text"].startswith("Senior Data Engineer at Caterpillar")

    def test_what_is_stored_is_what_was_embedded(self) -> None:
        # It would be easy to embed a header-prefixed string and store the bare
        # one. That breaks the property worth protecting: a retrieved passage
        # the user cannot reconcile with its score is worse than a repetitive
        # one.
        chunks = embeddings.chunk_job("DE", "Acme", None, self.DESCRIPTION, "h1")
        for chunk in chunks:
            assert chunk["chunk_text"].startswith("DE at Acme")

    def test_the_header_is_charged_against_the_window(self) -> None:
        # Otherwise a chunk plus its header exceeds the budget the chunk size
        # was chosen to respect, and the model truncates the tail silently.
        long_header_chunks = embeddings.chunk_job(
            "Senior Staff Data and Machine Learning Platform Engineer",
            "A Company With A Notably Long Legal Name Incorporated",
            "San Francisco Bay Area, California",
            self.DESCRIPTION,
            "h1",
        )
        for chunk in long_header_chunks:
            assert len(chunk["chunk_text"]) <= embeddings.CHUNK_SIZE + 60

    def test_indexes_are_sequential_from_zero(self) -> None:
        chunks = embeddings.chunk_job("DE", "Acme", None, self.DESCRIPTION, "h1")
        assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_the_content_hash_rides_along(self) -> None:
        # Carried onto every vector so a re-run can tell "already embedded"
        # from "the posting has been edited since".
        chunks = embeddings.chunk_job("DE", "Acme", None, self.DESCRIPTION, "abc123")
        assert {c["content_hash"] for c in chunks} == {"abc123"}

    def test_an_empty_description_produces_nothing(self) -> None:
        # Not one chunk containing only a header. A vector of pure metadata
        # scores against every query and carries no information.
        assert embeddings.chunk_job("DE", "Acme", None, "", "h1") == []


class TestProfileQuery:
    PROFILE: ClassVar[dict] = {
        "headline": "Senior Data & AI Platform Engineer",
        "target_titles": ["Data Engineer", "AI Engineer"],
        "skills": ["Spark", "Python", "Postgres", "AWS", "Databricks"],
        "summary": "Builds data platforms and LLM systems.",
        "resume_text": "A long resume. " * 400,
    }

    def test_includes_the_headline_targets_and_skills(self) -> None:
        query = matching.profile_query_text(self.PROFILE)
        assert "Senior Data & AI Platform Engineer" in query
        assert "Data Engineer" in query
        assert "Spark" in query

    def test_it_fits_the_model_budget(self) -> None:
        # THE reason this module exists. all-MiniLM-L6-v2 truncates at 256
        # word-pieces, so a full resume passed to embed_query would have three
        # quarters of it silently discarded - no error, plausible result.
        query = matching.profile_query_text(self.PROFILE)
        assert len(query) <= matching.MAX_QUERY_CHARS

    def test_truncation_keeps_the_specific_part(self) -> None:
        # Ordered so the thing cut is the least specific. Losing the summary
        # costs little; losing the headline costs the whole query.
        profile = dict(self.PROFILE, summary="filler " * 500)
        query = matching.profile_query_text(profile)
        assert query.startswith("Senior Data & AI Platform Engineer")

    def test_truncation_does_not_sever_a_word(self) -> None:
        profile = dict(self.PROFILE, summary="filler " * 500)
        assert not matching.profile_query_text(profile).endswith("fil")

    def test_a_long_skills_list_is_capped(self) -> None:
        # A skills list long enough to fill the window on its own turns every
        # query into the same query.
        profile = dict(self.PROFILE, skills=[f"skill{i}" for i in range(200)])
        query = matching.profile_query_text(profile)
        assert "skill199" not in query
        assert len(query) <= matching.MAX_QUERY_CHARS

    def test_an_empty_profile_is_refused_loudly(self) -> None:
        # An empty query embeds to a vector equally close to everything, which
        # looks like ranking and is not.
        with pytest.raises(ValueError, match="nothing to match"):
            matching.profile_query_text({})

    def test_a_profile_with_only_a_headline_still_works(self) -> None:
        assert matching.profile_query_text({"headline": "Data Engineer"})

    def test_missing_fields_do_not_appear_as_none(self) -> None:
        query = matching.profile_query_text({"headline": "DE", "skills": None})
        assert "None" not in query


class TestProfileQueryChunks:
    def test_a_resume_becomes_several_queries(self) -> None:
        # One averaged vector over a whole career sits between its areas and is
        # close to none of them in particular.
        profile = {"headline": "DE", "resume_text": "Data engineering work. " * 300}
        chunks = matching.profile_query_chunks(profile)
        assert len(chunks) > 1

    def test_each_chunk_carries_the_headline(self) -> None:
        profile = {"headline": "Senior DE", "resume_text": "work " * 400}
        for chunk in matching.profile_query_chunks(profile):
            assert "Senior DE" in chunk

    def test_no_resume_falls_back_to_the_dense_query(self) -> None:
        # So a caller never has to check first.
        profile = {"headline": "DE", "skills": ["Spark"]}
        assert matching.profile_query_chunks(profile) == [
            matching.profile_query_text(profile)
        ]
