"""The ingest rules, without a cluster.

Everything Spark-free lives in `jobradar.ingest` precisely so it can be tested
here. The notebook orchestrates; it does not decide. A rule about which
duplicate wins is a rule whether or not there is a SparkContext, and a rule
that only exists inside a notebook cell is a rule nothing can test.
"""

from __future__ import annotations

import pickle
from datetime import UTC, datetime, timedelta

import pytest
from pytest import MonkeyPatch

from jobradar import ingest, watchlist
from jobradar.models import Job
from jobradar.repository import content_hash, cross_source_key


def job(**overrides: object) -> Job:
    fields = {
        "source": "greenhouse",
        "source_id": "1",
        "company": "Caterpillar",
        "title": "Senior Data Engineer",
        "url": "https://example.test/1",
        "location": "Chicago, IL",
        "remote": False,
        "description": "Build Spark pipelines.",
        "posted_at": None,
        "fetched_at": datetime(2026, 8, 9, tzinfo=UTC),
    }
    fields.update(overrides)
    # `id` is derived from (source, source_id), not passed in - which is what
    # makes it stable across fetches.
    return Job(**fields)


class TestSourceSpecs:
    def test_every_source_is_represented(self) -> None:
        kinds = {spec.kind for spec in ingest.source_specs()}
        assert kinds == {
            "greenhouse", "ashby", "lever", "breezy",
            "remotive", "adzuna", "usajobs", "workday",
        }

    def test_the_watchlist_comes_across_whole(self) -> None:
        specs = ingest.source_specs()
        ats = [s for s in specs if s.kind in ("greenhouse", "ashby", "lever")]
        assert len(ats) == len(watchlist.WATCHLIST)

    def test_specs_pickle(self) -> None:
        # The entire reason this type exists. watchlist.all_sources() returns
        # closures, which are a fine shape for a thread pool and cannot cross
        # to a Spark executor. If this ever fails, the fan-out silently becomes
        # driver-only work.
        specs = ingest.source_specs()
        assert pickle.loads(pickle.dumps(specs)) == specs

    def test_a_spec_labels_itself_for_logs(self) -> None:
        spec = ingest.SourceSpec(kind="greenhouse", slug="nex", company="Nex")
        assert spec.label == "greenhouse:Nex"

    def test_workday_carries_all_three_address_parts(self) -> None:
        # Workday is the one board that needs tenant, site and host rather than
        # a slug. Dropping one produces a 404 that looks like a dead board.
        workdays = [s for s in ingest.source_specs() if s.kind == "workday"]
        assert workdays
        for spec in workdays:
            assert spec.tenant and spec.site and spec.host


class TestFetchSpec:
    def test_a_dead_board_returns_an_error_not_an_exception(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # This runs on an executor across 129 sources. One raising would take
        # the partition with it, and the fan-out exists so failures stay
        # independent.
        def explode(*args: object, **kwargs: object) -> list:
            raise ConnectionError("board is down")

        monkeypatch.setattr(ingest.greenhouse, "fetch_jobs", explode)
        jobs, error = ingest.fetch_spec(ingest.SourceSpec(kind="greenhouse", slug="dead"))
        assert jobs == []
        assert "ConnectionError" in error
        assert "board is down" in error

    def test_the_error_names_the_source(self, monkeypatch: MonkeyPatch) -> None:
        # A run that loses eleven boards has to be able to say which eleven.
        def explode(*args: object, **kwargs: object) -> list:
            raise ValueError("nope")

        monkeypatch.setattr(ingest.ashby, "fetch_jobs", explode)
        _, error = ingest.fetch_spec(
            ingest.SourceSpec(kind="ashby", slug="ramp", company="Ramp")
        )
        assert error.startswith("ashby:Ramp")

    def test_success_reports_no_error(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(ingest.greenhouse, "fetch_jobs", lambda *a, **k: [job()])
        jobs, error = ingest.fetch_spec(ingest.SourceSpec(kind="greenhouse", slug="nex"))
        assert len(jobs) == 1
        assert error is None

    def test_an_unknown_kind_is_reported_not_ignored(self) -> None:
        jobs, error = ingest.fetch_spec(ingest.SourceSpec(kind="linkedin"))
        assert jobs == []
        assert "no fetcher registered" in error


class TestToRow:
    def test_produces_every_declared_column(self) -> None:
        assert set(ingest.to_row(job())) == set(ingest.ROW_FIELDS)

    def test_strips_html_from_the_description(self) -> None:
        # The description feeds the chunker, the LLM scorer and the UI. Storing
        # markup means every one of them either strips it again or embeds
        # "<div>".
        row = ingest.to_row(job(description="<p>Build <b>Spark</b> pipelines.</p>"))
        assert "<" not in row["description"]
        assert "Spark" in row["description"]

    def test_hashes_the_stripped_text_not_the_markup(self) -> None:
        # Otherwise a board reformatting its HTML re-embeds every job it has.
        row = ingest.to_row(job(description="<p>Build Spark pipelines.</p>"))
        assert row["content_hash"] == content_hash(row["description"])

    def test_derives_the_cross_source_key(self) -> None:
        row = ingest.to_row(job())
        assert row["cross_source_key"] == cross_source_key(
            "Caterpillar", "Senior Data Engineer", "Chicago, IL"
        )

    def test_supplies_fetched_at_when_a_source_omits_it(self) -> None:
        assert ingest.to_row(job(fetched_at=None))["fetched_at"] is not None


class TestPriority:
    def test_an_ats_beats_an_aggregator(self) -> None:
        assert ingest.source_priority("greenhouse") < ingest.source_priority("adzuna")

    def test_an_unknown_source_sorts_last_rather_than_crashing(self) -> None:
        # A fetcher added later still ingests; it just never wins a tie.
        assert ingest.source_priority("brand-new") == ingest.DEFAULT_PRIORITY

    def test_prefer_keeps_the_ats_row(self) -> None:
        ats = {"source": "greenhouse", "description": "short"}
        aggregator = {"source": "adzuna", "description": "a much longer description"}
        # Longer, and still loses: an aggregator's text is a truncated relay,
        # and length is only the tiebreak within one priority level.
        assert ingest.prefer(ats, aggregator) is ats
        assert ingest.prefer(aggregator, ats) is ats

    def test_length_breaks_a_tie_within_one_level(self) -> None:
        short = {"source": "greenhouse", "description": "short"}
        long = {"source": "ashby", "description": "a considerably longer description"}
        assert ingest.prefer(short, long) is long

    def test_prefer_is_order_independent(self) -> None:
        # A run that picks a different winner each time re-embeds the same job
        # forever, because content_hash changes with the description.
        a = {"source": "lever", "description": "aaa"}
        b = {"source": "adzuna", "description": "bbbbbb"}
        assert ingest.prefer(a, b) is ingest.prefer(b, a)

    def test_a_missing_description_does_not_crash(self) -> None:
        assert ingest.prefer(
            {"source": "greenhouse", "description": None}, {"source": "adzuna"}
        )


class TestDeduplicate:
    def _row(self, **overrides: object) -> dict:
        return ingest.to_row(job(**overrides))

    def test_the_same_posting_twice_collapses(self) -> None:
        rows = [self._row(), self._row()]
        assert len(ingest.deduplicate(rows)) == 1

    def test_the_freshest_fetch_wins(self) -> None:
        older = self._row(fetched_at=datetime(2026, 8, 1, tzinfo=UTC))
        newer = self._row(fetched_at=datetime(2026, 8, 9, tzinfo=UTC))
        newer["description"] = "the updated description"
        result = ingest.deduplicate([older, newer])
        assert result[0]["description"] == "the updated description"

    def test_the_same_job_from_two_sources_collapses(self) -> None:
        # THE case round 1 cannot see. Different source ids, so different
        # primary keys, but one real job.
        greenhouse = self._row(source="greenhouse", source_id="gh-1")
        adzuna = self._row(source="adzuna", source_id="az-9")
        assert greenhouse["id"] != adzuna["id"]
        assert greenhouse["cross_source_key"] == adzuna["cross_source_key"]

        result = ingest.deduplicate([greenhouse, adzuna])
        assert len(result) == 1
        assert result[0]["source"] == "greenhouse"

    def test_order_does_not_change_the_survivor(self) -> None:
        greenhouse = self._row(source="greenhouse", source_id="gh-1")
        adzuna = self._row(source="adzuna", source_id="az-9")
        assert (
            ingest.deduplicate([greenhouse, adzuna])[0]["source"]
            == ingest.deduplicate([adzuna, greenhouse])[0]["source"]
        )

    def test_two_genuinely_different_jobs_both_survive(self) -> None:
        chicago = self._row(source_id="1", location="Chicago, IL")
        peoria = self._row(source_id="2", location="Peoria, IL")
        assert len(ingest.deduplicate([chicago, peoria])) == 2

    def test_different_companies_both_survive(self) -> None:
        cat = self._row(source_id="1", company="Caterpillar")
        food = self._row(source_id="2", company="Foodsmart")
        assert len(ingest.deduplicate([cat, food])) == 2

    def test_nothing_in_nothing_out(self) -> None:
        assert ingest.deduplicate([]) == []

    def test_a_missing_fetched_at_does_not_crash(self) -> None:
        row = self._row()
        row["fetched_at"] = None
        assert len(ingest.deduplicate([row, self._row()])) == 1

    @pytest.mark.parametrize("count", [1, 2, 10])
    def test_n_copies_collapse_to_one(self, count: int) -> None:
        rows = [self._row() for _ in range(count)]
        assert len(ingest.deduplicate(rows)) == 1

    def test_a_realistic_mix(self) -> None:
        now = datetime(2026, 8, 9, tzinfo=UTC)
        rows = [
            # one job, three sources
            self._row(source="greenhouse", source_id="gh-1", fetched_at=now),
            self._row(source="adzuna", source_id="az-1", fetched_at=now),
            self._row(source="remotive", source_id="rm-1", fetched_at=now),
            # a second, genuinely different job
            self._row(source="lever", source_id="lv-9", title="Staff Engineer"),
            # the first job again, fetched later
            self._row(
                source="greenhouse", source_id="gh-1", fetched_at=now + timedelta(hours=1)
            ),
        ]
        result = ingest.deduplicate(rows)
        assert len(result) == 2
        titles = {row["title"] for row in result}
        assert titles == {"Senior Data Engineer", "Staff Engineer"}
        assert {row["source"] for row in result} == {"greenhouse", "lever"}
