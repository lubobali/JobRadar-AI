"""The watchlist — probe-verified boards turned into pipeline fetchers."""

from __future__ import annotations

from jobradar.watchlist import (
    ADZUNA_QUERIES,
    REMOTIVE_SEARCHES,
    USAJOBS_QUERIES,
    WATCHLIST,
    WORKDAY_SEARCHES,
    AdzunaQuery,
    Board,
    UsaJobsQuery,
    adzuna_fetchers,
    all_sources,
    breezy_fetchers,
    remotive_fetchers,
    to_fetchers,
    usajobs_fetchers,
    workday_fetchers,
)


class TestWatchlist:
    def test_every_board_uses_a_known_fetcher(self) -> None:
        assert all(b.source in {"greenhouse", "lever", "ashby"} for b in WATCHLIST)

    def test_slugs_are_unique_per_source(self) -> None:
        """A duplicated board would fetch, score and pay for the same jobs twice."""
        keys = [(b.source, b.slug) for b in WATCHLIST]
        assert len(keys) == len(set(keys))

    def test_no_board_is_missing_a_company_name(self) -> None:
        assert all(b.company and b.slug for b in WATCHLIST)

    def test_the_list_is_non_trivial(self) -> None:
        assert len(WATCHLIST) >= 20


class TestToFetchers:
    def test_produces_one_callable_per_board(self) -> None:
        boards = [Board("greenhouse", "acme", "Acme"), Board("ashby", "beta", "Beta")]

        fetchers = to_fetchers(boards)

        assert len(fetchers) == 2
        assert all(callable(f) for f in fetchers)

    def test_binds_each_board_independently(self) -> None:
        """Late-binding closures are a classic loop bug: every fetcher must keep its OWN slug."""
        boards = [Board("ashby", "one", "One"), Board("ashby", "two", "Two")]

        fetchers = to_fetchers(boards)

        # partial keeps the bound slug as the first positional arg.
        assert fetchers[0].args[0] == "one"  # type: ignore[attr-defined]
        assert fetchers[1].args[0] == "two"  # type: ignore[attr-defined]

    def test_greenhouse_is_not_passed_a_company_kwarg(self) -> None:
        """Greenhouse's fetch_jobs takes no `company` (its API already names the employer); passing
        it raises TypeError and silently killed every Greenhouse board in the first real run."""
        fetcher = Board("greenhouse", "nex", "Nex").to_fetcher()

        assert fetcher.args == ("nex",)  # type: ignore[attr-defined]
        assert "company" not in fetcher.keywords  # type: ignore[attr-defined]

    def test_lever_and_ashby_do_get_the_company(self) -> None:
        for source in ("lever", "ashby"):
            fetcher = Board(source, "acme", "Acme Inc").to_fetcher()

            assert fetcher.keywords["company"] == "Acme Inc"  # type: ignore[attr-defined]

    def test_defaults_to_the_full_watchlist(self) -> None:
        assert len(to_fetchers()) == len(WATCHLIST)


class TestRemotiveSources:
    def test_one_fetcher_per_search(self) -> None:
        fetchers = remotive_fetchers(["data engineer", "AI engineer"])

        assert len(fetchers) == 2
        assert all(callable(f) for f in fetchers)

    def test_binds_the_search_term(self) -> None:
        fetcher = remotive_fetchers(["data platform"])[0]

        assert fetcher.args[0] == "data platform"  # type: ignore[attr-defined]

    def test_default_searches_cover_his_lane(self) -> None:
        assert "data engineer" in REMOTIVE_SEARCHES
        assert len(remotive_fetchers()) == len(REMOTIVE_SEARCHES)


class TestAdzunaSources:
    def test_queries_only_target_his_workable_metros(self) -> None:
        """Adzuna is LOCAL search — it must only hunt places he can work, never a random US city."""
        wheres = {q.where for q in ADZUNA_QUERIES}
        assert wheres == {"Sarasota", "Chicago"}

    def test_binds_phrase_where_and_distance(self) -> None:
        fetcher = AdzunaQuery("data engineer", "Sarasota", 60).to_fetcher()

        assert fetcher.args[0] == "data engineer"  # type: ignore[attr-defined]
        assert fetcher.keywords["where"] == "Sarasota"  # type: ignore[attr-defined]
        assert fetcher.keywords["distance"] == 60  # type: ignore[attr-defined]

    def test_one_fetcher_per_query(self) -> None:
        assert len(adzuna_fetchers()) == len(ADZUNA_QUERIES)


class TestWorkdaySources:
    def test_one_fetcher_per_board_and_search(self) -> None:
        # 3 gov/defense boards x the search terms.
        assert len(workday_fetchers()) == 3 * len(WORKDAY_SEARCHES)

    def test_binds_the_board_and_search(self) -> None:
        fetcher = workday_fetchers()[0]

        assert fetcher.args[0].tenant == "gdit"  # type: ignore[attr-defined]
        assert fetcher.args[1] in WORKDAY_SEARCHES  # type: ignore[attr-defined]


class TestUsaJobsSources:
    def test_scopes_are_his_metros_plus_remote(self) -> None:
        """Federal search targeted only where he can work: Chicago, Tampa, or remote-only."""
        for q in USAJOBS_QUERIES:
            assert q.location_name in ("Chicago, Illinois", "Tampa, Florida") or q.remote_only

    def test_binds_keyword_and_scope(self) -> None:
        fetcher = UsaJobsQuery("data engineer", "Tampa, Florida", 60, False).to_fetcher()

        assert fetcher.args[0] == "data engineer"  # type: ignore[attr-defined]
        assert fetcher.keywords["location_name"] == "Tampa, Florida"  # type: ignore[attr-defined]

    def test_one_fetcher_per_query(self) -> None:
        assert len(usajobs_fetchers()) == len(USAJOBS_QUERIES)


class TestAllSources:
    def test_combines_every_source_family(self) -> None:
        assert len(all_sources()) == (
            len(WATCHLIST)
            + len(REMOTIVE_SEARCHES)
            + len(ADZUNA_QUERIES)
            + len(workday_fetchers())
            + len(USAJOBS_QUERIES)
            + len(breezy_fetchers())
        )
