"""The gate in front of the gold demand table.

`check_gold` returns a list of failures. The notebook raises if the list is not empty, so a bad
run leaves the previous table unchanged instead of writing a wrong one.

The total check: every source row lands either in a group (`postings`) or in `no_week`.
sum(postings) + no_week must equal the number of rows in the source. Ingest already removed
cross-source duplicates, so one source row is one posting.
"""

from __future__ import annotations

from datetime import date

import pytest

from jobradar.gold_checks import check_gold

W1 = date(2026, 8, 3)
W2 = date(2026, 8, 10)


def row(
    week: date = W1,
    role_type: str = "data_engineer",
    work_mode: str = "remote",
    postings: int | None = 5,
    companies: int | None = 3,
) -> dict:
    return {
        "week_start": week,
        "role_type": role_type,
        "work_mode": work_mode,
        "postings": postings,
        "companies": companies,
    }


GOOD = [
    row(W1, "data_engineer", "remote", 5, 3),
    row(W1, "data_engineer", "not_remote", 4, 4),
    row(W2, "other", "remote", 1, 1),
]
GOOD_TOTAL = 10


class TestPasses:
    def test_good_rows_pass(self) -> None:
        assert check_gold(GOOD, source_rows=GOOD_TOTAL, no_week=0) == []

    def test_rows_with_no_week_count_toward_the_total(self) -> None:
        assert check_gold(GOOD, source_rows=GOOD_TOTAL + 2, no_week=2) == []


class TestFails:
    def test_empty(self) -> None:
        failures = check_gold([], source_rows=0, no_week=0)
        assert any("empty" in f for f in failures)

    def test_duplicate_grain(self) -> None:
        rows = [*GOOD, row(W1, "data_engineer", "remote", 1, 1)]
        failures = check_gold(rows, source_rows=GOOD_TOTAL + 1, no_week=0)
        assert any("grain" in f for f in failures)

    def test_unknown_role_type(self) -> None:
        rows = [*GOOD[:2], row(W2, "astronaut", "remote", 1, 1)]
        failures = check_gold(rows, source_rows=GOOD_TOTAL, no_week=0)
        assert any("role_type" in f and "astronaut" in f for f in failures)

    def test_unknown_work_mode(self) -> None:
        rows = [*GOOD[:2], row(W2, "other", "hybrid", 1, 1)]
        failures = check_gold(rows, source_rows=GOOD_TOTAL, no_week=0)
        assert any("work_mode" in f and "hybrid" in f for f in failures)

    @pytest.mark.parametrize("bad", [0, -1, None])
    def test_postings_not_positive(self, bad: int | None) -> None:
        rows = [*GOOD, row(W2, "data_analyst", "remote", bad, 1)]
        failures = check_gold(rows, source_rows=GOOD_TOTAL, no_week=0)
        assert any("postings" in f for f in failures)

    @pytest.mark.parametrize("bad", [0, -1, None])
    def test_companies_not_positive(self, bad: int | None) -> None:
        rows = [*GOOD, row(W2, "data_analyst", "remote", 1, bad)]
        failures = check_gold(rows, source_rows=GOOD_TOTAL + 1, no_week=0)
        assert any("companies" in f for f in failures)

    def test_total_short_of_source(self) -> None:
        failures = check_gold(GOOD, source_rows=GOOD_TOTAL + 2, no_week=0)
        assert any("total" in f for f in failures)

    def test_total_over_source(self) -> None:
        failures = check_gold(GOOD, source_rows=GOOD_TOTAL, no_week=3)
        assert any("total" in f for f in failures)

    def test_reports_every_failure_not_just_the_first(self) -> None:
        rows = [row(W1, "astronaut", "hybrid", 0, 0)]
        failures = check_gold(rows, source_rows=99, no_week=0)
        assert len(failures) >= 4
