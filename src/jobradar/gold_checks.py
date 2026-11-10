"""The gate in front of the gold demand table.

`check_gold` returns every failure it finds, not just the first. The notebook raises when the
list is not empty, so a bad run leaves the previous table in place instead of writing a wrong one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from jobradar.role_type import ROLE_TYPES, WORK_MODES

GRAIN: tuple[str, ...] = ("week_start", "role_type", "work_mode")
"""One row per week, role type and work mode."""


def _positive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def check_gold(rows: Sequence[Mapping[str, Any]], source_rows: int, no_week: int) -> list[str]:
    """Check the gold rows before they are written. An empty list means safe to write.

    The total check: every source row lands either in a group (`postings`) or in `no_week`,
    so sum(postings) + no_week must equal `source_rows`. Ingest already removed cross-source
    duplicates, so one source row is one posting.
    """
    failures: list[str] = []

    if not rows:
        failures.append("empty: the gold table has no rows")

    keys = Counter(tuple(r.get(c) for c in GRAIN) for r in rows)
    dupes = [k for k, n in keys.items() if n > 1]
    if dupes:
        failures.append(f"grain: {len(dupes)} duplicate {GRAIN} keys, e.g. {dupes[0]}")

    bad_roles = sorted({str(r.get("role_type")) for r in rows} - set(ROLE_TYPES))
    if bad_roles:
        failures.append(f"role_type: values not allowed {bad_roles}")

    bad_modes = sorted({str(r.get("work_mode")) for r in rows} - set(WORK_MODES))
    if bad_modes:
        failures.append(f"work_mode: values not allowed {bad_modes}")

    for column in ("postings", "companies"):
        bad = sum(1 for r in rows if not _positive(r.get(column)))
        if bad:
            failures.append(f"{column}: {bad} rows with a zero, negative or missing count")

    total = sum(r["postings"] for r in rows if isinstance(r.get("postings"), int))
    if total + no_week != source_rows:
        failures.append(
            f"total: sum(postings) {total} + no_week {no_week} = {total + no_week}, "
            f"but the source has {source_rows} rows"
        )

    return failures
