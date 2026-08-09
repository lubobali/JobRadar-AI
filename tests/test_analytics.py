"""The CDF analytics surface: the repository read and the Insights page.

Capstone requirement 6. The pipeline itself lives in
`notebooks/cdf_analytics.py` and runs on Spark against Delta, so what is
testable here is the contract either side of it: what the repository asks
Postgres for, and what the page does with the answer - including when the
answer is "the run has never happened".
"""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from flask.testing import FlaskClient

sys.path.insert(0, "app")

app_module = importlib.import_module("app")

from jobradar import repository  # noqa: E402

ROWS = [
    {"day": date(2026, 8, 9), "metric": "status_transitions", "dimension": "screening",
     "value": 3, "computed_at": datetime(2026, 8, 9, 20, tzinfo=UTC)},
    {"day": date(2026, 8, 9), "metric": "status_transitions", "dimension": "applied",
     "value": 1, "computed_at": datetime(2026, 8, 9, 20, tzinfo=UTC)},
    {"day": date(2026, 8, 8), "metric": "status_transitions", "dimension": "screening",
     "value": 2, "computed_at": datetime(2026, 8, 9, 20, tzinfo=UTC)},
    {"day": date(2026, 8, 9), "metric": "pipeline_now", "dimension": "applied",
     "value": 7, "computed_at": datetime(2026, 8, 9, 20, tzinfo=UTC)},
    {"day": date(2026, 8, 9), "metric": "jobs_saved", "dimension": None,
     "value": 4, "computed_at": datetime(2026, 8, 9, 20, tzinfo=UTC)},
]


class TestRepositoryRead:
    def test_the_window_is_passed_as_days(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict = {}

        def fake(sql: str, params: tuple) -> list:
            seen["sql"], seen["params"] = sql, params
            return []

        monkeypatch.setattr(repository.lakebase, "run_query", fake)
        repository.analytics(days=14)
        assert seen["params"] == (14,)
        assert "make_interval(days => %s)" in seen["sql"]

    def test_a_metric_filter_is_parameterised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Never interpolated. This value reaches SQL from a URL."""
        seen: dict = {}
        monkeypatch.setattr(
            repository.lakebase,
            "run_query",
            lambda sql, params: seen.update(sql=sql, params=params) or [],
        )
        repository.analytics(metric="jobs_saved", days=30)
        assert seen["params"] == (30, "jobs_saved")
        assert "jobs_saved" not in seen["sql"]

    def test_no_run_yet_reads_as_none_not_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing run and a run that found nothing are different facts.

        The page says so; conflating them would render an empty dashboard as if
        it were a real answer.
        """
        monkeypatch.setattr(repository.lakebase, "run_query_one", lambda *a, **k: {"at": None})
        assert repository.analytics_computed_at() is None


@pytest.fixture
def client() -> FlaskClient:
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


class TestInsightsPage:
    def test_it_renders_the_metrics(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_module.repository, "analytics", lambda days=90: ROWS)
        monkeypatch.setattr(
            app_module.repository,
            "analytics_computed_at",
            lambda: datetime(2026, 8, 9, 20, tzinfo=UTC),
        )
        html = client.get("/insights").get_data(as_text=True)
        assert "Status transitions" in html
        assert "Pipeline now" in html
        assert "screening" in html

    def test_days_are_summed_per_dimension(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """screening appears on two days, 3 and 2, and should read as 5.

        A list of per-day rows is not an answer to "where is everything".
        """
        monkeypatch.setattr(app_module.repository, "analytics", lambda days=90: ROWS)
        monkeypatch.setattr(
            app_module.repository,
            "analytics_computed_at",
            lambda: datetime(2026, 8, 9, 20, tzinfo=UTC),
        )
        html = client.get("/insights").get_data(as_text=True)
        assert ">5<" in html

    def test_it_says_so_when_the_run_never_happened(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty dashboard must not look like a dashboard reading zero."""
        monkeypatch.setattr(app_module.repository, "analytics", lambda days=90: [])
        monkeypatch.setattr(app_module.repository, "analytics_computed_at", lambda: None)
        html = client.get("/insights").get_data(as_text=True)
        assert "No analytics yet" in html
        assert "cdf_analytics.py" in html

    def test_the_page_names_its_source(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every panel states which change type it came from.

        A number with no provenance is indistinguishable from one queried
        straight out of the operational table, which is exactly what this
        requirement is not.
        """
        monkeypatch.setattr(app_module.repository, "analytics", lambda days=90: ROWS)
        monkeypatch.setattr(
            app_module.repository,
            "analytics_computed_at",
            lambda: datetime(2026, 8, 9, 20, tzinfo=UTC),
        )
        html = client.get("/insights").get_data(as_text=True)
        assert "CDF update_postimage" in html
        assert "table_changes()" in html

    def test_insights_is_in_the_nav(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_module.repository, "analytics", lambda days=90: [])
        monkeypatch.setattr(app_module.repository, "analytics_computed_at", lambda: None)
        assert 'href="/insights"' in client.get("/insights").get_data(as_text=True)


class TestNotebookContract:
    """The notebook is not importable here, so its guarantees are pinned as text.

    Cheap, and it catches the two edits that would silently break requirement 6:
    losing the CDF property, or swapping the MERGE for an overwrite.
    """

    @staticmethod
    def _source() -> str:
        return Path("notebooks/cdf_analytics.py").read_text()

    def test_cdf_is_enabled_on_the_mirrors(self) -> None:
        assert 'option("delta.enableChangeDataFeed", "true")' in self._source()

    def test_the_feed_is_actually_read(self) -> None:
        assert 'option("readChangeFeed", "true")' in self._source()

    def test_it_merges_rather_than_overwrites_the_mirrors(self) -> None:
        """An overwrite rewrites every row, so the feed would report the whole
        table as changed on every run and the transition counts would be noise."""
        source = self._source()
        assert ".merge(" in source
        assert ".whenMatchedUpdateAll()" in source

    def test_every_change_type_it_relies_on_is_used(self) -> None:
        source = self._source()
        for change_type in ("insert", "update_postimage", "delete"):
            assert f'"{change_type}"' in source
