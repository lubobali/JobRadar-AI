"""A guard against tests that VANISH instead of failing.

`@respx.mock` applied to a class returns a *function*, not a class. pytest only collects
classes named `Test*`, so it saw a function, skipped it, and said nothing — no error, no skip
marker. ~50 HTTP-layer tests across all five fetchers never ran for two days while the suite
reported green. When they were finally run, 13 were failing.

A rising pass count proved nothing: a test that disappears looks exactly like a test that
passes. This file makes that failure mode impossible rather than trusting anyone to remember.
(PLAN.md Decision Log #9.)
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

import pytest
import respx

TESTS_ROOT = Path(__file__).parent
TEST_FILES = sorted(TESTS_ROOT.rglob("test_*.py"))


def is_silently_uncollectable(name: str, obj: object) -> bool:
    """Report whether `name`/`obj` looks like a test class that pytest will silently skip.

    pytest collects only *classes* named `Test*`. Anything else with that name — typically a
    class a decorator quietly replaced with a function — is dropped without a word.

    >>> class TestThing: pass
    >>> is_silently_uncollectable("TestThing", TestThing)
    False
    >>> is_silently_uncollectable("TestThing", lambda: None)   # what @respx.mock leaves behind
    True
    >>> is_silently_uncollectable("FETCHED_AT", 42)            # not a test name; irrelevant
    False
    """
    if not name.startswith("Test"):
        return False
    return not inspect.isclass(obj)


def _load(path: Path) -> ModuleType:
    """Import a test module by path, reusing pytest's own import if it already did."""
    if path.stem in sys.modules:
        return sys.modules[path.stem]
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_there_are_test_files_to_guard() -> None:
    """If this file ever finds nothing, the guard itself has silently stopped working."""
    assert len(TEST_FILES) >= 2


@pytest.mark.parametrize("path", TEST_FILES, ids=lambda p: p.name)
def test_no_test_class_was_replaced_by_a_decorator(path: Path) -> None:
    """Every module-level `Test*` must still be a class by the time pytest collects it."""
    module = _load(path)

    broken = [
        f"{path.name}::{name} is a {type(obj).__name__}, not a class — pytest will SKIP it "
        f"silently. A class decorator (e.g. @respx.mock) probably replaced it; use the "
        f"respx_mock fixture on each test method instead."
        for name, obj in vars(module).items()
        if is_silently_uncollectable(name, obj)
    ]

    assert not broken, "\n".join(broken)


class TestGuardActuallyCatchesTheRealBug:
    """Proof the guard works, using the real decorator that caused the outage."""

    def test_catches_a_class_decorated_with_respx_mock(self) -> None:
        @respx.mock
        class TestWouldSilentlyVanish:
            def test_never_runs(self) -> None: ...

        assert is_silently_uncollectable("TestWouldSilentlyVanish", TestWouldSilentlyVanish)

    def test_accepts_a_class_using_the_respx_mock_fixture(self) -> None:
        class TestUsesFixture:
            def test_runs(self, respx_mock: object) -> None: ...

        assert not is_silently_uncollectable("TestUsesFixture", TestUsesFixture)
