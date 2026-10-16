"""Wf-A61 (2026-10-17) L19 — ``molmetal.eval.scoring.sascorer_aggregator``
provides a single canonical ``try_load`` entry point that absorbs the
three historical import patterns for ``sascorer``.
"""

from __future__ import annotations

from molmetal.eval.scoring.sascorer_aggregator import compute_sa, try_load


def test_try_load_returns_callable_or_none() -> None:
    """L19 — returns a callable OR None on a host where RDKit Contrib is missing."""
    fn = try_load()
    assert fn is None or callable(fn)


def test_compute_sa_none_input_returns_none() -> None:
    """L19 — ``compute_sa(None)`` is always None (no crash)."""
    assert compute_sa(None) is None