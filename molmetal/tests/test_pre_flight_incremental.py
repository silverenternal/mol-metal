"""Unit tests for pre_flight_check --changed-only incremental gating.

Pure-function tests of ``select_changed_gates`` + ``GATE_PATH_SCOPES`` — they
never execute a gate, so they run in milliseconds.

The safety invariants these pin:
  * an empty changed-file set runs ALL gates (never zero);
  * a gate absent from GATE_PATH_SCOPES ALWAYS runs;
  * a scoped gate runs when a changed file matches its glob, skips otherwise;
  * every key in GATE_PATH_SCOPES names a registered gate (no dead scopes).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import pre_flight_check as pf  # noqa: E402


def _fake_runners():
    return [("scoped_gate", lambda: None), ("global_gate", lambda: None)]


def test_scope_map_keys_are_all_registered() -> None:
    """Every scoped gate name must exist in build_gates('all')."""
    names = {n for n, _ in pf.build_gates("all")}
    missing = sorted(n for n in pf.GATE_PATH_SCOPES if n not in names)
    assert missing == [], f"GATE_PATH_SCOPES names no registered gate: {missing}"


def test_empty_changed_runs_all() -> None:
    """An empty changed-file list must NOT drop any gate."""
    runners_all = pf.build_gates("all")
    kept, skipped, _ = pf.select_changed_gates(runners_all, [])
    assert len(kept) == len(runners_all)
    assert skipped == []


def test_unscoped_gate_always_runs() -> None:
    """A gate absent from the map runs regardless of the changed-file set."""
    map_backup = dict(pf.GATE_PATH_SCOPES)
    pf.GATE_PATH_SCOPES.clear()
    pf.GATE_PATH_SCOPES.update({})
    try:
        kept, skipped, _ = pf.select_changed_gates(_fake_runners(), ["README.md"])
        assert [n for n, _ in kept] == ["scoped_gate", "global_gate"]
        assert skipped == []
    finally:
        pf.GATE_PATH_SCOPES.clear()
        pf.GATE_PATH_SCOPES.update(map_backup)


def test_scoped_gate_skips_on_unrelated_change() -> None:
    """A scoped gate whose globs are not hit is skipped; the unscoped one runs."""
    name = "p3_qed_score_metric_present"
    globs = pf.GATE_PATH_SCOPES[name]
    map_backup = dict(pf.GATE_PATH_SCOPES)
    pf.GATE_PATH_SCOPES.clear()
    # 'global_gate' is intentionally ABSENT from the map → must always run.
    pf.GATE_PATH_SCOPES.update({name: globs})
    runners = [(name, lambda: None), ("global_gate", lambda: None)]
    try:
        kept, skipped, _ = pf.select_changed_gates(runners, ["TODO/some_doc.md"])
        assert [n for n, _ in kept] == ["global_gate"]
        assert skipped == [name]
    finally:
        pf.GATE_PATH_SCOPES.clear()
        pf.GATE_PATH_SCOPES.update(map_backup)


def test_scoped_gate_runs_on_matching_change() -> None:
    """A scoped gate runs when a changed file matches one of its globs."""
    name = "p3_qed_score_metric_present"
    globs = pf.GATE_PATH_SCOPES[name]
    map_backup = dict(pf.GATE_PATH_SCOPES)
    pf.GATE_PATH_SCOPES.clear()
    pf.GATE_PATH_SCOPES.update({name: globs})
    try:
        kept, skipped, _ = pf.select_changed_gates(
            [("p3_qed_score_metric_present", lambda: None)],
            ["metrics/by_metric/qed_score.json"],
        )
        assert [n for n, _ in kept] == ["p3_qed_score_metric_present"]
        assert skipped == []
    finally:
        pf.GATE_PATH_SCOPES.clear()
        pf.GATE_PATH_SCOPES.update(map_backup)


def test_doc_only_change_is_strictly_smaller() -> None:
    """A TODO/*.md change runs fewer gates than the full set (dev-throughput
    win) but still runs the globals and never runs zero."""
    runners_all = pf.build_gates("all")
    kept, skipped, _ = pf.select_changed_gates(runners_all, ["TODO/x.md"])
    assert 0 < len(kept) < len(runners_all)
    # Every global gate (absent from the map) must survive.
    globals_ = [n for n, _ in runners_all if n not in pf.GATE_PATH_SCOPES]
    assert all(g in {n for n, _ in kept} for g in globals_)


def test_no_gitignored_scope() -> None:
    """No scoped gate may read a GITIGNORED path.

    ``--changed-only`` derives its changed set from ``git diff --cached
    --name-only``, which cannot report a gitignored file. A glob over a
    gitignored path therefore never matches, so such a gate is skipped
    UNCONDITIONALLY — even when its frozen data was tampered with. That turns a
    failing commit green, which the conservatism contract forbids.
    """
    offenders = {
        name: globs
        for name, globs in pf.GATE_PATH_SCOPES.items()
        if any(str(g).startswith(".data/") for g in globs)
    }
    assert offenders == {}, (
        "gitignored .data/ scopes are never matchable by `git diff --cached` "
        f"and are unconditionally skipped: {offenders}"
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
