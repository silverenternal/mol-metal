"""Regression tests for the tolerance-based metric diff (Paper-A repro).

``scripts/reproduce_paper_a.py`` decides PASS/FAIL by diffing a regenerated
artifact against the on-disk one. That comparator must be tolerant of float
noise (thread-nondeterministic scorers) yet strict about structure, types and
large value drift — otherwise the reproducibility claim is either vacuous (too
loose) or permanently red (too strict). These tests pin both edges. CPU-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._metric_diff import structural_diff


def test_identical_objects_match():
    obj = {"a": 1, "b": [1.0, 2.0], "c": {"d": "x"}}
    assert structural_diff(obj, obj) == []


def test_float_within_abs_tol_matches():
    assert structural_diff({"auc": 0.8362931423415294}, {"auc": 0.8362931423415296}) == []


def test_float_beyond_abs_tol_differs():
    d = structural_diff({"auc": 0.81}, {"auc": 0.82})
    assert len(d) == 1 and d[0].startswith("!auc")


def test_rel_tol_scales_with_magnitude():
    # abs_tol 0 but rel_tol 1e-3: a 1e-4 relative change on a large number passes.
    assert structural_diff({"n": 1e6}, {"n": 1e6 * (1 + 1e-4)},
                           abs_tol=0.0, rel_tol=1e-3) == []
    assert structural_diff({"n": 1e6}, {"n": 1e6 * (1 + 1e-2)},
                           abs_tol=0.0, rel_tol=1e-3) != []


def test_bool_is_not_int():
    assert structural_diff({"flag": True}, {"flag": 1}) != []
    assert structural_diff({"flag": False}, {"flag": 0}) != []


def test_missing_and_extra_keys_are_reported():
    assert structural_diff({"a": 1}, {"a": 1, "b": 2})[0].startswith("+b")
    assert structural_diff({"a": 1, "b": 2}, {"a": 1})[0].startswith("-b")


def test_list_length_mismatch():
    d = structural_diff({"v": [1, 2, 3]}, {"v": [1, 2]})
    assert len(d) == 1 and d[0].startswith("~v")


def test_type_mismatch():
    assert structural_diff({"a": 1}, {"a": "1"}) != []


def test_ignore_path_suppresses_subtree():
    expected = {"verdict": {"why": "old"}, "auc": 0.5}
    actual = {"verdict": {"why": "new"}, "auc": 0.5}
    assert structural_diff(expected, actual, ignore=["verdict"]) == []
    assert structural_diff(expected, actual, ignore=["verdict.why"]) == []
    assert structural_diff(expected, actual, ignore=["other"]) != []


def test_ignore_does_not_leak_across_sibling_keys():
    # "verdict" must not silence "verdict_extra".
    d = structural_diff({"verdict_extra": 1}, {"verdict_extra": 2}, ignore=["verdict"])
    assert len(d) == 1
