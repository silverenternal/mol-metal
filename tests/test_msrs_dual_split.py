"""Tests for the FIX-E1 dual-split gate (fragment_grouped + canonical_grouped)."""

from __future__ import annotations

import json

import pytest

from molmetal.scripts.msrs_dual_split import (
    DUAL_SPLIT_ARTIFACTS,
    REQUIRED_SPLITS,
    SPLIT_CANONICAL_GROUPED,
    SPLIT_FRAGMENT_GROUPED,
    DualSplitError,
    check_directory,
    normalize_splits,
    requires_dual_split,
    validate_payload,
    write_report,
)


def _report(**overrides):
    payload = {
        "artifact": "msrs_complex_features",
        "splits": {
            SPLIT_FRAGMENT_GROUPED: {"complex": {"r2": 0.4368}, "main": {"r2": 0.3366}},
            SPLIT_CANONICAL_GROUPED: {"complex": {"r2": 0.5012}, "main": {"r2": 0.3972}},
        },
    }
    payload.update(overrides)
    return payload


def test_required_splits_are_the_two_groupings():
    assert REQUIRED_SPLITS == ("fragment_grouped", "canonical_grouped")


def test_valid_report_has_no_problems():
    assert validate_payload(_report()) == []


def test_legacy_keys_are_renamed_on_read():
    payload = {
        "artifact": "msrs_complex_features",
        "splits": {
            "fragment": {"a": {"r2": 0.1}},
            "canonical": {"a": {"r2": 0.2}},
        },
    }
    assert normalize_splits(payload) == {
        SPLIT_FRAGMENT_GROUPED: {"a": {"r2": 0.1}},
        SPLIT_CANONICAL_GROUPED: {"a": {"r2": 0.2}},
    }
    assert validate_payload(payload) == []


def test_single_split_report_is_rejected():
    payload = _report()
    del payload["splits"][SPLIT_CANONICAL_GROUPED]
    problems = validate_payload(payload)
    assert len(problems) == 1
    assert "canonical_grouped" in problems[0]
    assert "not interpretable" in problems[0]


def test_empty_split_table_is_rejected():
    payload = _report(
        splits={
            SPLIT_FRAGMENT_GROUPED: {"a": {"r2": 0.1}},
            SPLIT_CANONICAL_GROUPED: {},
        }
    )
    assert any("no entries" in p for p in validate_payload(payload))


def test_protocol_report_shape_is_accepted():
    """Decision-layer reports carry metrics, not arms; both splits still required."""
    payload = {
        "artifact": "msrs_decision_policy_v2",
        "splits": {
            SPLIT_FRAGMENT_GROUPED: {"n_cohorts": 424, "full_cohort_regret": {"a": 0.1}},
            SPLIT_CANONICAL_GROUPED: {"n_cohorts": 424, "full_cohort_regret": {"a": 0.1}},
        },
    }
    assert validate_payload(payload) == []
    del payload["splits"][SPLIT_CANONICAL_GROUPED]
    assert validate_payload(payload)


def test_arm_sets_must_match_across_splits():
    payload = _report(
        splits={
            SPLIT_FRAGMENT_GROUPED: {"a": {"r2": 0.1}, "b": {"r2": 0.2}},
            SPLIT_CANONICAL_GROUPED: {"a": {"r2": 0.1}},
        }
    )
    problems = validate_payload(payload)
    assert any("arm sets differ" in p for p in problems)


def test_missing_r2_is_rejected():
    payload = _report(
        splits={
            SPLIT_FRAGMENT_GROUPED: {"a": {"mae": 0.1}},
            SPLIT_CANONICAL_GROUPED: {"a": {"r2": 0.2}},
        }
    )
    assert any("no 'r2'" in p for p in validate_payload(payload))


def test_top_level_split_tables_are_accepted():
    payload = {
        "artifact": "msrs_metal_block",
        SPLIT_FRAGMENT_GROUPED: {"a": {"r2": 0.1}},
        SPLIT_CANONICAL_GROUPED: {"a": {"r2": 0.2}},
    }
    assert validate_payload(payload) == []


def test_write_report_refuses_incomplete_payload(tmp_path):
    payload = _report()
    del payload["splits"][SPLIT_FRAGMENT_GROUPED]
    target = tmp_path / "arm.json"
    with pytest.raises(DualSplitError):
        write_report(target, payload)
    assert not target.exists()


def test_write_report_writes_complete_payload(tmp_path):
    target = tmp_path / "arm.json"
    write_report(target, _report())
    assert json.loads(target.read_text())["splits"][SPLIT_CANONICAL_GROUPED]["complex"][
        "r2"
    ] == pytest.approx(0.5012)


def test_registry_artifacts_are_in_scope_even_without_flag():
    payload = {"artifact": "msrs_complex_features"}
    assert requires_dual_split(payload) is True
    assert "msrs_complex_features" in DUAL_SPLIT_ARTIFACTS


def test_unlisted_artifact_is_out_of_scope():
    assert requires_dual_split({"artifact": "msrs_decision_policy"}) is False
    assert requires_dual_split({"artifact": "x", "requires_dual_split": True}) is True


def test_check_directory_reports_missing_split(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps(_report()))
    bad = _report()
    del bad["splits"][SPLIT_CANONICAL_GROUPED]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    (tmp_path / "out_of_scope.json").write_text(json.dumps({"artifact": "other"}))
    (tmp_path / "broken.json").write_text("{not json")

    checked, passed, failed = check_directory(tmp_path)
    assert checked == 2
    assert passed == ["good.json"]
    assert len(failed) == 1 and "bad.json" in failed[0]
