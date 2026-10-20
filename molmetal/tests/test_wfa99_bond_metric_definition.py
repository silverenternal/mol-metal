# SPDX-License-Identifier: Apache-2.0
"""Wf-A99 (2026-10-20) — ``bond_pr_auc`` must say WHICH quantity it is.

Wf-A97 corrected the trainer's ``bond_pr_auc`` from the NO_BOND one-vs-rest
AP (majority class, 0.62) to the existence AP (0.2659 on the same 64 val
clouds).  The name did not change, and no artefact recorded the definition,
so three consumers that read ``bond_pr_auc`` out of run directories written
across both eras could put the two numbers in one column and read the
difference as a regression.

This module tests the stamp and the guard that make that impossible.
"""
from __future__ import annotations

import torch

from molmetal.eval.bond_metric_schema import (
    BOND_AP_EXISTENCE,
    BOND_AP_NO_BOND_LEGACY,
    BOND_AP_UNDECLARED,
    MixedBondApDefinition,
    bond_ap_definition,
    bond_ap_legend,
    group_by_definition,
    require_single_definition,
    stamp_counts,
)


def _trainer():
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    return T


def _probs(rows):
    return [torch.tensor(rows, dtype=torch.float32)]


# --- the schema helper ----------------------------------------------------


def test_unstamped_row_is_undeclared_not_guessed():
    """A pre-Wf-A97 artefact has no stamp; the module must not infer one."""
    assert bond_ap_definition({"bond_pr_auc": 0.6209}) == BOND_AP_UNDECLARED
    assert bond_ap_definition({}) == BOND_AP_UNDECLARED
    assert bond_ap_definition({"bond_pr_auc_definition": ""}) == BOND_AP_UNDECLARED


def test_stamp_is_read_from_mapping_and_from_attribute_rows():
    class Row:
        bond_pr_auc_definition = BOND_AP_EXISTENCE

    assert bond_ap_definition({"bond_pr_auc_definition": BOND_AP_EXISTENCE}) == BOND_AP_EXISTENCE
    assert bond_ap_definition(Row()) == BOND_AP_EXISTENCE


def test_legend_distinguishes_the_two_definitions():
    assert "EXISTENCE" in bond_ap_legend(BOND_AP_EXISTENCE)
    assert "NO_BOND" in bond_ap_legend(BOND_AP_NO_BOND_LEGACY)
    assert "pre-Wf-A97" in bond_ap_legend(BOND_AP_UNDECLARED)
    assert "unknown definition" in bond_ap_legend("not_a_definition")


def test_single_definition_passes_through():
    rows = [{"run": "a", "bond_pr_auc_definition": BOND_AP_EXISTENCE}] * 3
    assert require_single_definition(rows) == BOND_AP_EXISTENCE


def test_mixing_definitions_raises_and_names_the_rows():
    rows = [
        {"run": "cfm_r9_fix_v2", "bond_pr_auc_definition": BOND_AP_EXISTENCE},
        {"run": "cfm_fm_control_2026_10_09", "bond_pr_auc_definition": BOND_AP_UNDECLARED},
    ]
    try:
        require_single_definition(rows, context="census table")
    except MixedBondApDefinition as exc:
        msg = str(exc)
        assert "2 definitions" in msg
        assert "census table" in msg
        assert "cfm_fm_control_2026_10_09" in msg
    else:  # pragma: no cover - the raise IS the contract
        raise AssertionError("mixing two definitions must raise")


def test_grouping_and_counts_preserve_rows():
    rows = [
        {"run": "new", "bond_pr_auc_definition": BOND_AP_EXISTENCE},
        {"run": "old1", "bond_pr_auc_definition": BOND_AP_NO_BOND_LEGACY},
        {"run": "old2"},
    ]
    groups = group_by_definition(rows)
    assert [r["run"] for r in groups[BOND_AP_EXISTENCE]] == ["new"]
    assert [r["run"] for r in groups[BOND_AP_NO_BOND_LEGACY]] == ["old1"]
    assert [r["run"] for r in groups[BOND_AP_UNDECLARED]] == ["old2"]
    assert stamp_counts(rows) == {
        BOND_AP_EXISTENCE: 1,
        BOND_AP_NO_BOND_LEGACY: 1,
        BOND_AP_UNDECLARED: 1,
    }


# --- the trainer must stamp ----------------------------------------------


def test_legacy_report_stamps_the_existence_definition():
    T = _trainer()
    out = T._bond_ranking_report([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert out["bond_pr_auc_definition"] == BOND_AP_EXISTENCE
    # and the empty branch stamps too, so a zero is still attributable
    empty = T._bond_ranking_report([], [])
    assert empty["bond_pr_auc_definition"] == BOND_AP_EXISTENCE


def test_per_class_report_stamps_the_existence_definition():
    T = _trainer()
    probs = _probs([[0.9, 0.1, 0.0, 0.0, 0.0]] * 10)
    truth = [0] * 8 + [1] * 2
    out = T._bond_ranking_report_per_class(probs, truth)
    assert out["bond_pr_auc_definition"] == BOND_AP_EXISTENCE
    assert out["bond_pr_auc"] == out["bond_existence_ap"]


def test_epoch_metrics_default_is_empty_so_old_rows_read_as_undeclared():
    """The dataclass default must NOT claim a definition.

    A default of ``existence_ap_v2`` would relabel every artefact written
    before this field existed, which is the exact failure the stamp is
    meant to prevent.
    """
    import dataclasses

    T = _trainer()
    fields = {f.name: f for f in dataclasses.fields(T.EpochMetrics)}
    assert "bond_pr_auc_definition" in fields
    assert fields["bond_pr_auc_definition"].default == ""
    # and the empty stamp reads as undeclared, not as either definition
    class Row:
        bond_pr_auc_definition = ""

    assert bond_ap_definition(Row()) == BOND_AP_UNDECLARED
