# SPDX-License-Identifier: Apache-2.0
"""T41 — the summary's ``best`` block must describe ONE epoch.

THE DEFECT
----------
``best.epoch`` is the epoch the early-stop gate selected.  The five headline
bond numbers written beside it were read from ``_bond_acc``, the evaluator's
accumulator, which after the training loop holds the LAST evaluated epoch.
On a 3-epoch run those coincide; on a 30-epoch run whose bond head peaked at
epoch 3 the artifact reported epoch 0's gate next to epoch 29's bond scores.

MEASURED: 87 of the 90 archived artifacts that carry both a summary JSON and
its per-epoch CSV disagree with their own CSV about which epoch the bond
numbers came from.

WHY THE RULE IS A MODULE-LEVEL FUNCTION
---------------------------------------
The block lives inside a 3000-line ``main()``; asserting on it end-to-end
would need a trained model.  ``resolve_best_block_bond_source`` exists so the
rule — "the fields come from the row whose epoch equals ``best.epoch``" — can
be tested against synthetic rows.  Same reason ``resolve_selection`` is
module-level.

WHAT THIS FILE DOES NOT CLAIM
-----------------------------
That the bond numbers are good.  Only that the ones in ``best`` and the epoch
labelled ``best.epoch`` refer to the same point in the run, or that the block
says out loud that they do not.

The second half of the file covers the related ``combined`` gate fix: a
product with a pinned factor is not a combination, it is the other factor
wearing the word "combined".
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_trainer():
    """Import the trainer module by path — it is a script, not a package API."""
    path = REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    spec = importlib.util.spec_from_file_location("cfm_trainer_t41", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class _Row:
    """The subset of EpochMetrics the rule reads."""

    epoch: int
    bond_precision: float
    bond_recall: float
    bond_f1: float
    bond_exact_type_rate: float
    bond_type_macro_f1_pooled: float
    bond_n_scored: int


def _rows():
    # Epoch 0 is the gate's pick and the bond head's peak; epoch 2 is the last
    # epoch and is deliberately WORSE on every bond field, so a fallback to the
    # accumulator is visible as a drop rather than a coincidence.
    return [
        _Row(0, 0.90, 0.80, 0.85, 0.40, 0.30, 192),
        _Row(1, 0.70, 0.60, 0.65, 0.20, 0.15, 192),
        _Row(2, 0.50, 0.40, 0.44, 0.10, 0.08, 192),
    ]


def test_fields_come_from_the_selected_epoch_not_the_last():
    resolve = _load_trainer().resolve_best_block_bond_source
    fields, epoch_of_fields = resolve(_rows(), 0, {"bond_f1": 0.44, "n_scored": 192})
    assert epoch_of_fields == 0
    assert fields["bond_f1"] == pytest.approx(0.85), (
        "best.bond_f1 came from the last epoch (0.44), not from the selected "
        "epoch (0.85) — the epochs_mixed defect is back"
    )
    assert fields["bond_exact_type_rate"] == pytest.approx(0.40)
    assert fields["bond_n_scored"] == 192


def test_middle_epoch_selection_also_works():
    """Not just epoch 0 — the lookup must not assume the first or last row."""
    resolve = _load_trainer().resolve_best_block_bond_source
    fields, epoch_of_fields = resolve(_rows(), 1, {"bond_f1": 0.44})
    assert epoch_of_fields == 1
    assert fields["bond_f1"] == pytest.approx(0.65)


def test_missing_epoch_falls_back_and_reports_mixed():
    """A 0-epoch run has no row.  Fall back, but say the block is mixed."""
    resolve = _load_trainer().resolve_best_block_bond_source
    fields, epoch_of_fields = resolve([], -1, {"bond_f1": 0.31, "n_scored": 7})
    assert epoch_of_fields is None, (
        "a missing row must be reported as mixed, not silently labelled with "
        "best_epoch"
    )
    assert fields["bond_f1"] == pytest.approx(0.31)
    assert fields["bond_n_scored"] == 7, (
        "the fallback kept the evaluator's own `n_scored` spelling, so the "
        "reader of bond_n_scored would see 0 while 7 pairs were scored"
    )


def test_fallback_without_scored_count_reports_zero():
    resolve = _load_trainer().resolve_best_block_bond_source
    fields, epoch_of_fields = resolve([], 3, {"bond_f1": 0.0})
    assert epoch_of_fields is None and fields["bond_n_scored"] == 0


def test_plain_dict_rows_are_accepted():
    """Eval-only passes have been known to hand back mappings, not dataclasses."""
    resolve = _load_trainer().resolve_best_block_bond_source
    rows = [{"epoch": 0, "bond_f1": 0.9}, {"epoch": 1, "bond_f1": 0.1}]
    fields, epoch_of_fields = resolve(rows, 0, {})
    assert epoch_of_fields == 0 and fields["bond_f1"] == pytest.approx(0.9)


def test_best_block_emits_the_two_provenance_keys():
    """Structural guard: the block must state which epoch its fields are from.

    Without these keys a reader has no way to tell a self-consistent block
    from a mixed one, which is exactly how the original defect stayed
    invisible for months.
    """
    src = (
        REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    ).read_text(encoding="utf-8")
    assert '"bond_fields_from_epoch"' in src
    assert '"epochs_mixed"' in src
    assert '"gate_epoch"' in src
    assert '"gate_value_at_epoch"' in src
    assert "resolve_best_block_bond_source(" in src


def test_early_stop_control_says_which_epoch_the_weights_are():
    """`best.epoch` is the selected epoch; `early_stop_control.best_epoch` is
    the gate's argmax.  They differ on a degenerate run, and without these two
    keys the same JSON reads as a contradiction."""
    src = (
        REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    ).read_text(encoding="utf-8")
    assert '"selected_epoch": _best_effective_epoch' in src
    assert '"selection_overrode_gate"' in src


def test_the_helper_is_wired_to_the_SELECTED_epoch():
    """A correct helper wired to the wrong epoch is still the defect.

    This file exists because the audit kept finding rules that were right in
    isolation and wrong at the call site.  `best.epoch` must be the epoch the
    named checkpoint holds, so the lookup has to use the post-selection
    epoch, not the gate's raw argmax.
    """
    src = (
        REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    ).read_text(encoding="utf-8")
    call = re.search(
        r"resolve_best_block_bond_source\(\s*per_epoch,\s*([A-Za-z_][A-Za-z0-9_]*),",
        src,
    )
    assert call is not None, "the helper is no longer called on per_epoch"
    assert call.group(1) == "_best_effective_epoch", (
        f"the lookup is wired to {call.group(1)!r}; it must use the epoch the "
        "artifact names"
    )
    assert '"epoch": _best_effective_epoch' in src, (
        "best.epoch must be the selected epoch, not the gate's argmax"
    )


# ---------------------------------------------------------------------------
# `combined` with a pinned factor is not a combination.
#
# MEASURED on results/wfa108j_gcv36/per_epoch_metrics.json: both halves read
# 0.000 at all 36 epochs, so `combined` was a constant 0.0 while the artifact
# named two axes.  A generation half pinned at 1.000 is the mirror case: the
# product equals the bond half exactly, so the generation axis contributed
# nothing while the name claimed it did.
# ---------------------------------------------------------------------------


def _resolve_combined(**over):
    resolve = _load_trainer().resolve_early_stop_criterion
    kw = dict(
        decode_ratio=0.8,
        gen_validity=0.875,
        gen_n=48,
        bond_exact_type_rate=0.0,
        bond_n_scored=192.0,
        bond_type_macro_f1=0.3634,
    )
    kw.update(over)
    return resolve("combined", **kw)


def test_combined_keeps_its_name_when_both_halves_move():
    name, value, note = _resolve_combined()
    assert name == "combined"
    assert value == pytest.approx(0.875 * 0.3634)
    assert "bond_type_macro_f1" in note and "structurally unreachable" in note


def test_combined_reduces_when_generation_half_is_pinned_at_one():
    name, value, note = _resolve_combined(gen_validity=1.0)
    assert name == "bond_type_macro_f1", (
        "a product with gen_validity pinned at 1.000 equals the bond half "
        "exactly; calling it `combined` claims a contribution that is zero"
    )
    assert value == pytest.approx(0.3634)
    assert "pinned" in note and "generation half" in note


def test_combined_reduces_when_generation_half_is_pinned_at_zero():
    name, value, note = _resolve_combined(gen_validity=0.0)
    assert name == "bond_type_macro_f1"
    assert value == pytest.approx(0.3634)
    assert "pinned" in note


def test_combined_reduces_when_bond_half_is_pinned_at_zero():
    name, value, note = _resolve_combined(bond_type_macro_f1=None)
    assert name == "gen_validity", (
        "a bond half pinned at 0.000 makes the product 0.000 at every epoch; "
        "gating on that constant is the Wf-A108k defect"
    )
    assert value == pytest.approx(0.875)
    assert "could not rank" in note


def test_both_halves_pinned_keeps_the_product_and_its_name():
    """A constant either way — the honest answer is to keep the name and let
    resolve_selection report the degeneracy, not to rename a constant."""
    name, value, _note = _resolve_combined(gen_validity=1.0, bond_type_macro_f1=None)
    assert name == "combined" and value == 0.0


def test_reduction_never_changes_the_value_at_one():
    """The fix is a labelling fix: x1.0 is identity, so no number moves."""
    _, pinned, _ = _resolve_combined(gen_validity=1.0)
    _, moving, _ = _resolve_combined(gen_validity=0.999999)
    assert pinned == pytest.approx(0.3634)
    assert moving == pytest.approx(0.999999 * 0.3634)


# ---------------------------------------------------------------------------
# The recorded artifact the defect was measured on.
#
# Asserting arithmetic on hand-picked numbers would pass while the real run
# kept selecting epoch 0.  This replays the 36 epochs the run actually wrote.
# ---------------------------------------------------------------------------

GCV36 = "results/wfa108j_gcv36/per_epoch_metrics.json"


def _gcv36_epochs():
    p = REPO_ROOT / GCV36
    if not p.is_file():
        pytest.skip(f"recorded run not present at {p}")
    import json

    doc = json.loads(p.read_text(encoding="utf-8"))
    return doc["per_epoch_metrics"]


def test_gcv36_combined_gate_is_not_a_constant():
    """MEASURED premise + regression guard on the real 36-epoch artifact.

    Before the fix this file's `combined` resolved to bond_exact_type, which
    read 0.000 at all 36 epochs, so `_best_gate` stayed at its -inf start and
    epoch 0 (the least-trained weights) was selected.
    """
    epochs = _gcv36_epochs()
    resolve = _load_trainer().resolve_early_stop_criterion
    resolved = [
        resolve(
            "combined",
            decode_ratio=float(r["decode_ratio_rdkit_strict"]),
            gen_validity=float(r["gen_validity"]),
            gen_n=int(r["gen_validity_n"]),
            bond_exact_type_rate=float(r["bond_exact_type_rate"]),
            bond_n_scored=float(r["bond_n_scored"]),
            bond_type_macro_f1=float(r["bond_type_macro_f1_pooled"]),
        )
        for r in epochs
    ]
    names = {n for n, _, _ in resolved}
    values = [v for _, v, _ in resolved]
    assert len(set(round(v, 6) for v in values)) > 1, (
        f"`combined` resolved to the constant {values[0]!r} across all "
        f"{len(values)} epochs of {GCV36}; a constant gate cannot rank them "
        "and selects epoch 0"
    )
    assert names == {"bond_type_macro_f1"}, (
        f"expected the gate to fall through to the moving half, got {names}"
    )


def test_gcv36_episode_zero_is_no_longer_the_argmax():
    """The selection, not just the values: epoch 0 must not win."""
    epochs = _gcv36_epochs()
    resolve = _load_trainer().resolve_early_stop_criterion
    values = [
        resolve(
            "combined",
            decode_ratio=float(r["decode_ratio_rdkit_strict"]),
            gen_validity=float(r["gen_validity"]),
            gen_n=int(r["gen_validity_n"]),
            bond_exact_type_rate=float(r["bond_exact_type_rate"]),
            bond_n_scored=float(r["bond_n_scored"]),
            bond_type_macro_f1=float(r["bond_type_macro_f1_pooled"]),
        )[1]
        for r in epochs
    ]
    best_i = max(range(len(values)), key=lambda i: values[i])
    assert int(epochs[best_i]["epoch"]) != 0, (
        "the gate still selects epoch 0 — the least-trained weights of the run"
    )

