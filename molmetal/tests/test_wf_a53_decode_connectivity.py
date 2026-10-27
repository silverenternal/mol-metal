"""Wf-A53 — the trainer's decode metric must not read "parses" as "connected".

`decode_ratio` counts RDKit-sanitised strings.  A cloud of loose atoms
sanitises perfectly, so a head that predicts BOND_NO_BOND everywhere scores
1.000.  MEASURED on the shipped 5-metal checkpoint: 256/256 "successes",
0/256 single-fragment, mean 20.5 fragments, max 45.

These tests pin the three pieces of the fix that do not need a model:

1. ``_decoded_smiles_stats`` measures connectivity from the decoded SMILES.
2. ``EpochMetrics`` carries the three connectivity fields (so the CSV/JSON
   writer cannot silently drop them).
3. ``evaluate_decode_ratio`` accepts ``stats_out`` and
   ``_decode_with_rdkit_strict`` keeps its old boolean contract.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.scripts import train_cfm_yuelbond_wasserstein as trainer  # noqa: E402


def test_decoded_smiles_stats_counts_connectivity():
    pytest.importorskip("rdkit")
    out = trainer._decoded_smiles_stats(
        [
            "CCO",  # 1 fragment
            "CCO.O.[HH]",  # 3 fragments
            "c1ccccc1.C[Rh](C)C",  # 2 fragments, metal present
        ]
    )
    assert out["decode_n_decoded"] == 3.0
    assert out["decode_n_single_fragment"] == 1.0
    assert out["decode_single_fragment_ratio"] == pytest.approx(1 / 3)
    assert out["decode_fragments_max"] == 3.0
    assert out["decode_fragments_mean"] == pytest.approx(2.0)
    assert out["decode_n_with_metal"] == 1.0
    assert out["decode_metal_ratio"] == pytest.approx(1 / 3)


def test_decoded_smiles_stats_skips_unparseable_and_empty():
    pytest.importorskip("rdkit")
    out = trainer._decoded_smiles_stats(["CCO", "not-a-smiles", ""])
    assert out["decode_n_decoded"] == 1.0
    assert out["decode_single_fragment_ratio"] == 1.0


def test_decoded_smiles_stats_is_empty_when_nothing_decodes():
    pytest.importorskip("rdkit")
    out = trainer._decoded_smiles_stats([])
    assert out["decode_n_decoded"] == 0
    assert out["decode_single_fragment_ratio"] == 0.0


def test_epoch_metrics_carries_connectivity_fields():
    fields = trainer.EpochMetrics.__dataclass_fields__
    for name in (
        "decode_n_single_fragment",
        "decode_single_fragment_ratio",
        "decode_fragments_mean",
    ):
        assert name in fields, f"{name} missing from EpochMetrics"


def test_evaluate_decode_ratio_accepts_stats_out():
    params = inspect.signature(trainer.evaluate_decode_ratio).parameters
    assert "stats_out" in params
    assert params["stats_out"].default is None


def test_decode_with_rdkit_strict_keeps_its_boolean_contract(monkeypatch):
    """The old helper must return only the boolean, not the tuple."""
    monkeypatch.setattr(
        trainer,
        "_decode_with_rdkit_strict_detail",
        lambda decoder, cloud: (True, "CCO"),
    )
    assert trainer._decode_with_rdkit_strict(None, None) is True

    monkeypatch.setattr(
        trainer,
        "_decode_with_rdkit_strict_detail",
        lambda decoder, cloud: (False, "C.C"),
    )
    assert trainer._decode_with_rdkit_strict(None, None) is False


def test_n_train_zero_means_every_row():
    """configs/cfm_5metal_v5_full.yaml relies on ``<= 0`` meaning "all rows".

    If ``_stratified_row_sample`` ever treats 0 as "no rows", that config
    would silently train on an empty cohort.  This pins the contract the
    config documents.
    """
    rows = [("CCO", "Ru", None)] * 10 + [("CCN", "Ir", None)] * 10
    out = trainer._stratified_row_sample(rows, 0, seed=0)
    assert len(out) == len(rows)
    out_neg = trainer._stratified_row_sample(rows, -1, seed=0)
    assert len(out_neg) == len(rows)
    capped = trainer._stratified_row_sample(rows, 4, seed=0)
    assert len(capped) == 4


# ---------------------------------------------------------------------------
# Wf-A113 (2026-10-27) / T47 — the metal set is ONE definition, and it has Pt
# ---------------------------------------------------------------------------


def test_pt_only_decode_counts_as_containing_metal():
    """MEASURED defect this pins: ``metal_z`` was a hand-written
    ``{44, 45, 75, 76, 77}`` that omitted Pt (78), so a Pt-only decode scored
    ``decode_metal_ratio = 0.0`` — "no metal" for the metal the project is
    named after — and ``wf_a53_eval_cfm_test_split`` re-published that number
    under the bare name ``metal_ratio``.
    """
    pytest.importorskip("rdkit")
    out = trainer._decoded_smiles_stats(["[Pt](N)(N)(Cl)Cl"])
    assert out["decode_n_with_metal"] == 1.0
    assert out["decode_metal_ratio"] == 1.0


def test_metal_cohort_is_the_target_cohort_and_is_reported():
    """The cohort is ``_TARGET_METALS``, and it travels with the number."""
    pytest.importorskip("rdkit")
    out = trainer._decoded_smiles_stats(["[Pt](N)(N)(Cl)Cl", "CCO"])
    assert out["decode_metal_cohort"] == sorted(trainer._TARGET_METALS)
    assert 78 in out["decode_metal_cohort"]
    empty = trainer._decoded_smiles_stats([])
    assert empty["decode_metal_cohort"] == sorted(trainer._TARGET_METALS)


def test_no_five_element_metal_literal_remains_in_the_trainer():
    """A second literal would re-create the divergence silently."""
    src = (
        _REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    ).read_text(encoding="utf-8")
    for literal in ("{44, 45, 75, 76, 77}", "{44,45,75,76,77}"):
        assert literal not in src, (
            f"{literal} is a 5-metal set without Pt; use _TARGET_METALS"
        )
