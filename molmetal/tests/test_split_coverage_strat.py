"""Coverage-stratified scaffold split.

The legacy `largest_first` path budgets capacities in scaffolds and breaks size
ties by file order.  On the PlatinAI 4-cell cohort that starved A2780cis to 27
test rows (see `metrics/by_metric/split_coverage_4cell.json`).  These tests pin
both halves: the legacy path still produces exactly the split it always did, and
the new path keeps the leakage invariant while making coverage order-invariant.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from molmetal.data.splits import LigandDeduplicatedSplitter, ScaffoldSplitter

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / ".data" / "molmetal" / "platinai_4cell_real_trainer.csv"
CELLS = {
    "A549": "pIC50_a549",
    "A2780": "pIC50_a2780",
    "A2780cis": "pIC50_a2780cis",
    "MCF-7": "pIC50_mcf7",
}

# Ten Bemis-Murcko scaffolds with deliberately unequal family sizes.  The
# "key" labels are concentrated on the three largest families, which is exactly
# the shape that largest_first starves.
FAMILIES = [
    ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "CCCCc1ccccc1",
     "Oc1ccccc1", "Nc1ccccc1", "Fc1ccccc1", "Clc1ccccc1", "Brc1ccccc1"],
    ["c1ccncc1", "Cc1ccncc1", "CCc1ccncc1", "Oc1ccncc1", "Nc1ccncc1",
     "Fc1ccncc1", "Clc1ccncc1", "Brc1ccncc1", "Ic1ccncc1"],
    ["c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1", "Oc1ccc2ccccc2c1",
     "Nc1ccc2ccccc2c1", "Fc1ccc2ccccc2c1", "Clc1ccc2ccccc2c1",
     "Brc1ccc2ccccc2c1", "Ic1ccc2ccccc2c1"],
    ["C1CCCCC1", "CC1CCCCC1", "OC1CCCCC1", "NC1CCCCC1", "FC1CCCCC1",
     "ClC1CCCCC1", "BrC1CCCCC1"],
    ["C1CCCC1", "CC1CCCC1", "OC1CCCC1", "NC1CCCC1", "FC1CCCC1", "ClC1CCCC1"],
    ["C1CCCCCC1", "CC1CCCCCC1", "OC1CCCCCC1", "NC1CCCCCC1", "FC1CCCCCC1"],
    ["c1ccc2[nH]ccc2c1", "Cc1ccc2[nH]ccc2c1", "Oc1ccc2[nH]ccc2c1",
     "Fc1ccc2[nH]ccc2c1"],
    ["c1ccc2ncccc2c1", "Cc1ccc2ncccc2c1", "Oc1ccc2ncccc2c1"],
    ["c1ccsc1", "Cc1ccsc1"],
    ["c1ccoc1"],
]
# Labelled rows per family for the synthetic key: 9, 8, 7, 3, 1, 0, ...
KEY_LABELS = [9, 8, 7, 3, 1, 0, 0, 0, 0, 0]


class _DS:
    def __init__(self, smiles):
        self.smiles = np.asarray(list(smiles), dtype=object)

    def __len__(self):
        return len(self.smiles)


def _fixture():
    smiles, mask = [], []
    for family, n_labelled in zip(FAMILIES, KEY_LABELS):
        for j, smi in enumerate(family):
            smiles.append(smi)
            mask.append(j < n_labelled)
    return _DS(smiles), np.array(mask, dtype=bool)


def _split_map(result, n):
    out = {}
    for name, idx in (
        ("train", result.train_idx),
        ("val", result.val_idx),
        ("test", result.test_idx),
    ):
        for i in idx:
            out[int(i)] = name
    assert len(out) == n
    return out


def test_legacy_path_still_starves_the_concentrated_key():
    ds, mask = _fixture()
    result = ScaffoldSplitter(fractions=(0.7, 0.1, 0.2))(ds)
    assign = _split_map(result, len(ds))
    test_rows = [i for i, s in assign.items() if s == "test"]
    assert int(mask[test_rows].sum()) == 0


def test_coverage_path_recovers_the_concentrated_key():
    ds, mask = _fixture()
    result = ScaffoldSplitter(
        fractions=(0.7, 0.1, 0.2), coverage={"key": mask}
    )(ds)
    assign = _split_map(result, len(ds))
    test_rows = [i for i, s in assign.items() if s == "test"]
    target = 0.2 * mask.sum()
    assert mask[test_rows].sum() >= 0.5 * target


def test_coverage_path_keeps_one_scaffold_per_split():
    ds, mask = _fixture()
    splitter = ScaffoldSplitter(fractions=(0.7, 0.1, 0.2), coverage={"key": mask})
    assign = _split_map(splitter(ds), len(ds))

    canon = splitter._canonicalize(ds.smiles)
    by_scaffold = {}
    for i, smi in enumerate(canon):
        by_scaffold.setdefault(splitter._scaffold_for(str(smi)), set()).add(assign[i])
    leaks = {k: v for k, v in by_scaffold.items() if len(v) > 1}
    assert not leaks, f"scaffold(s) split across folds: {list(leaks)[:3]}"


def test_coverage_path_is_invariant_to_row_order():
    ds, mask = _fixture()
    base = _split_map(
        ScaffoldSplitter(fractions=(0.7, 0.1, 0.2), coverage={"key": mask})(ds),
        len(ds),
    )
    rng = np.random.default_rng(0)
    for _ in range(3):
        perm = rng.permutation(len(ds))
        shuffled = _DS([ds.smiles[i] for i in perm])
        other = _split_map(
            ScaffoldSplitter(
                fractions=(0.7, 0.1, 0.2), coverage={"key": mask[perm]}
            )(shuffled),
            len(ds),
        )
        # Compare in the original row space.
        remapped = {int(perm[i]): other[i] for i in range(len(ds))}
        assert remapped == base


def test_coverage_path_is_deterministic():
    ds, mask = _fixture()
    a = ScaffoldSplitter(fractions=(0.7, 0.1, 0.2), coverage={"key": mask})(ds)
    b = ScaffoldSplitter(fractions=(0.7, 0.1, 0.2), coverage={"key": mask})(ds)
    assert np.array_equal(a.train_idx, b.train_idx)
    assert np.array_equal(a.val_idx, b.val_idx)
    assert np.array_equal(a.test_idx, b.test_idx)


def test_wrong_length_coverage_raises():
    ds, mask = _fixture()
    with pytest.raises(ValueError, match="expected"):
        ScaffoldSplitter(fractions=(0.7, 0.1, 0.2), coverage={"key": mask[:-1]})(ds)


def test_all_empty_coverage_raises():
    ds, _ = _fixture()
    with pytest.raises(ValueError, match="nothing to stratify"):
        ScaffoldSplitter(
            fractions=(0.7, 0.1, 0.2),
            coverage={"key": np.zeros(len(ds), dtype=bool)},
        )(ds)


def test_absent_coverage_leaves_the_legacy_assignment_untouched():
    """`coverage=None` must not reach the new path, even by accident."""
    ds, _ = _fixture()
    splitter = ScaffoldSplitter(fractions=(0.7, 0.1, 0.2))
    canon = splitter._canonicalize(ds.smiles)
    groups = {}
    for i, smi in enumerate(canon):
        key = splitter._scaffold_for(str(smi)) or f"__empty__{smi}"
        groups.setdefault(key, []).append(i)
    expected = LigandDeduplicatedSplitter(
        fractions=(0.7, 0.1, 0.2)
    )._assign_groups(list(groups.values()))
    assert _split_map(splitter(ds), len(ds)) == {int(k): v for k, v in expected.items()}


@pytest.mark.skipif(not COHORT.exists(), reason="REAL 4-cell cohort not on disk")
def test_real_cohort_legacy_split_is_the_one_that_was_trained():
    rows = list(csv.DictReader(open(COHORT, newline="")))
    ds = _DS([r["SMILES_Ligands"] for r in rows])
    result = ScaffoldSplitter(fractions=(0.7, 0.1, 0.2))(ds)
    assert len(result.train_idx) == 1650
    assert len(result.val_idx) == 105
    assert len(result.test_idx) == 211


@pytest.mark.skipif(not COHORT.exists(), reason="REAL 4-cell cohort not on disk")
def test_real_cohort_every_cell_line_clears_the_bar_in_test():
    from molmetal.scripts.train_cisplatin_resistance import MIN_EVAL_ROWS

    rows = list(csv.DictReader(open(COHORT, newline="")))
    ds = _DS([r["SMILES_Ligands"] for r in rows])
    masks = {cl: np.array([r[col] != "" for r in rows]) for cl, col in CELLS.items()}
    result = ScaffoldSplitter(
        fractions=(0.7, 0.1, 0.2), coverage=masks
    )(ds)
    for cl, mask in masks.items():
        assert int(mask[result.test_idx].sum()) >= MIN_EVAL_ROWS, cl
        assert int(mask[result.val_idx].sum()) >= MIN_EVAL_ROWS, cl
