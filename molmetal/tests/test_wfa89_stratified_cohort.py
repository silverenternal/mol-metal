# SPDX-License-Identifier: Apache-2.0
"""A capped cohort must still be a 5-metal cohort.

WHY THIS EXISTS
---------------
``build_csv_cohort`` used to select with ``rows[:n_clouds]``.  The corpus CSV
is grouped by metal — MEASURED: the first 512 train rows of
``metalcytotox_5metal_multitask_v3.csv`` are ALL Ru (full split: Ru 12,471 /
Ir 3,444 / Os 834 / Rh 696 / Re 692).  So every capped run, including the
``--n-train 64`` CPU smoke used for audits, trained a "5-metal" model on one
metal, and the val cohort was Ru-only.  The cohort line printed "64 inserted"
and looked healthy, because a metal count cannot distinguish one metal from
five.

RED CHECK: the proportional-quota case asserts exact per-metal counts, and the
real-CSV case fails if the selection ever collapses back to a prefix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _rows(metal: str, n: int):
    return [(f"C{metal}{i}", metal, None) for i in range(n)]


def test_prefix_selection_would_have_been_single_metal():
    """Guard the premise: the fixture really is metal-grouped."""
    rows = _rows("Ru", 512) + _rows("Ir", 64)
    assert {r[1] for r in rows[:64]} == {"Ru"}


def test_capped_selection_covers_every_metal():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _stratified_row_sample,
    )

    rows = (
        _rows("Ru", 400)
        + _rows("Ir", 100)
        + _rows("Os", 60)
        + _rows("Rh", 40)
        + _rows("Re", 40)
    )
    picked = _stratified_row_sample(rows, 64, seed=0)
    assert len(picked) == 64
    assert {r[1] for r in picked} == {"Ru", "Ir", "Os", "Rh", "Re"}


def test_quota_is_proportional():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _stratified_row_sample,
    )

    rows = _rows("Ru", 100) + _rows("Ir", 20)
    picked = _stratified_row_sample(rows, 12, seed=3)
    counts = {}
    for r in picked:
        counts[r[1]] = counts.get(r[1], 0) + 1
    assert counts == {"Ru": 10, "Ir": 2}


def test_selection_is_deterministic_and_seed_sensitive():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _stratified_row_sample,
    )

    rows = _rows("Ru", 200) + _rows("Ir", 80)
    a = _stratified_row_sample(rows, 40, seed=7)
    b = _stratified_row_sample(rows, 40, seed=7)
    c = _stratified_row_sample(rows, 40, seed=8)
    assert [r[0] for r in a] == [r[0] for r in b]
    assert [r[0] for r in a] != [r[0] for r in c]


def test_full_run_is_untouched():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _stratified_row_sample,
    )

    rows = _rows("Ru", 10) + _rows("Ir", 5)
    assert _stratified_row_sample(rows, 15, seed=0) == rows
    assert _stratified_row_sample(rows, 999, seed=0) == rows


def test_real_corpus_prefix_is_stratified():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _stratified_row_sample,
    )

    csv_path = Path(".data/molmetal/metalcytotox_5metal_multitask_v3.csv")
    if not csv_path.exists():
        pytest.skip("corpus CSV not on disk")
    import csv as _csv

    rows = [
        (
            (r.get("smiles") or r.get("SMILES_Ligands") or "").strip(),
            (r.get("Metal") or "").strip(),
            None,
        )
        for r in _csv.DictReader(csv_path.open(newline=""))
        if (r.get("split") or "").strip() == "train"
    ]
    picked = _stratified_row_sample(rows, 64, seed=0)
    metals = {r[1] for r in picked}
    assert len(picked) == 64
    assert len(metals) >= 4, f"capped cohort collapsed to {metals}"
