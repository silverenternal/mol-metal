"""Tests for Wf-312A Re cohort non-degenerate scaffold fallback.

Wf-312A PART 1: under the strict Krasnov 2024/2025 cutoffs the Re
cohort (868 rows / 250 unique SMILES / 123 Bemis-Murcko scaffolds)
collapses to n_test=1 (temporal degenerate). The driver auto-falls-back
to scaffold-OOD split so the per-seed AUC is non-degenerate.

These tests exercise the fallback + the scaffold group size invariant
on a tiny synthetic cohort so the suite stays fast on CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. Re cohort sizing — guard against the degenerate temporal split.
# ---------------------------------------------------------------------------
def test_re_cohort_size_invariants() -> None:
    """Re cohort must have >= 800 rows / >= 100 unique SMILES / >= 50 scaffolds.

    These are the Wf-312A pre-conditions for the scaffold fallback to
    be meaningful. If this test fails, the CSV has shrunk below the
    threshold needed for honest split evaluation.
    """
    import pandas as pd

    csv = PROJECT_ROOT / ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
    if not csv.exists():
        pytest.skip(f"missing {csv}; CSV path is project-local")
    df = pd.read_csv(csv, low_memory=False)
    re_df = df[df["metal"] == "Re"]
    assert len(re_df) >= 800, (
        f"Re cohort shrunk to {len(re_df)} rows; was 868 in Wf-312A. "
        f"Investigate data drift before re-running scaffold fallback."
    )
    assert (
        re_df["smiles"].nunique() >= 100
    ), f"Re unique SMILES = {re_df['smiles'].nunique()}; need >= 100."
    assert "scaffold_id" in re_df.columns, (
        "Re cohort is missing scaffold_id column; scaffold fallback "
        "needs scaffold_id (Wf-312A)."
    )
    assert re_df["scaffold_id"].nunique() >= 50, (
        f"Re distinct scaffolds = {re_df['scaffold_id'].nunique()}; "
        f"need >= 50 for scaffold-OOD fallback."
    )


# ---------------------------------------------------------------------------
# 2. Temporal split is degenerate — proves the bug existed.
# ---------------------------------------------------------------------------
def test_re_temporal_split_is_degenerate() -> None:
    """Under strict Krasnov cutoffs (train<2024, test>=2025), Re test
    window has only 1 row. This is the bug Wf-312A fixes."""
    import pandas as pd

    csv = PROJECT_ROOT / ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
    raw_csv = PROJECT_ROOT / ".data/molmetal/MetalCytoToxDB.csv"
    if not csv.exists() or not raw_csv.exists():
        pytest.skip("CSV / raw CSV not present")
    df = pd.read_csv(csv, low_memory=False)
    df = df[df["metal"] == "Re"].reset_index(drop=True)
    df_raw = pd.read_csv(raw_csv, low_memory=False)
    yr_lookup = (
        df_raw[["SMILES_Ligands", "Metal", "Year"]]
        .rename(columns={"SMILES_Ligands": "smiles", "Metal": "metal"})
        .groupby(["smiles", "metal"], as_index=False)["Year"]
        .min()
    )
    df = df.merge(yr_lookup, on=["smiles", "metal"], how="left")
    n_test_2025 = int((df["Year"] >= 2025).sum())
    assert n_test_2025 < 50, (
        f"Re test_window(Year>=2025) = {n_test_2025}; if >=50 then "
        f"the bug has been fixed at the data layer (Year distribution "
        f"changed) and this guard test should be retired."
    )


# ---------------------------------------------------------------------------
# 3. Scaffold split returns a non-degenerate test set.
# ---------------------------------------------------------------------------
def test_re_scaffold_split_non_degenerate() -> None:
    """_scaffold_split_indices must yield n_test >= 50 for every seed."""
    import pandas as pd

    csv = PROJECT_ROOT / ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
    if not csv.exists():
        pytest.skip("CSV not present")
    # Defer chemprop_v2 import so the test does not require cuik_molmaker.
    sys.path.insert(0, str(PROJECT_ROOT / ".cloned" / "chemprop_v2"))
    try:
        from molmetal.scripts.wf219_temporal_split_driver import (  # noqa: E402
            _scaffold_split_indices,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"driver import failed (cuik_molmaker missing?): {exc}")
    df = pd.read_csv(csv, low_memory=False)
    re_df = df[df["metal"] == "Re"].reset_index(drop=True)
    smiles = re_df["smiles"].astype(str).tolist()
    active = re_df["activity"].astype(int).to_numpy()
    for seed in (0, 1, 2, 3, 4):
        out = _scaffold_split_indices(
            smiles_subset=smiles,
            active_subset=active,
            seed=seed,
        )
        assert out is not None, (
            f"seed={seed}: no scaffold group >= 50 rows; Re cohort "
            f"distribution has shifted; fallback path is broken."
        )
        train_idx, val_idx, test_idx, _label = out
        assert len(test_idx) >= 50, (
            f"seed={seed}: scaffold test set has only {len(test_idx)} "
            f"rows; need >= 50 for honest AUC."
        )
        assert len(train_idx) >= 50, (
            f"seed={seed}: scaffold train set has only {len(train_idx)} "
            f"rows; need >= 50 to train."
        )
        # Sanity: train/val/test disjoint.
        all_idx = (
            set(train_idx.tolist()) | set(val_idx.tolist()) | set(test_idx.tolist())
        )
        assert len(all_idx) == len(train_idx) + len(val_idx) + len(
            test_idx
        ), f"seed={seed}: train/val/test overlap detected."


# ---------------------------------------------------------------------------
# 4. CLI --split auto wires correctly (dry-run).
# ---------------------------------------------------------------------------
def test_wf312a_split_auto_cli() -> None:
    """`--split auto` is accepted by the driver CLI."""
    sys.path.insert(0, str(PROJECT_ROOT / ".cloned" / "chemprop_v2"))
    try:
        from molmetal.scripts.wf219_temporal_split_driver import build_argparser
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"driver import failed: {exc}")
    p = build_argparser()
    for split in ("temporal", "scaffold", "auto"):
        args = p.parse_args(["--split", split, "--dry-run"])
        assert args.split == split
    # bad value
    with pytest.raises(SystemExit):
        p.parse_args(["--split", "random"])


# ---------------------------------------------------------------------------
# 5. run_one_per_metal with split_mode='auto' falls back to scaffold for Re.
# ---------------------------------------------------------------------------
def test_wf312a_run_one_per_metal_auto_uses_scaffold_for_re() -> None:
    """When split_mode='auto' and Re test_window is degenerate, the
    driver must use scaffold-OOD. Verified by inspecting the returned
    ``split`` field on a tiny 1-epoch CPU smoke (no AUC assertion, just
    routing)."""
    sys.path.insert(0, str(PROJECT_ROOT / ".cloned" / "chemprop_v2"))
    try:
        from molmetal.scripts.wf219_temporal_split_driver import (  # noqa: E402
            run_one_per_metal,
        )
        import torch
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"driver / torch import failed: {exc}")

    import pandas as pd

    csv = PROJECT_ROOT / ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
    if not csv.exists():
        pytest.skip("CSV not present")
    df = pd.read_csv(csv, low_memory=False)
    re_df = df[df["metal"] == "Re"].reset_index(drop=True)
    # Use the FULL Re cohort so the scaffold-OOD fallback finds a group
    # of >= 50 rows. Synthetic year = all 2020 so test_window=0 in
    # temporal mode and the driver must fall back to scaffold.
    smiles = re_df["smiles"].astype(str).tolist()
    active = re_df["activity"].astype(int).to_numpy()
    year = np.array([2020] * len(smiles), dtype=np.int64)
    run = run_one_per_metal(
        smiles_subset=smiles,
        active_subset=active,
        year_subset=year,
        train_cutoff="2024-01-01",
        test_cutoff="2025-01-01",
        epochs=1,
        batch_size=32,
        lr=1e-3,
        device=torch.device("cpu"),
        seed=0,
        atom_fdim=72,
        bond_fdim=14,
        hidden_dim=32,
        depth=2,
        num_workers=0,
        split_mode="auto",
    )
    # With n_test(>=2025)=0 in the synthetic all-2020 year vector the
    # driver must fall back to scaffold. Otherwise the n_test would be 0
    # and AUC would be NaN.
    assert (
        "scaffold" in run.split or "fallback" in run.split or run.n_test >= 50
    ), f"auto mode did not fall back: split={run.split} n_test={run.n_test}"
    assert run.n_test >= 50, (
        f"non-degenerate scaffold fallback still gave n_test="
        f"{run.n_test}; expected >= 50."
    )
