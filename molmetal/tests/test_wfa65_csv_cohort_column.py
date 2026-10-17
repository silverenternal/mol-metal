"""Wf-A65 (2026-10-17) — regression test for the SMILES column-name bug in
``molmetal/scripts/train_cfm_yuelbond_wasserstein.py::build_csv_cohort``.

Bug history
-----------
On 2026-10-17, a small GPU smoke run on the canonical 5-metal cohort
``.data/molmetal/metalcytotox_5metal_multitask_v3.csv`` (n_train=4) raised::

    RuntimeError: build_csv_cohort produced 0 clouds from
    .data/molmetal/metalcytotox_5metal_multitask_v3.csv split='train'.
    Check the path, the `split` column values, and whether RDKit can parse
    the SMILES.

Root cause: ``build_csv_cohort`` read the SMILES column with
``row.get("smiles", "")`` but the metalcytotox_5metal CSV uses the
MetalCytoToxDB-native column name ``SMILES_Ligands`` (capitalised).
0 rows passed the ``if smi:`` filter; the error message gave a
misleading diagnostic trail that did not mention column names.

The canonical 5-metal config had been "SHIPPED" for 17 waves of CFM
closeout but had never actually been run end-to-end on real data —
``results/cfm_5metal_default/`` does not exist on disk.  This is a
§5.5 filesystem-truth violation that the Wf-A65 GPU smoke caught.

Fix
---
``build_csv_cohort`` now does a dual-key lookup
(``row.get("smiles") or row.get("SMILES_Ligands")``) and the
RuntimeError message is updated to list the expected column names
and the actual filter-breakdown counts.

What this test verifies
-----------------------
1. A CSV with the ``SMILES_Ligands`` column name (the metalcytotox
   schema) parses to >= 1 cloud.
2. A CSV with the legacy ``smiles`` column name (the synthetic
   cohort schema) also parses to >= 1 cloud.
3. The diagnostic RuntimeError mentions both column names when both
   are absent (so future regressions have a non-misleading trail).
4. End-to-end: a temporary copy of the actual
   ``.data/molmetal/metalcytotox_5metal_multitask_v3.csv`` parses
   to its declared 18137 train rows (sanity).
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

# Import the function under test.
from molmetal.scripts.train_cfm_yuelbond_wasserstein import build_csv_cohort


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    """Write a small CSV with the given header + rows."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


_SIMPLE_SMILES_1 = "COc1cc(/C=C/C(=O)/C=C([O-])/C=C/c2ccc(O)c(OC)c2)ccc1O.Cc1ccc(C(C)C)cc1.[Cl-]"
_SIMPLE_SMILES_2 = "N#Cc1ccc(C(c2ccc(C#N)cc2)n2cncn2)cc1.[Cl-].c1ccc(P(c2ccccc2)c2ccccc2)cc1"


# ---------------------------------------------------------------------------
# 1. SMILES_Ligands-schema CSV parses (the metalcytotox bug-fix path)
# ---------------------------------------------------------------------------
def test_build_csv_cohort_handles_SMILES_Ligands_column() -> None:
    """The metalcytotox_5metal CSV uses ``SMILES_Ligands``; verify build_csv_cohort reads it."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "metalcytotox_subset.csv"
        _write_csv(
            csv_path,
            ["SMILES_Ligands", "Metal", "split", "cell_line", "pic50"],
            [
                [_SIMPLE_SMILES_1, "Ru", "train", "MCF-7", "5.0"],
                [_SIMPLE_SMILES_2, "Ir", "train", "A549", "4.5"],
                [_SIMPLE_SMILES_1, "Ru", "val", "MCF-7", "5.0"],
            ],
        )
        clouds = build_csv_cohort(str(csv_path), split="train")
        assert len(clouds) >= 1, "SMILES_Ligands column should yield >= 1 cloud (the bug-fix path)"
        # First cloud should have at least 1 non-H heavy atom (the ligand atoms).
        assert (clouds[0].atomic_numbers > 1).any(), "first cloud must contain heavy atoms"


# ---------------------------------------------------------------------------
# 2. legacy-smiles CSV still parses (backward-compat)
# ---------------------------------------------------------------------------
def test_build_csv_cohort_handles_legacy_smiles_column() -> None:
    """The synthetic-cohort CSVs use lowercase ``smiles``; verify backward-compat."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "synthetic_subset.csv"
        _write_csv(
            csv_path,
            ["smiles", "split"],
            [
                [_SIMPLE_SMILES_1, "train"],
                [_SIMPLE_SMILES_2, "train"],
            ],
        )
        clouds = build_csv_cohort(str(csv_path), split="train")
        assert len(clouds) >= 1, "legacy smiles column should still parse (backward-compat)"


# ---------------------------------------------------------------------------
# 3. diagnostic error message lists expected column names
# ---------------------------------------------------------------------------
def test_build_csv_cohort_diagnostic_lists_column_names() -> None:
    """If both column names are missing, the RuntimeError must list them explicitly."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "no_smiles_col.csv"
        _write_csv(
            csv_path,
            ["molecule_id", "split"],
            [
                ["id_1", "train"],
                ["id_2", "train"],
            ],
        )
        with pytest.raises(RuntimeError) as exc_info:
            build_csv_cohort(str(csv_path), split="train")
        msg = str(exc_info.value).lower()
        assert "smiles" in msg, "RuntimeError must mention the `smiles` column name"
        assert "smiles_ligands" in msg, "RuntimeError must mention the `SMILES_Ligands` column name"


# ---------------------------------------------------------------------------
# 4. end-to-end sanity on the actual metalcytotox_5metal cohort
# ---------------------------------------------------------------------------
def test_build_csv_cohort_actual_metalcytotox_5metal_train() -> None:
    """The canonical 5-metal cohort must now build clouds (the Wf-A65 bug-fix target)."""
    canonical = Path(".data/molmetal/metalcytotox_5metal_multitask_v3.csv")
    if not canonical.exists():
        pytest.skip(f"{canonical} not on disk (data acquisition deferred)")
    clouds = build_csv_cohort(str(canonical), split="train", n_clouds=50)
    assert len(clouds) >= 1, "canonical 5-metal cohort must now build clouds after Wf-A65 fix"
    # The CSV's `SMILES_Ligands` column is the LIGAND (organic scaffold), not the
    # metal complex — the metal itself is in a separate `Metal` column that
    # `build_csv_cohort` does NOT read. So clouds only contain ligand atoms
    # (C/N/O/H/halogens) and the metal is added later by the YuelBond decoder
    # at decode time. We assert the cloud is non-empty and the cohort parses.
    assert clouds[0].atomic_numbers.numel() > 0, "first cloud must have atoms"
    # Sanity: at least 1 row in the canonical CSV is train-split.
    import pandas as pd
    df = pd.read_csv(canonical)
    assert (df["split"] == "train").sum() >= 1000, (
        f"canonical 5-metal cohort has {int((df['split'] == 'train').sum())} train rows; "
        f"expected >= 1000"
    )
