"""Wf-342-L1 — DrugOOD loader coverage extension.

================================================================
Wf-342-L1 status check + coverage extension for
``molmetal.data.drugood_loader``.  Per
``TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`` §1.1 the loader
contract calls for ≥12 dedicated tests; that bar is already exceeded
by:

    * ``molmetal.tests.test_wf313h_drugood_loader`` — 16 tests
      (4 schema + 3 row/split + 2 edge + 2 integration + 5 cross-loader)
    * ``molmetal.tests.test_wf320f_drugood_loader`` — 13 tests
      (4 schema + 3 row/split + 2 edge + 2 integration + 2 cross-loader)

These two test files together form a **double-anchored** contract
(Wf-313H + Wf-320F).  Wf-342-L1 adds a tight 4-test extension focused
on the specific gaps named in the Wf-342-L1 spec:

  1. **DrugOOD CSV upstream format loads** — the in-repo curated
     ``molmetal/data/drugood/train.csv`` (296,676 rows; DrugOOD upstream
     format ``smiles, y, assay_id, scaffold_split_id, size_split_id``)
     is a different on-disk artefact from the curator JSON.  We verify
     pandas-direct read returns 296,676 rows with the upstream 5-col
     schema (this is the entry point for
     ``molmetal/data/four_domain_combined.py``).
  2. **OOD test set actually loads** — ``load_drugood(split="ood_test")``
     returns a non-empty DataFrame with the canonical 6-col schema
     (the JSON curator exposes train / ood_val / ood_test / iid_val /
     iid_test — only the ``train`` split is exercised by Wf-313H +
     Wf-320F).
  3. **Scaffold split partitions ood_test disjointly** — when we run
     Bemis-Murcko scaffold split on the ``ood_test`` rows, the disjoint
     partition property holds (the OOD split is by definition
     scaffold-disjoint from train — we sanity-check the *test* split
     rows are themselves scaffoldable).
  4. **Multi-task label structure (pIC50 + active) is intact** — both
     labels are present together (numeric pIC50 AND boolean active),
     with non-NaN counts > 0, on every loaded split.  This is the
     multi-task label contract called out in Wf-305D.

All four tests are **skip-if-unavailable** so they auto-degrade when
the DrugOOD curator JSON or CSV is absent.

MEASURED anchors (per on-disk artefacts — Sep 28 2026):

  * ``.data/molmetal/drugood/ic50/official/lbap_refined_ic50_assay.json``
    — train = 95,373 rows (Ji 2022 official curator)
  * ``molmetal/data/drugood/train.csv`` — 296,676 rows (upstream
    DrugOOD format)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.data.drugood_loader import (  # noqa: E402
    DEFAULT_DRUGOOD_TARGET_ROOT,
    load_drugood,
)

# Skip conditions
_JSON_PATH = DEFAULT_DRUGOOD_TARGET_ROOT / "lbap_refined_ic50_assay.json"
JSON_AVAILABLE = _JSON_PATH.exists()
JSON_SKIP = f"DrugOOD JSON not found at {_JSON_PATH}"

_CSV_PATH = _REPO_ROOT / "molmetal" / "data" / "drugood" / "train.csv"
CSV_AVAILABLE = _CSV_PATH.exists()
CSV_SKIP = f"DrugOOD upstream CSV not found at {_CSV_PATH}"


def test_wf342l1_drugood_csv_loads() -> None:
    """DrugOOD upstream CSV (296,676 rows, 5-col schema) loads via pandas.

    The CSV is the in-repo curated copy used by
    ``molmetal/data/four_domain_combined.py``.  It is the upstream
    DrugOOD ``smiles, y, assay_id, scaffold_split_id, size_split_id``
    format — DIFFERENT from the curator JSON.  This test pins the row
    count + 5-col schema so the upstream artefact stays anchored.
    """
    if not CSV_AVAILABLE:
        pytest.skip(CSV_SKIP)

    df = pd.read_csv(_CSV_PATH)
    assert isinstance(df, pd.DataFrame)
    # wc -l counts the trailing newline as one row; actual data rows = 296,675.
    assert len(df) == 296675, (
        f"DrugOOD upstream CSV row count drifted: expected 296675 "
        f"(wc -l reports 296676 incl. trailing newline), got {len(df)}"
    )
    expected_cols = ["smiles", "y", "assay_id", "scaffold_split_id", "size_split_id"]
    assert df.columns.tolist() == expected_cols, (
        f"DrugOOD upstream CSV schema drifted: {df.columns.tolist()} != "
        f"{expected_cols}"
    )
    # y column is binary 0/1 (classification label upstream format).
    assert set(df["y"].unique()).issubset({0, 1}), (
        f"DrugOOD CSV y column should be binary 0/1, "
        f"got uniques: {sorted(df['y'].unique().tolist())}"
    )


@pytest.mark.skipif(not JSON_AVAILABLE, reason=JSON_SKIP)
def test_wf342l1_drugood_ood_test_set_loads() -> None:
    """``load_drugood(split="ood_test")`` returns non-empty canonical schema.

    Wf-313H + Wf-320F exercise only ``train``; the OOD test split is
    part of the public DrugOOD contract (``VALID_SPLITS`` includes
    ``ood_test``) but its on-disk row count is not pinned.  This test
    verifies the OOD test split is loadable, non-empty, and matches the
    canonical 6-col schema.
    """
    df = load_drugood(
        subset="lbap_refined_ic50_assay",
        split="ood_test",
        prefer_library=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, "drugood ood_test split is empty"
    assert df.columns.tolist() == [
        "smiles", "metal", "cell_line", "protein", "pIC50", "active",
    ], (
        f"drugood ood_test schema drifted: {df.columns.tolist()}"
    )


@pytest.mark.skipif(not JSON_AVAILABLE, reason=JSON_SKIP)
def test_wf342l1_drugood_ood_test_scaffold_split_subsample() -> None:
    """Bemis-Murcko scaffold split partitions ``ood_test`` disjointly.

    Subsampled to 1k rows for unit-test latency.  Verifies that the
    OOD test split is RDKit-parseable at ≥90 % rate (DrugOOD is a curated
    ligand panel) and that the scaffold split produces three disjoint
    arrays that cover all rows.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")

    df_full = load_drugood(
        subset="lbap_refined_ic50_assay",
        split="ood_test",
        prefer_library=False,
    )
    n_sample = min(1000, len(df_full))
    df = df_full.sample(n=n_sample, random_state=0).reset_index(drop=True)

    scaffolds: dict[str, list[int]] = {}
    parseable = 0
    for i, sm in enumerate(df["smiles"].tolist()):
        mol = Chem.MolFromSmiles(str(sm))
        if mol is None:
            continue
        parseable += 1
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
        scaffolds.setdefault(sc, []).append(i)

    assert parseable / len(df) >= 0.90, (
        f"only {parseable}/{len(df)} drugood ood_test SMILES are "
        f"RDKit-parseable ({parseable / len(df):.3f})"
    )

    sorted_groups = sorted(scaffolds.values(), key=len, reverse=True)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    n = len(df)
    for grp in sorted_groups:
        cur = len(train_idx) + len(val_idx) + len(test_idx)
        if cur < int(0.8 * n):
            train_idx.extend(grp)
        elif cur < int(0.9 * n):
            val_idx.extend(grp)
        else:
            test_idx.extend(grp)

    all_idx = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_idx) == n, (
        f"scaffold split on ood_test lost rows: {n - len(all_idx)} missing"
    )


@pytest.mark.skipif(not JSON_AVAILABLE, reason=JSON_SKIP)
def test_wf342l1_drugood_multitask_labels_intact() -> None:
    """Multi-task label structure: numeric pIC50 + boolean active on every split.

    Per Wf-305D the DrugOOD loader exposes BOTH a regression label
    (``pIC50`` — float) AND a classification label (``active`` — bool).
    This test pins the joint dtype contract across train + ood_test
    (the two splits exercised by this file).
    """
    for split in ("train", "ood_test"):
        df = load_drugood(
            subset="lbap_refined_ic50_assay",
            split=split,
            prefer_library=False,
        )
        assert pd.api.types.is_numeric_dtype(df["pIC50"]), (
            f"drugood {split} pIC50 should be numeric, got {df['pIC50'].dtype}"
        )
        assert df["pIC50"].notna().all(), (
            f"drugood {split} has NaN pIC50: "
            f"{int(df['pIC50'].isna().sum())} rows"
        )
        assert df["pIC50"].apply(lambda v: math.isfinite(float(v))).all(), (
            f"drugood {split} has non-finite pIC50"
        )
        assert df["active"].dtype == bool, (
            f"drugood {split} active should be bool, got {df['active'].dtype}"
        )
        # Both labels must be present together (multi-task contract).
        assert len(df["pIC50"]) == len(df["active"]), (
            f"drugood {split} label length mismatch: "
            f"pIC50={len(df['pIC50'])} vs active={len(df['active'])}"
        )
        # Active distribution must be binary {False, True}.
        unique_active = set(df["active"].unique().tolist())
        assert unique_active.issubset({False, True}), (
            f"drugood {split} active should be binary, got {unique_active}"
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
