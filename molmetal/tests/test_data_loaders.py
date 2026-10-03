"""Smoke tests for the three canonical-6-col data loaders.

================================================================
BD-1 / Wf-130 (TODO-pending/99_drugood_loader.md) — wire the
DrugOOD library into the project's canonical schema and verify that
``load_binding_db``, ``load_nci60`` and ``load_drugood`` all return
DataFrames with **identical** column names:

    smiles, metal, cell_line, protein, pIC50, active

Test design
-----------
1.  ``test_binding_db_loader`` — ``load_binding_db()`` returns a
    non-empty DataFrame with the canonical 6-col schema, all
    ``pIC50`` values finite.
2.  ``test_nci60_loader``     — ``load_nci60()`` returns a non-empty
    DataFrame with the canonical 6-col schema (post-Wf-130
    ``cell`` → ``cell_line`` rename + ``metal``/``protein`` NA fill);
    all ``pIC50`` values finite.
3.  ``test_drugood_loader``   — ``load_drugood()`` returns a non-empty
    DataFrame with the canonical 6-col schema, using the
    ``lbap_refined_ic50_assay`` subset (Ji et al. NeurIPS 2022
    refined curator — assay-domain OOD split).  Runs against the
    project's DrugOOD JSON payload
    (``molmetal/.data/molmetal/drugood/ic50/official/``); uses the
    raw-JSON fallback path so the test is independent of whether the
    ``drugood`` conda env is active.
4.  ``test_all_three_canonical_schema_match`` — the three loaders'
    DataFrame column lists are identical and exactly equal to the
    canonical 6-col schema.  This is the GATE-6 contract.

Backward-compat sanity checks
-----------------------------
The tests use the documented public functions (``load_binding_db``,
``load_nci60``, ``load_drugood``) and check that no caller uses any
removed/renamed column.  In particular, ``load_nci60`` was previously
4-col (``smiles, cell, pIC50, active``); the canonical 6-col update
keeps the loader importable and returns a superset of the old
columns plus two new NA-filled ones — no existing pipeline breaks
(``pre_featurize_dmpnn.py`` only reads ``["smiles"]``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable for ``molmetal.data`` and the local
# ``conftest`` if any.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.data.binding_db import load_binding_db  # noqa: E402
from molmetal.data.nci60 import load_nci60  # noqa: E402
from molmetal.data.drugood_loader import (  # noqa: E402
    DEFAULT_DRUGOOD_TARGET_ROOT,
    load_drugood,
)


# ---------------------------------------------------------------------------
# Canonical-schema reference (GATE-6)
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = ("smiles", "metal", "cell_line", "protein", "pIC50", "active")


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------
_DRUGOOD_JSON = DEFAULT_DRUGOOD_TARGET_ROOT / "lbap_refined_ic50_assay.json"


# ---------------------------------------------------------------------------
# 1. binding_db loader
# ---------------------------------------------------------------------------
class TestBindingDBLoader:
    def test_binding_db_loader(self) -> None:
        """``load_binding_db`` returns canonical 6-col, non-empty, finite pIC50."""
        df = load_binding_db()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "binding_db loader returned 0 rows"

        # GATE-3 + GATE-6: canonical 6-col schema, exact column order.
        assert (
            tuple(df.columns) == CANONICAL_COLUMNS
        ), f"binding_db columns {df.columns.tolist()} != canonical {CANONICAL_COLUMNS}"

        # Type contract
        assert (
            df["active"].dtype == bool
        ), f"active should be bool, got {df['active'].dtype}"
        assert pd.api.types.is_numeric_dtype(
            df["pIC50"]
        ), f"pIC50 should be numeric, got {df['pIC50'].dtype}"

        # All pIC50 finite
        n_nan = int(df["pIC50"].isna().sum())
        assert n_nan == 0, f"binding_db has {n_nan} NaN pIC50 rows"
        assert (
            df["pIC50"].apply(lambda v: math.isfinite(float(v))).all()
        ), "binding_db has non-finite pIC50 values"

        # No empty SMILES / metal / cell_line / protein
        for c in ("smiles", "metal", "cell_line", "protein"):
            assert (df[c].astype(str) != "").all(), f"binding_db has empty {c} entries"

        # Documented behaviour: 5257 rows for the project-shipped CSV.
        assert (
            len(df) == 5257
        ), f"binding_db row count drifted: expected 5257, got {len(df)}"


# ---------------------------------------------------------------------------
# 2. nci60 loader
# ---------------------------------------------------------------------------
class TestNCI60Loader:
    def test_nci60_loader(self) -> None:
        """``load_nci60`` returns canonical 6-col (post-Wf-130 update)."""
        df = load_nci60()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "nci60 loader returned 0 rows"

        # GATE-4 + GATE-6: canonical 6-col schema (was 4-col pre-Wf-130).
        assert (
            tuple(df.columns) == CANONICAL_COLUMNS
        ), f"nci60 columns {df.columns.tolist()} != canonical {CANONICAL_COLUMNS}"

        # metal + protein are NA-filled (NCI-60 has no per-row metal /
        # protein identifier in the source CSV).
        assert (
            df["metal"] == "NA"
        ).all(), f"nci60 metal should be 'NA'-filled, got uniques: {df['metal'].unique()[:5]}"
        assert (
            df["protein"] == "NA"
        ).all(), f"nci60 protein should be 'NA'-filled, got uniques: {df['protein'].unique()[:5]}"

        # cell_line still carries real NCI-60 identifiers (was ``cell``
        # pre-Wf-130 — verify it is a non-trivial set).
        unique_cells = df["cell_line"].unique()
        assert (
            len(unique_cells) >= 3
        ), f"nci60 cell_line should have ≥3 unique NCI-60 lines, got {unique_cells}"

        # Type contract + finite pIC50
        assert (
            df["active"].dtype == bool
        ), f"active should be bool, got {df['active'].dtype}"
        assert pd.api.types.is_numeric_dtype(
            df["pIC50"]
        ), f"pIC50 should be numeric, got {df['pIC50'].dtype}"
        n_nan = int(df["pIC50"].isna().sum())
        assert n_nan == 0, f"nci60 has {n_nan} NaN pIC50 rows"

        # Documented behaviour: 7823 rows for the project-shipped CSV.
        assert len(df) == 7823, f"nci60 row count drifted: expected 7823, got {len(df)}"


# ---------------------------------------------------------------------------
# 3. drugood loader
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _DRUGOOD_JSON.exists(),
    reason=f"DrugOOD subset JSON not found at {_DRUGOOD_JSON}",
)
class TestDrugOODLoader:
    def test_drugood_loader(self) -> None:
        """``load_drugood`` returns canonical 6-col from the lbap_refined_ic50_assay subset.

        Uses ``prefer_library=False`` so the test runs under the
        project's primary Python env (the ``drugood`` conda env
        brings mmcv-1.5.0, which is not installed in ``.venv``).  Both
        code paths flatten into the same canonical schema — this
        asserts the schema, not which path was taken.
        """
        df = load_drugood(
            subset="lbap_refined_ic50_assay",
            split="train",
            prefer_library=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "drugood loader returned 0 rows"

        # GATE-1 + GATE-2 + GATE-6: canonical 6-col schema, exact order.
        assert (
            tuple(df.columns) == CANONICAL_COLUMNS
        ), f"drugood columns {df.columns.tolist()} != canonical {CANONICAL_COLUMNS}"

        # Type contract
        assert (
            df["active"].dtype == bool
        ), f"active should be bool, got {df['active'].dtype}"
        assert pd.api.types.is_numeric_dtype(
            df["pIC50"]
        ), f"pIC50 should be numeric, got {df['pIC50'].dtype}"

        # DrugOOD has no per-row metal / cell_line identifier — both
        # columns are NA-filled by design.
        assert (
            df["metal"] == "NA"
        ).all(), f"drugood metal should be 'NA'-filled, got uniques: {df['metal'].unique()[:5]}"
        assert (
            df["cell_line"] == "NA"
        ).all(), f"drugood cell_line should be 'NA'-filled, got uniques: {df['cell_line'].unique()[:5]}"

        # protein carries the assay_id proxy (stringified); should be
        # non-empty and reasonably diverse (DrugOOD assay split has
        # thousands of unique assay ids).
        unique_assays = df["protein"].unique()
        assert len(unique_assays) >= 100, (
            f"drugood protein (assay_id) should have ≥100 unique values, "
            f"got {len(unique_assays)}"
        )

        # All pIC50 finite (no NaN, no inf)
        n_nan = int(df["pIC50"].isna().sum())
        assert n_nan == 0, f"drugood has {n_nan} NaN pIC50 rows"
        assert (
            df["pIC50"].apply(lambda v: math.isfinite(float(v))).all()
        ), "drugood has non-finite pIC50 values"

        # Documented behaviour for lbap_refined_ic50_assay train split
        # is 95,373 rows (Ji 2022 official curator output).
        assert len(df) == 95373, (
            f"drugood lbap_refined_ic50_assay train row count drifted: "
            f"expected 95373, got {len(df)}"
        )


# ---------------------------------------------------------------------------
# 4. Cross-loader canonical-schema match (GATE-6 contract)
# ---------------------------------------------------------------------------
class TestCanonicalSchemaMatch:
    def test_all_three_canonical_schema_match(self) -> None:
        """All three loaders return the same canonical column list.

        This is the GATE-6 contract: ``smiles, metal, cell_line,
        protein, pIC50, active`` in this exact order — for
        ``load_binding_db``, ``load_nci60`` and ``load_drugood``.
        Downstream concatenation in
        ``molmetal/scripts/pre_featurize_dmpnn.py`` and the
        multi-domain featurisers relies on this invariant.
        """
        df_bd = load_binding_db()
        df_nci = load_nci60()

        bd_cols = tuple(df_bd.columns)
        nci_cols = tuple(df_nci.columns)
        assert (
            bd_cols == CANONICAL_COLUMNS
        ), f"binding_db columns {bd_cols} != canonical {CANONICAL_COLUMNS}"
        assert (
            nci_cols == CANONICAL_COLUMNS
        ), f"nci60 columns {nci_cols} != canonical {CANONICAL_COLUMNS}"
        assert (
            bd_cols == nci_cols
        ), f"binding_db and nci60 columns diverge: {bd_cols} vs {nci_cols}"

        # DrugOOD is optional — only run if the JSON is available.
        if _DRUGOOD_JSON.exists():
            df_do = load_drugood(
                subset="lbap_refined_ic50_assay",
                split="train",
                prefer_library=False,
            )
            do_cols = tuple(df_do.columns)
            assert (
                do_cols == CANONICAL_COLUMNS
            ), f"drugood columns {do_cols} != canonical {CANONICAL_COLUMNS}"
            assert do_cols == bd_cols == nci_cols, (
                f"canonical-schema mismatch: "
                f"binding_db={bd_cols}, nci60={nci_cols}, drugood={do_cols}"
            )
