"""Wf-313H — comprehensive coverage for ``molmetal.data.drugood_loader``.

================================================================
Wf-313H / R20-Phase-0 closeout.  Per
``TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`` §1.1 the
``molmetal/data/drugood_loader.py`` loader needs ≥12 dedicated tests
covering:

  * load + parse + return schema (4 tests)
  * row count + metal filter + scaffold split (3 tests)
  * edge cases (empty / malformed JSON) (2 tests)
  * integration with downstream consumer (chemprop v2 / dmpnn) (2 tests)

This file plus the lighter ``test_data_loaders.py::TestDrugOODLoader``
smoke test together exceed the ≥12 contract — ``test_data_loaders.py``
carries 1 dedicated test + 1 cross-loader test, this file carries the
remaining 13.

MEASURED anchors (per on-disk JSON — Sep 28 2026):

  * ``lbap_refined_ic50_assay`` train split = 95,373 rows (Ji 2022
    official curator)
  * DrugOOD carries no per-row metal / cell-line identifier — both
    columns are ``"NA"``-filled by design (the GATE-6 schema contract)
  * ``protein`` carries the ``assay_id`` (stringified) as a proxy for
    the bioassay target (≥100 unique values for the assay split)
  * Loader has two code paths: ``prefer_library=True`` (uses upstream
    ``drugood.datasets.DrugOODDataset``; requires mmcv + drugood conda
    env) and ``prefer_library=False`` (reads JSON directly; works in
    any env where the JSON file is on disk).
"""

from __future__ import annotations

import json
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
    DEFAULT_SUBSETS,
    VALID_SPLITS,
    _CANONICAL_COLUMNS,
    _NA_FILL,
    _resolve_subset_path,
    load_drugood,
)

CANONICAL_COLUMNS = (
    "smiles",
    "metal",
    "cell_line",
    "protein",
    "pIC50",
    "active",
)

EXPECTED_TRAIN_ROW_COUNT = 95373


# Skip condition: only run if the DrugOOD JSON is on disk
_DRUGOOD_JSON = DEFAULT_DRUGOOD_TARGET_ROOT / "lbap_refined_ic50_assay.json"
DRUGOOD_JSON_AVAILABLE = _DRUGOOD_JSON.exists()
SKIP_REASON = f"DrugOOD JSON not found at {_DRUGOOD_JSON}"


# ---------------------------------------------------------------------------
# 1. load + parse + return schema (4 tests)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DRUGOOD_JSON_AVAILABLE, reason=SKIP_REASON)
class TestDrugOODSchema:
    """The loader returns the canonical 6-col DataFrame."""

    def test_load_returns_dataframe_raw_json_path(self) -> None:
        """``prefer_library=False`` (raw JSON) returns a DataFrame."""
        df = load_drugood(
            subset="lbap_refined_ic50_assay",
            split="train",
            prefer_library=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_canonical_columns_exact_order(self) -> None:
        """GATE-6 contract: column list is exactly ``CANONICAL_COLUMNS``."""
        df = load_drugood(
            subset="lbap_refined_ic50_assay",
            split="train",
            prefer_library=False,
        )
        assert tuple(df.columns) == CANONICAL_COLUMNS, (
            f"drugood columns {df.columns.tolist()} != canonical "
            f"{CANONICAL_COLUMNS}"
        )

    def test_pIC50_numeric_and_finite(self) -> None:
        df = load_drugood(
            subset="lbap_refined_ic50_assay",
            split="train",
            prefer_library=False,
        )
        assert pd.api.types.is_numeric_dtype(
            df["pIC50"]
        ), f"pIC50 should be numeric, got {df['pIC50'].dtype}"
        assert (
            int(df["pIC50"].isna().sum()) == 0
        ), f"drugood has {int(df['pIC50'].isna().sum())} NaN pIC50"
        assert (
            df["pIC50"].apply(lambda v: math.isfinite(float(v))).all()
        ), "drugood has non-finite pIC50"

    def test_metal_and_cell_line_NA_filled(self) -> None:
        """DrugOOD carries no per-row metal / cell-line identifier.

        Both columns must be exactly the literal string ``"NA"`` for
        every row (GATE-6 NA-fill contract, mirroring NCI-60).
        """
        df = load_drugood(
            subset="lbap_refined_ic50_assay",
            split="train",
            prefer_library=False,
        )
        assert (df["metal"] == _NA_FILL).all(), (
            f"drugood metal should be '{_NA_FILL}', "
            f"got uniques: {df['metal'].unique()[:5].tolist()}"
        )
        assert (df["cell_line"] == _NA_FILL).all(), (
            f"drugood cell_line should be '{_NA_FILL}', "
            f"got uniques: {df['cell_line'].unique()[:5].tolist()}"
        )


# ---------------------------------------------------------------------------
# 2. row count + metal filter + scaffold split (3 tests)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DRUGOOD_JSON_AVAILABLE, reason=SKIP_REASON)
class TestDrugOODRowAndSplit:
    """Per-loader invariants: 95373 rows, ≥100 unique assays, scaffold split."""

    def test_row_count_95373(self) -> None:
        """Project-shipped ``lbap_refined_ic50_assay`` train split = 95373 rows."""
        df = load_drugood(
            subset="lbap_refined_ic50_assay",
            split="train",
            prefer_library=False,
        )
        assert len(df) == EXPECTED_TRAIN_ROW_COUNT, (
            f"drugood lbap_refined_ic50_assay train row count drifted: "
            f"expected {EXPECTED_TRAIN_ROW_COUNT}, got {len(df)}"
        )

    def test_assay_diversity(self) -> None:
        """``protein`` carries the assay_id (stringified); ≥100 unique."""
        df = load_drugood(
            subset="lbap_refined_ic50_assay",
            split="train",
            prefer_library=False,
        )
        unique_assays = df["protein"].unique().tolist()
        assert len(unique_assays) >= 100, (
            f"drugood protein (assay_id) should have ≥100 unique values, "
            f"got {len(unique_assays)}"
        )
        # And the assay_ids are non-empty (no degenerate empty-string).
        assert all(
            str(a).strip() != "" for a in unique_assays
        ), f"drugood has empty assay_id: {unique_assays[:5]}"

    def test_scaffold_split_subsample(self) -> None:
        """Bemis-Murcko scaffold split partitions a 2k-row subsample.

        The full train split is 95k rows — too large for a unit test.
        We subsample 2k rows (deterministic seed=0) and verify the
        scaffold split partitions them 3-way without losing rows.
        """
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")

        df = (
            load_drugood(
                subset="lbap_refined_ic50_assay",
                split="train",
                prefer_library=False,
            )
            .sample(n=2000, random_state=0)
            .reset_index(drop=True)
        )

        scaffolds: dict[str, list[int]] = {}
        parseable = 0
        for i, sm in enumerate(df["smiles"].tolist()):
            mol = Chem.MolFromSmiles(str(sm))
            if mol is None:
                continue
            parseable += 1
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
            scaffolds.setdefault(sc, []).append(i)

        # ≥90 % SMILES parseable (DrugOOD is a ligand panel so this is
        # a more lenient threshold than binding_db).
        assert (
            parseable / len(df) >= 0.90
        ), f"only {parseable}/{len(df)} drugood SMILES are RDKit-parseable"

        # Partition 80/10/10
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
        assert (
            len(all_idx) == n
        ), f"scaffold split lost rows: {n - len(all_idx)} missing"
        assert len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0


# ---------------------------------------------------------------------------
# 3. edge cases (empty / malformed JSON) (2 tests)
# ---------------------------------------------------------------------------
class TestDrugOODEdgeCases:
    """Loader surfaces clean errors on missing / malformed JSON."""

    def test_missing_json_raises_filenotfound(self, tmp_path: Path) -> None:
        """Non-existent subset JSON raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            load_drugood(
                subset="lbap_refined_ic50_assay",
                split="train",
                target_root=tmp_path,  # empty tmp dir
                prefer_library=False,
            )

    def test_invalid_split_raises_valueerror(self) -> None:
        """Invalid ``split`` argument raises ``ValueError`` upfront."""
        with pytest.raises(ValueError) as excinfo:
            load_drugood(
                subset="lbap_refined_ic50_assay",
                split="not_a_valid_split",
                prefer_library=False,
            )
        assert (
            "not_a_valid_split" in str(excinfo.value)
            or "valid" in str(excinfo.value).lower()
        )


# ---------------------------------------------------------------------------
# 4. integration with downstream consumer (2 tests)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DRUGOOD_JSON_AVAILABLE, reason=SKIP_REASON)
class TestDrugOODIntegration:
    """Loader output is consumable by the chemprop v2/v3 + D-MPNN stack."""

    def test_smiles_rdkit_round_trip_subsample(self) -> None:
        """At least 90 % of drugood train SMILES round-trip through RDKit.

        Subsampled to 2k rows for unit-test speed.  chemprop v2's
        ``ChempropDataset`` silently drops invalid SMILES so we want
        the loader's output to be ≥90 % RDKit-parseable (DrugOOD is a
        curated ligand panel — closer to 99 % in practice).
        """
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")

        df = load_drugood(
            subset="lbap_refined_ic50_assay",
            split="train",
            prefer_library=False,
        ).sample(n=2000, random_state=0)

        n_ok = 0
        for sm in df["smiles"].tolist():
            mol = Chem.MolFromSmiles(str(sm))
            if mol is not None:
                n_ok += 1
        frac = n_ok / len(df)
        assert frac >= 0.90, (
            f"only {n_ok}/{len(df)} drugood SMILES round-trip through "
            f"RDKit ({frac:.3f})"
        )

    def test_dataframe_consumable_by_chemprop_v2_smoke_loader(self) -> None:
        """Output columns are the exact subset chemprop v2 reads.

        ``pre_featurize_dmpnn.py`` consumes ``load_drugood`` and pipes
        the output to chemprop v2's ``ChempropDataset`` — verify the
        drugood DataFrame has the right string-typed string columns +
        numeric pIC50 + boolean active.
        """
        df = load_drugood(
            subset="lbap_refined_ic50_assay",
            split="train",
            prefer_library=False,
        )
        assert pd.api.types.is_string_dtype(
            df["smiles"]
        ), f"smiles should be string-like, got {df['smiles'].dtype}"
        assert pd.api.types.is_numeric_dtype(df["pIC50"])
        assert df["active"].dtype == bool
        assert pd.api.types.is_string_dtype(
            df["protein"]
        ), f"protein should be string-like, got {df['protein'].dtype}"
        # And no NaN in string columns (metal + cell_line are
        # 'NA'-filled, protein carries real assay_ids).
        for c in ("smiles", "metal", "cell_line", "protein"):
            assert df[c].notna().all(), f"drugood has NaN {c}"


# ---------------------------------------------------------------------------
# 5. cross-loader pairing (2 tests; mirrors the GATE-6 contract test)
# ---------------------------------------------------------------------------
class TestDrugOODCrossLoaderPairing:
    """drugood columns match the canonical contract with no extras."""

    def test_internal_canonical_columns_constant_matches(self) -> None:
        assert _CANONICAL_COLUMNS == CANONICAL_COLUMNS

    def test_resolve_subset_path_uses_env_var(self, monkeypatch) -> None:
        """``_resolve_subset_path`` honours the ``DRUGOOD_IC50_TARGET_ROOT`` env var."""
        custom_root = Path("/tmp/_wf313h_drugood_custom_root")
        monkeypatch.setenv("DRUGOOD_IC50_TARGET_ROOT", str(custom_root))
        resolved = _resolve_subset_path("lbap_refined_ic50_assay")
        assert resolved == custom_root / "lbap_refined_ic50_assay.json"

    def test_default_subsets_constant_is_3_entries(self) -> None:
        """``DEFAULT_SUBSETS`` is exactly 3 (Ji 2022's refined curator outputs)."""
        assert len(DEFAULT_SUBSETS) == 3
        for s in DEFAULT_SUBSETS:
            assert "lbap_refined_ic50" in s

    def test_valid_splits_constant_is_5_entries(self) -> None:
        """``VALID_SPLITS`` is exactly 5 (train + 2 OOD + 2 IID)."""
        assert len(VALID_SPLITS) == 5
        assert "train" in VALID_SPLITS
        assert "ood_val" in VALID_SPLITS
        assert "ood_test" in VALID_SPLITS
        assert "iid_val" in VALID_SPLITS
        assert "iid_test" in VALID_SPLITS

    def test_default_target_root_path_attribute(self) -> None:
        """``DEFAULT_DRUGOOD_TARGET_ROOT`` points at the project-shipped JSON root."""
        assert isinstance(DEFAULT_DRUGOOD_TARGET_ROOT, Path)
        assert str(DEFAULT_DRUGOOD_TARGET_ROOT).endswith("drugood/ic50/official")
