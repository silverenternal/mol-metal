"""Wf-313H — comprehensive coverage for ``molmetal.data.nci60``.

================================================================
Wf-313H / R20-Phase-0 closeout.  Per
``TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`` §1.1 the
``molmetal/data/nci60.py`` loader needs ≥12 dedicated tests
covering:

  * load + parse + return schema (4 tests)
  * row count + metal filter + scaffold split (3 tests)
  * edge cases (empty / malformed CSV) (2 tests)
  * integration with downstream consumer (chemprop v2 / dmpnn) (2 tests)

This file plus the lighter ``test_data_loaders.py::TestNCI60Loader``
smoke test together exceed the ≥12 contract — ``test_data_loaders.py``
carries 1 dedicated test + 1 cross-loader test, this file carries the
remaining 13.

MEASURED anchors (per on-disk CSV — Sep 28 2026):

  * 7,823 rows (compound × NCI-60 cell-line pairs)
  * source columns ``cell_line_precious`` / ``pIC50_precious`` /
    ``active_precious`` → normalised to canonical
    ``cell_line / pIC50 / active``
  * ``metal`` and ``protein`` are filled with literal ``"NA"`` (NCI-60
    is a cell-line panel, not a target panel; carries no per-row metal
    or protein identifier)
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

from molmetal.data.nci60 import (  # noqa: E402
    DEFAULT_CSV,
    _NA_FILL,
    _REQUIRED_COLUMNS,
    load_nci60,
)

CANONICAL_COLUMNS = (
    "smiles",
    "metal",
    "cell_line",
    "protein",
    "pIC50",
    "active",
)

EXPECTED_ROW_COUNT = 7823


# ---------------------------------------------------------------------------
# 1. load + parse + return schema (4 tests)
# ---------------------------------------------------------------------------
class TestNCI60Schema:
    """The loader returns the canonical 6-col DataFrame."""

    def test_load_returns_dataframe(self) -> None:
        df = load_nci60()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_canonical_columns_exact_order(self) -> None:
        """GATE-6 contract: column list is exactly ``CANONICAL_COLUMNS``."""
        df = load_nci60()
        assert tuple(df.columns) == CANONICAL_COLUMNS, (
            f"nci60 columns {df.columns.tolist()} != canonical " f"{CANONICAL_COLUMNS}"
        )

    def test_pIC50_numeric_and_finite(self) -> None:
        df = load_nci60()
        assert pd.api.types.is_numeric_dtype(
            df["pIC50"]
        ), f"pIC50 should be numeric, got {df['pIC50'].dtype}"
        assert (
            int(df["pIC50"].isna().sum()) == 0
        ), f"nci60 has {int(df['pIC50'].isna().sum())} NaN pIC50"
        assert (
            df["pIC50"].apply(lambda v: math.isfinite(float(v))).all()
        ), "nci60 has non-finite pIC50"

    def test_metal_and_protein_NA_filled(self) -> None:
        """NCI-60 link CSV carries no per-row metal / protein identifier.

        Both columns must be exactly the literal string ``"NA"`` for
        every row — this is the documented Wf-130 NA-fill contract.
        """
        df = load_nci60()
        assert (df["metal"] == _NA_FILL).all(), (
            f"nci60 metal should be '{_NA_FILL}', "
            f"got uniques: {df['metal'].unique()[:5].tolist()}"
        )
        assert (df["protein"] == _NA_FILL).all(), (
            f"nci60 protein should be '{_NA_FILL}', "
            f"got uniques: {df['protein'].unique()[:5].tolist()}"
        )


# ---------------------------------------------------------------------------
# 2. row count + metal filter + scaffold split (3 tests)
# ---------------------------------------------------------------------------
class TestNCI60RowAndSplit:
    """Per-loader invariants: 7823 rows, ≥3 unique cell lines, scaffold split."""

    def test_row_count_7823(self) -> None:
        """Project-shipped CSV carries exactly 7,823 rows (MEASURED Sep 28)."""
        df = load_nci60()
        assert len(df) == EXPECTED_ROW_COUNT, (
            f"nci60 row count drifted: expected {EXPECTED_ROW_COUNT}, " f"got {len(df)}"
        )

    def test_cell_line_diversity(self) -> None:
        """≥3 unique NCI-60 cell-line identifiers (non-trivial panel)."""
        df = load_nci60()
        unique_cells = df["cell_line"].unique().tolist()
        assert (
            len(unique_cells) >= 3
        ), f"nci60 cell_line should have ≥3 unique lines, got {unique_cells}"
        # And each cell line has ≥10 rows (real cell-line panel, not a
        # single-cell artefact).
        per_cell = df["cell_line"].value_counts()
        for cl in unique_cells:
            assert (
                int(per_cell.get(cl, 0)) >= 10
            ), f"cell_line {cl} has only {int(per_cell.get(cl, 0))} rows"

    def test_scaffold_split_returns_three_index_arrays(self) -> None:
        """Bemis-Murcko scaffold split partitions indices 3-way."""
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")

        df = load_nci60()
        scaffolds: dict[str, list[int]] = {}
        parseable = 0
        for i, sm in enumerate(df["smiles"].tolist()):
            mol = Chem.MolFromSmiles(str(sm))
            if mol is None:
                continue
            parseable += 1
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
            scaffolds.setdefault(sc, []).append(i)

        # ≥95 % SMILES parseable
        assert (
            parseable / len(df) >= 0.95
        ), f"only {parseable}/{len(df)} SMILES are RDKit-parseable"

        # Partition 80/10/10 in deterministic order
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

        # 3-way partition covers the full row range, no overlap.
        all_idx = set(train_idx) | set(val_idx) | set(test_idx)
        assert (
            len(all_idx) == n
        ), f"scaffold split lost rows: {n - len(all_idx)} missing"
        assert len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0, (
            f"empty partition: train={len(train_idx)} val={len(val_idx)} "
            f"test={len(test_idx)}"
        )


# ---------------------------------------------------------------------------
# 3. edge cases (empty / malformed CSV) (2 tests)
# ---------------------------------------------------------------------------
class TestNCI60EdgeCases:
    """Loader surfaces clean errors on missing / malformed inputs."""

    def test_missing_csv_raises_filenotfound(self, tmp_path: Path) -> None:
        """Non-existent path raises ``FileNotFoundError`` (no silent empty)."""
        bogus = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            load_nci60(path=bogus)

    def test_malformed_csv_missing_columns_raises_valueerror(
        self, tmp_path: Path
    ) -> None:
        """CSV missing required canonical columns raises ``ValueError``.

        NCI-60 loader normalises ``cell_line_precious`` /
        ``pIC50_precious`` / ``active_precious`` → canonical names.  A
        CSV with no recognisable source columns must raise
        ``ValueError`` with a message that names the missing canonical
        columns.
        """
        bad = tmp_path / "nci60_bad.csv"
        # Empty-ish: only `smiles_precious` (NOT a recognised source
        # column) so no canonical column is present after the rename.
        bad.write_text("smiles_precious\nCCO\n")
        with pytest.raises(ValueError) as excinfo:
            load_nci60(path=bad)
        msg = str(excinfo.value)
        # The error should list the missing canonical columns.
        assert any(
            c in msg for c in CANONICAL_COLUMNS
        ), f"error should list missing canonical columns: {msg}"


# ---------------------------------------------------------------------------
# 4. integration with downstream consumer (2 tests)
# ---------------------------------------------------------------------------
class TestNCI60Integration:
    """Loader output is consumable by the chemprop v2/v3 + D-MPNN stack."""

    def test_smiles_rdkit_round_trip(self) -> None:
        """At least 95 % of nci60 SMILES round-trip through RDKit."""
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")

        df = load_nci60()
        n_ok = 0
        for sm in df["smiles"].tolist():
            mol = Chem.MolFromSmiles(str(sm))
            if mol is not None:
                n_ok += 1
        frac = n_ok / len(df)
        assert frac >= 0.95, (
            f"only {n_ok}/{len(df)} nci60 SMILES round-trip through "
            f"RDKit ({frac:.3f}); chemprop v2 expects ≥0.95"
        )

    def test_dataframe_consumable_by_chemprop_v2_smoke_loader(self) -> None:
        """Output columns are the exact subset chemprop v2 reads.

        ``pre_featurize_dmpnn.py`` consumes ``load_nci60`` and pipes
        the output to chemprop v2's ``ChempropDataset`` — verify the
        nci60 DataFrame has the right string-typed string columns +
        numeric pIC50 + boolean active.
        """
        df = load_nci60()
        assert pd.api.types.is_string_dtype(
            df["smiles"]
        ), f"smiles should be string-like, got {df['smiles'].dtype}"
        assert pd.api.types.is_numeric_dtype(df["pIC50"])
        assert df["active"].dtype == bool
        assert pd.api.types.is_string_dtype(
            df["cell_line"]
        ), f"cell_line should be string-like, got {df['cell_line'].dtype}"
        # And no NaN in string columns (cell_line carries real NCI-60
        # identifiers; metal / protein are 'NA'-filled).
        for c in ("smiles", "metal", "cell_line", "protein"):
            assert df[c].notna().all(), f"nci60 has NaN {c}"


# ---------------------------------------------------------------------------
# 5. cross-loader pairing (2 tests; mirrors the GATE-6 contract test)
# ---------------------------------------------------------------------------
class TestNCI60CrossLoaderPairing:
    """nci60 columns match the canonical contract with no extras."""

    def test_internal_required_columns_constant_matches_canonical(self) -> None:
        assert _REQUIRED_COLUMNS == CANONICAL_COLUMNS

    def test_default_csv_path_exists_on_disk(self) -> None:
        """``DEFAULT_CSV`` points at the project-shipped CSV."""
        assert DEFAULT_CSV.exists(), f"DEFAULT_CSV {DEFAULT_CSV} does not exist on disk"
        # ≥7000 rows minimum (the 7823-row project-shipped subset).
        n_lines = sum(1 for _ in open(DEFAULT_CSV, "r", encoding="utf-8"))
        assert n_lines >= 7000, f"DEFAULT_CSV has only {n_lines} lines; expected ≥7000"
