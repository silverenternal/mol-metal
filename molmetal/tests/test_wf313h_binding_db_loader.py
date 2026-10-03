"""Wf-313H — comprehensive coverage for ``molmetal.data.binding_db``.

================================================================
Wf-313H / R20-Phase-0 closeout.  Per
``TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`` §1.1 the
``molmetal/data/binding_db.py`` loader needs ≥12 dedicated tests
covering:

  * load + parse + return schema (4 tests)
  * row count + metal filter + scaffold split (3 tests)
  * edge cases (empty / malformed CSV) (2 tests)
  * integration with downstream consumer (chemprop v2 / dmpnn) (2 tests)

This file plus the lighter ``test_data_loaders.py::TestBindingDBLoader``
smoke test together exceed the ≥12 contract — ``test_data_loaders.py``
carries 1 dedicated test + 1 cross-loader test, this file carries the
remaining 13 (4 schema + 3 counts/scaffold + 2 edge + 2 integration +
2 cross-loader pairing).  The two files together = 14+ tests
covering :func:`load_binding_db`.

MEASURED anchors (per on-disk CSV — Sep 28 2026):

  * 5,257 rows
  * 5 d-block metals (Ru / Ir / Re / Os / Rh)
  * canonical 6-col schema: ``smiles, metal, cell_line, protein, pIC50,
    active``
  * active is boolean; pIC50 is finite float
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

from molmetal.data.binding_db import (  # noqa: E402
    DEFAULT_CSV,
    _REQUIRED_COLUMNS,
    load_binding_db,
)

CANONICAL_COLUMNS = (
    "smiles",
    "metal",
    "cell_line",
    "protein",
    "pIC50",
    "active",
)

EXPECTED_METALS = ("Ru", "Ir", "Re", "Os", "Rh")
EXPECTED_ROW_COUNT = 5257


# ---------------------------------------------------------------------------
# 1. load + parse + return schema (4 tests)
# ---------------------------------------------------------------------------
class TestBindingDBSchema:
    """The loader returns the canonical 6-col DataFrame."""

    def test_load_returns_dataframe(self) -> None:
        df = load_binding_db()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_canonical_columns_exact_order(self) -> None:
        """GATE-6 contract: column list is exactly ``CANONICAL_COLUMNS``."""
        df = load_binding_db()
        assert tuple(df.columns) == CANONICAL_COLUMNS, (
            f"binding_db columns {df.columns.tolist()} != canonical "
            f"{CANONICAL_COLUMNS}"
        )

    def test_pIC50_numeric_and_finite(self) -> None:
        df = load_binding_db()
        assert pd.api.types.is_numeric_dtype(
            df["pIC50"]
        ), f"pIC50 should be numeric, got {df['pIC50'].dtype}"
        # No NaN, no inf
        assert (
            int(df["pIC50"].isna().sum()) == 0
        ), f"binding_db has {int(df['pIC50'].isna().sum())} NaN pIC50"
        assert (
            df["pIC50"].apply(lambda v: math.isfinite(float(v))).all()
        ), "binding_db has non-finite pIC50"

    def test_active_is_boolean(self) -> None:
        df = load_binding_db()
        assert (
            df["active"].dtype == bool
        ), f"active should be bool, got {df['active'].dtype}"
        # And it's a sensible binary (no NaN)
        assert df["active"].notna().all()


# ---------------------------------------------------------------------------
# 2. row count + metal filter + scaffold split (3 tests)
# ---------------------------------------------------------------------------
class TestBindingDBRowAndSplit:
    """Per-loader invariants: 5257 rows, 5 d-block metals, scaffold split."""

    def test_row_count_5257(self) -> None:
        """Project-shipped CSV carries exactly 5,257 rows (MEASURED Sep 28)."""
        df = load_binding_db()
        assert len(df) == EXPECTED_ROW_COUNT, (
            f"binding_db row count drifted: expected {EXPECTED_ROW_COUNT}, "
            f"got {len(df)}"
        )

    def test_metal_filter_5_metals(self) -> None:
        """The metal column is exactly the 5 d-block metals from §1.1 plan.

        Filters out empty strings + verifies the canonical set.
        """
        df = load_binding_db()
        unique_metals = sorted(df["metal"].unique().tolist())
        assert unique_metals == sorted(
            EXPECTED_METALS
        ), f"binding_db metals {unique_metals} != expected {sorted(EXPECTED_METALS)}"

        # Per-metal row count is non-trivial (≥10) — no degenerate single-
        # metal subset.
        per_metal = df["metal"].value_counts()
        for m in EXPECTED_METALS:
            assert int(per_metal.get(m, 0)) >= 10, (
                f"metal {m} has only {int(per_metal.get(m, 0))} rows "
                f"(too few for downstream training)"
            )

    def test_scaffold_split_returns_three_index_arrays(self) -> None:
        """Scaffold split produces train/val/test integer index arrays.

        Uses Bemis-Murcko scaffolds via RDKit (same algorithm as
        :class:`molmetal.data.splits.ScaffoldSplitter`).  The loader does
        not expose the splitter directly — this test consumes the loader
        output and applies the scaffold grouping itself, verifying that
        the loader's SMILES column is RDKit-parseable for ≥95 % of rows
        AND that a Bemis-Murcko split partitions the indices into 3
        disjoint sets covering the full row range.
        """
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")

        df = load_binding_db()
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

        # Now partition: sorted by scaffold size (descending) and bucket
        # 80/10/10 in deterministic order.  Verifies the loader output
        # is consumable by an OOD-style scaffold split.
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
class TestBindingDBEdgeCases:
    """Loader surfaces clean errors on missing / malformed inputs."""

    def test_missing_csv_raises_filenotfound(self, tmp_path: Path) -> None:
        """Non-existent path raises ``FileNotFoundError`` (no silent empty)."""
        bogus = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            load_binding_db(path=bogus)

    def test_malformed_csv_missing_columns_raises_valueerror(
        self, tmp_path: Path
    ) -> None:
        """CSV missing required canonical columns raises ``ValueError``."""
        bad = tmp_path / "binding_db_bad.csv"
        bad.write_text("smiles,metal,protein,pIC50\nCCO,Ru,XXX,5.0\n")
        with pytest.raises(ValueError) as excinfo:
            load_binding_db(path=bad)
        msg = str(excinfo.value)
        # Error message names the missing columns
        assert (
            "cell_line" in msg or "active" in msg
        ), f"error should list missing required columns: {msg}"


# ---------------------------------------------------------------------------
# 4. integration with downstream consumer (2 tests)
# ---------------------------------------------------------------------------
class TestBindingDBIntegration:
    """Loader output is consumable by the chemprop v2/v3 + D-MPNN stack."""

    def test_smiles_rdkit_round_trip(self) -> None:
        """At least 95 % of binding_db SMILES round-trip through RDKit.

        This is the contract chemprop v2's ``ChempropDataset`` assumes:
        invalid SMILES are silently dropped, so we want the loader to
        return SMILES that survive RDKit sanitisation.
        """
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")

        df = load_binding_db()
        n_ok = 0
        for sm in df["smiles"].tolist():
            mol = Chem.MolFromSmiles(str(sm))
            if mol is not None:
                n_ok += 1
        frac = n_ok / len(df)
        assert frac >= 0.95, (
            f"only {n_ok}/{len(df)} binding_db SMILES round-trip through "
            f"RDKit ({frac:.3f}); chemprop v2 expects ≥0.95"
        )

    def test_dataframe_consumable_by_chemprop_v2_smoke_loader(self) -> None:
        """Output columns are the exact subset chemprop v2 reads.

        ``train_chemprop_v2_v3.py`` and ``pre_featurize_dmpnn.py`` both
        pull ``['smiles', 'pIC50', 'metal']`` from any canonical
        loader.  We verify the loader exposes those columns with the
        right types so the chemprop v2 CSV→DataFrame→Dataset path is
        a no-op for binding_db rows.
        """
        df = load_binding_db()
        # chemprop v2 reads 'smiles' as str, 'pIC50' as float.
        assert pd.api.types.is_string_dtype(
            df["smiles"]
        ), f"smiles should be string-like, got {df['smiles'].dtype}"
        assert pd.api.types.is_numeric_dtype(df["pIC50"])
        # 'metal' column is read for per-metal task conditioning.
        assert pd.api.types.is_string_dtype(
            df["metal"]
        ), f"metal should be string-like, got {df['metal'].dtype}"
        # 'cell_line' is used as an auxiliary label by chemprop v2's
        # multi-task head (NOT a per-row task; consumed as a string
        # feature).
        assert pd.api.types.is_string_dtype(
            df["cell_line"]
        ), f"cell_line should be string-like, got {df['cell_line'].dtype}"
        assert pd.api.types.is_string_dtype(
            df["protein"]
        ), f"protein should be string-like, got {df['protein'].dtype}"
        # And no NaN cell_line / metal / protein / smiles
        for c in ("smiles", "metal", "cell_line", "protein"):
            assert df[c].notna().all(), f"binding_db has NaN {c}"


# ---------------------------------------------------------------------------
# 5. cross-loader pairing (2 tests; mirrors the GATE-6 contract test)
# ---------------------------------------------------------------------------
class TestBindingDBCrossLoaderPairing:
    """binding_db columns match the canonical contract with no extras."""

    def test_internal_required_columns_constant_matches_canonical(self) -> None:
        assert _REQUIRED_COLUMNS == CANONICAL_COLUMNS

    def test_default_csv_path_exists_on_disk(self) -> None:
        """``DEFAULT_CSV`` points at the project-shipped CSV."""
        assert DEFAULT_CSV.exists(), f"DEFAULT_CSV {DEFAULT_CSV} does not exist on disk"
        # ≥5000 rows minimum (the 5257-row project-shipped subset).
        n_lines = sum(1 for _ in open(DEFAULT_CSV, "r", encoding="utf-8"))
        assert n_lines >= 5000, f"DEFAULT_CSV has only {n_lines} lines; expected ≥5000"
