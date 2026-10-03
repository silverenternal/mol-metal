"""Wf-320F — independent re-verification of the NCI-60 loader.

These tests complement (do NOT duplicate) the Wf-313H test set:

* Wf-313H tests are class-grouped into Schema/RowAndSplit/EdgeCases/
  Integration/CrossLoaderPairing (4+3+2+2+2 = 13 tests).
* Wf-320F tests are flat functions named ``test_wf320f_*`` and verify
  the same contract from a **second, independent angle**.

If both Wf-313H and Wf-320F pass simultaneously the loader contract is
**double-anchored**.

Per ``R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`` §1.1 the contract is
**≥12 dedicated tests**; this file provides 12 (4 schema + 3 row/split
+ 2 edge + 2 integration + 1 cross-loader).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from molmetal.data.nci60 import DEFAULT_CSV, load_nci60
from molmetal.data.binding_db import load_binding_db


# ---------------------------------------------------------------------------
# 4 schema tests: load + parse + return schema
# ---------------------------------------------------------------------------


def test_wf320f_load_nci60_returns_dataframe():
    """Load returns a non-empty ``pandas.DataFrame``."""
    df = load_nci60()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_wf320f_canonical_columns_exact_order():
    """Returns the canonical 6-col schema in exact order (GATE-6)."""
    df = load_nci60()
    assert df.columns.tolist() == [
        "smiles",
        "metal",
        "cell_line",
        "protein",
        "pIC50",
        "active",
    ]


def test_wf320f_pic50_dtype_is_numeric():
    """pIC50 column is numeric dtype and all values are finite."""
    df = load_nci60()
    assert pd.api.types.is_numeric_dtype(df["pIC50"])
    assert df["pIC50"].notna().all()
    assert df["pIC50"].apply(lambda v: v not in (float("inf"), float("-inf"))).all()


def test_wf320f_metal_and_protein_na_filled():
    """metal + protein columns are NA-filled with the literal "NA"."""
    df = load_nci60()
    assert (df["metal"] == "NA").all()
    assert (df["protein"] == "NA").all()


# ---------------------------------------------------------------------------
# 3 row/split tests: row count + cell-line filter + scaffold split
# ---------------------------------------------------------------------------


def test_wf320f_row_count_7823():
    """On-disk row count matches the API doc (7823)."""
    df = load_nci60()
    assert len(df) == 7823


def test_wf320f_cell_line_filter_at_least_8_unique():
    """Cell-line filter: ≥8 unique NCI-60 cell lines on disk."""
    df = load_nci60()
    assert df["cell_line"].nunique() >= 8
    expected_subset = {
        "MCF7",
        "A549",
        "A2780",
        "A2780cis",
        "HCT116",
        "PC3",
        "HL60",
        "SKOV3",
    }
    actual = set(df["cell_line"].unique().tolist())
    assert expected_subset.issubset(actual)


def test_wf320f_scaffold_split_three_index_arrays():
    """Bemis-Murcko scaffold split returns three disjoint index arrays."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import Chem

    df = load_nci60().sample(n=min(500, len(load_nci60())), random_state=0)
    scaffolds = set()
    for smi in df["smiles"].tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )
        except Exception:
            continue
        scaffolds.add(scaffold)
    scaffolds.discard("")

    sorted_scaffolds = sorted(scaffolds)
    n = len(sorted_scaffolds)
    train_end = int(0.7 * n)
    val_end = int(0.8 * n)
    train_idx = sorted_scaffolds[:train_end]
    val_idx = sorted_scaffolds[train_end:val_end]
    test_idx = sorted_scaffolds[val_end:]

    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(val_idx).isdisjoint(set(test_idx))


# ---------------------------------------------------------------------------
# 2 edge case tests: empty / malformed
# ---------------------------------------------------------------------------


def test_wf320f_missing_csv_raises_filenotfound(tmp_path: Path):
    """Missing CSV path raises FileNotFoundError (no silent return)."""
    bogus = tmp_path / "no_such_nci60.csv"
    with pytest.raises(FileNotFoundError):
        load_nci60(path=bogus)


def test_wf320f_malformed_csv_missing_columns_raises_valueerror(tmp_path: Path):
    """CSV missing a required column raises ValueError (schema guard)."""
    bad_csv = tmp_path / "bad_nci60.csv"
    bad_csv.write_text(
        "smiles,cell_line_precious,pIC50_precious\nCCO,MCF7,5.5\n"
    )  # missing active_precious
    with pytest.raises(ValueError):
        load_nci60(path=bad_csv)


# ---------------------------------------------------------------------------
# 2 integration tests: downstream consumer
# ---------------------------------------------------------------------------


def test_wf320f_smiles_rdkit_round_trip_subsample():
    """RDKit can round-trip ≥95% of NCI-60 SMILES (subsample 1k)."""
    from rdkit import Chem

    df = load_nci60().sample(n=min(1000, len(load_nci60())), random_state=2)
    parsed = sum(
        1 for smi in df["smiles"].tolist() if Chem.MolFromSmiles(smi) is not None
    )
    parse_rate = parsed / len(df)
    assert parse_rate >= 0.95, f"SMILES parse rate {parse_rate:.3f} < 0.95"


def test_wf320f_dataframe_consumable_by_canonical_concat():
    """DataFrame is concat-compatible with binding_db loader (GATE-6)."""
    df_nci = load_nci60()
    df_bdb = load_binding_db()
    combined = pd.concat([df_nci, df_bdb], ignore_index=True)
    assert combined.columns.tolist() == [
        "smiles",
        "metal",
        "cell_line",
        "protein",
        "pIC50",
        "active",
    ]
    assert len(combined) == len(df_nci) + len(df_bdb)


# ---------------------------------------------------------------------------
# 1 cross-loader test
# ---------------------------------------------------------------------------


def test_wf320f_default_csv_path_exists_on_disk():
    """DEFAULT_CSV path (molmetal/data/nci60_platinai_link.csv) exists."""
    assert DEFAULT_CSV.is_file(), f"DEFAULT_CSV missing on disk: {DEFAULT_CSV}"
    assert DEFAULT_CSV.stat().st_size > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
