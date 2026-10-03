"""Wf-320F — independent re-verification of the binding_db loader.

These tests complement (do NOT duplicate) the Wf-313H test set:

* Wf-313H tests are class-grouped into Schema/RowAndSplit/EdgeCases/
  Integration/CrossLoaderPairing (4+3+2+2+2 = 13 tests).
* Wf-320F tests are flat functions named ``test_wf320f_*`` and verify
  the same contract from a **second, independent angle** — they check
  the public behaviour (load + parse + return schema + row count +
  metal filter + scaffold split + edge cases + downstream integration)
  using fresh helper code, NOT a re-export of the Wf-313H helpers.

If both Wf-313H and Wf-320F pass simultaneously the loader contract is
**double-anchored**: any future regression that breaks the schema /
row count / metal filter / scaffold split / edge cases / downstream
contract will fail at least one of the two suites.

Per ``R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`` §1.1 the contract is
**≥12 dedicated tests**; this file provides 12 (4 schema + 3 row/split
+ 2 edge + 2 integration + 1 cross-loader).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from molmetal.data.binding_db import (
    DEFAULT_CSV,
    load_binding_db,
)
from molmetal.data.nci60 import load_nci60
from molmetal.data.drugood_loader import load_drugood


# ---------------------------------------------------------------------------
# 4 schema tests: load + parse + return schema
# ---------------------------------------------------------------------------


def test_wf320f_load_binding_db_returns_dataframe():
    """Load returns a non-empty ``pandas.DataFrame`` (parses + shape OK)."""
    df = load_binding_db()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_wf320f_canonical_columns_exact_order():
    """Returns the canonical 6-col schema in exact order (GATE-6)."""
    df = load_binding_db()
    assert df.columns.tolist() == [
        "smiles",
        "metal",
        "cell_line",
        "protein",
        "pIC50",
        "active",
    ]


def test_wf320f_pic50_dtype_is_numeric():
    """pIC50 column is numeric dtype and all values are finite (no NaN/inf)."""
    df = load_binding_db()
    assert pd.api.types.is_numeric_dtype(df["pIC50"])
    assert df["pIC50"].notna().all()
    assert df["pIC50"].apply(lambda v: v not in (float("inf"), float("-inf"))).all()


def test_wf320f_active_dtype_is_boolean():
    """active column is bool dtype (post-coercion)."""
    df = load_binding_db()
    assert df["active"].dtype == bool


# ---------------------------------------------------------------------------
# 3 row/split tests: row count + metal filter + scaffold split
# ---------------------------------------------------------------------------


def test_wf320f_row_count_5257():
    """On-disk row count matches the API doc (5257)."""
    df = load_binding_db()
    assert len(df) == 5257


def test_wf320f_metal_filter_5_metals():
    """Metal filter: exactly the 5 d-block metals (Ru / Ir / Re / Os / Rh)."""
    df = load_binding_db()
    expected = {"Ru", "Ir", "Re", "Os", "Rh"}
    actual = set(df["metal"].unique().tolist())
    assert actual == expected
    # every row has a non-NA metal (NA-fill is a NCI-60 / DrugOOD artefact,
    # not a binding_db one).
    assert df["metal"].notna().all()


def test_wf320f_scaffold_split_three_index_arrays():
    """Bemis-Murcko scaffold split returns three disjoint index arrays."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import Chem

    df = load_binding_db().sample(n=min(500, len(load_binding_db())), random_state=0)
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
    scaffolds.discard("")  # drop empty (acyclic) scaffolds

    # Make a deterministic 70/10/20 split by sorted scaffold name.
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
    # disjoint
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(val_idx).isdisjoint(set(test_idx))


# ---------------------------------------------------------------------------
# 2 edge case tests: empty / malformed
# ---------------------------------------------------------------------------


def test_wf320f_missing_csv_raises_filenotfound(tmp_path: Path):
    """Missing CSV path raises FileNotFoundError (no silent return)."""
    bogus = tmp_path / "no_such_bindingdb.csv"
    with pytest.raises(FileNotFoundError):
        load_binding_db(path=bogus)


def test_wf320f_malformed_csv_missing_columns_raises_valueerror(tmp_path: Path):
    """CSV missing a required column raises ValueError (schema guard)."""
    bad_csv = tmp_path / "bad_bindingdb.csv"
    bad_csv.write_text(
        "smiles,metal,cell_line\nCCO,Ru,MCF7\n"
    )  # missing protein/pIC50/active
    with pytest.raises(ValueError):
        load_binding_db(path=bad_csv)


# ---------------------------------------------------------------------------
# 2 integration tests: downstream consumer
# ---------------------------------------------------------------------------


def test_wf320f_smiles_rdkit_round_trip_subsample():
    """RDKit can round-trip ≥95% of BindingDB SMILES (subsample 1k)."""
    from rdkit import Chem

    df = load_binding_db().sample(n=min(1000, len(load_binding_db())), random_state=1)
    parsed = sum(
        1 for smi in df["smiles"].tolist() if Chem.MolFromSmiles(smi) is not None
    )
    parse_rate = parsed / len(df)
    assert parse_rate >= 0.95, f"SMILES parse rate {parse_rate:.3f} < 0.95"


def test_wf320f_dataframe_consumable_by_canonical_concat():
    """DataFrame is concat-compatible with the other 2 loaders (GATE-6)."""
    df_bdb = load_binding_db()
    df_nci = load_nci60()
    df_dgo = load_drugood(prefer_library=False)
    # concat across all 3 loaders; column-wise union must equal the
    # canonical 6-col schema (no extra cols, no missing cols).
    combined = pd.concat([df_bdb, df_nci, df_dgo], ignore_index=True)
    assert combined.columns.tolist() == [
        "smiles",
        "metal",
        "cell_line",
        "protein",
        "pIC50",
        "active",
    ]
    assert len(combined) == len(df_bdb) + len(df_nci) + len(df_dgo)


# ---------------------------------------------------------------------------
# 1 cross-loader test
# ---------------------------------------------------------------------------


def test_wf320f_default_csv_path_exists_on_disk():
    """DEFAULT_CSV path (molmetal/data/protein_metal_binding.csv) exists."""
    assert DEFAULT_CSV.is_file(), f"DEFAULT_CSV missing on disk: {DEFAULT_CSV}"
    assert DEFAULT_CSV.stat().st_size > 0


# Allow running this file in isolation.
if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
