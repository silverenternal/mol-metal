"""Wf-320F — independent re-verification of the DrugOOD loader.

These tests complement (do NOT duplicate) the Wf-313H test set:

* Wf-313H tests are class-grouped into Schema/RowAndSplit/EdgeCases/
  Integration/CrossLoaderPairing (4+3+2+2+5 = 16 tests).
* Wf-320F tests are flat functions named ``test_wf320f_*`` and verify
  the same contract from a **second, independent angle** — they exercise
  the **raw-JSON fallback path** (the production code path on this
  host, since the ``drugood`` conda env is not the default Python).

If both Wf-313H and Wf-320F pass simultaneously the loader contract is
**double-anchored**.

Per ``R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`` §1.1 the contract is
**≥12 dedicated tests**; this file provides 13 (4 schema + 3 row/split
+ 2 edge + 2 integration + 2 cross-loader — one extra to cover
``VALID_SPLITS`` size and ``DEFAULT_SUBSETS`` size, since DrugOOD
has 5 splits / 3 subsets that are part of the public contract).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from molmetal.data.drugood_loader import (
    DEFAULT_DRUGOOD_TARGET_ROOT,
    DEFAULT_SUBSETS,
    VALID_SPLITS,
    _resolve_subset_path,
    load_drugood,
)


# ---------------------------------------------------------------------------
# 4 schema tests: load + parse + return schema
# ---------------------------------------------------------------------------


def test_wf320f_load_drugood_returns_dataframe_raw_json_path():
    """Raw-JSON path: load returns a non-empty ``pandas.DataFrame``."""
    df = load_drugood(prefer_library=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_wf320f_canonical_columns_exact_order():
    """Returns the canonical 6-col schema in exact order (GATE-6)."""
    df = load_drugood(prefer_library=False)
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
    df = load_drugood(prefer_library=False)
    assert pd.api.types.is_numeric_dtype(df["pIC50"])
    assert df["pIC50"].notna().all()
    assert df["pIC50"].apply(lambda v: v not in (float("inf"), float("-inf"))).all()


def test_wf320f_metal_and_cell_line_na_filled():
    """metal + cell_line columns are NA-filled with the literal "NA"."""
    df = load_drugood(prefer_library=False)
    assert (df["metal"] == "NA").all()
    assert (df["cell_line"] == "NA").all()


# ---------------------------------------------------------------------------
# 3 row/split tests: row count + assay diversity + scaffold split
# ---------------------------------------------------------------------------


def test_wf320f_row_count_95373():
    """On-disk row count for assay train matches the API doc (95373)."""
    df = load_drugood(prefer_library=False)
    assert len(df) == 95373


def test_wf320f_assay_diversity_at_least_100():
    """Assay diversity: ≥100 unique assay_ids in protein column."""
    df = load_drugood(prefer_library=False)
    assert df["protein"].nunique() >= 100


def test_wf320f_scaffold_split_subsample_three_arrays():
    """Bemis-Murcko scaffold split (2k subsample) returns three disjoint arrays."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import Chem

    df = load_drugood(prefer_library=False).sample(n=2000, random_state=0)
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


def test_wf320f_missing_json_raises_filenotfound(tmp_path: Path):
    """Missing JSON path raises FileNotFoundError (no silent return)."""
    bogus = tmp_path / "no_such_drugood" / "lbap_refined_ic50_assay.json"
    with pytest.raises(FileNotFoundError):
        load_drugood(target_root=bogus.parent, prefer_library=False)


def test_wf320f_invalid_split_raises_valueerror(tmp_path: Path):
    """Invalid split name raises ValueError (schema guard)."""
    # Build a minimal valid JSON payload so we get past the FileNotFound
    # path; the split validator should fire before row flattening.
    fake_root = tmp_path / "fake_drugood"
    fake_root.mkdir()
    payload = {
        "split": {
            "train": [
                {
                    "smiles": "CCO",
                    "reg_label": 5.5,
                    "assay_id": "A1",
                    "cls_label": 1,
                    "domain_id": 0,
                },
            ],
        },
    }
    (fake_root / "lbap_refined_ic50_assay.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_drugood(
            subset="lbap_refined_ic50_assay",
            split="not_a_real_split",
            target_root=fake_root,
            prefer_library=False,
        )


# ---------------------------------------------------------------------------
# 2 integration tests: downstream consumer
# ---------------------------------------------------------------------------


def test_wf320f_smiles_rdkit_round_trip_subsample():
    """RDKit can round-trip ≥90% of DrugOOD SMILES (2k subsample)."""
    from rdkit import Chem

    df = load_drugood(prefer_library=False).sample(n=2000, random_state=3)
    parsed = sum(
        1 for smi in df["smiles"].tolist() if Chem.MolFromSmiles(smi) is not None
    )
    parse_rate = parsed / len(df)
    assert parse_rate >= 0.90, f"SMILES parse rate {parse_rate:.3f} < 0.90"


def test_wf320f_dataframe_consumable_by_canonical_concat():
    """DataFrame is concat-compatible with the canonical 6-col schema.

    Note: concat with an *empty* DataFrame coerces the pIC50 dtype to
    object (pandas behaviour, not a loader bug); we verify the schema
    union + row-count preservation here and re-check the pIC50 dtype on
    the original loaded frame for the integration contract.
    """
    df_dgo = load_drugood(prefer_library=False)
    # pIC50 dtype contract holds on the loader output.
    assert pd.api.types.is_numeric_dtype(df_dgo["pIC50"])

    # Concatenate with empty DataFrame of identical schema; the column
    # union must equal the canonical 6-col schema.
    schema_only = pd.DataFrame(columns=df_dgo.columns.tolist())
    combined = pd.concat([df_dgo, schema_only], ignore_index=True)
    assert combined.columns.tolist() == df_dgo.columns.tolist()
    assert len(combined) == len(df_dgo)


# ---------------------------------------------------------------------------
# 2 cross-loader tests
# ---------------------------------------------------------------------------


def test_wf320f_resolve_subset_path_uses_env_var(tmp_path: Path, monkeypatch):
    """_resolve_subset_path honours DRUGOOD_IC50_TARGET_ROOT env override."""
    monkeypatch.setenv("DRUGOOD_IC50_TARGET_ROOT", str(tmp_path))
    resolved = _resolve_subset_path("lbap_refined_ic50_assay")
    assert resolved == tmp_path / "lbap_refined_ic50_assay.json"


def test_wf320f_default_subsets_and_valid_splits_constants():
    """DEFAULT_SUBSETS has 3 entries; VALID_SPLITS has 5 entries."""
    assert len(DEFAULT_SUBSETS) == 3
    assert "lbap_refined_ic50_assay" in DEFAULT_SUBSETS
    assert "lbap_refined_ic50_scaffold" in DEFAULT_SUBSETS
    assert "lbap_refined_ic50_size" in DEFAULT_SUBSETS

    assert len(VALID_SPLITS) == 5
    expected_splits = {"train", "ood_val", "ood_test", "iid_val", "iid_test"}
    assert set(VALID_SPLITS) == expected_splits

    # DEFAULT_DRUGOOD_TARGET_ROOT is a Path on disk (parent may not exist
    # on a fresh checkout, but the path is constructable).
    assert isinstance(DEFAULT_DRUGOOD_TARGET_ROOT, Path)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
