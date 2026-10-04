"""Tests for the opt-in paper-funnel cohort builder + DOI-split CV.

Covers the Wf paper-replication brief:
  * DOI split leakage: every row of a DOI lands on one side only
  * fold partition: every row lands in exactly one validation fold
  * cohort builder determinism across calls + filter semantics
  * frozen anchor on the real MetalCytoToxDB v2 CSV (skipped when absent)

These tests exercise additive helpers only; no existing default is involved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from molmetal.data import cytotox

# Resolved at import time from MOLMETAL_DATA_DIR (v2 release, md5 f270dc90...).
REAL_CSV = cytotox.DEFAULT_CSV


def _toy_frame(n_doi: int = 25, rows_per_doi: int = 8) -> pd.DataFrame:
    rows = []
    for d in range(n_doi):
        doi = f"10.1000/toy.{d:03d}"
        for i in range(rows_per_doi):
            rows.append(
                {
                    "SMILES_Ligands": f"CC{d}-{i}",
                    "Metal": "Ru",
                    "Cell_line": "MCF-7",
                    "Time(h)": 24.0,
                    "IC50_Dark_value": 1.0 + 0.5 * i,
                    "DOI": doi,
                    "Year": 2020,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DOI splitter
# ---------------------------------------------------------------------------
def test_doi_split_no_leakage():
    """Every row of a DOI lands on one side only, in every fold."""
    df = _toy_frame()
    folds = cytotox.build_doi_split_folds(df, doi_col="DOI", n_folds=10, seed=0)
    assert len(folds) == 10
    for f in folds:
        train = df.iloc[f["train_idx"]]
        val = df.iloc[f["val_idx"]]
        assert len(val) > 0, f"fold {f['fold']} has empty validation"
        shared = set(train["DOI"]) & set(val["DOI"])
        assert not shared, f"fold {f['fold']} leaks DOIs: {sorted(shared)}"
        val_dois = set(val["DOI"])
        assert not train["DOI"].isin(val_dois).any(), (
            f"fold {f['fold']}: rows of a validation DOI appear in train"
        )


def test_doi_folds_partition_exactly():
    """Validation folds are disjoint and cover every row exactly once."""
    df = _toy_frame()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {"SMILES_Ligands": "X1", "Metal": "Ru", "Cell_line": "MCF-7",
                     "Time(h)": 24.0, "IC50_Dark_value": 2.0, "DOI": np.nan, "Year": 2020},
                    {"SMILES_Ligands": "X2", "Metal": "Ru", "Cell_line": "MCF-7",
                     "Time(h)": 24.0, "IC50_Dark_value": 2.0, "DOI": np.nan, "Year": 2020},
                ]
            ),
        ],
        ignore_index=True,
    )
    folds = cytotox.build_doi_split_folds(df, n_folds=10, seed=0)
    all_val = np.sort(np.concatenate([f["val_idx"] for f in folds]))
    assert np.array_equal(all_val, np.arange(len(df)))  # exactly-once coverage
    for f in folds:
        assert np.intersect1d(f["train_idx"], f["val_idx"]).size == 0
        combined = np.sort(np.concatenate([f["train_idx"], f["val_idx"]]))
        assert np.array_equal(combined, np.arange(len(df)))


def test_doi_split_deterministic_across_calls():
    df = _toy_frame()
    a = cytotox.build_doi_split_folds(df, n_folds=10, seed=0)
    b = cytotox.build_doi_split_folds(df, n_folds=10, seed=0)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa["val_idx"], fb["val_idx"])
        assert np.array_equal(fa["train_idx"], fb["train_idx"])


# ---------------------------------------------------------------------------
# Cohort builder
# ---------------------------------------------------------------------------
def test_cohort_builder_deterministic_and_semantics():
    df = pd.DataFrame(
        [
            {"SMILES_Ligands": "A", "Metal": "Ru", "Cell_line": "MCF-7",
             "Time(h)": 24.0, "IC50_Dark_value": 5.0, "DOI": "d1"},
            {"SMILES_Ligands": "A", "Metal": "Ru", "Cell_line": "MCF-7",
             "Time(h)": 24.0, "IC50_Dark_value": 5.0, "DOI": "d1"},  # dup key -> dropped
            {"SMILES_Ligands": "B", "Metal": "Ru", "Cell_line": "HEK293",
             "Time(h)": 24.0, "IC50_Dark_value": 5.0, "DOI": "d1"},  # non-cancer -> excluded
            {"SMILES_Ligands": "C", "Metal": "Ir", "Cell_line": "MCF-7",
             "Time(h)": 24.0, "IC50_Dark_value": 5.0, "DOI": "d2"},  # other metal
            {"SMILES_Ligands": "D", "Metal": "Ru", "Cell_line": "MCF-7",
             "Time(h)": 24.0, "IC50_Dark_value": np.nan, "DOI": "d2"},  # no IC50
        ]
    )
    idx1 = cytotox.build_paper_cohort_indices(df)
    idx2 = cytotox.build_paper_cohort_indices(df)
    assert np.array_equal(idx1, idx2)
    assert idx1.tolist() == [0]  # dedup keeps the FIRST occurrence
    cohort = cytotox.build_paper_cohort(df)
    assert cohort.index.tolist() == [0]  # input index preserved
    assert cohort.iloc[0]["SMILES_Ligands"] == "A"
    assert len(df) == 5  # input frame untouched
    # opt-out levers, one at a time
    assert cytotox.build_paper_cohort_indices(df, dedup_keys=None).tolist() == [0, 1]
    assert cytotox.build_paper_cohort_indices(df, exclude_non_cancer=False).tolist() == [0, 2]
    # exact-name matching: HLF (tumour-derived) must NOT be swept up by HFL-1
    df2 = df.copy()
    df2.loc[0, "Cell_line"] = "HLF"
    assert 0 in cytotox.build_paper_cohort_indices(df2).tolist()


# ---------------------------------------------------------------------------
# Real-data anchor (v2 CSV; skipped when the data file is absent)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not REAL_CSV.exists(), reason=f"MetalCytoToxDB v2 not at {REAL_CSV}")
def test_real_csv_paper_cohort_reproduces_frozen_funnel():
    df = pd.read_csv(REAL_CSV, low_memory=False)
    idx = cytotox.build_paper_cohort_indices(df)
    cohort = df.iloc[idx]
    # Frozen calibration anchor for MetalCytoToxDB v2 (md5 f270dc90...),
    # measured 2026-10-04; see the funnel-replication phase record.
    assert len(idx) == 15799
    assert cohort["SMILES_Ligands"].nunique() == 4793
    assert cohort["Cell_line"].nunique() == 591
    # deterministic across calls on the real frame
    assert np.array_equal(idx, cytotox.build_paper_cohort_indices(df))
    # no excluded line survives (exact normalised match)
    excluded = {n.strip().upper() for n in cytotox.NON_CANCER_CELL_LINES}
    surviving = set(cohort["Cell_line"].astype(str).str.strip().str.upper())
    assert not (surviving & excluded)


# ---------------------------------------------------------------------------
# Complex-level coherence (the `smiles=` path)
# ---------------------------------------------------------------------------
#
# 2026-10-04 — `doi_split_cv.py` called `assign_doi_folds(dois, ...)` with no
# `smiles=`, producing a split that is coherent by DOI but NOT by complex.
# The D-MPNN trainer resolves each row through a SMILES-keyed dict, so a
# complex appearing in two articles had its articles land in different folds
# and the trainer silently collapsed them.  MEASURED on the real cohort, 6.1%
# of complexes span more than one DOI (up to 13).  These tests pin the
# coherence the trainer actually depends on.

def _linked_frame() -> pd.DataFrame:
    """Two DOIs joined by a shared complex, plus one independent DOI."""
    return pd.DataFrame(
        {
            "SMILES_Ligands": [
                "CC_A", "CC_B", "CC_A",   # CC_A appears in BOTH doi.1 and doi.2
                "CC_C", "CC_D",           # doi.3, independent
            ],
            "Metal": ["Ru"] * 5,
            "Cell_line": ["MCF-7"] * 5,
            "Time(h)": [24.0] * 5,
            "IC50_Dark_value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "DOI": [
                "10.1000/doi.1", "10.1000/doi.1",
                "10.1000/doi.2",
                "10.1000/doi.3", "10.1000/doi.3",
            ],
        }
    )


def test_doi_group_ids_merges_dois_sharing_a_complex():
    """The two articles sharing CC_A must share one group id."""
    df = _linked_frame()
    g = cytotox.doi_group_ids(df["DOI"], smiles=df["SMILES_Ligands"])
    assert g[0] == g[1] == g[2], (
        "doi.1 and doi.2 are linked by complex CC_A and must be one group"
    )
    assert g[3] == g[4] != g[0], "the independent DOI must be its own group"


def test_doi_group_ids_without_smiles_is_doi_only():
    """Without `smiles` the grouping is by DOI alone — the old behaviour."""
    df = _linked_frame()
    g = cytotox.doi_group_ids(df["DOI"])
    assert g[0] == g[1] and g[0] != g[2], "doi.1 and doi.2 stay separate"


def test_every_complex_lands_in_one_fold_when_smiles_is_passed():
    """THE regression guard: no SMILES may straddle two folds.

    This is the property the trainer's SMILES-keyed split depends on.  With
    `smiles=` omitted the shared complex CC_A would sit in two folds.
    """
    df = _linked_frame()
    coh = cytotox.assign_doi_folds(
        df["DOI"], n_folds=2, seed=0, smiles=df["SMILES_Ligands"]
    )
    assert coh[0] == coh[1] == coh[2], (
        "complex CC_A straddles folds — the D-MPNN trainer keys its split on "
        "SMILES and would collapse these rows, leaking across the split"
    )

    leak = cytotox.assign_doi_folds(df["DOI"], n_folds=2, seed=0)
    # The DOI-only form is allowed to leak; assert the premise so this test
    # keeps documenting the real difference rather than passing vacuously.
    assert len(set(leak[[0, 1, 2]].tolist())) > 1, (
        "premise: without smiles the DOI-only split DOES straddle CC_A"
    )


def test_doi_split_cv_payload_passes_smiles():
    """Source guard on the omission itself.

    The bug was a missing keyword, not a wrong computation, so no numerical
    assertion can catch a future regression.  Pin the call.
    """
    src = (
        Path(__file__).resolve().parents[1] / "scripts" / "doi_split_cv.py"
    ).read_text()
    assert "smiles=sub[" in src, (
        "doi_split_cv no longer passes smiles= to assign_doi_folds — the "
        "folds it writes will be DOI-coherent but not complex-coherent, and "
        "the D-MPNN trainer will collapse the straddling complexes"
    )
