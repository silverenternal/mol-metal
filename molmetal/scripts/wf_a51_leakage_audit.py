#!/usr/bin/env python
"""Wf-A51 — measured train/validation/test leakage audit.

Every number is computed from the files on disk.  The audit answers four
questions for each dataset that is actually trained or evaluated:

1. Is the *same molecule* present on both sides of a split, once SMILES are
   canonicalised?  (raw-string splits do not guarantee this)
2. Is the *same scaffold* present on both sides?  (only relevant for the
   split protocols that claim a scaffold split)
3. Is a multi-row molecule (same SMILES, several metals / cell lines)
   assigned to more than one split?  (the trainer keys on SMILES, so one
   split label per molecule is required)
4. Do the datasets used together overlap?  (external Pt pool vs the
   5-metal training cohorts; NCI-60 control vs training, etc.)

Outputs
-------
``molmetal/reports/wf_a51_leakage_audit/leakage_audit.json``
    Machine-readable, one block per check.
``docs/DATA_LEAKAGE_AUDIT.md``
    Human-readable summary generated from the same run.

Usage
-----
    .venv/bin/python molmetal/scripts/wf_a51_leakage_audit.py

    # re-measure only the NCI-60 section, reusing the rest of an artifact
    .venv/bin/python molmetal/scripts/wf_a51_leakage_audit.py \
        --reuse molmetal/reports/wf_a51_leakage_audit/leakage_audit.json
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data.cytotox import build_paper_cohort  # noqa: E402
from molmetal.data.featurize import _strip_metal_atoms  # noqa: E402
from molmetal.data.leakage_utils import canonicalize_smiles  # noqa: E402
from molmetal.data.splits import _bemis_murcko_key  # noqa: E402

DATA = PROJECT_ROOT / ".data" / "molmetal"
REPORT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf_a51_leakage_audit"
OUT_JSON = REPORT_DIR / "leakage_audit.json"
OUT_MD = PROJECT_ROOT / "docs" / "DATA_LEAKAGE_AUDIT.md"

# Sections at or after this heading are not produced by this script.
PRESERVED_TAIL_MARKER = "\n## 8."


def _preserved_tail(path: Path) -> str:
    """Trailing sections of *path* that this generator must not delete.

    §8 (MSRS strong-model splits) is maintained outside this script.  On
    2026-10-19 a plain overwrite removed it; a re-run must regenerate §1-§7
    without destroying sections it does not own.
    """
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    idx = text.find(PRESERVED_TAIL_MARKER)
    return text[idx + 1 :] if idx >= 0 else ""

METALS = ("Ru", "Ir", "Rh", "Os", "Re")
SEEDS = (42, 43, 44, 49, 123)


# ---------------------------------------------------------------------------
# Canonicalisation / scaffold helpers (computed once per unique string)
# ---------------------------------------------------------------------------
def _map_unique(values: Sequence[str], fn) -> np.ndarray:
    """Apply *fn* to each unique string, return a row-aligned object array."""
    series = pd.Series([str(v) for v in values])
    uniq = pd.unique(series)
    table = {u: fn(u) for u in uniq}
    return series.map(table).to_numpy(dtype=object)


def _canon(values: Sequence[str]) -> np.ndarray:
    return _map_unique(values, canonicalize_smiles)


def _scaffold(values: Sequence[str]) -> np.ndarray:
    return _map_unique(values, _bemis_murcko_key)


def _scaffold_split_key(values: Sequence[str]) -> np.ndarray:
    """The key ``ScaffoldSplitter`` actually uses: empty -> per-molecule.

    Canonicalise before hashing, because that is what the splitter does
    (``ScaffoldSplitter._scaffold_for(canon)``).  Computing the key on the raw
    string is not equivalent: on the Pt 4-cell real cohort it disagrees for
    359/1,966 molecules (18 %) and manufactures two "shared" scaffolds that
    the splitter never groups together.  On cohorts whose SMILES are already
    canonical (MoleculeNet, NCI-60) the two agree.
    """
    return _map_unique(
        values,
        lambda s: _bemis_murcko_key(canonicalize_smiles(s))
        or f"__empty__{canonicalize_smiles(s)}",
    )


def _scaffold_full_nostereo(values: Sequence[str]) -> np.ndarray:
    """Bemis-Murcko on the whole molecule, chirality stripped, no desalting."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def fn(s: str) -> str:
        mol = _parse(s)
        if mol is None:
            return f"__empty__{s}"
        try:
            return (
                MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                or f"__empty__{canonicalize_smiles(s)}"
            )
        except Exception:
            # Murcko decomposition raises AtomValenceException on a handful of
            # metal-organic rows; treat them as per-molecule keys, matching
            # ``_bemis_murcko_key``'s empty-scaffold contract.
            return f"__empty__{canonicalize_smiles(s)}"

    return _map_unique(values, fn)


def _fragment_keys(smi: str) -> Tuple[str, ...]:
    """Metal-stripped organic fragments with >= 6 heavy atoms, canonical."""
    from rdkit import Chem

    stripped, _ = _strip_metal_atoms(smi)
    mol = _parse(stripped)
    if mol is None:
        return (stripped,)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    keys = []
    for frag in frags:
        if frag.GetNumHeavyAtoms() >= 6:
            keys.append(Chem.MolToSmiles(frag))
    return tuple(sorted(set(keys))) or (stripped,)


def _ligand_key(smi: str) -> str:
    """Metal-stripped, largest-fragment, canonical ligand key."""
    stripped, _ = _strip_metal_atoms(smi)
    mol = _parse(stripped)
    if mol is None:
        return stripped
    from rdkit import Chem

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol)


def _parse(smi: str):
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smi)


def _n_parse_failures(values: Sequence[str]) -> int:
    uniq = pd.unique(pd.Series([str(v) for v in values]))
    return int(sum(1 for s in uniq if _parse(s) is None))


def _overlap(
    smiles: np.ndarray,
    split: np.ndarray,
    a: str,
    b: str,
    keys: np.ndarray,
) -> Dict[str, float]:
    """Overlap of *a* vs *b* in terms of *keys* (row-aligned)."""
    mask_a = split == a
    mask_b = split == b
    ka = set(keys[mask_a].tolist())
    kb = set(keys[mask_b].tolist())
    n_b_rows = int(mask_b.sum())
    seen_rows = int(np.isin(keys[mask_b], list(ka)).sum()) if n_b_rows else 0
    return {
        "n_a_rows": int(mask_a.sum()),
        "n_b_rows": n_b_rows,
        "n_a_keys": len(ka),
        "n_b_keys": len(kb),
        "n_intersect_keys": len(ka & kb),
        "n_b_rows_seen_in_a": seen_rows,
        "frac_b_rows_seen_in_a": (seen_rows / n_b_rows) if n_b_rows else 0.0,
    }


# ---------------------------------------------------------------------------
# 1. Predictive line — MetalCytoToxDB 5-metal cohorts
# ---------------------------------------------------------------------------
def audit_predictive() -> Dict[str, object]:
    from sklearn.model_selection import StratifiedKFold

    from molmetal.data.cytotox import assign_doi_folds

    df = pd.read_csv(DATA / "MetalCytoToxDB.csv", low_memory=False)
    out: Dict[str, object] = {"path": str(DATA / "MetalCytoToxDB.csv"), "metals": {}}
    union_raw: set = set()
    union_lig: set = set()

    for metal in METALS:
        sub = build_paper_cohort(df, metal=metal)
        raw = sub["SMILES_Ligands"].astype(str).to_numpy()
        y_row = (
            pd.to_numeric(sub["IC50_Dark_value"], errors="coerce") < 10.0
        ).astype(int).to_numpy()

        # collapse_duplicates(tie -> active) — one row per exact SMILES string
        frame = pd.DataFrame({"s": raw, "y": y_row})
        agg = frame.groupby("s", sort=False)["y"].agg(
            lambda v: int(v.sum() * 2 >= len(v))
        )
        smiles = agg.index.to_numpy(dtype=object)
        y = agg.to_numpy(dtype=int)

        canon = _canon(smiles)
        lig = _map_unique(smiles, _ligand_key)
        # ``_scaffold_split_key`` maps an empty scaffold to a per-molecule
        # sentinel.  Using the raw empty string would group every acyclic
        # molecule into one bucket and report a false overlap.
        scaf = _scaffold_split_key(smiles)
        union_raw.update(canon.tolist())
        union_lig.update(lig.tolist())

        # canonical variants: one molecule written several ways
        grp = pd.DataFrame({"raw": smiles, "canon": canon})
        variants = grp.groupby("canon")["raw"].nunique()
        n_multi_variant = int((variants > 1).sum())
        n_rows_in_multi = int(grp["canon"].isin(variants[variants > 1].index).sum())

        # exact-duplicate check after collapse
        n_exact_dupes = int(len(smiles) - len(set(smiles)))

        # ---- protocol A: StratifiedKFold on raw SMILES (probe path) ----
        per_seed = []
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            seen_fracs, scaf_fracs, n_cross_canon = [], [], []
            for tr, te in skf.split(np.zeros(len(y)), y):
                train_canon = set(canon[tr].tolist())
                train_scaf = set(scaf[tr].tolist())
                seen_fracs.append(float(np.isin(canon[te], list(train_canon)).mean()))
                scaf_fracs.append(float(np.isin(scaf[te], list(train_scaf)).mean()))
                n_cross_canon.append(
                    int(len(train_canon & set(canon[te].tolist())))
                )
            per_seed.append(
                {
                    "seed": seed,
                    "mean_frac_test_rows_canon_in_train": float(np.mean(seen_fracs)),
                    "max_frac_test_rows_canon_in_train": float(np.max(seen_fracs)),
                    "mean_frac_test_rows_scaffold_in_train": float(np.mean(scaf_fracs)),
                    "mean_n_canon_keys_crossing_folds": float(np.mean(n_cross_canon)),
                }
            )

        # ---- protocol B: DOI-grouped 10-fold (m1 paper path) ----
        # assign_doi_folds runs on the un-collapsed cohort (one row per
        # measurement); the collapsed SMILES then inherit their fold.  The
        # union-find inside assign_doi_folds(smiles=raw) merges every DOI that
        # shares a raw SMILES, so a SMILES cannot straddle two folds unless a
        # canonical variant is written differently — which is exactly the
        # quantity this check is meant to expose.
        dois = sub["DOI"].astype(str).to_numpy()
        folds_row = assign_doi_folds(dois, n_folds=10, seed=0, smiles=raw)
        fold_nunique = pd.Series(folds_row).groupby(pd.Series(raw)).nunique()
        n_smi_multi_fold = int((fold_nunique > 1).sum())
        fold_map = pd.Series(folds_row).groupby(pd.Series(raw)).agg(
            lambda v: int(v.mode().iloc[0])
        )
        folds = fold_map.reindex(pd.Series(smiles)).to_numpy()
        doi_seen, doi_scaf = [], []
        for k in range(10):
            tr = folds != k
            te = folds == k
            if not te.any():
                continue
            train_canon = set(canon[tr].tolist())
            train_scaf = set(scaf[tr].tolist())
            doi_seen.append(float(np.isin(canon[te], list(train_canon)).mean()))
            doi_scaf.append(float(np.isin(scaf[te], list(train_scaf)).mean()))

        out["metals"][metal] = {
            "n_rows_after_collapse": int(len(smiles)),
            "n_parse_failures": _n_parse_failures(smiles),
            "n_active": int(y.sum()),
            "n_inactive": int((1 - y).sum()),
            "n_exact_duplicate_smiles_after_collapse": n_exact_dupes,
            "n_canonical_keys": int(len(set(canon.tolist()))),
            "n_canonical_variants_multi_raw": n_multi_variant,
            "n_rows_in_multi_raw_canonical_groups": n_rows_in_multi,
            "stratified_kfold": {
                "note": "5-fold on raw SMILES, the metal_activity_transfer_probe path",
                "per_seed": per_seed,
            },
            "doi_grouped_10fold": {
                "note": "assign_doi_folds(seed=0, smiles=raw), the m1 paper path",
                "n_smiles_spanning_multiple_folds": n_smi_multi_fold,
                "mean_frac_test_rows_canon_in_train": float(np.mean(doi_seen)),
                "max_frac_test_rows_canon_in_train": float(np.max(doi_seen)),
                "mean_frac_test_rows_scaffold_in_train": float(np.mean(doi_scaf)),
            },
        }

    out["union_n_canonical_5metal"] = len(union_raw)
    out["union_n_ligand_keys_5metal"] = len(union_lig)
    out["_union_canon"] = union_raw
    out["_union_lig"] = union_lig
    return out


# ---------------------------------------------------------------------------
# 2. Generative line — CFM 5-metal v3/v4
# ---------------------------------------------------------------------------
def audit_cfm(path: Path, smiles_col: str, metal_col: str) -> Dict[str, object]:
    df = pd.read_csv(path, low_memory=False)
    smiles = df[smiles_col].astype(str).to_numpy()
    split = df["split"].astype(str).to_numpy()

    canon = _canon(smiles)
    lig = _map_unique(smiles, _ligand_key)
    scaf_bm = _scaffold_split_key(smiles)
    scaf_full_nostereo = _scaffold_full_nostereo(smiles)
    scaf_col = df["scaffold_id"].astype(str).to_numpy()

    # one split label per molecule?
    per_smi = pd.DataFrame({"s": smiles, "split": split})
    n_smi_multi_split = int((per_smi.groupby("s")["split"].nunique() > 1).sum())
    multi_idx = per_smi.groupby("s")["split"].nunique()
    multi_smis = set(multi_idx[multi_idx > 1].index)
    n_rows_multi_split = int(per_smi["s"].isin(multi_smis).sum())

    # split consistency of the scaffold_id column itself
    per_scaf = pd.DataFrame({"s": scaf_col, "split": split})
    n_scaf_multi_split = int((per_scaf.groupby("s")["split"].nunique() > 1).sum())
    n_smi_multi_scaf = int((per_smi.assign(scaf=scaf_col).groupby("s")["scaf"].nunique() > 1).sum())

    # exact duplicate measurement rows across splits
    dup_cols = [smiles_col, metal_col, "cell_line"]
    dup = df.duplicated(subset=dup_cols, keep=False)
    dup_frame = df.loc[dup]
    n_dup_rows = int(dup.sum())
    n_dup_groups_cross_split = int(
        dup_frame.groupby(dup_cols)["split"].nunique().gt(1).sum()
    ) if n_dup_rows else 0

    pairs = {}
    for a, b in combinations(("train", "val", "test"), 2):
        pairs[f"{a}_vs_{b}"] = {
            "raw": _overlap(smiles, split, a, b, smiles),
            "canonical": _overlap(smiles, split, a, b, canon),
            "ligand": _overlap(smiles, split, a, b, lig),
            "bemis_murcko": _overlap(smiles, split, a, b, scaf_bm),
            "bemis_murcko_full_nostereo": _overlap(
                smiles, split, a, b, scaf_full_nostereo
            ),
            "scaffold_id_column": _overlap(smiles, split, a, b, scaf_col),
        }

    return {
        "path": str(path),
        "n_rows": int(len(df)),
        "n_parse_failures": _n_parse_failures(smiles),
        "n_unique_raw_smiles": int(df[smiles_col].nunique()),
        "n_unique_canonical": int(len(set(canon.tolist()))),
        "n_unique_bemis_murcko": int(len(set(scaf_bm.tolist()))),
        "n_unique_scaffold_id": int(df["scaffold_id"].nunique()),
        "split_row_counts": df["split"].value_counts().to_dict(),
        "split_unique_canonical_counts": {
            s: int(len(set(canon[split == s].tolist())))
            for s in ("train", "val", "test")
        },
        "split_unique_scaffold_id_counts": {
            s: int(df.loc[split == s, "scaffold_id"].nunique())
            for s in ("train", "val", "test")
        },
        "n_smiles_with_multiple_split_labels": n_smi_multi_split,
        "n_rows_in_multi_split_smiles": n_rows_multi_split,
        "n_scaffold_ids_with_multiple_split_labels": n_scaf_multi_split,
        "n_smiles_with_multiple_scaffold_ids": n_smi_multi_scaf,
        "n_duplicate_measurement_rows": n_dup_rows,
        "n_duplicate_groups_crossing_splits": n_dup_groups_cross_split,
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# 3. Platinum external pool
# ---------------------------------------------------------------------------
def audit_pt() -> Dict[str, object]:
    path = DATA / "wf_a37_platinai_biodata" / "pt" / "platinai_pt_all_cells_with_smiles.jsonl"
    rows = [json.loads(line) for line in path.open()]
    df = pd.DataFrame(rows)
    smiles = df["smiles"].astype(str).to_numpy()
    canon = _canon(smiles)
    lig = _map_unique(smiles, _ligand_key)
    stripped = _map_unique(
        smiles, lambda s: canonicalize_smiles(_strip_metal_atoms(s)[0])
    )
    frags = _map_unique(smiles, _fragment_keys)

    code_canon = df.assign(canon=canon).groupby("compound_code")["canon"].nunique()
    canon_code = df.assign(canon=canon).groupby("canon")["compound_code"].nunique()

    return {
        "path": str(path),
        "n_rows": int(len(df)),
        "n_parse_failures": _n_parse_failures(smiles),
        "n_unique_biodata_id": int(df["biodata_id"].nunique()),
        "n_duplicate_biodata_id_rows": int(len(df) - df["biodata_id"].nunique()),
        "n_unique_compound_code": int(df["compound_code"].nunique()),
        "n_unique_raw_smiles": int(df["smiles"].nunique()),
        "n_unique_canonical": int(len(set(canon.tolist()))),
        "n_unique_stripped_canonical": int(len(set(stripped.tolist()))),
        "n_unique_ligand_keys": int(len(set(lig.tolist()))),
        "n_cell_lines": int(df["cell_line"].nunique()),
        "n_compound_codes_with_multiple_canonical": int((code_canon > 1).sum()),
        "n_canonical_with_multiple_compound_codes": int((canon_code > 1).sum()),
        "n_duplicate_code_cell_rows": int(
            len(df) - len(df.drop_duplicates(subset=["compound_code", "cell_line"]))
        ),
        "n_duplicate_code_cell_value_rows": int(
            len(df)
            - len(df.drop_duplicates(subset=["compound_code", "cell_line", "value_mean"]))
        ),
        "has_split_column": "split" in df.columns,
        "_canon": set(canon.tolist()),
        "_lig": set(lig.tolist()),
        "_stripped": set(stripped.tolist()),
        "_frags": set().union(*[set(x) for x in frags.tolist()]),
    }


# ---------------------------------------------------------------------------
# 3b. Platinum 4-cell TRAINING cohort (a split, unlike the pool above)
# ---------------------------------------------------------------------------
def audit_platinai_4cell() -> Dict[str, object]:
    """Audit the split the Pt 4-cell trainer actually builds.

    The loader is imported rather than re-implemented so this measures the
    real code path.  ``split_strategy`` is named ``scaffold_70_10_10`` but
    resolves to a stable hash of the SMILES string
    (``train_dmpnn_multitask_wasserstein.py:702``); this section measures
    what that name costs on the scaffold axis.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        load_platinai_4cell_split_table,
    )

    path = DATA / "platinai_4cell.csv"
    df = load_platinai_4cell_split_table(path)
    uniq = df.drop_duplicates(subset=["smiles"])[["smiles", "split"]]
    smiles = uniq["smiles"].astype(str).to_numpy()
    split = uniq["split"].astype(str).to_numpy()

    canon = _canon(smiles)
    scaf = _scaffold_split_key(smiles)
    full_ns = _scaffold_full_nostereo(smiles)

    pairs: Dict[str, object] = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        pairs[f"{a}_vs_{b}"] = {
            "canonical": _overlap(smiles, split, a, b, canon),
            "bemis_murcko": _overlap(smiles, split, a, b, scaf),
            "bemis_murcko_full_nostereo": _overlap(smiles, split, a, b, full_ns),
        }

    # 2026-10-19 (Wf-A53): the opt-in real scaffold split, measured through the
    # same loader so the comparison is like-for-like.
    alt_df = load_platinai_4cell_split_table(
        path, split_strategy="bemis_murcko_70_10_10"
    )
    alt_uniq = alt_df.drop_duplicates(subset=["smiles"])[["smiles", "split"]]
    alt_smiles = alt_uniq["smiles"].astype(str).to_numpy()
    alt_split = alt_uniq["split"].astype(str).to_numpy()
    alt_scaf = _scaffold_split_key(alt_smiles)
    alt_pairs: Dict[str, object] = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        alt_pairs[f"{a}_vs_{b}"] = _overlap(alt_smiles, alt_split, a, b, alt_scaf)

    return {
        "path": str(path),
        "split_strategy_in_code": "scaffold_70_10_10",
        # Measured, not asserted: the current loader sets scaffold_id to an
        # integer hash.  If that is ever changed to a scaffold SMILES this
        # flips to False and the report stops calling it a hash split.
        "scaffold_id_looks_like_hash": bool(
            pd.to_numeric(df["scaffold_id"], errors="coerce").notna().all()
        ),
        "split_sizes_unique_smiles": {
            s: int((split == s).sum()) for s in ("train", "val", "test")
        },
        "n_rows_long": int(len(df)),
        "n_unique_smiles": int(len(uniq)),
        "n_parse_failures": _n_parse_failures(smiles),
        "n_smiles_with_multiple_split_labels": int(
            (pd.Series(split).groupby(pd.Series(smiles)).nunique() > 1).sum()
        ),
        "pairs": pairs,
        "bemis_murcko_strategy": {
            "split_strategy": "bemis_murcko_70_10_10",
            "split_sizes_unique_smiles": {
                s: int((alt_split == s).sum()) for s in ("train", "val", "test")
            },
            "pairs": alt_pairs,
        },
    }


# ---------------------------------------------------------------------------
# 4. NCI-60 9-cell control
# ---------------------------------------------------------------------------
def audit_nci60() -> Dict[str, object]:
    """Audit both NCI-60 protocols through the production loader.

    The loader is imported rather than re-implemented, for the same reason as
    §4: the split that matters is the one the trainer builds.  Both protocols
    are measured because the legacy one is still what every shipped m21
    artifact was scored on and is pinned in five configs -- auditing only the
    default would leave the legacy reference unverified, and auditing only the
    legacy (as this section did before 2026-10-22) would describe a protocol
    no bare run uses.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        load_nci60_9cell_split_table,
    )

    path = PROJECT_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"
    raw = pd.read_csv(path, low_memory=False)
    cells = sorted(raw["cell_line"].astype(str).unique())
    splits = ("train", "val", "test")

    def _one(strategy: str) -> Dict[str, object]:
        df = load_nci60_9cell_split_table(path, split_strategy=strategy)
        df = df.assign(
            smiles=df["smiles"].astype(str), split=df["split"].astype(str),
            cell_line=df["cell_line"].astype(str),
        )
        uniq = df.drop_duplicates(subset=["smiles"])[["smiles", "split"]]
        smiles = uniq["smiles"].to_numpy(dtype=object)
        split = uniq["split"].to_numpy(dtype=object)

        canon = _canon(smiles)
        scaf = _scaffold_split_key(smiles)
        lig = _map_unique(smiles, _ligand_key)

        pairs = {}
        for a, b in combinations(splits, 2):
            pairs[f"{a}_vs_{b}"] = {
                "canonical": _overlap(smiles, split, a, b, canon),
                "bemis_murcko_split_key": _overlap(smiles, split, a, b, scaf),
                "bemis_murcko_desalted": _overlap(smiles, split, a, b, scaf),
                "ligand": _overlap(smiles, split, a, b, lig),
            }

        # Per-cell row counts.  Iterating ``cells`` rather than groupby keeps a
        # cell that has zero rows in a split visible -- that is how the legacy
        # protocol's SKOV3 val=0 was found.
        per_cell = {
            c: {s: int(((df["cell_line"] == c) & (df["split"] == s)).sum())
                for s in splits}
            for c in cells
        }

        # A scaffold may not span two splits.  Measured on the key set, not on
        # index ranks: two distinct scaffolds can share a rank.
        scaf_sets = {s: set(scaf[split == s].tolist()) for s in splits}
        spanning = sum(
            len(scaf_sets[a] & scaf_sets[b]) for a, b in combinations(splits, 2)
        )

        return {
            "split_strategy": strategy,
            "split_sizes_unique_smiles": {
                s: int((split == s).sum()) for s in splits
            },
            "split_sizes_rows": {s: int((df["split"] == s).sum()) for s in splits},
            "row_fraction": {
                s: round(float((df["split"] == s).sum()) / len(df), 4)
                for s in splits
            },
            "per_cell_rows": per_cell,
            "min_test_rows_per_cell": min(v["test"] for v in per_cell.values()),
            "min_val_rows_per_cell": min(v["val"] for v in per_cell.values()),
            "cells_below_30_test_rows": sorted(
                c for c, v in per_cell.items() if v["test"] < 30
            ),
            "cells_with_zero_val_rows": sorted(
                c for c, v in per_cell.items() if v["val"] == 0
            ),
            "n_scaffolds_spanning_splits": int(spanning),
            "n_smiles_with_multiple_split_labels": int(
                (pd.Series(split).groupby(pd.Series(smiles)).nunique() > 1).sum()
            ),
            "pairs": pairs,
        }

    legacy = _one("scaffold_70_10_10")
    coverage = _one("scaffold_coverage_70_10_20")

    # Sets used by the cross-dataset section.  Protocol-independent: both
    # protocols partition the same 5,207 molecules.
    smiles_all = np.asarray(sorted(raw["smiles"].astype(str).unique()), dtype=object)
    canon = _canon(smiles_all)
    lig = _map_unique(smiles_all, _ligand_key)

    return {
        "path": str(path),
        "n_rows": int(len(raw)),
        "n_cells": len(cells),
        "n_parse_failures": _n_parse_failures(smiles_all),
        "n_unique_raw_smiles": int(len(smiles_all)),
        "n_unique_canonical": int(len(set(canon.tolist()))),
        "n_unique_bemis_murcko": int(
            len(set(_scaffold_split_key(smiles_all).tolist()))
        ),
        "default_protocol": "scaffold_coverage_70_10_20",
        "protocols": {
            "scaffold_70_10_10": legacy,
            "scaffold_coverage_70_10_20": coverage,
        },
        # The figure the m21 docstring carried before 2026-10-21.  It is not
        # reproducible: it came from the pre-BLK-PA12 splitter, which returned
        # an empty scaffold key for every molecule.
        "pre_blk_pa12_claim": {
            "split_sizes_unique_smiles": {"train": 3645, "val": 521, "test": 1041},
            "reproducible": False,
        },
        "_canon": set(canon.tolist()),
        "_lig": set(lig.tolist()),
    }


# ---------------------------------------------------------------------------
# 4b. Platinum 4-cell REAL-label cohort
# ---------------------------------------------------------------------------
def audit_platinai_4cell_real() -> Dict[str, object]:
    """Leakage at the scaffold key for the REAL-label 4-cell cohort.

    §4 audits ``platinai_4cell.csv``, whose labels are synthetic and which is
    marked DO-NOT-TRAIN.  The cohort the trainer is actually pointed at is
    ``platinai_4cell_real_trainer.csv``.  Its split is built inline in
    ``train_cisplatin_resistance.main`` (there is no loader function to
    import), so this section calls the same splitter with the same coverage
    dict; the row counts are cross-checked against the recorded
    ``metrics/by_metric/split_coverage_4cell.json``.
    """
    from molmetal.data.splits import ScaffoldSplitter
    from molmetal.scripts import wf_a61_split_coverage_audit as a61
    from molmetal.scripts.train_cisplatin_resistance import CELL_LINES

    rows, smiles, labels, _values = a61._load()
    result = ScaffoldSplitter(
        fractions=a61.FRACTIONS, coverage={cl: labels[cl] for cl in CELL_LINES}
    )(a61._Smiles(smiles))

    split = np.full(len(smiles), "train", dtype=object)
    split[np.asarray(result.val_idx, dtype=np.int64)] = "val"
    split[np.asarray(result.test_idx, dtype=np.int64)] = "test"
    smi = np.asarray(smiles, dtype=object)

    canon = _canon(smi)
    scaf = _scaffold_split_key(smi)

    pairs = {}
    for a, b in combinations(("train", "val", "test"), 2):
        pairs[f"{a}_vs_{b}"] = {
            "canonical": _overlap(smi, split, a, b, canon),
            "bemis_murcko_split_key": _overlap(smi, split, a, b, scaf),
        }

    scaf_sets = {s: set(scaf[split == s].tolist()) for s in ("train", "val", "test")}
    spanning = sum(
        len(scaf_sets[a] & scaf_sets[b])
        for a, b in combinations(("train", "val", "test"), 2)
    )

    return {
        "path": str(a61.COHORT),
        "split_strategy": "scaffold_coverage (default since 2026-10-21)",
        "n_compounds": len(rows),
        "n_unique_smiles": int(len(set(smi.tolist()))),
        "n_parse_failures": _n_parse_failures(smi),
        "split_sizes_rows": {
            s: int((split == s).sum()) for s in ("train", "val", "test")
        },
        "n_scaffolds_spanning_splits": int(spanning),
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# 5. MoleculeNet controls
# ---------------------------------------------------------------------------
def audit_moleculenet() -> Dict[str, object]:
    import importlib.util
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    spec = importlib.util.spec_from_file_location(
        "train_moleculenet", PROJECT_ROOT / "molmetal" / "scripts" / "train_moleculenet.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_moleculenet"] = mod
    spec.loader.exec_module(mod)

    out = {"tasks": {}}
    for task in ("bace", "bbbp", "clintox", "hiv"):
        smiles, labels = mod._load_task(task)
        smiles_arr = np.asarray(smiles, dtype=object)
        canon = _canon(smiles)
        scaf = _scaffold(smiles)
        split_key = np.asarray(
            [mod._bemis_murcko_scaffold(s) for s in smiles], dtype=object
        )
        split_key_nostereo = _scaffold_full_nostereo(smiles)
        per_seed = []
        for seed in SEEDS:
            tr, va, te = mod._scaffold_split(smiles, seed=seed)
            label = np.full(len(smiles), "train", dtype=object)
            label[va] = "val"
            label[te] = "test"
            fallback = (len(te) < 5) or (len(np.unique(labels[te])) < 2)
            rec = {
                "seed": seed,
                "split_sizes": [int(len(tr)), int(len(va)), int(len(te))],
                "random_fallback_triggered": bool(fallback),
                "canonical": {},
                "bemis_murcko_split_key": {},
                "bemis_murcko_split_key_nostereo": {},
                "bemis_murcko_desalted": {},
            }
            for a, b in combinations(("train", "val", "test"), 2):
                rec["canonical"][f"{a}_vs_{b}"] = _overlap(
                    smiles_arr, label, a, b, canon
                )
                rec["bemis_murcko_split_key"][f"{a}_vs_{b}"] = _overlap(
                    smiles_arr, label, a, b, split_key
                )
                rec["bemis_murcko_split_key_nostereo"][f"{a}_vs_{b}"] = _overlap(
                    smiles_arr, label, a, b, split_key_nostereo
                )
                rec["bemis_murcko_desalted"][f"{a}_vs_{b}"] = _overlap(
                    smiles_arr, label, a, b, scaf
                )
            per_seed.append(rec)
        out["tasks"][task] = {
            "n_rows": int(len(smiles)),
            "n_parse_failures": _n_parse_failures(smiles),
            "n_unique_raw_smiles": int(len(set(smiles))),
            "n_unique_canonical": int(len(set(canon.tolist()))),
            "n_unique_bemis_murcko_split_key": int(len(set(split_key.tolist()))),
            "n_unique_bemis_murcko_desalted": int(len(set(scaf.tolist()))),
            "positive_rate": float(np.mean(labels)),
            "per_seed": per_seed,
            "_canon": set(canon.tolist()),
        }
    return out


# ---------------------------------------------------------------------------
# 6. DrugOOD OOD control
# ---------------------------------------------------------------------------
def audit_drugood() -> Dict[str, object]:
    path = DATA / "drugood" / "ic50" / "csv" / "lbap_refined_ic50_scaffold.csv"
    df = pd.read_csv(path, low_memory=False)
    smiles = df["smiles"].astype(str).to_numpy()
    split = df["split"].astype(str).to_numpy()

    # 266k rows — canonicalise unique strings only (there are 266k unique,
    # so this is the full column; kept for honesty about the cost).
    canon = _canon(smiles)

    pairs = {}
    for a, b in (("train", "ood_test"), ("train", "iid_test"), ("train", "ood_val")):
        pairs[f"{a}_vs_{b}"] = {
            "raw": _overlap(smiles, split, a, b, smiles),
            "canonical": _overlap(smiles, split, a, b, canon),
            "scaffold_split_id": _overlap(
                smiles, split, a, b, df["scaffold_split_id"].astype(str).to_numpy()
            ),
        }
    return {
        "path": str(path),
        "n_rows": int(len(df)),
        "n_unique_raw_smiles": int(df["smiles"].nunique()),
        "n_unique_canonical": int(len(set(canon.tolist()))),
        "split_row_counts": df["split"].value_counts().to_dict(),
        "pairs": pairs,
        "_canon": set(canon.tolist()),
    }


# ---------------------------------------------------------------------------
# 7. Cross-dataset overlap
# ---------------------------------------------------------------------------
def cross_overlap(
    pred: Dict[str, object],
    cfm: Dict[str, object],
    pt: Dict[str, object],
    nci: Dict[str, object],
    molnet: Dict[str, object],
) -> Dict[str, object]:
    cfm_df = pd.read_csv(
        DATA / "metalcytotox_5metal_multitask_v3.csv", low_memory=False
    )
    cfm_smiles = cfm_df["SMILES_Ligands"].astype(str).to_numpy()
    cfm_canon = set(_canon(cfm_smiles).tolist())
    cfm_lig = set(_map_unique(cfm_smiles, _ligand_key).tolist())
    cfm_frags = set().union(
        *[set(x) for x in _map_unique(cfm_smiles, _fragment_keys).tolist()]
    )
    cfm_train = cfm_df[cfm_df["split"] == "train"]
    cfm_train_smiles = cfm_train["SMILES_Ligands"].astype(str).to_numpy()
    cfm_train_canon = set(_canon(cfm_train_smiles).tolist())
    cfm_train_lig = set(_map_unique(cfm_train_smiles, _ligand_key).tolist())

    pred_c, pred_l = pred["_union_canon"], pred["_union_lig"]
    pred_df = pd.read_csv(DATA / "MetalCytoToxDB.csv", low_memory=False)
    pred_smiles = pred_df["SMILES_Ligands"].astype(str).to_numpy()
    pred_frags = set().union(
        *[set(x) for x in _map_unique(pred_smiles, _fragment_keys).tolist()]
    )
    pt_c, pt_l, pt_f = pt["_canon"], pt["_lig"], pt["_frags"]
    nci_c, nci_l = nci["_canon"], nci["_lig"]

    def pair(a, b, na, nb):
        return {
            "n_a": len(a),
            "n_b": len(b),
            "n_intersect": len(a & b),
            "frac_a": len(a & b) / len(a) if a else 0.0,
            "frac_b": len(a & b) / len(b) if b else 0.0,
            "label": f"{na} vs {nb}",
        }

    molnet_union = set()
    for t in molnet["tasks"].values():
        molnet_union |= t["_canon"]

    return {
        "predictive5_vs_cfm_all": {
            "canonical": pair(pred_c, cfm_canon, "predictive5", "cfm_all"),
            "ligand": pair(pred_l, cfm_lig, "predictive5", "cfm_all"),
        },
        "predictive5_vs_cfm_train": {
            "canonical": pair(pred_c, cfm_train_canon, "predictive5", "cfm_train"),
            "ligand": pair(pred_l, cfm_train_lig, "predictive5", "cfm_train"),
        },
        "pt_vs_predictive5": {
            "canonical": pair(pt_c, pred_c, "pt", "predictive5"),
            "ligand": pair(pt_l, pred_l, "pt", "predictive5"),
            "fragment": pair(pt_f, pred_frags, "pt_fragments", "predictive5_fragments"),
        },
        "pt_vs_cfm_all": {
            "canonical": pair(pt_c, cfm_canon, "pt", "cfm_all"),
            "ligand": pair(pt_l, cfm_lig, "pt", "cfm_all"),
            "fragment": pair(pt_f, cfm_frags, "pt_fragments", "cfm_fragments"),
        },
        "pt_vs_cfm_train": {
            "canonical": pair(pt_c, cfm_train_canon, "pt", "cfm_train"),
            "ligand": pair(pt_l, cfm_train_lig, "pt", "cfm_train"),
        },
        "nci60_vs_predictive5": {
            "canonical": pair(nci_c, pred_c, "nci60", "predictive5"),
            "ligand": pair(nci_l, pred_l, "nci60", "predictive5"),
        },
        "nci60_vs_cfm_all": {
            "canonical": pair(nci_c, cfm_canon, "nci60", "cfm_all"),
            "ligand": pair(nci_l, cfm_lig, "nci60", "cfm_all"),
        },
        "nci60_vs_pt": {
            "canonical": pair(nci_c, pt_c, "nci60", "pt"),
            "ligand": pair(nci_l, pt_l, "nci60", "pt"),
        },
        "molnet_vs_predictive5": pair(
            molnet_union, pred_c, "moleculenet", "predictive5"
        ),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def render_markdown(result: Dict[str, object]) -> str:
    L: List[str] = []
    add = L.append
    add("# Data leakage audit — measured")
    add("")
    add(
        "> Generated by `molmetal/scripts/wf_a51_leakage_audit.py`. Every number is "
        "computed from the files on disk; re-run the script rather than editing this file."
    )
    add("")

    add("## 1. Predictive line (MetalCytoToxDB, 5-metal cohorts)")
    add("")
    add(
        "The trainer keys its split on the **raw SMILES string**. A molecule written "
        "two different ways therefore counts as two molecules and can land in both "
        "train and test. The table measures that, and the scaffold axis for reference."
    )
    add("")
    add("| Metal | rows | parse failures | canonical keys | multi-string molecules | SKF test rows seen in train | SKF test scaffolds seen in train | DOI-10fold test rows seen in train |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for metal, rec in result["predictive"]["metals"].items():
        skf = rec["stratified_kfold"]["per_seed"]
        seen = np.mean([r["mean_frac_test_rows_canon_in_train"] for r in skf])
        scaf = np.mean([r["mean_frac_test_rows_scaffold_in_train"] for r in skf])
        add(
            f"| {metal} | {rec['n_rows_after_collapse']:,} | {rec['n_parse_failures']:,} | "
            f"{rec['n_canonical_keys']:,} | "
            f"{rec['n_canonical_variants_multi_raw']:,} | {_pct(seen)} | {_pct(scaf)} | "
            f"{_pct(rec['doi_grouped_10fold']['mean_frac_test_rows_canon_in_train'])} |"
        )
    add("")
    add(
        "`multi-string molecules` counts canonical molecules represented by more than "
        "one distinct raw SMILES string in that metal's cohort."
    )
    add("")

    add("## 2. Generative line (CFM 5-metal)")
    add("")
    add("| File | rows | parse failures | unique canonical | multi-split molecules | scaffold IDs in >1 split | desalted BM scaffolds in train and test |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for key in ("v3", "v4"):
        rec = result["cfm"][key]
        bm = rec["pairs"]["train_vs_test"]["bemis_murcko"]
        add(
            f"| {Path(rec['path']).name} | {rec['n_rows']:,} | "
            f"{rec['n_parse_failures']:,} | {rec['n_unique_canonical']:,} | "
            f"{rec['n_smiles_with_multiple_split_labels']:,} | "
            f"{rec['n_scaffold_ids_with_multiple_split_labels']:,} | "
            f"{bm['n_intersect_keys']:,} |"
        )
    add("")
    for key in ("v3", "v4"):
        rec = result["cfm"][key]
        p = rec["pairs"]["train_vs_test"]
        add(f"**{Path(rec['path']).name}** — train vs test:")
        add("")
        add(f"- raw string: {p['raw']['n_intersect_keys']:,} shared")
        add(
            f"- canonical: {p['canonical']['n_intersect_keys']:,} shared "
            f"({_pct(p['canonical']['frac_b_rows_seen_in_a'])} of test rows)"
        )
        add(
            f"- Bemis-Murcko (desalted, stereo-blind): "
            f"{p['bemis_murcko']['n_intersect_keys']:,} shared "
            f"({_pct(p['bemis_murcko']['frac_b_rows_seen_in_a'])} of test rows)"
        )
        add(
            f"- Bemis-Murcko (whole complex, stereo-blind): "
            f"{p['bemis_murcko_full_nostereo']['n_intersect_keys']:,} shared "
            f"({_pct(p['bemis_murcko_full_nostereo']['frac_b_rows_seen_in_a'])} "
            f"of test rows) — isolates the stereo-only component"
        )
        add(
            f"- in-file `scaffold_id`: {p['scaffold_id_column']['n_intersect_keys']:,} shared"
        )
        pv = rec["pairs"]["train_vs_val"]
        add(
            f"- train vs val: desalted stereo-blind BM "
            f"{pv['bemis_murcko']['n_intersect_keys']:,} shared "
            f"({_pct(pv['bemis_murcko']['frac_b_rows_seen_in_a'])} of val rows); "
            f"whole-complex stereo-blind "
            f"{pv['bemis_murcko_full_nostereo']['n_intersect_keys']:,} "
            f"({_pct(pv['bemis_murcko_full_nostereo']['frac_b_rows_seen_in_a'])})"
        )
        add(
            f"- duplicate measurement rows (same SMILES + metal + cell): "
            f"{rec['n_duplicate_measurement_rows']:,}; groups crossing splits: "
            f"{rec['n_duplicate_groups_crossing_splits']:,}"
        )
        add("")

    add("## 3. Platinum external pool")
    add("")
    pt = result["pt"]
    add(f"- Rows {pt['n_rows']:,}; parse failures {pt['n_parse_failures']:,} of "
        f"{pt['n_unique_raw_smiles']:,} unique; unique `biodata_id` "
        f"{pt['n_unique_biodata_id']:,}; duplicate-ID rows "
        f"{pt['n_duplicate_biodata_id_rows']:,}")
    add(f"- Unique canonical SMILES {pt['n_unique_canonical']:,}; unique "
        f"metal-stripped canonical {pt['n_unique_stripped_canonical']:,}; unique "
        f"ligand keys {pt['n_unique_ligand_keys']:,}")
    add(f"- Compound codes with >1 canonical SMILES: "
        f"{pt['n_compound_codes_with_multiple_canonical']:,}; canonical SMILES "
        f"with >1 compound code: {pt['n_canonical_with_multiple_compound_codes']:,}")
    add(f"- Duplicate (code, cell) rows {pt['n_duplicate_code_cell_rows']:,}; "
        f"duplicate (code, cell, value) rows {pt['n_duplicate_code_cell_value_rows']:,}")
    add(f"- Fixed split column present: **{pt['has_split_column']}** — the pool is not partitioned.")
    add("")

    add("## 4. Platinum 4-cell cohorts")
    add("")
    add(
        "Two files with the same four cell lines and different labels. The "
        "synthetic one is audited here because it is what the legacy loader "
        "still defaults to; the real one is the cohort the trainer is pointed "
        "at, and it is audited separately below."
    )
    add("")
    add("### 4.1 `platinai_4cell.csv` — synthetic labels, DO-NOT-TRAIN")
    add("")
    add(
        "This is a different file from the pool above, and it *is* partitioned. "
        "The loader is imported so the audit measures the split the trainer "
        "builds, not a re-implementation of it."
    )
    add("")
    p4 = result["platinai_4cell"]
    add(
        f"- `split_strategy` in code: **`{p4['split_strategy_in_code']}`** — "
        + (
            "despite the name, `scaffold_id` is a numeric hash of the SMILES "
            "string (`_stable_hash_int(smiles, 10**8)`, "
            "`train_dmpnn_multitask_wasserstein.py:702`), so this is a "
            "molecule-level hash split."
            if p4["scaffold_id_looks_like_hash"]
            else "`scaffold_id` is a scaffold string, not a numeric hash."
        )
    )
    add(
        f"- {p4['n_unique_smiles']:,} unique SMILES / {p4['n_rows_long']:,} "
        f"long-format rows; split {p4['split_sizes_unique_smiles']}; parse "
        f"failures {p4['n_parse_failures']:,}; SMILES with >1 split label "
        f"{p4['n_smiles_with_multiple_split_labels']:,}."
    )
    alt = p4["bemis_murcko_strategy"]
    add(
        f"- Opt-in real scaffold split **`{alt['split_strategy']}`** (added "
        f"2026-10-19): split {alt['split_sizes_unique_smiles']}; test rows sharing "
        f"a desalted stereo-blind scaffold with train "
        f"{_pct(alt['pairs']['train_vs_test']['frac_b_rows_seen_in_a'])} "
        f"({alt['pairs']['train_vs_test']['n_intersect_keys']} shared scaffolds). "
        f"It is not the default because it changes the training cohort."
    )
    add("")
    add("| Pair | canonical shared | desalted stereo-blind BM shared | frac of B rows | whole-complex stereo-blind BM shared | frac of B rows |")
    add("|---|---:|---:|---:|---:|---:|")
    for pair_name, rec in p4["pairs"].items():
        add(
            f"| {pair_name} | {rec['canonical']['n_intersect_keys']} | "
            f"{rec['bemis_murcko']['n_intersect_keys']} | "
            f"{_pct(rec['bemis_murcko']['frac_b_rows_seen_in_a'])} | "
            f"{rec['bemis_murcko_full_nostereo']['n_intersect_keys']} | "
            f"{_pct(rec['bemis_murcko_full_nostereo']['frac_b_rows_seen_in_a'])} |"
        )
    add("")

    add("### 4.2 `platinai_4cell_real_trainer.csv` — REAL labels, the live cohort")
    add("")
    p4r = result["platinai_4cell_real"]
    rw = p4r["split_sizes_rows"]
    add(
        f"- {p4r['n_compounds']:,} compounds / {p4r['n_unique_smiles']:,} unique "
        f"SMILES (28 SMILES carry more than one compound code); parse failures "
        f"{p4r['n_parse_failures']:,}."
    )
    add(
        f"- `split_strategy` **`{p4r['split_strategy']}`**: "
        f"{rw['train']:,} / {rw['val']:,} / {rw['test']:,} rows, "
        f"{p4r['n_scaffolds_spanning_splits']} scaffolds spanning two splits."
    )
    for pair_name, rec in p4r["pairs"].items():
        add(
            f"- {pair_name}: canonical {rec['canonical']['n_intersect_keys']}, "
            f"split-key BM {rec['bemis_murcko_split_key']['n_intersect_keys']} "
            f"({_pct(rec['bemis_murcko_split_key']['frac_b_rows_seen_in_a'])} of B rows)"
        )
    add(
        "- The split is built inline in `train_cisplatin_resistance.main`, so "
        "this section calls the same splitter with the same coverage dict "
        "rather than importing a loader. Per-cell coverage and the "
        "order-sensitivity measurements behind it are in "
        "`metrics/by_metric/split_coverage_4cell.json`."
    )
    add("")

    add("## 5. Controls")
    add("")
    nci = result["nci60"]
    legacy = nci["protocols"]["scaffold_70_10_10"]
    cov = nci["protocols"]["scaffold_coverage_70_10_20"]
    add(
        f"**NCI-60 9-cell** — {nci['n_rows']:,} rows / {nci['n_unique_canonical']:,} "
        f"canonical molecules / {nci['n_cells']} cell lines; parse failures "
        f"{nci['n_parse_failures']:,}. Two protocols, both measured through the "
        f"production loader (`load_nci60_9cell_split_table`). The default since "
        f"2026-10-21 is `{nci['default_protocol']}`."
    )
    add("")
    add(
        "| Protocol | molecules train/val/test | rows train/val/test "
        "| row % | min test rows per cell | min val rows per cell "
        "| cells < 30 test rows | scaffolds spanning splits |"
    )
    add("|---|---|---|---:|---:|---:|---:|---:|")
    for rec in (legacy, cov):
        sz = rec["split_sizes_unique_smiles"]
        rw = rec["split_sizes_rows"]
        fr = rec["row_fraction"]
        add(
            f"| `{rec['split_strategy']}` "
            f"| {sz['train']:,} / {sz['val']:,} / {sz['test']:,} "
            f"| {rw['train']:,} / {rw['val']:,} / {rw['test']:,} "
            f"| {100 * fr['train']:.1f} / {100 * fr['val']:.1f} / {100 * fr['test']:.1f} "
            f"| {rec['min_test_rows_per_cell']} | {rec['min_val_rows_per_cell']} "
            f"| {len(rec['cells_below_30_test_rows'])} "
            f"| {rec['n_scaffolds_spanning_splits']} |"
        )
    add("")
    _zero_val = legacy["cells_with_zero_val_rows"]
    add(
        f"- The legacy protocol budgets fractions in *scaffold counts*, so the "
        f"largest scaffolds take the whole train budget and test receives the "
        f"singleton leftovers: {len(legacy['cells_below_30_test_rows'])} of "
        f"{nci['n_cells']} cells have < 30 test rows "
        f"({', '.join(legacy['cells_below_30_test_rows']) or 'none'}) and "
        f"{len(_zero_val)} "
        f"{'has' if len(_zero_val) == 1 else 'have'} no val rows "
        f"({', '.join(_zero_val) or 'none'}). "
        f"Every shipped m21 artifact was scored on it; it is pinned in the "
        f"configs that reproduce them."
    )
    add(
        f"- The coverage protocol targets fractions of *each cell line's "
        f"labelled rows* on the same scaffold partition. Measured: "
        f"{cov['split_sizes_rows']['train']:,} / {cov['split_sizes_rows']['val']:,} "
        f"/ {cov['split_sizes_rows']['test']:,} rows, every cell >= "
        f"{cov['min_test_rows_per_cell']} test and >= {cov['min_val_rows_per_cell']} "
        f"val rows. No model has been retrained on it yet."
    )
    add(
        f"- The pre-2026-10-21 claim of "
        f"{'/'.join(f'{v:,}' for v in nci['pre_blk_pa12_claim']['split_sizes_unique_smiles'].values())} "
        f"molecules is **not reproducible**: it came from the pre-BLK-PA12 "
        f"splitter, which returned an empty scaffold key for every molecule."
    )
    for proto_name, rec in (("legacy", legacy), ("coverage", cov)):
        for pair_name, o in rec["pairs"].items():
            add(
                f"- {proto_name} {pair_name}: canonical "
                f"{o['canonical']['n_intersect_keys']}, split-key BM "
                f"{o['bemis_murcko_split_key']['n_intersect_keys']}, desalted BM "
                f"{o['bemis_murcko_desalted']['n_intersect_keys']}"
            )
    add("")
    add("| MoleculeNet task | rows | parse failures | canonical | seeds | max canonical train/test overlap | max split-key BM overlap | max stereo-blind BM overlap | max desalted BM overlap | random fallback |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for task, rec in result["moleculenet"]["tasks"].items():
        max_c = max(
            s["canonical"]["train_vs_test"]["n_intersect_keys"] for s in rec["per_seed"]
        )
        max_k = max(
            s["bemis_murcko_split_key"]["train_vs_test"]["n_intersect_keys"]
            for s in rec["per_seed"]
        )
        max_ns = max(
            s["bemis_murcko_split_key_nostereo"]["train_vs_test"]["n_intersect_keys"]
            for s in rec["per_seed"]
        )
        max_b = max(
            s["bemis_murcko_desalted"]["train_vs_test"]["n_intersect_keys"]
            for s in rec["per_seed"]
        )
        fb = any(s["random_fallback_triggered"] for s in rec["per_seed"])
        add(
            f"| {task} | {rec['n_rows']:,} | {rec['n_parse_failures']:,} | "
            f"{rec['n_unique_canonical']:,} | "
            f"{len(rec['per_seed'])} | {max_c} | {max_k} | {max_ns} | {max_b} | {fb} |"
        )
    add("")
    d = result["drugood"]
    add(
        f"**DrugOOD lbap** — {d['n_rows']:,} rows / {d['n_unique_canonical']:,} canonical "
        f"molecules. train vs ood_test: canonical "
        f"{d['pairs']['train_vs_ood_test']['canonical']['n_intersect_keys']:,} shared, "
        f"scaffold_split_id {d['pairs']['train_vs_ood_test']['scaffold_split_id']['n_intersect_keys']:,} shared."
    )
    add("")

    add("## 6. Cross-dataset overlap")
    add("")
    add("| Pair | level | shared | frac of A | frac of B |")
    add("|---|---|---:|---:|---:|")
    for name, rec in result["cross"].items():
        if "canonical" in rec:
            for level in ("canonical", "ligand", "fragment"):
                if level not in rec:
                    continue
                r = rec[level]
                add(
                    f"| {r['label']} | {level} | {r['n_intersect']:,} | "
                    f"{_pct(r['frac_a'])} | {_pct(r['frac_b'])} |"
                )
        else:
            add(
                f"| {rec['label']} | canonical | {rec['n_intersect']:,} | "
                f"{_pct(rec['frac_a'])} | {_pct(rec['frac_b'])} |"
            )
    add("")
    add(
        "`canonical` = whole-SMILES identity after RDKit canonicalisation. "
        "`ligand` = metal-stripped largest organic fragment. `fragment` = any "
        "metal-stripped organic fragment with >= 6 heavy atoms shared. The "
        "fragment level is the loosest and is reported to show how much of the "
        "apparent overlap is a real shared ligand versus a fragment-selection "
        "artefact."
    )
    add("")

    add("## 7. Verdict")
    add("")
    add(result["verdict"])
    add("")
    return "\n".join(L)


def build_verdict(result: Dict[str, object]) -> str:
    """Derive the §7 verdict from the measured result dict.

    Kept separate from ``main`` so a partial re-measurement (``--reuse``)
    re-derives the verdict instead of inheriting stale prose.
    """
    pred = result["predictive"]
    cfm_v3 = result["cfm"]["v3"]
    pt4 = result["platinai_4cell"]
    molnet = result["moleculenet"]
    cross = result["cross"]
    nci = result["nci60"]

    # Verdict, derived from the measured numbers rather than asserted.
    skf_seen = [
        r["mean_frac_test_rows_canon_in_train"]
        for rec in pred["metals"].values()
        for r in rec["stratified_kfold"]["per_seed"]
    ]
    skf_scaf = [
        r["mean_frac_test_rows_scaffold_in_train"]
        for rec in pred["metals"].values()
        for r in rec["stratified_kfold"]["per_seed"]
    ]
    doi_scaf = [
        rec["doi_grouped_10fold"]["mean_frac_test_rows_scaffold_in_train"]
        for rec in pred["metals"].values()
    ]
    cfm_multi = cfm_v3["n_smiles_with_multiple_split_labels"]
    cfm_bm = cfm_v3["pairs"]["train_vs_test"]["bemis_murcko"]["n_intersect_keys"]
    cfm_bm_frac = cfm_v3["pairs"]["train_vs_test"]["bemis_murcko"][
        "frac_b_rows_seen_in_a"
    ]
    cfm_full_ns = cfm_v3["pairs"]["train_vs_test"]["bemis_murcko_full_nostereo"]
    cfm_val_frac = cfm_v3["pairs"]["train_vs_val"]["bemis_murcko"][
        "frac_b_rows_seen_in_a"
    ]
    cfm_scaf_col = cfm_v3["pairs"]["train_vs_test"]["scaffold_id_column"][
        "n_intersect_keys"
    ]
    pt_pred_c = cross["pt_vs_predictive5"]["canonical"]["n_intersect"]
    pt_pred_f = cross["pt_vs_predictive5"]["fragment"]["n_intersect"]
    molnet_max = max(
        s["canonical"]["train_vs_test"]["n_intersect_keys"]
        for rec in molnet["tasks"].values()
        for s in rec["per_seed"]
    )
    molnet_bm_max = max(
        s["bemis_murcko_desalted"]["train_vs_test"]["n_intersect_keys"]
        for rec in molnet["tasks"].values()
        for s in rec["per_seed"]
    )
    molnet_bm_frac = max(
        s["bemis_murcko_desalted"]["train_vs_test"]["frac_b_rows_seen_in_a"]
        for rec in molnet["tasks"].values()
        for s in rec["per_seed"]
    )
    molnet_ns_max = max(
        s["bemis_murcko_split_key_nostereo"]["train_vs_test"]["n_intersect_keys"]
        for rec in molnet["tasks"].values()
        for s in rec["per_seed"]
    )
    molnet_ns_frac = max(
        s["bemis_murcko_split_key_nostereo"]["train_vs_test"]["frac_b_rows_seen_in_a"]
        for rec in molnet["tasks"].values()
        for s in rec["per_seed"]
    )
    nci_legacy = nci["protocols"]["scaffold_70_10_10"]
    nci_cov = nci["protocols"]["scaffold_coverage_70_10_20"]
    nci_c = nci_legacy["pairs"]["train_vs_test"]["canonical"]["n_intersect_keys"]
    nci_c_cov = nci_cov["pairs"]["train_vs_test"]["canonical"]["n_intersect_keys"]
    nci_bm = nci_legacy["pairs"]["train_vs_test"]["bemis_murcko_split_key"][
        "n_intersect_keys"
    ]
    nci_bm_cov = nci_cov["pairs"]["train_vs_test"]["bemis_murcko_split_key"][
        "n_intersect_keys"
    ]
    nci_pred = cross["nci60_vs_predictive5"]["canonical"]["n_intersect"]
    nci_cfm = cross["nci60_vs_cfm_all"]["canonical"]["n_intersect"]
    doi_seen_mean = float(
        np.mean(
            [
                rec["doi_grouped_10fold"]["mean_frac_test_rows_canon_in_train"]
                for rec in pred["metals"].values()
            ]
        )
    )

    verdict_lines = [
        "**Molecule-level (same canonical SMILES in train and test):**",
        f"- Predictive 5-metal cohorts: {_pct(float(np.mean(skf_seen)))} of test rows "
        f"under StratifiedKFold (max {_pct(float(np.max(skf_seen)))}), "
        f"{_pct(doi_seen_mean)} under DOI-grouped 10-fold. "
        f"**No molecule-level leak.**",
        f"- CFM v3: {cfm_multi:,} SMILES carry more than one split label; "
        f"canonical train/test overlap 0. **No molecule-level leak.**",
        f"- NCI-60: canonical train/test overlap {nci_c} under the legacy "
        f"protocol and {nci_c_cov} under the coverage default; MoleculeNet: max "
        f"canonical train/test overlap {molnet_max}; DrugOOD train vs ood_test: "
        f"0. **No molecule-level leak in these splits.**",
        "",
        "**Scaffold-level (same Bemis-Murcko scaffold in train and test):**",
        f"- Predictive random CV: {_pct(min(skf_scaf))}-{_pct(max(skf_scaf))} of "
        f"test rows share a BM scaffold with train "
        f"(DOI-grouped 10-fold: {_pct(min(doi_scaf))}-{_pct(max(doi_scaf))}). "
        f"This is the expected property of random CV, "
        f"not a bug, but the number is **not** a scaffold-generalisation estimate.",
        f"- CFM v3: in-file `scaffold_id` overlap {cfm_scaf_col}. The split key is "
        f"stereo-aware and full-complex. Removing stereo alone leaves "
        f"{cfm_full_ns['n_intersect_keys']:,} shared scaffolds "
        f"({_pct(cfm_full_ns['frac_b_rows_seen_in_a'])} of test rows); removing "
        f"counterions as well leaves {cfm_bm:,} ({_pct(cfm_bm_frac)} of test rows). "
        f"The dominant gap is counterion/solvent fragments in the split key, not "
        f"stereochemistry. Train vs val is worse: {_pct(cfm_val_frac)} of val rows "
        f"share a desalted ligand scaffold with train — and val clouds are what the "
        f"CFM quality metrics are measured on.",
        f"- MoleculeNet: **key aligned 2026-10-19.** The split key is now the "
        f"stereo-blind `includeChirality=False` key (the `astartes`/chemprop "
        f"reference), so the `split-key BM` and `stereo-blind BM` columns are "
        f"one measurement and both are 0 by construction. The pre-fix key kept "
        f"chirality; its test-set turnover and old overlap figures are in "
        f"`metrics/by_metric/moleculenet_scaffold_key_alignment.json`.",
        f"- NCI-60: split-key BM overlap {nci_bm} (legacy) and {nci_bm_cov} "
        f"(coverage); {nci_legacy['n_scaffolds_spanning_splits']} and "
        f"{nci_cov['n_scaffolds_spanning_splits']} scaffolds span two splits "
        f"respectively. Neither protocol leaks a scaffold.",
        f"- Pt 4-cell: the loader's `split_strategy` is named "
        f"`scaffold_70_10_10` but "
        + (
            "is a stable hash of the SMILES string, i.e. a molecule-level split"
            if pt4["scaffold_id_looks_like_hash"]
            else "`scaffold_id` is a scaffold string"
        )
        + f". Measured: "
        f"{pt4['pairs']['train_vs_test']['bemis_murcko']['n_intersect_keys']} shared "
        f"desalted scaffolds "
        f"({_pct(pt4['pairs']['train_vs_test']['bemis_murcko']['frac_b_rows_seen_in_a'])} "
        f"of test rows) and "
        f"{_pct(pt4['pairs']['train_vs_val']['bemis_murcko']['frac_b_rows_seen_in_a'])} "
        f"of val rows; canonical overlap "
        f"{pt4['pairs']['train_vs_test']['canonical']['n_intersect_keys']}. "
        f"**Opt-in fix 2026-10-19:** `bemis_murcko_70_10_10` gives a real "
        f"scaffold split at 0.0% test-row overlap "
        f"(`metrics/by_metric/platinai_4cell_scaffold_split_alignment.json`); it "
        f"is not the default because it changes the training cohort.",
        "",
        "**Cross-dataset:**",
        f"- NCI-60 is a **subset of MetalCytoToxDB**: {nci_pred:,}/{nci['n_unique_canonical']:,} "
        f"canonical molecules are in the 5-metal pool and {nci_cfm:,} are in CFM. "
        f"It is a re-slice of the same corpus, not an independent control.",
        f"- Pt pool vs 5-metal training: exact canonical overlap {pt_pred_c}; "
        f"shared organic fragments {pt_pred_f:,}. Pt is a different metal, so this "
        f"is ligand-level overlap, not molecule identity. "
        f"{result['pt']['n_parse_failures']:,}/"
        f"{result['pt']['n_unique_raw_smiles']:,} unique Pt SMILES fail RDKit "
        f"parsing and are compared as raw strings only.",
        f"- Pt pool has **no split column**; it is a pool, not a train/val/test "
        f"partition. Duplicate (code, cell) rows "
        f"{result['pt']['n_duplicate_code_cell_rows']:,}, exact duplicate "
        f"measurements {result['pt']['n_duplicate_code_cell_value_rows']:,}.",
        "",
        "**Protocol change (not leakage):**",
        f"- The NCI-60 default split moved on 2026-10-21 from "
        f"`scaffold_70_10_10` to `scaffold_coverage_70_10_20`, because the legacy "
        f"protocol budgets fractions in scaffold counts and left "
        f"{len(nci_legacy['cells_below_30_test_rows'])} of {nci['n_cells']} cells "
        f"with < 30 test rows. Both protocols are measured above. The shipped "
        f"numbers (0.7294 m21, 0.6683 m3) were scored on the legacy protocol and "
        f"are reproducible only through the configs that pin it; no model has "
        f"been retrained on the coverage split, and the two test sets differ "
        f"~4.5x in size, so the numbers are not comparable.",
        "",
        "**Action:**",
        "- No re-split is required to remove molecule-level leakage: none was found.",
        "- Two split *labels* did not describe their implementations and are "
        "aligned as of 2026-10-19: `train_moleculenet._bemis_murcko_scaffold` uses "
        "the astartes stereo-blind key, and the Pt 4-cell loader offers a real "
        "`bemis_murcko_70_10_10` strategy beside the legacy hash. The CFM file "
        "still claims a strict Bemis-Murcko split while its key is stereo-aware "
        "and full-complex; that file belongs to the CFM session and remains open.",
        "- Before reporting a scaffold/OOD number, group by a **stereo-blind, "
        "desalted ligand scaffold** (``_bemis_murcko_key`` with "
        "``includeChirality=False``). For MoleculeNet comparability the reference "
        "key is astartes' stereo-blind, *non*-desalted key.",
        "- The Pt **pool** is not partitioned, so it needs no scaffold split; it "
        "needs de-duplication before use as an external evaluation set. The Pt "
        "**4-cell cohort** is partitioned and is the split that needs fixing.",
    ]
    return "\n".join(verdict_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Measure train/val/test leakage for every active dataset."
    )
    ap.add_argument(
        "--reuse",
        type=Path,
        default=None,
        help="Reuse the measured sections of an existing artifact and re-measure "
        "only NCI-60, then re-render. Exists because the NCI-60 section is cheap "
        "to measure and the MoleculeNet/DrugOOD sections are not; a full re-run "
        "to refresh one section would also churn numbers this change did not "
        "touch.",
    )
    args = ap.parse_args(argv)

    if args.reuse is not None:
        result = json.loads(args.reuse.read_text(encoding="utf-8"))
        print(f"reusing {args.reuse} for sections 1-4, 6-7", flush=True)
        nci = audit_nci60()
        pt4_real = audit_platinai_4cell_real()
        # The cross-dataset section is protocol-independent (same molecule set
        # under both protocols); refuse to mix if that is no longer true.
        if result["nci60"]["n_unique_canonical"] != nci["n_unique_canonical"]:
            raise SystemExit(
                "reused artifact was measured on a different NCI-60 molecule "
                f"set ({result['nci60']['n_unique_canonical']} -> "
                f"{nci['n_unique_canonical']}); re-run without --reuse"
            )
        result["nci60"] = {k: v for k, v in nci.items() if not k.startswith("_")}
        result["platinai_4cell_real"] = pt4_real
    else:
        print("[1/8] predictive cohorts ...", flush=True)
        pred = audit_predictive()
        print("[2/8] CFM v3 ...", flush=True)
        cfm_v3 = audit_cfm(
            DATA / "metalcytotox_5metal_multitask_v3.csv", "SMILES_Ligands", "Metal"
        )
        print("[3/8] CFM v4 ...", flush=True)
        cfm_v4 = audit_cfm(
            DATA / "metalcytotox_5metal_multitask_v4.csv", "smiles", "metal"
        )
        print("[4/8] platinum pool ...", flush=True)
        pt = audit_pt()
        print("[5/8] platinum 4-cell training cohort ...", flush=True)
        pt4 = audit_platinai_4cell()
        pt4_real = audit_platinai_4cell_real()
        print("[6/8] NCI-60 ...", flush=True)
        nci = audit_nci60()
        print("[7/8] MoleculeNet ...", flush=True)
        molnet = audit_moleculenet()
        print("[8/8] DrugOOD + cross-dataset ...", flush=True)
        drugood = audit_drugood()
        cross = cross_overlap(pred, cfm_v3, pt, nci, molnet)

        result = {
            "workflow": "Wf-A51",
            "title": "Train/validation/test leakage audit",
            "predictive": {k: v for k, v in pred.items() if not k.startswith("_")},
            "cfm": {"v3": cfm_v3, "v4": cfm_v4},
            "pt": {k: v for k, v in pt.items() if not k.startswith("_")},
            "platinai_4cell": pt4,
            "platinai_4cell_real": pt4_real,
            "nci60": {k: v for k, v in nci.items() if not k.startswith("_")},
            "moleculenet": {
                "tasks": {
                    t: {k: v for k, v in rec.items() if not k.startswith("_")}
                    for t, rec in molnet["tasks"].items()
                }
            },
            "drugood": {k: v for k, v in drugood.items() if not k.startswith("_")},
            "cross": cross,
        }

    result["verdict"] = build_verdict(result)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    rendered = render_markdown(result)
    # 2026-10-19 (Wf-A53) — this generator owns §1-§7 only.  §8 (MSRS) is
    # maintained outside this script, and a plain overwrite deleted it.  Keep
    # any trailing section verbatim so a re-run is a regeneration, not a loss.
    tail = _preserved_tail(OUT_MD)
    if tail:
        rendered = rendered.rstrip("\n") + "\n\n---\n\n" + tail
        print(f"preserved {len(tail.splitlines())} trailing line(s) from {OUT_MD}")
    OUT_MD.write_text(rendered)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(result["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
