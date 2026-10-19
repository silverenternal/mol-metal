#!/usr/bin/env python3
"""MET-10 — dataset card + leakage audit numbers for the strong-model corpus.

The existing ``docs/DATA_LEAKAGE_AUDIT.md`` covers MoleculeNet, CFM, Pt and
NCI-60 but not the corpus the strong model is trained on (MSRS potency).  This
script measures that corpus so the card and the audit can quote numbers that
have a producer.

What is measured
----------------
* corpus size, endpoint/censoring breakdown, metal/cell/ligand/DOI counts;
* for each of the four MET-2 protocols: folds, groups, rows evaluated, and the
  fraction of test rows whose ligand / DOI / scaffold / cell line was also seen
  in training (the leakage quantity that protocol is supposed to control);
* duplicate keys on ``(ligand, metal, cell, DOI, time)`` and whether any
  duplicate crosses a fold under any protocol;
* the de-duplication and tie rules as implemented, not as intended.

Output: ``metrics/by_metric/msrs_data_card.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_external_validation import murcko_scaffold  # noqa: E402
from molmetal.scripts.msrs_generalization import scheme_folds  # noqa: E402
from molmetal.scripts.msrs_joint_representation import ENDPOINT  # noqa: E402
from molmetal.scripts.msrs_surface_model import _sha256  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

PROTOCOLS = ("ligand", "doi", "scaffold", "year")
KEYS = ["ligand_canonical", "metal", "cell_line_key", "doi", "time_h"]


def cross_fold_rate(
    values: np.ndarray, fold: np.ndarray
) -> Dict[str, object]:
    """Fraction of evaluated test rows whose key also appears in train.

    ``fold < 0`` rows train in every fold and are never evaluated; they are
    included in the train pool, matching ``fit_arm``.
    """
    seen = 0
    total = 0
    per_fold = []
    for k in sorted(set(fold[fold >= 0].tolist())):
        te = fold == k
        tr = fold != k
        train_keys = set(values[tr].tolist())
        hit = int(np.isin(values[te], list(train_keys)).sum())
        seen += hit
        total += int(te.sum())
        per_fold.append(
            {
                "fold": int(k),
                "n_test": int(te.sum()),
                "n_seen_in_train": hit,
                "rate": round(hit / max(int(te.sum()), 1), 4),
            }
        )
    return {
        "n_test_rows": total,
        "n_seen_in_train": seen,
        "rate": round(seen / max(total, 1), 4),
        "per_fold": per_fold,
    }


def duplicate_audit(sub: pd.DataFrame) -> Dict[str, object]:
    grouped = sub.groupby(KEYS, dropna=False)["value_log10"]
    sizes = grouped.size()
    unique = grouped.nunique()
    repeated = sizes[sizes > 1]
    conflicting = (sizes > 1) & (unique > 1)
    identical_cols = KEYS + ["value_log10"]
    return {
        "n_rows": int(len(sub)),
        "n_unique_keys": int(len(sizes)),
        "n_keys_repeated": int(len(repeated)),
        "n_rows_in_repeated_keys": int(repeated.sum()),
        "n_keys_conflicting_values": int(conflicting.sum()),
        "n_rows_in_conflicting_keys": int(sizes[conflicting].sum()),
        "n_exact_duplicate_rows": int(sub.duplicated(identical_cols).sum()),
        "rule": (
            "No de-duplication is applied.  Every row passing the filter is a "
            "training row; repeated measurements of the same key contribute "
            "multiple targets.  This is deliberate: the label is a measured "
            "value, and dropping a replicate would discard variance information."
        ),
    }


def duplicates_cross_folds(
    sub: pd.DataFrame, folds: Dict[str, np.ndarray]
) -> Dict[str, object]:
    """Does any duplicate key straddle a fold boundary?"""
    key_id = pd.factorize(
        sub["ligand_canonical"].astype(str)
        + "|"
        + sub["metal"].astype(str)
        + "|"
        + sub["cell_line_key"].astype(str)
        + "|"
        + sub["doi"].astype(str)
        + "|"
        + sub["time_h"].astype(str)
    )[0]
    out: Dict[str, object] = {}
    n_nan = int(np.sum(key_id < 0))
    for protocol, fold in folds.items():
        crossing = 0
        for g in np.unique(key_id):
            if g < 0:
                continue  # factorize's sentinel for an unparseable key
            rows = fold[key_id == g]
            rows = rows[rows >= 0]
            if len(set(rows.tolist())) > 1:
                crossing += 1
        out[protocol] = {"crossing_keys": int(crossing), "nan_keys": n_nan}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_data_card.json",
    )
    args = parser.parse_args(argv)

    raw = pd.read_parquet(args.long)
    censoring = raw["censoring"].astype(str)
    potency = raw[raw["endpoint"] == ENDPOINT]

    sub = raw[
        (raw["endpoint"] == ENDPOINT)
        & raw["value_log10"].notna()
        & (raw["censoring"].astype(str) == "none")
        & raw["ligand_main_fragment"].notna()
        & raw["ligand_canonical"].notna()
        & raw["oxidation_state"].notna()
    ].copy().reset_index(drop=True)

    folds: Dict[str, np.ndarray] = {}
    protocols: Dict[str, object] = {}
    for protocol in PROTOCOLS:
        fold, _, info = scheme_folds(sub, protocol)
        folds[protocol] = fold
        protocols[protocol] = {
            **info,
            "ligand_cross_fold": cross_fold_rate(
                sub["ligand_canonical"].astype(str).to_numpy(), fold
            ),
            "doi_cross_fold": cross_fold_rate(
                sub["doi"].astype(str).to_numpy(), fold
            ),
            "scaffold_cross_fold": cross_fold_rate(
                sub["ligand_canonical"].map(murcko_scaffold).astype(str).to_numpy(),
                fold,
            ),
            "cell_cross_fold": cross_fold_rate(
                sub["cell_line_key"].astype(str).to_numpy(), fold
            ),
        }

    dup = duplicate_audit(sub)
    dup_cross = duplicates_cross_folds(sub, folds)

    payload = {
        "artifact": "msrs_data_card",
        "version": "1.0",
        "requires_dual_split": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        },
        "corpus": {
            "n_rows_total": int(len(raw)),
            "endpoint_counts": {str(k): int(v) for k, v in raw["endpoint"].value_counts().items()},
            "censoring_counts": {str(k): int(v) for k, v in censoring.value_counts().items()},
            "potency_rows": int(len(potency)),
            "potency_censoring_counts": {
                str(k): int(v)
                for k, v in potency["censoring"].astype(str).value_counts().items()
            },
            "model_rows": int(len(sub)),
            "n_metals": int(sub["metal"].nunique()),
            "metal_counts": {str(k): int(v) for k, v in sub["metal"].value_counts().items()},
            "n_cell_lines": int(sub["cell_line_key"].nunique()),
            "n_ligands": int(sub["ligand_canonical"].nunique()),
            "n_dois": int(sub["doi"].nunique()),
            "n_time_points": int(sub["time_h"].nunique()),
            "value_units": sorted(set(map(str, sub["value_units"].unique()))),
            "metal_in_smiles_rows": int(
                sub["ligand_canonical"].astype(str)
                .str.contains(r"\[(Ru|Ir|Rh|Os|Re|Pt)", regex=True)
                .sum()
            ),
            "endpoint": ENDPOINT,
            "label": "log10(IC50 / uM); lower is more potent",
        },
        "filter": {
            "expression": (
                "endpoint == 'potency' AND value_log10.notna() AND "
                "censoring == 'none' AND ligand_main_fragment.notna() AND "
                "ligand_canonical.notna() AND oxidation_state.notna()"
            ),
            "rows_dropped": int(len(raw) - len(sub)),
        },
        "protocols": protocols,
        "duplicates": {**dup, "keys_crossing_folds": dup_cross},
        "tie_rule": (
            "Decision-layer rankings use pandas sort_values(kind='stable') on the "
            "predicted score, so exact ties are broken by original row order "
            "(dataset order).  For the pairwise Delta surface, metals are ordered "
            "lexicographically (Ir < Os < Re < Rh < Ru) before pairing, so a pair "
            "is never counted twice as (a,b) and (b,a)."
        ),
        "split_rules": {
            "ligand": "GroupKFold(5) on ligand_canonical",
            "doi": "GroupKFold(5) on doi",
            "scaffold": "GroupKFold(5) on the Bemis-Murcko scaffold of ligand_canonical",
            "year": (
                "LeaveOneGroupOut on the year parsed from the DOI; years with "
                "fewer than MIN_YEAR_ROWS rows are pooled; unparseable DOIs get "
                "fold -1 (train in every fold, never evaluated)"
            ),
        },
    }
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[met10] wrote {args.out}")
    for protocol in PROTOCOLS:
        p = protocols[protocol]
        print(
            f"  {protocol:8s} folds={p['n_folds']} groups={p['n_groups']} "
            f"ligand_x={p['ligand_cross_fold']['rate']} "
            f"doi_x={p['doi_cross_fold']['rate']} "
            f"scaffold_x={p['scaffold_cross_fold']['rate']} "
            f"cell_x={p['cell_cross_fold']['rate']}"
        )
    print(f"  duplicates: {dup['n_keys_repeated']} keys / {dup['n_rows_in_repeated_keys']} rows")
    print(f"  duplicate keys crossing folds: {dup_cross}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
