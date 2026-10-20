#!/usr/bin/env python3
"""C6 — is the LOMO shortfall a model failure, or an unresolvable fold?

The headline in ``metal_lomo_auc_wasserstein.json`` is the equal-weight mean of
five leave-one-metal-out AUCs: 0.5575 against a 0.6 target.  That mean treats
Rh (n = 51 unique ligands), Os (36) and Re (74) as if their AUCs were measured
with the same precision as Ru (1,255) and Ir (534).

The project already has a rule for this — ``MIN_UNIQUE_FOR_GATE = 200`` in
``metal_activity_transfer_probe.py``: "below this many unique ligands a metal's
AUC is not resolvable and must not be gated".  The LOMO headline never applied
it.  This audit applies it mechanically and reports both aggregates, so the
shortfall can be attributed rather than assumed:

* ``all_metals`` — the published 0.5575, kept for continuity;
* ``resolvable_only`` — Ru + Ir, the only folds the rule allows into a gate.

Per fold it also reports the Hanley-McNeil standard error, which is the
sampling uncertainty of an AUC given n_pos / n_neg.  Seed spread is *not* that
uncertainty: five seeds on one fold share the same test set.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/lomo_resolvability_audit.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from molmetal.scripts.metal_activity_transfer_probe import MIN_UNIQUE_FOR_GATE  # noqa: E402
from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

METALS = ("Ru", "Ir", "Os", "Rh", "Re")
TARGET = 0.6


def hanley_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of a single AUC."""
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    return float(math.sqrt(max(var, 0.0)))


def fold_counts(csv_path: Path) -> Dict[str, Dict[str, int]]:
    """Unique ligands and label balance per metal, after the trainer's dedup."""
    import pandas as pd

    from molmetal.scripts.train_metal_lomo_wasserstein import collapse_ligand_duplicates

    frame = pd.read_csv(csv_path, low_memory=False)
    collapsed, _info = collapse_ligand_duplicates(frame)
    out: Dict[str, Dict[str, int]] = {}
    for metal, group in collapsed.groupby("Metal"):
        n_pos = int(group["active"].sum())
        out[str(metal)] = {
            "n_unique": int(len(group)),
            "n_pos": n_pos,
            "n_neg": int(len(group) - n_pos),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "data" / "metal_lomo.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "lomo_resolvability_audit.json",
    )
    args = parser.parse_args(argv)

    artifact = json.loads(args.artifact.read_text())
    per_run = np.asarray(artifact["per_run_auc"], dtype=float)
    n_seeds = len(per_run) // len(METALS)
    counts = fold_counts(args.csv)

    folds: List[Dict[str, object]] = []
    for index, metal in enumerate(METALS):
        runs = per_run[index * n_seeds : (index + 1) * n_seeds]
        info = counts.get(metal, {})
        mean = float(runs.mean())
        se = hanley_se(mean, int(info.get("n_pos", 0)), int(info.get("n_neg", 0)))
        folds.append(
            {
                "metal": metal,
                "n_unique": int(info.get("n_unique", 0)),
                "n_pos": int(info.get("n_pos", 0)),
                "n_neg": int(info.get("n_neg", 0)),
                "auc_mean": round(mean, 4),
                "auc_seed_sd": round(float(runs.std(ddof=1)), 4),
                "auc_seed_min": round(float(runs.min()), 4),
                "auc_seed_max": round(float(runs.max()), 4),
                "hanley_se": round(se, 4),
                "ci95_hanley": [round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)],
                "resolvable": bool(int(info.get("n_unique", 0)) >= MIN_UNIQUE_FOR_GATE),
            }
        )

    resolvable = [f for f in folds if f["resolvable"]]
    all_mean = round(float(np.mean([f["auc_mean"] for f in folds])), 4)
    res_mean = round(float(np.mean([f["auc_mean"] for f in resolvable])), 4)

    payload = {
        "artifact": "lomo_resolvability_audit",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/lomo_resolvability_audit.py",
            "git_commit": _git_commit(),
            "source_artifact": "metrics/by_metric/metal_lomo_auc_wasserstein.json",
            "cohort": "molmetal/data/metal_lomo.csv after collapse_ligand_duplicates",
        },
        "design": {
            "rule": "MIN_UNIQUE_FOR_GATE",
            "min_unique_for_gate": MIN_UNIQUE_FOR_GATE,
            "rule_source": "molmetal/scripts/metal_activity_transfer_probe.py:185",
            "target_overall_auc": TARGET,
            "n_seeds": n_seeds,
            "uncertainty": "Hanley-McNeil SE on n_pos/n_neg; seed spread is model variance, not sampling error",
        },
        "folds": folds,
        "aggregates": {
            "all_metals": {"value": all_mean, "n_metals": len(folds), "vs_target": round(all_mean - TARGET, 4)},
            "resolvable_only": {
                "value": res_mean,
                "metals": [f["metal"] for f in resolvable],
                "n_metals": len(resolvable),
                "vs_target": round(res_mean - TARGET, 4),
            },
        },
        "verdict": {
            "gate_target_met_all_metals": bool(all_mean >= TARGET),
            "gate_target_met_resolvable_only": bool(res_mean >= TARGET),
            "reading": "neither aggregate clears 0.6; the all-metal mean overstates the shortfall because three of five folds are unresolvable",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    for f in folds:
        print(
            f"[c6] {f['metal']:2s} n={f['n_unique']:5d} pos={f['n_pos']:4d} "
            f"auc={f['auc_mean']:.4f} ci={f['ci95_hanley']} resolvable={f['resolvable']}"
        )
    print(f"[c6] all_metals={all_mean} resolvable_only={res_mean} target={TARGET}")
    print(f"[c6] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
