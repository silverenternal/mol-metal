#!/usr/bin/env python3
"""Assemble the in-house PA3 metric (D-MPNN cross-metal LOMO, 24 h probe cohort).

PA3 in Paper A is the "representation-dependence" paragraph: replacing the
fingerprint scorer with a stronger backbone on the 24 h probe cohort, under the
same ligand-disjoint leave-one-metal-out protocol.  The published number is
``metrics/by_metric/pa3_chemprop_lomo_probe_cohort.json`` (chemprop v2).  The
project red line requires the predictive line to be backed by the
self-developed D-MPNN, so this synthesis rebuilds the SAME three columns from
in-house artifacts:

  * ``cross``   — the D-MPNN trained on the 4 source metals, scored zero-shot on
                  the entire held-out metal cohort.  Source:
                  ``metrics/by_metric/ex1_lomo_24h_none.json`` (arm=none,
                  ``no_adapt=true``), 5-seed mean per metal.
  * ``within``  — the D-MPNN's within-metal 5-fold CV, from the committed
                  ``metrics/by_metric/dmpnn_within_metal_auc.json`` (24 h cohort,
                  same folds).  Carried, not recomputed.
  * the label-permutation null and its Holm correction — recomputed from the
    seed-42 per-ligand scores dumped by the ex1 run
    (``molmetal/reports/ex1_lomo_24h/none_scores.jsonl``) via the probe's own
    ``permutation_null`` (1,000 shuffles, seed 42), mirroring
    ``metal_activity_transfer_permutation_null.json``.

Capacity note (honest): the chemprop reference used ``d_h=300 depth=3``; the
in-house D-MPNN is ``hidden=128 depth=3``.  The difference is disclosed in the
artifact rather than hidden — a like-for-like capacity claim is not made.

Usage
-----
::

    .venv/bin/python molmetal/scripts/synthesize_pa3_inhouse.py \
        --out-metric metrics/by_metric/pa3_inhouse_lomo_probe_cohort.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.metal_activity_transfer_probe import permutation_null  # noqa: E402

EX1_CROSS = "metrics/by_metric/ex1_lomo_24h_none.json"
WITHIN_METRIC = "metrics/by_metric/dmpnn_within_metal_auc.json"
SCORES_DUMP = "molmetal/reports/ex1_lomo_24h/none_scores.jsonl"
CHEMPROP_PA3 = "metrics/by_metric/pa3_chemprop_lomo_probe_cohort.json"

METALS = ("Ru", "Ir", "Rh", "Os", "Re")
PERM_SEED = 42
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 2000
BOOT_SEED = 42

WHY = (
    "PA3 in-house migration: Paper A's 'representation-dependence' paragraph "
    "cited a chemprop v2 scorer.  The project red line requires the predictive "
    "line to be backed by the self-developed D-MPNN.  This artifact re-measures "
    "the same cross-metal LOMO column with the in-house MetalAwareDMPNN "
    "(metal_injection='none', zero-shot) on the same 24 h probe cohort and the "
    "same ligand-disjoint LOMO split."
)

CAVEATS: List[str] = [
    "Capacity is NOT matched: chemprop v2 used d_h=300 depth=3, the in-house "
    "D-MPNN uses hidden=128 depth=3.  Differences here are reported as a "
    "scorer/representation comparison, not a like-for-like capacity comparison.",
    "Per-metal test folds are small for Rh (n=51), Os (n=36) and Re (n=74).  "
    "The label-permutation null on the fixed seed-42 ranking is the reference "
    "(1,000 shuffles); at those sizes the null sd is ~0.09-0.11, so a nominal "
    "0.5 is not the right line.",
    "Only the CROSS column is seed-42 for the permutation null and the bootstrap "
    "CI; the reported cross AUCs are 5-seed means.  The CI's observed_auc is the "
    "seed-42 ranking, so it sits inside the seed spread around the 5-seed mean.",
    "The within-metal column is carried from the committed 24 h artifact "
    "(epochs=30, 5 split seeds); the cross column is epochs=50.  The two columns "
    "are not the same training budget — the same asymmetry exists in the "
    "chemprop reference, where within and cross are separate runs.",
]


def _load(rel: str) -> Dict[str, object]:
    return json.loads((PROJECT_ROOT / rel).read_text())


def _cross_cells(ex1: Dict[str, object]) -> Dict[str, Dict[int, float]]:
    """``{metal: {seed: zero_shot_auc}}`` from the ex1 per_run rows (arm=none)."""
    out: Dict[str, Dict[int, float]] = {m: {} for m in METALS}
    for r in ex1["per_run"]:  # type: ignore[index]
        if r["arm"] != "none":
            continue
        out[str(r["metal"])][int(r["seed"])] = float(r["zero_shot_auc"])
    return out


def _seed42_scores(dump: Path) -> Dict[str, Dict[str, List[float]]]:
    """``{metal: {labels, probs}}`` for seed 42 from the dumped JSONL."""
    out: Dict[str, Dict[str, List[float]]] = {}
    for line in dump.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if int(rec["seed"]) != PERM_SEED or rec["arm"] != "none":
            continue
        out[str(rec["metal"])] = {"labels": rec["y"], "probs": rec["p_zero"]}
    return out


def _verdict(p_above: float, p_below: float) -> str:
    if p_above < 0.05:
        return "above (sig)"
    if p_below < 0.05:
        return "below (sig)"
    return "in null"


def _holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n = len(items)
    adj: Dict[str, float] = {}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, float(p) * (n - rank)))
        adj[k] = running
    return adj


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cross", default=EX1_CROSS)
    p.add_argument("--within-metric", default=WITHIN_METRIC)
    p.add_argument("--scores-dump", default=SCORES_DUMP)
    p.add_argument("--chemprop-pa3", default=CHEMPROP_PA3)
    p.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    p.add_argument("--perm-seed", type=int, default=PERM_SEED)
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/pa3_inhouse_lomo_probe_cohort.json",
    )
    args = p.parse_args(argv)

    ex1 = _load(args.cross)
    within = _load(args.within_metric)
    cells = _cross_cells(ex1)

    dump_path = PROJECT_ROOT / args.scores_dump
    seed42 = _seed42_scores(dump_path) if dump_path.exists() else {}

    from molmetal.data.leakage_utils import bootstrap_auc

    comparison: Dict[str, Dict[str, object]] = {}
    cross_ci: Dict[str, Dict[str, object]] = {}
    p_two_sided: Dict[str, float] = {}

    for m in METALS:
        seeds = sorted(cells.get(m, {}).keys())
        cross_vals = [cells[m][s] for s in seeds]
        cross_mean = float(np.mean(cross_vals))
        cross_std = float(np.std(cross_vals))
        wrec = within["per_metal"].get(m, {})  # type: ignore[index]
        within_mean = (
            float(wrec["dmpnn_auc"]) if wrec.get("status") == "MEASURED" else None
        )
        within_std = (
            float(wrec.get("dmpnn_auc_across_folds_std"))
            if wrec.get("status") == "MEASURED"
            else None
        )

        entry: Dict[str, object] = {
            "n_test": int(wrec.get("n_unique", 0)),
            "n_seeds": len(seeds),
            "inhouse_cross_mean_auc": cross_mean,
            "inhouse_cross_std": cross_std,
            "inhouse_cross_per_seed": {str(s): cells[m][s] for s in seeds},
            "inhouse_within_mean_auc": within_mean,
            "inhouse_within_std": within_std,
            "inhouse_penalty_within_minus_cross": (
                None if within_mean is None else within_mean - cross_mean
            ),
        }

        if m in seed42:
            perm = permutation_null(
                seed42[m]["labels"], seed42[m]["probs"],
                args.n_permutations, args.perm_seed,
            )
            p_two_sided[m] = float(perm["p_two_sided"])
            entry["permutation_null_verdict"] = _verdict(
                float(perm["p_above_chance"]), float(perm["p_below_chance"])
            )
            entry["permutation_p_above_chance"] = float(perm["p_above_chance"])
            entry["permutation_null_sd"] = float(perm["null_sd"])
            _mean, _lo, _hi, _ = bootstrap_auc(
                seed42[m]["labels"], seed42[m]["probs"],
                args.n_bootstrap, args.boot_seed,
            )
            cross_ci[m] = {
                "n_boot": int(args.n_bootstrap),
                "observed_auc_seed42": float(perm["observed_auc"]),
                "ci95": [float(_lo), float(_hi)],
                "ci_excludes_chance": bool(float(_lo) > 0.5 or float(_hi) < 0.5),
            }
        else:
            entry["permutation_null_verdict"] = "NOT_COMPUTED"
            entry["permutation_null_reason"] = f"no seed-{PERM_SEED} dump for {m}"
        comparison[m] = entry

    _adj = _holm(p_two_sided) if p_two_sided else {}
    holm = {
        "correction": "Holm-Bonferroni (family = 5 held-out metals)",
        "n_family": len(p_two_sided),
        "p_two_sided_holm": _adj,
        "significant_after_holm_0.05": [m for m, q in _adj.items() if q < 0.05],
    }

    # Optional side-by-side against the chemprop PA3 canonical (reference only).
    chem_ref: Dict[str, object] = {"checked": False}
    chem_path = PROJECT_ROOT / args.chemprop_pa3
    if chem_path.exists():
        chem = json.loads(chem_path.read_text())
        chem_cross = {
            f["metal"]: float(f["mean_auc"]) for f in chem.get("per_fold", [])
        }
        chem_ref = {
            "checked": True,
            "reference_artifact": args.chemprop_pa3,
            "reference_only": True,
            "chemprop_cross_mean_auc": chem_cross,
            "inhouse_minus_chemprop_cross": {
                m: comparison[m]["inhouse_cross_mean_auc"] - chem_cross[m]
                for m in METALS
                if m in chem_cross
            },
            "note": (
                "chemprop v2 is a REFERENCE scorer here, not the paper's model "
                "(project red line: the predictive line is the in-house D-MPNN)."
            ),
        }

    out = {
        "metric": "pa3_inhouse_lomo_probe_cohort",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/synthesize_pa3_inhouse.py",
        "description": (
            "In-house D-MPNN cross-metal LOMO on the 24 h probe cohort — the "
            "migration of Paper A's PA3 'representation-dependence' column off "
            "the chemprop v2 reference onto the self-developed model."
        ),
        "model": "MetalAwareDMPNN(metal_injection='none'), hidden=128 depth=3 "
        "dropout=0.1, epochs=50",
        "protocol": {
            "cohort": "MetalCytotoxDataset + CytotoxFilter(24h, ic50>=0.01uM) "
            "+ collapse_duplicates, per metal",
            "split": "leave-one-metal-out, ligand-disjoint (held-out metal's "
            "ligands dropped from the train pool)",
            "cross_query": "entire held-out metal cohort (no_adapt=true; no "
            "support draw)",
            "within_query": "StratifiedKFold(5) per metal, 5 split seeds "
            "(carried from dmpnn_within_metal_auc.json)",
            "seeds": list(ex1.get("seeds", [])),
            "metals": list(METALS),
            "significance": f"label-permutation null ({args.n_permutations} "
            f"shuffles, seed {args.perm_seed}) on the seed-{PERM_SEED} ranking; "
            "Holm-Bonferroni over the 5-metal family",
        },
        "cross_metal": comparison,
        "cross_metal_bootstrap_ci": cross_ci,
        "permutation_null_holm": holm,
        "chemprop_reference": chem_ref,
        "why": WHY,
        "caveats": CAVEATS,
        "source_artifacts": {
            "cross": args.cross,
            "within": args.within_metric,
            "scores_dump": args.scores_dump,
        },
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    dest = Path(args.out_metric)
    atomic_write_file(dest, json.dumps(out, indent=2, default=float, ensure_ascii=False))

    for m in METALS:
        d = comparison[m]
        w = d["inhouse_within_mean_auc"]
        print(
            f"{m:>4}  cross {d['inhouse_cross_mean_auc']:.4f} "
            f"within {'  nan ' if w is None else f'{w:.4f}'}  "
            f"{d['permutation_null_verdict']}"
        )
    print(f"holm significant: {holm['significant_after_holm_0.05']}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
