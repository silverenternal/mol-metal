#!/usr/bin/env python3
"""Assemble the in-house PA15 metric (static metal conditioning, 24 h probe cohort).

BLK-PA15 asks whether an explicit metal input recovers any of Paper A's
cross-metal (LOMO) collapse — i.e. whether the paper can offer a *method* and not
only a benchmark.  The published number is
``metrics/by_metric/pa15_metalcond_lomo.json`` (chemprop v2).  The project red
line requires the predictive line to be backed by the self-developed D-MPNN, so
this synthesis rebuilds the same four-arm comparison from in-house ex1 runs.

Arm mapping (chemprop -> in-house MetalAwareDMPNN), all zero-shot (no adaptation):

  * ``none``  — metal-blind baseline (``metal_injection='none'``);
  * ``token`` — metal vector concatenated at the pooled READOUT
                (``metal_injection='readout'``, base 18-d metal features);
  * ``desc``  — same, plus the transferable descriptor blocks (``--rich-metal``
                + ``--chem-metal``);
  * ``film``  — per-layer FiLM modulation (``metal_injection='film'``).

The comparison is **paired**: for each of the 5 metals x 5 seeds = 25
(metal, seed) cells the conditioned arm's cross AUC pairs with the baseline's on
the identical split.  Significance is a paired bootstrap over those 25 cells
(10,000 resamples, seed 42) with a one-sided p for "conditioning helps", then
Holm-Bonferroni across the three conditioned arms.

Capacity note (honest): the chemprop reference used d_h=300 depth=3; the
in-house D-MPNN is hidden=128 depth=3.

Usage
-----
::

    .venv/bin/python molmetal/scripts/synthesize_pa15_inhouse.py \
        --out-metric metrics/by_metric/pa15_inhouse_metalcond_lomo.json
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

ARMS: Dict[str, str] = {
    "none": "metrics/by_metric/ex1_lomo_24h_none.json",
    "token": "metrics/by_metric/ex1_pa15_24h_token.json",
    "desc": "metrics/by_metric/ex1_pa15_24h_desc.json",
    "film": "metrics/by_metric/ex1_pa15_24h_film.json",
}
CHEMPROP_PA15 = "metrics/by_metric/pa15_metalcond_lomo.json"
METALS = ("Ru", "Ir", "Rh", "Os", "Re")
SEEDS = (42, 43, 44, 49, 123)

N_BOOTSTRAP = 10000
BOOT_SEED = 42


def _load(rel: str) -> Dict[str, object]:
    return json.loads((PROJECT_ROOT / rel).read_text())


def _cross_cells(ex1: Dict[str, object]) -> Dict[str, Dict[int, float]]:
    """``{metal: {seed: zero_shot_auc}}`` for the single arm in this ex1 report."""
    out: Dict[str, Dict[int, float]] = {m: {} for m in METALS}
    for r in ex1["per_run"]:  # type: ignore[index]
        out[str(r["metal"])][int(r["seed"])] = float(r["zero_shot_auc"])
    return out


def _paired_bootstrap(delta: np.ndarray, n_boot: int, seed: int) -> Dict[str, object]:
    """Paired bootstrap over the (metal, seed) cells (mirrors pa15_metalcond_analysis)."""
    rng = np.random.default_rng(seed)
    n = delta.size
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(delta[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_greater = float(np.mean(means <= 0.0))
    return {
        "n_cells": int(n),
        "n_boot": int(n_boot),
        "mean_delta": float(delta.mean()),
        "delta_std": float(delta.std(ddof=1)) if n > 1 else 0.0,
        "ci95_delta": [float(lo), float(hi)],
        "ci_excludes_zero": bool(float(lo) > 0.0 or float(hi) < 0.0),
        "p_one_sided_greater": p_greater,
        "p_one_sided_less": 1.0 - p_greater,
    }


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
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/pa15_inhouse_metalcond_lomo.json",
    )
    args = p.parse_args(argv)

    missing = [a for a, rel in ARMS.items() if not (PROJECT_ROOT / rel).exists()]
    if missing:
        raise SystemExit(
            f"missing ex1 arm artifact(s) for {missing}; run the corresponding "
            f"configs/ex1_*_24h*.yaml first (MISSING != dropped)"
        )

    reports = {arm: _load(rel) for arm, rel in ARMS.items()}
    cross = {arm: _cross_cells(r) for arm, r in reports.items()}
    base = cross["none"]

    arm_cross_summary: Dict[str, object] = {}
    for arm in ARMS:
        per_metal = {
            m: float(np.mean([cross[arm][m][s] for s in SEEDS])) for m in METALS
        }
        arm_cross_summary[arm] = {
            "mean_across_metals": float(np.mean(list(per_metal.values()))),
            "per_metal_mean": per_metal,
            "per_metal_std": {
                m: float(np.std([cross[arm][m][s] for s in SEEDS])) for m in METALS
            },
        }

    conditioned = ("token", "desc", "film")
    deltas: Dict[str, Dict[str, object]] = {}
    per_metal_delta: Dict[str, Dict[str, float]] = {}
    pvals: Dict[str, float] = {}
    for arm in conditioned:
        d = np.asarray(
            [cross[arm][m][s] - base[m][s] for m in METALS for s in SEEDS],
            dtype=np.float64,
        )
        bt = _paired_bootstrap(d, args.n_bootstrap, args.boot_seed)
        deltas[arm] = bt
        pvals[arm] = float(bt["p_one_sided_greater"])
        per_metal_delta[arm] = {
            m: float(np.mean([cross[arm][m][s] - base[m][s] for s in SEEDS]))
            for m in METALS
        }

    holm_adj = _holm(pvals)
    holm = {
        "correction": "Holm-Bonferroni (family = 3 conditioned arms vs the same baseline)",
        "n_family": len(pvals),
        "p_one_sided_raw": pvals,
        "p_one_sided_holm": holm_adj,
        "significant_after_holm_0.05": [a for a, q in holm_adj.items() if q < 0.05],
    }

    best_arm = max(conditioned, key=lambda a: deltas[a]["mean_delta"])
    best = deltas[best_arm]
    if best["ci95_delta"][0] > 0.0 and holm_adj[best_arm] < 0.05:
        verdict = (
            f"CONDITIONING HELPS: the {best_arm} arm beats the metal-blind "
            f"baseline by {best['mean_delta']:+.4f} cross-metal AUC "
            f"(paired bootstrap 95% CI [{best['ci95_delta'][0]:+.4f}, "
            f"{best['ci95_delta'][1]:+.4f}], Holm p={holm_adj[best_arm]:.4f})."
        )
        contributes_method = True
    else:
        verdict = (
            f"CONDITIONING DOES NOT RECOVER THE COLLAPSE: the best arm "
            f"({best_arm}) moves cross-metal AUC by {best['mean_delta']:+.4f} "
            f"(paired bootstrap 95% CI [{best['ci95_delta'][0]:+.4f}, "
            f"{best['ci95_delta'][1]:+.4f}], Holm p={holm_adj[best_arm]:.4f}); the "
            f"cross-metal collapse is intrinsic to the metal axis, not an "
            f"artefact of an unconditioned scorer."
        )
        contributes_method = False

    chem_ref: Dict[str, object] = {"checked": False}
    chem_path = PROJECT_ROOT / CHEMPROP_PA15
    if chem_path.exists():
        chem = json.loads(chem_path.read_text())
        chem_means = {
            a: v.get("mean_across_metals")
            for a, v in chem.get("cross_metal", {}).items()
        }
        chem_ref = {
            "checked": True,
            "reference_artifact": CHEMPROP_PA15,
            "reference_only": True,
            "chemprop_mean_across_metals": chem_means,
            "note": (
                "chemprop v2 is a REFERENCE scorer here, not the paper's model "
                "(project red line: the predictive line is the in-house D-MPNN)."
            ),
        }

    out = {
        "metric": "pa15_inhouse_metalcond_lomo",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/synthesize_pa15_inhouse.py",
        "description": (
            "Metal-conditioning ablation on the 24 h probe cohort under the "
            "ligand-disjoint leave-one-metal-out protocol, with the in-house "
            "D-MPNN (migration of BLK-PA15 off the chemprop v2 reference).  "
            "Four arms (none / token / desc / film) share cohort, split and "
            "seeds; a paired bootstrap over the 5 metals x 5 seeds cells tests "
            "whether an explicit metal input recovers cross-metal transfer."
        ),
        "model": "MetalAwareDMPNN, hidden=128 depth=3 dropout=0.1, epochs=50, "
        "zero-shot (no adaptation)",
        "protocol": {
            "cohort": "MetalCytotoxDataset + CytotoxFilter(24h, ic50>=0.01uM) "
            "+ collapse_duplicates, per metal",
            "split": "leave-one-metal-out, ligand-disjoint (held-out metal's "
            "ligands dropped from the train pool)",
            "query": "entire held-out metal cohort (no_adapt=true)",
            "seeds": list(SEEDS),
            "metals": list(METALS),
            "epochs": 50,
            "significance": "paired bootstrap over 25 (metal, seed) cells, "
            f"{args.n_bootstrap} resamples, seed {args.boot_seed}; "
            "Holm-Bonferroni over the 3 conditioned arms",
        },
        "arms": {
            arm: {
                "inhouse_metal_injection": {
                    "none": "none", "token": "readout",
                    "desc": "readout + rich_metal + chem_metal", "film": "film",
                }[arm],
                "artifact": ARMS[arm],
            }
            for arm in ARMS
        },
        "cross_metal": arm_cross_summary,
        "paired_delta_vs_none": deltas,
        "paired_delta_per_metal": per_metal_delta,
        "holm": holm,
        "chemprop_reference": chem_ref,
        "contributes_method": bool(contributes_method),
        "verdict": verdict,
        "caveats": [
            "The in-house `token` arm concatenates the projected metal VECTOR "
            "(18-d base metal features: identity + oxidation state + charge) at "
            "the readout — the closest in-house analogue of chemprop's learnable "
            "metal token.",
            "The `desc` arm adds the transferable descriptor blocks "
            "(--rich-metal + --chem-metal).  The chem-metal block is all-zero if "
            "molmetal/data/metal_chemistry_5.csv is absent, in which case the "
            "arm degenerates to token.",
            "Rh (n=51), Os (n=36) and Re (n=74) test folds are small; per-metal "
            "deltas on those cells carry large variance.",
            "Capacity is NOT matched to the chemprop reference (128 vs 300); the "
            "comparison is reported as a scorer/representation result.",
            "The in-house model is bit-reproducible on CPU (OMP_NUM_THREADS=1), "
            "so the 5-seed spread here reflects split/init variance, not run "
            "nondeterminism.",
        ],
        "honest_framing_note": (
            "Every number is MEASURED from the four ex1 arm reports on disk; the "
            "verdict and contributes_method flag are derived from the bootstrap "
            "CI and the Holm-adjusted p, not asserted."
        ),
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    dest = Path(args.out_metric)
    atomic_write_file(dest, json.dumps(out, indent=2, default=float, ensure_ascii=False))

    print(f"{'arm':>6}  cross-mean   delta-vs-none   p1(H1:>0)  holm")
    for arm in ("none",) + conditioned:
        cm = arm_cross_summary[arm]["mean_across_metals"]
        if arm == "none":
            print(f"{arm:>6}  {cm:.4f}       —")
        else:
            d = deltas[arm]
            print(
                f"{arm:>6}  {cm:.4f}   {d['mean_delta']:+.4f} "
                f"[{d['ci95_delta'][0]:+.4f},{d['ci95_delta'][1]:+.4f}]  "
                f"{d['p_one_sided_greater']:.4f}    {holm_adj[arm]:.4f}"
            )
    print(f"verdict: {verdict}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
