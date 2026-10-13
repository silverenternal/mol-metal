#!/usr/bin/env python3
"""Assemble the PA15 metal-conditioning metric from the three arm reports.

BLK-PA15 asks whether an explicit metal input recovers any of Paper A's
cross-metal (LOMO) collapse — i.e. whether the paper can offer a *method* and
not only a benchmark.  Three arms share the cohort, the ligand-disjoint LOMO
split and the seeds, and differ only in the pooled head's metal input:

  * ``none``  — metal-agnostic (the PA3 scorer);
  * ``token`` — learnable per-metal embedding (held-out row is UNTRAINED in
    LOMO, so it is a random vector at test time);
  * ``desc``  — fixed periodic-table descriptor vector through a shared learned
    projection (the only arm whose held-out representation is known).

The comparison is **paired**: for each of the 5 metals x 5 seeds = 25
(metal, seed) cells the conditioned arm's cross AUC pairs with the baseline's
on the identical split.  Significance is a paired bootstrap over those 25
cells (10,000 resamples, seed 42) with a one-sided p for "conditioning helps",
then Holm-Bonferroni across the two conditioned arms.

The report also carries a baseline-reproduction check: the ``none`` arm's
per-metal cross mean is compared against the canonical PA3 metric on disk
(``metrics/by_metric/pa3_chemprop_lomo_probe_cohort.json``), which is a GPU
bf16 artifact and drifts ~0.01-0.03 per-seed run-to-run.

Usage
-----
::

    .venv/bin/python molmetal/scripts/pa15_metalcond_analysis.py \
        --out-metric metrics/by_metric/pa15_metalcond_lomo.json
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
    "none": "molmetal/reports/pa15_metalcond_lomo_none/metrics.json",
    "token": "molmetal/reports/pa15_metalcond_lomo_token/metrics.json",
    "desc": "molmetal/reports/pa15_metalcond_lomo_desc/metrics.json",
}
PA3_CANONICAL = "metrics/by_metric/pa3_chemprop_lomo_probe_cohort.json"
METALS = ("Ru", "Ir", "Rh", "Os", "Re")
SEEDS = (42, 43, 44, 49, 123)

N_BOOTSTRAP = 10000
BOOT_SEED = 42


def _load(rel: str) -> Dict[str, object]:
    return json.loads((PROJECT_ROOT / rel).read_text())


def _cross_cells(report: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    """``{metal: {seed_str: auc}}`` from a report's per_fold per_seed blocks."""
    out: Dict[str, Dict[str, float]] = {}
    for fold in report["per_fold"]:  # type: ignore[index]
        m = str(fold["metal"])
        out[m] = {
            str(s): float(a) for s, a in fold["per_seed_auc"].items()
        }
    return out


def _within_means(report: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m, v in report["within_metal"].items():  # type: ignore[index]
        if v.get("status") == "MEASURED":
            out[m] = float(v["mean_auc"])
    return out


def _paired_bootstrap(delta: np.ndarray, n_boot: int, seed: int) -> Dict[str, object]:
    """Paired bootstrap over the (metal, seed) cells.

    ``delta`` is the array of per-cell (conditioned - baseline) AUC differences.
    Resamples cells with replacement; reports the mean, a 95% percentile CI and
    a one-sided p for H1: mean delta > 0.
    """
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
        "ci95": [float(lo), float(hi)],
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
    p.add_argument("--out-metric", default="metrics/by_metric/pa15_metalcond_lomo.json")
    args = p.parse_args(argv)

    reports = {arm: _load(rel) for arm, rel in ARMS.items()}
    cross = {arm: _cross_cells(r) for arm, r in reports.items()}
    within = {arm: _within_means(r) for arm, r in reports.items()}

    base = cross["none"]
    arms_meta = {
        arm: {
            "metric_name": reports[arm].get("metric"),
            "metal_mode": reports[arm].get("metal_mode"),
            "metal_embed_dim": reports[arm].get("metal_embed_dim"),
            "report": ARMS[arm],
        }
        for arm in ARMS
    }

    # Per-arm cross means + per-metal table.
    arm_cross_summary: Dict[str, object] = {}
    for arm in ARMS:
        per_metal = {
            m: float(np.mean([cross[arm][m][str(s)] for s in SEEDS])) for m in METALS
        }
        arm_cross_summary[arm] = {
            "mean_across_metals": float(np.mean(list(per_metal.values()))),
            "per_metal_mean": per_metal,
            "per_metal_std": {
                m: float(np.std([cross[arm][m][str(s)] for s in SEEDS])) for m in METALS
            },
        }

    # Paired deltas (conditioned - none) over the 25 (metal, seed) cells.
    deltas: Dict[str, Dict[str, object]] = {}
    per_metal_delta: Dict[str, Dict[str, float]] = {}
    pvals: Dict[str, float] = {}
    for arm in ("token", "desc"):
        d = np.asarray(
            [cross[arm][m][str(s)] - base[m][str(s)] for m in METALS for s in SEEDS],
            dtype=np.float64,
        )
        bt = _paired_bootstrap(d, args.n_bootstrap, args.boot_seed)
        bt["ci95_delta"] = bt.pop("ci95")
        deltas[arm] = bt
        pvals[arm] = float(bt["p_one_sided_greater"])
        per_metal_delta[arm] = {
            m: float(
                np.mean([cross[arm][m][str(s)] - base[m][str(s)] for s in SEEDS])
            )
            for m in METALS
        }

    holm_adj = _holm(pvals)
    holm = {
        "correction": "Holm-Bonferroni (family = 2 conditioned arms vs the same baseline)",
        "n_family": len(pvals),
        "p_one_sided_raw": pvals,
        "p_one_sided_holm": holm_adj,
        "significant_after_holm_0.05": [a for a, p in holm_adj.items() if p < 0.05],
    }

    # Within-metal column (a control: conditioning should not change the
    # in-distribution setting, where the metal id is constant).
    within_summary: Dict[str, object] = {}
    for arm in ARMS:
        per_metal = {m: within[arm].get(m) for m in METALS}
        within_summary[arm] = {
            "mean_across_metals": float(
                np.mean([v for v in per_metal.values() if v is not None])
            ),
            "per_metal_mean": per_metal,
        }

    # Baseline reproduction vs the canonical PA3 metric on disk.
    repro: Dict[str, object] = {"checked": False}
    pa3_path = PROJECT_ROOT / PA3_CANONICAL
    if pa3_path.exists():
        pa3 = json.loads(pa3_path.read_text())
        pa3_cross = {
            f["metal"]: float(f["mean_auc"]) for f in pa3.get("per_fold", [])
        }
        diffs = {
            m: float(arm_cross_summary["none"]["per_metal_mean"][m] - pa3_cross[m])
            for m in METALS
            if m in pa3_cross
        }
        repro = {
            "checked": True,
            "canonical": PA3_CANONICAL,
            "none_arm_per_metal_mean": arm_cross_summary["none"]["per_metal_mean"],
            "pa3_canonical_per_metal_mean": pa3_cross,
            "abs_diff": {m: abs(v) for m, v in diffs.items()},
            "max_abs_diff": max(abs(v) for v in diffs.values()) if diffs else None,
            "note": (
                "GPU bf16 training is not bit-reproducible; a per-metal drift of "
                "<=0.02 is expected (see synthesize_pa3_metric.py caveats)."
            ),
        }

    # Verdict — derived, not asserted.
    best_arm = max(("token", "desc"), key=lambda a: deltas[a]["mean_delta"])
    best = deltas[best_arm]
    if best["ci95_delta"][0] > 0.0 and holm_adj[best_arm] < 0.05:
        verdict = (
            f"CONDITIONING HELPS: the {best_arm} arm beats the metal-agnostic "
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

    out = {
        "metric": "pa15_metalcond_lomo",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/pa15_metalcond_analysis.py",
        "description": (
            "Metal-conditioning ablation on the Paper-A probe cohort under the "
            "PA3 ligand-disjoint leave-one-metal-out protocol.  Three arms "
            "(none / token / desc) share cohort, split and seeds; the paired "
            "bootstrap over the 5 metals x 5 seeds cells tests whether an "
            "explicit metal input recovers cross-metal transfer (BLK-PA15)."
        ),
        "protocol": {
            "cohort": "MetalCytotoxDataset + CytotoxFilter(24h, ic50>=0.01uM) "
            "+ collapse_duplicates, per metal",
            "split": "leave-one-metal-out, ligand-disjoint (held-out metal's "
            "ligands dropped from the train pool)",
            "seeds": list(SEEDS),
            "metals": list(METALS),
            "epochs": 50,
            "backbone": "chemprop v2 AtomMessagePassing d_h=300 depth=3",
            "head": "single pooled BinaryClassificationFFN (1x300)",
            "significance": "paired bootstrap over 25 (metal, seed) cells, "
            f"{args.n_bootstrap} resamples, seed {args.boot_seed}; "
            "Holm-Bonferroni over the 2 conditioned arms",
        },
        "arms": arms_meta,
        "cross_metal": arm_cross_summary,
        "paired_delta_vs_none": deltas,
        "paired_delta_per_metal": per_metal_delta,
        "holm": holm,
        "within_metal": within_summary,
        "baseline_reproduction_check": repro,
        "contributes_method": bool(contributes_method),
        "verdict": verdict,
        "caveats": [
            "The token arm's held-out metal embedding is never trained in LOMO, "
            "so it is a random vector at test time — the arm bounds what an "
            "UNLEARNED token can do, not what a learned one could.",
            "The desc arm's descriptors are public periodic-table constants read "
            "from molmetal/data/transition_metals_30.csv; no label information "
            "crosses the split.  They encode periodic position (atomic number, "
            "group, period), formal oxidation state, a d-count proxy and a "
            "coordination-geometry one-hot — not ligand-field or redox potentials.",
            "Rh (n=51), Os (n=36) and Re (n=74) test folds are small; per-metal "
            "deltas on those cells carry large variance.",
            "GPU bf16 training is not bit-reproducible (~0.01-0.03 per-seed AUC "
            "drift), which is why the baseline arm is re-run in-session and the "
            "test is paired rather than against the PA3 canonical directly.",
        ],
        "honest_framing_note": (
            "Every number is MEASURED from the three arm reports on disk; the "
            "verdict and contributes_method flag are derived from the bootstrap "
            "CI and the Holm-adjusted p, not asserted."
        ),
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    dest = Path(args.out_metric)
    atomic_write_file(dest, json.dumps(out, indent=2, default=float, ensure_ascii=False))

    print(f"{'arm':>6}  cross-mean   delta-vs-none   p1(H1:>0)  holm")
    for arm in ("none", "token", "desc"):
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
    if repro.get("checked"):
        print(f"baseline repro max |diff| vs PA3 canonical: {repro['max_abs_diff']:.4f}")
    print(f"verdict: {verdict}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
