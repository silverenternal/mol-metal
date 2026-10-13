#!/usr/bin/env python
"""L3 — assemble the metal-conditioning "pull-up" metric from the arm reports.

Two arms, both under the identical ligand-disjoint leave-one-metal-out split
and the same 5 seeds as PA15's ``none`` baseline:

  * ``film``  — the metal descriptor drives a per-channel FiLM (scale+shift) on
    the pooled graph readout, instead of being concatenated (PA15's ``desc``).
    Compared to ``none`` over the 25 (metal, seed) cells.
  * ``kshot`` — the frozen zero-shot ``none`` model is fine-tuned on K=10
    labelled support ligands from the held-out metal and re-scored on the
    disjoint query set.  Compared zero-shot-query vs k-shot-query on the
    IDENTICAL query rows.

Significance: paired bootstrap (10,000 resamples, seed 42) with a one-sided p
for "the arm helps", then Holm-Bonferroni across the two arms.

MEASURED only.  Every number comes from the arm reports on disk.
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

FILM_REPORT = "molmetal/reports/l3_metalcond_film/metrics.json"
KSHOT_REPORT = "molmetal/reports/l3_metalcond_kshot/metrics.json"
NONE_REPORT = "molmetal/reports/pa15_metalcond_lomo_none/metrics.json"
METALS = ("Ru", "Ir", "Rh", "Os", "Re")
SEEDS = (42, 43, 44, 49, 123)


def _load(rel: str) -> Dict[str, object]:
    return json.loads((PROJECT_ROOT / rel).read_text())


def _cross_cells(report: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for fold in report["per_fold"]:  # type: ignore[index]
        out[str(fold["metal"])] = {
            str(s): float(a) for s, a in fold["per_seed_auc"].items()
        }
    return out


def _kshot_cells(report: Dict[str, object]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """``{metal: {seed: {zs, ks}}}`` from the k-shot per-seed blocks."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for fold in report["per_fold"]:  # type: ignore[index]
        m = str(fold["metal"])
        out[m] = {}
        for s, rec in fold["per_seed"].items():
            if "kshot" in rec:
                out[m][str(s)] = {
                    "zs": float(rec["kshot"]["zero_shot_query_auc"]),
                    "ks": float(rec["kshot"]["kshot_query_auc"]),
                }
    return out


def _paired_bootstrap(delta: np.ndarray, n_boot: int, seed: int) -> Dict[str, object]:
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
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--boot-seed", type=int, default=42)
    p.add_argument("--out-metric", default="metrics/by_metric/l3_metalcond_lomo.json")
    args = p.parse_args(argv)

    none = _load(NONE_REPORT)
    film = _load(FILM_REPORT)
    kshot = _load(KSHOT_REPORT)

    base = _cross_cells(none)
    film_c = _cross_cells(film)
    ks_c = _kshot_cells(kshot)

    def _mean_per_metal(cells, key=None):
        out = {}
        for m in METALS:
            vals = [
                cells[m][str(s)] if key is None else cells[m][str(s)][key]
                for s in SEEDS
                if str(s) in cells.get(m, {})
            ]
            out[m] = float(np.mean(vals)) if vals else float("nan")
        return out

    # --- film vs none (paired over 25 cells) ---
    film_delta = np.asarray(
        [film_c[m][str(s)] - base[m][str(s)] for m in METALS for s in SEEDS],
        dtype=np.float64,
    )
    film_bt = _paired_bootstrap(film_delta, args.n_bootstrap, args.boot_seed)

    # --- k-shot vs zero-shot on the SAME query rows (paired over 25 cells) ---
    ks_delta = np.asarray(
        [ks_c[m][str(s)]["ks"] - ks_c[m][str(s)]["zs"] for m in METALS for s in SEEDS],
        dtype=np.float64,
    )
    ks_bt = _paired_bootstrap(ks_delta, args.n_bootstrap, args.boot_seed)

    pvals = {
        "film_vs_none": float(film_bt["p_one_sided_greater"]),
        "kshot_vs_zeroshot": float(ks_bt["p_one_sided_greater"]),
    }
    holm_adj = _holm(pvals)

    report: Dict[str, object] = {
        "metric": "l3_metalcond_lomo",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/l3_metalcond_analysis.py",
        "description": (
            "L3 pull-up: does a stronger readout-level conditioning (FiLM) or a "
            "small labelled support set from the held-out metal (K-shot "
            "adaptation) recover Paper A's cross-metal collapse?"
        ),
        "protocol": {
            "cohort": "MetalCytotoxDataset + CytotoxFilter(24h, ic50>=0.01uM) + collapse_duplicates, per metal",
            "split": "leave-one-metal-out, ligand-disjoint",
            "seeds": list(SEEDS),
            "metals": list(METALS),
            "backbone": "chemprop v2 AtomMessagePassing d_h=300 depth=3",
            "k_shot": 10,
            "k_shot_ft_epochs": 20,
            "significance": "paired bootstrap over 25 (metal, seed) cells, 10000 resamples, seed 42; Holm-Bonferroni over the 2 arms",
        },
        "arms": {
            "none_baseline": {
                "report": NONE_REPORT,
                "cross_mean_across_metals": float(
                    np.mean(list(_mean_per_metal(base).values()))
                ),
                "per_metal_mean": _mean_per_metal(base),
            },
            "film": {
                "report": FILM_REPORT,
                "cross_mean_across_metals": float(
                    np.mean(list(_mean_per_metal(film_c).values()))
                ),
                "per_metal_mean": _mean_per_metal(film_c),
                "paired_delta_vs_none": film_bt,
                "per_metal_delta": {
                    m: float(
                        np.mean(
                            [film_c[m][str(s)] - base[m][str(s)] for s in SEEDS]
                        )
                    )
                    for m in METALS
                },
            },
            "kshot": {
                "report": KSHOT_REPORT,
                "k": 10,
                "zero_shot_query_mean": float(
                    np.mean(list(_mean_per_metal(ks_c, "zs").values()))
                ),
                "kshot_query_mean": float(
                    np.mean(list(_mean_per_metal(ks_c, "ks").values()))
                ),
                "per_metal_zero_shot": _mean_per_metal(ks_c, "zs"),
                "per_metal_kshot": _mean_per_metal(ks_c, "ks"),
                "paired_delta_vs_zeroshot": ks_bt,
                "per_metal_delta": {
                    m: float(
                        np.mean(
                            [ks_c[m][str(s)]["ks"] - ks_c[m][str(s)]["zs"] for s in SEEDS]
                        )
                    )
                    for m in METALS
                },
            },
        },
        "holm": {
            "correction": "Holm-Bonferroni (family = 2 arms vs their paired baseline)",
            "n_family": len(pvals),
            "p_one_sided_raw": pvals,
            "p_one_sided_holm": holm_adj,
            "significant_after_holm_0.05": [a for a, v in holm_adj.items() if v < 0.05],
        },
    }

    film_ok = holm_adj["film_vs_none"] < 0.05 and film_bt["mean_delta"] > 0
    ks_ok = holm_adj["kshot_vs_zeroshot"] < 0.05 and ks_bt["mean_delta"] > 0
    report["verdict"] = {
        "film_recovers_collapse": bool(film_ok),
        "kshot_recovers_collapse": bool(ks_ok),
        "contributes_method": bool(film_ok or ks_ok),
        "summary": (
            f"film delta {film_bt['mean_delta']:+.4f} "
            f"(95% CI {film_bt['ci95'][0]:+.4f},{film_bt['ci95'][1]:+.4f}, "
            f"Holm p={holm_adj['film_vs_none']:.4f}); "
            f"k-shot delta {ks_bt['mean_delta']:+.4f} "
            f"(95% CI {ks_bt['ci95'][0]:+.4f},{ks_bt['ci95'][1]:+.4f}, "
            f"Holm p={holm_adj['kshot_vs_zeroshot']:.4f})"
        ),
    }
    report["honest_framing_note"] = (
        "MEASURED only.  The k-shot arm is an ADAPTATION result (it uses "
        "held-out-metal labels); the zero-shot LOMO column remains the paper's "
        "Table-1 cross number.  Every value traces to the arm reports on disk."
    )

    out = PROJECT_ROOT / args.out_metric
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=1))
    print(json.dumps(report["holm"], ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
