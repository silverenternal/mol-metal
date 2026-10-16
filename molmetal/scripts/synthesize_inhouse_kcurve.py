#!/usr/bin/env python3
"""Assemble the budget-matched in-house FSMA K-curve (C1 migration).

Paper A's ``tab:kshot`` / ``fig2_kshot_curve`` / ``recovery_to_ceiling`` currently
cite ``metrics/by_metric/fsma_paper_stats.json``, whose source runs
(``molmetal/reports/fsma_c4_head_k*/metrics.json``) are **chemprop v2** with
``config.epochs = 50``.  The project red line requires the method result to be
backed by the self-developed D-MPNN.

The in-house curve exists (``metrics/by_metric/fsma_inhouse_kcurve_headonly.json``)
but was run at ``epochs = 20`` — i.e. the in-house SOURCE model was trained at
2.5x less than the reference it is compared against.  At that under-budget
setting the in-house paired delta is **negative** at ``K=5`` and ``K=10``, so the
paper's "the curve is positive at every K" claim cannot be carried over as-is.

Two independent ``epochs=50`` runs already on disk
(``fsma_l1_ep50.json``, ``fsma_ep50_lowk.json``) both give a positive low-K
delta at the reference budget, so the sign flip is a source-budget artefact, not
a property of the in-house model.  This synthesis reports the FULL curve at
``epochs=50`` (the matched budget, from ``fsma_inhouse_kcurve_ep50.json``) as the
primary column, carries the ``epochs=20`` curve as an explicit sensitivity, and
keeps the chemprop reference alongside.

Statistics mirror the chemprop synthesis: per ``K`` the 50 ``(metal, seed)``
paired deltas ``adapted - zero_shot``; a one-sided t-test for ``mean > 0``; a
t-based 95 % CI; Holm-Bonferroni across the five ``K`` in the family.

Usage
-----
::

    .venv/bin/python molmetal/scripts/synthesize_inhouse_kcurve.py \
        --out-metric metrics/by_metric/fsma_inhouse_kcurve_ep50_synth.json
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

EP50_METRIC = "metrics/by_metric/fsma_inhouse_kcurve_ep50.json"
EP20_METRIC = "metrics/by_metric/fsma_l1_ep20.json"
CHEMPROP_STATS = "metrics/by_metric/fsma_paper_stats.json"
#: capacity ablation (hidden 512, no adaptation levers) vs the h128 primary.
CAP512_METRIC = "metrics/by_metric/fsma_inhouse_kcurve_ep50_cap512.json"

METALS = ("Ru", "Ir", "Rh", "Os", "Re")
KS = (5, 10, 20, 40, 80)
#: chemprop paper-funnel runs backing the current tab:kshot (families.C1_kshot_curve_headonly)
CHEMPROP_RUNS = {"5": "head_k5", "10": "head_k10", "20": "head_k20",
                 "40": "head_k40", "80": "head_k80"}

BUDGET_NOTE = (
    "The chemprop reference this migrates trains its source model for "
    "config.epochs=50 (fsma_c4_head_k*.config.epochs).  The original in-house "
    "curve used epochs=20 and is NEGATIVE at K=5/K=10; at the matched epochs=50 "
    "budget the same model is positive at every K.  Both budgets are reported "
    "(inhouse_ep50 = primary, inhouse_ep20 = sensitivity) so the sign flip is "
    "visible rather than hidden."
)


def _load(rel: str) -> Dict[str, object]:
    return json.loads((PROJECT_ROOT / rel).read_text())


def _deltas(metric: Dict[str, object]) -> Dict[int, List[float]]:
    """``{K: [adapted - zero_shot per (metal, seed) cell]}`` from an ex1 report."""
    out: Dict[int, List[float]] = {}
    for r in metric["per_run"]:  # type: ignore[index]
        z, a = r.get("zero_shot_auc"), r.get("adapted_auc")
        if z is None or a is None:
            continue
        if not (np.isfinite(z) and np.isfinite(a)):
            continue
        out.setdefault(int(r["k"]), []).append(float(a) - float(z))
    return out


def _per_metal_deltas(metric: Dict[str, object]) -> Dict[int, Dict[str, List[float]]]:
    out: Dict[int, Dict[str, List[float]]] = {}
    for r in metric["per_run"]:  # type: ignore[index]
        z, a = r.get("zero_shot_auc"), r.get("adapted_auc")
        if z is None or a is None or not (np.isfinite(z) and np.isfinite(a)):
            continue
        out.setdefault(int(r["k"]), {}).setdefault(str(r["metal"]), []).append(
            float(a) - float(z)
        )
    return out


def _abs_means(metric: Dict[str, object]) -> Dict[int, float]:
    """``{K: mean adapted absolute AUC}``.

    The capacity ablation is judged on the ABSOLUTE AUC, not the paired delta:
    a stronger source raises ``zero_shot_auc`` more than ``adapted_auc``, so the
    delta shrinks even when the treatment is better (see
    metrics/by_metric/... docstring / the paired-delta baseline illusion).
    """
    acc: Dict[int, List[float]] = {}
    for r in metric["per_run"]:  # type: ignore[index]
        a = r.get("adapted_auc")
        if a is None or not np.isfinite(a):
            continue
        acc.setdefault(int(r["k"]), []).append(float(a))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _stats(deltas: List[float]) -> Dict[str, object]:
    from scipy import stats as st

    x = np.asarray(deltas, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return {"n_cells": 0, "status": "SKIPPED", "reason": "no cells"}
    m = float(x.mean())
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    se = s / np.sqrt(n) if n > 0 else float("nan")
    if se > 0:
        t = m / se
        p = float(st.t.sf(t, df=n - 1))          # H1: mean delta > 0
        crit = float(st.t.ppf(0.975, df=n - 1))
        ci = [m - crit * se, m + crit * se]
    else:
        t, p, ci = float("inf"), 0.0, [m, m]
    return {
        "status": "MEASURED",
        "n_cells": n,
        "mean": m,
        "std": s,
        "t": t,
        "p_one_sided_gt0": p,
        "ci95_ttest": [float(ci[0]), float(ci[1])],
        "n_positive": int((x > 0).sum()),
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
    p.add_argument("--ep50-metric", default=EP50_METRIC)
    p.add_argument("--ep20-metric", default=EP20_METRIC)
    p.add_argument("--cap512-metric", default=CAP512_METRIC)
    p.add_argument("--chemprop-stats", default=CHEMPROP_STATS)
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/fsma_inhouse_kcurve_ep50_synth.json",
    )
    args = p.parse_args(argv)

    missing = [
        rel for rel in (args.ep50_metric, args.ep20_metric, args.cap512_metric,
                        args.chemprop_stats)
        if not (PROJECT_ROOT / rel).exists()
    ]
    if missing:
        raise SystemExit(
            f"missing input artifact(s) {missing}; run "
            f"configs/ex1_fsma_ep50_kcurve.yaml first (MISSING != dropped)"
        )

    ep50 = _load(args.ep50_metric)
    ep20 = _load(args.ep20_metric)
    cap512 = _load(args.cap512_metric)
    chem = _load(args.chemprop_stats)

    d50 = _deltas(ep50)
    d20 = _deltas(ep20)
    dcap = _deltas(cap512)
    pm50 = _per_metal_deltas(ep50)
    abs50, abscap = _abs_means(ep50), _abs_means(cap512)

    per_k: Dict[str, Dict[str, object]] = {}
    for k in KS:
        entry: Dict[str, object] = {}
        entry["inhouse_ep50"] = _stats(d50.get(k, []))
        entry["inhouse_ep20"] = _stats(d20.get(k, []))
        entry["inhouse_ep50_cap512"] = _stats(dcap.get(k, []))
        entry["adapted_abs_auc"] = {
            "inhouse_ep50_h128": abs50.get(k),
            "inhouse_ep50_cap512": abscap.get(k),
        }
        run = CHEMPROP_RUNS.get(str(k))
        if run is not None:
            try:
                pcell = chem["per_run"][run]["arms"]["kshot"]["_pooled_ru_dominated"]["per_cell"]
                entry["chemprop"] = _stats([float(v) for v in pcell])
            except (KeyError, TypeError):
                entry["chemprop"] = {"status": "SKIPPED", "reason": f"no {run} in chemprop stats"}
        if k in pm50:
            entry["inhouse_ep50_per_metal"] = {
                m: _stats(pm50[k][m]) for m in METALS if m in pm50[k]
            }
        per_k[str(k)] = entry

    def _holm_for(arm: str) -> Dict[str, object]:
        pv = {
            str(k): float(per_k[str(k)][arm]["p_one_sided_gt0"])
            for k in KS
            if per_k[str(k)].get(arm, {}).get("status") == "MEASURED"
        }
        adj = _holm(pv) if pv else {}
        return {
            "correction": "Holm-Bonferroni (family = 5 K values)",
            "n_family": len(pv),
            "p_one_sided_raw": pv,
            "p_one_sided_holm": adj,
            "significant_after_holm_0.05": sorted(
                (k for k, q in adj.items() if q < 0.05), key=lambda s: int(s)
            ),
        }

    holm50, holm20, holmcp = _holm_for("inhouse_ep50"), _holm_for("inhouse_ep20"), _holm_for("chemprop")
    holmcap = _holm_for("inhouse_ep50_cap512")

    ep50_means = {
        int(k): float(per_k[str(k)]["inhouse_ep50"]["mean"])
        for k in KS if per_k[str(k)]["inhouse_ep50"].get("status") == "MEASURED"
    }
    positive_every_k = bool(ep50_means) and all(v > 0 for v in ep50_means.values())
    sig = holm50["significant_after_holm_0.05"]
    if positive_every_k and sig:
        verdict = (
            f"REPRODUCED at the matched budget: the in-house head-only K-curve is "
            f"positive at every K (K=5 {ep50_means.get(5, float('nan')):+.4f} .. "
            f"K=80 {ep50_means.get(80, float('nan')):+.4f}) and family-significant "
            f"at K={','.join(sig)}; the same model at epochs=20 was negative at "
            f"K=5/K=10, so the sign is set by the source budget, not the model."
        )
    elif sig:
        verdict = (
            f"PARTIAL: the in-house epochs=50 curve is family-significant at "
            f"K={','.join(sig)} but not positive at every K."
        )
    else:
        verdict = (
            "NOT REPRODUCED at the matched budget: no K survives the family "
            "correction on the in-house model."
        )

    out = {
        "metric": "fsma_inhouse_kcurve_ep50",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "paired ROC-AUC delta (adapted - zero-shot), higher better",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/synthesize_inhouse_kcurve.py",
        "description": (
            "FSMA head-only K-shot recovery curve re-measured on the in-house "
            "D-MPNN at the chemprop-matched source budget (epochs=50), with the "
            "epochs=20 curve and the chemprop v2 reference carried alongside."
        ),
        "model": "MetalAwareDMPNN(metal_injection='none'), hidden=128 depth=3 "
        "dropout=0.1, epochs=50; encoder frozen, head-only adaptation "
        "(capacity ablation: hidden=512, no adaptation levers)",
        "protocol": {
            "cohort": "paper funnel (build_paper_cohort, no time window)",
            "split": "leave-one-metal-out, ligand-disjoint",
            "adaptation": "stratified K support; head-only; adapt_steps = 20*ceil(K/16)",
            "seeds": list(ep50.get("seeds", [])),
            "metals": list(METALS),
            "k": list(KS),
            "epochs": 50,
            "significance": "one-sided t-test (H1: mean paired delta > 0) on the "
            "50 (metal, seed) cells per K; t-based 95% CI; Holm-Bonferroni across "
            "the 5 K values",
        },
        "per_k": per_k,
        "holm": {
            "inhouse_ep50": holm50,
            "inhouse_ep20": holm20,
            "inhouse_ep50_cap512": holmcap,
            "chemprop_reference": holmcp,
        },
        "positive_at_every_k": {
            "inhouse_ep50": positive_every_k,
            "inhouse_ep20": all(
                float(per_k[str(k)]["inhouse_ep20"]["mean"]) > 0
                for k in KS if per_k[str(k)]["inhouse_ep20"].get("status") == "MEASURED"
            ),
            "inhouse_ep50_cap512": all(
                float(per_k[str(k)]["inhouse_ep50_cap512"]["mean"]) > 0
                for k in KS if per_k[str(k)]["inhouse_ep50_cap512"].get("status") == "MEASURED"
            ),
        },
        "verdict": verdict,
        "budget_note": BUDGET_NOTE,
        "caveats": [
            "The adaptation step is NOT bit-reproducible on CUDA: two epochs=50 "
            "runs of the same config agree to ~1e-3 on the 50-cell mean but differ "
            "by up to 0.079 on an individual cell.  Only the aggregate is reported; "
            "per-cell values are noisy.",
            "Capacity is NOT matched to the chemprop reference (hidden=128 vs "
            "d_h=300) in the PRIMARY column; the hidden=512 capacity ablation is "
            "carried alongside.  For the ablation the criterion is the ADAPTED "
            "ABSOLUTE AUC (adapted_abs_auc per K), NOT the paired delta: a "
            "stronger source raises zero_shot_auc more than adapted_auc, so the "
            "delta shrinks at low K (K5 +0.0129 -> -0.0033) even though the "
            "absolute AUC rises.  The 'delta positive at every K' claim is made "
            "for the h128 primary only.",
            "The epochs=20 column is carried as a sensitivity, not discarded: it is "
            "the budget at which the in-house low-K delta is negative.",
            "chemprop v2 is a REFERENCE scorer here, not the paper's model "
            "(project red line: the method result is the in-house D-MPNN).",
        ],
        "source_artifacts": {
            "inhouse_ep50": args.ep50_metric,
            "inhouse_ep20": args.ep20_metric,
            "inhouse_ep50_cap512": args.cap512_metric,
            "chemprop_reference": args.chemprop_stats,
        },
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    dest = Path(args.out_metric)
    atomic_write_file(dest, json.dumps(out, indent=2, default=float, ensure_ascii=False))

    print(f"{'K':>4} {'ep50(in-house)':>15} {'ep20(in-house)':>15} {'chemprop':>10}  "
          f"{'cap512 delta':>13} {'cap512 abs':>11} {'h128 abs':>9}  {'holm50':>9}")
    for k in KS:
        e = per_k[str(k)]
        s50, s20, scp = e["inhouse_ep50"], e["inhouse_ep20"], e.get("chemprop", {})
        sc = e["inhouse_ep50_cap512"]
        f = lambda s: (f"{s['mean']:+.4f}" if s.get("status") == "MEASURED" else "  n/a  ")
        q = holm50["p_one_sided_holm"].get(str(k), float("nan"))
        ab = e["adapted_abs_auc"]
        print(f"{k:>4} {f(s50):>15} {f(s20):>15} {f(scp):>10}  "
              f"{f(sc):>13} {ab['inhouse_ep50_cap512']:>11.4f} "
              f"{ab['inhouse_ep50_h128']:>9.4f}  {q:9.2e}")
    print(f"ep50 significant K: {sig}")
    print(f"cap512 significant K: {holmcap['significant_after_holm_0.05']}")
    print(f"verdict: {verdict}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
