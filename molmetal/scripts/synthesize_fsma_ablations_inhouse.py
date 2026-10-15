#!/usr/bin/env python3
"""Assemble the in-house FSMA ablation families (C1/C1b/C2/C3/C4) at the
chemprop-matched source budget.

Paper A's ablation tables (``05_ablation.tex``) and the ``families`` block of
``metrics/by_metric/fsma_paper_stats.json`` are backed by chemprop v2 runs whose
source training is ``config.epochs = 50``.  The project red line requires the
method result to be backed by the self-developed D-MPNN, so this synthesis
rebuilds the SAME five families from in-house ``ex1`` artifacts re-run at
``epochs = 50`` and the chemprop K/seed settings:

  * ``C1``  head-only K-curve  — ``fsma_inhouse_kcurve_ep50.json``
  * ``C1b`` full-scope K-curve — ``fsma_inhouse_c4_full_ep50.json``
  * ``C2``  budget-matched controls — the K-curve (kshot) plus
            ``fsma_inhouse_c2_from_scratch_ep50.json`` and
            ``fsma_inhouse_c2_concat_retrain_ep50.json``
  * ``C3``  support sampler at K=20 — ``fsma_inhouse_c3_{random,scaffold_diverse,
            activity_scaffold}_ep50.json`` plus the K=20 cell of the K-curve
            restricted to the five chemprop seeds (label-stratified)
  * ``C4``  adaptation scope — head (K-curve) vs full (C1b) vs partial
            (``fsma_inhouse_c4_partial_ep50.json``)

Statistics mirror ``fsma_synthesize.py``: per K a paired one-sided t-test for
``mean(delta) > 0`` on the 50 (metal, seed) cells (25 for C3, which uses the five
chemprop seeds), a t-based 95 % CI, and Holm-Bonferroni within each family.

The C2 controls emit ``zero_shot_auc = NaN`` (a from-scratch / concat-retrain
model has no zero-shot).  Their paired delta is therefore the control's
``adapted_auc`` minus the SOURCE model's zero-shot AUC on the identical
``(metal, seed, K)`` cell, read from the K-curve artifact.  That join is exact:
the two runs share the source cache and ``zero_shot_auc`` is bit-identical across
them (asserted at runtime below).

Usage
-----
::

    .venv/bin/python molmetal/scripts/synthesize_fsma_ablations_inhouse.py \
        --out-metric metrics/by_metric/fsma_ablations_inhouse_ep50.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KCURVE = "metrics/by_metric/fsma_inhouse_kcurve_ep50.json"
C4_FULL = "metrics/by_metric/fsma_inhouse_c4_full_ep50.json"
C4_PARTIAL = "metrics/by_metric/fsma_inhouse_c4_partial_ep50.json"
C2_FROM_SCRATCH = "metrics/by_metric/fsma_inhouse_c2_from_scratch_ep50.json"
C2_CONCAT = "metrics/by_metric/fsma_inhouse_c2_concat_retrain_ep50.json"
C3_RANDOM = "metrics/by_metric/fsma_inhouse_c3_random_ep50.json"
C3_SCAFFOLD = "metrics/by_metric/fsma_inhouse_c3_scaffold_diverse_ep50.json"
C3_ACTIVITY = "metrics/by_metric/fsma_inhouse_c3_activity_scaffold_ep50.json"
CHEMPROP_STATS = "metrics/by_metric/fsma_paper_stats.json"

METALS = ("Ru", "Ir", "Rh", "Os", "Re")
HEAD_KS = (5, 10, 20, 40, 80)
SCOPE_KS = (5, 20, 40, 80)
#: the five seeds the chemprop C3 sampler runs used (c3_*_k20 reports)
C3_SEEDS = (42, 43, 44, 49, 123)


def _load(rel: str) -> Dict[str, object]:
    return json.loads((PROJECT_ROOT / rel).read_text())


def _index(metric: Dict[str, object]) -> Dict[Tuple[str, int, int], Dict[str, object]]:
    """``{(metal, k, seed): run_row}``."""
    out: Dict[Tuple[str, int, int], Dict[str, object]] = {}
    for r in metric["per_run"]:  # type: ignore[index]
        out[(str(r["metal"]), int(r["k"]), int(r["seed"]))] = r
    return out


def _finite(x: object) -> bool:
    return x is not None and np.isfinite(float(x))


def _paired_deltas(
    metric: Dict[str, object],
    k: int,
    *,
    seeds: Optional[Tuple[int, ...]] = None,
) -> List[float]:
    """``adapted - zero_shot`` per (metal, seed) cell at ``k``."""
    out: List[float] = []
    for r in metric["per_run"]:  # type: ignore[index]
        if int(r["k"]) != k:
            continue
        if seeds is not None and int(r["seed"]) not in seeds:
            continue
        z, a = r.get("zero_shot_auc"), r.get("adapted_auc")
        if not (_finite(z) and _finite(a)):
            continue
        out.append(float(a) - float(z))
    return out


def _control_deltas(
    control: Dict[str, object],
    zero_shot: Dict[Tuple[str, int, int], Dict[str, object]],
    k: int,
    *,
    strict: bool = True,
) -> Tuple[List[float], int]:
    """``control.adapted_auc - source.zero_shot_auc`` joined on (metal, seed, k).

    Returns ``(deltas, n_missing)`` where ``n_missing`` counts control cells whose
    ``(metal, seed, k)`` had no finite zero-shot in the K-curve (MISSING !=
    dropped: the caller records the count).
    """
    out: List[float] = []
    missing = 0
    for r in control["per_run"]:  # type: ignore[index]
        if int(r["k"]) != k:
            continue
        a = r.get("adapted_auc")
        key = (str(r["metal"]), k, int(r["seed"]))
        src = zero_shot.get(key)
        if not _finite(a) or src is None or not _finite(src.get("zero_shot_auc")):
            missing += 1
            continue
        out.append(float(a) - float(src["zero_shot_auc"]))
    if strict and missing:
        # a non-zero join miss means the two runs disagree on the cell grid
        raise SystemExit(f"C2 join missed {missing} cell(s) at K={k}")
    return out, missing


def _stats(deltas: List[float]) -> Dict[str, object]:
    from scipy import stats as st

    x = np.asarray(deltas, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return {"n_cells": 0, "status": "SKIPPED", "reason": "no cells"}
    m = float(x.mean())
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    se = s / np.sqrt(n)
    if se > 0:
        t = m / se
        # One-sided p in the DIRECTION of the observed effect, matching
        # fsma_synthesize.py: t.sf(t) for a positive mean, t.cdf(t) for a
        # negative one (the C2 controls are strongly negative, and the paper
        # reports that as "significant" too).
        p = float(st.t.sf(t, df=n - 1)) if t > 0 else 1.0 - float(st.t.sf(t, df=n - 1))
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
        "p_one_sided_directional": p,
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


def _family(members: Dict[str, List[float]]) -> Dict[str, object]:
    """Stats per member + Holm across the family."""
    per = {name: _stats(d) for name, d in members.items()}
    pv = {
        name: float(per[name]["p_one_sided_directional"])
        for name in per
        if per[name].get("status") == "MEASURED"
    }
    adj = _holm(pv) if pv else {}
    return {
        "members": per,
        "p_one_sided_raw": pv,
        "p_one_sided_holm": adj,
        "correction": f"Holm-Bonferroni (family = {len(pv)} arms)",
        "note": "p is one-sided in the direction of each arm's observed effect "
        "(so a strongly negative control arm is also 'significant')",
        "significant_after_holm_0.05": sorted(
            (k for k, q in adj.items() if q < 0.05)
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kcurve", default=KCURVE)
    p.add_argument("--c4-full", default=C4_FULL)
    p.add_argument("--c4-partial", default=C4_PARTIAL)
    p.add_argument("--c2-from-scratch", default=C2_FROM_SCRATCH)
    p.add_argument("--c2-concat", default=C2_CONCAT)
    p.add_argument("--c3-random", default=C3_RANDOM)
    p.add_argument("--c3-scaffold", default=C3_SCAFFOLD)
    p.add_argument("--c3-activity", default=C3_ACTIVITY)
    p.add_argument("--chemprop-stats", default=CHEMPROP_STATS)
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/fsma_ablations_inhouse_ep50.json",
    )
    args = p.parse_args(argv)

    rels = {
        "kcurve": args.kcurve,
        "c4_full": args.c4_full,
        "c4_partial": args.c4_partial,
        "c2_from_scratch": args.c2_from_scratch,
        "c2_concat": args.c2_concat,
        "c3_random": args.c3_random,
        "c3_scaffold": args.c3_scaffold,
        "c3_activity": args.c3_activity,
        "chemprop_stats": args.chemprop_stats,
    }
    missing = [rel for rel in rels.values() if not (PROJECT_ROOT / rel).exists()]
    if missing:
        raise SystemExit(
            f"missing input artifact(s) {missing}; run the configs/ex1_fsma_ep50_*.yaml "
            f"runs first (MISSING != dropped)"
        )

    m = {k: _load(v) for k, v in rels.items()}
    for name, mm in m.items():
        if name == "chemprop_stats":
            continue
        if int(mm.get("epochs", -1)) != 50:
            raise SystemExit(
                f"{name} is not the epochs=50 (chemprop-matched) artifact "
                f"(epochs={mm.get('epochs')}); refusing to mix budgets"
            )

    zs = _index(m["kcurve"])

    # The C2 join below assumes the source model's zero_shot_auc is identical
    # across runs (they share the source cache).  Assert it rather than trust it.
    for (metal, k, seed), row in _index(m["c4_full"]).items():
        ref = zs.get((metal, k, seed))
        if ref is None:
            continue
        if ref.get("zero_shot_auc") != row.get("zero_shot_auc"):
            raise SystemExit(
                f"zero_shot_auc disagrees between the K-curve and C4-full runs at "
                f"({metal}, K={k}, seed={seed}): {ref.get('zero_shot_auc')} vs "
                f"{row.get('zero_shot_auc')} — the C2 join would not be exact"
            )

    # ---- C1 / C1b: the two K-curves -----------------------------------------
    c1 = _family({f"head_k{k}": _paired_deltas(m["kcurve"], k) for k in HEAD_KS})
    c1b = _family({f"full_k{k}": _paired_deltas(m["c4_full"], k) for k in SCOPE_KS})

    # ---- C2: budget-matched controls ----------------------------------------
    c2: Dict[str, object] = {}
    for k in (20, 40):
        c2[f"k{k}"] = _family({
            "kshot": _paired_deltas(m["kcurve"], k),
            "from_scratch": _control_deltas(m["c2_from_scratch"], zs, k)[0],
            "concat_retrain": _control_deltas(m["c2_concat"], zs, k)[0],
        })

    # ---- C3: support sampler at K=20 ----------------------------------------
    c3 = _family({
        "random": _paired_deltas(m["c3_random"], 20),
        "stratified": _paired_deltas(m["kcurve"], 20, seeds=C3_SEEDS),
        "scaffold_diverse": _paired_deltas(m["c3_scaffold"], 20),
        "activity_scaffold": _paired_deltas(m["c3_activity"], 20),
    })

    # ---- C4: adaptation scope ------------------------------------------------
    c4_members: Dict[str, List[float]] = {}
    for k in SCOPE_KS:
        c4_members[f"head_k{k}"] = _paired_deltas(m["kcurve"], k)
        c4_members[f"full_k{k}"] = _paired_deltas(m["c4_full"], k)
        c4_members[f"partial_k{k}"] = _paired_deltas(m["c4_partial"], k)
    c4 = _family(c4_members)

    out = {
        "metric": "fsma_ablations_inhouse_ep50",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "paired ROC-AUC delta (adapted - zero-shot), higher better",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/synthesize_fsma_ablations_inhouse.py",
        "description": (
            "In-house D-MPNN FSMA ablation families (C1/C1b/C2/C3/C4) at the "
            "chemprop-matched source budget (epochs=50), replacing the chemprop v2 "
            "backing of paper_a's ablation tables."
        ),
        "model": "MetalAwareDMPNN(metal_injection='none'), hidden=128 depth=3 "
        "dropout=0.1, epochs=50",
        "families": {
            "C1_kshot_curve_headonly": c1,
            "C1b_kshot_curve_fullscope": c1b,
            "C2_controls_matched": c2,
            "C3_support_sampler": c3,
            "C4_adapt_scope": c4,
        },
        "protocol": {
            "cohort": "paper funnel (build_paper_cohort, no time window)",
            "split": "leave-one-metal-out, ligand-disjoint",
            "adaptation": "stratified K support; adapt_steps = 20*ceil(K/16)",
            "c2_control_join": (
                "control.adapted_auc - source.zero_shot_auc on the identical "
                "(metal, seed, K); zero_shot is bit-identical across the runs "
                "(shared source cache), asserted at runtime"
            ),
            "c3_seeds": list(C3_SEEDS),
            "significance": "one-sided t-test (H1: mean paired delta > 0) per arm; "
            "t-based 95% CI; Holm-Bonferroni within each family",
        },
        "source_artifacts": rels,
        "caveats": [
            "Adaptation is NOT bit-reproducible on CUDA (~1e-3 on the 50-cell mean); "
            "only the aggregate is reported.",
            "Capacity is NOT matched to the chemprop reference (hidden=128 vs "
            "d_h=300); the comparison is a scorer/representation result.",
            "chemprop v2 is a REFERENCE scorer here, not the paper's model "
            "(project red line: the method result is the in-house D-MPNN).",
        ],
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    dest = Path(args.out_metric)
    atomic_write_file(dest, json.dumps(out, indent=2, default=float, ensure_ascii=False))

    print(f"{'family':>28} {'arm':>18} {'mean':>9} {'pos':>7} {'holm':>9}")
    for fam_name, fam in out["families"].items():
        if fam_name == "C2_controls_matched":
            for kk, sub in fam.items():
                for arm, s in sub["members"].items():
                    q = sub["p_one_sided_holm"].get(arm, float("nan"))
                    print(f"{fam_name+'@'+kk:>28} {arm:>18} {s['mean']:>+9.4f} "
                          f"{str(s['n_positive'])+'/'+str(s['n_cells']):>7} {q:>9.2e}")
            continue
        for arm, s in fam["members"].items():
            q = fam["p_one_sided_holm"].get(arm, float("nan"))
            print(f"{fam_name:>28} {arm:>18} {s['mean']:>+9.4f} "
                  f"{str(s['n_positive'])+'/'+str(s['n_cells']):>7} {q:>9.2e}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
