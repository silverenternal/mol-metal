#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""C3 follow-up — the support-sampler PAIRWISE contrast that was never run.

=====================================================================
Why this file exists
=====================================================================
``metrics/by_metric/fsma_ablations_inhouse_ep50.json`` reports the C3 family
(four support-sampling rules) as four *one-sample* tests against zero:

    random            +0.0359   Holm 0.0101   SIGNIFICANT
    scaffold_diverse  +0.0249   Holm 0.0430   SIGNIFICANT
    stratified        +0.0116   Holm 0.3609   n.s.
    activity_scaffold +0.0031   Holm 0.4000   n.s.

``paper_a/main.tex`` (abstract, line ~70) reads that table as "four samplers
(random, stratified, scaffold-diverse, activity-scaffold) are statistically
indistinguishable".  That is a category error: **four separate tests against
zero carry no information about whether the arms differ from each other**, and
two of the four arms are themselves significant.  The sampler-vs-sampler
contrast was never computed, so the sentence has no backing artifact.

This script computes it.  It is pure re-analysis of artifacts already on disk
(no GPU, no retraining).

=====================================================================
Pairing
=====================================================================
The four C3 runs are paired on ``(metal, seed)`` at ``K=20``:

* ``random`` / ``scaffold_diverse`` / ``activity_scaffold`` come from
  ``fsma_inhouse_c3_*_ep50.json`` (25 cells = 5 metals x 5 seeds).
* ``stratified`` is the ``head_k20`` slice of ``fsma_inhouse_kcurve_ep50.json``
  restricted to the same 5 seeds — the K-curve itself ran 10 seeds, so the
  extra 5 are dropped to keep the contrast paired.  The intersection is
  asserted, not assumed.

=====================================================================
Statistics
=====================================================================
* Friedman test across the four samplers on the 25 paired cells (the
  non-parametric repeated-measures omnibus) — this is the direct test of
  "indistinguishable".
* All 6 pairwise **two-sided** paired t-tests (two-sided because the claim
  under test is *no difference*, not a signed direction), Holm-corrected
  within the 6-test family.
* Per-metal means, since Ru is ~64% of the deduplicated cohort and can drive
  a pooled mean on its own.

Usage::

    python molmetal/scripts/fsma_c3_sampler_pairwise.py \
        --out metrics/by_metric/fsma_c3_sampler_pairwise.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

METRICS = PROJECT_ROOT / "metrics" / "by_metric"
SAMPLERS = ("random", "scaffold_diverse", "stratified", "activity_scaffold")
K = 20


def _load_cells(path: Path, k: int) -> Dict[Tuple[str, int], float]:
    d = json.loads(path.read_text())
    return {
        (r["metal"], int(r["seed"])): float(r["adapted_auc"])
        for r in d["per_run"]
        if int(r["k"]) == k and r.get("arm") == "none" and np.isfinite(r["adapted_auc"])
    }


def _load_zero(path: Path, k: int) -> Dict[Tuple[str, int], float]:
    """Zero-shot AUC per cell — the FSMA delta's baseline.

    NOTE: unlike the C2 controls, the three C3 runs do NOT share a query set
    with the stratified arm.  The sampler draws the K support rows, and the
    query is held-out-minus-support, so changing the rule swaps K rows between
    support and query.  ``n_query`` is identical across samplers but the
    zero-shot AUC is not, which is why the contrast is reported both on the
    absolute adapted AUC and on the delta-vs-zero-shot.
    """
    d = json.loads(path.read_text())
    return {
        (r["metal"], int(r["seed"])): float(r["zero_shot_auc"])
        for r in d["per_run"]
        if int(r["k"]) == k and r.get("arm") == "none" and np.isfinite(r["zero_shot_auc"])
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=METRICS / "fsma_c3_sampler_pairwise.json")
    args = ap.parse_args(argv)

    from scipy import stats

    sources = {
        "random": METRICS / "fsma_inhouse_c3_random_ep50.json",
        "scaffold_diverse": METRICS / "fsma_inhouse_c3_scaffold_diverse_ep50.json",
        "activity_scaffold": METRICS / "fsma_inhouse_c3_activity_scaffold_ep50.json",
        "stratified": METRICS / "fsma_inhouse_kcurve_ep50.json",
    }
    cells = {s: _load_cells(p, K) for s, p in sources.items()}
    zeros = {s: _load_zero(p, K) for s, p in sources.items()}

    shared = set.intersection(*[set(c) for c in cells.values()])
    if len(shared) < 5:
        raise SystemExit(f"too few paired cells: {len(shared)}")
    keys = sorted(shared)
    x = {s: np.array([cells[s][key] for key in keys]) for s in SAMPLERS}
    # FSMA delta = adapted - zero-shot, per cell (the family's reported quantity).
    delta = {s: x[s] - np.array([zeros[s][key] for key in keys]) for s in SAMPLERS}

    # Guard the premise: the arms must be the same (metal, seed) cells, and the
    # stratified arm must be the 5-seed subset of the 10-seed K-curve, i.e. the
    # intersection must be exactly the 25 cells the three C3 runs carry.
    if len(keys) != 25:
        raise SystemExit(
            f"expected 25 paired cells, found {len(keys)} "
            f"(per-sampler counts: { {s: len(cells[s]) for s in SAMPLERS} })"
        )

    friedman = stats.friedmanchisquare(*[x[s] for s in SAMPLERS])
    friedman_delta = stats.friedmanchisquare(*[delta[s] for s in SAMPLERS])

    def _pair_block(arrays):
        out, pv = {}, {}
        for a, b in combinations(SAMPLERS, 2):
            d = arrays[a] - arrays[b]
            t, p = stats.ttest_rel(arrays[a], arrays[b])  # two-sided: H0 = no difference
            se = d.std(ddof=1) / np.sqrt(len(d))
            crit = stats.t.ppf(0.975, len(d) - 1)
            out[f"{a}_vs_{b}"] = {
                "delta_mean": float(d.mean()),
                "delta_std": float(d.std(ddof=1)),
                "t": float(t),
                "p_two_sided": float(p),
                "ci95_ttest": [float(d.mean() - crit * se), float(d.mean() + crit * se)],
                "n_cells": int(len(d)),
                "n_a_better": int((d > 0).sum()),
            }
            pv[f"{a}_vs_{b}"] = float(p)
        order = sorted(pv, key=lambda k: pv[k])
        holm, running = {}, 0.0
        for i, name in enumerate(order):
            running = max(running, min(1.0, (len(order) - i) * pv[name]))
            holm[name] = running
        return out, pv, holm, order

    pairs, pvals, holm, order = _pair_block(x)
    pairs_d, _, holm_d, _ = _pair_block(delta)

    per_metal = {}
    for m in sorted({k[0] for k in keys}):
        sel = [i for i, k in enumerate(keys) if k[0] == m]
        per_metal[m] = {s: float(x[s][sel].mean()) for s in SAMPLERS}

    means = {s: float(x[s].mean()) for s in SAMPLERS}
    delta_means = {s: float(delta[s].mean()) for s in SAMPLERS}
    spread = max(means.values()) - min(means.values())
    max_pair = max(pairs.items(), key=lambda kv: abs(kv[1]["delta_mean"]))
    significant_pairs = [n for n, p in holm.items() if p < 0.05]
    significant_pairs_delta = [n for n, p in holm_d.items() if p < 0.05]

    verdict = (
        "SAMPLERS_DIFFER"
        if (friedman.pvalue < 0.05 or significant_pairs or significant_pairs_delta)
        else "SAMPLERS_INDISTINGUISHABLE"
    )

    rep = {
        "metric": "fsma_c3_sampler_pairwise",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "description": (
            "C3 follow-up: the support-sampler PAIRWISE contrast. The published C3 family "
            "reports four separate one-sample tests against zero, which cannot support the "
            "abstract's claim that the four samplers are statistically indistinguishable. "
            "This is the missing sampler-vs-sampler test, on the same 25 paired (metal, seed) "
            "cells at K=20. Pure re-analysis of on-disk artifacts."
        ),
        "unit": "paired ROC-AUC delta",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/fsma_c3_sampler_pairwise.py",
        "k": K,
        "n_cells": len(keys),
        "cells": [f"{m}:{s}" for m, s in keys],
        "samplers": list(SAMPLERS),
        "source_artifacts": {s: str(p.relative_to(PROJECT_ROOT)) for s, p in sources.items()},
        "sampler_mean": means,
        "sampler_mean_delta_vs_zero_shot": delta_means,
        "mean_spread": float(spread),
        "per_metal_mean": per_metal,
        "friedman": {"statistic": float(friedman.statistic), "p": float(friedman.pvalue)},
        "friedman_on_delta": {
            "statistic": float(friedman_delta.statistic),
            "p": float(friedman_delta.pvalue),
        },
        "pairwise": pairs,
        "pairwise_on_delta": pairs_d,
        "p_two_sided_holm": holm,
        "p_two_sided_holm_on_delta": holm_d,
        "max_abs_pairwise_delta": {"pair": max_pair[0], **max_pair[1]},
        "significant_pairs_holm_0.05": significant_pairs,
        "significant_pairs_holm_0.05_on_delta": significant_pairs_delta,
        "verdict": verdict,
        "honest_framing": (
            "MEASURED only. A negative verdict (SAMPLERS_INDISTINGUISHABLE) licenses the "
            "abstract's sentence; a positive one means the sentence must be reworded. The test "
            "is two-sided because the hypothesis under test is 'no difference', and it is "
            "Holm-corrected within the 6-pair family. Pooled means are reported but Ru is ~64% "
            "of the cohort, so per-metal means are the load-bearing view. "
            "PAIRING CAVEAT: the sampler draws the K support rows and the query is "
            "held-out-minus-support, so the four arms do NOT share a query set — n_query is "
            "identical but the zero-shot AUC is not. The contrast is therefore reported twice, "
            "on the absolute adapted AUC and on the delta-vs-zero-shot, and the verdict requires "
            "both to be indistinguishable. This is the same (metal, seed) cell structure the "
            "published C3 family uses (n_cells=25), so the test is consistent with it."
        ),
    }
    atomic_write_file(args.out, json.dumps(rep, indent=2, default=float, ensure_ascii=False))

    print(f"[c3-pairwise] verdict={verdict}")
    print(f"[c3-pairwise] sampler means: " + "  ".join(f"{s}={means[s]:+.4f}" for s in SAMPLERS))
    print(f"[c3-pairwise] delta-vs-zero means: " + "  ".join(f"{s}={delta_means[s]:+.4f}" for s in SAMPLERS))
    print(f"[c3-pairwise] spread={spread:.4f}  Friedman p={friedman.pvalue:.4g} "
          f"(on delta {friedman_delta.pvalue:.4g})")
    for name in order:
        v = pairs[name]
        flag = "SIG" if holm[name] < 0.05 else "   "
        print(f"  {flag} {name:>40}: delta={v['delta_mean']:+.4f} "
              f"ci95=[{v['ci95_ttest'][0]:+.4f},{v['ci95_ttest'][1]:+.4f}] "
              f"p2={v['p_two_sided']:.4f} holm={holm[name]:.4f} "
              f"| on-delta holm={holm_d[name]:.4f}")
    print(f"[c3-pairwise] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
