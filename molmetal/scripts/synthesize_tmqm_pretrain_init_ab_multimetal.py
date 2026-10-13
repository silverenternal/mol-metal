#!/usr/bin/env python3
"""BLK-PA4 — multi-metal tmQM-pretrain A/B synthesis (Ru + Ir).

=====================================================================
Why this exists
=====================================================================
BLK-PA4's finding was that Claim 2 ("tmQM pretraining transfers") rested on a
single metal (Ru), 5 seeds. On the fixed scaffold split the Ru effect is NULL
(paired delta -0.0094, CI includes 0; see
``metrics/by_metric/tmqm_pretrain_init_ab_ru.json``). The blocker's fix step 1
is to extend the A/B to >=2 metals and report per-metal paired deltas with a
multiple-comparison correction.

This script reads the per-arm trainer reports for each metal and emits
``metrics/by_metric/tmqm_pretrain_init_ab_multimetal.json``: per-metal paired
delta + bootstrap CI, plus a Holm-Bonferroni-corrected family verdict.

It does NOT recompute the Ru supersedes history — that lives in the Ru
single-metal artifact and is referenced, not duplicated (AGENTS.md §5.5).

Usage
-----
::

    .venv/bin/python molmetal/scripts/synthesize_tmqm_pretrain_init_ab_multimetal.py \
        --out-metric metrics/by_metric/tmqm_pretrain_init_ab_multimetal.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

BOOTSTRAP_N = 20000
BOOTSTRAP_SEED = 0

# metal -> (on-arm report, off-arm report). Both arms come from the same fixed
# Bemis-Murcko scaffold split (BLK-PA12); the only difference is init.
ARM_PATHS: Dict[str, Dict[str, str]] = {
    "Ru": {
        "on": "molmetal/reports/tmqm_ab_metalcytotox_30ep_init_on/finetune_tmqm_Ru.json",
        "off": "molmetal/reports/tmqm_ab_metalcytotox_30ep_init_off/finetune_random_Ru.json",
    },
    "Ir": {
        "on": "molmetal/reports/tmqm_ab_metalcytotox_ir_30ep_init_on/finetune_tmqm_Ir.json",
        "off": "molmetal/reports/tmqm_ab_metalcytotox_ir_30ep_init_off/finetune_random_Ir.json",
    },
}


def _paired_stats(deltas: np.ndarray) -> Dict[str, object]:
    from scipy import stats as _st

    n = int(deltas.size)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1)) if n > 1 else 0.0
    se = float(sd / np.sqrt(n)) if n > 0 else 0.0
    t = float(mean / se) if se else 0.0

    # Conventional paired test for the 5 per-seed deltas: one-sided t.
    # This is the p used for the family correction. The percentile bootstrap
    # below is reported for reference only: at n=5 it cannot generate enough
    # distinct resamples and is anti-conservative (MEASURED: Ir bootstrap
    # p=0.012 vs t p=0.068 vs Wilcoxon p=0.156 on the same 5 deltas).
    p_t = float(_st.t.sf(t, df=n - 1)) if (n > 1 and se) else 1.0
    try:
        p_w: Optional[float] = float(_st.wilcoxon(deltas, alternative="greater").pvalue)
    except Exception:
        p_w = None

    tcrit = float(_st.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    ci_t = [round(mean - tcrit * se, 6), round(mean + tcrit * se, 6)]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = deltas[rng.integers(0, n, size=(BOOTSTRAP_N, n))].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    p_leq0 = float((boot <= 0.0).mean())
    return {
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "se": round(se, 6),
        "ci95": ci_t,
        "ci95_ttest": ci_t,
        "ci95_bootstrap_reference_only": [round(lo, 6), round(hi, 6)],
        "t": round(t, 4) if se else None,
        "n_seeds": n,
        "n_seeds_positive": int((deltas > 0).sum()),
        "all_seeds_positive": bool((deltas > 0).all()),
        "p_one_sided_ttest": round(p_t, 6),
        "p_one_sided_wilcoxon": (round(p_w, 6) if p_w is not None else None),
        "p_one_sided_bootstrap_reference_only": round(p_leq0, 6),
        "p_used_for_family_correction": "p_one_sided_ttest",
        "ci_used_for_verdict": "ci95_ttest",
        "ci_method": (
            f"t-based 95% CI (t_{{.975,{n - 1}}}={tcrit:.3f}); percentile "
            f"bootstrap ({BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED}) is "
            "reported as reference only and is anti-conservative at n=5"
        ),
    }


def _holm(pvals: Dict[str, float]) -> Dict[str, object]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, p * (m - rank)))
        adj[k] = round(running, 6)
    return {
        "correction": f"Holm-Bonferroni (family = {m} metals)",
        "n_family": m,
        "p_one_sided_holm": adj,
        "significant_after_holm_0.05": [k for k, p in adj.items() if p < 0.05],
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Multi-metal tmQM-pretrain A/B synthesis (BLK-PA4)."
    )
    p.add_argument("--metals", nargs="+", default=["Ru", "Ir"])
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/tmqm_pretrain_init_ab_multimetal.json",
    )
    args = p.parse_args(argv)

    per_metal: Dict[str, object] = {}
    pvals: Dict[str, float] = {}
    for metal in args.metals:
        paths = ARM_PATHS[metal]
        on = json.loads((PROJECT_ROOT / paths["on"]).read_text())
        off = json.loads((PROJECT_ROOT / paths["off"]).read_text())
        a = np.asarray([float(r["test_auc"]) for r in on["per_seed"]], float)
        b = np.asarray([float(r["test_auc"]) for r in off["per_seed"]], float)
        if a.size != b.size:
            raise SystemExit(f"{metal}: arm seed counts differ ({a.size} vs {b.size})")
        stats = _paired_stats(a - b)
        pvals[metal] = float(stats["p_one_sided_ttest"])
        per_metal[metal] = {
            "n_rows": on["n_rows"],
            "n_test": on["per_seed"][0]["n_test"],
            "epochs": on["epochs"],
            "batch_size": on["batch_size"],
            "lr": on["lr"],
            "arms": {
                "tmqm": {
                    "mean": round(float(a.mean()), 6),
                    "std": round(float(a.std(ddof=1)), 6),
                    "per_seed": [round(x, 6) for x in a.tolist()],
                },
                "random": {
                    "mean": round(float(b.mean()), 6),
                    "std": round(float(b.std(ddof=1)), 6),
                    "per_seed": [round(x, 6) for x in b.tolist()],
                },
            },
            "paired_delta": stats,
            "verdict": (
                "NULL"
                if stats["ci95"][0] <= 0.0 <= stats["ci95"][1]
                else ("POSITIVE" if stats["mean"] > 0 else "NEGATIVE")
            ),
        }
        print(
            f"[blk-pa4] {metal:>3} tmqm={per_metal[metal]['arms']['tmqm']['mean']:.4f} "
            f"random={per_metal[metal]['arms']['random']['mean']:.4f} "
            f"delta={stats['mean']:+.4f} CI[{stats['ci95'][0]:+.4f},{stats['ci95'][1]:+.4f}] "
            f"p_t={stats['p_one_sided_ttest']:.4f} "
            f"({stats['n_seeds_positive']}/{stats['n_seeds']}+) -> {per_metal[metal]['verdict']}",
            flush=True,
        )

    holm = _holm(pvals)
    any_holm_positive = bool(holm["significant_after_holm_0.05"])
    report = {
        "metric": "tmqm_pretrain_init_ab_multimetal",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "paired delta in test ROC-AUC (tmqm init minus random init)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/synthesize_tmqm_pretrain_init_ab_multimetal.py",
        "source_script": "molmetal/scripts/finetune_metalcytotox_from_tmqm.py",
        "why": (
            "BLK-PA4 fix step 1: the tmQM-pretrain A/B was Ru-only (5 seeds). "
            "Extend to a second metal and report per-metal paired deltas with a "
            "Holm-Bonferroni family correction."
        ),
        "protocol": (
            "per metal: MetalCytoToxDB subset; FIXED Bemis-Murcko scaffold split "
            "(BLK-PA12); seeds vary model init only; the ONLY difference between "
            "arms is init: tmqm|random"
        ),
        "metals": list(args.metals),
        "per_metal": per_metal,
        "holm": holm,
        "family_verdict": (
            "AT_LEAST_ONE_METAL_SIGNIFICANT_POSITIVE_AFTER_HOLM_T"
            if any_holm_positive
            else "NO_METAL_SIGNIFICANT_POSITIVE_AFTER_HOLM_T"
        ),
        "significance_note": (
            "The family correction uses the paired one-sided t-test on the 5 "
            "per-seed deltas, not the percentile bootstrap. At n=5 the bootstrap "
            "is anti-conservative: on the Ir arm it returns p=0.0120 while the "
            "t-test returns 0.0680 and Wilcoxon 0.1562 on the identical deltas, "
            "because 5 points cannot generate enough distinct resamples. The "
            "bootstrap CI is kept in each arm as *_reference_only."
        ),
        "honest_negative": {
            "claim_2_status": (
                "Metal-specific, not general. On the fixed scaffold split the "
                "tmQM-init benefit is positive only where it is Holm-significant "
                "(see family_verdict) and null elsewhere; it must not be written "
                "as a blanket 'pretraining transfers'."
            ),
            "ir_is_suggestive_not_confirmed": (
                "Ir's point estimate is +0.0486 but only 3/5 seeds are positive "
                "and the t-based 95% CI includes zero, so Ir is reported as a "
                "suggestive benefit, not a confirmed one."
            ),
            "ru_reference": (
                "the Ru single-metal record (with its supersedes history) lives "
                "in metrics/by_metric/tmqm_pretrain_init_ab_ru.json; not "
                "duplicated here (AGENTS.md §5.5)"
            ),
            "do_not_claim": (
                "that a result on Ru+Ir generalises to all five metals — only "
                "the two tested are covered"
            ),
        },
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = PROJECT_ROOT / args.out_metric
    atomic_write_file(out, json.dumps(report, indent=2, default=float, ensure_ascii=False))

    print(f"\nHolm: significant = {holm['significant_after_holm_0.05']}")
    print(f"family verdict: {report['family_verdict']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
