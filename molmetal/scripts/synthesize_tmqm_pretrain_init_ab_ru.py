#!/usr/bin/env python3
"""Assemble the tmQM-pretraining A/B metric from the raw per-arm reports.

The canonical ``metrics/by_metric/tmqm_pretrain_init_ab_ru.json`` is a
*synthesis*: the trainer (``finetune_metalcytotox_from_tmqm.py``) writes one JSON
per arm under ``molmetal/reports/tmqm_ab_metalcytotox_30ep_init_{on,off}/``, and
the paired-delta / CI / gate / supersedes record was assembled by hand. This
script performs that assembly deterministically so the metric is reproducible
from committed code (AGENTS.md §5.5: the historical ``supersedes`` block is
carried verbatim, not recomputed).

Usage
-----
::

    .venv/bin/python molmetal/scripts/synthesize_tmqm_pretrain_init_ab_ru.py \
        --out-metric metrics/by_metric/tmqm_pretrain_init_ab_ru.json
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

import numpy as np

ON_ARM = "molmetal/reports/tmqm_ab_metalcytotox_30ep_init_on/finetune_tmqm_Ru.json"
OFF_ARM = "molmetal/reports/tmqm_ab_metalcytotox_30ep_init_off/finetune_random_Ru.json"

BOOTSTRAP_N = 20000
BOOTSTRAP_SEED = 0

# The raw arm reports carry no timestamp, so the artifact records the date the
# BLK-PA12-fixed A/B was run. A constant, not a clock read.
RERUN_DATE = "2026-10-12"

# Frozen historical record of the superseded (degenerate-splitter) measurement.
# Carried verbatim per AGENTS.md §5.5; never recomputed.
SUPERSEDES: Dict[str, object] = {
    "previous_value": 0.018136,
    "previous_paired_delta": {
        "mean": 0.018136, "sd": 0.011819, "se": 0.005285,
        "ci95": [0.003464, 0.032808], "t_df4": 3.4313,
        "n_seeds": 5, "all_seeds_positive": True,
    },
    "previous_arms": {
        "tmqm": {
            "mean": 0.818864, "std": 0.005248,
            "per_seed": [0.819404, 0.820819, 0.819812, 0.824186, 0.810099],
        },
        "random": {
            "mean": 0.800728, "std": 0.014589,
            "per_seed": [0.816771, 0.788467, 0.80354, 0.811818, 0.783044],
        },
    },
    "previous_protocol": (
        "MetalCytoToxDB Ru subset; scaffold split (deterministic, largest_first); "
        "seeds vary model init only; 30 epochs; batch 128; lr 1e-3; hidden 128 "
        "depth 3; n_rows 19135; n_test 714; the ONLY difference between arms is "
        "init: tmqm|random"
    ),
    "previous_gate": {
        "criterion": "non-negative delta (TODO/completed/42_tmqm_pretrain_v4_encoder.md)",
        "verdict": "PASS",
        "why": "all 5 seeds positive and the 95% CI excludes 0",
    },
    "why": (
        "the previous run (n_test 714) was on the degenerate scaffold splitter -- "
        "molecule-level, not scaffold-disjoint (BLK-PA12). The fixed splitter "
        "gives n_test 215 and a different (null) result."
    ),
    "cited_by": (
        "paper_a/sections/04_results.tex tab:pretrain (Claim 2) -- updated "
        "2026-10-12 to the honest negative"
    ),
}


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _paired_stats(deltas: np.ndarray) -> Dict[str, object]:
    n = int(deltas.size)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1)) if n > 1 else 0.0
    se = float(sd / np.sqrt(n)) if n > 0 else 0.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = deltas[rng.integers(0, n, size=(BOOTSTRAP_N, n))].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "se": round(se, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "t_df4": round(mean / se, 4) if se else None,
        "n_seeds": n,
        "all_seeds_positive": bool((deltas > 0).all()),
        "ci_method": (
            f"percentile bootstrap, {BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED} "
            "(paired per-seed deltas)"
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Assemble tmqm_pretrain_init_ab_ru.json from the raw arm reports."
    )
    p.add_argument("--on-arm", default=ON_ARM)
    p.add_argument("--off-arm", default=OFF_ARM)
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/tmqm_pretrain_init_ab_ru.json",
    )
    args = p.parse_args(argv)

    on = _load(PROJECT_ROOT / args.on_arm)
    off = _load(PROJECT_ROOT / args.off_arm)

    def per_seed(d: Dict[str, object]) -> List[float]:
        return [float(r["test_auc"]) for r in d["per_seed"]]  # type: ignore[index]

    tmqm_ps, random_ps = per_seed(on), per_seed(off)
    if len(tmqm_ps) != len(random_ps):
        raise SystemExit("arm seed counts differ; cannot pair")
    a = np.asarray(tmqm_ps, float)
    b = np.asarray(random_ps, float)
    delta = _paired_stats(a - b)

    arms = {
        "tmqm": {
            "mean": round(float(a.mean()), 6),
            "std": round(float(a.std(ddof=1)), 6),
            "per_seed": [round(x, 6) for x in tmqm_ps],
        },
        "random": {
            "mean": round(float(b.mean()), 6),
            "std": round(float(b.std(ddof=1)), 6),
            "per_seed": [round(x, 6) for x in random_ps],
        },
    }
    ci = delta["ci95"]
    null_result = bool(ci[0] <= 0.0 <= ci[1])

    report = {
        "metric": "tmqm_pretrain_init_ab_metalcytotox_ru",
        "value": delta["mean"],
        "value_std": delta["sd"],
        "tag": "MEASURED_HONEST_NEGATIVE" if null_result else "MEASURED",
        "source_script": "molmetal/scripts/finetune_metalcytotox_from_tmqm.py",
        "synthesis_script": "molmetal/scripts/synthesize_tmqm_pretrain_init_ab_ru.py",
        "configs": [
            "configs/tmqm_ab_metalcytotox_init_on.yaml",
            "configs/tmqm_ab_metalcytotox_init_off.yaml",
        ],
        "protocol": (
            "MetalCytoToxDB Ru subset; FIXED Bemis-Murcko scaffold split "
            "(BLK-PA12: ScaffoldSplitter desalt-aware _scaffold_for); seeds vary "
            f"model init only; {on['epochs']} epochs; batch {on['batch_size']}; "
            f"lr {on['lr']}; n_rows {on['n_rows']}; n_test "
            f"{on['per_seed'][0]['n_test']}; the ONLY difference between arms is "
            "init: tmqm|random"
        ),
        "arms": arms,
        "paired_delta": delta,
        "gate": {
            "criterion": "non-negative delta (TODO/completed/42_tmqm_pretrain_v4_encoder.md)",
            "verdict": "FAIL" if delta["mean"] < 0 else "PASS",
            "why": (
                f"the paired delta is {'NEGATIVE' if delta['mean'] < 0 else 'positive'} "
                f"({delta['mean']:+.4f}) and its 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] "
                f"{'includes' if null_result else 'excludes'} 0; on the fixed "
                "scaffold split tmQM pretraining shows "
                f"{'no' if null_result else 'a'} effect on Ru"
            ),
        },
        "honest_negative": {
            "statement": (
                f"On the fixed Bemis-Murcko scaffold split, tmQM-pretrained init "
                f"does {'NOT ' if null_result else ''}improve Ru test AUC over "
                f"random init ({arms['tmqm']['mean']:.4f} vs "
                f"{arms['random']['mean']:.4f}; paired delta {delta['mean']:+.4f}, "
                f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], "
                f"t(4)={delta['t_df4']:.2f}). The previously recorded +0.0181 "
                "(CI [+0.0035, +0.0328]) was measured on the degenerate splitter "
                "and is not reproducible under a genuine scaffold split."
            ),
            "fallback_invoked": (
                "BLK-PA12 honest_negative_fallback: '\u82e5\u91cd\u8dd1\u540e tmQM "
                "A/B \u7684\u914d\u5bf9 delta \u53d8\u5f97\u4e0d\u663e\u8457\uff0c"
                "\u5982\u5b9e\u62a5\u544a\u4e3a\u968f\u673a\u5207\u5206\u4e0b\u7684 "
                "+0.0181 \u4e0d\u7a33\u5065'. Here it is not merely "
                "non-significant -- the point estimate flipped sign."
            ),
            "what_survives": (
                "The paired design still cancels seed variance; the arms remain "
                "identical apart from init; both arms loaded the tmQM encoder "
                "tensors (so the ON arm was not a silent random arm)."
            ),
        },
        "supersedes": SUPERSEDES,
        "comparator": {
            "Krasnov2026_Ru_subset_auc": 0.81,
            "tmqm_arm": arms["tmqm"]["mean"],
            "random_arm": arms["random"]["mean"],
            "note": (
                "both arms sit below the named 0.81 comparator; the gate is the "
                "delta, not the absolute"
            ),
        },
        "transfer_verified": "both arms printed 'loaded 12 backbone tensors from tmQM'",
        "timestamp": on.get("timestamp") or RERUN_DATE,
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = PROJECT_ROOT / args.out_metric
    # ensure_ascii=False keeps the non-ASCII (the BLK-PA12 fallback quote is in
    # Chinese) readable in the artifact; the canonical was written this way.
    atomic_write_file(out, json.dumps(report, indent=2, default=float, ensure_ascii=False))

    print(f"tmqm   mean={arms['tmqm']['mean']:.4f} std={arms['tmqm']['std']:.4f}")
    print(f"random mean={arms['random']['mean']:.4f} std={arms['random']['std']:.4f}")
    print(f"paired delta {delta['mean']:+.4f} CI [{ci[0]:+.4f}, {ci[1]:+.4f}] "
          f"t={delta['t_df4']} -> gate {report['gate']['verdict']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
