#!/usr/bin/env python3
"""Device-matched capacity pair analysis: h128 vs h512 within-metal D-MPNN.

=====================================================================
Why this exists
=====================================================================
Table 1's within-metal column is produced by ``dmpnn_within_metal_auc.py``. The
published column was measured on CPU at ``hidden=128``. Comparing a GPU h512 run
against it would confound capacity with device, so #184 re-measured BOTH widths
on the same device (``cuda:0``, ``--deterministic``) with ``model.hidden`` as the
only free variable.

This script is a pure derivation over two MEASURED JSONs — it trains nothing. It
emits per-metal paired per-seed capacity deltas (h512 - h128) with a paired
one-sided t-test and Holm correction, plus the two facts that make the pairing
credible:

1. **Invariant check** — every non-capacity field (cohort, folds, epochs, lr,
   depth, dropout, batch size, split seeds, device, deterministic) must match.
2. **Fold-identity check** — the XGBoost column is deterministic given the fold
   split, so ``xgb_auc`` must be bit-identical per (metal, split_seed) across the
   two runs. If it is, both runs saw the same folds.

Honest framing: a positive paired delta means *capacity helps on these folds*,
not that the D-MPNN beats XGBoost. The XGB column is carried through unchanged so
the remaining G1 gap is visible next to every delta.

Usage
-----
::

    .venv/bin/python molmetal/scripts/wfA_capacity_pair_analysis.py \
        --h128 metrics/by_metric/dmpnn_within_metal_cap128_pf_gpu0.json \
        --h512 metrics/by_metric/dmpnn_within_metal_cap512_pf_gpu0.json \
        --out-metric metrics/by_metric/dmpnn_within_metal_capacity_pair_gpu0.json
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

from molmetal.scripts.synthesize_tmqm_pretrain_init_ab_multimetal import (  # noqa: E402
    _holm,
    _paired_stats,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

# Fields that must be identical for the pair to isolate capacity. Anything that
# changes the data, the folds, or the optimisation budget belongs here.
INVARIANT_ARCH_KEYS = ("depth", "dropout", "lr", "epochs", "batch_size", "class_weight", "mask_padding")
INVARIANT_TOP_KEYS = ("cohort", "cohort_spec", "fold_protocol", "data_source", "device", "deterministic")
XGB_TOL = 1e-12


def _per_seed_auc(metal_block: Dict[str, object], key: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for split in metal_block["per_split"]:
        out[int(split["split_seed"])] = float(split[key])
    return out


def _check_invariants(h128: Dict[str, object], h512: Dict[str, object]) -> Dict[str, object]:
    mismatches: List[str] = []
    for k in INVARIANT_TOP_KEYS:
        if h128.get(k) != h512.get(k):
            mismatches.append(f"{k}: h128={h128.get(k)!r} vs h512={h512.get(k)!r}")
    a128, a512 = h128["dmpnn_arch"], h512["dmpnn_arch"]
    for k in INVARIANT_ARCH_KEYS:
        if a128.get(k) != a512.get(k):
            mismatches.append(f"dmpnn_arch.{k}: {a128.get(k)!r} vs {a512.get(k)!r}")
    if int(a128["hidden"]) != 128:
        mismatches.append(f"h128 file has hidden={a128['hidden']}")
    if int(a512["hidden"]) != 512:
        mismatches.append(f"h512 file has hidden={a512['hidden']}")
    if list(h128["split_seeds"]) != list(h512["split_seeds"]):
        mismatches.append(f"split_seeds differ: {h128['split_seeds']} vs {h512['split_seeds']}")
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "checked_keys": list(INVARIANT_TOP_KEYS) + [f"dmpnn_arch.{k}" for k in INVARIANT_ARCH_KEYS],
        "hidden": {"h128": int(a128["hidden"]), "h512": int(a512["hidden"])},
        "mismatches": mismatches,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="h128 vs h512 device-matched capacity pair analysis.")
    p.add_argument("--h128", default="metrics/by_metric/dmpnn_within_metal_cap128_pf_gpu0.json")
    p.add_argument("--h512", default="metrics/by_metric/dmpnn_within_metal_cap512_pf_gpu0.json")
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/dmpnn_within_metal_capacity_pair_gpu0.json",
    )
    args = p.parse_args(argv)

    h128 = json.loads((PROJECT_ROOT / args.h128).read_text())
    h512 = json.loads((PROJECT_ROOT / args.h512).read_text())

    invariants = _check_invariants(h128, h512)
    if invariants["status"] != "PASS":
        print("INVARIANT FAIL — refusing to derive a capacity delta:", file=sys.stderr)
        for m in invariants["mismatches"]:
            print(f"  {m}", file=sys.stderr)
        return 1

    per_metal: Dict[str, object] = {}
    pvals: Dict[str, float] = {}
    fold_identity: Dict[str, object] = {}
    all_identical = True

    for metal in h128["per_metal"]:
        b128 = h128["per_metal"][metal]
        b512 = h512["per_metal"][metal]
        seeds = sorted(_per_seed_auc(b128, "dmpnn_auc"))
        if seeds != sorted(_per_seed_auc(b512, "dmpnn_auc")):
            print(f"FAIL: {metal} split seeds differ between runs", file=sys.stderr)
            return 1

        xgb128 = _per_seed_auc(b128, "xgb_auc")
        xgb512 = _per_seed_auc(b512, "xgb_auc")
        worst = max(abs(xgb128[s] - xgb512[s]) for s in seeds)
        identical = worst <= XGB_TOL
        all_identical = all_identical and identical
        fold_identity[metal] = {
            "max_abs_xgb_diff_across_seeds": worst,
            "identical": identical,
        }

        auc128 = _per_seed_auc(b128, "dmpnn_auc")
        auc512 = _per_seed_auc(b512, "dmpnn_auc")
        deltas = np.array([auc512[s] - auc128[s] for s in seeds], dtype=float)
        stats = _paired_stats(deltas)
        pvals[metal] = float(stats["p_one_sided_ttest"])

        xgb_auc = float(b128["xgb_auc_same_folds"])
        gap128 = float(b128["delta_dmpnn_minus_xgb"])
        gap512 = float(b512["delta_dmpnn_minus_xgb"])
        # Fraction of the h128 gap to XGB closed by widening. Only meaningful
        # when h128 was below XGB; left null otherwise.
        closed = (gap512 - gap128) / abs(gap128) if gap128 < 0 else None

        per_metal[metal] = {
            "n_unique": int(b128["n_unique"]),
            "krasnov_reported_auc": b128.get("krasnov_reported_auc"),
            "xgb_auc_same_folds": xgb_auc,
            "h128_auc": float(b128["dmpnn_auc"]),
            "h128_std": float(b128["dmpnn_auc_across_folds_std"]),
            "h512_auc": float(b512["dmpnn_auc"]),
            "h512_std": float(b512["dmpnn_auc_across_folds_std"]),
            "std_direction": (
                "wider" if b512["dmpnn_auc_across_folds_std"] > b128["dmpnn_auc_across_folds_std"]
                else "narrower" if b512["dmpnn_auc_across_folds_std"] < b128["dmpnn_auc_across_folds_std"]
                else "unchanged"
            ),
            "per_seed": {
                str(s): {
                    "h128": round(auc128[s], 6),
                    "h512": round(auc512[s], 6),
                    "capacity_delta": round(auc512[s] - auc128[s], 6),
                }
                for s in seeds
            },
            "capacity_delta": stats,
            "gap_to_xgb_h128": round(gap128, 6),
            "gap_to_xgb_h512": round(gap512, 6),
            "gap_closed_fraction": (round(closed, 4) if closed is not None else None),
            "h512_beats_xgb": bool(gap512 > 0),
            "h512_beats_krasnov": (
                None if b128.get("krasnov_reported_auc") is None
                else bool(float(b512["dmpnn_auc"]) > float(b128["krasnov_reported_auc"]))
            ),
            "wall_h128_sec": float(b128.get("wall_sec") or 0.0) or None,
            "wall_h512_sec": float(b512.get("wall_sec") or 0.0) or None,
        }

    holm = _holm(pvals)

    n_positive = sum(1 for m in per_metal if per_metal[m]["capacity_delta"]["mean"] > 0)
    sig = set(holm["significant_after_holm_0.05"])
    report = {
        "metric": "dmpnn_within_metal_capacity_pair_delta",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "description": (
            "Device-matched capacity ablation on the Table-1 within-metal column. Both widths "
            "were trained on cuda:0 with --deterministic; model.hidden is the only free variable. "
            "Delta = h512 AUC - h128 AUC on the same folds and the same split seeds."
        ),
        "unit": "ROC-AUC difference (paired per split_seed)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/wfA_capacity_pair_analysis.py",
        "inputs": {"h128": args.h128, "h512": args.h512},
        "pairing": {
            "invariants": invariants,
            "fold_identity_check": {
                "criterion": f"max |xgb_auc(h128) - xgb_auc(h512)| <= {XGB_TOL} per (metal, split_seed)",
                "per_metal": fold_identity,
                "all_identical": all_identical,
                "interpretation": (
                    "XGBoost is deterministic given the fold split, so an identical column means "
                    "both runs used the same cohort and the same folds."
                    if all_identical else
                    "XGB column differs — the two runs did NOT see the same folds; deltas are not paired."
                ),
            },
        },
        "statistics": {
            "test": "paired one-sided t-test on the 5 per-seed deltas (H1: h512 > h128)",
            "family_correction": holm["correction"],
            "p_one_sided_holm": holm["p_one_sided_holm"],
            "significant_after_holm_0.05": holm["significant_after_holm_0.05"],
            "why_not_bootstrap": (
                "At n=5 a percentile bootstrap cannot produce enough distinct resamples and is "
                "anti-conservative (see synthesize_tmqm_pretrain_init_ab_multimetal._paired_stats)."
            ),
        },
        "per_metal": per_metal,
        "summary": {
            "metals_positive": n_positive,
            "metals_total": len(per_metal),
            "metals_significant_after_holm": sorted(sig),
            "all_seeds_positive_metals": [
                m for m in per_metal if per_metal[m]["capacity_delta"]["all_seeds_positive"]
            ],
            "h512_beats_xgb_metals": [m for m in per_metal if per_metal[m]["h512_beats_xgb"]],
        },
        "honest_framing": {
            "do_not_claim": [
                "that widening closes the gap to ECFP4->XGBoost — report h512_beats_xgb_metals",
                "that capacity monotonically helps — small cohorts are reported individually",
                "any pooled 5-metal mean (the metals differ in n by ~19x and are not exchangeable)",
            ],
            "what_this_says": (
                "Per-metal, on identical folds and identical device, how much of the h128 gap "
                "is width rather than architecture or featurisation."
            ),
        },
    }

    out = PROJECT_ROOT / args.out_metric
    atomic_write_file(out, json.dumps(report, indent=2, default=float, ensure_ascii=False))

    print(f"\n{'metal':>5} {'n':>6} {'h128':>7} {'h512':>7} {'delta':>8} {'+seeds':>6} {'p':>7} {'pHolm':>7}")
    for m, r in per_metal.items():
        s = r["capacity_delta"]
        print(
            f"{m:>5} {r['n_unique']:>6} {r['h128_auc']:>7.4f} {r['h512_auc']:>7.4f} "
            f"{s['mean']:>+8.4f} {s['n_seeds_positive']:>3}/{s['n_seeds']} "
            f"{s['p_one_sided_ttest']:>7.4f} {holm['p_one_sided_holm'][m]:>7.4f}"
        )
    print(f"\nfold identity: {'PASS' if all_identical else 'FAIL'} (all 5 metals, all seeds)")
    print(f"Holm significant: {holm['significant_after_holm_0.05']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
