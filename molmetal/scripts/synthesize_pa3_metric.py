#!/usr/bin/env python3
"""Assemble the PA3 chemprop-LOMO metric from the trainer report + the XGB probe.

``train_chemprop_lomo_paper_cohort.py`` writes a per-run report to
``molmetal/reports/pa3_chemprop_lomo/metrics.json``. The canonical
``metrics/by_metric/pa3_chemprop_lomo_probe_cohort.json`` is that report plus
seven keys assembled by hand (``/tmp/promote_pa3_metric_v2.py``, ephemeral):
a comparison against the Morgan-FP+XGB probe, a label-permutation null on the
PA3 ranking, and four prose blocks. This script performs that assembly
deterministically so the canonical is reproducible from committed code.

What is recomputed vs carried
-----------------------------
* ``comparison_vs_xgb_probe`` — recomputed: the chemprop cross/within means come
  from the report, the XGB cross from ``metal_activity_transfer.json`` (seed 42)
  and the XGB within from ``metal_activity_transfer_seed_stability_v2.json``.
* the permutation null — recomputed from the report's stored seed-42
  ``test_probs``/``test_labels`` via the probe's own ``permutation_null``.
* ``why`` / ``caveats`` / ``verdict_vs_BLK_PA3`` / ``cross_only_first_run`` —
  carried as constants (commentary, not measurement).

The report itself is a GPU (bf16) artifact and drifts run-to-run; this synthesis
is deterministic *given* that report, exactly like the tmQM syntheses.

Usage
-----
::

    .venv/bin/python molmetal/scripts/synthesize_pa3_metric.py \
        --out-metric metrics/by_metric/pa3_chemprop_lomo_probe_cohort.json
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

from molmetal.scripts.metal_activity_transfer_probe import permutation_null  # noqa: E402

REPORT = "molmetal/reports/pa3_chemprop_lomo/metrics.json"
PROBE_METRIC = "metrics/by_metric/metal_activity_transfer.json"
STABILITY_METRIC = "metrics/by_metric/metal_activity_transfer_seed_stability_v2.json"
CROSS_ONLY_REPORT = "molmetal/reports/pa3_chemprop_lomo/metrics_cross_only_2026_10_12.json"

PERM_SEED = 42
N_PERMUTATIONS = 1000

# --- prose carried verbatim (commentary, not measurement) -------------------
WHY = (
    "BLK-PA3: the paper's Claim-1 collapse is measured with a Morgan-FP + XGBoost "
    "probe, and the 'it is the metal axis, not a broken backbone' argument rested "
    "on MoleculeNet (where this repo's D-MPNN loses to the chemprop reference on "
    "4/4 tasks).  This run replaces the scorer with the official chemprop v2 "
    "message-passing backbone on the SAME cohort and the SAME splits, so the "
    "comparison isolates the scorer."
)

CAVEATS: List[str] = [
    "Per-metal test folds are small for Rh (n=51), Os (n=36) and Re (n=74).  The "
    "label-permutation null on the fixed seed-42 ranking is the reference used "
    "here (1,000 shuffles, matching metal_activity_transfer_permutation_null.json); "
    "at those sizes the null sd is ~0.09-0.11, so a nominal 0.5 is not the right line.",
    "Only the CROSS column is seed-42 for the permutation null; the reported cross "
    "AUCs are 5-seed means.",
    "GPU bf16 training is not bit-reproducible run-to-run (~0.01-0.03 per-seed AUC "
    "drift).  Metal means are stable to <=0.02 across two independent launches.",
    "The scorer is a single pooled BinaryClassificationFFN over the chemprop encoder "
    "with no metal features \u2014 deliberately matching the XGB probe's contract.  "
    "The existing per-metal-head chemprop LOMO (metal_lomo_chemprop_v2.json) is a "
    "different, non-comparable design.",
]

VERDICT_VS_BLK_PA3 = (
    "CHEMPROP REPRODUCES THE CROSS-METAL COLLAPSE ON 2 OF 5 METALS (Ir, Rh), with a "
    "within-minus-cross penalty that is positive on 4 OF 5 (Ru, Ir, Rh, Re) \u2014 "
    "the same sign as the XGB probe.  It is significantly ABOVE chance only on Ru, "
    "Re and significantly BELOW chance on Ir; Rh, Os sit in the null band.  "
    "Ru: cross 0.5621 (XGB 0.6179, -0.0558); within 0.7853 (XGB 0.8271); penalty "
    "+0.2232. Ir: cross 0.4841 (XGB 0.5276, -0.0435); within 0.7901 (XGB 0.8430); "
    "penalty +0.3060. Rh: cross 0.4915 (XGB 0.5190, -0.0275); within 0.6658 "
    "(XGB 0.6974); penalty +0.1743. Os: cross 0.5559 (XGB 0.4950, +0.0609); within "
    "0.4133 (XGB 0.4960); penalty -0.1425. Re: cross 0.7368 (XGB 0.4429, +0.2939); "
    "within 0.8972 (XGB 0.9480); penalty +0.1604."
)

CROSS_ONLY_NOTE = (
    "The first PA3 launch ran --mode cross only.  The rerun (mode=both) reproduced "
    "the cross column within GPU bf16 run-to-run variation: e.g. Ru seed42 0.5697 -> "
    "0.5467, Re seed42 0.7583 -> 0.7607.  The 5-seed metal means moved by <=0.02.  "
    "Non-determinism is from cuDNN/bf16 reductions; the script seeds torch and numpy "
    "but does not call torch.use_deterministic_algorithms(True)."
)


def _load(rel: str) -> Dict[str, object]:
    return json.loads((PROJECT_ROOT / rel).read_text())


def _verdict(p_above: float, p_below: float) -> str:
    if p_above < 0.05:
        return "above (sig)"
    if p_below < 0.05:
        return "below (sig)"
    return "in null"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Assemble the PA3 probe-cohort metric from the trainer report."
    )
    p.add_argument("--report", default=REPORT)
    p.add_argument("--probe-metric", default=PROBE_METRIC)
    p.add_argument("--stability-metric", default=STABILITY_METRIC)
    p.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    p.add_argument("--perm-seed", type=int, default=PERM_SEED)
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/pa3_chemprop_lomo_probe_cohort.json",
    )
    args = p.parse_args(argv)

    report = _load(args.report)
    probe = _load(args.probe_metric)
    stability = _load(args.stability_metric)

    xgb_cross = {
        f["metal"]: round(float(f["roc_auc"]), 4)
        for f in probe["cross_metal_lomo_dedup"]["per_fold"]  # type: ignore[index]
        if f.get("status") == "MEASURED"
    }
    xgb_within = {
        m: float(d["mean"])
        for m, d in stability["within_metal_stability"].items()  # type: ignore[union-attr]
    }

    comparison: Dict[str, Dict[str, object]] = {}
    for fold in report["per_fold"]:  # type: ignore[index]
        m = fold["metal"]
        within = report["within_metal"][m]  # type: ignore[index]
        seed42 = fold["per_seed"]["42"]  # type: ignore[index]
        perm = permutation_null(
            seed42["test_labels"], seed42["test_probs"],
            args.n_permutations, args.perm_seed,
        )
        cross_mean = float(fold["mean_auc"])
        within_mean = float(within["mean_auc"])
        xc, xw = xgb_cross[m], xgb_within[m]
        comparison[m] = {
            "n_test": int(fold["n_test"]),
            "n_train": int(fold["n_train"]),
            "chemprop_cross_mean_auc": cross_mean,
            "chemprop_cross_std": float(fold["std"]),
            "xgb_cross_seed42_auc": xc,
            "cross_delta": cross_mean - xc,
            "chemprop_within_mean_auc": within_mean,
            "chemprop_within_std": float(within["std"]),
            "xgb_within_auc": xw,
            "within_delta": within_mean - xw,
            "chemprop_penalty_within_minus_cross": within_mean - cross_mean,
            "xgb_penalty_within_minus_cross": xw - xc,
            "permutation_null_verdict": _verdict(
                float(perm["p_above_chance"]), float(perm["p_below_chance"])
            ),
            "permutation_p_above_chance": float(perm["p_above_chance"]),
        }

    out = dict(report)
    out.update({
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "why": WHY,
        "comparison_vs_xgb_probe": comparison,
        "cross_only_first_run": {"report": CROSS_ONLY_REPORT, "note": CROSS_ONLY_NOTE},
        "verdict_vs_BLK_PA3": VERDICT_VS_BLK_PA3,
        "caveats": CAVEATS,
    })

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    dest = Path(args.out_metric)
    # ensure_ascii=False keeps the em-dashes readable in the artifact (it is cited
    # by the paper); the canonical was written this way.
    atomic_write_file(dest, json.dumps(out, indent=2, default=float, ensure_ascii=False))

    for m, d in comparison.items():
        print(f"{m:>4}  cross {d['chemprop_cross_mean_auc']:.4f} "
              f"(XGB {d['xgb_cross_seed42_auc']:.4f})  within "
              f"{d['chemprop_within_mean_auc']:.4f}  "
              f"{d['permutation_null_verdict']}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
