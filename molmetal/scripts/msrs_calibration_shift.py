#!/usr/bin/env python3
"""UP-6 — conditional coverage, stratified by the stratum that actually fails.

=====================================================================
What the existing artifact shows
=====================================================================
``msrs_calibration.json`` (DOI protocol, alpha=0.10):

    variant      coverage   median width   worst stratum
    global        0.9004      2.0052        0.8096  (cohort size 2, n=851)
    metal         0.9002      2.0231        0.8331  (cohort size 2)
    difficulty    0.8789      1.9195        0.8073  (cohort size 2)

The failure is **not** by metal — metal-Mondrian moves the failing group from
0.8096 to 0.8331 and pays for it with width.  It is by **cohort size**: rows
whose (ligand, cell) pair was measured with two metals are the cross-metal
comparison cohort, and they are exactly the rows the model is worst on.

So the fix is a Mondrian split on the stratum that fails:

    cohort         one conformal quantile per cohort size
    cohort_metal   per (cohort size, metal), falling back to cohort, then global

Width is a different matter.  A 90% interval of width W implies a residual SD
of about W/3.29; width 2.0052 therefore implies SD 0.61, which is what
R2 = 0.38 on a target with SD ~0.78 produces.  **No calibration scheme can
shrink width without improving accuracy** — the gate allows width <= 1.75, or
a width increase under 5% paired with coverage >= 0.88, and the second branch
is the realistic one.

=====================================================================
Gate (frozen 2026-10-21T15:10+08:00)
=====================================================================
    worst stratum coverage >= 0.87
    AND (median width <= 1.75 OR (width increase < 5% AND coverage >= 0.88))

=====================================================================
Usage
=====================================================================
::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_calibration_shift.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_actionability import ALPHA  # noqa: E402
from molmetal.scripts.msrs_calibration import (  # noqa: E402
    ARM,
    cohort_sizes,
    run_protocol,
)
from molmetal.scripts.msrs_surface_model import _git_commit, _sha256  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

EXTRA = ("cohort", "cohort_metal")
REFERENCE = {"coverage": 0.9004, "median_width": 2.0052, "worst_stratum": 0.8096}
COVERAGE_GATE = 0.87
COVERAGE_STRONG = 0.88
WIDTH_GATE = 1.75
WIDTH_TOLERANCE = 0.05


def worst_stratum(block: Dict[str, object]) -> Dict[str, object]:
    """Lowest coverage over metal groups and cohort-size groups."""
    candidates = []
    for key in ("by_group", "by_cohort_size"):
        for name, v in (block.get(key) or {}).items():
            if isinstance(v, dict) and v.get("coverage") is not None and v.get("n", 0) >= 100:
                candidates.append((key, name, float(v["coverage"]), int(v["n"])))
    if not candidates:
        return {"status": "NO_STRATA"}
    key, name, cov, n = min(candidates, key=lambda t: t[2])
    return {"stratum": f"{key}:{name}", "coverage": round(cov, 4), "n": n}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--oof", type=Path, default=PROJECT_ROOT / "results" / "msrs_generalization" / "oof.npz"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_calibration_shift.json",
    )
    ap.add_argument("--protocol", default="doi")
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args(argv)

    if not args.oof.exists():
        print(f"FAIL: missing {args.oof}")
        return 1

    data = np.load(args.oof, allow_pickle=False)
    y = data["y"].astype(float)
    metal = data["metal"].astype(str)
    cell = data["cell_line_key"].astype(str)
    frame = pd.DataFrame(
        {
            "ligand_canonical": data["ligand_canonical"].astype(str),
            "cell_line_key": cell,
            "doi": data["doi"].astype(str),
            "time_h": data["time_h"].astype(float),
            "metal": metal,
        }
    )
    ck = cohort_sizes(frame)

    block = run_protocol(
        y,
        data[f"{args.protocol}__{args.arm}"].astype(float),
        data[f"{args.protocol}__fold"].astype(int),
        metal,
        cell,
        ck,
        alpha=args.alpha,
        extra_variants=EXTRA,
    )

    summary: Dict[str, object] = {}
    for name, v in block["variants"].items():
        if v.get("status") == "INCOMPLETE":
            summary[name] = {"status": "INCOMPLETE"}
            continue
        w = worst_stratum(v)
        summary[name] = {
            "coverage": v.get("coverage"),
            "median_width": v.get("median_width"),
            "mean_width": v.get("mean_width"),
            "winkler": v.get("winkler"),
            "worst_stratum": w,
        }

    best = max(
        (n for n in EXTRA if summary.get(n, {}).get("coverage") is not None),
        key=lambda n: summary[n]["worst_stratum"].get("coverage", 0.0),
        default=None,
    )
    if best is None:
        print("FAIL: no extra variant produced an interval")
        return 1
    b = summary[best]
    worst = b["worst_stratum"].get("coverage", 0.0)
    width = b["median_width"] or 99.0
    width_ratio = width / REFERENCE["median_width"] - 1.0
    coverage_pass = worst >= COVERAGE_GATE
    width_pass = width <= WIDTH_GATE or (
        width_ratio < WIDTH_TOLERANCE and worst >= COVERAGE_STRONG
    )
    passed = bool(coverage_pass and width_pass)

    out = {
        "artifact": "msrs_calibration_shift",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L4 uncertainty / UP-6",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "oof": str(args.oof),
            "oof_sha256": _sha256(args.oof),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "protocol": args.protocol,
            "arm": args.arm,
            "alpha": args.alpha,
            "extra_variants": list(EXTRA),
            "reference": REFERENCE,
            "gate": {
                "worst_stratum_coverage": COVERAGE_GATE,
                "width": WIDTH_GATE,
                "width_tolerance": WIDTH_TOLERANCE,
                "coverage_strong": COVERAGE_STRONG,
                "frozen_at": "2026-10-21T15:10+08:00",
            },
            "why_cohort": (
                "the failing group is cohort size 2 (n=851, 0.8096), not a metal; "
                "metal-Mondrian only moves it to 0.8331"
            ),
        },
        "variants": summary,
        "best_extra_variant": best,
        "gate": {
            "coverage_pass": bool(coverage_pass),
            "width_pass": bool(width_pass),
            "worst_stratum": worst,
            "width": width,
            "width_change_vs_reference": round(width_ratio, 4),
        },
        "verdict": "PASS" if passed else "FAIL",
        "verdict_rule": (
            "worst stratum >= 0.87 AND (width <= 1.75 OR (width increase < 5% AND "
            "worst stratum >= 0.88))"
        ),
        "caveats": [
            "Base predictions are out-of-fold from the same models used to fit the calibration "
            "quantiles, so residuals are mildly optimistic and coverage is a lower bound.",
            "Width cannot be improved by calibration alone: a 90% interval of width W implies a "
            "residual SD of about W/3.29, and that SD is set by the model's R2.",
            "The cohort_metal PIT is computed against the global pool because that variant has no "
            "single calibration pool per test row; treat its histogram as indicative.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[UP-6] best={best} worst={worst:.4f} width={width:.4f} "
        f"({width_ratio:+.1%}) -> {out['verdict']}"
    )
    print(f"json: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
