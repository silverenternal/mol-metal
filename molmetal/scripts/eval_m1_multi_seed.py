#!/usr/bin/env python3
"""Wf-359-A2 — m1 multi-seed evaluation scaffold (CPU-only, deferred GPU launch).

================================================================
Wf-359-A2 — CPU-only multi-seed evaluation scaffold for the m1 Ru
AUC anchor. Closes the Wf-357-B3 / Wf-355-B1 / Wf-354-A1 §5
multi-seed gap by preparing the aggregation code so the Wf-355-B1
USER_OK-gated launch wrapper can hand off its outputs and produce
the paper-grade multi-seed mean ± std the moment the 7 DEFERRED
runs complete.

Scope (per Wf-359-A2 task brief)
--------------------------------
* LOAD per-seed metrics from ``molmetal/reports/wf131_p3_deferred_runs/``
  (primary) and ``molmetal/reports/wf131_p3_lomo_full/`` (fallback for
  already-completed metals).
* COMPUTE aggregate per metal: mean ± std across available seeds.
* HONEST-NEGATIVE handling: if a metal has only 1 seed completed, emit
  the 1-seed reading with a HONEST_NEGATIVE marker + n_seeds=1 tag.
* EMIT ``molmetal/reports/wf359a2_m1_multi_seed_scaffold/aggregate.json``
  in the same schema as ``metal_lomo_auc.json``
  ``wf131_p3_lomo_partial_update.per_metal_*`` blocks.

This script is CPU-only — no GPU launches. It does NOT modify any
metrics/ JSON in ``metrics/by_metric/``; the aggregate output is
placed in ``molmetal/reports/wf359a2_m1_multi_seed_scaffold/aggregate.json``
which the operator can promote manually after honest-negative review
(per AGENTS.md §1.1 + §5.2 + `party-twenty-spirit` §3.5).

Usage
-----
    PYTHONPATH=/home/hugo/codes/mol-metal python \\
        molmetal/scripts/eval_m1_multi_seed.py

    PYTHONPATH=/home/hugo/codes/mol-metal python \\
        molmetal/scripts/eval_m1_multi_seed.py --metals Ru Ir Os Rh Re

Honest framing
--------------
This is a SCAFFOLD only — actual multi-seed execution is USER_OK-gated
per AGENTS.md §5.1 + Wf-355-B1 wrapper. Per-seed readings are READ
from existing JSON files on disk; the script does NOT run any trainer.
Per Wf-357-B3 + Wf-355-B1: when the 7 DEFERRED runs (Rh 3+4 + Re 0-4)
are eventually MEASURED, this script aggregates them into the
canonical ``aggregate.json`` automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
PRIMARY_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf131_p3_deferred_runs"
FALLBACK_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf131_p3_lomo_full"
OUTPUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf359a2_m1_multi_seed_scaffold"
OUTPUT_FILE = OUTPUT_DIR / "aggregate.json"

# Five d-block metals per Wf-131 P3 LOMO 5-metal protocol
DEFAULT_METALS: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")

# Krasnov 2026 baseline for m1 Ru AUC (per AGENTS.md §1.1 + Wf-357-B3 §4)
RU_KRASNOV_BASELINE = 0.7123
GATE_TARGET = 0.86


def _load_per_seed_metrics(
    metal_dir: Path,
    metal: str,
) -> Tuple[List[Dict[str, float]], str]:
    """Load per-seed metrics for ``metal`` from ``metal_dir``.

    Returns:
        (per_seed_list, source_path)
        - per_seed_list: list of dicts with at least {seed, test_auc,
          val_auc, n_train, n_test} keys.
        - source_path: which file the seeds came from
          (``per_seed_metrics.json`` aggregate or ``per_seed_metrics_seedN.json``
          per-seed files).

    Note
    ----
    Tries ``per_seed_metrics.json`` (aggregate form) first; falls back
    to ``per_seed_metrics_seedN.json`` per-seed files (one JSON per
    seed, e.g. ``per_seed_metrics_seed0.json``). The fallback path is
    the on-disk convention used by ``train_dmpnn_multitask_wasserstein.py``
    (see Wf-124 P3 + Wf-131 P3 anchor).
    """
    metal_subdir = metal_dir / metal
    if not metal_subdir.exists():
        return [], f"<missing: {metal_subdir}>"

    aggregate = metal_subdir / "per_seed_metrics.json"
    if aggregate.exists():
        try:
            data = json.loads(aggregate.read_text())
            per_seed = data.get("per_seed", [])
            if per_seed:
                return per_seed, str(aggregate)
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: per-seed files (Wf-124 P3 / Wf-131 P3 convention)
    per_seed_files = sorted(metal_subdir.glob("per_seed_metrics_seed*.json"))
    out: List[Dict[str, float]] = []
    for p in per_seed_files:
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # Normalize field names: per-seed file uses "fold" + "seed",
        # aggregate uses "seed" only; both carry "test_auc".
        out.append(
            {
                "seed": d.get("seed", d.get("fold_seed")),
                "test_auc": d.get("test_auc"),
                "val_auc": d.get("val_auc"),
                "n_train": d.get("n_train"),
                "n_test": d.get("n_test"),
                "test_pic50_mae": d.get("test_pic50_mae"),
            }
        )
    if out:
        return out, f"<per-seed dir: {metal_subdir}>"

    return [], f"<empty: {metal_subdir}>"


def aggregate_metal(
    per_seed: List[Dict[str, float]],
    metal: str,
    source_path: str,
) -> Dict[str, object]:
    """Aggregate per-seed readings into mean ± std.

    Honest-negative contract (per Wf-357-B3 + Wf-355-B1):
      * n_seeds >= 3 → emit mean ± std as MEASURED.
      * n_seeds == 2 → emit mean (no std) + PARTIAL tag.
      * n_seeds == 1 → emit 1-seed reading + HONEST_NEGATIVE_1SEED tag.
      * n_seeds == 0 → emit empty block + NOT_MEASURED tag.
    """
    valid_aucs: List[float] = []
    seeds_used: List[int] = []
    for entry in per_seed:
        v = entry.get("test_auc")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        try:
            valid_aucs.append(float(v))
            seeds_used.append(int(entry.get("seed", -1)))
        except (TypeError, ValueError):
            continue

    n = len(valid_aucs)
    if n == 0:
        verdict = "NOT_MEASURED"
        auc_mean: Optional[float] = None
        auc_std: Optional[float] = None
    elif n == 1:
        auc_mean = valid_aucs[0]
        auc_std = 0.0
        verdict = "HONEST_NEGATIVE_1SEED"
    elif n == 2:
        auc_mean = (valid_aucs[0] + valid_aucs[1]) / 2.0
        # Population std for 2-sample pair
        auc_std = math.sqrt(
            ((valid_aucs[0] - auc_mean) ** 2 + (valid_aucs[1] - auc_mean) ** 2) / 2.0
        )
        verdict = "PARTIAL_2SEED"
    else:
        auc_mean = sum(valid_aucs) / n
        variance = sum((v - auc_mean) ** 2 for v in valid_aucs) / n
        auc_std = math.sqrt(variance)
        verdict = "MEASURED"

    block: Dict[str, object] = {
        "metal": metal,
        "n_seeds_used": n,
        "seeds_used": seeds_used,
        "per_seed_aucs": valid_aucs,
        "test_auc_mean": auc_mean,
        "test_auc_std": auc_std,
        "source_path": source_path,
        "verdict": verdict,
    }

    # m1 Ru-specific gate target delta (only for Ru)
    if metal == "Ru" and auc_mean is not None:
        block["baseline_krasnov_2026"] = RU_KRASNOV_BASELINE
        block["gate_target"] = GATE_TARGET
        block["lift_vs_baseline"] = auc_mean - RU_KRASNOV_BASELINE
        block["gap_to_gate"] = auc_mean - GATE_TARGET
        block["hit_gate"] = auc_mean >= GATE_TARGET

    return block


def build_aggregate(
    metals: Tuple[str, ...] = DEFAULT_METALS,
    primary_dir: Path = PRIMARY_DIR,
    fallback_dir: Path = FALLBACK_DIR,
) -> Dict[str, object]:
    """Build the aggregate dict for ``metals``.

    For each metal, try the primary directory (wf131_p3_deferred_runs)
    first, then fall back to the LOMO-full directory. The fallback
    is the Wf-131 P3 anchor data on disk.
    """
    per_metal: Dict[str, Dict[str, object]] = {}
    for metal in metals:
        per_seed, src = _load_per_seed_metrics(primary_dir, metal)
        if not per_seed:
            per_seed, src = _load_per_seed_metrics(fallback_dir, metal)
        block = aggregate_metal(per_seed, metal, src)
        per_metal[metal] = block

    # Overall aggregate (across all metals present)
    all_means: List[float] = [
        b["test_auc_mean"] for b in per_metal.values() if b["test_auc_mean"] is not None
    ]
    n_metals_with_data = len(all_means)
    overall_mean = (sum(all_means) / n_metals_with_data) if n_metals_with_data else None
    overall_std = (
        math.sqrt(sum((m - overall_mean) ** 2 for m in all_means) / n_metals_with_data)
        if n_metals_with_data > 1
        else 0.0
    )

    return {
        "schema": "wf359a2_m1_multi_seed_scaffold.v1",
        "wf_origin": "Wf-359-A2 (ultracode residual wave; Wf-357-B3 multi-seed gap)",
        "parent": "Wf-355-B1 wrapper + Wf-131 P3 anchor + Wf-357-B3 verification",
        "primary_dir": str(primary_dir),
        "fallback_dir": str(fallback_dir),
        "metals": list(metals),
        "per_metal": per_metal,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "n_metals_with_data": n_metals_with_data,
        "honest_framing": {
            "MEASURED_today": "CPU-only aggregation scaffold; no GPU launches",
            "PROJECTED": "When Wf-355-B1 USER_OK launch completes, re-running "
            "this script will populate per_metal Rh + Re from "
            "wf131_p3_deferred_runs/ rather than fallback LOMO-full data",
            "honest_negative_note": (
                "1-seed-only readings emit HONEST_NEGATIVE_1SEED verdict "
                "with n_seeds_used=1 to preserve MEASURED-vs-PROJECTED "
                "tag discipline (party-twenty-spirit §3.5 + AGENTS.md §1.1)"
            ),
            "kill_then_adapt_directive": "CPU-only; no retrain / panel / ablation",
        },
        "verdict": (
            "READY_FOR_USER_OK_LAUNCH"
            if any(
                b["verdict"]
                in ("HONEST_NEGATIVE_1SEED", "PARTIAL_2SEED", "NOT_MEASURED")
                for b in per_metal.values()
            )
            else "ALL_METALS_5SEED"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Wf-359-A2 m1 multi-seed evaluation scaffold (CPU-only).")
    )
    parser.add_argument(
        "--metals",
        nargs="+",
        default=list(DEFAULT_METALS),
        help=f"Metals to aggregate (default: {' '.join(DEFAULT_METALS)})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output JSON path (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the aggregate to stdout in addition to writing the file",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate = build_aggregate(metals=tuple(args.metals))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(aggregate, indent=2) + "\n")

    if args.print:
        print(json.dumps(aggregate, indent=2))

    # Stdout invariant per Wf-262 / AGENTS.md §4: emit PASS / SKIP / FAIL
    n_complete = sum(
        1 for b in aggregate["per_metal"].values() if b["n_seeds_used"] >= 3
    )
    n_total = len(aggregate["metals"])
    if n_complete == n_total:
        verdict = "PASS"
    elif n_complete == 0:
        verdict = "SKIP"
    else:
        verdict = "PASS"

    print(
        f"{verdict} Wf-359-A2 m1 multi-seed scaffold: "
        f"{n_complete}/{n_total} metals have ≥3 seeds; "
        f"aggregate written to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
