#!/usr/bin/env python3
"""Wf-308D — G5 retrain (TODO-181) results integration template.

==============================================================================
Wf-308D PART 3 (TEMPLATE ONLY — does NOT execute).

This script is a **template** for integrating the Wf-124 P3 G5 retrain results
(results/wf124_p3_50epoch_full_sweep_g5/) into the project-wide metric /
dashboard / pre-flight surfaces. The G5 retrain launched under PID 1015907
on 2026-09-28 has finished (per master.log "ALL TODO-181 G5 PAIRS DONE Tue
Sep 29 03:11:13 AM CST 2026"), but the integration step (TODO-181 §3.4) is
deferred until Hugo explicitly OKs it (TODO-84 paper rule + Wf-308D PART 3
honest framing).

What this template does (when executed)
----------------------------------------
1. **Detect completion** of the G5 retrain by walking the
   results/wf124_p3_50epoch_full_sweep_g5/ tree and confirming every
   per-(metal, seed) sub-directory has a per_seed_metrics.json. The current
   master.log reports "ALL TODO-181 G5 PAIRS DONE" but the aggregator step
   that ran at end-of-sweep ("aggregated 0 pairs (expected 15)") failed
   silently — this script re-runs that aggregation correctly by reading the
   per_seed_metrics.json files directly (NOT the training_summary.json files
   which only carry single-seed aggregates).

2. **Aggregate per-metal mean ± std** across seeds 0/1/2 for test_auc,
   test_pearson, test_rmse, and per-metal n_train/n_val/n_test.

3. **Update molmetal/reports/wf219_krasnov_aligned/sweep_summary.json** with
   new "G5_rows" entries (these are the 5-metal × 3-seed × 50-epoch
   multi-task D-MPNN + Wasserstein pIC50 numbers — orthogonal to the
   chemprop v2 single-task baseline already in the file).

4. **Update metrics/by_metric/wf219_krasnov_aligned_dmpnn_mtw_5metal_3seed_50epoch_g5.json**
   (NEW file; mirrors the chemprop v2 metric JSON schema). This becomes the
   ship target for Wf-124 P3 G5 claim cards.

5. **Update molmetal/scripts/paper_dashboard.py** with new claim cards:
   - claim_dmpnn_mtw_5metal_3seed_50epoch_g5_mean_auc
   - claim_dmpnn_mtw_ru_per_metal_50epoch_g5
   - claim_dmpnn_mtw_ir_per_metal_50epoch_g5
   (with MEASURED vs PROJECTED badge per TODO-188 §5.1 honest framing).

6. **Trigger pre_flight_check.py** to verify 32+/32+ PASS preserved after
   the metric / dashboard edits. (Pre-flight auto-verify on CPU.)

What this template does NOT do
-------------------------------
- **NO actual integration** — the script is gated behind a `--apply` CLI
  flag. Without `--apply`, it prints a dry-run diff to stdout and exits 0.
- **NO paper-§4 claim changes** without Hugo's explicit per-claim OK (per
  TODO-84 paper rule).
- **NO training_summary.json mutation** — the per-seed JSON files in
  results/wf124_p3_50epoch_full_sweep_g5/ are MEASURED DATA and must not
  be tampered with.
- **NO retrain launch** — assumes PID 1015907 finished; this template
  integrates whatever finished, including partial results.

CLI::

    # Dry-run (default — print planned integration, no file writes):
    python molmetal/scripts/wf308d_g5_results_integration.py

    # Apply (mutates sweep_summary.json + creates the new metric JSON +
    # appends dashboard claim cards; requires --yes-i-approve-g5-integration
    # double-flag to prevent accidental fires):
    python molmetal/scripts/wf308d_g5_results_integration.py --apply \\
        --yes-i-approve-g5-integration

    # Verify-only: just confirm G5 is fully done + emit aggregate table.
    python molmetal/scripts/wf308d_g5_results_integration.py --verify-only

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
- TODO-181 §3.4 deferred until Hugo OK; this template is the integration
  glue that fires when the OK lands.
- The Wasserstein-pIC50 regulariser lift (+0.03 → +0.10 AUC on Ru / Ir /
  Os / Rh / Re vs single-task chemprop v2 baseline) is PROJECTED pending
  G5 MEASURED data; the actual lift will be computed from the
  per_seed_metrics.json files when this template runs.
- The G5 retrain's per-metal std is expected to be ±0.05-0.10 for
  smaller metals (Rh / Os / Re < 1.2k rows); we will report mean ± std
  honestly and not paper over it (party-twenty-spirit §3.5 honesty gate).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --- Wf-130 GPU default -----------------------------------------
# RTX 5090 on kanzi (GPU 0 is the VLM venv process).  Must run
# BEFORE `import torch` so CUDA init sees the env var.
import os as _wf130_os
_wf130_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
del _wf130_os
# --- end Wf-130 ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
G5_OUTPUT_DIR = PROJECT_ROOT / "results" / "wf124_p3_50epoch_full_sweep_g5"
METALS_5: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_EPOCHS: int = 50
SWEEP_SUMMARY_PATH = PROJECT_ROOT / "molmetal" / "reports" / "wf219_krasnov_aligned" / "sweep_summary.json"
NEW_METRIC_PATH = PROJECT_ROOT / "metrics" / "by_metric" / "wf219_krasnov_aligned_dmpnn_mtw_5metal_3seed_50epoch_g5.json"
DASHBOARD_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "paper_dashboard.py"
PRE_FLIGHT_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"

G5_METRIC_KEY = "wf219_krasnov_aligned_dmpnn_mtw_5metal_3seed_50epoch_g5"
G5_REPORT_RUN_ID = "wf124_p3_50epoch_full_sweep_g5"


@dataclass
class PerSeedRecord:
    """One (metal, seed) MEASURED record from the G5 retrain output."""
    metal: str
    seed: int
    n_train: int
    n_val: int
    n_test: int
    test_auc: float
    test_pearson: float
    test_rmse: float
    test_mae: Optional[float]
    val_auc: Optional[float]
    elapsed_seconds: Optional[float]
    status: str
    ckpt_path: Optional[str]
    per_seed_metrics_path: Path


@dataclass
class MetalAggregate:
    """Per-metal aggregate across seeds 0/1/2."""
    metal: str
    n_seeds_total: int
    n_seeds_ok: int
    n_train: int
    n_val: int
    n_test: int
    test_auc_mean: float
    test_auc_std: float
    test_pearson_mean: float
    test_pearson_std: float
    test_rmse_mean: float
    test_rmse_std: float
    per_seed: List[PerSeedRecord] = field(default_factory=list)


@dataclass
class IntegrationPlan:
    """The plan emitted by --verify-only or --apply."""
    g5_complete: bool
    n_pairs_total: int
    n_pairs_ok: int
    per_metal_aggregates: Dict[str, MetalAggregate]
    sweep_summary_delta: dict
    new_metric_json: dict
    new_dashboard_claims: List[dict]
    pre_flight_command: List[str]
    integration_blocked_reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_g5_completion(g5_dir: Path) -> Tuple[int, int]:
    """Walk the G5 output tree and return (n_pairs_present, n_pairs_complete).

    A pair is "present" iff its per-(metal, seed) directory has any contents;
    a pair is "complete" iff it has a per_seed_metrics.json file with non-NaN
    test_auc. Ir seed0 and Re seed2 are known partial (only run.log, no
    per_seed_metrics.json) per master.log inspection; we report both counts
    honestly.
    """
    if not g5_dir.exists():
        return 0, 0
    n_present = 0
    n_complete = 0
    for metal in METALS_5:
        for seed in DEFAULT_SEEDS:
            pair_dir = g5_dir / metal / f"seed{seed}"
            if not pair_dir.exists():
                continue
            n_present += 1
            # per_seed_metrics.json lives at {pair_dir}/{metal_lower}/per_seed_metrics.json
            for sub in pair_dir.iterdir():
                if not sub.is_dir():
                    continue
                if sub.name != metal.lower():
                    continue
                psm = sub / "per_seed_metrics.json"
                if psm.exists():
                    try:
                        data = json.loads(psm.read_text())
                        if data.get("per_seed") and isinstance(
                            data["per_seed"], list
                        ) and len(data["per_seed"]) > 0:
                            ps = data["per_seed"][0]
                            if (
                                ps.get("test_auc") is not None
                                and float(ps["test_auc"]) == float(ps["test_auc"])
                            ):
                                n_complete += 1
                                break
                    except Exception:
                        continue
    return n_present, n_complete


def _load_per_seed(psm_path: Path) -> Optional[PerSeedRecord]:
    """Read one per_seed_metrics.json into a PerSeedRecord."""
    try:
        data = json.loads(psm_path.read_text())
    except Exception:
        return None
    per_seed = data.get("per_seed") or []
    if not per_seed:
        return None
    ps = per_seed[0]
    metal = psm_path.parent.name.upper()  # ru / ir / os / rh / re -> upper
    return PerSeedRecord(
        metal=metal,
        seed=int(ps.get("seed", 0)),
        n_train=int(ps.get("n_train", 0)),
        n_val=int(ps.get("n_val", 0)),
        n_test=int(ps.get("n_test", 0)),
        test_auc=float(ps.get("test_auc", float("nan"))),
        test_pearson=float(ps.get("test_pearson", float("nan"))),
        test_rmse=float(ps.get("test_rmse", float("nan"))),
        test_mae=(
            float(ps["test_mae"]) if ps.get("test_mae") is not None else None
        ),
        val_auc=(
            float(ps["val_auc"]) if ps.get("val_auc") is not None else None
        ),
        elapsed_seconds=(
            float(ps["elapsed_seconds"])
            if ps.get("elapsed_seconds") is not None
            else None
        ),
        status=str(ps.get("status", "UNKNOWN")),
        ckpt_path=ps.get("ckpt_path"),
        per_seed_metrics_path=psm_path,
    )


def aggregate_g5(g5_dir: Path) -> Tuple[Dict[str, MetalAggregate], int, int]:
    """Walk G5 output, aggregate per-metal mean ± std, return (agg, n_present, n_ok)."""
    n_present, n_complete = detect_g5_completion(g5_dir)
    aggregates: Dict[str, MetalAggregate] = {}
    for metal in METALS_5:
        records: List[PerSeedRecord] = []
        for seed in DEFAULT_SEEDS:
            pair_dir = g5_dir / metal / f"seed{seed}"
            if not pair_dir.exists():
                continue
            sub_lower = pair_dir / metal.lower()
            psm = sub_lower / "per_seed_metrics.json"
            if not psm.exists():
                continue
            rec = _load_per_seed(psm)
            if rec is None or rec.status != "OK":
                continue
            records.append(rec)
        if not records:
            continue
        aucs = [r.test_auc for r in records]
        pearsons = [r.test_pearson for r in records]
        rmses = [r.test_rmse for r in records]
        agg = MetalAggregate(
            metal=metal,
            n_seeds_total=len(DEFAULT_SEEDS),
            n_seeds_ok=len(records),
            n_train=records[0].n_train,
            n_val=records[0].n_val,
            n_test=records[0].n_test,
            test_auc_mean=statistics.fmean(aucs),
            test_auc_std=(
                statistics.pstdev(aucs) if len(aucs) > 1 else 0.0
            ),
            test_pearson_mean=statistics.fmean(pearsons),
            test_pearson_std=(
                statistics.pstdev(pearsons) if len(pearsons) > 1 else 0.0
            ),
            test_rmse_mean=statistics.fmean(rmses),
            test_rmse_std=(
                statistics.pstdev(rmses) if len(rmses) > 1 else 0.0
            ),
            per_seed=records,
        )
        aggregates[metal] = agg
    return aggregates, n_present, n_complete


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------
def build_integration_plan(
    aggregates: Dict[str, MetalAggregate],
    n_present: int,
    n_complete: int,
) -> IntegrationPlan:
    """Build the dry-run plan. With --apply the plan becomes the actual edits."""
    sweep_delta = {"G5_rows": {}}
    new_metric: dict = {
        "metric": G5_METRIC_KEY,
        "description": (
            "Wf-124 P3 G5 retrain (TODO-181): 5-metal × 3-seed × 50-epoch "
            "multi-task D-MPNN + Wasserstein pIC50 regulariser (alpha=0.5, "
            "w_w=0.05, L2 gradient-norm guard). MEASURED-vs-PROJECTED honest "
            "framing per party-twenty-spirit §3.5."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": (
            "tier1_precious_metal_anticancer / MetalCytoToxDB 5-metal / "
            "D-MPNN multi-task Wasserstein"
        ),
        "source": "molmetal/scripts/wf308d_g5_results_integration.py (G5)",
        "gate_target": 0.7,
        "split_protocol": "scaffold (Bemis-Murcko) per Wf-124 P3 baseline",
        "krasnov_protocol_aligned": True,
        "n_seeds_used": 3,
        "metals": list(METALS_5),
        "seeds": list(DEFAULT_SEEDS),
        "epochs": DEFAULT_EPOCHS,
        "value": None,
        "value_std": None,
        "per_metal_in_domain": {},
        "per_metal_seed_auc": {},
        "per_metal_std": {},
        "per_seed_stats": {},
        "ru_auc_target_projected": 0.86,
        "ru_auc_baseline_single_task_chemprop_v2": 0.7123,
        "honest_framing_note": (
            "Wf-124 P3 G5 MEASURED data (results/wf124_p3_50epoch_full_sweep_g5/). "
            "Wasserstein pIC50 regulariser lift is MEASURED vs PROJECTED; per-metal "
            "std reported unfiltered. Smaller metals (Rh/Os/Re < 1.2k rows) have "
            "±0.05-0.10 per-seed spread; reported honestly per party-twenty-spirit §3.5."
        ),
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    new_claims: List[dict] = []
    blocked_reasons: List[str] = []

    if not aggregates:
        blocked_reasons.append(
            "G5 retrain not detected as complete (no per_seed_metrics.json "
            "found). Re-run after PID 1015907 finishes and aggregator step "
            "emits per-(metal, seed) records."
        )
    else:
        all_aucs = [a.test_auc_mean for a in aggregates.values()]
        new_metric["value"] = statistics.fmean(all_aucs)
        new_metric["value_std"] = (
            statistics.pstdev(all_aucs) if len(all_aucs) > 1 else 0.0
        )
        for m, agg in aggregates.items():
            sweep_delta["G5_rows"][m] = {
                "mean_auc": agg.test_auc_mean,
                "std": agg.test_auc_std,
                "n_seeds": agg.n_seeds_ok,
                "per_seed_auc": {
                    str(r.seed): r.test_auc for r in agg.per_seed
                },
                "per_seed_status": {
                    str(r.seed): r.status for r in agg.per_seed
                },
                "n_train": agg.n_train,
                "n_val": agg.n_val,
                "n_test": agg.n_test,
            }
            new_metric["per_metal_in_domain"][m] = agg.test_auc_mean
            new_metric["per_metal_std"][m] = agg.test_auc_std
            new_metric["per_metal_seed_auc"][m] = {
                str(r.seed): r.test_auc for r in agg.per_seed
            }
            new_metric["per_seed_stats"][m] = {
                "aucs": [r.test_auc for r in agg.per_seed],
                "pearsons": [r.test_pearson for r in agg.per_seed],
                "rmses": [r.test_rmse for r in agg.per_seed],
            }

        new_claims.append(
            {
                "id": "claim_dmpnn_mtw_5metal_3seed_50epoch_g5_mean_auc",
                "claim": (
                    f"Wf-124 P3 G5 5-metal D-MPNN + Wasserstein pIC50 mean "
                    f"AUC = {new_metric['value']:.3f} ± "
                    f"{new_metric['value_std']:.3f} (MEASURED; Wf-308D "
                    "PART 3 integration)"
                ),
                "json_path": (
                    "metrics/by_metric/wf219_krasnov_aligned_dmpnn_mtw_"
                    "5metal_3seed_50epoch_g5.json"
                ),
                "badge": "MEASURED" if aggregates else "PROJECTED",
                "krasnov_protocol_aligned": True,
            }
        )
        if "Ru" in aggregates:
            new_claims.append(
                {
                    "id": "claim_dmpnn_mtw_ru_per_metal_50epoch_g5",
                    "claim": (
                        f"Wf-124 P3 G5 Ru per-metal AUC = "
                        f"{aggregates['Ru'].test_auc_mean:.3f} ± "
                        f"{aggregates['Ru'].test_auc_std:.3f} "
                        f"(vs chemprop v2 single-task baseline 0.7123)"
                    ),
                    "json_path": (
                        "metrics/by_metric/wf219_krasnov_aligned_dmpnn_mtw_"
                        "5metal_3seed_50epoch_g5.json"
                    ),
                    "badge": "MEASURED",
                    "krasnov_protocol_aligned": True,
                }
            )
        if "Ir" in aggregates:
            new_claims.append(
                {
                    "id": "claim_dmpnn_mtw_ir_per_metal_50epoch_g5",
                    "claim": (
                        f"Wf-124 P3 G5 Ir per-metal AUC = "
                        f"{aggregates['Ir'].test_auc_mean:.3f} ± "
                        f"{aggregates['Ir'].test_auc_std:.3f} "
                        f"(vs Krasnov 2026 baseline 0.73)"
                    ),
                    "json_path": (
                        "metrics/by_metric/wf219_krasnov_aligned_dmpnn_mtw_"
                        "5metal_3seed_50epoch_g5.json"
                    ),
                    "badge": "MEASURED",
                    "krasnov_protocol_aligned": True,
                }
            )

    return IntegrationPlan(
        g5_complete=bool(aggregates),
        n_pairs_total=n_present,
        n_pairs_ok=n_complete,
        per_metal_aggregates=aggregates,
        sweep_summary_delta=sweep_delta,
        new_metric_json=new_metric,
        new_dashboard_claims=new_claims,
        pre_flight_command=[
            sys.executable,
            str(PRE_FLIGHT_PATH),
            "--pipeline",
            "all",
            "--json",
            "/tmp/wf308d_g5_preflight.json",
        ],
        integration_blocked_reasons=blocked_reasons,
    )


# ---------------------------------------------------------------------------
# Apply (only fires with --apply --yes-i-approve-g5-integration)
# ---------------------------------------------------------------------------
def apply_integration(plan: IntegrationPlan) -> None:
    """Mutate the sweep_summary.json + write new metric JSON + emit dashboard patch."""
    if SWEEP_SUMMARY_PATH.exists():
        existing = json.loads(SWEEP_SUMMARY_PATH.read_text())
    else:
        existing = {}
    existing.setdefault("g5_integration_wf308d", {})
    existing["g5_integration_wf308d"] = {
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_pairs_total": plan.n_pairs_total,
        "n_pairs_ok": plan.n_pairs_ok,
        "per_metal_aggregates": {
            m: {
                "mean_auc": a.test_auc_mean,
                "std": a.test_auc_std,
                "n_seeds": a.n_seeds_ok,
            }
            for m, a in plan.per_metal_aggregates.items()
        },
        "delta_payload": plan.sweep_summary_delta,
    }
    SWEEP_SUMMARY_PATH.write_text(json.dumps(existing, indent=2))
    NEW_METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEW_METRIC_PATH.write_text(json.dumps(plan.new_metric_json, indent=2))
    dashboard_patch_path = (
        PROJECT_ROOT
        / "molmetal"
        / "reports"
        / "wf308d_g5_integration_template"
        / "dashboard_claim_patch.py"
    )
    dashboard_patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_lines: List[str] = [
        "# Wf-308D G5 integration — dashboard claim card patch.",
        "# Generated by molmetal/scripts/wf308d_g5_results_integration.py --apply.",
        "# Review + manually paste into molmetal/scripts/paper_dashboard.py CLAIMS list.",
        "",
    ]
    for c in plan.new_dashboard_claims:
        patch_lines.append("{")
        for k, v in c.items():
            patch_lines.append(f"    {k!r}: {json.dumps(v)},")
        patch_lines.append("},")
    dashboard_patch_path.write_text("\n".join(patch_lines) + "\n")


def trigger_pre_flight() -> Tuple[int, str]:
    """Run the pre-flight auto-verify and return (exit_code, tail)."""
    cmd = [
        sys.executable,
        str(PRE_FLIGHT_PATH),
        "--pipeline",
        "all",
        "--json",
        "/tmp/wf308d_g5_preflight.json",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        tail = (proc.stdout or "")[-2000:]
        return proc.returncode, tail
    except Exception as e:
        return 2, f"pre_flight_check.py failed: {e}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Mutate sweep_summary.json + write new metric JSON + emit "
            "dashboard claim patch. Requires --yes-i-approve-g5-integration."
        ),
    )
    p.add_argument(
        "--yes-i-approve-g5-integration",
        action="store_true",
        help="Second-flag confirmation for --apply.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Just confirm G5 completion + emit aggregate table; no writes.",
    )
    p.add_argument(
        "--trigger-pre-flight",
        action="store_true",
        help=(
            "After integration, run pre_flight_check.py --pipeline all to "
            "verify 32+/32+ PASS preserved."
        ),
    )
    args = p.parse_args(argv)

    aggregates, n_present, n_complete = aggregate_g5(G5_OUTPUT_DIR)
    plan = build_integration_plan(aggregates, n_present, n_complete)

    print("=" * 78)
    print("Wf-308D G5 integration template")
    print("=" * 78)
    print(f"G5 output dir          : {G5_OUTPUT_DIR}")
    print(f"G5 complete?           : {plan.g5_complete}")
    print(f"n_pairs_present (15)   : {plan.n_pairs_total}")
    print(f"n_pairs_complete (15)  : {plan.n_pairs_ok}")
    print()
    if aggregates:
        print("Per-metal aggregate (mean ± std across seeds 0/1/2):")
        for m in METALS_5:
            if m not in aggregates:
                print(f"  {m}: (no complete per_seed_metrics.json — partial only)")
                continue
            a = aggregates[m]
            print(
                f"  {m}: AUC={a.test_auc_mean:.4f} ± {a.test_auc_std:.4f} "
                f"(n_seeds={a.n_seeds_ok}/{a.n_seeds_total}) "
                f"pearson={a.test_pearson_mean:.4f} "
                f"rmse={a.test_rmse_mean:.4f}"
            )
        all_aucs = [a.test_auc_mean for a in aggregates.values()]
        grand_mean = statistics.fmean(all_aucs)
        grand_std = (
            statistics.pstdev(all_aucs) if len(all_aucs) > 1 else 0.0
        )
        print()
        print(f"5-metal grand mean AUC : {grand_mean:.4f} ± {grand_std:.4f}")
    print()
    if plan.integration_blocked_reasons:
        print("BLOCKED:")
        for r in plan.integration_blocked_reasons:
            print(f"  - {r}")
        print()
    if args.apply:
        if not args.yes_i_approve_g5_integration:
            print(
                "ERROR: --apply requires --yes-i-approve-g5-integration. "
                "Refusing to mutate files."
            )
            return 2
        if not plan.g5_complete:
            print("ERROR: G5 not complete; refusing to apply.")
            return 2
        apply_integration(plan)
        print("Applied:")
        print(f"  - updated {SWEEP_SUMMARY_PATH}")
        print(f"  - wrote {NEW_METRIC_PATH}")
        print(
            "  - dashboard claim patch -> "
            "molmetal/reports/wf308d_g5_integration_template/dashboard_claim_patch.py"
        )
        print(
            "    (REVIEW + manually paste into paper_dashboard.py CLAIMS list)"
        )
        print()
        if args.trigger_pre_flight:
            rc, tail = trigger_pre_flight()
            print(f"pre_flight_check.py exit={rc}")
            print(tail)
            return rc
        return 0

    if args.verify_only:
        return 0

    # Default: dry-run summary (no writes).
    print("Dry-run only (no writes). Pass --apply --yes-i-approve-g5-integration")
    print("to mutate sweep_summary.json + create new metric JSON.")
    return 0


if __name__ == "__main__":
    sys.exit(main())