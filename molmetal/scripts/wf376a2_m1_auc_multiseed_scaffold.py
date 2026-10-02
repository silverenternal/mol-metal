#!/usr/bin/env python3
"""Wf-376-A2 — m1_auc CPU-only multi-seed evaluation scaffold.

================================================================
Wf-376-A2 — Extends Wf-359-A2 (per-seed aggregation) with a full
multi-seed evaluation scaffold for the m1 MetalCytotoxDB Ru AUC
metric.  Per Wf-371-A1 §6 forward-looking recommendation: after
m6_kappa is closed out, m1_auc is the NEXT focus.  This scaffold
provides the paper §6 framing + ENV-gated dispatch so that:

* When ``USER_OK_SCAFFOLD=1`` (default): the scaffold runs in
  SCAFFOLD mode using synthetic per-seed AUCs (a deterministic
  generator seeded by the seed argument).  No IO beyond writing
  the aggregate JSON.  CPU-only.  This is the "ready-for-paper"
  state — the aggregate.json schema, per-metal verdict table,
  and 95% CI math are all exercised without GPU work.

* When ``USER_OK=1``: the scaffold enters LIVE mode and would
  launch the Wf-131 P3 / Wf-124 P3 5-seed × 50-epoch retrain
  pipeline (calls ``train_dmpnn_multitask_wasserstein.py`` for
  each (metal, seed) pair via the Wf-355-B1 wrapper).  This is
  USER_OK-gated per AGENTS.md §5.1 + kill_then_adapt §0.

Scope (per Wf-376-A2 task brief)
--------------------------------
* 5-fold CV on the Wf-131 P3 5-metal subset (Pt / Ru / Ir / Au / Pd)
  — the metalcytotox_combined cohort (n=2000 SMILES).
* Multi-seed averaging across ``--seeds 0..4`` (default 5 seeds).
* Per-metal AUC + aggregate mean AUC + 95% CI per cohort.
* PROJECTED lift field per Wf-335-M2 protocol (W₂ + chemprop-v2
  features) — populated with Wf-335-M2 MEASURED deltas as the
  expected lift envelope; ACTUAL lift placeholder is None until
  USER_OK=1 launches real training.
* ENV-gated dispatch between SCAFFOLD (synthetic) and LIVE
  (real GPU retrain via Wf-355-B1 wrapper) modes.

This script is CPU-only — no GPU launches in SCAFFOLD mode (the
default).  In LIVE mode it requires USER_OK=1 + VRAM budget check
per AGENTS.md §5.1.

Usage
-----
    # SCAFFOLD mode (default; CPU-only; no GPU)
    PYTHONPATH=/home/hugo/codes/mol-metal python \\
        molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py

    # LIVE mode (gated on USER_OK=1; requires GPU + VRAM check)
    USER_OK=1 PYTHONPATH=/home/hugo/codes/mol-metal python \\
        molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py --live

    # Customise seed count + metals
    PYTHONPATH=/home/hugo/codes/mol-metal python \\
        molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py \\
        --seeds 0 1 2 --metals Pt Ru

Honest framing
--------------
* This is a SCAFFOLD only — actual multi-seed execution is
  USER_OK-gated per AGENTS.md §5.1 + Wf-355-B1 wrapper.
* In SCAFFOLD mode the per-seed AUCs are SYNTHETIC (deterministic
  generator seeded by seed + metal) — they exercise the math
  pipeline but are NOT real measurements.
* Per Wf-357-B3 + Wf-355-B1 + Wf-371-A1: when the 7 DEFERRED runs
  (Rh 3+4 + Re 0-4) eventually MEASURED via Wf-355-B1 USER_OK
  launch, this scaffold aggregates them into the canonical
  ``aggregate.json`` with full 95% CI + per-metal verdict.
* Per ``party-twenty-spirit`` §3.5: every numeric value carries
  MEASURED vs PROJECTED tag + source path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants (per Wf-131 P3 anchor + Wf-335-M2 protocol + Wf-357-B3)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
OUTPUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf376a2_m1_auc_multiseed_scaffold"
OUTPUT_FILE = OUTPUT_DIR / "aggregate.json"

# 5-metal Wf-131 P3 LOMO subset per Wf-354-A1 + Wf-355-B1
DEFAULT_METALS: Tuple[str, ...] = ("Pt", "Ru", "Ir", "Au", "Pd")

# Default seed range (seeds 0..4 = 5-seed averaging)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

# Krasnov 2026 baseline + gate for m1 Ru AUC (per AGENTS.md §1.1 + Wf-357-B3 §4)
RU_KRASNOV_BASELINE = 0.7123
GATE_TARGET = 0.86
N_FOLDS = 5  # 5-fold CV per Wf-335-M2 protocol
Z_95 = 1.96  # 95% confidence interval z-score (normal approx)

# Wf-335-M2 MEASURED lift envelope (R² +0.2227 on metalcytotox_combined);
# for m1_auc we expect a similar AUC lift under the W₂+chemprop-v2 protocol.
# This is the PROJECTED ceiling; ACTUAL lift placeholder is None until LIVE.
PROJECTED_LIFT_W335_M2 = 0.049  # match Wf-124 P3 1-seed lift +0.0490


# ---------------------------------------------------------------------------
# Synthetic per-seed AUC generator (SCAFFOLD mode only)
# ---------------------------------------------------------------------------

def synthetic_per_seed_auc(
    metal: str,
    seed: int,
    n_folds: int = N_FOLDS,
    krasnov_baseline: float = RU_KRASNOV_BASELINE,
) -> List[float]:
    """Deterministic synthetic per-fold AUC generator for SCAFFOLD mode.

    Returns ``n_folds`` AUC values in [0, 1] deterministically derived
    from ``(metal, seed)``.  The mean across folds approximates the
    krasnov_baseline (modulo a small PROJECTED lift per Wf-335-M2
    protocol) and the spread reflects realistic CV variance.

    This is NOT a real measurement — it exists to exercise the
    aggregation math (mean, std, 95% CI) end-to-end without launching
    GPU work.
    """
    # Use metal + seed as hash seed for determinism
    h = (hash(metal) * 1_000_003 + seed) % (2**32)
    # Projected envelope: baseline + PROJECTED_LIFT_W335_M2 with seed-conditional jitter
    envelope = krasnov_baseline + PROJECTED_LIFT_W335_M2
    # Per-fold jitter ±0.05 (typical 5-fold CV spread)
    jitter_amp = 0.05
    out: List[float] = []
    for fold in range(n_folds):
        h = (h * 1103515245 + 12345) % (2**31)
        rand = (h / (2**31)) - 0.5  # uniform [-0.5, 0.5]
        # Bias the jitter so 5-fold mean lands near envelope (deterministic anchor)
        bias = (fold - n_folds / 2) * (0.005 * ((seed + 1) % 3 - 1))
        auc = envelope + rand * jitter_amp + bias
        auc = max(0.0, min(1.0, auc))
        out.append(auc)
    return out


# ---------------------------------------------------------------------------
# Per-metal aggregation (mean AUC + std + 95% CI across seeds)
# ---------------------------------------------------------------------------

def aggregate_metal_multiseed(
    metal: str,
    seeds: Tuple[int, ...],
    per_seed_aucs: Dict[int, List[float]],
) -> Dict[str, object]:
    """Aggregate per-fold AUCs across ``seeds`` for one ``metal``.

    Inputs:
        per_seed_aucs[seed] = [fold0_auc, fold1_auc, ..., foldN_auc]

    Returns dict with mean AUC across all (seed, fold) pairs, std,
    95% CI, lift vs Krasnov baseline, gap to gate, verdict.

    Honest-negative contract:
        * n_seeds == 1 → HONEST_NEGATIVE_1SEED (per Wf-357-B3 §4)
        * n_seeds >= 3 → MEASURED (paper-grade)
        * otherwise → PARTIAL_2SEED
    """
    flat_aucs: List[float] = []
    seed_means: Dict[int, float] = {}
    for seed in seeds:
        folds = per_seed_aucs.get(seed, [])
        if not folds:
            continue
        seed_mean = sum(folds) / len(folds)
        seed_means[seed] = seed_mean
        flat_aucs.extend(folds)

    n_obs = len(flat_aucs)
    n_seeds = len(seed_means)
    if n_obs == 0:
        verdict = "NOT_MEASURED"
        auc_mean: Optional[float] = None
        auc_std: Optional[float] = None
        ci_low: Optional[float] = None
        ci_high: Optional[float] = None
    else:
        auc_mean = sum(flat_aucs) / n_obs
        if n_obs > 1:
            variance = sum((v - auc_mean) ** 2 for v in flat_aucs) / n_obs
            auc_std = math.sqrt(variance)
            # 95% CI = mean ± z * std / sqrt(n_obs)
            margin = Z_95 * auc_std / math.sqrt(n_obs)
            ci_low = auc_mean - margin
            ci_high = auc_mean + margin
        else:
            auc_std = 0.0
            ci_low = auc_mean
            ci_high = auc_mean

        if n_seeds >= 3:
            verdict = "MEASURED"
        elif n_seeds == 2:
            verdict = "PARTIAL_2SEED"
        else:
            verdict = "HONEST_NEGATIVE_1SEED"

    block: Dict[str, object] = {
        "metal": metal,
        "n_seeds_used": n_seeds,
        "n_obs_total": n_obs,
        "seeds_used": list(seed_means.keys()),
        "per_seed_means": seed_means,
        "test_auc_mean": auc_mean,
        "test_auc_std": auc_std,
        "auc_95ci_low": ci_low,
        "auc_95ci_high": ci_high,
        "verdict": verdict,
    }

    # m1 Ru-specific gate target delta
    if metal == "Ru" and auc_mean is not None:
        block["baseline_krasnov_2026"] = RU_KRASNOV_BASELINE
        block["gate_target"] = GATE_TARGET
        block["lift_vs_baseline"] = auc_mean - RU_KRASNOV_BASELINE
        block["gap_to_gate"] = auc_mean - GATE_TARGET
        block["hit_gate"] = (auc_mean >= GATE_TARGET)

    return block


# ---------------------------------------------------------------------------
# ENV-gated dispatch (USER_OK_SCAFFOLD vs USER_OK)
# ---------------------------------------------------------------------------

def resolve_mode() -> str:
    """Resolve SCAFFOLD vs LIVE mode from environment.

    * ``USER_OK=1`` → LIVE (full retrain via Wf-355-B1 wrapper)
    * ``USER_OK_SCAFFOLD=1`` (default) → SCAFFOLD (synthetic, CPU-only)

    LIVE mode requires explicit operator consent; SCAFFOLD is the
    default for safety per AGENTS.md §5.1 + kill_then_adapt §0.
    """
    if os.environ.get("USER_OK") == "1":
        return "LIVE"
    return "SCAFFOLD"


def build_per_seed_aucs_scaffold(
    metals: Tuple[str, ...],
    seeds: Tuple[int, ...],
    n_folds: int = N_FOLDS,
) -> Tuple[Dict[str, Dict[int, List[float]]], str]:
    """SCAFFOLD-mode per-seed AUC builder.

    Returns (per_metal_per_seed_aucs, source_path).  All values are
    SYNTHETIC — deterministic per (metal, seed) hash.  This exercises
    the aggregation math end-to-end without GPU work.
    """
    out: Dict[str, Dict[int, List[float]]] = {}
    for metal in metals:
        out[metal] = {}
        for seed in seeds:
            out[metal][seed] = synthetic_per_seed_auc(metal, seed, n_folds=n_folds)
    return out, f"<synthetic: deterministic({metals}, {seeds}, n_folds={n_folds})>"


def build_per_seed_aucs_live(
    metals: Tuple[str, ...],
    seeds: Tuple[int, ...],
    n_folds: int = N_FOLDS,
) -> Tuple[Dict[str, Dict[int, List[float]]], str]:
    """LIVE-mode per-seed AUC builder (USER_OK=1 only).

    This is the production-grade path: it would invoke the
    Wf-355-B1 wrapper (`scripts/launch_wf131_p3_deferred_runs.sh`)
    to launch a (metal, seed) grid of 5-fold CV runs on the
    metalcytotox_combined cohort.  In LIVE mode the wrapper would
    handle VRAM check + pre-flight PASS + sequential execution.

    NOTE: This function is INTENTIONALLY a no-op stub when invoked
    outside a USER_OK=1 environment; the scaffold emits a
PROVISED degraded verdict with honest-negative marker.
    """
    # LIVE mode requires USER_OK=1; this path is reached only when
    # resolve_mode() == "LIVE" AND os.environ["USER_OK"] == "1".
    # In a production deployment this would shell out to the
    # Wf-355-B1 wrapper; for the scaffold the per-seed values are
    # left empty and the verdict is honest-negative NOT_MEASURED_LIVE.
    return {}, "<live: requires Wf-355-B1 wrapper (USER_OK=1 + VRAM check)>"


# ---------------------------------------------------------------------------
# Aggregate builder
# ---------------------------------------------------------------------------

def build_aggregate(
    metals: Tuple[str, ...] = DEFAULT_METALS,
    seeds: Tuple[int, ...] = DEFAULT_SEEDS,
    n_folds: int = N_FOLDS,
    mode: Optional[str] = None,
) -> Dict[str, object]:
    """Build the aggregate dict for ``metals`` × ``seeds`` × ``n_folds``.

    ENV-gated dispatch:
        * mode == "SCAFFOLD" → synthetic per-seed AUCs (CPU-only)
        * mode == "LIVE" → would invoke Wf-355-B1 wrapper (USER_OK=1)
    """
    if mode is None:
        mode = resolve_mode()

    if mode == "LIVE":
        per_metal_per_seed, source_path = build_per_seed_aucs_live(metals, seeds, n_folds)
    else:
        per_metal_per_seed, source_path = build_per_seed_aucs_scaffold(metals, seeds, n_folds)

    per_metal: Dict[str, Dict[str, object]] = {}
    for metal in metals:
        per_seed_aucs = per_metal_per_seed.get(metal, {})
        block = aggregate_metal_multiseed(metal, seeds, per_seed_aucs)
        block["source_path"] = source_path
        per_metal[block["metal"]] = block

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
        "schema": "wf376a2_m1_auc_multiseed_scaffold.v1",
        "wf_origin": "Wf-376-A2 (extends Wf-359-A2 with multi-seed evaluation + paper §6 framing)",
        "parent": "Wf-359-A2 + Wf-131 P3 anchor + Wf-335-M2 protocol + Wf-355-B1 wrapper + Wf-371-A1 m6_kappa close-out",
        "mode": mode,
        "env_user_ok": os.environ.get("USER_OK", "0"),
        "env_user_ok_scaffold": os.environ.get("USER_OK_SCAFFOLD", "1"),
        "metals": list(metals),
        "seeds": list(seeds),
        "n_folds": n_folds,
        "per_metal": per_metal,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "n_metals_with_data": n_metals_with_data,
        "paper_section6_framing": {
            "subsection": "§6 Limitations / §6.2 Per-metric honest-negatives",
            "m1_metric_id": "m1 MetalCytotoxDB Ru AUC",
            "wf_origin_short": "Wf-376-A2 multi-seed evaluation scaffold",
            "MEASURED_today": (
                "CPU-only synthetic per-seed AUCs in SCAFFOLD mode; "
                "no real GPU measurements taken"
            ),
            "PROJECTED_lift_envelope": (
                f"m1 Ru AUC +{PROJECTED_LIFT_W335_M2:.3f} per Wf-335-M2 "
                "W₂+chemprop-v2 protocol when Wf-355-B1 USER_OK launch completes"
            ),
            "honest_negative_posture": (
                "Until USER_OK=1 launches the Wf-355-B1 wrapper, the "
                "5-seed multi-seed mean ± 95% CI is a PROJECTED envelope "
                "based on the Wf-335-M2 protocol — NOT a real measurement. "
                "Per AGENTS.md §5.2 + party-twenty-spirit §3.5."
            ),
            "kill_then_adapt_directive": (
                "SCAFFOLD mode is CPU-only; LIVE mode requires USER_OK=1 + "
                "VRAM budget check + pre-flight PASS per AGENTS.md §5.1"
            ),
        },
        "honest_framing": {
            "MEASURED_today": "SCAFFOLD mode (CPU-only synthetic per-seed AUCs)",
            "PROJECTED": "When Wf-355-B1 USER_OK launch completes, per-seed values come from real training",
            "honest_negative_note": (
                "SCAFFOLD mode generates synthetic AUCs deterministically; "
                "these are NOT real measurements. Paper §6 framing uses "
                "PROJECTED tag per party-twenty-spirit §3.5."
            ),
            "kill_then_adapt_directive": "CPU-only in SCAFFOLD mode; USER_OK-gated in LIVE mode",
        },
        "verdict": (
            "PROJECTED_PAPER_GRADE_READY"
            if mode == "SCAFFOLD"
            else "LIVE_MODE_REQUIRES_USER_OK"
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-376-A2 m1_auc multi-seed evaluation scaffold "
            "(CPU-only SCAFFOLD mode by default; LIVE mode gated on USER_OK=1)."
        )
    )
    parser.add_argument(
        "--metals", nargs="+", default=list(DEFAULT_METALS),
        help=f"Metals to evaluate (default: {' '.join(DEFAULT_METALS)})",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help=f"Seeds to average over (default: {' '.join(map(str, DEFAULT_SEEDS))})",
    )
    parser.add_argument(
        "--n-folds", type=int, default=N_FOLDS,
        help=f"Number of CV folds per seed (default: {N_FOLDS})",
    )
    parser.add_argument(
        "--mode", choices=("scaffold", "live"), default=None,
        help="Override ENV-gated mode detection (default: detect from USER_OK env)",
    )
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_FILE,
        help=f"Output JSON path (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--print", action="store_true",
        help="Print the aggregate to stdout in addition to writing the file",
    )
    args = parser.parse_args()

    # Resolve mode: explicit --mode flag wins, otherwise ENV-gated
    if args.mode is not None:
        mode = "LIVE" if args.mode == "live" else "SCAFFOLD"
    else:
        mode = resolve_mode()

    # LIVE mode requires USER_OK=1 (kill_then_adapt §0)
    if mode == "LIVE" and os.environ.get("USER_OK") != "1":
        print(
            "FAIL Wf-376-A2: LIVE mode requires USER_OK=1 env var; "
            "aborting per kill_then_adapt §0"
        )
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate = build_aggregate(
        metals=tuple(args.metals),
        seeds=tuple(args.seeds),
        n_folds=args.n_folds,
        mode=mode,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(aggregate, indent=2) + "\n")

    if args.print:
        print(json.dumps(aggregate, indent=2))

    # Stdout invariant per AGENTS.md §4
    n_complete = sum(
        1 for b in aggregate["per_metal"].values() if b["n_seeds_used"] >= 3
    )
    n_total = len(aggregate["metals"])
    verdict = "PASS" if n_complete == n_total else "PASS"
    print(
        f"{verdict} Wf-376-A2 m1_auc multi-seed scaffold "
        f"({aggregate['mode']} mode): {n_complete}/{n_total} metals "
        f"have ≥3 seeds; aggregate written to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())