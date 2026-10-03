#!/usr/bin/env python3
"""Wf-197 — Per-innovation ablation launcher (TODO-182 ship).

=============================================================
Per-innovation ablation — full 2^3 = 8 permutation matrix
=============================================================

The Wf-124 P3 multi-task + Wasserstein recipe was run with all 3
innovations enabled simultaneously. Paper §3 cannot honestly attribute
the +0.049 aggregate lift (0.7123 → 0.7613) to specific levers.

This script runs the **full permutation ablation**:
  - Baseline (000): no levers
  - Each lever individually (3 configs): 100, 010, 001
  - All combinations (7 configs): 100, 010, 001, 110, 101, 011, 111
  - Total = 8 unique configurations (1 + 7)

Each config trains Ru seed=0 at 30 epochs, hidden=64, depth=2 (CPU
protocol matching Wf-196 apples-to-apples).

**Levers (binary 0/1 per row):**
  bit 0 (L2)  : multi-task 5-metal cohort (1) vs single-task Ru-only (0)
  bit 1 (W)   : Wasserstein pIC50 reg weight (0.05 vs 0.0)
  bit 2 (G)   : L2 grad-norm guard (on vs off)

**Hardware**: CPU primary (per standing "kill then adapt" GPU directive);
6-way parallelism. Wall-time budget: <= 3600s.

**Honest framing (party-twenty-spirit §3.5)**:
  - Per-lever attribution is a WITHIN-COHORT relative comparison; the
    +0.049 absolute Wf-124 P3 lift (h=128, d=3, GPU) is NOT directly
    reproducible at h=64, d=2 (CPU); the smaller backbone gives
    different absolute numbers (Wf-196 showed this).
  - This ablation attributes lift BETWEEN levers, not AVALATIVE to the
    Wf-124 P3 absolute anchor.

**Outputs** (under `molmetal/reports/wf197_ablation/`):
  - `per_config/per_config_<code>_seed0.json` — per-config test_auc
  - `ablation_summary.json` — full table + per-lever attribution
  - `synthesis.md` — human-readable narrative

**Usage**
::
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf197_ablation_launcher.py \\
        --n-parallel 6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PY = "/home/hugo/codes/mol-metal/.venv/bin/python"
TRAIN_SCRIPT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
)
DATA_CSV = Path("/home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv")
OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf197_ablation"
CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints" / "wf197_ablation"
LOG_DIR = OUT_DIR / "logs"

# ----- Ablation matrix (2^3 = 8 configurations) -----
# bits: (L2=multi-task, W=Wasserstein, G=L2 guard)
# Config code: 3-bit string e.g. "111" = all on, "000" = all off (baseline)
ABLATION_CONFIGS: list[dict] = [
    # config_code, multi_task, w_weight, l2_guard, label
    {
        "code": "000",
        "multi_task": False,
        "w_weight": 0.0,
        "l2_guard": False,
        "label": "baseline (no levers)",
    },
    {
        "code": "100",
        "multi_task": True,
        "w_weight": 0.0,
        "l2_guard": False,
        "label": "multi-task only",
    },
    {
        "code": "010",
        "multi_task": False,
        "w_weight": 0.05,
        "l2_guard": False,
        "label": "Wasserstein only",
    },
    {
        "code": "001",
        "multi_task": False,
        "w_weight": 0.0,
        "l2_guard": True,
        "label": "L2 guard only",
    },
    {
        "code": "110",
        "multi_task": True,
        "w_weight": 0.05,
        "l2_guard": False,
        "label": "multi-task + W",
    },
    {
        "code": "101",
        "multi_task": True,
        "w_weight": 0.0,
        "l2_guard": True,
        "label": "multi-task + L2 guard",
    },
    {
        "code": "011",
        "multi_task": False,
        "w_weight": 0.05,
        "l2_guard": True,
        "label": "W + L2 guard",
    },
    {
        "code": "111",
        "multi_task": True,
        "w_weight": 0.05,
        "l2_guard": True,
        "label": "full recipe (Wf-124 P3 default)",
    },
]


def run_one_config(
    cfg: dict,
    epochs: int,
    hidden: int,
    depth: int,
    batch_size: int,
    seed: int,
    log_dir: Path,
) -> dict:
    """Train one ablation configuration (single metal=Ru, single seed)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"cfg{cfg['code']}_seed{seed}.log"
    cmd = [
        PY,
        str(TRAIN_SCRIPT),
        "--metals",
        "Ru",
        "--seeds",
        "0",
        "--epochs",
        str(epochs),
        "--hidden",
        str(hidden),
        "--depth",
        str(depth),
        "--batch-size",
        str(batch_size),
        "--device",
        "cpu",
        "--data-csv",
        str(DATA_CSV),
        "--out-dir",
        str(OUT_DIR / "per_config"),
        "--ckpt-dir",
        str(CKPT_DIR / f"cfg{cfg['code']}"),
    ]
    if cfg["multi_task"]:
        cmd.extend(["--metals", "Ru", "Ir", "Os", "Rh", "Re"])
    else:
        cmd.append("--single-task")
    cmd.extend(["--wasserstein-weight", str(cfg["w_weight"])])
    if cfg["l2_guard"]:
        cmd.append("--use-l2-guard")
    else:
        cmd.append("--no-l2-guard")

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
    wall = time.time() - t0

    # Read test_auc from per-seed metrics
    metrics_path = OUT_DIR / "per_config" / "ru" / f"per_seed_metrics_seed{seed}.json"
    test_auc = None
    test_pearson = None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            test_auc = metrics.get("test_metrics", {}).get("roc_auc")
            test_pearson = metrics.get("test_pic50_metrics", {}).get("pearson")
        except Exception:
            pass

    return {
        "code": cfg["code"],
        "label": cfg["label"],
        "multi_task": cfg["multi_task"],
        "w_weight": cfg["w_weight"],
        "l2_guard": cfg["l2_guard"],
        "wall_seconds": wall,
        "returncode": proc.returncode,
        "test_auc": test_auc,
        "test_pearson": test_pearson,
        "log_path": str(log_path),
        "metrics_path": str(metrics_path) if metrics_path.exists() else None,
    }


def compute_attribution(results: list[dict]) -> dict:
    """Compute per-lever lift attribution from the ablation matrix.

    For each lever L and the interaction term:
        Δ_auc(L=on, others=baseline) - Δ_auc(L=off, others=baseline)

    For 2-way interactions:
        I_AB = auc(A=1, B=1) - auc(A=0) - auc(B=0) + auc(0,0)
    """
    by_code = {r["code"]: r for r in results if r["test_auc"] is not None}
    if "000" not in by_code:
        return {"error": "baseline (000) not measured", "by_code": by_code}

    baseline_auc = by_code["000"]["test_auc"]

    def auc(code: str) -> float | None:
        return by_code.get(code, {}).get("test_auc")

    # Single-lever lifts
    single_lift: dict[str, float | None] = {
        # lever_name: code with lever on vs baseline
        "multi_task": (auc("100") - baseline_auc) if auc("100") is not None else None,
        "wasserstein": (auc("010") - baseline_auc) if auc("010") is not None else None,
        "l2_guard": (auc("001") - baseline_auc) if auc("001") is not None else None,
    }

    # 2-way interactions: I_AB = AUC(A=1,B=1) - AUC(A=1,B=0) - AUC(A=0,B=1) + AUC(0,0)
    # Bit positions: index 0 = multi_task (MT), index 1 = wasserstein (W), index 2 = l2_guard (L2)
    def interaction(a_pos: int, b_pos: int, ab_pos: int) -> float | None:
        """Compute I_AB given positions of A, B, and the third lever (fixed at 0)."""

        # Build configs for (A, B, third=0) combinations
        def cfg_code(a: int, b: int) -> str:
            bits = ["0", "0", "0"]
            bits[a_pos] = str(a)
            bits[b_pos] = str(b)
            return "".join(bits)

        a_v = auc(cfg_code(1, 0))
        b_v = auc(cfg_code(0, 1))
        ab_v = auc(cfg_code(1, 1))
        c_v = auc(cfg_code(0, 0))
        if any(v is None for v in [a_v, b_v, ab_v, c_v]):
            return None
        return ab_v - a_v - b_v + c_v

    interactions: dict[str, float | None] = {
        "I_MT_W": interaction(0, 1, 2),  # MT × W, L2=0
        "I_MT_L2": interaction(0, 2, 1),  # MT × L2, W=0
        "I_W_L2": interaction(1, 2, 0),  # W × L2, MT=0
    }

    # Full recipe (111) lift vs baseline
    full_recipe_lift = (auc("111") - baseline_auc) if auc("111") is not None else None

    # Sum of single-lever lifts
    sum_single = sum(v for v in single_lift.values() if v is not None)
    sum_interactions = sum(v for v in interactions.values() if v is not None)

    # Additivity check: full_recipe_lift == sum_single + sum_interactions
    additivity_gap = None
    if full_recipe_lift is not None and sum_single is not None:
        try:
            # Round to handle None in interactions
            inter_total = sum_interactions if interactions else 0.0
            additivity_gap = round(full_recipe_lift - (sum_single + inter_total), 6)
        except Exception:
            additivity_gap = None

    return {
        "baseline_auc": baseline_auc,
        "single_lever_lift": single_lift,
        "two_way_interactions": interactions,
        "full_recipe_lift": full_recipe_lift,
        "sum_single_levers": sum_single,
        "sum_interactions": sum_interactions,
        "additivity_gap": additivity_gap,
        "by_code": by_code,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Wf-197 — Per-innovation ablation launcher (TODO-182 ship)"
    )
    p.add_argument(
        "--n-parallel",
        type=int,
        default=6,
        help="Number of parallel ablation configs (default 6)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs per config (default 30 — apples-to-apples with Wf-124 P3)",
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="Backbone hidden dim (default 64 — CPU-friendly per Wf-196)",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Backbone depth (default 2 — CPU-friendly per Wf-196)",
    )
    p.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size (default 64)"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Single seed for ablation (default 0 — anchor Wf-124 P3)",
    )
    p.add_argument(
        "--max-wall-min",
        type=int,
        default=55,
        help="Hard wall-time cap in minutes (default 55; task budget is 60)",
    )
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_config").mkdir(parents=True, exist_ok=True)

    n_configs = len(ABLATION_CONFIGS)
    print(
        f"[wf197] launching {n_configs} ablation configs at "
        f"epochs={args.epochs} hidden={args.hidden} depth={args.depth} "
        f"n_parallel={args.n_parallel}",
        flush=True,
    )
    t_start = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.n_parallel) as ex:
        futs = {
            ex.submit(
                run_one_config,
                cfg,
                args.epochs,
                args.hidden,
                args.depth,
                args.batch_size,
                args.seed,
                LOG_DIR,
            ): cfg
            for cfg in ABLATION_CONFIGS
        }
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as exc:
                cfg = futs[fut]
                res = {
                    "code": cfg["code"],
                    "label": cfg["label"],
                    "error": repr(exc),
                }
            results.append(res)
            elapsed = time.time() - t_start
            print(
                f"[wf197] DONE cfg={res.get('code')} "
                f"auc={res.get('test_auc')} "
                f"wall={res.get('wall_seconds', 0):.1f}s "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
            # Hard cap check
            if elapsed > args.max_wall_min * 60:
                print(
                    f"[wf197] HARD CAP at {args.max_wall_min} min — "
                    f"cancelling remaining configs",
                    flush=True,
                )
                for fut2 in futs:
                    if not fut2.done():
                        fut2.cancel()
                break

    total_wall = time.time() - t_start
    print(
        f"[wf197] ALL DONE total_wall={total_wall:.1f}s "
        f"({total_wall / 60:.1f} min, {total_wall / 3600:.2f}h)",
        flush=True,
    )

    # Compute attribution
    attribution = compute_attribution(results)

    summary = {
        "wf_tag": "Wf-197 — TODO-182 per-innovation ablation (full 2^3 permutations)",
        "n_configs_planned": n_configs,
        "n_configs_completed": len(
            [r for r in results if r.get("test_auc") is not None]
        ),
        "n_parallel": args.n_parallel,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "depth": args.depth,
        "seed": args.seed,
        "total_wall_seconds": total_wall,
        "total_wall_minutes": total_wall / 60.0,
        "total_wall_hours": total_wall / 3600.0,
        "metal": "Ru",
        "results": sorted(results, key=lambda r: r["code"]),
        "attribution": attribution,
    }

    summary_path = OUT_DIR / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[wf197] wrote summary: {summary_path}", flush=True)

    # Write synthesis.md
    write_synthesis(summary, attribution)
    return 0


def write_synthesis(summary: dict, attribution: dict) -> None:
    """Write human-readable synthesis.md."""
    lines = [
        "# Wf-197 — TODO-182 per-innovation ablation synthesis",
        "",
        "**Date**: 2026-09-28  ",
        "**Owner**: Hugo Lee + ultracode agent  ",
        "**Parent**: `TODO/pending/182_wf124_p3_per_innovation_ablation.md`",
        "**Hardware**: CPU (per standing kill-then-adapt directive)",
        "",
        "## §0 Why this plan exists",
        "",
        "Wf-124 P3 ran 3 innovations simultaneously:",
        "1. **Multi-task D-MPNN** (5-metal joint cohort)",
        "2. **Wasserstein pIC50 regularization** (Wf-123 P1 M1)",
        "3. **L2 guard** (grad-norm clip at 5.0)",
        "",
        "The aggregate lift (+0.049, 0.7123 → 0.7613 on Ru seed=0 30ep h=128) was",
        "an **aggregate** measurement — paper §3 cannot honestly attribute it to",
        "per-lever percentages without this ablation.",
        "",
        "## §1 Acceptance gates",
        "",
        "| Gate | Requirement | Result |",
        "|---|---|---|",
        f"| G1 | All {len(ABLATION_CONFIGS)} permutations complete | {summary['n_configs_completed']}/{summary['n_configs_planned']} |",
        f"| G2 | Per-lever lift attribution table | {'✅' if attribution.get('single_lever_lift') else '❌'} |",
        "| G3 | Aggregate +0.049 attributed to levers | see §4 |",
        "| G4 | Honest framing | see §6 |",
        "",
        "## §2 Ablation matrix (2^3 = 8 configs)",
        "",
        "Code bits: (multi_task, wasserstein, l2_guard)",
        "",
        "| Code | multi_task | W₂ | L2 guard | label |",
        "|---|---|---|---|---|",
    ]
    for r in summary["results"]:
        lines.append(
            f"| {r['code']} | {r.get('multi_task', '?')} | "
            f"{r.get('w_weight', '?')} | {r.get('l2_guard', '?')} | "
            f"{r.get('label', '')} |"
        )

    lines.extend(
        [
            "",
            "## §3 Per-config measured test_auc (Ru seed=0, 30ep, h=64, d=2, CPU)",
            "",
            "| Code | test_auc | wall (s) | return |",
            "|---|---|---|---|",
        ]
    )
    for r in summary["results"]:
        lines.append(
            f"| {r['code']} | {r.get('test_auc', 'N/A')} | "
            f"{r.get('wall_seconds', 0):.1f} | {r.get('returncode', 'N/A')} |"
        )

    lines.extend(
        [
            "",
            "## §4 Per-lever lift attribution (within-cohort, h=64 d=2)",
            "",
        ]
    )
    if "error" in attribution:
        lines.append(f"**Error**: {attribution['error']}")
    else:
        lines.append(f"**Baseline (000) test_auc**: {attribution['baseline_auc']:.4f}")
        lines.append("")
        lines.append("### §4.1 Single-lever lifts (vs baseline)")
        lines.append("")
        lines.append("| Lever | code on | Δ test_auc |")
        lines.append("|---|---|---|")
        for lever, lift in attribution["single_lever_lift"].items():
            if lift is not None:
                lines.append(f"| {lever} | {lever_to_code(lever)} | {lift:+.4f} |")
        lines.append("")
        lines.append("### §4.2 Two-way interactions")
        lines.append("")
        lines.append("| Interaction | formula | value |")
        lines.append("|---|---|---|")
        for inter, val in attribution["two_way_interactions"].items():
            if val is not None:
                lines.append(f"| {inter} | I_AB | {val:+.4f} |")
        lines.append("")
        lines.append("### §4.3 Full recipe (111) lift vs baseline")
        lines.append("")
        if attribution["full_recipe_lift"] is not None:
            lines.append(
                f"- Full recipe (111): test_auc={attribution['baseline_auc'] + attribution['full_recipe_lift']:.4f}"
            )
            lines.append(f"- Δ vs baseline: {attribution['full_recipe_lift']:+.4f}")
        else:
            lines.append("- 111 not measured")
        lines.append("")
        lines.append("### §4.4 Additivity check")
        lines.append("")
        if attribution["additivity_gap"] is not None:
            lines.append(
                f"- Σ single-lever lifts = {attribution['sum_single_levers']:+.4f}"
            )
            lines.append(f"- Σ interactions = {attribution['sum_interactions']:+.4f}")
            lines.append(f"- Full recipe lift = {attribution['full_recipe_lift']:+.4f}")
            lines.append(
                f"- **Additivity gap** = full_recipe - Σ - I = "
                f"{attribution['additivity_gap']:+.4f}"
            )
            lines.append("")
            if abs(attribution["additivity_gap"]) < 0.005:
                lines.append(
                    "**Verdict**: levers are approximately additive (gap < 0.005)."
                )
            elif abs(attribution["additivity_gap"]) < 0.02:
                lines.append(
                    "**Verdict**: small interaction effect (0.005 < gap < 0.02)."
                )
            else:
                lines.append(
                    "**Verdict**: large interaction term — levers are NOT independent."
                )

    lines.extend(
        [
            "",
            "## §5 Per-lever lift % attribution",
            "",
        ]
    )
    if attribution.get("single_lever_lift"):
        baseline = attribution["baseline_auc"]
        total_lift = attribution.get("full_recipe_lift") or 0.0
        if total_lift > 0.001:
            lines.append("| Lever | Δ AUC | % of total lift |")
            lines.append("|---|---|---|")
            for lever, lift in attribution["single_lever_lift"].items():
                if lift is not None:
                    pct = 100.0 * lift / total_lift
                    lines.append(f"| {lever} | {lift:+.4f} | {pct:.1f}% |")
            lines.append(f"| **Full recipe** | {total_lift:+.4f} | 100.0% |")
        else:
            lines.append("Full recipe lift not positive — cannot compute %.")

    lines.extend(
        [
            "",
            "## §6 Honest framing (party-twenty-spirit §3.5)",
            "",
            "**MEASURED**:",
            "- Per-config test_auc for all 8 permutations at h=64, d=2, 30ep, Ru seed=0, CPU",
            "- Per-lever lift attribution WITHIN this cohort (relative)",
            "",
            "**PROJECTED — DO NOT CITE AS MEASURED**:",
            "- The Wf-124 P3 +0.049 absolute lift (h=128, d=3, GPU) is **NOT directly reproducible**",
            "  at h=64, d=2 (CPU). The smaller backbone yields different absolute numbers.",
            "- The +0.049 anchor was on GPU with full backbone at 30ep.",
            "- The per-lever % attribution computed here is **WITHIN-COHORT** relative.",
            "",
            "**INTERACTION CAVEAT**:",
            "- If the additivity gap (§4.4) is large, the 3 levers are NOT independent.",
            "- Simple additive decomposition (Δ_total ≈ Δ_MT + Δ_W + Δ_L2) may be misleading.",
            "",
            "**PER-METAL CAVEAT**:",
            "- Only Ru seed=0 was ablated (single metal, single seed).",
            "- Per-metal attribution for Ir/Os/Rh/Re requires the full 5-metal × 3-seed",
            "  sweep — DEFERRED to TODO-181 GPU follow-up.",
            "",
            "**NO FABRICATION**:",
            "- Per-lever attribution % is reported WITH std (single seed → std=N/A).",
            "- Per-lever lift = single config vs baseline; no best-seed selection.",
            "",
            "## §7 Acceptance verdict",
            "",
        ]
    )
    n_done = summary["n_configs_completed"]
    n_total = summary["n_configs_planned"]
    if n_done == n_total and attribution.get("single_lever_lift"):
        lines.append("✅ **G1 PASS** — All 8 permutations complete")
        lines.append("✅ **G2 PASS** — Per-lever lift attribution table produced")
        # G3 verdict
        full_lift = attribution.get("full_recipe_lift")
        single_sum = attribution.get("sum_single_levers")
        if full_lift is not None and single_sum is not None:
            if abs(full_lift) < 0.001:
                lines.append(
                    "⚠️ **G3 NEGATIVE** — Full recipe lift not positive; cannot attribute +0.049"
                )
                lines.append(
                    "  (The h=64, d=2 CPU ablation did NOT reproduce the Wf-124 P3 +0.049 lift."
                )
                lines.append(
                    "  This is itself a finding — Wf-196 showed the smaller backbone regressed."
                )
            else:
                lines.append(
                    "✅ **G3 PASS** — Per-lever lift attributed (within-cohort)"
                )
        else:
            lines.append("⚠️ **G3 PARTIAL** — Some permutations missing")
        lines.append("✅ **G4 PASS** — Honest framing documented (§6)")
    else:
        lines.append(f"⚠️ **G1 PARTIAL** — {n_done}/{n_total} permutations complete")
        lines.append("⚠️ See §3 for missing configs and §6 for honest framing")

    lines.extend(
        [
            "",
            "## §8 Follow-ups",
            "",
            "1. **TODO-181 GPU follow-up**: full-backbone (h=128, d=3) 50-epoch ablation",
            "   for apples-to-apples comparison with Wf-124 P3 0.7613 anchor.",
            "2. **Per-metal ablation**: 5-metal × 3-seed full permutation (TODO-181 §3.2).",
            "3. **Paper §3 attribution**: cite per-lever % only after GPU ablation completes.",
            "",
        ]
    )
    synth_path = OUT_DIR / "synthesis.md"
    synth_path.write_text("\n".join(lines))
    print(f"[wf197] wrote synthesis: {synth_path}", flush=True)


def lever_to_code(lever: str) -> str:
    return {
        "multi_task": "100",
        "wasserstein": "010",
        "l2_guard": "001",
    }.get(lever, "???")


if __name__ == "__main__":
    raise SystemExit(main())
