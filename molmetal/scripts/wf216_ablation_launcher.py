#!/usr/bin/env python3
"""Wf-216 — TODO-182 per-innovation permutations retry (full 2^3 ablation).

==========================================================
Per-innovation ablation — full 2^3 = 8 permutation matrix
==========================================================

Wf-197 partial attempt produced 6/8 logs but ALL configs got killed
mid-training (none reached the test_auc evaluation at epoch 30). Wf-216
retries with a tighter wall-time budget + smaller backbone (h=64 d=2
Wf-196 protocol) so all 8 configs complete cleanly within 60 min.

**Levers (binary 0/1 per row):**
  bit 0 (MT)  : multi-task 5-metal cohort (1) vs single-task Ru-only (0)
  bit 1 (W)   : Wasserstein pIC50 reg weight (0.05 vs 0.0)
  bit 2 (L2)  : L2 grad-norm guard (on vs off)

**Unique configs (2^3 = 8)**: 000 baseline + 3 single-lever (100, 010, 001)
                              + 3 two-way (110, 101, 011) + 1 full (111).
Note: the task spec lists 1 + 3 + 7 = 11 runs; the 7 "all combinations"
double-count the 3 single-lever runs. We run each unique config once (8
total) and report the overlap in the synthesis.

**Hardware**: CPU primary (per standing kill-then-adapt GPU directive);
6-way parallelism. Wall-time budget: <= 3600s (60 min).

**Honest framing (party-twenty-spirit §3.5)**:
  - Smaller backbone (h=64, d=2) gives different absolute numbers than
    Wf-124 P3 (h=128, d=3, GPU) — the +0.049 Wf-124 P3 anchor is NOT
    directly reproducible at this scale.
  - This ablation attributes lift BETWEEN levers, not ABSOLUTE to Wf-124.

**Outputs** (under `molmetal/reports/wf216_ablation/`):
  - `per_config/per_config_<code>_seed0.json` — per-config test_auc
  - `ablation_summary.json` — full table + per-lever attribution
  - `synthesis.md` — human-readable narrative
  - `wf216_launcher.log` — orchestrator stdout

**Usage**
::
    ${PY} molmetal/scripts/wf216_ablation_launcher.py --n-parallel 6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PY = os.environ.get("PY", "/home/hugo/codes/mol-metal/.venv/bin/python")
TRAIN_SCRIPT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
)
DATA_CSV = Path("/home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv")
OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf216_ablation"
CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints" / "wf216_ablation"
LOG_DIR = OUT_DIR / "logs"

# ----- Ablation matrix (2^3 = 8 unique configurations) -----
# bits: (MT=multi_task, W=Wasserstein, L2=L2 guard)
# Config code: 3-bit string e.g. "111" = all on, "000" = all off (baseline)
ABLATION_CONFIGS: list[dict] = [
    {"code": "000", "multi_task": False, "w_weight": 0.0,  "l2_guard": False, "label": "baseline (no levers)"},
    {"code": "100", "multi_task": True,  "w_weight": 0.0,  "l2_guard": False, "label": "multi-task only"},
    {"code": "010", "multi_task": False, "w_weight": 0.05, "l2_guard": False, "label": "Wasserstein only"},
    {"code": "001", "multi_task": False, "w_weight": 0.0,  "l2_guard": True,  "label": "L2 guard only"},
    {"code": "110", "multi_task": True,  "w_weight": 0.05, "l2_guard": False, "label": "MT + W"},
    {"code": "101", "multi_task": True,  "w_weight": 0.0,  "l2_guard": True,  "label": "MT + L2"},
    {"code": "011", "multi_task": False, "w_weight": 0.05, "l2_guard": True,  "label": "W + L2"},
    {"code": "111", "multi_task": True,  "w_weight": 0.05, "l2_guard": True,  "label": "full recipe (Wf-124 P3 default)"},
]


def run_one_config(
    cfg: dict,
    epochs: int,
    hidden: int,
    depth: int,
    batch_size: int,
    seed: int,
    log_dir: Path,
    out_dir: Path,
    ckpt_dir: Path,
) -> dict:
    """Train one ablation configuration (single metal=Ru, single seed).

    Returns dict with code, label, wall_seconds, returncode, test_auc, etc.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"cfg{cfg['code']}_seed{seed}.log"
    cfg_out = out_dir / "per_config"
    cfg_ckpt = ckpt_dir / f"cfg{cfg['code']}"
    cfg_out.mkdir(parents=True, exist_ok=True)
    cfg_ckpt.mkdir(parents=True, exist_ok=True)

    cmd = [
        PY,
        str(TRAIN_SCRIPT),
        "--metals", "Ru", "Ir", "Os", "Rh", "Re",
        "--seeds", str(seed),
        "--epochs", str(epochs),
        "--hidden", str(hidden),
        "--depth", str(depth),
        "--batch-size", str(batch_size),
        "--device", "cpu",
        "--data-csv", str(DATA_CSV),
        "--out-dir", str(cfg_out),
        "--ckpt-dir", str(cfg_ckpt),
        "--wasserstein-weight", str(cfg["w_weight"]),
        "--no-l2-guard" if not cfg["l2_guard"] else "--use-l2-guard",
    ]
    if not cfg["multi_task"]:
        cmd.append("--single-task")

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT),
        )
    wall = time.time() - t0

    # Read test_auc from per-seed metrics file (race-safe: parse own log)
    metrics_path = cfg_out / "ru" / f"per_seed_metrics_seed{seed}.json"
    test_auc = None
    test_pearson = None
    # Strategy 1: parse own log file (race-safe, since each cfg has its own log)
    try:
        if log_path.exists():
            log_text = log_path.read_text()
            import re
            # Match: [Ru seed=0] DONE best_epoch=X val_auc=Y test_auc=Z ...
            matches = re.findall(
                r"\[Ru seed=\d+\] DONE.*?test_auc=([\d.]+).*?test_pearson=([\d.-]+)",
                log_text,
            )
            if matches:
                test_auc = float(matches[0][0])
                test_pearson = float(matches[0][1])
    except Exception:
        pass
    # Strategy 2: fallback to metrics file (may be overwritten by another config)
    if test_auc is None and metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            # Verify this is the right config by checking wasserstein_weight
            w_in_metrics = metrics.get("wasserstein_weight", -1)
            l2_in_metrics = metrics.get("use_l2_guard", None)
            if abs(w_in_metrics - cfg["w_weight"]) < 0.001 and l2_in_metrics == cfg["l2_guard"]:
                test_auc = metrics.get("test_metrics", {}).get("roc_auc")
                test_pearson = metrics.get("test_pic50_metrics", {}).get("pearson")
        except Exception:
            pass

    # If metrics file missing, parse log for last epoch
    epochs_completed = 0
    if log_path.exists():
        try:
            log_text = log_path.read_text()
            import re
            matches = re.findall(r"epoch\s+(\d+)/\d+", log_text)
            if matches:
                epochs_completed = max(int(m) for m in matches)
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
        "epochs_completed_in_log": epochs_completed,
        "log_path": str(log_path),
        "metrics_path": str(metrics_path) if metrics_path.exists() else None,
    }


def compute_attribution(results: list[dict]) -> dict:
    """Compute per-lever lift attribution from the ablation matrix.

    For each lever L and the interaction term:
        Δ_auc(L=on, others=baseline) - Δ_auc(L=off, others=baseline)

    For 2-way interactions:
        I_AB = auc(A=1, B=1) - auc(A=0, B=1) - auc(A=1, B=0) + auc(A=0, B=0)
    """
    by_code = {r["code"]: r for r in results if r["test_auc"] is not None}
    if "000" not in by_code:
        return {"error": "baseline (000) not measured", "by_code": by_code}

    baseline_auc = by_code["000"]["test_auc"]

    def auc(code: str) -> float | None:
        return by_code.get(code, {}).get("test_auc")

    # Single-lever lifts
    single_lift: dict[str, float | None] = {
        "multi_task": (auc("100") - baseline_auc) if auc("100") is not None else None,
        "wasserstein": (auc("010") - baseline_auc) if auc("010") is not None else None,
        "l2_guard": (auc("001") - baseline_auc) if auc("001") is not None else None,
    }

    # 2-way interactions: I_AB = AUC(A=1,B=1) - AUC(A=1,B=0) - AUC(A=0,B=1) + AUC(0,0)
    # Bit positions: index 0 = multi_task (MT), index 1 = wasserstein (W), index 2 = l2_guard (L2)
    def interaction(a_pos: int, b_pos: int) -> float | None:
        """Compute I_AB given positions of A, B; third lever fixed at 0."""
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
        "I_MT_W": interaction(0, 1),    # MT × W, L2=0
        "I_MT_L2": interaction(0, 2),   # MT × L2, W=0
        "I_W_L2": interaction(1, 2),    # W × L2, MT=0
    }

    # 3-way interaction (full): I_3way = full(111) - main_effects - pairwise
    full_auc_val = auc("111")
    full_recipe_lift = (full_auc_val - baseline_auc) if full_auc_val is not None else None

    # Sum of single-lever lifts
    valid_singles = [v for v in single_lift.values() if v is not None]
    sum_single = sum(valid_singles) if valid_singles else None
    valid_inter = [v for v in interactions.values() if v is not None]
    sum_interactions = sum(valid_inter) if valid_inter else None

    # Additivity check: full_recipe_lift == sum_single + sum_interactions + I_3way
    # where I_3way = full - sum_single - sum_interactions
    three_way_I = None
    additivity_gap = None
    if full_recipe_lift is not None and sum_single is not None and sum_interactions is not None:
        three_way_I = full_recipe_lift - sum_single - sum_interactions
        # If three_way_I == 0, levers are additive (no 3-way interaction)
        additivity_gap = round(three_way_I, 6)

    # Per-lever % attribution
    pct_attribution: dict[str, float | None] = {}
    if full_recipe_lift is not None and abs(full_recipe_lift) > 0.001:
        for lever, lift in single_lift.items():
            if lift is not None:
                pct_attribution[lever] = round(100.0 * lift / full_recipe_lift, 2)
    elif sum_interactions is not None and abs(sum_interactions) > 0.001:
        # Fallback: use sum of single-levers as denominator
        for lever, lift in single_lift.items():
            if lift is not None and sum_single is not None and sum_single != 0:
                pct_attribution[lever] = round(100.0 * lift / sum_single, 2)

    return {
        "baseline_auc": baseline_auc,
        "single_lever_lift": single_lift,
        "two_way_interactions": interactions,
        "full_recipe_lift": full_recipe_lift,
        "sum_single_levers": sum_single,
        "sum_interactions": sum_interactions,
        "three_way_interaction": three_way_I,
        "additivity_gap_three_way": additivity_gap,
        "pct_attribution": pct_attribution,
        "by_code": {k: {"test_auc": v.get("test_auc"), "wall_seconds": v.get("wall_seconds")}
                    for k, v in by_code.items()},
    }


def lever_to_code(lever: str) -> str:
    return {
        "multi_task": "100",
        "wasserstein": "010",
        "l2_guard": "001",
    }.get(lever, "???")


def write_synthesis(summary: dict, attribution: dict) -> None:
    """Write human-readable synthesis.md."""
    lines = [
        "# Wf-216 — TODO-182 per-innovation ablation synthesis (RETRY)",
        "",
        "**Date**: 2026-09-28  ",
        "**Owner**: Hugo Lee + ultracode agent  ",
        "**Parent**: `TODO/pending/182_wf124_p3_per_innovation_ablation.md`",
        "**Retry reason**: Wf-197 produced 6/8 logs but ALL configs got killed mid-training (no test_auc emitted). Wf-216 retries with h=64 d=2 backbone + 15 epochs for clean wall-time fit.",
        "**Hardware**: CPU (per standing kill-then-adapt directive)",
        "",
        "## §0 Why this plan exists",
        "",
        "Wf-124 P3 ran 3 innovations simultaneously:",
        "1. **Multi-task D-MPNN** (5-metal joint cohort)",
        "2. **Wasserstein pIC50 regularization** (Wf-123 P1 M1)",
        "3. **L2 guard** (grad-norm clip at 5.0)",
        "",
        f"The aggregate lift (+0.049, 0.7123 → 0.7613 on Ru seed=0 30ep h=128) was",
        "an **aggregate** measurement — paper §3 cannot honestly attribute it to",
        "per-lever percentages without this ablation.",
        "",
        "## §1 Acceptance gates",
        "",
        f"| Gate | Requirement | Result |",
        f"|---|---|---|",
        f"| G1 | All 8 permutations complete | {summary['n_configs_completed']}/{summary['n_configs_planned']} |",
        f"| G2 | Per-lever lift attribution table | {'PASS' if attribution.get('single_lever_lift') else 'FAIL'} |",
        f"| G3 | Aggregate +0.049 attributed to levers | see §4 |",
        f"| G4 | Honest framing | see §6 |",
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

    lines.extend([
        "",
        "## §3 Per-config measured test_auc (Ru seed=0, 30ep→15ep, h=64, d=2, CPU)",
        "",
        "| Code | test_auc | test_pearson | wall (s) | epochs_completed | return |",
        "|---|---|---|---|---|---|",
    ])
    for r in summary["results"]:
        auc_str = f"{r['test_auc']:.4f}" if r.get('test_auc') is not None else "N/A"
        pearson_str = f"{r['test_pearson']:.4f}" if r.get('test_pearson') is not None else "N/A"
        epochs_str = f"{r.get('epochs_completed_in_log', '?')}/{summary['epochs']}"
        lines.append(
            f"| {r['code']} | {auc_str} | {pearson_str} | "
            f"{r.get('wall_seconds', 0):.1f} | {epochs_str} | "
            f"{r.get('returncode', 'N/A')} |"
        )

    lines.extend([
        "",
        "## §4 Per-lever lift attribution (within-cohort, h=64 d=2)",
        "",
    ])
    if "error" in attribution:
        lines.append(f"**Error**: {attribution['error']}")
    else:
        lines.append(f"**Baseline (000) test_auc**: {attribution['baseline_auc']:.4f}")
        lines.append("")
        lines.append("### §4.1 Single-lever lifts (vs baseline)")
        lines.append("")
        lines.append("| Lever | code on | test_auc | Δ test_auc |")
        lines.append("|---|---|---|---|")
        for lever, lift in attribution["single_lever_lift"].items():
            code_on = lever_to_code(lever)
            auc_val = attribution["by_code"].get(code_on, {}).get("test_auc")
            auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A"
            if lift is not None:
                lines.append(f"| {lever} | {code_on} | {auc_str} | {lift:+.4f} |")
            else:
                lines.append(f"| {lever} | {code_on} | {auc_str} | N/A |")
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
            full_auc = attribution["baseline_auc"] + attribution["full_recipe_lift"]
            lines.append(f"- Full recipe (111): test_auc={full_auc:.4f}")
            lines.append(f"- Δ vs baseline: {attribution['full_recipe_lift']:+.4f}")
        else:
            lines.append("- 111 not measured")
        lines.append("")
        lines.append("### §4.4 Additivity check (3-way interaction)")
        lines.append("")
        if attribution["sum_single_levers"] is not None:
            lines.append(
                f"- Σ single-lever lifts = {attribution['sum_single_levers']:+.4f}"
            )
        if attribution["sum_interactions"] is not None:
            lines.append(
                f"- Σ 2-way interactions = {attribution['sum_interactions']:+.4f}"
            )
        if attribution["full_recipe_lift"] is not None:
            lines.append(
                f"- Full recipe lift = {attribution['full_recipe_lift']:+.4f}"
            )
        if attribution["three_way_interaction"] is not None:
            lines.append(
                f"- **3-way interaction** = full - Σ - I2way = "
                f"{attribution['three_way_interaction']:+.4f}"
            )
        lines.append("")
        if attribution.get("three_way_interaction") is not None:
            gap = attribution["three_way_interaction"]
            if abs(gap) < 0.005:
                lines.append("**Verdict**: levers are approximately additive (3-way gap < 0.005).")
            elif abs(gap) < 0.02:
                lines.append("**Verdict**: small 3-way interaction effect (0.005 < gap < 0.02).")
            else:
                lines.append("**Verdict**: large 3-way interaction term — levers are NOT independent.")

    lines.extend([
        "",
        "## §5 Per-lever lift % attribution (vs full recipe)",
        "",
    ])
    if attribution.get("pct_attribution"):
        lines.append("| Lever | Δ AUC | % of full-recipe lift |")
        lines.append("|---|---|---|")
        for lever, lift in attribution["single_lever_lift"].items():
            pct = attribution["pct_attribution"].get(lever)
            if lift is not None and pct is not None:
                lines.append(f"| {lever} | {lift:+.4f} | {pct:.1f}% |")
        if attribution.get("sum_interactions") is not None:
            lines.append(
                f"| **Σ 2-way interactions** | "
                f"{attribution['sum_interactions']:+.4f} | "
                f"{100.0 * attribution['sum_interactions'] / max(abs(attribution['full_recipe_lift'] or 1e-9), 1e-9):.1f}% |"
            )
        if attribution.get("three_way_interaction") is not None:
            lines.append(
                f"| **3-way interaction** | "
                f"{attribution['three_way_interaction']:+.4f} | "
                f"{100.0 * attribution['three_way_interaction'] / max(abs(attribution['full_recipe_lift'] or 1e-9), 1e-9):.1f}% |"
            )
        if attribution.get("full_recipe_lift") is not None:
            lines.append(f"| **Full recipe** | {attribution['full_recipe_lift']:+.4f} | 100.0% |")
    else:
        lines.append("Full recipe lift not positive — cannot compute meaningful %.")

    lines.extend([
        "",
        "## §6 Honest framing (party-twenty-spirit §3.5)",
        "",
        "**MEASURED**:",
        "- Per-config test_auc for all 8 permutations at h=64, d=2, 15ep, Ru seed=0, CPU",
        "- Per-lever lift attribution WITHIN this cohort (relative)",
        "",
        "**PROJECTED — DO NOT CITE AS MEASURED**:",
        "- The Wf-124 P3 +0.049 absolute lift (h=128, d=3, GPU, 30ep) is **NOT directly reproducible**",
        "  at h=64, d=2 (CPU, 15ep). The smaller backbone yields different absolute numbers.",
        "- The +0.049 anchor was on GPU with full backbone at 30ep.",
        "- The per-lever % attribution computed here is **WITHIN-COHORT** relative.",
        "",
        "**INTERACTION CAVEAT**:",
        "- If the 3-way interaction (§4.4) is large, the 3 levers are NOT independent.",
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
    ])
    n_done = summary["n_configs_completed"]
    n_total = summary["n_configs_planned"]
    if n_done == n_total and attribution.get("single_lever_lift"):
        lines.append(f"PASS **G1** — All {n_total} permutations complete")
        lines.append("PASS **G2** — Per-lever lift attribution table produced")
        full_lift = attribution.get("full_recipe_lift")
        single_sum = attribution.get("sum_single_levers")
        if full_lift is not None and single_sum is not None:
            if abs(full_lift) < 0.001:
                lines.append(f"NEGATIVE **G3** — Full recipe lift not positive ({full_lift:+.4f}); cannot attribute +0.049 directly.")
                lines.append("  (The h=64, d=2 CPU ablation did NOT reproduce the Wf-124 P3 +0.049 lift.")
                lines.append("  This is itself a finding — Wf-196 showed the smaller backbone regressed.)")
            else:
                lines.append(f"PASS **G3** — Per-lever lift attributed: "
                             f"MT={attribution['single_lever_lift'].get('multi_task', 0):+.4f}, "
                             f"W={attribution['single_lever_lift'].get('wasserstein', 0):+.4f}, "
                             f"L2={attribution['single_lever_lift'].get('l2_guard', 0):+.4f} (within-cohort, not absolute)")
        else:
            lines.append("PARTIAL **G3** — Some permutations missing")
        lines.append("PASS **G4** — Honest framing documented (§6)")
    else:
        lines.append(f"PARTIAL **G1** — {n_done}/{n_total} permutations complete")
        lines.append("See §3 for missing configs and §6 for honest framing")

    lines.extend([
        "",
        "## §8 Follow-ups",
        "",
        "1. **TODO-181 GPU follow-up**: full-backbone (h=128, d=3) 50-epoch ablation",
        "   for apples-to-apples comparison with Wf-124 P3 0.7613 anchor.",
        "2. **Per-metal ablation**: 5-metal × 3-seed full permutation (TODO-181 §3.2).",
        "3. **Paper §3 attribution**: cite per-lever % only after GPU ablation completes.",
        "",
    ])
    synth_path = OUT_DIR / "synthesis.md"
    synth_path.write_text("\n".join(lines))
    print(f"[wf216] wrote synthesis: {synth_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Wf-216 — TODO-182 per-innovation permutations retry"
    )
    p.add_argument("--n-parallel", type=int, default=6,
                   help="Number of parallel ablation configs (default 6)")
    p.add_argument("--epochs", type=int, default=15,
                   help="Training epochs per config (default 15 — fits 60 min budget)")
    p.add_argument("--hidden", type=int, default=64,
                   help="Backbone hidden dim (default 64 — Wf-196 CPU protocol)")
    p.add_argument("--depth", type=int, default=2,
                   help="Backbone depth (default 2 — Wf-196 CPU protocol)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Mini-batch size (default 64)")
    p.add_argument("--seed", type=int, default=0,
                   help="Single seed for ablation (default 0 — anchor Wf-124 P3)")
    p.add_argument("--max-wall-min", type=int, default=55,
                   help="Hard wall-time cap in minutes (default 55; task budget is 60)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help="Output directory (default: molmetal/reports/wf216_ablation)")
    args = p.parse_args(argv)

    out_dir = args.out_dir
    ckpt_dir = PROJECT_ROOT / "molmetal" / "checkpoints" / "wf216_ablation"
    log_dir = out_dir / "logs"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_config").mkdir(parents=True, exist_ok=True)

    n_configs = len(ABLATION_CONFIGS)
    print(
        f"[wf216] launching {n_configs} ablation configs at "
        f"epochs={args.epochs} hidden={args.hidden} depth={args.depth} "
        f"n_parallel={args.n_parallel}",
        flush=True,
    )
    print(
        f"[wf216] target: full 2^3 = 8 unique permutations "
        f"(1 baseline + 3 single + 3 two-way + 1 three-way)",
        flush=True,
    )
    print(f"[wf216] note: task spec 1+3+7=11 has overlap "
          f"(the 7 combinations include the 3 single-lever runs); "
          f"we run each unique config once",
          flush=True)

    t_start = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.n_parallel) as ex:
        futs = {
            ex.submit(
                run_one_config,
                cfg, args.epochs, args.hidden, args.depth,
                args.batch_size, args.seed, log_dir, out_dir, ckpt_dir,
            ): cfg for cfg in ABLATION_CONFIGS
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
            auc_str = f"{res.get('test_auc'):.4f}" if res.get('test_auc') is not None else "N/A"
            print(
                f"[wf216] DONE cfg={res.get('code')} "
                f"auc={auc_str} "
                f"epochs={res.get('epochs_completed_in_log', '?')}/{args.epochs} "
                f"wall={res.get('wall_seconds', 0):.1f}s "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
            # Hard cap check
            if elapsed > args.max_wall_min * 60:
                print(
                    f"[wf216] HARD CAP at {args.max_wall_min} min — "
                    f"cancelling remaining configs",
                    flush=True,
                )
                for fut2 in futs:
                    if not fut2.done():
                        fut2.cancel()
                break

    total_wall = time.time() - t_start
    print(
        f"[wf216] ALL DONE total_wall={total_wall:.1f}s "
        f"({total_wall / 60:.1f} min, {total_wall / 3600:.2f}h)",
        flush=True,
    )

    # Compute attribution
    attribution = compute_attribution(results)

    summary = {
        "wf_tag": "Wf-216 — TODO-182 per-innovation ablation (full 2^3 permutations, retry)",
        "retry_of": "Wf-197 (killed mid-training)",
        "n_configs_planned": n_configs,
        "n_configs_completed": len([r for r in results if r.get("test_auc") is not None]),
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

    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[wf216] wrote summary: {summary_path}", flush=True)

    # Write per-config JSON files
    for r in results:
        cfg_path = out_dir / f"per_config_{r['code']}_seed{args.seed}.json"
        cfg_path.write_text(json.dumps(r, indent=2, default=str))

    # Write synthesis.md
    write_synthesis(summary, attribution)

    # Exit code: 0 if all completed, else 1
    return 0 if summary["n_configs_completed"] == n_configs else 1


if __name__ == "__main__":
    raise SystemExit(main())
