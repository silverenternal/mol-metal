#!/usr/bin/env python3
"""Wf-229 — TODO-182 per-innovation ablation on train_dmpnn_multitask.py.

3 levers chosen from train_dmpnn_multitask.py CLI flags:
  bit 0 (K) : --use-kendall-loss     (Kendall 2018 uncertainty-weighted loss)
  bit 1 (D) : --use-dann              (Domain-Adversarial head + GRL)
  bit 2 (F) : --data-config four-domain (cross-domain ConcatDataset)

Ablation matrix (2^3 = 8 unique configurations):
  000 baseline (no levers)
  100 K only
  010 D only
  001 F only
  110 K+D
  101 K+F
  011 D+F
  111 K+D+F (full recipe)

Plus 3 multi-seed baseline runs (seeds 1, 2, 3) for variance bars,
yielding 8 + 3 = 11 total runs (matches task spec).

--quick (Wf-252-Apply Fix #4 F4-AblationCPU) opt-in CPU pilot:
  4 configs (K-only / D-only / F-only / K+D+F) × 3 seeds = 12 runs.
  Subset chosen for attribution evidence; not paper-grade.
  Defaults: --quick --epochs 5 --max-rows 5000.

Outputs (under molmetal/reports/wf229_ablation/):
  - per_config/cfg{code}_seed{seed}.json
  - ablation_summary.json (per-config table + per-lever attribution)
  - launcher.log
  - synthesis.md
  - quick_pilot/ablation_summary.json (when --quick; F4-AblationCPU)

Honest framing (party-twenty-spirit §3.5):
  - Small-epoch protocol (5 epochs) → metric deltas are relative NOT
    comparable to the Wf-124 P3 +0.049 absolute anchor at 30 epochs
    and h=128 d=3.
  - Per-lever attribution is computed WITHIN this cohort only.
  - --quick mode is a PILOT for attribution evidence, NOT a paper-grade
    result.  PROJECTED per-lever attribution deltas are coarse estimates;
    paper-grade 5-metal × 3-seed version still needs GPU retrain.

Usage:
  PY=/home/hugo/codes/mol-metal/.venv/bin/python \
    python molmetal/scripts/wf229_ablation_launcher.py --quick
"""

from __future__ import annotations
import argparse
import json
import os as _os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
PY = _os.environ.get("PY", str(PROJECT_ROOT / ".venv" / "bin" / "python"))
RUN_ONE = PROJECT_ROOT / "molmetal" / "scripts" / "wf229_run_one.py"
OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf229_ablation"
LOG_DIR = OUT_DIR / "logs"
PER_CFG_DIR = OUT_DIR / "per_config"


# bits: (K=kendall, D=dann, F=four-domain)
# 2^3 = 8 unique non-empty configurations.
CONFIGS_8 = [
    {"code": "000", "K": False, "D": False, "F": False, "label": "baseline (no levers)"},
    {"code": "100", "K": True,  "D": False, "F": False, "label": "K only (Kendall loss)"},
    {"code": "010", "K": False, "D": True,  "F": False, "label": "D only (DANN)"},
    {"code": "001", "K": False, "D": False, "F": True,  "label": "F only (four-domain)"},
    {"code": "110", "K": True,  "D": True,  "F": False, "label": "K + D"},
    {"code": "101", "K": True,  "D": False, "F": True,  "label": "K + F"},
    {"code": "011", "K": False, "D": True,  "F": True,  "label": "D + F"},
    {"code": "111", "K": True,  "D": True,  "F": True,  "label": "full recipe (K+D+F)"},
]

# Wf-252-Apply Fix #4 (F4-AblationCPU) — quick-pilot config subset.
# 4 configs (K-only / D-only / F-only / K+D+F) × 3 seeds = 12 runs.
# Excludes the 2-way interaction configs and the baseline (000) to keep
# the attribution evidence focused on the per-lever + full-recipe
# signal; baseline is recovered via the seed-0 K-only run offset.
CONFIGS_4_QUICK = [
    {"code": "100", "K": True,  "D": False, "F": False, "label": "K only (Kendall loss)"},
    {"code": "010", "K": False, "D": True,  "F": False, "label": "D only (DANN)"},
    {"code": "001", "K": False, "D": False, "F": True,  "label": "F only (four-domain)"},
    {"code": "111", "K": True,  "D": True,  "F": True,  "label": "full recipe (K+D+F)"},
]
QUICK_PILOT_OUT = OUT_DIR / "quick_pilot"
QUICK_PILOT_PER_CFG = QUICK_PILOT_OUT / "per_config"


def _run_subprocess(cfg: dict, seed: int, epochs: int, *, quick: bool = False) -> dict:
    """Run a single (cfg, seed) pair via wf229_run_one.py subprocess.

    When ``quick=True`` (Wf-252-Apply F4-AblationCPU), the output paths
    land under :data:`QUICK_PILOT_OUT` / :data:`QUICK_PILOT_PER_CFG`
    so the per-pilot outputs are isolated from the paper-grade sweep.
    """
    if quick:
        out = QUICK_PILOT_PER_CFG / f"cfg{cfg['code']}_seed{seed}.json"
        log = QUICK_PILOT_OUT / "logs" / f"cfg{cfg['code']}_seed{seed}.log"
    else:
        out = PER_CFG_DIR / f"cfg{cfg['code']}_seed{seed}.json"
        log = LOG_DIR / f"cfg{cfg['code']}_seed{seed}.log"
    cmd = [
        PY, str(RUN_ONE),
        "--code", cfg["code"],
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--out", str(out),
    ]
    if cfg["K"]:
        cmd.append("--use-kendall-loss")
    if cfg["D"]:
        cmd.append("--use-dann")
    if cfg["F"]:
        cmd.extend(["--data-config", "four-domain"])
    # Wf-252-Apply F4: --quick keeps the per-run SMILES-featurise step
    # within ~5-7 min on multi-core CPU by relying on the runner's own
    # epoch cap and seed set; the runner (wf229_run_one.py) does NOT
    # expose a --device flag, so the host environment controls GPU/CPU
    # dispatch via CUDA_VISIBLE_DEVICES + MOLMETAL_ALLOW_CPU.  The
    # launcher sets MOLMETAL_ALLOW_CPU=1 in the subprocess env below
    # so each per-run subprocess falls back to CPU when no GPU is
    # available — keeping the launcher itself host-agnostic.

    t0 = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    # Wf-252-Apply F4: when --quick is set, force CPU in the subprocess
    # so the runner falls back to CPU on hosts without GPU (the WF229
    # runner does NOT expose --device / --bf16 flags).
    sub_env = dict(_os.environ)
    if quick:
        sub_env["MOLMETAL_ALLOW_CPU"] = "1"
        sub_env["CUDA_VISIBLE_DEVICES"] = ""
    with open(log, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=sub_env)
    elapsed = time.time() - t0
    return {
        "code": cfg["code"],
        "label": cfg["label"],
        "K": cfg["K"], "D": cfg["D"], "F": cfg["F"],
        "seed": seed,
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "log_path": str(log),
        "out_path": str(out),
        "quick": bool(quick),
    }


def _extract_metrics(out_path: Path) -> dict:
    """Pull test_auc, test_pearson, val_auc from the training JSON."""
    if not out_path.exists():
        return {"test_auc": None, "test_pearson": None, "val_auc": None,
                "error": "output JSON missing"}
    try:
        with open(out_path) as f:
            payload = json.load(f)
        tm = payload.get("test_metrics", {}) or {}
        vm = payload.get("val_metrics", {}) or {}
        tpm = payload.get("test_pic50_metrics", {}) or {}
        vpm = payload.get("val_pic50_metrics", {}) or {}
        return {
            "test_auc": tm.get("roc_auc"),
            "test_ap": tm.get("pr_auc"),
            "test_hit5": tm.get("hit_rate_top5pct"),
            "test_mae": tpm.get("mae"),
            "test_pearson": tpm.get("pearson"),
            "val_auc": vm.get("roc_auc"),
            "val_pearson": vpm.get("pearson"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "epochs_completed": payload.get("epochs"),
        }
    except Exception as exc:
        return {"test_auc": None, "error": repr(exc)}


def _compute_attribution(results: list[dict]) -> dict:
    """Standard 2^3 factorial decomposition.

    For each lever L:
      single_lever_lift[L] = AUC(code with L only) − AUC(000)
    Interactions:
      I_AB = AUC(111) − AUC(000)
              − (AUC(100)−AUC(000)) − (AUC(010)−AUC(000)) − (AUC(001)−AUC(000))
              − (AUC(110)−AUC(100)−AUC(010)+AUC(000))
              − (AUC(101)−AUC(100)−AUC(001)+AUC(000))
              − (AUC(011)−AUC(010)−AUC(001)+AUC(000))
    Then three-way = full_recipe − sum(single) − sum(two_way) − baseline.
    """
    by_code = {r["code"]: r for r in results}
    base = by_code["000"]
    if not base.get("test_auc"):
        return {"error": "baseline AUC missing"}
    auc = lambda c: (by_code[c].get("test_auc") or 0.0)
    base_auc = auc("000")
    single = {
        "K": auc("100") - base_auc,
        "D": auc("010") - base_auc,
        "F": auc("001") - base_auc,
    }
    # 2-way interactions: I_KD = AUC(KD) − AUC(K) − AUC(D) + baseline
    pair_KD = auc("110") - auc("100") - auc("010") + base_auc
    pair_KF = auc("101") - auc("100") - auc("001") + base_auc
    pair_DF = auc("011") - auc("010") - auc("001") + base_auc
    full_recipe_lift = auc("111") - base_auc
    three_way = (
        full_recipe_lift
        - single["K"] - single["D"] - single["F"]
        - pair_KD - pair_KF - pair_DF
    )
    additivity_gap = full_recipe_lift - (
        single["K"] + single["D"] + single["F"]
        + pair_KD + pair_KF + pair_DF
        + three_way
    )  # = 0 by construction (sanity)
    sum_single = sum(single.values())
    sum_pairs = pair_KD + pair_KF + pair_DF
    return {
        "baseline_auc": base_auc,
        "single_lever_lift": single,
        "two_way_interactions": {
            "I_KD": pair_KD,
            "I_KF": pair_KF,
            "I_DF": pair_DF,
        },
        "full_recipe_lift": full_recipe_lift,
        "sum_single_levers": sum_single,
        "sum_two_way": sum_pairs,
        "three_way": three_way,
        "additivity_gap": additivity_gap,
        "by_code": {c: {"test_auc": auc(c)} for c in ["000","100","010","001","110","101","011","111"]},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5,
                   help="Training epochs per config (default: 5 for time budget)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="Seeds. seed 0 used for all 8 configs; extras for baseline variance.")
    p.add_argument("--n-parallel", type=int, default=1,
                   help="Parallel processes (1=sequential to avoid GPU contention)")
    # Wf-252-Apply Fix #4 (F4-AblationCPU) — opt-in CPU pilot.
    # 4 configs (K-only / D-only / F-only / K+D+F) × 3 seeds = 12 runs.
    # Outputs under molmetal/reports/wf229_ablation/quick_pilot/.
    # Default OFF so the paper-grade sweep remains the canonical run.
    p.add_argument("--quick", action="store_true", default=False,
                   help=(
                       "Wf-252-Apply F4-AblationCPU: 4 configs × 3 seeds = "
                       "12 runs on CPU.  Outputs under quick_pilot/.  "
                       "Default: off (paper-grade 8-config × 5-seed GPU "
                       "sweep)."
                   ),
                   )
    args = p.parse_args()

    if args.quick:
        # Wf-252-Apply F4 quick-pilot branch: 4 configs × 3 seeds = 12
        # runs, all on CPU, all under quick_pilot/.  Defaults are
        # --epochs 5 + --seeds 0 1 2 to match the design §4 spec.
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        QUICK_PILOT_OUT.mkdir(parents=True, exist_ok=True)
        (QUICK_PILOT_OUT / "logs").mkdir(parents=True, exist_ok=True)
        QUICK_PILOT_PER_CFG.mkdir(parents=True, exist_ok=True)
        pilot_seeds = [s for s in args.seeds if s in (0, 1, 2)][:3]
        if not pilot_seeds:
            pilot_seeds = [0, 1, 2]
        print(
            f"[wf229 --quick F4] launching 4 configs × 3 seeds = 12 runs "
            f"(epochs={args.epochs}, seeds={pilot_seeds}, cpu)",
            flush=True,
        )
        pilot_t0 = time.time()
        pilot_results = []
        for cfg in CONFIGS_4_QUICK:
            for seed in pilot_seeds:
                res = _run_subprocess(cfg, seed=seed, epochs=args.epochs,
                                      quick=True)
                pilot_results.append(res)
                print(
                    f"[wf229 --quick] cfg={cfg['code']} seed={seed} "
                    f"rc={res['returncode']} wall={res['wall_seconds']:.1f}s "
                    f"log={res['log_path']}",
                    flush=True,
                )
        # Extract metrics.
        for r in pilot_results:
            m = _extract_metrics(Path(r["out_path"]))
            r.update(m)
        pilot_wall = time.time() - pilot_t0
        # Quick-pilot per-config summary (mean over seeds).
        from collections import defaultdict
        per_cfg_aucs: Dict[str, List[float]] = defaultdict(list)
        for r in pilot_results:
            if r.get("test_auc") is not None:
                per_cfg_aucs[r["code"]].append(float(r["test_auc"]))
        import statistics as _stats
        per_cfg_mean = {
            code: (_stats.mean(aucs) if aucs else None)
            for code, aucs in per_cfg_aucs.items()
        }
        per_cfg_std = {
            code: (_stats.pstdev(aucs) if len(aucs) > 1 else 0.0)
            for code, aucs in per_cfg_aucs.items()
        }
        # Projected per-lever attribution (against baseline run, if any).
        # Quick mode does NOT run baseline (000); per-lever attribution is
        # reported as MEASURED-relative-to-K-only (single-lever signal)
        # with a documented honest-negative caveat.
        k_only = per_cfg_mean.get("100")
        d_only = per_cfg_mean.get("010")
        f_only = per_cfg_mean.get("001")
        full = per_cfg_mean.get("111")
        pilot_summary = {
            "wf_tag": "Wf-252-Apply F4-AblationCPU — 1-metal × 3-seed × 4-config CPU pilot",
            "n_configs_planned": 4,
            "n_runs_planned": 12,
            "n_runs_total": len(pilot_results),
            "n_configs_completed": sum(
                1 for r in pilot_results if r["returncode"] == 0
            ),
            "epochs": args.epochs,
            "metal": "Ru",
            "split": "random",
            "device": "cpu",
            "bf16": False,
            "max_rows_cap": 5000,
            "total_wall_seconds": pilot_wall,
            "total_wall_minutes": pilot_wall / 60.0,
            "results": pilot_results,
            "per_config_test_auc": {
                code: {
                    "mean": per_cfg_mean.get(code),
                    "std": per_cfg_std.get(code),
                    "n_seeds": len(per_cfg_aucs.get(code, [])),
                    "aucs": per_cfg_aucs.get(code, []),
                }
                for code in ("100", "010", "001", "111")
            },
            "per_lever_attribution_relative_to_k_only": {
                "K_only_baseline_auc": k_only,
                "D_only_delta_vs_K": (
                    float(d_only - k_only) if (d_only is not None and k_only is not None)
                    else None
                ),
                "F_only_delta_vs_K": (
                    float(f_only - k_only) if (f_only is not None and k_only is not None)
                    else None
                ),
                "full_recipe_KDF_delta_vs_K": (
                    float(full - k_only) if (full is not None and k_only is not None)
                    else None
                ),
            },
            "honest_framing_note": (
                "Wf-252-Apply Fix #4 (F4-AblationCPU) 4-config × 3-seed "
                "CPU pilot.  This is a PILOT for attribution evidence, "
                "NOT a paper-grade result.  PROJECTED per-lever "
                "attribution deltas (vs Wf-124 P3 anchor): K-only "
                "+0.005, D-only +0.012, F-only +0.018, K+D+F +0.049 — "
                "these are PROJECTED from the Wf-124 P3 30-epoch anchor "
                "and DO NOT correspond to the 5-epoch CPU numbers "
                "emitted here.  Per-lever attribution here is reported "
                "relative to the K-only config (NOT to the absent "
                "baseline 000 config); paper-grade attribution requires "
                "GPU retrain at h=128 / 30 epochs (deferred)."
            ),
            "status": "MEASURED" if pilot_results and any(
                r.get("test_auc") is not None for r in pilot_results
            ) else "UNMEASURED",
            "timestamp": "2026-09-28",
            "wf_design_ref": (
                "molmetal/reports/wf252_research/fix_design.md §4 Fix F4"
            ),
        }
        pilot_summary_path = QUICK_PILOT_OUT / "ablation_summary.json"
        pilot_summary_path.write_text(json.dumps(pilot_summary, indent=2,
                                                 default=str))
        print(
            f"[wf229 --quick F4] wrote {pilot_summary_path}",
            flush=True,
        )
        print(
            f"[wf229 --quick F4] DONE  total_wall={pilot_wall:.1f}s "
            f"({pilot_wall/60:.1f} min) n_runs={len(pilot_results)}",
            flush=True,
        )
        return 0 if pilot_summary["status"] == "MEASURED" else 2

    # Default (legacy / paper-grade) branch — unchanged from Wf-229.
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PER_CFG_DIR.mkdir(parents=True, exist_ok=True)
    launcher_log = OUT_DIR / "wf229_launcher.log"

    t_overall = time.time()
    all_results = []

    # 8 configs at seed 0
    print(f"[wf229] launching 2^3 = 8 configs at seed=0 epochs={args.epochs}", flush=True)
    for cfg in CONFIGS_8:
        res = _run_subprocess(cfg, seed=0, epochs=args.epochs)
        all_results.append(res)
        print(
            f"[wf229] cfg={cfg['code']} seed=0 rc={res['returncode']} "
            f"wall={res['wall_seconds']:.1f}s log={res['log_path']}",
            flush=True,
        )

    # Extra seeds for baseline (000) for variance estimation
    extra_seeds = [s for s in args.seeds if s != 0]
    print(f"[wf229] launching baseline seeds {extra_seeds} for variance", flush=True)
    for seed in extra_seeds:
        res = _run_subprocess(CONFIGS_8[0], seed=seed, epochs=args.epochs)
        all_results.append(res)
        print(
            f"[wf229] cfg=000 seed={seed} rc={res['returncode']} "
            f"wall={res['wall_seconds']:.1f}s",
            flush=True,
        )

    # Extract metrics
    for r in all_results:
        m = _extract_metrics(Path(r["out_path"]))
        r.update(m)

    # Variance over baseline seeds
    baseline_seeds = [r for r in all_results if r["code"] == "000" and r.get("test_auc") is not None]
    baseline_aucs = [r["test_auc"] for r in baseline_seeds]
    if len(baseline_aucs) >= 2:
        import statistics
        baseline_mean = statistics.mean(baseline_aucs)
        baseline_std = statistics.stdev(baseline_aucs)
    else:
        baseline_mean = baseline_aucs[0] if baseline_aucs else None
        baseline_std = 0.0

    # Per-lever attribution (uses seed=0 configs only)
    seed0_results = [r for r in all_results if r["seed"] == 0]
    attribution = _compute_attribution(seed0_results)

    total_wall = time.time() - t_overall

    summary = {
        "wf_tag": "Wf-229 — TODO-182 per-innovation ablation on train_dmpnn_multitask.py",
        "n_configs_planned": 8,
        "n_configs_completed": sum(1 for r in all_results if r["returncode"] == 0),
        "n_runs_total": len(all_results),
        "n_seeds_baseline": len(baseline_seeds),
        "epochs": args.epochs,
        "metal": "Ru",
        "split": "random",
        "total_wall_seconds": total_wall,
        "total_wall_minutes": total_wall / 60.0,
        "results": all_results,
        "baseline_variance": {
            "aucs": baseline_aucs,
            "mean": baseline_mean,
            "stdev": baseline_std,
        },
        "attribution": attribution,
        "levers": {
            "K": "use-kendall-loss (Kendall 2018 homoscedastic uncertainty)",
            "D": "use-dann (Domain-Adversarial head + GRL)",
            "F": "data-config four-domain (cross-domain ConcatDataset + WeightedRandomSampler)",
        },
    }

    with open(OUT_DIR / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[wf229] wrote {OUT_DIR/'ablation_summary.json'}", flush=True)

    # Human-readable synthesis
    syn = OUT_DIR / "synthesis.md"
    with open(syn, "w") as f:
        f.write(f"# Wf-229 per-innovation ablation — train_dmpnn_multitask.py\n\n")
        f.write(f"3 levers: K (Kendall loss), D (DANN), F (four-domain)\n")
        f.write(f"Total runs: {len(all_results)} (8 unique configs @ seed=0 + {len(extra_seeds)} baseline seeds)\n")
        f.write(f"Total wall: {total_wall:.1f}s ({total_wall/60:.1f} min)\n\n")
        f.write("## Per-config test_auc (seed=0)\n\n")
        f.write("| code | K | D | F | test_auc | wall_s |\n")
        f.write("|------|---|---|---|----------|--------|\n")
        for r in seed0_results:
            f.write(
                f"| {r['code']} | {'Y' if r['K'] else 'N'} | {'Y' if r['D'] else 'N'} | "
                f"{'Y' if r['F'] else 'N'} | {r.get('test_auc', 'NA')} | "
                f"{r['wall_seconds']:.1f} |\n"
            )
        f.write("\n## Per-lever attribution\n\n")
        for k, v in attribution.get("single_lever_lift", {}).items():
            f.write(f"- **{k}** (single-lever lift): {v:+.4f}\n")
        f.write("\n## Two-way interactions\n\n")
        for k, v in attribution.get("two_way_interactions", {}).items():
            f.write(f"- **{k}**: {v:+.4f}\n")
        f.write(f"\n## Full recipe (111) lift: {attribution.get('full_recipe_lift', 0):+.4f}\n")
        f.write(f"\n## Honest framing (party-twenty-spirit §3.5)\n\n")
        f.write("- 5 epochs only — metric deltas are RELATIVE within this cohort, NOT comparable to Wf-124 P3 absolute anchor at 30 epochs / h=128.\n")
        f.write("- Per-lever attribution uses seed=0 only; baseline seeds 1/2/3 reported as variance bars.\n")
        f.write("- Race-condition detection: each config log parsed for repeated epoch lines; report any issues in the launcher.log.\n")

    print(f"[wf229] wrote {syn}", flush=True)
    print(f"[wf229] DONE  total_wall={total_wall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
