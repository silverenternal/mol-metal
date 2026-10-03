#!/usr/bin/env python3
"""Wf-240 — EXTENSION of Wf-229 with longer wall-time (50 epochs) + all 11 configs.

CONTEXT: Wf-229 only completed 2/11 configs at 5 epochs (K-only lift=0 within noise).
This extension re-runs ALL configs at 50 epochs to get a MEASURED signal whether
the K/D/F levers actually lift beyond noise when the model has time to converge.

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

Outputs (under molmetal/reports/wf240_ablation_extended/):
  - per_config/cfg{code}_seed{seed}.json
  - ablation_summary.json (per-config table + per-lever attribution)
  - launcher.log
  - synthesis.md

Honest framing (party-twenty-spirit §3.5):
  - 50 epochs (vs Wf-229's 5) is closer to a converged signal but still
    NOT comparable to the Wf-124 P3 +0.049 absolute anchor at 30 epochs
    and h=128 d=3 (different model scale).
  - Per-lever attribution is computed WITHIN this cohort only.
  - Wf-229 result was "K-only lift=0 within noise" at 5 epochs — the
    extension verifies whether this negative holds at 50 epochs OR
    whether the K lever is just slow-converging.

Usage:
  PY=/home/hugo/codes/mol-metal/.venv/bin/python \
    CUDA_VISIBLE_DEVICES=0 \
    python molmetal/scripts/wf240_ablation_launcher.py
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )

    _WF424_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF424_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]


PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
PY = os.environ.get("PY", str(PROJECT_ROOT / ".venv" / "bin" / "python"))
RUN_ONE = PROJECT_ROOT / "molmetal" / "scripts" / "wf229_run_one.py"
OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf240_ablation_extended"
LOG_DIR = OUT_DIR / "logs"
PER_CFG_DIR = OUT_DIR / "per_config"


# bits: (K=kendall, D=dann, F=four-domain)
# 2^3 = 8 unique non-empty configurations.
CONFIGS_8 = [
    {
        "code": "000",
        "K": False,
        "D": False,
        "F": False,
        "label": "baseline (no levers)",
    },
    {
        "code": "100",
        "K": True,
        "D": False,
        "F": False,
        "label": "K only (Kendall loss)",
    },
    {"code": "010", "K": False, "D": True, "F": False, "label": "D only (DANN)"},
    {"code": "001", "K": False, "D": False, "F": True, "label": "F only (four-domain)"},
    {"code": "110", "K": True, "D": True, "F": False, "label": "K + D"},
    {"code": "101", "K": True, "D": False, "F": True, "label": "K + F"},
    {"code": "011", "K": False, "D": True, "F": True, "label": "D + F"},
    {"code": "111", "K": True, "D": True, "F": True, "label": "full recipe (K+D+F)"},
]


def _wf424_atomic_write_text(
    path, content: str, min_lines: int = 50, expected_min_length: int = 200
) -> None:
    """Wf-424-A4 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF424_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk,
                    path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] detect_truncation reason=%r at %s",
                    _trunc,
                    path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")


def _run_subprocess(cfg: dict, seed: int, epochs: int) -> dict:
    """Run a single (cfg, seed) pair via wf229_run_one.py subprocess."""
    out = PER_CFG_DIR / f"cfg{cfg['code']}_seed{seed}.json"
    log = LOG_DIR / f"cfg{cfg['code']}_seed{seed}.log"
    cmd = [
        PY,
        str(RUN_ONE),
        "--code",
        cfg["code"],
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--out",
        str(out),
    ]
    if cfg["K"]:
        cmd.append("--use-kendall-loss")
    if cfg["D"]:
        cmd.append("--use-dann")
    if cfg["F"]:
        cmd.extend(["--data-config", "four-domain"])

    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    return {
        "code": cfg["code"],
        "label": cfg["label"],
        "K": cfg["K"],
        "D": cfg["D"],
        "F": cfg["F"],
        "seed": seed,
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "log_path": str(log),
        "out_path": str(out),
    }


def _extract_metrics(out_path: Path) -> dict:
    """Pull test_auc, test_pearson, val_auc from the training JSON."""
    if not out_path.exists():
        return {
            "test_auc": None,
            "test_pearson": None,
            "val_auc": None,
            "error": "output JSON missing",
        }
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
    """Standard 2^3 factorial decomposition."""
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
    pair_KD = auc("110") - auc("100") - auc("010") + base_auc
    pair_KF = auc("101") - auc("100") - auc("001") + base_auc
    pair_DF = auc("011") - auc("010") - auc("001") + base_auc
    full_recipe_lift = auc("111") - base_auc
    three_way = (
        full_recipe_lift
        - single["K"]
        - single["D"]
        - single["F"]
        - pair_KD
        - pair_KF
        - pair_DF
    )
    additivity_gap = full_recipe_lift - (
        single["K"]
        + single["D"]
        + single["F"]
        + pair_KD
        + pair_KF
        + pair_DF
        + three_way
    )
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
        "by_code": {
            c: {"test_auc": auc(c)}
            for c in ["000", "100", "010", "001", "110", "101", "011", "111"]
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per config (default: 50 for MEASURED signal)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Seeds. seed 0 used for all 8 configs; extras for baseline variance.",
    )
    p.add_argument(
        "--n-parallel",
        type=int,
        default=1,
        help="Parallel processes (1=sequential to avoid GPU contention)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs that already have a JSON output.",
    )
    args = p.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PER_CFG_DIR.mkdir(parents=True, exist_ok=True)
    launcher_log = OUT_DIR / "wf240_launcher.log"

    t_overall = time.time()
    all_results = []

    # 8 configs at seed 0
    print(
        f"[wf240] launching 2^3 = 8 configs at seed=0 epochs={args.epochs}", flush=True
    )
    for cfg in CONFIGS_8:
        out = PER_CFG_DIR / f"cfg{cfg['code']}_seed0.json"
        if args.resume and out.exists():
            try:
                with open(out) as f:
                    payload = json.load(f)
                if payload.get("test_metrics"):
                    print(
                        f"[wf240] cfg={cfg['code']} seed=0 SKIP (resume, already done)",
                        flush=True,
                    )
                    all_results.append(
                        {
                            "code": cfg["code"],
                            "label": cfg["label"],
                            "K": cfg["K"],
                            "D": cfg["D"],
                            "F": cfg["F"],
                            "seed": 0,
                            "returncode": 0,
                            "wall_seconds": 0.0,
                            "log_path": str(LOG_DIR / f"cfg{cfg['code']}_seed0.log"),
                            "out_path": str(out),
                            "_resumed": True,
                        }
                    )
                    continue
            except Exception:
                pass  # fall through to run
        res = _run_subprocess(cfg, seed=0, epochs=args.epochs)
        all_results.append(res)
        print(
            f"[wf240] cfg={cfg['code']} seed=0 rc={res['returncode']} "
            f"wall={res['wall_seconds']:.1f}s log={res['log_path']}",
            flush=True,
        )

    # Extra seeds for baseline (000) for variance estimation
    extra_seeds = [s for s in args.seeds if s != 0]
    print(f"[wf240] launching baseline seeds {extra_seeds} for variance", flush=True)
    for seed in extra_seeds:
        out = PER_CFG_DIR / f"cfg000_seed{seed}.json"
        if args.resume and out.exists():
            try:
                with open(out) as f:
                    payload = json.load(f)
                if payload.get("test_metrics"):
                    print(
                        f"[wf240] cfg=000 seed={seed} SKIP (resume, already done)",
                        flush=True,
                    )
                    all_results.append(
                        {
                            "code": "000",
                            "label": "baseline (no levers)",
                            "K": False,
                            "D": False,
                            "F": False,
                            "seed": seed,
                            "returncode": 0,
                            "wall_seconds": 0.0,
                            "log_path": str(LOG_DIR / f"cfg000_seed{seed}.log"),
                            "out_path": str(out),
                            "_resumed": True,
                        }
                    )
                    continue
            except Exception:
                pass
        res = _run_subprocess(CONFIGS_8[0], seed=seed, epochs=args.epochs)
        all_results.append(res)
        print(
            f"[wf240] cfg=000 seed={seed} rc={res['returncode']} "
            f"wall={res['wall_seconds']:.1f}s",
            flush=True,
        )

    # Extract metrics
    for r in all_results:
        m = _extract_metrics(Path(r["out_path"]))
        r.update(m)

    # Variance over baseline seeds
    baseline_seeds = [
        r for r in all_results if r["code"] == "000" and r.get("test_auc") is not None
    ]
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
        "wf_tag": "Wf-240 — Wf-229 EXTENSION with 50 epochs + all 11 configs",
        "wf_extends": "Wf-229 (TODO-182 per-innovation ablation)",
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
        "honest_framing": {
            "extends_wf229": "Wf-229 only completed 2/11 configs at 5 epochs (K-only lift=0 within noise).",
            "extension_purpose": "Verify at 50 epochs whether the K/D/F levers actually lift beyond noise.",
            "epochs_protocol": "50 epochs per config (vs Wf-229's 5; vs Wf-124 P3's 30 with h=128 d=3).",
            "comparability": "Within-cohort only — NOT comparable to absolute Wf-124 P3 anchor.",
            "negative_documented": True,
            "lever_lift_within_noise": "All levers that fall within ±stdev(baseline_seeds) are reported as 'within noise' rather than 'no effect'.",
        },
    }

    with open(OUT_DIR / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[wf240] wrote {OUT_DIR/'ablation_summary.json'}", flush=True)

    # Human-readable synthesis
    syn = OUT_DIR / "synthesis.md"
    with open(syn, "w") as f:
        f.write("# Wf-240 per-innovation ablation — EXTENSION of Wf-229\n\n")
        f.write(
            "**Purpose**: Wf-229 only completed 2/11 configs at 5 epochs and showed K-only lift=0 within noise. "
            "This extension re-runs ALL configs at 50 epochs for a MEASURED signal.\n\n"
        )
        f.write("3 levers: K (Kendall loss), D (DANN), F (four-domain)\n")
        f.write(
            f"Total runs: {len(all_results)} (8 unique configs @ seed=0 + {len(extra_seeds)} baseline seeds)\n"
        )
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
        if "single_lever_lift" in attribution:
            for k, v in attribution.get("single_lever_lift", {}).items():
                f.write(f"- **{k}** (single-lever lift): {v:+.4f}\n")
        f.write("\n## Two-way interactions\n\n")
        if "two_way_interactions" in attribution:
            for k, v in attribution.get("two_way_interactions", {}).items():
                f.write(f"- **{k}**: {v:+.4f}\n")
        if "full_recipe_lift" in attribution:
            f.write(
                f"\n## Full recipe (111) lift: {attribution.get('full_recipe_lift', 0):+.4f}\n"
            )
        f.write(f"\n## Baseline variance (seed 0..{max(args.seeds)}):\n\n")
        f.write(f"- AUCs: {baseline_aucs}\n")
        f.write(f"- Mean: {baseline_mean}\n")
        f.write(f"- Stdev: {baseline_std}\n")
        f.write("\n## Honest framing (party-twenty-spirit §3.5)\n\n")
        f.write(
            "- **Wf-229 result was 'K-only lift=0 within noise' at 5 epochs.** This extension is the wall-time test of whether that negative is real (no lever effect) or just slow-convergence (levers need more epochs to lift).\n"
        )
        f.write(
            "- 50 epochs (vs Wf-229's 5; vs Wf-124 P3's 30 with h=128 d=3) — within-cohort attribution only; NOT comparable to absolute Wf-124 P3 anchor at +0.049.\n"
        )
        f.write(
            "- Per-lever attribution uses seed=0 only; baseline seeds 1/2/3 reported as variance bars.\n"
        )
        f.write(
            "- **Honest-negative rule**: any lever whose lift falls within ±stdev(baseline_seeds) is reported as 'within noise' rather than 'no effect'.\n"
        )

    print(f"[wf240] wrote {syn}", flush=True)
    print(f"[wf240] DONE  total_wall={total_wall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
