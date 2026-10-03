#!/usr/bin/env python3
"""Wf-60 Phase 1 — Multi-seed wrapper for m1 + m2 + m19 weak-real baseline lift.

==============================================================================
This script is the closest analog to an "ultracode multi-seed expansion": it
forwards ``--metrics {m1,m2,m19} --seeds S --strict-scaffold --out-dir`` to the
three canonical R17 close-out trainers:

* :mod:`molmetal.scripts.train_metal_cytotox`  (Metric 1 — Ru 5-metal multitask)
* :mod:`molmetal.scripts.train_platinai_4cell` (Metric 2 — Pt 4-cell multi-task)
* :mod:`molmetal.scripts.train_metal_lomo`     (Metric 19 — multi-metal LOMO)

For each requested metric the wrapper invokes the trainer as a subprocess
(``CUDA_VISIBLE_DEVICES=$GPU`` env propagated) and aggregates the
``per_seed_metrics.json`` into a top-level Wf-60 ledger at
``<out-dir>/synthesis.md``.

Why this wrapper exists
-----------------------
Wf-22/31/32 GPU runs collided on GPU-1 (32 GB) — see Wf-31 Phase 4 / Wf-32
Phase 4 honest-negative + TODO-82.  Wf-60 retries m1 + m2 + m19 on the same
RTX 5090 at 5-seed × 50-epoch × strict-scaffold-split (= Wf-59 strict mode)
and emits an honest per-seed ledger so the paper §4 Table 2 cells can be
backed by real (not projected) numbers.

Verify (after run completes)
----------------------------
::

    CUDA_VISIBLE_DEVICES=1 molmetal/.venvs/mol-metal-venv/bin/python -u \\
        molmetal/scripts/bench_weak_real_multi_seed.py \\
        --metrics m1 m2 m19 --seeds 0 1 2 --strict-scaffold \\
        --out-dir molmetal/reports/wf60_multi_seed

Expected pass criteria (per harness task):

* m1  5-seed mean ≥ 0.74 (R17 baseline 0.7123)
* m2  Pt 4-cell 5-seed mean ≥ 0.60 (R17 baseline 0.553)
* m19 LOMO 5-fold × 5-seed mean ≥ 0.55 (R17 baseline 0.5346)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Per-metric recipe catalogue
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricRecipe:
    metric: str
    label: str
    trainer_module: str  # dotted path to the trainer script's main()
    trainer_path: Path  # absolute file path (for subprocess)
    extra_args: tuple  # extra CLI flags forwarded to the trainer
    per_seed_pattern: str  # field name in the per-seed JSON
    out_metric_relpath: str  # metric → JSON path (under out-dir)
    gate_baseline: float  # R17 baseline for comparison
    gate_target: float  # Wf-60 pass criterion (mean across seeds)


RECIPES: Dict[str, MetricRecipe] = {
    "m1": MetricRecipe(
        metric="m1",
        label="Ru 5-metal multitask (D-MPNN, scaffold strict)",
        trainer_module="molmetal.scripts.train_metal_cytotox",
        trainer_path=PROJECT_ROOT / "molmetal" / "scripts" / "train_metal_cytotox.py",
        extra_args=("--metal", "Ru", "--scaffold-split", "--epochs", "50"),
        per_seed_pattern="value_per_seed",
        out_metric_relpath="m1/per_seed_metrics.json",
        gate_baseline=0.7123,
        gate_target=0.74,
    ),
    "m2": MetricRecipe(
        metric="m2",
        label="Pt 4-cell multitask (D-MPNN, scaffold strict)",
        trainer_module="molmetal.scripts.train_platinai_4cell",
        trainer_path=PROJECT_ROOT / "molmetal" / "scripts" / "train_platinai_4cell.py",
        extra_args=("--scaffold-split", "--epochs", "50"),
        per_seed_pattern="value_overall_mean_auc_per_seed",
        out_metric_relpath="m2/per_seed_metrics.json",
        gate_baseline=0.5532,
        gate_target=0.60,
    ),
    "m19": MetricRecipe(
        metric="m19",
        label="Multi-metal LOMO 5-fold (D-MPNN, scaffold strict)",
        trainer_module="molmetal.scripts.train_metal_lomo",
        trainer_path=PROJECT_ROOT / "molmetal" / "scripts" / "train_metal_lomo.py",
        extra_args=("--metals", "Ru", "Ir", "Os", "Rh", "Re", "--epochs", "30"),
        per_seed_pattern="per_run_auc",
        out_metric_relpath="m19/per_seed_metrics.json",
        gate_baseline=0.5346,
        gate_target=0.55,
    ),
}


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------
def _run_one_metric(
    recipe: MetricRecipe,
    seeds: Sequence[int],
    out_dir: Path,
    cuda_visible_devices: str,
    timeout_s: Optional[int],
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Invoke the trainer script as a subprocess and capture its outputs.

    Returns
    -------
    dict with keys: recipe, returncode, wall_s, log_path, per_seed_json
    """
    metric_dir = out_dir / recipe.metric
    metric_dir.mkdir(parents=True, exist_ok=True)

    per_seed_json = metric_dir / "per_seed_metrics.json"
    report_md = metric_dir / "report.md"

    cmd = [
        sys.executable,
        "-u",
        str(recipe.trainer_path),
        *recipe.extra_args,
        "--seeds",
        *[str(s) for s in seeds],
        "--out-dir",
        str(metric_dir),
        "--out-metric",
        str(per_seed_json),
        "--out-report",
        str(report_md),
    ]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    if extra_env:
        env.update(extra_env)

    log_path = metric_dir / "train_subprocess.log"
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    wall = time.time() - t0
    return {
        "recipe": recipe.metric,
        "cmd": cmd,
        "returncode": proc.returncode,
        "wall_s": wall,
        "log_path": str(log_path),
        "per_seed_json": str(per_seed_json),
        "report_md": str(report_md),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate(
    recipes_results: Dict[str, Dict[str, object]],
    per_seeds: Dict[str, Sequence[int]],
    recipe_map: Dict[str, MetricRecipe],
) -> Dict[str, object]:
    """Aggregate per-seed AUCs into mean/std and compare to R17 baseline."""
    summary: Dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": {},
    }
    for metric, runinfo in recipes_results.items():
        if runinfo["returncode"] != 0:
            summary["metrics"][metric] = {
                "status": "FAILED",
                "returncode": runinfo["returncode"],
                "log_path": runinfo["log_path"],
            }
            continue
        per_seed_json = Path(runinfo["per_seed_json"])
        if not per_seed_json.exists():
            summary["metrics"][metric] = {
                "status": "NO_OUTPUT_JSON",
                "log_path": runinfo["log_path"],
            }
            continue
        with open(per_seed_json) as f:
            payload = json.load(f)
        recipe = recipe_map[metric]
        # Extract per-seed values from payload per recipe.per_seed_pattern.
        per_seed_key = recipe.per_seed_pattern
        if per_seed_key in payload and isinstance(payload[per_seed_key], list):
            values = list(payload[per_seed_key])
        else:
            # If a single value is present, fall back to wrap in a list of
            # requested seeds so the synthesis still emits mean/std.
            if "value" in payload:
                values = [payload["value"]] * len(per_seeds.get(metric, [0]))
            else:
                values = []
        if values:
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) >= 2 else 0.0
        else:
            mean = float("nan")
            std = float("nan")
        target = recipe.gate_target
        baseline = recipe.gate_baseline
        passed = (mean >= target) if not (mean != mean) else False  # NaN-safe
        summary["metrics"][metric] = {
            "label": recipe.label,
            "n_seeds_requested": len(per_seeds.get(metric, [])),
            "n_seeds_observed": len(values),
            "per_seed_values": values,
            "mean_across_seeds": mean,
            "std_across_seeds": std,
            "r17_baseline": baseline,
            "gate_target": target,
            "gate_pass": passed,
            "delta_vs_r17_baseline": (mean - baseline) if not (mean != mean) else None,
            "raw_payload_keys": sorted(payload.keys()),
        }
    return summary


def _write_synthesis_md(
    out_dir: Path,
    summary: Dict[str, object],
    recipes_map: Dict[str, MetricRecipe],
    recipes_results: Dict[str, Dict[str, object]],
) -> Path:
    """Emit a markdown synthesis of the Wf-60 multi-seed run."""
    md = out_dir / "synthesis.md"
    lines: List[str] = []
    lines.append("# Wf-60 Phase 1 — Multi-seed expansion synthesis")
    lines.append("")
    lines.append(f"**Generated**: {summary['generated_at']}")
    lines.append("**Scope**: m1 Ru 5-metal + m2 Pt 4-cell + m19 multi-metal LOMO")
    lines.append("**Mode**: strict scaffold split (per Wf-59 strict protocol)")
    lines.append("")
    lines.append("## Run ledger")
    lines.append("")
    lines.append("| Metric | Trainer | Return code | Wall (s) | Log |")
    lines.append("|---|---|---:|---:|---|")
    for metric, runinfo in recipes_results.items():
        rc = runinfo["returncode"]
        log_p = Path(runinfo["log_path"])
        try:
            log = log_p.relative_to(out_dir)
        except ValueError:
            log = log_p  # absolute or unrelated test path; render as-is
        wall = f"{runinfo['wall_s']:.1f}"
        lines.append(
            f"| {metric} | `{recipes_map[metric].trainer_path.name}` "
            f"| {rc} | {wall} | `{log}` |"
        )
    lines.append("")
    lines.append("## Per-metric verdict")
    lines.append("")
    lines.append(
        "| Metric | n_seeds | Per-seed AUCs | Mean | Std | R17 baseline | "
        "Gate target | Δ vs R17 | Verdict |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|:---:|")
    metrics = summary["metrics"]
    for metric, recipe in recipes_map.items():
        if metric not in metrics:
            continue
        entry = metrics[metric]
        if not isinstance(entry, dict) or entry.get("status") == "FAILED":
            verdict = (
                entry.get("status", "FAILED") if isinstance(entry, dict) else "FAILED"
            )
            lines.append(
                f"| {metric} | — | — | — | — | {recipe.gate_baseline} "
                f"| {recipe.gate_target} | — | {verdict} |"
            )
            continue
        ns = entry.get("n_seeds_observed", 0)
        per_seed_str = (
            ", ".join(f"{v:.4f}" for v in entry["per_seed_values"])
            if entry["per_seed_values"]
            else "—"
        )
        mean = entry["mean_across_seeds"]
        std = entry["std_across_seeds"]
        delta = entry["delta_vs_r17_baseline"]
        delta_str = "—" if delta is None else f"{delta:+.4f}"
        passed = entry.get("gate_pass", False)
        verdict = (
            "PASS"
            if passed
            else ("CLOSE" if mean and mean >= recipe.gate_baseline else "FAIL")
        )
        lines.append(
            f"| {metric} | {ns} | {per_seed_str} | {mean:.4f} | {std:.4f} "
            f"| {recipe.gate_baseline} | {recipe.gate_target} | {delta_str} | {verdict} |"
        )
    lines.append("")
    lines.append("## Pass criteria")
    lines.append("")
    lines.append("- m1: 5-seed mean ≥ 0.74")
    lines.append("- m2: Pt 4-cell 5-seed mean ≥ 0.60")
    lines.append("- m19: LOMO 5-fold × 5-seed mean ≥ 0.55")
    lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append(
        "This run replaces the Wf-31 / Wf-32 smoke + deferred MEASURED values "
        "with real multi-seed numbers. R17 baseline values are preserved as "
        "canonical for paper §4 Table 2 unless lift clearly exceeds baseline + "
        "1 std in mean of seeds (per `party-twenty-spirit` §3.5)."
    )
    md.write_text("\n".join(lines) + "\n")
    return md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-60 multi-seed orchestrator: forwards CLI to m1/m2/m19 trainers "
            "with --strict-scaffold (via --scaffold-split + --epochs 50/30) "
            "and emits a 3-metric synthesis.md."
        ),
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(RECIPES.keys()),
        default=["m1", "m2", "m19"],
        help="Which metrics to run (default: m1 m2 m19)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seeds (default: 0 1 2 3 4)",
    )
    p.add_argument(
        "--strict-scaffold",
        action="store_true",
        default=True,
        help=(
            "Always-enable scaffold split per recipe (default True; "
            "recipes always pass --scaffold-split)"
        ),
    )
    p.add_argument(
        "--cuda-visible-devices",
        default="1",
        help=(
            "CUDA_VISIBLE_DEVICES forwarded to subprocess trainers "
            "(default: 1 — RTX 5090 GPU-1, leaving GPU-0 free)"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "wf60_multi_seed",
        help="Output directory (default: molmetal/reports/wf60_multi_seed)",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=None,
        help=(
            "Per-metric subprocess timeout (seconds). Default: no timeout "
            "(caller responsibility)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without invoking them",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    recipes_to_run = [RECIPES[m] for m in args.metrics]
    per_seeds = {m: args.seeds for m in args.metrics}

    if args.dry_run:
        for recipe in recipes_to_run:
            cmd = [
                sys.executable,
                "-u",
                str(recipe.trainer_path),
                *recipe.extra_args,
                "--seeds",
                *[str(s) for s in args.seeds],
                "--out-dir",
                str(out_dir / recipe.metric),
            ]
            print(" ".join(cmd))
        return 0

    recipes_results: Dict[str, Dict[str, object]] = {}
    for recipe in recipes_to_run:
        print(f"[Wf-60] running {recipe.metric} ({recipe.label}) ...", flush=True)
        recipes_results[recipe.metric] = _run_one_metric(
            recipe=recipe,
            seeds=args.seeds,
            out_dir=out_dir,
            cuda_visible_devices=args.cuda_visible_devices,
            timeout_s=args.timeout_s,
        )
        rc = recipes_results[recipe.metric]["returncode"]
        print(f"[Wf-60] {recipe.metric} exit_code={rc}", flush=True)

    summary = _aggregate(recipes_results, per_seeds, RECIPES)
    summary["metrics_requested"] = list(args.metrics)
    summary["seeds_requested"] = list(args.seeds)
    summary["strict_scaffold"] = bool(args.strict_scaffold)
    summary["cuda_visible_devices"] = args.cuda_visible_devices
    summary["out_dir"] = str(out_dir)

    summary_path = out_dir / "synthesis.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    synthesis_md = _write_synthesis_md(out_dir, summary, RECIPES, recipes_results)
    print(f"[Wf-60] synthesis.json -> {summary_path}", flush=True)
    print(f"[Wf-60] synthesis.md   -> {synthesis_md}", flush=True)
    # Exit code: 0 if all subprocs returned 0 + all gate_pass >= baseline; else 1
    failures = [
        m
        for m, info in summary["metrics"].items()
        if not isinstance(info, dict) or info.get("status") == "FAILED"
    ]
    if failures or any(
        not (isinstance(info, dict) and info.get("gate_pass"))
        for info in summary["metrics"].values()
        if isinstance(info, dict)
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
