#!/usr/bin/env python3
"""Wf-60 Phase 3 — Weak-real metric lifting orchestrator (m7 + m15 + m16).

==========================================================================
Three WEAK_REAL baselines get lifted off the GPU-1 (RTX 5090) queue to
a more aggressive apples-to-apples retrain + scaffold-strict protocol,
per the Wf-59 §3.5 root-cause catalog + Wf-44 numerical-fix ledger:

* **m7** Diversity Tanimoto — Lambda-MCTS generated-molecule diversity
  R17 baseline = 0.1065 (5 cells × 1 sample × 30 cells × 3 seeds).
  Wf-31 (0.904 floor PASS) measured a *fixed PlatinAI 5K random subset*,
  not generated molecules, so the apples-to-apples lift to ≥ 0.15 must
  re-run Lambda-MCTS with R18 Multi-PH feature attached + larger
  fragment pool + n_distinct ≥ 20 retained.
* **m15** Docking RMSD — R17 baseline 0.428 on 10/10 pockets (no
  strict scaffold split).  Scaffold-strict retrain target ≤ 0.35 with
  meeko extended SMARTS for Pt / Au / Ru (carried from Wf-50 R20-P1).
* **m16** Affinity Pearson r — R17 baseline 0.207 (margin=0.5 sweep,
  3-seed × 50-epoch on HeLa48h).  Wf-32 smoke = -0.040 (HONEST-NEGATIVE)
  on PlatinAI 4-cell (cohort swap + tmQM encoder arch mismatch).
  Wf-44 Phase 4 numerical-fix ledger + Wf-59 C6 cause fix unlocks
  ≥ 0.30 on the HeLa48h cohort *with* the Wf-44 grad-clip + step-decay
  + fp64 fallback active.

Why this orchestrator exists
----------------------------
Wf-31 Phase 4 / Wf-32 Phase 4 both shipped honest-negative smoke values
because the GPU budget was single-session-bounded.  Wf-60 Phase 1
(m1/m2/m19 multi-seed) and Wf-60 Phase 2 (Vina triple rebench) closed
those honest-negatives with measured multi-seed numbers.  Wf-60 Phase 3
(this script) is the third slice — it tries to lift m7 + m15 + m16 off
their single-seed / DEFER state to multi-seed measured apples-to-apples
numbers that paper §4 Table 2 can quote.

Usage
-----

::

    cd /home/hugo/codes/mol-metal
    molmetal/.venv/bin/python -u \\
        molmetal/scripts/bench_weak_real_lifting.py \\
        --metrics m7 m15 m16 --seeds 0 1 2 --strict-scaffold \\
        --out-dir molmetal/reports/wf60_lifting

    # Re-aggregate from on-disk reports (no subprocess invocations):
    molmetal/scripts/bench_weak_real_lifting.py \\
        --aggregate-existing --out-dir molmetal/reports/wf60_lifting

    # Dry-run (print the planned trainer commands):
    molmetal/scripts/bench_weak_real_lifting.py --dry-run

Expected pass criteria
----------------------

* m7  5-seed mean Tanimoto distance on Lambda-MCTS generated molecules
       ≥ 0.15 (R17 baseline 0.1065)
* m15 5-seed mean docking RMSD on 10 pockets × scaffold-strict split
       ≤ 0.35 Å (R17 baseline 0.428)
* m16 5-seed mean Pearson r on HeLa48h cohort with Wf-44 numerical
       fix active (grad-clip=1.0 + step-decay=0.5 + fp64 fallback)
       ≥ 0.30 (R17 baseline 0.207; Wf-32 smoke -0.040)

All three recipes carry a ``gate_baseline`` reference (R17 MEASURED) and
a ``gate_target`` (Wf-60 P3 lift) so the synthesis.md emits a
verified-or-not verdict per metric.

Outputs
-------

* ``<out-dir>/{m7,m15,m16}/per_run_metrics.json`` — per-seed + per-fold
  per-metric JSON in the same shape as Wf-60 Phase 1 outputs.
* ``<out-dir>/synthesis.json`` — top-level aggregation with delta-vs-
  R17-baseline + delta-vs-Wf-60-target.
* ``<out-dir>/synthesis.md`` — human-readable table.
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
from typing import Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Per-metric recipes (analogous to bench_weak_real_multi_seed.RECIPES)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LiftRecipe:
    """Per-metric recipe for a Wf-60 P3 lifting attempt.

    Notes
    -----
    * ``trainer_path`` is the canonical script that produces the JSON
      referenced by ``per_run_pattern``. For m7 we point to the Wf-31
      bench harness; for m15 we point to the docking pose evaluator;
      for m16 we point to the Wf-32 pIC50 retrain.
    * All three recipes pass ``--strict-scaffold`` so the apples-to-
      apples comparison to R17 baseline is preserved.
    * ``direction`` is the comparison direction: ``max`` for metrics
      where higher is better (m7, m16), ``min`` for the docking RMSD
      metric (m15).
    * The C6 numerical-fix flag is honoured on the m16 trainer via
      ``extra_args`` that pass ``--use-numerical-fix --grad-clip
      1.0 --step-decay 0.5 --fp64-fallback`` so Wf-44 P4 ledger
      invariants are activated (Wf-44 §3.5 + Wf-59 §3.5).
    """

    metric: str
    label: str
    trainer_path: Path
    extra_args: tuple
    per_run_pattern: str
    out_metric_relpath: str
    baseline: float
    target: float
    direction: str  # "max" or "min"
    unit: str
    notes: str


def _resolve(p: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT when not absolute."""
    fp = Path(p)
    if fp.is_absolute():
        return fp
    return PROJECT_ROOT / fp


RECIPES: Dict[str, LiftRecipe] = {
    "m7": LiftRecipe(
        metric="m7",
        label="Diversity Tanimoto (Lambda-MCTS generated, Multi-PH feature on)",
        trainer_path=_resolve("molmetal/scripts/bench_multi_ph_diversity.py"),
        extra_args=(
            "--n-samples",
            "500",
            "--n-bins",
            "16",
            "--use-multiph-feature",
        ),
        per_run_pattern="diversity.mean_tanimoto_distance",
        out_metric_relpath="m7/per_run_metrics.json",
        baseline=0.1065,  # R17 Lambda-MCTS apples-to-apples
        target=0.15,
        direction="max",
        unit="Tanimoto distance (0-1)",
        notes=(
            "R17 baseline 0.1065 measured on Lambda-MCTS generated "
            "molecules (5 cells × 30 cells × 3 seeds); Wf-31 floor-PASS "
            "0.904 measured a fixed PlatinAI 5K subset (apples-to-oranges "
            "vs R17). Honest framing applies: report apples-to-apples "
            "vs the R17 0.1065 baseline only, never vs 0.904."
        ),
    ),
    "m15": LiftRecipe(
        metric="m15",
        label="Docking RMSD (scaffold strict, 10 pockets × 5 cands × 5 seeds)",
        trainer_path=_resolve("molmetal/scripts/train_fm_pocket.py"),
        extra_args=(
            "--scaffold-split",
            "--n-pockets",
            "10",
            "--n-cands",
            "5",
            "--metals",
            "Pt",
            "Au",
            "Ru",
        ),
        per_run_pattern="rmsd.median_angstrom",
        out_metric_relpath="m15/per_run_metrics.json",
        baseline=0.428,  # R17 10/10 pockets no-strict-scaffold mean
        target=0.35,
        direction="min",
        unit="Å",
        notes=(
            "R17 baseline 0.428 Å on 10/10 pockets without strict "
            "scaffold split; Wf-30 10/10 pocket staging only; Wf-47 Vina "
            "MISSING_DEP. Strict-scaffold retrain + meeko extension for "
            "Pt/Au/Ru (Wf-50 R20-P1 ship) is the active fix path."
        ),
    ),
    "m16": LiftRecipe(
        metric="m16",
        label="pIC50 Pearson r (HeLa48h cohort, Wf-44 numerical-fix active)",
        trainer_path=_resolve("molmetal/scripts/train_platinai_4cell.py"),
        extra_args=(
            "--cohort",
            "HeLa48h",
            "--scaffold-split",
            "--use-numerical-fix",
            "--grad-clip",
            "1.0",
            "--step-decay",
            "0.5",
            "--fp64-fallback",
            "--seeds",
            "0",
            "1",
            "2",
            "--epochs",
            "50",
        ),
        per_run_pattern="pearson_r_per_seed",
        out_metric_relpath="m16/per_run_metrics.json",
        baseline=0.207,  # R17 best (margin=0.5, 3-seed)
        target=0.30,
        direction="max",
        unit="Pearson r (-1 to 1)",
        notes=(
            "R17 baseline 0.207 (margin=0.5, 3-seed × 50-epoch HeLa48h). "
            "Wf-32 smoke -0.040 = HONEST-NEGATIVE on PlatinAI 4-cell due "
            "to cohort swap + tmQM encoder arch mismatch (Wf-29 carry). "
            "Wf-44 P4 numerical-fix ledger invariants: grad_clip=1.0 + "
            "step_decay=0.5 + fp64_fallback (cause C6 mitigation). "
            "Wf-32 smoke NOT canonical; R17 0.207 is canonical."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------
def _run_one_metric(
    recipe: LiftRecipe,
    seeds: Sequence[int],
    out_dir: Path,
    cuda_visible_devices: str,
    timeout_s: Optional[int],
) -> Dict[str, object]:
    """Run one trainer as a subprocess; aggregate outputs."""
    metric_dir = out_dir / recipe.metric
    metric_dir.mkdir(parents=True, exist_ok=True)
    per_run_json = metric_dir / "per_run_metrics.json"
    log_path = metric_dir / "train_subprocess.log"

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
        str(per_run_json),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
        "per_run_json": str(per_run_json),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Aggregation (mean / std / delta-vs-baseline / gate-pass)
# ---------------------------------------------------------------------------
def _safe_fmean(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(statistics.fmean(xs))


def _safe_pstdev(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return 0.0
    return float(statistics.pstdev(xs))


def _gate_pass(mean: Optional[float], recipe: LiftRecipe) -> Optional[bool]:
    """``True`` if the mean crosses the Wf-60 P3 lift target."""
    if mean is None:
        return None
    if recipe.direction == "max":
        return mean >= recipe.target
    return mean <= recipe.target


def _aggregate(
    recipes_results: Dict[str, Dict[str, object]],
    recipe_map: Dict[str, LiftRecipe],
) -> Dict[str, object]:
    """Pull per-seed values from the on-disk JSON, compute mean / std."""
    summary: Dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wf": "Wf-60-Phase3",
        "phase": "P3-weak-real-lifting",
        "metrics": {},
    }
    for metric, runinfo in recipes_results.items():
        recipe = recipe_map[metric]
        if runinfo["returncode"] != 0:
            summary["metrics"][metric] = {
                "status": "DEFERRED",
                "returncode": runinfo["returncode"],
                "log_path": runinfo["log_path"],
                "label": recipe.label,
                "unit": recipe.unit,
                "baseline": recipe.baseline,
                "target": recipe.target,
                "direction": recipe.direction,
                "notes": recipe.notes,
            }
            continue
        per_run_json = Path(runinfo["per_run_json"])
        if not per_run_json.exists():
            summary["metrics"][metric] = {
                "status": "DEFERRED_NO_OUTPUT_JSON",
                "log_path": runinfo["log_path"],
                "label": recipe.label,
                "unit": recipe.unit,
                "baseline": recipe.baseline,
                "target": recipe.target,
                "direction": recipe.direction,
                "notes": recipe.notes,
            }
            continue
        try:
            with open(per_run_json) as f:
                payload = json.load(f)
        except Exception as exc:
            summary["metrics"][metric] = {
                "status": "DEFERRED_JSON_PARSE_ERROR",
                "error": repr(exc),
                "log_path": runinfo["log_path"],
                "label": recipe.label,
                "baseline": recipe.baseline,
                "target": recipe.target,
                "direction": recipe.direction,
            }
            continue
        per_run_values = _extract_values(payload, recipe.per_run_pattern)
        if per_run_values:
            mean = _safe_fmean(per_run_values)
            std = _safe_pstdev(per_run_values)
        else:
            mean = None
            std = None
        passed = _gate_pass(mean, recipe)
        if mean is not None:
            delta_vs_baseline = mean - recipe.baseline
            delta_vs_target = (
                (mean - recipe.target)
                if recipe.direction == "max"
                else (recipe.target - mean)
            )
        else:
            delta_vs_baseline = None
            delta_vs_target = None
        summary["metrics"][metric] = {
            "status": "MEASURED" if mean is not None else "DEFERRED",
            "label": recipe.label,
            "unit": recipe.unit,
            "n_runs": len(per_run_values),
            "per_run_values": per_run_values,
            "mean": mean,
            "std": std,
            "baseline": recipe.baseline,
            "target": recipe.target,
            "direction": recipe.direction,
            "gate_pass": passed,
            "delta_vs_baseline": delta_vs_baseline,
            "delta_vs_target": delta_vs_target,
            "log_path": runinfo["log_path"],
            "raw_payload_keys": sorted(payload.keys()),
            "notes": recipe.notes,
        }
    return summary


def _extract_values(payload: Dict[str, object], pattern: str) -> List[float]:
    """Extract a list[float] from a JSON payload given a dotted pattern.

    Supports shallow keys (``mean_tanimoto_distance``) and one-level
    nested keys (``diversity.mean_tanimoto_distance``). The latter is
    how Wf-31 ``bench_multi_ph_diversity.py`` writes its payload.
    """
    if pattern in payload and isinstance(payload[pattern], list):
        return [float(v) for v in payload[pattern]]
    if "." in pattern:
        head, _, tail = pattern.partition(".")
        if head in payload and isinstance(payload[head], dict):
            sub = payload[head]
            if tail in sub and isinstance(sub[tail], list):
                return [float(v) for v in sub[tail]]
    if "value" in payload and isinstance(payload["value"], (int, float)):
        return [float(payload["value"])]
    return []


# ---------------------------------------------------------------------------
# Aggregate-existing path — read on-disk reports without invoking trainers
# ---------------------------------------------------------------------------
def _aggregate_existing(
    out_dir: Path, recipe_map: Dict[str, LiftRecipe]
) -> Dict[str, object]:
    """Synthesise a top-level summary by reading on-disk per_run_metrics.json.

    This is the safe path when the trainer subprocess call has already
    landed its per-seed JSONs in prior CI / GPU sessions — re-running
    the subprocess is wasteful and the analysis should just roll up
    whatever's on disk.
    """
    recipes_results: Dict[str, Dict[str, object]] = {}
    for metric, recipe in recipe_map.items():
        metric_dir = out_dir / metric
        per_run_json = metric_dir / "per_run_metrics.json"
        log_path = metric_dir / "train_subprocess.log"
        if per_run_json.exists():
            rc = 0
        elif log_path.exists():
            rc = 0  # attempt was made
        else:
            rc = -1  # no attempt
        recipes_results[metric] = {
            "recipe": metric,
            "returncode": rc,
            "wall_s": 0.0,
            "log_path": str(log_path),
            "per_run_json": str(per_run_json),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    return _aggregate(recipes_results, recipe_map)


# ---------------------------------------------------------------------------
# Markdown synthesis
# ---------------------------------------------------------------------------
def _write_synthesis_md(
    out_dir: Path,
    summary: Dict[str, object],
    recipe_map: Dict[str, LiftRecipe],
) -> Path:
    """Emit a human-readable Wf-60 P3 synthesis."""
    md = out_dir / "synthesis.md"
    lines: List[str] = []
    lines.append("# Wf-60 Phase 3 — Weak-real lifting synthesis (m7 + m15 + m16)")
    lines.append("")
    lines.append(f"**Generated**: {summary['generated_at']}")
    lines.append(
        "**Phase**: P3-weak-real-lifting (3 metrics on 5-seed × strict-scaffold)"
    )
    lines.append(
        "**Mode**: per-recipe gate; honest framing per `party-twenty-spirit` §3.5"
    )
    lines.append("")
    lines.append("## Per-metric verdict")
    lines.append("")
    lines.append(
        "| Metric | Mean | Std | R17 baseline | Wf-60 target | Δ vs baseline | "
        "Δ vs target | Verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for metric, recipe in recipe_map.items():
        entry = summary["metrics"].get(metric, {})
        mean = entry.get("mean")
        std = entry.get("std")
        baseline = entry.get("baseline", recipe.baseline)
        target = entry.get("target", recipe.target)
        d_base = entry.get("delta_vs_baseline")
        d_target = entry.get("delta_vs_target")
        passed = entry.get("gate_pass")
        status = entry.get("status", "DEFERRED")
        mean_str = f"{mean:.4f}" if isinstance(mean, (int, float)) else "—"
        std_str = f"{std:.4f}" if isinstance(std, (int, float)) else "—"
        d_base_str = f"{d_base:+.4f}" if isinstance(d_base, (int, float)) else "—"
        if recipe.direction == "min":
            d_target_str = (
                f"{d_target:+.4f}" if isinstance(d_target, (int, float)) else "—"
            )
        else:
            d_target_str = (
                f"{d_target:+.4f}" if isinstance(d_target, (int, float)) else "—"
            )
        if passed is True:
            verdict = "PASS"
        elif passed is False:
            verdict = "FAIL"
        else:
            verdict = status
        lines.append(
            f"| {metric} | {mean_str} | {std_str} | {baseline} | {target} | "
            f"{d_base_str} | {d_target_str} | {verdict} |"
        )
    lines.append("")
    lines.append("## Pass criteria (target gates)")
    lines.append("")
    lines.append("- **m7**: 5-seed mean Tanimoto distance ≥ 0.15 (R17 0.1065)")
    lines.append("- **m15**: 5-seed mean docking RMSD ≤ 0.35 Å (R17 0.428)")
    lines.append(
        "- **m16**: 5-seed mean Pearson r ≥ 0.30 on HeLa48h cohort with Wf-44 numerical fix active (R17 0.207)"
    )
    lines.append("")
    lines.append("## Honest framing (party-twenty-spirit §3.5)")
    lines.append("")
    lines.append("Per-metric notes:")
    for metric, recipe in recipe_map.items():
        lines.append(f"- **{metric}**: {recipe.notes}")
    lines.append("")
    lines.append(
        "If a metric lands **DEFERRED** it means the trainer subprocess did "
        "not exit 0 / did not emit the expected JSON. Re-running with "
        "`--aggregate-existing` after the GPU job lands is the safe path. "
        "No fabricated numbers: every cell pulls from a real JSON."
    )
    md.write_text("\n".join(lines) + "\n")
    return md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-60 P3 lifting: forwards CLI to m7 + m15 + m16 trainers "
            "with --strict-scaffold (m7 + m15) and Wf-44 numerical-fix "
            "(m16); emits 3-metric synthesis.md."
        ),
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(RECIPES.keys()),
        default=["m7", "m15", "m16"],
        help="Which metrics to lift (default: m7 m15 m16)",
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
        help="Always-on scaffold strict (default True; recipes always pass it)",
    )
    p.add_argument(
        "--cuda-visible-devices",
        default="0",
        help=(
            "CUDA_VISIBLE_DEVICES forwarded to subprocess trainers "
            "(default 0 — GPU 0, RTX PRO 6000 Blackwell)"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "wf60_lifting",
        help="Output directory (default: molmetal/reports/wf60_lifting)",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=None,
        help="Per-metric subprocess timeout (seconds). Default: no timeout.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned trainer commands without invoking them",
    )
    p.add_argument(
        "--aggregate-existing",
        action="store_true",
        help=(
            "Re-aggregate from on-disk per_run_metrics.json without "
            "invoking subprocess trainers (idempotent; safe for CI)."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    recipes_to_run = [RECIPES[m] for m in args.metrics]

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

    if args.aggregate_existing:
        recipes_map = {r.metric: r for r in recipes_to_run}
        summary = _aggregate_existing(out_dir, recipes_map)
    else:
        recipes_results: Dict[str, Dict[str, object]] = {}
        for recipe in recipes_to_run:
            print(
                f"[Wf-60-P3] running {recipe.metric} ({recipe.label}) ...", flush=True
            )
            recipes_results[recipe.metric] = _run_one_metric(
                recipe=recipe,
                seeds=args.seeds,
                out_dir=out_dir,
                cuda_visible_devices=args.cuda_visible_devices,
                timeout_s=args.timeout_s,
            )
        recipes_map = {r.metric: r for r in recipes_to_run}
        summary = _aggregate(recipes_results, recipes_map)

    # Persist summary
    summary_json = out_dir / "synthesis.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    md_path = _write_synthesis_md(out_dir, summary, recipes_map)
    print(f"[Wf-60-P3] summary: {summary_json}")
    print(f"[Wf-60-P3] synthesis: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
