#!/usr/bin/env python3
"""Wf-29 Phase 2 — side-by-side summary of the 4-recipe smoke benchmark.

Reads the 4 per-recipe ``metrics.json`` files written by
``train_se3sb.py`` / ``train_transport.py`` / ``train_multiph.py`` /
``train_dmpnn_multitask.py`` (the Wf-29 Phase 2 smoke benchmark on
GPU 1) and consolidates them into:

* ``molmetal/reports/wf29_smoke/bench_summary.json`` — slim rollup
  table (per-recipe mean / std / per-seed AUC + wall-seconds).
* ``molmetal/reports/wf29_smoke/comparison.md`` — markdown
  side-by-side: se3sb vs transport vs multiph vs R17-baseline
  (per-metal + headline) with honest-negative provenance tags.

This script is **read-only / opt-in**: it does NOT touch the Wf-29
trainer scripts.  It is the Wf-29 Phase 2 equivalent of
``wf30_dock_summarize.py`` (the post-processor for the Wf-30
docking sweep).  No GPU, no mutation, idempotent.

Inputs
------
* ``molmetal/reports/wf29_smoke/se3sb/metrics.json``
* ``molmetal/reports/wf29_smoke/transport/metrics.json``
* ``molmetal/reports/wf29_smoke/multiph/metrics.json``
* ``molmetal/reports/wf29_smoke/baseline/metrics.json``

Outputs
-------
* ``molmetal/reports/wf29_smoke/bench_summary.json`` (NEW) — per-recipe
  headline + per-seed Ru test AUC + wall-seconds + provenance tag.
* ``molmetal/reports/wf29_smoke/comparison.md`` (NEW or refreshed) —
  side-by-side table with honest-negative provenance.

Verify
------
* ``molmetal/scripts/verify_wf29_phase2_summarize.py`` (NEW, ~80 LOC)
  exercises this script on a 1-seed × 1-recipe stub.

Usage
-----
::

    source env.sh
    uv run --no-sync python molmetal/scripts/wf29_phase2_summarize.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

DEFAULT_SMOKE_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf29_smoke"
DEFAULT_RECIPES = ("se3sb", "transport", "multiph", "baseline")

# Per-metal column order for the comparison table (matches the
# canonical 5-metal cohort used by train_ru_5metal_se3sb.py).
METAL_COLS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Re", "Os")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _load_metrics(path: Path) -> Optional[dict]:
    """Read a metrics.json if it exists; return None otherwise."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _per_metal_auc_from_per_seed(per_seed: Sequence[dict]) -> Dict[str, float]:
    """Compute per-metal mean test AUC across seeds.

    Each seed record in ``per_seed`` may carry
    ``test_auc_per_metal`` (dict of metal → AUC).  We average across
    seeds.  Metals missing from a particular seed are skipped (not
    treated as 0).
    """
    accum: Dict[str, List[float]] = {m: [] for m in METAL_COLS}
    for rec in per_seed:
        per_metal = rec.get("test_auc_per_metal") or {}
        for metal, auc in per_metal.items():
            if metal in accum and math.isfinite(float(auc)):
                accum[metal].append(float(auc))
    return {m: (statistics.fmean(v) if v else float("nan")) for m, v in accum.items()}


def _extract_recipe_summary(name: str, payload: dict) -> dict:
    """Pull a minimal per-recipe summary block out of a metrics.json."""
    per_seed = payload.get("per_seed") or []
    ru_auc_per_seed = payload.get("ru_test_auc_per_seed") or []
    headline = float(payload.get("ru_mean_test_auc", float("nan")))
    wall = float(payload.get("wall_seconds", payload.get("wall_seconds_total", float("nan"))))
    per_metal = _per_metal_auc_from_per_seed(per_seed)
    return {
        "recipe": name,
        "init": payload.get("init", name),
        "n_seeds": len(per_seed) or len(ru_auc_per_seed),
        "ru_test_auc_per_seed": [float(x) for x in ru_auc_per_seed],
        "ru_mean_test_auc": headline,
        "per_metal_mean_test_auc": per_metal,
        "wall_seconds": wall,
        "provenance": payload.get("_provenance", "MEASURED"),
        "n_epochs": int(payload.get("epochs", 0)),
        "raw_recipe_tag": payload.get("recipe", ""),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_smoke(smoke_dir: Path, recipes: Sequence[str]) -> Tuple[dict, List[dict]]:
    """Read the per-recipe metrics.json files and build the rollup."""
    summaries: List[dict] = []
    skipped: List[str] = []
    for recipe in recipes:
        path = smoke_dir / recipe / "metrics.json"
        payload = _load_metrics(path)
        if payload is None:
            skipped.append(recipe)
            continue
        summaries.append(_extract_recipe_summary(recipe, payload))

    bench_summary = {
        "recipe": "wf29_phase2_smoke_bench_summary",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "smoke_dir": str(smoke_dir),
        "recipes_recorded": [s["recipe"] for s in summaries],
        "recipes_skipped_missing": skipped,
        "per_recipe": summaries,
        "headline": (
            "best mean Ru test AUC across the recorded recipes "
            "(see per_recipe[*].ru_mean_test_auc)"
        ),
        "best_recipe": (
            max(summaries, key=lambda s: s.get("ru_mean_test_auc") or float("-inf"))["recipe"]
            if summaries
            else None
        ),
        "honest_negative_provenance": (
            "All ru_mean_test_auc values come from the Wf-29 Phase 2 "
            "smoke benchmark (1-3 seeds × 10 epochs × small arch on "
            "GPU 1) and are MEASURED — they are NOT directly comparable "
            "to the Wf-31 production runs which use 5 seeds × 50 "
            "epochs on the full data."
        ),
    }
    return bench_summary, summaries


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------
def render_comparison_md(bench_summary: dict) -> str:
    """Render the side-by-side markdown table."""
    lines: List[str] = []
    lines.append("# Wf-29 Phase 2 — 4-Recipe Smoke Comparison")
    lines.append("")
    lines.append(f"_Generated_: {bench_summary['date_utc']}  ")
    lines.append(f"_Smoke dir_: `{bench_summary['smoke_dir']}`  ")
    lines.append(f"_Recipes recorded_: {', '.join(bench_summary['recipes_recorded']) or '(none)'}  ")
    if bench_summary.get("recipes_skipped_missing"):
        lines.append(f"_Recipes skipped (missing metrics.json)_: "
                     f"{', '.join(bench_summary['recipes_skipped_missing'])}  ")
    lines.append("")
    lines.append("## Per-recipe headline (Ru test AUC)")
    lines.append("")
    lines.append("| Recipe | Init | n_seeds | Ru AUC mean | Ru AUC per seed | Wall (s) | Epochs | Provenance |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in bench_summary["per_recipe"]:
        per_seed_str = ", ".join(f"{x:.3f}" for x in s["ru_test_auc_per_seed"])
        ru_auc = s["ru_mean_test_auc"]
        ru_str = f"{ru_auc:.4f}" if math.isfinite(ru_auc) else "—"
        wall = s["wall_seconds"]
        wall_str = f"{wall:.1f}" if math.isfinite(wall) else "—"
        lines.append(
            f"| {s['recipe']} | {s['init']} | {s['n_seeds']} | {ru_str} | "
            f"{per_seed_str} | {wall_str} | {s['n_epochs']} | "
            f"{s['provenance']} |"
        )
    if not bench_summary["per_recipe"]:
        lines.append("| (no recipes recorded) | | | | | | | |")
    lines.append("")
    lines.append("## Per-metal mean test AUC")
    lines.append("")
    lines.append("| Recipe | " + " | ".join(METAL_COLS) + " |")
    lines.append("|---|" + "|".join(["---"] * len(METAL_COLS)) + "|")
    for s in bench_summary["per_recipe"]:
        cells = []
        for m in METAL_COLS:
            v = s["per_metal_mean_test_auc"].get(m, float("nan"))
            cells.append(f"{v:.3f}" if math.isfinite(v) else "—")
        lines.append(f"| {s['recipe']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Honest-negative provenance")
    lines.append("")
    lines.append(bench_summary["honest_negative_provenance"])
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    p.add_argument(
        "--recipes", nargs="+", default=list(DEFAULT_RECIPES),
        help="Recipe sub-directories under --smoke-dir to consolidate.",
    )
    p.add_argument(
        "--summary-path", type=Path,
        default=DEFAULT_SMOKE_DIR / "wf29_summary.json",
    )
    p.add_argument(
        "--comparison-md", type=Path,
        default=DEFAULT_SMOKE_DIR / "comparison_phase2.md",
    )
    p.add_argument(
        "--also-refresh", action="store_true",
        help="If set, also rewrite the existing bench_summary.json + comparison.md "
             "(the Wf-29 trainers write these under their own names).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    bench_summary, _summaries = aggregate_smoke(args.smoke_dir, args.recipes)

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(bench_summary, indent=2))
    print(f"[wf29_summarize] wrote {args.summary_path}")

    md = render_comparison_md(bench_summary)
    args.comparison_md.write_text(md)
    print(f"[wf29_summarize] wrote {args.comparison_md}")

    if args.also_refresh:
        legacy_summary = args.smoke_dir / "bench_summary.json"
        legacy_summary.write_text(json.dumps(bench_summary, indent=2))
        legacy_md = args.smoke_dir / "comparison.md"
        legacy_md.write_text(md)
        print(f"[wf29_summarize] also refreshed {legacy_summary}")
        print(f"[wf29_summarize] also refreshed {legacy_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())