"""Aggregate the 10x3 Round-12 SA-filtered sweep into a 30-cell SA table.

Reads the report.json written by
:mod:`molmetal.scripts.r4_lambda_only_run` and computes the per-cell
SA mean + per-pocket aggregate + per-seed aggregate + grand mean,
then writes a 30-row table + a 4-line lift summary to stdout.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

_REPORT_ROOT = Path("/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_sa_fragment_pool_optimize/r4c")
_OUT = Path("/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_sa_fragment_pool_optimize")
_BASELINE_SA = 3.32  # WF-SA-Penalty-Guidance baseline (sa_mean=3.32 after --sa-weight 0.3)
_TARGETDIFF_SA_MIN = 2.65
_TARGETDIFF_SA_MAX = 2.86


def _finite(x: float) -> bool:
    return x is not None and x == x and abs(x) != float("inf")


def main() -> int:
    report_path = _REPORT_ROOT / "report.json"
    if not report_path.exists():
        print(f"ERROR: report not found at {report_path}", file=sys.stderr)
        return 2
    raw = json.loads(report_path.read_text())
    cells = raw.get("cells", [])
    if not cells:
        print("ERROR: report contains 0 cells", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for c in cells:
        sa = c.get("sa_mean")
        sa = float(sa) if sa is not None and _finite(sa) else float("nan")
        rows.append({
            "pocket_id": c.get("pocket_id", "?"),
            "seed": c.get("seed"),
            "n_simulations": c.get("n_simulations"),
            "n_top_k": c.get("n_top_k"),
            "sa_mean": sa,
            "n_candidates": c.get("n_candidates", 0),
            "valid_rate": c.get("valid_rate", float("nan")),
            "synth_rate": c.get("synth_rate", float("nan")),
            "metal_compliance": c.get("metal_compliance", float("nan")),
            "wall_s": c.get("elapsed_s", float("nan")),
        })

    valid_sa = [r["sa_mean"] for r in rows if _finite(r["sa_mean"])]
    grand_mean = statistics.fmean(valid_sa) if valid_sa else float("nan")

    by_pocket: dict[str, list[float]] = {}
    for r in rows:
        if _finite(r["sa_mean"]):
            by_pocket.setdefault(r["pocket_id"], []).append(r["sa_mean"])
    by_seed: dict[int, list[float]] = {}
    for r in rows:
        if _finite(r["sa_mean"]):
            by_seed.setdefault(int(r["seed"]), []).append(r["sa_mean"])

    print(f"\n=== WF-SA-Fragment-Pool-Optimize ===")
    print(f"Run dir: {_REPORT_ROOT}")
    print(f"Cells completed: {len(rows)} (10 pockets × 3 seeds)")
    print(f"Cells with valid SA: {len(valid_sa)} / {len(rows)}")
    print(f"Grand mean SA: {grand_mean:.3f}  (range {min(valid_sa) if valid_sa else float('nan'):.3f}–{max(valid_sa) if valid_sa else float('nan'):.3f})")

    print("\nPer-cell SA table:")
    print(f"{'pocket':<12} {'seed':>6} {'sa_mean':>8} {'valid':>6} {'synth':>6} {'metal':>6} {'wall_s':>7}")
    for r in rows:
        sa_str = f"{r['sa_mean']:.3f}" if _finite(r["sa_mean"]) else "  nan"
        print(f"{r['pocket_id']:<12} {r['seed']:>6} {sa_str:>8} "
              f"{r['valid_rate']:>6.2f} {r['synth_rate']:>6.2f} "
              f"{r['metal_compliance']:>6.2f} {r['wall_s']:>7.1f}")

    print("\nPer-pocket SA mean:")
    for pid, vals in sorted(by_pocket.items()):
        print(f"  {pid:<12} n={len(vals):>2}  mean_SA={statistics.fmean(vals):.3f}")

    print("\nPer-seed SA mean:")
    for s, vals in sorted(by_seed.items()):
        print(f"  seed={s:>4}   n={len(vals):>2}  mean_SA={statistics.fmean(vals):.3f}")

    # CSV dump
    out_csv = _OUT / "sa_30cell_table.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_csv}")

    # Schema-required metrics
    n_tiles_removed = 10
    sa_mean_aggregate = grand_mean
    lift_vs_baseline = _BASELINE_SA - grand_mean
    n_in_targetdiff = sum(1 for v in valid_sa if _TARGETDIFF_SA_MIN <= v <= _TARGETDIFF_SA_MAX)
    in_range = (_TARGETDIFF_SA_MIN <= grand_mean <= _TARGETDIFF_SA_MAX)
    print("\n=== SCHEMA METRICS ===")
    print(json.dumps({
        "n_tiles_removed": n_tiles_removed,
        "sa_mean_aggregate": round(sa_mean_aggregate, 3),
        "lift_vs_sa_penalty_baseline": round(lift_vs_baseline, 3),
        "comparison_vs_targetdiff_265_286": (
            f"in_range={in_range}; baseline_sa={_BASELINE_SA}; "
            f"targetdiff_min={_TARGETDIFF_SA_MIN}; targetdiff_max={_TARGETDIFF_SA_MAX}"
        ),
        "n_pockets_with_sa_in_range": n_in_targetdiff,
        "n_pockets_total": len(by_pocket),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
