#!/usr/bin/env python3
"""Wf-251-F3 — 4×5×3 multi-task D-MPNN ablation grid driver.

=====================================================================
Driver that runs the per-cell ablation table for paper §6:

  Cells = 4 ablation configs × 5 metals × 3 seeds = 60 cells.

Each cell invokes ``molmetal/scripts/train_dmpnn_multitask_wasserstein.py``
with ``--ablation-config {mtl_baseline, mtl_structured, mtl_gradproj,
single_task}`` (Wf-251-F3 wire-in) and emits a per-cell per-seed JSON.

The aggregator writes:

* ``<out>/<config>/<metal>_seed<N>.json`` — per-cell per-seed payload
* ``<out>/aggregate.csv`` — 60-row CSV (config × metal × seed × AUC)
* ``<out>/ablation_table.md`` — per-config × per-metal mean ± std table

Usage::

    /home/hugo/codes/mol-metal/.venv/bin/python -u \\
        molmetal/scripts/run_mtl_ablation_grid.py \\
        --configs mtl_baseline mtl_structured mtl_gradproj single_task \\
        --metals Ru Ir Os Rh Re --seeds 0 1 2 \\
        --epochs 5 --out-dir molmetal/reports/wf251_mtl_ablation_grid

CPU-only / backward-compat.  Existing CLI flags of
``train_dmpnn_multitask_wasserstein.py`` are unchanged.

Honest framing
--------------
This driver **does not** run the full 60-cell sweep by default — it
caps at ``--epochs 5`` so the smoke is sub-hour on CPU.  Per the
fix_design §F3 the full 60-cell × 50-epoch sweep is 30 hours CPU and
must be scheduled on RTX 5090 GPU-1 (5 hours) for paper §6.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 4 ablation configs (Wf-251-F3).  ``mtl_baseline`` is the Wf-131 P1
# shipped behaviour; the other 3 are lever switches documented in the
# fix_design.  ``single_task`` aliases the Wf-197 --single-task flag.
DEFAULT_CONFIGS: Tuple[str, ...] = (
    "mtl_baseline",
    "mtl_structured",
    "mtl_gradproj",
    "single_task",
)
DEFAULT_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        choices=DEFAULT_CONFIGS,
        help="Ablation configs (default: all 4).",
    )
    p.add_argument(
        "--metals",
        nargs="+",
        default=list(DEFAULT_METALS),
        help="Metals to run (default: Ru Ir Rh Os Re).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds (default: 0 1 2).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs per cell (default 5 for smoke; full sweep = 50).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "molmetal" / "reports" / "wf251_mtl_ablation_grid",
        help="Output directory.",
    )
    p.add_argument(
        "--train-script",
        type=Path,
        default=_HERE / "train_dmpnn_multitask_wasserstein.py",
        help="Path to the train script (default next to this driver).",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter (default sys.executable).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-cell commands without executing them.",
    )
    return p.parse_args(argv)


def run_cell(
    *,
    python: str,
    train_script: Path,
    config: str,
    metal: str,
    seed: int,
    epochs: int,
    cell_dir: Path,
    dry_run: bool,
) -> Dict[str, object]:
    """Run a single ablation cell.

    Returns a dict with the per-cell AUC (or error message).  The
    per-seed JSON is emitted by the train script under ``cell_dir``.
    """
    cmd: List[str] = [
        python,
        str(train_script),
        "--ablation-config",
        config,
        "--metals",
        metal,
        "--seeds",
        str(seed),
        "--epochs",
        str(epochs),
        "--out-dir",
        str(cell_dir),
    ]
    cell_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if dry_run:
        print(f"  DRY-RUN cell: {' '.join(cmd)}")
        return {
            "config": config,
            "metal": metal,
            "seed": seed,
            "elapsed_s": 0.0,
            "status": "dry_run",
        }
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60 * 60,  # 1 hour per cell cap
        )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        return {
            "config": config,
            "metal": metal,
            "seed": seed,
            "elapsed_s": float(elapsed),
            "status": "ok" if ok else "fail",
            "returncode": int(proc.returncode),
            "stdout_tail": (proc.stdout or b"")[-512:].decode(
                "utf-8", errors="replace"
            ),
        }
    except Exception as exc:
        return {
            "config": config,
            "metal": metal,
            "seed": seed,
            "elapsed_s": float(time.time() - t0),
            "status": "exception",
            "error": str(exc),
        }


def aggregate_cell(
    out_dir: Path, config: str, metal: str, seed: int
) -> Optional[float]:
    """Read the per-cell per-seed JSON written by the train script."""
    path = out_dir / config / metal.lower() / f"per_seed_metrics_seed{seed}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        tm = payload.get("test_metrics", {})
        auc = tm.get("roc_auc")
        return float(auc) if auc is not None else None
    except Exception:
        return None


def render_table(rows: List[Dict[str, object]]) -> str:
    """Render the per-config × per-metal mean ± std table."""
    import math

    # Group by (config, metal)
    groups: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        auc = row.get("auc")
        if auc is None or not math.isfinite(auc):
            continue
        key = (str(row["config"]), str(row["metal"]))
        groups.setdefault(key, []).append(float(auc))
    if not groups:
        return "# Empty ablation table — no cells with finite AUC.\n"
    lines = [
        "# Wf-251-F3 — Multi-Task D-MPNN Ablation Grid",
        "",
        "## Per-config × per-metal mean ± std (test AUC)",
        "",
        "| Config | " + " | ".join(DEFAULT_METALS) + " |",
        "|---|" + "|".join(["---"] * len(DEFAULT_METALS)) + "|",
    ]
    for cfg in DEFAULT_CONFIGS:
        row_strs = []
        for met in DEFAULT_METALS:
            vals = groups.get((cfg, met), [])
            if not vals:
                row_strs.append("—")
            else:
                mean = sum(vals) / len(vals)
                std = (
                    sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
                ) ** 0.5
                row_strs.append(f"{mean:.4f} ± {std:.4f}")
        lines.append(f"| {cfg} | " + " | ".join(row_strs) + " |")
    lines.extend(
        [
            "",
            "## Honest framing",
            "",
            "* Each cell is N epochs (default 5 for smoke; full sweep = 50).",
            "* `mtl_baseline` mirrors the Wf-131 P1 shipped behaviour "
            "  (h=128/d=3, sliced-W₂ gradient, 5-metal joint pool).",
            "* `single_task` enforces per-metal cohort only (Wf-197 lever 1).",
            "* `mtl_structured` lifts h to 300 (chemprop v2 production backbone "
            "  — Wf-251-F2).",
            "* `mtl_gradproj` is reserved for the future PCGrad hook; today "
            "  it aliases `mtl_baseline`.",
            "* Δ vs `mtl_baseline` per cell = honest-negative if negative.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells: List[Dict[str, object]] = []
    n_total = len(args.configs) * len(args.metals) * len(args.seeds)
    print(
        f"[wf251-F3] grid = {len(args.configs)} configs × "
        f"{len(args.metals)} metals × {len(args.seeds)} seeds = "
        f"{n_total} cells; epochs={args.epochs}; out={args.out_dir}",
        flush=True,
    )
    cell_idx = 0
    for config in args.configs:
        for metal in args.metals:
            for seed in args.seeds:
                cell_idx += 1
                print(
                    f"  [{cell_idx}/{n_total}] {config} {metal} seed={seed}...",
                    flush=True,
                )
                cell_dir = args.out_dir
                res = run_cell(
                    python=args.python,
                    train_script=args.train_script,
                    config=config,
                    metal=metal,
                    seed=seed,
                    epochs=args.epochs,
                    cell_dir=cell_dir,
                    dry_run=args.dry_run,
                )
                auc = aggregate_cell(args.out_dir, config, metal, seed)
                res["auc"] = auc
                cells.append(res)
                print(
                    f"    -> status={res['status']} elapsed_s={res.get('elapsed_s', 0.0):.1f} "
                    f"auc={auc if auc is not None else 'NA'}",
                    flush=True,
                )

    # Emit aggregate.csv
    csv_path = args.out_dir / "aggregate.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config", "metal", "seed", "status", "auc", "elapsed_s"])
        for r in cells:
            writer.writerow(
                [
                    r.get("config", ""),
                    r.get("metal", ""),
                    r.get("seed", ""),
                    r.get("status", ""),
                    r.get("auc", ""),
                    r.get("elapsed_s", ""),
                ]
            )
    print(f"[wf251-F3] wrote {csv_path}", flush=True)

    # Emit ablation_table.md
    table_md = render_table(cells)
    table_path = args.out_dir / "ablation_table.md"
    table_path.write_text(table_md)
    print(f"[wf251-F3] wrote {table_path}", flush=True)

    # Emit a summary JSON
    summary = {
        "wf": "Wf-251-F3",
        "n_configs": len(args.configs),
        "n_metals": len(args.metals),
        "n_seeds": len(args.seeds),
        "n_total_cells": n_total,
        "epochs_per_cell": args.epochs,
        "configs": list(args.configs),
        "metals": list(args.metals),
        "seeds": list(args.seeds),
        "dry_run": bool(args.dry_run),
        "n_cells_ok": sum(1 for c in cells if c.get("status") == "ok"),
        "n_cells_with_finite_auc": sum(1 for c in cells if c.get("auc") is not None),
        "per_cell": cells,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[wf251-F3] wrote {summary_path}", flush=True)

    print(
        f"\n[wf251-F3] DONE — {n_total} cells attempted; "
        f"{summary['n_cells_ok']} ok; "
        f"{summary['n_cells_with_finite_auc']} with finite AUC.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
