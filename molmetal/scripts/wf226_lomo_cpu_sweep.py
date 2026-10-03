#!/usr/bin/env python3
"""Wf-226 — 50-epoch CPU sweep launcher for Wf-131 P3 LOMO missing pairs.

Runs the Wf-131 Phase 3 LOMO trainer (``train_metal_lomo_wasserstein.py``)
at 50 epochs with a smaller backbone (hidden=64, depth=2) for CPU speed,
on the 7 PENDING (metal, seed) pairs:

    Rh × seeds 3, 4   (2 pairs)
    Re × seeds 0,1,2,3,4   (5 pairs)
    ============================
                          7 pairs total

Honest framing note
===================
Per `party-twenty-spirit §3.5` the Wf-131 P3 on-disk state (post Wf-193
extension) shows **18 + 2 = 20 pairs done** (Ir/Os/Ru × seeds 0-4 + Rh
× seeds 0-2 + Wf-193 Ir × seeds 1,2).  The "14 missing pairs" framing in
the task description reflects the original TODO-180 (only Ru seed=0
done).  On disk today only 7 pairs are actually missing to complete the
5-metal × 5-seed matrix (25 total).  This launcher fills those 7 pairs.

Wall-time budget: <= 8h on 32-core x86_64 CPU.  With 7-way parallel and
hidden=64 depth=2 each pair takes ~3-4 min on the smaller-metal cohorts
(Os/Rh/Re) and ~6-8 min on Ir (largest cohort); the Rh-oversample cohort
is appended only when the held-out metal is Rh (2 of the 7 pairs).

Usage:
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf226_lomo_cpu_sweep.py \\
        --n-parallel 4 \\
        --epochs 50 \\
        --hidden 64 --depth 2
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
PY = "/home/hugo/codes/mol-metal/.venv/bin/python"
TRAIN_SCRIPT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "train_metal_lomo_wasserstein.py"
)
DEFAULT_CSV = PROJECT_ROOT / "molmetal" / "data" / "metal_lomo.csv"
OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf226_lomo_sweep"
CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints" / "wf226_lomo"

# 7 PENDING (metal, seed) pairs to complete the 5×5 LOMO matrix.
# (Ru seed=0 done at Wf-131 P3; Ir/Os/Ru × seeds 0-4 + Rh × seeds 0-2
# already on disk in Wf-131 P3; Wf-193 added Ir × seeds 1,2.)
MISSING_PAIRS: list[tuple[str, int]] = [
    ("Rh", 3), ("Rh", 4),
    ("Re", 0), ("Re", 1), ("Re", 2), ("Re", 3), ("Re", 4),
]


def run_one_pair(
    metal: str,
    seed: int,
    epochs: int,
    hidden: int,
    depth: int,
    batch_size: int,
    wasserstein_weight: float,
    rh_oversample: int,
    log_dir: Path,
) -> dict:
    """Train one (metal, seed) LOMO fold and return a summary."""
    log_path = log_dir / f"{metal.lower()}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(TRAIN_SCRIPT),
        "--metals", metal,
        "--seeds", str(seed),
        "--epochs", str(epochs),
        "--hidden", str(hidden),
        "--depth", str(depth),
        "--batch-size", str(batch_size),
        "--wasserstein-weight", str(wasserstein_weight),
        "--rh-oversample", str(rh_oversample),
        "--csv", str(DEFAULT_CSV),
        "--out-dir", str(OUT_DIR),
    ]
    env = os.environ.copy()
    # Cap threads per process so 4 parallel runs share the 32 cores.
    env.setdefault("OMP_NUM_THREADS", "6")
    env.setdefault("MKL_NUM_THREADS", "6")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Force CPU (the LOMO trainer defaults to GPU if available)
    env["CUDA_VISIBLE_DEVICES"] = ""
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
    return {
        "metal": metal,
        "seed": seed,
        "wall_seconds": wall,
        "returncode": proc.returncode,
        "log_path": str(log_path),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Wf-226 — LOMO 50-epoch CPU sweep")
    p.add_argument("--n-parallel", type=int, default=4,
                   help="Number of parallel (metal, seed) pair runs (default 4)")
    p.add_argument("--epochs", type=int, default=50, help="Epochs per pair")
    p.add_argument("--hidden", type=int, default=64, help="Backbone hidden dim")
    p.add_argument("--depth", type=int, default=2, help="Backbone depth")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--wasserstein-weight", type=float, default=0.05,
                   help="Wasserstein transfer loss weight")
    p.add_argument("--rh-oversample", type=int, default=500,
                   help="Synthetic Rh cohort count (applied to all folds)")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "wf226_lomo_sweep" / "logs",
    )
    p.add_argument(
        "--pairs",
        type=str,
        default=None,
        help=(
            "Comma-separated 'metal:seed' list (e.g. 'Re:0,Re:1'). "
            "Default = all 7 MISSING_PAIRS."
        ),
    )
    args = p.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs:
        pairs: list[tuple[str, int]] = []
        for tok in args.pairs.split(","):
            metal, seed = tok.split(":")
            pairs.append((metal, int(seed)))
    else:
        pairs = list(MISSING_PAIRS)

    print(
        f"[wf226] sweeping {len(pairs)} LOMO pairs at "
        f"epochs={args.epochs} hidden={args.hidden} depth={args.depth} "
        f"n_parallel={args.n_parallel}",
        flush=True,
    )
    t_start = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.n_parallel) as ex:
        futs = {
            ex.submit(
                run_one_pair,
                metal, seed, args.epochs, args.hidden, args.depth,
                args.batch_size, args.wasserstein_weight,
                args.rh_oversample, args.log_dir,
            ): (metal, seed)
            for metal, seed in pairs
        }
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            ms, sd = res["metal"], res["seed"]
            wall = res["wall_seconds"]
            rc = res["returncode"]
            print(
                f"[wf226] DONE {ms} seed={sd} "
                f"wall={wall:.1f}s returncode={rc}",
                flush=True,
            )
    total_wall = time.time() - t_start
    print(
        f"[wf226] ALL DONE total_wall={total_wall:.1f}s "
        f"({total_wall / 60:.1f} min, {total_wall / 3600:.2f} h)",
        flush=True,
    )

    summary = {
        "wf_tag": "Wf-226 — 50-epoch CPU LOMO sweep of 7 PENDING (metal, seed) pairs",
        "n_pairs": len(pairs),
        "n_parallel": args.n_parallel,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "depth": args.depth,
        "total_wall_seconds": total_wall,
        "total_wall_minutes": total_wall / 60.0,
        "total_wall_hours": total_wall / 3600.0,
        "results": sorted(results, key=lambda r: (r["metal"], r["seed"])),
    }
    summary_path = OUT_DIR / "wf226_launch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[wf226] wrote launch summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
