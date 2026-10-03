#!/usr/bin/env python3
"""Wf-196 — 50-epoch CPU sweep launcher.

Runs the 14 missing (metal, seed) pairs (Ru s1/s2 + Ir/Os/Rh/Re s0/s1/s2)
at 50 epochs with a smaller backbone (hidden=64, depth=2) for CPU speed.

Wall-time budget: <= 8h on 32-core x86_64 CPU.

Usage:
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf196_50ep_cpu_sweep.py \\
        --n-parallel 6 \\
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
    PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
)
DATA_CSV = Path("/home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv")
OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf196_50ep_sweep"
CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints" / "wf196_50ep"

# 14 PENDING (metal, seed) pairs (Ru seed=0 already MEASURED at 30 epochs;
# all others are PENDING per TODO-180 §2 + TODO-181 §3).
MISSING_PAIRS: list[tuple[str, int]] = [
    ("Ru", 1),
    ("Ru", 2),
    ("Ir", 0),
    ("Ir", 1),
    ("Ir", 2),
    ("Os", 0),
    ("Os", 1),
    ("Os", 2),
    ("Rh", 0),
    ("Rh", 1),
    ("Rh", 2),
    ("Re", 0),
    ("Re", 1),
    ("Re", 2),
]


def run_one_pair(
    metal: str,
    seed: int,
    epochs: int,
    hidden: int,
    depth: int,
    batch_size: int,
    log_dir: Path,
) -> dict:
    """Train a single (metal, seed) pair and return its result summary."""
    log_path = log_dir / f"{metal.lower()}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(TRAIN_SCRIPT),
        "--metals",
        metal,
        "--seeds",
        str(seed),
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
        str(OUT_DIR),
        "--ckpt-dir",
        str(CKPT_DIR),
    ]
    env = os.environ.copy()
    # Cap threads per process so 6 parallel runs share the 32 cores.
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
    return {
        "metal": metal,
        "seed": seed,
        "wall_seconds": wall,
        "returncode": proc.returncode,
        "log_path": str(log_path),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Wf-196 — 50-epoch CPU sweep launcher")
    p.add_argument(
        "--n-parallel",
        type=int,
        default=6,
        help="Number of parallel (metal, seed) pair runs (default 6)",
    )
    p.add_argument("--epochs", type=int, default=50, help="Epochs per pair")
    p.add_argument("--hidden", type=int, default=64, help="Backbone hidden dim")
    p.add_argument("--depth", type=int, default=2, help="Backbone depth")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "wf196_50ep_sweep" / "logs",
    )
    p.add_argument(
        "--pairs",
        type=str,
        default=None,
        help=(
            "Comma-separated 'metal:seed' list (e.g. 'Ir:0,Os:1'). "
            "Default = all 14 MISSING_PAIRS."
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
        f"[wf196] sweeping {len(pairs)} pairs at "
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
                metal,
                seed,
                args.epochs,
                args.hidden,
                args.depth,
                args.batch_size,
                args.log_dir,
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
                f"[wf196] DONE {ms} seed={sd} " f"wall={wall:.1f}s returncode={rc}",
                flush=True,
            )
    total_wall = time.time() - t_start
    print(
        f"[wf196] ALL DONE total_wall={total_wall:.1f}s "
        f"({total_wall / 60:.1f} min, {total_wall / 3600:.2f} h)",
        flush=True,
    )

    summary = {
        "wf_tag": "Wf-196 — 50-epoch CPU sweep of 14 missing (metal, seed) pairs",
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
    summary_path = OUT_DIR / "wf196_launch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[wf196] wrote launch summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
