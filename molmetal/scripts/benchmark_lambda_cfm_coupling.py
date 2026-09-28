"""Benchmark — Lambda × CFM CPU joint inference coupling (TODO-21).

================================================================
W2.3 / R17 ultracode dispatch (2026-09-26).

This script benchmarks the CPU-only joint inference skeleton on the
PlatinAI 5K CSV (``molmetal/data/platinai_5k.csv``) and writes a
structured JSON summary to
``molmetal/reports/wf_lambda_cfm_coupling_audit/benchmark.json``.

Measurements
------------
For ``--n-samples`` PlatinAI SMILES (default 100, deterministic head
of the CSV), we record:

* ``wall_time_s``  per-sample wall-clock time,
* ``peak_memory_mb``  Python tracemalloc peak delta across the run,
* ``n_atoms``  length of the decoded ``lambda_features`` vector (the
  32-d λ-feature is the proxy "atom count" in this skeleton; the
  joint inference skeleton does not emit 3-D coordinates).

Output schema
-------------
::

    {
      "n_samples": 100,
      "mean_wall_time_s": 0.0123,
      "std_wall_time_s":  0.0014,
      "peak_memory_mb": 3.21,
      "mean_n_atoms":    32.0,
      "n_decoded_smiles": 100,
      "bridge_source":   "stub" | "<path>",
      "alpha":           0.5,
      "n_steps":         50,
      "timestamp_utc":   "2026-09-26T00:00:00Z"
    }

Honest framing
--------------
* This is a **CPU joint inference skeleton** benchmark, NOT a GPU
  joint training benchmark.  The "CFM ODE" integrated is a closed-
  form analytic linear-path CFM; the speed-up story is deferred to
  the post-GPU-retrain work in
  ``wf_cfm_gpu_retrain/final.md``.
* The "atom count" is the 32-d λ-feature dimension, not 3-D atom
  positions — there are no 3-D coordinates emitted by the skeleton.
* The default ``--n-samples 100`` runs in < 5 s on a single CPU core.

Usage::

    uv run --no-sync python molmetal/scripts/benchmark_lambda_cfm_coupling.py --n-samples 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import List

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (  # noqa: E402
    load_joint_pipeline,
)


def _load_platinai_smiles(csv_path: Path, n: int) -> List[str]:
    """Return the first ``n`` SMILES from the PlatinAI 5K CSV."""
    out: List[str] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                out.append(smi)
            if len(out) >= n:
                break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of PlatinAI samples to benchmark (default: 100).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="CFM ODE integration steps per sample (default: 50).",
    )
    parser.add_argument(
        "--pocket-size",
        type=int,
        default=8,
        help="Pocket conditioning vector size (default: 8).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Convex weight on lambda vs pocket in CFM target (default: 0.5).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(_REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"),
        help="PlatinAI CSV path (default: molmetal/data/platinai_5k.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(
            _REPO_ROOT
            / "molmetal"
            / "reports"
            / "wf_lambda_cfm_coupling_audit"
        ),
        help="Output directory for benchmark.json.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"[bench] platinai CSV not found: {csv_path}", file=sys.stderr)
        return 1

    smiles = _load_platinai_smiles(csv_path, int(args.n_samples))
    if not smiles:
        print(f"[bench] no SMILES in {csv_path}", file=sys.stderr)
        return 1

    pipe = load_joint_pipeline(alpha=float(args.alpha), seed=0)
    bridge_source = (
        str(pipe.bridge.source_path)
        if pipe.bridge.source_path is not None
        else "stub"
    )

    pocket = np.zeros(int(args.pocket_size), dtype=np.float32)
    wall_times: List[float] = []
    n_atoms_list: List[int] = []
    n_decoded = 0

    tracemalloc.start()
    base_mem = tracemalloc.get_traced_memory()[0]
    for smi in smiles:
        t0 = time.perf_counter()
        out = pipe.run_full_pipeline(smi, pocket, n_steps=int(args.n_steps))
        elapsed = time.perf_counter() - t0
        wall_times.append(float(elapsed))
        n_atoms_list.append(int(out["lambda_features"].shape[0]))
        if out["decoded_smiles"]:
            n_decoded += 1
    peak_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    summary = {
        "n_samples": int(len(smiles)),
        "mean_wall_time_s": float(mean(wall_times)),
        "std_wall_time_s": float(pstdev(wall_times)) if len(wall_times) > 1 else 0.0,
        "peak_memory_mb": float((peak_mem - base_mem) / (1024 * 1024)),
        "mean_n_atoms": float(mean(n_atoms_list)),
        "n_decoded_smiles": int(n_decoded),
        "bridge_source": bridge_source,
        "alpha": float(args.alpha),
        "n_steps": int(args.n_steps),
        "pocket_size": int(args.pocket_size),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"[bench] n_samples={summary['n_samples']} "
        f"mean_wall_time_s={summary['mean_wall_time_s']:.4f} "
        f"peak_memory_mb={summary['peak_memory_mb']:.2f} "
        f"mean_n_atoms={summary['mean_n_atoms']:.1f} "
        f"decoded={summary['n_decoded_smiles']}/{summary['n_samples']}"
    )
    print(f"[bench] wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())