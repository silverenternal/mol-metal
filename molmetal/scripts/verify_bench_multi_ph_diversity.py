#!/usr/bin/env python3
"""Wf-31 Phase 3 — Smoke verification of ``bench_multi_ph_diversity.py``.

Runs the bench on a **minimal** configuration:

* ``--n-samples 8`` (deterministic SMILES pool — small enough to run in
  <30 s on any laptop).
* ``--n-bins 8`` (small Multi-PH feature map — feature_dim = 64).

The verification is runnable on a fresh checkout with no staged data;
it loads the real PlatinAI 5K CSV, samples 8 SMILES deterministically
(seed=42), extracts the real Wf-1 T5 Multi-PH features, and reports the
expressivity + diversity numbers.  A successful smoke run produces a
``wf31_multi_ph_diversity_verify.json`` metrics file with non-empty
expressivity + diversity fields.

Usage
-----
::

    source env.sh
    uv run --no-sync python molmetal/scripts/verify_bench_multi_ph_diversity.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Repo-root path setup so the bench script can be imported as a library.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_multi_ph_diversity as bench  # type: ignore[import-not-found]


log = logging.getLogger("verify_bench_multi_ph_diversity")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", default=bench._HERE.parent / "data" / "platinai_5k.csv",
                   help="PlatinAI 5K CSV (default: data/platinai_5k.csv).")
    p.add_argument("--n-samples", type=int, default=8,
                   help="Number of SMILES to sample (default: 8).")
    p.add_argument("--n-bins", type=int, default=8,
                   help="Multi-PH bin count (default: 8).")
    p.add_argument("--out-dir", type=Path,
                   default=bench._HERE.parent.parent
                   / "molmetal" / "reports" / "wf31_multi_ph_diversity",
                   help="Output directory.")
    p.add_argument("--metrics-path", type=Path,
                   default=None,
                   help="Output metrics path (default: <out-dir>/"
                        "wf31_multi_ph_diversity_verify.json).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        args.metrics_path
        if args.metrics_path is not None
        else args.out_dir / "wf31_multi_ph_diversity_verify.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "verify_bench_multi_ph_diversity smoke: n_samples=%d n_bins=%d",
        args.n_samples, args.n_bins,
    )

    metrics = bench.run_bench(
        platinai_csv=Path(args.data),
        n_samples=int(args.n_samples),
        n_bins=int(args.n_bins),
        seed=42,
    )

    metrics_payload = {
        "recipe": "wf31_multi_ph_diversity_verify",
        **metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")
    log.info("wrote %s", metrics_path)

    exp = metrics["expressivity"]
    div = metrics["diversity"]
    if (
        metrics["n_parsed_multiph"] < 2
        or metrics["n_parsed_diversity"] < 2
    ):
        log.error(
            "verify_bench_multi_ph_diversity failed: too few samples "
            "(parsed Multi-PH=%d, diversity=%d)",
            metrics["n_parsed_multiph"], metrics["n_parsed_diversity"],
        )
        return 1
    log.info(
        "  expressivity multi_ph=%.3f single_ph=%.3f delta=%+.3f",
        exp["multi_ph"], exp["single_ph"], exp["delta"],
    )
    log.info(
        "  diversity tanimoto=%.3f meets_m7_floor=%s",
        div["mean_tanimoto_distance"], div["meets_m7_floor"],
    )
    log.info(
        "verify_bench_multi_ph_diversity PASSED "
        "(parsed_multi_ph=%d, parsed_diversity=%d)",
        metrics["n_parsed_multiph"], metrics["n_parsed_diversity"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
