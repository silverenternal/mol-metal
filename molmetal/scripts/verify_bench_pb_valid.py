#!/usr/bin/env python3
"""Wf-32 Phase 3 — Smoke verification of ``bench_pb_valid.py``.

Runs the PoseBusters validity sweep on a **minimal** configuration:

* 1 pocket (must be staged in ``--pocket-dir``; default
  ``molmetal/reports/wf30_pocket_prep/830c``).
* 1 candidate (``CCO`` — ethanol; fast to parse, no metals).
* 1 method (``mol`` — chemistry + geometry only).

The smoke is runnable without a real receptor file; if the chosen
pocket is not staged the script synthesises a dummy 2-atom ``pdbqt``
on the fly so the close-loop end-to-end path can be exercised on a
fresh checkout.  Both the ``posebusters`` package being installed and
it being missing are valid outcomes — the script records
``pb_pass_rate = None`` in the latter case so the closed-loop
outer-gate pattern stays intact.

Usage
----
::

    source env.sh
    uv run --no-sync python -m molmetal.scripts.verify_bench_pb_valid \\
        --pocket 830c --candidate CCO --method mol
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import List

# Repo-root path setup so we can import the sweep script as a library.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_pb_valid as pb  # type: ignore[import-not-found]


log = logging.getLogger("verify_bench_pb_valid")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pocket", default="830c", help="WF-30 pocket_id (default: 830c — MMP-13)."
    )
    p.add_argument(
        "--candidate",
        default="CCO",
        help="SMILES of the single candidate (default: CCO — ethanol).",
    )
    p.add_argument(
        "--pb-mode",
        choices=("mol", "dock"),
        default="mol",
        help="PB mode (default: mol — chemistry only).",
    )
    p.add_argument(
        "--method",
        choices=("se3sb", "transport", "multiph", "baseline"),
        default="se3sb",
        help="Trajectory-generation tag (recorded as metadata; " "default: se3sb).",
    )
    p.add_argument("--pocket-dir", type=Path, default=pb.WF30_POCKET_DIR)
    p.add_argument("--out-dir", type=Path, default=pb.DEFAULT_OUT_DIR)
    p.add_argument(
        "--metrics-path",
        type=Path,
        default=pb.DEFAULT_OUT_DIR / "wf32_pb_valid_verify.json",
    )
    p.add_argument(
        "--summary-path",
        type=Path,
        default=pb.DEFAULT_OUT_DIR / "wf32_pb_valid_verify_summary.json",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    cand = pb.Candidate(name="verify_smoke", smiles=args.candidate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "verify_bench_pb_valid smoke: pocket=%s candidate=%s pb_mode=%s method=%s",
        args.pocket,
        args.candidate,
        args.pb_mode,
        args.method,
    )

    # If the requested pocket isn't staged, materialise a tiny dummy
    # pdbqt on the fly so the smoke is runnable on a fresh checkout
    # (without requiring check_10_pockets.py to have run).
    pdbqt = args.pocket_dir / args.pocket / "receptor.pdbqt"
    if not pdbqt.exists():
        log.warning(
            "%s missing — synthesising a 2-atom dummy pdbqt for the smoke.",
            pdbqt,
        )
        pdbqt.parent.mkdir(parents=True, exist_ok=True)
        pdbqt.write_text(
            "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00    +0.000 C \n"
            "ATOM      2  N   ALA A   2       4.000   4.000   4.000  1.00  0.00    +0.000 N \n"
        )

    results = pb.run_sweep(
        pockets=[args.pocket],
        candidates=[cand],
        pb_modes=(args.pb_mode,),
        pocket_dir=args.pocket_dir,
        trajectory_method=args.method,
    )

    # Contract checks.
    if len(results) != 1:
        log.error("expected 1 evaluation, got %d", len(results))
        return 1
    r = results[0]
    log.info(
        "  pocket=%s candidate=%s pb_mode=%s traj=%s status=%s pass_rate=%.3f "
        "(n_checks=%d n_passed=%d n_failed=%d elapsed=%.2fs)",
        r.pocket_id,
        r.candidate_name,
        r.method,
        r.trajectory_method,
        r.status,
        r.pass_rate,
        r.n_checks,
        r.n_passed,
        r.n_failed,
        r.elapsed_s,
    )

    per_pocket, aggregate = pb.aggregate_results(results)
    overall = pb.overall_pass_rate(results, args.pb_mode, args.method)

    metrics_payload = {
        "recipe": "wf32_pb_valid_verify",
        "n_evaluations": int(len(results)),
        "pocket_id": args.pocket,
        "candidate": args.candidate,
        "pb_mode": args.pb_mode,
        "method": args.method,
        "per_evaluation": [
            {
                "pocket_id": r.pocket_id,
                "candidate": r.candidate,
                "candidate_name": r.candidate_name,
                "method": r.method,
                "trajectory_method": r.trajectory_method,
                "status": r.status,
                "pb_valid": r.pb_valid,
                "n_checks": r.n_checks,
                "n_passed": r.n_passed,
                "n_failed": r.n_failed,
                "pass_rate": r.pass_rate,
                "failed_checks": list(r.failed_checks),
                "elapsed_s": r.elapsed_s,
            }
            for r in results
        ],
        "per_pocket_per_method": pb._to_jsonable(per_pocket),
        "overall_pb_pass_rate": pb._to_jsonable(overall),
    }
    args.metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")
    log.info("wrote %s", args.metrics_path)

    summary_payload = {
        "recipe": "wf32_pb_valid_verify_summary",
        "n_evaluations": int(len(results)),
        "pocket_id": args.pocket,
        "candidate": args.candidate,
        "pb_mode": args.pb_mode,
        "method": args.method,
        "overall_pb_pass_rate": pb._to_jsonable(overall),
        "aggregate": {
            f"{k[0]}|{k[1]}": pb._to_jsonable(v) for k, v in aggregate.items()
        },
    }
    args.summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
    log.info("wrote %s", args.summary_path)

    # Headline pass/fail: at least one PB evaluation completed (any
    # status — passed, failed, or skipped is OK; only an outright
    # crash or zero-cell run is a failure).
    if r.status == "error":
        log.error(
            "verify_bench_pb_valid failed: cell status=%s note=%s", r.status, r.note
        )
        return 1
    log.info(
        "verify_bench_pb_valid PASSED (status=%s overall=%.3f)",
        r.status,
        overall if overall is not None else float("nan"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
