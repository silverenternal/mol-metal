#!/usr/bin/env python3
"""Wf-30 Phase 2 — Smoke verification of ``dock_10pockets_se3sb.py``.

Runs the docking sweep on a **minimal** configuration:

* 1 pocket (must be staged in ``--pocket-dir``; default
  ``molmetal/reports/wf30_pocket_prep/830c``).
* 1 candidate (``CCO`` — ethanol; fast to parse, no metals).
* 2 methods (``se3sb``, ``baseline``).
* 2 seeds.

This is the Wf-30 Phase 2 contract: the script must run end-to-end
on a single (pocket, candidate, method, seed) tuple without raising,
write both JSON outputs, and report a finite Vina kcal/mol value.

If Vina / meeko are available on PATH the script uses the real
``VinaDockingAdapter``; otherwise it falls back to the deterministic
stub scorer (matching the unit-test path). The smoke test treats
both paths as success as long as ``vina_kcal_mol`` is finite.

Usage
-----
::

    source env.sh
    uv run --no-sync python -m molmetal.scripts.verify_dock_10pockets \\
        --pocket 830c --candidate CCO
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
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dock_10pockets_se3sb as dock  # type: ignore[import-not-found]


log = logging.getLogger("verify_dock_10pockets")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pocket", default="830c",
                   help="WF-30 pocket_id (default: 830c — MMP-13).")
    p.add_argument("--candidate", default="CCO",
                   help="SMILES of the single candidate (default: CCO — ethanol).")
    p.add_argument("--pocket-dir", type=Path, default=dock.WF30_POCKET_DIR)
    p.add_argument("--out-dir", type=Path, default=dock.DEFAULT_OUT_DIR)
    p.add_argument("--metrics-path", type=Path,
                   default=dock.DEFAULT_OUT_DIR / "wf30_dock_verify.json")
    p.add_argument("--summary-path", type=Path,
                   default=dock.DEFAULT_OUT_DIR / "wf30_dock_verify_summary.json")
    p.add_argument("--use-real-vina", action="store_true")
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

    cand = dock.Candidate(name="verify_smoke", smiles=args.candidate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "verify_dock_10pockets smoke: pocket=%s candidate=%s methods=(se3sb,baseline)",
        args.pocket, args.candidate,
    )

    # If the requested pocket isn't staged, materialise a tiny dummy
    # receptor.pdbqt on the fly so the smoke is runnable on a fresh
    # checkout (without requiring check_10_pockets.py to have run).
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

    results = dock.run_sweep(
        pockets=[args.pocket],
        candidates=[cand],
        methods=("se3sb", "baseline"),
        seeds=(0, 1),
        pocket_dir=args.pocket_dir,
        n_steps=dock.SE3_INTEGRATION_STEPS,
        cpu_count=1,
        use_real_vina=bool(args.use_real_vina),
    )

    # Contract checks.
    n_expected = 4  # 1 pocket × 1 candidate × 2 methods × 2 seeds
    if len(results) != n_expected:
        log.error(
            "expected %d evaluations, got %d",
            n_expected, len(results),
        )
        return 1
    for r in results:
        log.info(
            "  pocket=%s method=%s seed=%d status=%s vina=%.3f elapsed=%.2fs",
            r.pocket_id, r.method, r.seed, r.status,
            r.vina_kcal_mol if r.vina_kcal_mol is not None else float("nan"),
            r.elapsed_s,
        )
        if r.status == "docked":
            if r.vina_kcal_mol is None:
                log.error("docked status but no score")
                return 1
            if not math.isfinite(r.vina_kcal_mol):
                log.error("non-finite vina_kcal_mol=%s", r.vina_kcal_mol)
                return 1
        else:
            log.warning("  -> non-docked status %s (note=%s)",
                        r.status, r.note)

    per_pocket, aggregate = dock.aggregate_results(results)
    win_tie_loss = dock._win_tie_loss(aggregate, ("se3sb", "baseline"))

    metrics_payload = {
        "recipe": "wf30_dock_verify_smoke",
        "n_evaluations": int(len(results)),
        "pocket_id": args.pocket,
        "candidate": args.candidate,
        "use_real_vina": bool(args.use_real_vina),
        "per_evaluation": [
            {
                "pocket_id": r.pocket_id,
                "method": r.method,
                "seed": r.seed,
                "status": r.status,
                "vina_kcal_mol": r.vina_kcal_mol,
                "elapsed_s": r.elapsed_s,
            }
            for r in results
        ],
        "per_pocket_per_method": per_pocket,
        "aggregate": {
            f"{k[0]}|{k[1]}": v for k, v in aggregate.items()
        },
        "win_tie_loss_vs_baseline": win_tie_loss,
    }
    args.metrics_path.write_text(
        json.dumps(metrics_payload, indent=2) + "\n"
    )
    log.info("wrote %s", args.metrics_path)

    summary_payload = {
        "recipe": "wf30_dock_verify_smoke_summary",
        "n_evaluations": int(len(results)),
        "n_docked": int(sum(1 for r in results if r.status == "docked")),
        "n_failed": int(sum(1 for r in results if r.status != "docked")),
        "aggregate": {
            f"{k[0]}|{k[1]}": v for k, v in aggregate.items()
        },
    }
    args.summary_path.write_text(
        json.dumps(summary_payload, indent=2) + "\n"
    )
    log.info("wrote %s", args.summary_path)

    # Headline pass/fail: at least one docked evaluation with a finite score.
    n_docked = sum(1 for r in results if r.status == "docked")
    if n_docked == 0:
        log.error("verify_dock_10pockets failed: zero docked evaluations")
        return 1
    log.info("verify_dock_10pockets PASSED (%d/%d docked)", n_docked, len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())