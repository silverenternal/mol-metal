"""Wf-A22 (2026-10-13) step-5: CLI for the per-pocket decode + ADMET panel.

Wraps :func:`molmetal.eval.scoring.per_pocket_metric.build_per_pocket_panel`
for shell usage from ``scripts/launch_*.sh`` wrappers.  Behaviour is
bit-exact identical when NOT invoked (no file written, no import in
the hot path).

Usage:

    .venv/bin/python molmetal/scripts/per_pocket_panel.py \\
        --smiles-csv /tmp/cfm_generated_with_pocket_id.csv \\
        --out-json   metrics/by_metric/cfm_per_pocket_metric.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from molmetal.eval.scoring.per_pocket_metric import build_per_pocket_panel


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="per_pocket_panel.py",
        description=(
            "Wf-A22 L8: per-pocket decode + ADMET panel from a generated-"
            "SMILES CSV with a 'pocket_id' column.  Writes "
            "metrics/by_metric/cfm_per_pocket_metric.json with status "
            "fields preserved (EMPTY_COHORT / MEASURED)."
        ),
    )
    p.add_argument(
        "--smiles-csv",
        required=True,
        type=Path,
        help="CSV with at minimum a 'smiles' column and ideally a 'pocket_id' column.",
    )
    p.add_argument(
        "--out-json",
        required=True,
        type=Path,
        help="Output JSON path; will be created or overwritten.",
    )
    p.add_argument(
        "--largest-frac-min",
        type=float,
        default=0.80,
        help="Largest-fragment minimum for the per-pocket decode gate (default 0.80).",
    )
    p.add_argument(
        "--min-atoms",
        type=int,
        default=2,
        help="Minimum atom count for the per-pocket decode gate (default 2).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    panel = build_per_pocket_panel(
        smiles_csv=args.smiles_csv,
        out_json=args.out_json,
        largest_frac_min=args.largest_frac_min,
        min_atoms=args.min_atoms,
    )
    n_pockets = panel["n_pockets"]
    n_measured = sum(
        1 for v in panel["decode"].values() if v.get("status") == "MEASURED"
    )
    print(
        f"[wf-a22 L8] per_pocket_panel: n_pockets={n_pockets} "
        f"n_measured={n_measured} out={args.out_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())