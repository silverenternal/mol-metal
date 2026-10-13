"""Canonical paper-table CLI for Track E (m14 + 8 ADMET + m4/m5/m7).

Track E (Wf-60231cff-03e-15, m14-admet-paper-align) — single CLI that
wraps :func:`molmetal.eval.scoring.paper_table.build_paper_table` for
the paper §X table evaluator.  Reads an optional generated-SMILES
CSV cohort + the existing side-JSONs in ``metrics/by_metric/`` and
emits ONE canonical row per cohort (m14 + ADMET8 + m4 + m5 + m7) to
``metrics/by_metric/cfm_paper_table_<cohort>.json``.

Honest-negative preservation
----------------------------
m5 MCF7 pIC50 ``PENDING_GPU_RETRAIN`` and m7 IntDiv1
``~0.017 BELOW training distribution`` are preserved verbatim.

§5.6 atomic-write protocol
--------------------------
Writes use ``paper_table.write_paper_table_json`` (POSIX ``os.replace``
round-trip via temp-file).

Run
---
::

    python molmetal/scripts/bench_paper_table.py \\
        --cohort platinai_pt_subset_5k \\
        --smiles-csv molmetal/reports/wf298d/generated_smiles.csv \\
        --out-dir metrics/by_metric \\
        --emit

Or without a cohort (yields side-JSONs only; m14/ADMET8 sentinels)::

    python molmetal/scripts/bench_paper_table.py \\
        --cohort lambda_pilot_5x1 \\
        --out-dir metrics/by_metric \\
        --emit

§5.7 trainer-flag wire-in smoke
--------------------------------
A 4-mol cohort ``__main__`` smoke runs by default if no ``--cohort``
flag is given; prints ``SMOKE PASS 4/4`` and exits 0.  CPU-only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Make the ``molmetal`` package importable when this script is run
# directly (``python molmetal/scripts/bench_paper_table.py ...``).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from molmetal.eval.scoring.paper_table import (
    COHORT_KEYS,
    PROTOCOL_VERSION,
    PaperTableRow,
    build_paper_table,
    write_paper_table_json,
)


DEFAULT_OUT_DIR = _PROJECT_ROOT / "metrics" / "by_metric"


def _parse(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort",
        choices=COHORT_KEYS,
        default="platinai_pt_subset_5k",
        help=(
            "Cohort key feeding paper §X table.  One of "
            f"{COHORT_KEYS!r}.  Default: platinai_pt_subset_5k."
        ),
    )
    p.add_argument(
        "--smiles-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to a generated-SMILES CSV with a 'smiles' "
            "column.  When given, m14 + ADMET8 are computed on the "
            "cohort on-the-fly."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "Output directory for the canonical JSON.  Default: "
            f"{DEFAULT_OUT_DIR}."
        ),
    )
    p.add_argument(
        "--largest-frac-min",
        type=float,
        default=0.80,
        help="Per-atom largest-fragment gate threshold (default 0.80).",
    )
    p.add_argument(
        "--min-atoms",
        type=int,
        default=2,
        help="Reject mols with < N heavy atoms (default 2).",
    )
    p.add_argument(
        "--emit",
        action="store_true",
        help=(
            "If set, write the canonical JSON to "
            "<out-dir>/cfm_paper_table_<cohort>.json.  Without this "
            "flag, the row is printed to stdout and not persisted."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help=(
            "DEPRECATED — defaults to True.  Use --emit to persist.  "
            "Kept as a flag for explicit user override clarity."
        ),
    )
    p.add_argument(
        "--no-dry-run",
        dest="dry_run", action="store_false",
        help="Disable dry-run; persist output even without --emit.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse(argv)
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        # If the metrics dir doesn't exist (e.g. fresh worktree),
        # create it so the writer doesn't blow up.
        out_dir.mkdir(parents=True, exist_ok=True)

    row = build_paper_table(
        cohort_id=args.cohort,
        smiles_csv=args.smiles_csv,
        metrics_dir=out_dir,
        largest_frac_min=args.largest_frac_min,
        min_atoms=args.min_atoms,
    )

    import json as _json
    print(_json.dumps(row.to_dict(), indent=2, sort_keys=True))

    if args.emit and not args.dry_run:
        # Emit is gated; off by default.  Operator-only path.
        out_path = out_dir / f"cfm_paper_table_{args.cohort}.json"
        write_paper_table_json(row, out_path)
        print(f"[bench_paper_table] wrote {out_path}", file=sys.stderr)

    return 0


def _smoke() -> int:
    """Self-test (per §5.7).

    Runs ``build_paper_table`` against a synthetic 4-mol cohort on the
    project's own ``metrics/by_metric`` directory (which exists in
    the public-release worktree).  Asserts the structure of the
    output row + that ``write_paper_table_json`` round-trips.
    """
    metrics_dir = _PROJECT_ROOT / "metrics" / "by_metric"
    if not metrics_dir.exists():
        # Fall back to a temp dir; assertions are structural, not
        # data-dependent.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            metrics_dir = Path(td)
    cohort = ["CCO", "c1ccccc1", "C.C.C.C", "nonsense"]
    row = build_paper_table(
        cohort_id="lambda_pilot_5x1",  # one of COHORT_KEYS
        smiles=cohort,
        metrics_dir=metrics_dir,
    )
    assert row.n_smiles == 4, row
    assert row.protocol_version == PROTOCOL_VERSION, row.protocol_version
    # m14 + ADMET8 always populated when cohort non-empty
    assert row.m14_decode["per_atom_validity"] >= 0.0
    assert "per_metric_mean" in row.admet8
    # m4/m5/m7 status keys always present
    for field_name in ("m4_sa", "m5_mcf7_pic50", "m7_intdiv1"):
        sub = getattr(row, field_name)
        assert "status" in sub, sub
        assert "source" in sub, sub
    # Round-trip via the writer
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "round_trip.json"
        write_paper_table_json(row, out)
        assert out.exists()
        import json as _json
        with out.open("r") as f:
            rd = _json.load(f)
        assert rd["cohort_id"] == "lambda_pilot_5x1"
        assert rd["n_smiles"] == 4
    print(f"[bench_paper_table] SMOKE PASS  4/4  protocol={PROTOCOL_VERSION}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1:]))
    sys.exit(_smoke())
