#!/usr/bin/env python
"""Evaluate the TODO-15 anticancer metric suite over a SMILES file.

Usage
-----

    uv run python molmetal/scripts/evaluate_anticancer_metrics.py \\
        --input <smiles.txt> --output <out.csv> [--name-column NAME] [--smiles-column SMILES]

Input format
------------

Either:

  * a plain text file with one SMILES per line (line N = SMILES N,
    optional ``# name`` comment on the same line), or
  * a CSV with two columns ``name`` and ``smiles`` (configurable
    via ``--name-column`` / ``--smiles-column``).

Output format
-------------

A CSV with one row per input molecule plus all
:func:`AnticancerMetricSuite.composite_score` columns:

    name,smiles,logP,TPSA,RotB,MW,NumHA,NumHD,iv_window_ok,mw_iv_flag,
    metal_score,gsh_evasion,dna_proxy,is_metal_complex,anticancer_index

Demo
----

The included ``molmetal/scripts/_demo_5_metaldrugs.txt`` ships 5
hand-checked reference metal drugs:

    cisplatin, carboplatin, oxaliplatin, auranofin, ruthenium-arene.

Running the script with no arguments points at this file and emits a
table to ``molmetal/reports/_demo_anticancer_metrics.csv``.

See ``molmetal/reports/todo15_anticancer_metric_suite.md`` for the
implementation report and the embedded demo table.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so the absolute imports work
# when the script is invoked directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal_lam.metrics.anticancer_metric_suite import AnticancerMetricSuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to input SMILES file (one SMILES per line, optional "
             "'# name' comment). If omitted, defaults to the 5-metal-drug demo.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to output CSV. If omitted, defaults to "
             "molmetal/reports/_demo_anticancer_metrics.csv.",
    )
    parser.add_argument(
        "--name-column", type=str, default="name",
        help="Name of the name column when reading a CSV.",
    )
    parser.add_argument(
        "--smiles-column", type=str, default="smiles",
        help="Name of the SMILES column when reading a CSV.",
    )
    return parser.parse_args()


def default_input() -> Path:
    return PROJECT_ROOT / "molmetal" / "scripts" / "_demo_5_metaldrugs.txt"


def default_output() -> Path:
    return PROJECT_ROOT / "molmetal" / "reports" / "_demo_anticancer_metrics.csv"


def read_inputs(path: Path, name_column: str, smiles_column: str):
    """Yield (name, smiles) tuples from either .txt or .csv."""
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get(name_column) or "").strip() or f"row_{reader.line_num}"
                smiles = (row.get(smiles_column) or "").strip()
                if not smiles:
                    continue
                yield name, smiles
        return

    # Plain text fallback: one SMILES per line, optional "# name" comment.
    with path.open("r") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Allow "<smiles>  # name" inline comment.
            parts = line.split("#", 1)
            smiles = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else f"row_{line_no}"
            if smiles:
                yield name, smiles


CSV_COLUMNS = [
    "name", "smiles",
    "logP", "TPSA", "RotB", "MW", "NumHA", "NumHD",
    "iv_window_ok", "mw_iv_flag",
    "metal_score", "gsh_evasion", "dna_proxy",
    "is_metal_complex", "anticancer_index",
]


def _fmt(value) -> str:
    """Format a value for CSV: floats as-is, NaN as empty, bools as 0/1."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        # NaN / inf -> empty cell (CSV-safe)
        if value != value or value in (float("inf"), float("-inf")):
            return ""
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    args = parse_args()
    suite = AnticancerMetricSuite()

    in_path = Path(args.input) if args.input else default_input()
    out_path = Path(args.output) if args.output else default_output()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[evaluate_anticancer_metrics] input : {in_path}", flush=True)
    print(f"[evaluate_anticancer_metrics] output: {out_path}", flush=True)

    n_rows = 0
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for name, smiles in read_inputs(in_path, args.name_column, args.smiles_column):
            composite = suite.composite_score(smiles)
            row = [_fmt(name), smiles] + [
                _fmt(composite.get(col)) for col in CSV_COLUMNS[2:]
            ]
            writer.writerow(row)
            n_rows += 1

    print(f"[evaluate_anticancer_metrics] wrote {n_rows} row(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
