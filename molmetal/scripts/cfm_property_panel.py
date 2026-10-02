#!/usr/bin/env python3
"""Wf-264-SA — CFM generated molecule SA + QED property panel.

======================================================================
Wf-264-SA — SA (Synthetic Accessibility) + QED (Quantitative Estimate
of Drug-likeness) panel for CFM-generated molecules.
======================================================================

Reads a CSV with a ``smiles`` column (one molecule per row), computes the
RDKit SA score (Ertl & Schuffenhauer 2009, 1-10 scale, **lower = more
synthesizable**) and the RDKit QED score (Bickerton 2012, 0-1 scale,
**higher = more drug-like**) for each molecule, and emits a JSON file
with per-molecule results plus aggregate statistics.

Why this panel exists
---------------------
* **SA score** measures how hard a target molecule is to synthesize.
  It is the standard panel metric used by GuacaMol, MolDQN, and most
  CFM/diffusion generative chemistry papers (Ertl & Schuffenhauer 2009).
* **QED** is Bickerton 2012's weighted multi-property drug-likeness
  estimate (MW / LogP / HBA / HBD / PSA / ROTB / AROM / ALERTS).
* Together they form the canonical "is the generated molecule any
  good?" diagnostic — without them, decode_ratio is meaningless
  because a model could "succeed" at decoding garbage.

Inputs
------
* ``--smiles-csv PATH``  CSV with at minimum a ``smiles`` column.

Outputs
-------
* ``--out-json PATH``  JSON with shape::

    {
      "panel_version": "wf264-sa/v1",
      "n_input": int,
      "n_parsed": int,
      "n_failed": int,
      "per_molecule": [
        {"smiles": str, "sa": float, "qed": float},
        ...
      ],
      "aggregate": {
        "sa": {"mean": float, "std": float,
               "p10": float, "p25": float, "p50": float,
               "p75": float, "p90": float, "min": float, "max": float,
               "n": int},
        "qed": { ... same shape ... },
        "parsed_ratio": float,
        "n_input": int,
        "n_parsed": int,
        "n_failed": int
      }
    }

  ``sa`` / ``qed`` are ``null`` for SMILES that fail to parse.

Usage
-----
::

    .venv/bin/python molmetal/scripts/cfm_property_panel.py \\
        --smiles-csv /tmp/cfm_generated.csv \\
        --out-json molmetal/reports/wf264_sa_qed/panel.json

Constraints
-----------
* RDKit is mandatory; SA/QED are standard RDKit calls.
* No model checkpoint is loaded; this panel is **post-hoc** — it
  operates on any SMILES file.
* Honest-framing: ``n_parsed / n_input`` is the parse-rate; a low
  parse-rate is itself a HALT signal worth reporting.

Attribution
-----------
* SA score: Ertl P., Schuffenhauer A. (2009) ``Estimation of synthetic
  accessibility score of drug-like molecules based on molecular
  complexity and fragment contributions``.
* QED: Bickerton G.R. et al. (2012) ``Quantifying the chemical beauty
  of drugs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


PANEL_VERSION = "wf264-sa/v1"


def _load_sascorer():
    """Late-import RDKit's bundled SA scorer with the upstream import path.

    The standard rdkit Contrib SA_Score package is importable as a
    regular Python module from its install dir; we import it lazily so
    the script's docstring/help works even when RDKit SA data is
    missing.  Returns the module (which exposes ``calculateScore``).
    """
    from rdkit.Chem import RDConfig  # noqa: WPS433
    import os as _os
    sa_dir = _os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    import sascorer  # type: ignore[import-not-found]  # noqa: WPS433
    return sascorer


def _percentiles(values: list[float]) -> dict[str, float]:
    """Return mean / std / percentiles / min / max for non-NaN values.

    NaN values are filtered out before computing statistics; if no
    values remain, all keys are returned as ``None``.
    """
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return {
            "mean": None, "std": None,
            "p10": None, "p25": None, "p50": None,
            "p75": None, "p90": None, "min": None, "max": None,
            "n": 0,
        }
    n = len(clean)
    mean = sum(clean) / n
    var = sum((v - mean) ** 2 for v in clean) / n if n > 0 else 0.0
    std = math.sqrt(var)
    srt = sorted(clean)

    def _q(q: float) -> float:
        # Linear-interpolated quantile on the sorted sample.
        if n == 1:
            return srt[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return srt[lo]
        frac = pos - lo
        return srt[lo] + (srt[hi] - srt[lo]) * frac

    return {
        "mean": mean,
        "std": std,
        "p10": _q(0.10),
        "p25": _q(0.25),
        "p50": _q(0.50),
        "p75": _q(0.75),
        "p90": _q(0.90),
        "min": srt[0],
        "max": srt[-1],
        "n": n,
    }


def compute_one(smiles: str, sascorer) -> dict[str, Any]:
    """Compute SA + QED for a single SMILES string.

    Returns ``{"smiles": ..., "sa": float|None, "qed": float|None}``.
    ``sa`` / ``qed`` are ``None`` if RDKit cannot parse the SMILES.
    """
    from rdkit import Chem  # noqa: WPS433
    from rdkit.Chem import QED  # noqa: WPS433

    sa_val: Any = None
    qed_val: Any = None
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            sa_val = float(sascorer.calculateScore(mol))
        except Exception:  # noqa: BLE001
            sa_val = None
        try:
            qed_val = float(QED.qed(mol))
        except Exception:  # noqa: BLE001
            qed_val = None
    return {"smiles": smiles, "sa": sa_val, "qed": qed_val}


def read_smiles_csv(path: Path) -> list[str]:
    """Read a CSV with a ``smiles`` column.

    Accepts both quoted and unquoted SMILES, strips whitespace, drops
    empty rows.  Raises ``ValueError`` if no ``smiles`` column is found.
    """
    smiles_list: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header: list[str] | None = None
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if header is None:
                # Normalise header and find the smiles column index.
                lowered = [c.strip().lower() for c in row]
                if "smiles" not in lowered:
                    raise ValueError(
                        f"CSV at {path} has no 'smiles' column "
                        f"(header={row!r})"
                    )
                header = row
                col_idx = lowered.index("smiles")
                continue
            if col_idx >= len(row):
                continue
            s = row[col_idx].strip()
            if s:
                smiles_list.append(s)
    return smiles_list


def build_panel(smiles_list: list[str]) -> dict[str, Any]:
    """Compute the full SA+QED panel for an iterable of SMILES."""
    sascorer = _load_sascorer()
    per_molecule = [compute_one(s, sascorer) for s in smiles_list]

    sa_vals = [m["sa"] for m in per_molecule]
    qed_vals = [m["qed"] for m in per_molecule]
    n_parsed = sum(
        1 for m in per_molecule
        if m["sa"] is not None and m["qed"] is not None
    )
    n_input = len(per_molecule)
    parsed_ratio = (n_parsed / n_input) if n_input > 0 else 0.0

    aggregate = {
        "sa": _percentiles([v for v in sa_vals if v is not None]),
        "qed": _percentiles([v for v in qed_vals if v is not None]),
        "parsed_ratio": parsed_ratio,
        "n_input": n_input,
        "n_parsed": n_parsed,
        "n_failed": n_input - n_parsed,
    }

    return {
        "panel_version": PANEL_VERSION,
        "n_input": n_input,
        "n_parsed": n_parsed,
        "n_failed": n_input - n_parsed,
        "per_molecule": per_molecule,
        "aggregate": aggregate,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cfm_property_panel.py",
        description=(
            "Wf-264-SA: SA + QED property panel for CFM-generated "
            "(or any) SMILES list.  Computes RDKit SA score "
            "(Ertl 2009, lower=better) + QED (Bickerton 2012, "
            "higher=better) per molecule and emits aggregate JSON."
        ),
    )
    p.add_argument(
        "--smiles-csv", required=True, type=Path,
        help="CSV with a 'smiles' column (one molecule per row).",
    )
    p.add_argument(
        "--out-json", required=True, type=Path,
        help="Output JSON path; will be created or overwritten.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    smiles_list = read_smiles_csv(args.smiles_csv)
    panel = build_panel(smiles_list)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as fh:
        json.dump(panel, fh, indent=2)
    print(
        f"[wf264-sa] n_input={panel['n_input']} "
        f"n_parsed={panel['n_parsed']} "
        f"n_failed={panel['n_failed']} "
        f"parsed_ratio={panel['aggregate']['parsed_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())