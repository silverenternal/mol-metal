#!/usr/bin/env python3
"""Wf-380-A6 — CFM SA-aware panel (per-molecule SA evaluation + aggregation).

================================================================
Why this script exists
================================================================
The CFM property panel
(:mod:`molmetal.scripts.cfm_property_panel.py`, Wf-264-SA) computes
SA + QED per-molecule from a SMILES CSV.  Wf-380-A6 extends that
panel with the **SA penalty** layer (TODO-205 / WEAK_TO_STRONG §2.4):

* ``per_molecule_sa_evaluation`` — compute SA score per molecule
  via :func:`molmetal.adapters.flow_matching_lipman.sa_penalty.compute_sa_score`.
* ``aggregate_sa_distribution`` — aggregate SA distribution + penalty
  stats across the cohort, including
  ``mean_sa`` / ``median_sa`` / ``std_sa`` / ``fraction_sa_le_2.8`` /
  ``fraction_sa_le_4.0`` / ``mean_penalty`` / ``fraction_above_target``.

The script supports two modes:

1. ``--smiles-csv PATH`` — read SMILES from a CSV with a ``smiles`` column.
2. ``--smiles-literal S1,S2,...`` — supply SMILES directly on the CLI for
   smoke testing.

The output JSON matches the schema of ``cfm_property_panel.py`` so
downstream dashboards can ingest it via the same parser.

Wf-380-A6 honest framing (party-twenty-spirit §3.5):
* SCAFFOLD aggregator — integration with the CFM trainer (i.e. using
  the penalty to nudge sampling) is deferred to Wf-380-B1.
* PROJECTED lift on m4 SA: ``3.099 -> 2.5-2.7`` (WEAK_TO_STRONG §3.6).
* No GPU launches; pure CPU RDKit SA evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Optional numpy (only used for percentiles + mean — keep the
# import lazy so the script is usable in minimal envs).
try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Lazy imports of the SA penalty adapter module
# ---------------------------------------------------------------------------

def _load_sa_penalty_module():
    """Import :mod:`molmetal.adapters.flow_matching_lipman.sa_penalty`.

    Defensive wrapper: returns ``None`` if the module is not on
    ``sys.path`` (the script may be invoked directly via
    ``python molmetal/scripts/cfm_sa_aware_panel.py`` without the
    project root on ``sys.path``).  In that case we add the project
    root derived from ``__file__`` (``../../`` relative to
    ``molmetal/scripts/``) before re-attempting the import.
    """
    import importlib
    try:
        return importlib.import_module("molmetal.adapters.flow_matching_lipman.sa_penalty")
    except Exception:
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        # __file__ = <repo>/molmetal/scripts/cfm_sa_aware_panel.py
        # we want <repo>/ on sys.path
        repo_root = os.path.abspath(os.path.join(here, "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        return importlib.import_module("molmetal.adapters.flow_matching_lipman.sa_penalty")
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"[cfm_sa_aware_panel] WARN cannot import sa_penalty: {exc}\n")
        return None


# ---------------------------------------------------------------------------
# Per-molecule evaluation
# ---------------------------------------------------------------------------

def per_molecule_sa_evaluation(
    mols: Sequence,
    target_sa: float = 2.8,
    sa_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """Compute SA + SA penalty for a list of RDKit ``Mol`` objects.

    Parameters
    ----------
    mols : sequence of ``rdkit.Chem.Mol``
        Molecules to evaluate.  ``None`` entries are recorded as
        ``{"smiles": None, "sa": NaN, "penalty": NaN, "status": "NO_MOL"}``.
    target_sa : float
        Synthesizable threshold passed to :func:`sa_aware_sampling_penalty`.
    sa_weight : float
        Linear penalty weight.

    Returns
    -------
    list of dict
        Per-molecule result dicts with keys ``smiles`` (best-effort),
        ``sa`` (float or NaN), ``penalty`` (float or NaN),
        ``status`` (``OK`` | ``NO_MOL`` | ``SA_FAIL``).
    """
    mod = _load_sa_penalty_module()
    results: List[Dict[str, Any]] = []
    for mol in mols:
        if mol is None:
            results.append(
                {"smiles": None, "sa": float("nan"), "penalty": float("nan"),
                 "status": "NO_MOL"}
            )
            continue
        sa = float(mod.compute_sa_score(mol)) if mod is not None else float("nan")
        if math.isnan(sa):
            results.append(
                {"smiles": None, "sa": float("nan"), "penalty": float("nan"),
                 "status": "SA_FAIL"}
            )
            continue
        penalty = float(
            mod.sa_aware_sampling_penalty([sa], target_sa=target_sa, sa_weight=sa_weight)[0]
        ) if mod is not None else 0.0
        # Best-effort SMILES round-trip for downstream trace.
        try:
            from rdkit import Chem
            smi = Chem.MolToSmiles(mol)
        except Exception:
            smi = None
        results.append({"smiles": smi, "sa": sa, "penalty": penalty, "status": "OK"})
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_sa_distribution(
    cohort_results: Sequence[Dict[str, Any]],
    target_sa: float = 2.8,
) -> Dict[str, Any]:
    """Aggregate per-cohort SA penalty + distribution stats.

    Parameters
    ----------
    cohort_results : list of dict
        Output of :func:`per_molecule_sa_evaluation`.
    target_sa : float
        Synthesizable threshold used in the penalty calculation.

    Returns
    -------
    dict
        Aggregate stats with keys:

        ``n_total``      — number of cohort entries (including NO_MOL / SA_FAIL)
        ``n_ok``         — number of entries with status == "OK"
        ``mean_sa``      — mean SA across ``n_ok`` entries
        ``median_sa``    — median SA
        ``std_sa``       — std SA
        ``min_sa``, ``max_sa`` — SA range
        ``fraction_sa_le_2.8`` — fraction with SA <= 2.8 (synthesizable)
        ``fraction_sa_le_4.0`` — fraction with SA <= 4.0 (synth with effort)
        ``mean_penalty`` — mean sa_aware_sampling_penalty
        ``fraction_above_target`` — fraction with SA > target_sa
        ``target_sa``    — threshold echoed for downstream traceability
    """
    ok_sa: List[float] = []
    ok_pen: List[float] = []
    n_total = len(cohort_results)
    n_ok = 0
    for r in cohort_results:
        if r.get("status") != "OK":
            continue
        n_ok += 1
        sa = float(r["sa"])
        pen = float(r["penalty"])
        if not math.isnan(sa):
            ok_sa.append(sa)
        if not math.isnan(pen):
            ok_pen.append(pen)

    if _HAS_NUMPY and ok_sa:
        sa_arr = np.asarray(ok_sa, dtype=np.float64)
        mean_sa = float(np.mean(sa_arr))
        median_sa = float(np.median(sa_arr))
        std_sa = float(np.std(sa_arr))
        min_sa = float(np.min(sa_arr))
        max_sa = float(np.max(sa_arr))
        frac_le_28 = float(np.mean(sa_arr <= 2.8))
        frac_le_40 = float(np.mean(sa_arr <= 4.0))
        frac_above_target = float(np.mean(sa_arr > target_sa))
    elif ok_sa:
        mean_sa = sum(ok_sa) / len(ok_sa)
        sorted_sa = sorted(ok_sa)
        mid = len(sorted_sa) // 2
        median_sa = (
            sorted_sa[mid]
            if len(sorted_sa) % 2 == 1
            else 0.5 * (sorted_sa[mid - 1] + sorted_sa[mid])
        )
        var = sum((x - mean_sa) ** 2 for x in ok_sa) / len(ok_sa)
        std_sa = math.sqrt(var)
        min_sa = min(ok_sa)
        max_sa = max(ok_sa)
        frac_le_28 = sum(1 for x in ok_sa if x <= 2.8) / len(ok_sa)
        frac_le_40 = sum(1 for x in ok_sa if x <= 4.0) / len(ok_sa)
        frac_above_target = sum(1 for x in ok_sa if x > target_sa) / len(ok_sa)
    else:
        mean_sa = median_sa = std_sa = float("nan")
        min_sa = max_sa = frac_le_28 = frac_le_40 = frac_above_target = float("nan")

    mean_penalty = (
        sum(ok_pen) / len(ok_pen) if ok_pen else float("nan")
    )

    return {
        "n_total": int(n_total),
        "n_ok": int(n_ok),
        "mean_sa": float(mean_sa),
        "median_sa": float(median_sa),
        "std_sa": float(std_sa),
        "min_sa": float(min_sa),
        "max_sa": float(max_sa),
        "fraction_sa_le_2_8": float(frac_le_28),
        "fraction_sa_le_4_0": float(frac_le_40),
        "mean_penalty": float(mean_penalty),
        "fraction_above_target": float(frac_above_target),
        "target_sa": float(target_sa),
    }


# ---------------------------------------------------------------------------
# CSV / CLI helpers
# ---------------------------------------------------------------------------

def read_smiles_csv(path: Path) -> List[str]:
    """Read SMILES from a CSV with a ``smiles`` column.

    Strips ``\n`` / whitespace and drops empty rows.
    """
    out: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "smiles" not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV {path} missing required 'smiles' column; "
                f"found columns: {reader.fieldnames}"
            )
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                out.append(smi)
    return out


def _parse_smiles_literal(literal: str) -> List[str]:
    return [s.strip() for s in literal.split(",") if s.strip()]


def build_panel(
    smiles_list: Sequence[str],
    target_sa: float = 2.8,
    sa_weight: float = 0.3,
) -> Dict[str, Any]:
    """Build the full SA-aware panel from a SMILES list.

    Returns a dict with::

        {
          "panel_version": "wf380a6/v1",
          "n_input": int,
          "n_parsed": int,
          "n_failed": int,
          "per_molecule": [ {smiles, sa, penalty, status}, ... ],
          "aggregate": aggregate_sa_distribution(...)
        }

    Mirrors the schema of ``cfm_property_panel.build_panel`` so
    downstream dashboards can ingest via the same parser.
    """
    from rdkit import Chem

    mols: List[Any] = []
    parse_status: List[str] = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            mol = None
        if mol is None:
            parse_status.append("PARSE_FAIL")
        else:
            parse_status.append("OK")
        mols.append(mol)

    per_mol = per_molecule_sa_evaluation(mols, target_sa=target_sa, sa_weight=sa_weight)
    aggregate = aggregate_sa_distribution(per_mol, target_sa=target_sa)
    n_parsed = sum(1 for s in parse_status if s == "OK")
    n_failed = len(parse_status) - n_parsed
    return {
        "panel_version": "wf380a6/v1",
        "n_input": len(smiles_list),
        "n_parsed": int(n_parsed),
        "n_failed": int(n_failed),
        "per_molecule": per_mol,
        "aggregate": aggregate,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cfm_sa_aware_panel",
        description=(
            "Wf-380-A6 — CFM SA-aware panel (TODO-205). Computes SA score + "
            "SA-aware sampling penalty for a SMILES cohort and emits an "
            "aggregate JSON."
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles-csv", type=str, help="CSV with a 'smiles' column.")
    src.add_argument(
        "--smiles-literal",
        type=str,
        help="Comma-separated SMILES list (smoke mode).",
    )
    p.add_argument("--out-json", type=str, required=True, help="Output JSON path.")
    p.add_argument("--target-sa", type=float, default=2.8,
                   help="Synthesizable threshold (default 2.8, Wf-139 P6).")
    p.add_argument("--sa-weight", type=float, default=0.3,
                   help="Linear SA penalty weight (default 0.3, "
                        "matches metrics/by_metric/sa_mean.json 2026-09-14).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smiles_csv:
        smiles_list = read_smiles_csv(Path(args.smiles_csv))
    else:
        smiles_list = _parse_smiles_literal(args.smiles_literal)

    panel = build_panel(
        smiles_list,
        target_sa=float(args.target_sa),
        sa_weight=float(args.sa_weight),
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(panel, fh, indent=2, default=lambda o: float(o) if isinstance(o, float) else str(o))
    sys.stdout.write(
        f"[cfm_sa_aware_panel] wrote {out_path} (n_input={panel['n_input']} "
        f"n_parsed={panel['n_parsed']} n_failed={panel['n_failed']} "
        f"mean_sa={panel['aggregate']['mean_sa']:.4f} "
        f"fraction_sa_le_2.8={panel['aggregate']['fraction_sa_le_2_8']:.4f})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())