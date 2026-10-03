#!/usr/bin/env python3
"""Wf-323F — m4 Success Rate SA/QED sub-criterion (CPU smoke).

======================================================================
Wf-323F — m4 Success Rate SA/QED sub-criterion (WEAK_TO_STRONG §2.4)
======================================================================

Per `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §2.4: the m4 Success Rate
criterion is

    Vina < −8.18  ∧  QED > 0.25  ∧  SA > 0.59

where the Vina sub-criterion currently binds at 0/100 on the
MEASURED cohort (cascades from m3, see WEAK_TO_STRONG §2.3). The
SA/QED sub-criterion is the AND of the two physico-chemical gates
*independent of Vina* — this script computes its rate on the canonical
449 metallo-drug SMILES cohort (per Wf-315A: SA + QED pass-through
emission).

Wf-315A normalised the per-metric thresholds to the Wf-139 P6 bands:

  * SA < 2.8 (Ertl 2009 synthesizable cutoff; lower=better)
  * QED > 0.5 (Bickerton 2012 druglikeness cutoff; higher=better)

This script emits the SA/QED sub-criterion rate as a separate
:class:`GateResult` flavour (sub_rate_sa_qed = (SA<2.8 ∧ QED>0.5) / n)
plus per-pool distribution buckets. **No Vina sub-criterion is
computed** — that requires the GPU portion (Wf-310E / Wf-319B / etc.,
USER_OK-gated per mol-metal-gpu-execution §0).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
The MEASURED sub-rate for the 449 cohort is expected to be small
because (a) the Wf-139 P6 thresholds are deliberately stringent and
(b) the metallo-drug cohort is biased toward polycyclic + click-handle
scaffolds that are intrinsically SA- and QED-penalised. The CPU
output is therefore a **sub-rate smoke** that establishes a baseline
for the Vina-lift PROJECTED delta; if the Vina sub-criterion lifts
from 0 → 0.6 on a 100-pocket retrain (Wf-264 / Wf-310E), the
*intersection* (sub_rate_sa_qed × sub_rate_vina) is the projected m4
total — but that intersection is NOT computed here (Vina = out of
scope for this CPU-only workflow).

Outputs (CPU-only, no GPU, no chemprop)
----------------------------------------
* ``--out-json PATH`` : JSON with shape::

    {
      "metric": "m4_success_rate_sa_qed_subcriterion",
      "status": "MEASURED",
      "n_input": int,
      "n_parsed": int,
      "n_failed": int,
      "thresholds": {"sa_max": 2.8, "qed_min": 0.5},
      "sub_rate_sa_qed": float,   # P(SA<2.8 ∧ QED>0.5)
      "sub_rate_sa_only": float,  # P(SA<2.8)
      "sub_rate_qed_only": float, # P(QED>0.5)
      "per_pool_distribution": {
        "sa_bucket_lt_2_8": int,
        "sa_bucket_2_8_to_4": int,
        "sa_bucket_4_to_6": int,
        "sa_bucket_6_plus": int,
        "qed_bucket_lt_0_25": int,
        "qed_bucket_0_25_to_0_5": int,
        "qed_bucket_0_5_to_0_75": int,
        "qed_bucket_0_75_plus": int,
      },
      "wf_design_ref": "TODO/03_plans/WEAK_TO_STRONG_PLAN.md §2.4",
      "wf_tag": "Wf-323F m4 SA/QED sub-criterion smoke (CPU-only)",
    }
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


# Wf-139 P6 thresholds (mirrors sa_score.json / qed_score.json contracts).
SA_MAX_THRESHOLD = 2.8
QED_MIN_THRESHOLD = 0.5


def _load_sascorer():
    """Late-import RDKit SA scorer (same trick as cfm_property_panel.py)."""
    from rdkit.Chem import RDConfig  # noqa: WPS433
    import os as _os
    sa_dir = _os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    import sascorer  # type: ignore[import-not-found]  # noqa: WPS433
    return sascorer


def _compute_one(smiles: str, sascorer) -> dict[str, Any]:
    """Return ``{"smiles": ..., "sa": float|None, "qed": float|None}``."""
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


def _read_smiles_csv(path: Path) -> list[str]:
    """Read the canonical metallo_drugs_500_train.csv 'smiles' column.

    Same semantics as cfm_property_panel.py:stripped, drop blanks, raise
    if no 'smiles' column.  We do NOT touch RDKit here — SMILES are
    returned as raw strings.
    """
    out: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header: list[str] | None = None
        col_idx: int = -1
        for row in reader:
            if not row:
                continue
            if header is None:
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
                out.append(s)
    return out


def _bucket(per_molecule: list[dict[str, Any]]) -> dict[str, int]:
    """Bucket parsed molecules by SA / QED bands.

    Returns counts for the 8 buckets in the spec. Failed-to-parse
    molecules are NOT bucketed (they show up in n_failed).
    """
    buckets = {
        "sa_bucket_lt_2_8": 0,
        "sa_bucket_2_8_to_4": 0,
        "sa_bucket_4_to_6": 0,
        "sa_bucket_6_plus": 0,
        "qed_bucket_lt_0_25": 0,
        "qed_bucket_0_25_to_0_5": 0,
        "qed_bucket_0_5_to_0_75": 0,
        "qed_bucket_0_75_plus": 0,
    }
    for entry in per_molecule:
        sa = entry.get("sa")
        qed = entry.get("qed")
        if isinstance(sa, (int, float)):
            if sa < 2.8:
                buckets["sa_bucket_lt_2_8"] += 1
            elif sa < 4.0:
                buckets["sa_bucket_2_8_to_4"] += 1
            elif sa < 6.0:
                buckets["sa_bucket_4_to_6"] += 1
            else:
                buckets["sa_bucket_6_plus"] += 1
        if isinstance(qed, (int, float)):
            if qed < 0.25:
                buckets["qed_bucket_lt_0_25"] += 1
            elif qed < 0.5:
                buckets["qed_bucket_0_25_to_0_5"] += 1
            elif qed < 0.75:
                buckets["qed_bucket_0_5_to_0_75"] += 1
            else:
                buckets["qed_bucket_0_75_plus"] += 1
    return buckets


def compute_sub_rate(smiles_csv: Path) -> dict[str, Any]:
    """Compute the m4 SA/QED sub-criterion rate from the source CSV.

    Returns a dict matching the spec in the module docstring. Uses
    the canonical Wf-139 P6 thresholds (SA<2.8 + QED>0.5).
    """
    smiles_list = _read_smiles_csv(smiles_csv)
    sascorer = _load_sascorer()
    per_molecule = [_compute_one(s, sascorer) for s in smiles_list]

    parsed = [m for m in per_molecule
              if isinstance(m.get("sa"), (int, float))
              and isinstance(m.get("qed"), (int, float))]
    n_input = len(per_molecule)
    n_parsed = len(parsed)
    n_failed = n_input - n_parsed

    sub_sa_qed = sum(
        1 for m in parsed
        if m["sa"] < SA_MAX_THRESHOLD and m["qed"] > QED_MIN_THRESHOLD
    )
    sub_sa_only = sum(1 for m in parsed if m["sa"] < SA_MAX_THRESHOLD)
    sub_qed_only = sum(1 for m in parsed if m["qed"] > QED_MIN_THRESHOLD)

    return {
        "metric": "m4_success_rate_sa_qed_subcriterion",
        "status": "MEASURED",
        "n_input": n_input,
        "n_parsed": n_parsed,
        "n_failed": n_failed,
        "thresholds": {
            "sa_max": SA_MAX_THRESHOLD,
            "qed_min": QED_MIN_THRESHOLD,
        },
        "sub_rate_sa_qed": (sub_sa_qed / n_parsed) if n_parsed > 0 else 0.0,
        "sub_rate_sa_only": (sub_sa_only / n_parsed) if n_parsed > 0 else 0.0,
        "sub_rate_qed_only": (sub_qed_only / n_parsed) if n_parsed > 0 else 0.0,
        "n_sub_rate_sa_qed_pass": sub_sa_qed,
        "n_sub_rate_sa_only_pass": sub_sa_only,
        "n_sub_rate_qed_only_pass": sub_qed_only,
        "per_pool_distribution": _bucket(parsed),
        "wf_design_ref": "TODO/03_plans/WEAK_TO_STRONG_PLAN.md §2.4",
        "wf_tag": "Wf-323F m4 SA/QED sub-criterion smoke (CPU-only)",
        "honest_framing_note": (
            "MEASURED on the 449 metallo-drug cohort (Wf-315A canonical "
            "pass-through). Sub-rate is the SA/QED AND rate independent of "
            "Vina — Vina sub-criterion binding at 0/100 per WEAK_TO_STRONG "
            "§2.4 (cascades from m3). m4 = SA/QED sub-rate ∩ Vina sub-rate; "
            "Vina portion is OUT OF SCOPE for this CPU smoke (USER_OK-gated "
            "GPU work per mol-metal-gpu-execution §0)."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wf323f_m4_sa_qed_subcriterion.py",
        description=(
            "Wf-323F: m4 Success Rate SA/QED sub-criterion smoke (CPU-only). "
            "Reads the canonical metallo-drug SMILES CSV, computes per-molecule "
            "SA + QED, and emits the AND rate (SA<2.8 ∧ QED>0.5) plus per-bucket "
            "distribution. NO Vina sub-criterion is computed here."
        ),
    )
    p.add_argument(
        "--smiles-csv", required=True, type=Path,
        help="CSV with a 'smiles' column (canonical: metallo_drugs_500_train.csv).",
    )
    p.add_argument(
        "--out-json", required=True, type=Path,
        help="Output JSON path (will be created or overwritten).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = compute_sub_rate(args.smiles_csv)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(
        f"[wf323f] n_input={result['n_input']} "
        f"n_parsed={result['n_parsed']} "
        f"sub_rate_sa_qed={result['sub_rate_sa_qed']:.4f} "
        f"sa_only={result['sub_rate_sa_only']:.4f} "
        f"qed_only={result['sub_rate_qed_only']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
