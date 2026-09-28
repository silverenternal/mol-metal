"""Wf-252 F2 — emit m6_kappa Shape Kappa Index JSON from metallo_drugs_500_train.csv.

================================================================================
Wf-252-Apply Fix #2 (F2-M6Kappa) — additive Shape Kappa Index metric.
================================================================================

Background
----------
RDKit Bickerton 2012 QED under-rates Pt/Pd/Au coordination centers per
Casini & Pöthig 2024 (ACS Cent Sci).  The Shape Kappa Index (ChemAxon /
PMC11387870) is a normalised aggregate of RDKit Kappa1/2/3 in [0, 1] that
complements QED on metallodrug cohorts.

This script
-----------
1. Reads the same 449-SMILES metallo-drug cohort as ``metrics/by_metric/qed.json``.
2. Computes :func:`m6_kappa_from_smiles` per SMILES + the cohort mean.
3. Emits ``metrics/by_metric/m6_kappa.json`` in the standard FLAT schema
   (mirrors :file:`qed.json` field-for-field).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
- The Shape Kappa Index is a **normalised topological index**, NOT a
  learned metallodrug score.  Validate against MetalCytoToxDB labelled
  pIC50 in a follow-up workstream (not in this fix).
- QED computation is unchanged — the new metric is opt-in via
  ``metrics_v2.m6_kappa_from_smiles`` and shown side-by-side with QED
  in paper §3.

Usage
-----

    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \\
        molmetal/reports/wf252_fix/m6_kappa_emit.py

Writes ``metrics/by_metric/m6_kappa.json``.
"""

from __future__ import annotations

import csv
import json
import os as _os
import sys
from pathlib import Path
from typing import List

# Project root on sys.path so we can import molmetal.* modules.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in _os.sys.path:  # type: ignore[attr-defined]
    _os.sys.path.insert(0, str(PROJECT_ROOT))  # type: ignore[attr-defined]

try:
    from molmetal.molmetal_lam.sbdd_env.metrics_v2 import (
        m6_kappa_from_smiles,
        m6_kappa_mean,
    )
    _IMPORT_OK = True
except Exception as exc:  # pragma: no cover — best-effort
    _IMPORT_OK = False
    _IMPORT_ERR = exc
    m6_kappa_from_smiles = None  # type: ignore
    m6_kappa_mean = None  # type: ignore

DATA_CSV = PROJECT_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
METRICS_OUT = PROJECT_ROOT / "metrics" / "by_metric" / "m6_kappa.json"


def _load_smiles() -> List[str]:
    """Return the 449 metallo-drug SMILES (column 'smiles')."""
    if not DATA_CSV.exists():
        return []
    out: List[str] = []
    with open(DATA_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row.get("smiles", "").strip()
            if s:
                out.append(s)
    return out


def main() -> int:
    smiles = _load_smiles()
    n_total = len(smiles)
    if not _IMPORT_OK:
        # RDKit unavailable — emit a degraded-passthrough record.
        out = {
            "metric": "m6_kappa",
            "description": (
                "Shape Kappa Index — mean of normalised RDKit "
                "Kappa1/2/3 in [0, 1] (ChemAxon PMC11387870 convention)."
            ),
            "unit": "ratio (0-1, higher = more branched kappa shape)",
            "scope": "drug-likeness panel (metrics_v2 Wf-252 F2 add-on)",
            "source": (
                "molmetal/molmetal_lam/sbdd_env/metrics_v2.py:"
                "m6_kappa_from_smiles"
            ),
            "value": None,
            "n_samples": int(n_total),
            "data_source": "metallo_drugs_500_train.csv",
            "timestamp": "2026-09-28",
            "status": "DEGRADED_PASSTHROUGH",
            "honest_framing_note": (
                f"m6_kappa emission aborted: import failed "
                f"({_IMPORT_ERR!r}).  Emit a null record so the panel "
                f"schema stays consistent and downstream tooling can "
                f"branch on status."
            ),
        }
        METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
        METRICS_OUT.write_text(json.dumps(out, indent=2))
        print(f"[wf252-f2] DEGRADED_PASSTHROUGH: {METRICS_OUT}", flush=True)
        return 0

    vals: List[float] = []
    per_smiles: List[dict] = []
    n_parsed = 0
    for smi in smiles:
        v = m6_kappa_from_smiles(smi)
        if v is None:
            per_smiles.append({"smiles": smi[:80], "value": None})
            continue
        vals.append(v)
        n_parsed += 1
        per_smiles.append({"smiles": smi[:80], "value": float(v)})

    mean_val = (sum(vals) / len(vals)) if vals else None
    std_val = None
    if len(vals) > 1:
        import statistics
        std_val = statistics.pstdev(vals)

    out = {
        "metric": "m6_kappa",
        "description": (
            "Shape Kappa Index — mean of normalised RDKit Kappa1/2/3 "
            "in [0, 1] (ChemAxon PMC11387870 convention).  "
            "Wf-252-Apply Fix #2 (F2-M6Kappa) additive metric, "
            "complementing RDKit Bickerton 2012 QED for metallodrugs."
        ),
        "unit": "ratio (0-1, higher = more branched kappa shape)",
        "scope": "drug-likeness panel (metrics_v2 Wf-252 F2 add-on)",
        "source": (
            "molmetal/molmetal_lam/sbdd_env/metrics_v2.py:"
            "m6_kappa_from_smiles"
        ),
        "value": (float(mean_val) if mean_val is not None else None),
        "std": (float(std_val) if std_val is not None else None),
        "n_samples": int(n_total),
        "n_parsed": int(n_parsed),
        "n_failed": int(n_total - n_parsed),
        "data_source": "metallo_drugs_500_train.csv",
        "timestamp": "2026-09-28",
        "status": "MEASURED",
        "wf_tag": "Wf-252-Apply Fix #2 (F2-M6Kappa)",
        "wf_design_ref": (
            "molmetal/reports/wf252_research/fix_design.md §4 Fix F2"
        ),
        "qed_comparison": {
            "qed_value": 0.6123,
            "qed_path": "metrics/by_metric/qed.json",
            "honest_framing": (
                "Shape Kappa Index is a normalised topological index, "
                "NOT a learned metallodrug score.  Paper §3 shows it "
                "side-by-side with QED.  Validation against "
                "MetalCytoToxDB labelled pIC50 deferred to a follow-up "
                "workstream."
            ),
        },
        "honest_framing_note": (
            f"Wf-252-Apply Fix #2 (F2-M6Kappa) Shape Kappa Index "
            f"emission.  {n_parsed}/{n_total} SMILES parsed "
            f"({n_total - n_parsed} parse failures, gracefully skipped).  "
            f"Mean = {mean_val}.  Std = {std_val}.  This is the "
            f"side-by-side complement to RDKit Bickerton 2012 QED; do "
            f"NOT claim it as a learned metallodrug score without "
            f"validation against MetalCytoToxDB labelled pIC50."
        ),
        "_per_smiles_preview": per_smiles[:5],
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(out, indent=2))
    print(
        f"[wf252-f2] wrote {METRICS_OUT} n_parsed={n_parsed}/{n_total} "
        f"mean={mean_val} std={std_val}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())