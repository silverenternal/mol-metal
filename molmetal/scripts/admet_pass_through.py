#!/usr/bin/env python3
"""Wf-303C T1 / R03 — ADMET pass-through metric emission.

================================================================================
Wf-303C T1 — close 3 honest-negative ADMET metrics with one CPU pass-through
================================================================================

Background (per `molmetal/reports/wf297d_remediation_plan/remediation_plan.md`
§R03):

> "Add `molmetal/scripts/admet_pass_through.py`: take 449 metallo-drug SMILES
> (existing in `metallo_drugs_500_train.csv`), call `rdkit.Chem.Descriptors`
> for logP/Crippen + `rdkit.Chem.FilterCatalog` for mutagen; emit 3 JSON files."

This script closes 3 honest-negatives from `wf297c_gap_compute/gap_table.md`
Section A N12/N13/N14 with **pure RDKit pass-through** — no retrain, no GPU,
no chemprop.  The three metrics are:

1. **ames_mutagen** — AMES mutagenicity risk (RDKit FilterCatalog + structural
   alerts).  Target band per Wf-297D: **< 0.30** (mean across cohort).
   Ref: Benigni-Bossa 2012; Kazius 2005; RDKit `FilterCatalog` 2024.
2. **cell_permeability_logPapp** — Caco-2 / MDCK log Papp proxy from
   Crippen logP + TPSA.  Target band per Wf-297D: **> -6.0** (i.e. Papp >
   10^-6 cm/s).  Ref: Veber 2002; Hou 2007.
3. **aqueous_solubility_logS** — aqueous solubility logS from Crippen logP +
   MW + aromatic ring count.  Target band per Wf-297D: **> -4.0** (i.e.
   0.1 mM solubility).  Ref: Delaney 2004 ESOL; Hou 2007.

These three ADMET metrics are already implemented as per-SMILES functions in
:mod:`molmetal.molmetal_lam.sbdd_env.metrics_v2` — `ames_mutagen`,
`cell_permeability_logPapp`, `aqueous_solubility_logS`.  The pass-through is
**the cheapest possible honest-negative close**: call the existing function,
write the per-cohort mean + std + n_parsed + n_failed to 3 JSON files in the
standard FLAT schema (mirrors `qed.json` field-for-field).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
- The three ADMET metrics are **descriptive physico-chemical proxies**, NOT
  learned metallodrug scores.  They are RDKit / Crippen / Veber / Delaney
  heuristics — useful for cohort-level drug-likeness panels but **not**
  substitutable for actual AMES assay / Caco-2 / kinetic-solubility
  measurements.
- Wf-297D §R03 lists these as "AMES / cell_permeability_logPapp /
  aqueous_solubility_logS not run → < 0.30 / > -6.0 / > -4.0".  This script
  closes the gap by RUNNING them and recording MEASURED values.
- The pass-through cohort is the 449 metallo-drug SMILES in
  `metallo_drugs_500_train.csv` (same cohort as `qed.json`,
  `m6_kappa.json`, etc.).  Cross-cohort consistency is enforced by the
  `n_samples` / `n_parsed` / `n_failed` fields.

Usage
-----

    # Default cohort (449 metallo-drug SMILES)
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/admet_pass_through.py

    # Custom SMILES list (one SMILES per line, no header)
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/admet_pass_through.py --input my_smiles.smi

Outputs
-------
``metrics/by_metric/ames_mutagen.json``
``metrics/by_metric/cell_permeability_logPapp.json``
``metrics/by_metric/aqueous_solubility_logS.json``

Each JSON has the standard FLAT schema (mirror of `qed.json` /
`m6_kappa.json`).
"""

from __future__ import annotations

import argparse
import csv
import json
import os as _os
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Project root on sys.path so we can import molmetal.* modules.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in _os.sys.path:  # type: ignore[attr-defined]
    _os.sys.path.insert(0, str(PROJECT_ROOT))  # type: ignore[attr-defined]

try:
    from molmetal.molmetal_lam.sbdd_env.metrics_v2 import (
        ames_mutagen,
        ames_mutagen_mean,
        aqueous_solubility_logS,
        aqueous_solubility_logS_mean,
        bbb,
        bbb_mean,
        caco2_perm,
        caco2_perm_mean,
        cell_permeability_logPapp,
        cell_permeability_logPapp_mean,
        clearance,
        clearance_mean,
        half_life,
        half_life_mean,
        qed_score,
        qed_score_mean,
        sa_score,
        sa_score_mean,
        vdss,
        vdss_mean,
    )

    _IMPORT_OK = True
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover — best-effort
    _IMPORT_OK = False
    _IMPORT_ERR = exc
    ames_mutagen = None  # type: ignore
    ames_mutagen_mean = None  # type: ignore
    aqueous_solubility_logS = None  # type: ignore
    aqueous_solubility_logS_mean = None  # type: ignore
    bbb = None  # type: ignore
    bbb_mean = None  # type: ignore
    caco2_perm = None  # type: ignore
    caco2_perm_mean = None  # type: ignore
    cell_permeability_logPapp = None  # type: ignore
    cell_permeability_logPapp_mean = None  # type: ignore
    clearance = None  # type: ignore
    clearance_mean = None  # type: ignore
    half_life = None  # type: ignore
    half_life_mean = None  # type: ignore
    qed_score = None  # type: ignore
    qed_score_mean = None  # type: ignore
    sa_score = None  # type: ignore
    sa_score_mean = None  # type: ignore
    vdss = None  # type: ignore
    vdss_mean = None  # type: ignore

DEFAULT_DATA_CSV = PROJECT_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "metrics" / "by_metric"


# Per-metric schema definition.  Each entry is (function, mean_function,
# target_band_string, output_filename).
#
# Wf-312C — ADMET extension: 8 metrics total (3 Wf-303C T1 + 5 NEW
# Wf-312C R01).  All 8 are pure RDKit descriptor-based pass-through
# (no model, no GPU); 5 NEW are heuristics from Hou 2007 (caco2_perm),
# Clark 1999 (bbb), Lobell/Obach 1996-1997 (clearance), Obach 1997
# (half_life), and Lombardo/Obach (vdss).
METRIC_SPECS: List[Tuple[str, str, Callable, Callable, str, str]] = [
    # ---- 3 Wf-303C T1 / Wf-308B R03 ADMET pass-through (already shipped) ----
    (
        "ames_mutagen",
        "ratio (0-1, lower = less mutagenic; AMES FilterCatalog structural alerts)",
        ames_mutagen if _IMPORT_OK else None,
        ames_mutagen_mean if _IMPORT_OK else None,
        "< 0.30 (cohort mean)",
        "ames_mutagen.json",
    ),
    (
        "cell_permeability_logPapp",
        "log Papp proxy (Crippen logP + TPSA, higher = better permeability; -6 = Papp 1e-6 cm/s)",
        cell_permeability_logPapp if _IMPORT_OK else None,
        cell_permeability_logPapp_mean if _IMPORT_OK else None,
        "> -6.0 (i.e. Papp > 1e-6 cm/s)",
        "cell_permeability_logPapp.json",
    ),
    (
        "aqueous_solubility_logS",
        "log S proxy (Crippen logP + MW + aromatic rings; higher = more soluble)",
        aqueous_solubility_logS if _IMPORT_OK else None,
        aqueous_solubility_logS_mean if _IMPORT_OK else None,
        "> -4.0 (i.e. 0.1 mM solubility)",
        "aqueous_solubility_logS.json",
    ),
    # ---- 5 NEW Wf-312C R01 ADMET extension metrics ----
    (
        "caco2_perm",
        "Caco-2 logPapp from Hou 2007 refined regression (5-descriptor); higher = better permeability",
        caco2_perm if _IMPORT_OK else None,
        caco2_perm_mean if _IMPORT_OK else None,
        "> -5.15 (well-permeable threshold per Hou 2007)",
        "caco2_perm.json",
    ),
    (
        "bbb",
        "logBB blood-brain barrier penetration (Clark 1999 regression on TPSA + logP); higher = more BBB",
        bbb if _IMPORT_OK else None,
        bbb_mean if _IMPORT_OK else None,
        "-1 < logBB < 1 (BBB+/- threshold per Clark 1999)",
        "bbb.json",
    ),
    (
        "clearance",
        "Hepatic intrinsic clearance CLint in mL/min/kg (Lobell/Obach 1996-1997); typical 5-30",
        clearance if _IMPORT_OK else None,
        clearance_mean if _IMPORT_OK else None,
        "5-30 mL/min/kg (typical oral drug sweet spot)",
        "clearance.json",
    ),
    (
        "half_life",
        "Elimination half-life t1/2 in hours (Obach 1997); higher = slower elimination",
        half_life if _IMPORT_OK else None,
        half_life_mean if _IMPORT_OK else None,
        "1-24 hours (oral drug sweet spot)",
        "half_life.json",
    ),
    (
        "vdss",
        "Volume of distribution at steady state in L/kg (Lombardo 2006 / Obach 1997); higher = more tissue",
        vdss if _IMPORT_OK else None,
        vdss_mean if _IMPORT_OK else None,
        "0.5-10 L/kg (typical drug range)",
        "vdss.json",
    ),
    # ---- 2 NEW Wf-315A SA + QED evaluators (WEAK_TO_STRONG §3.6) ----
    # Per WEAK_TO_STRONG_PLAN.md §3.6: m5 SA (Ertl-Schuffenhauer 2009) +
    # m6 QED (Bickerton 2012) evaluators were implemented in
    # sa_score.py / qed_scorer.py but not wired into the panel.  Wire
    # them here as 2 NEW pass-through entries; the pass-through cohort
    # is the same 449 metallo-drug SMILES as the rest of the Wf-303C /
    # Wf-312C panel.  Verdict logic per Wf-139 P6 specs:
    #   * sa < 2.8 → PASS (synthesizable); SA 2.8-4.0 → WARN; SA >= 4.0 → FAIL
    #   * qed > 0.5 → PASS (drug-like); qed <= 0.5 → FAIL honest-negative
    # Lit: Ertl-Schuffenhauer 2009 (J. Cheminf. 1:11); Bickerton 2012
    # (Nat. Chem. 4:90).  Both are RDKit pass-through heuristics — NOT
    # learned metallodrug scores.
    (
        "sa_score",
        "Synthetic accessibility score (Ertl 2009) in [1, 10]; lower = easier to synthesize; 2.8 = synthesizable threshold per Wf-139 P6",
        sa_score if _IMPORT_OK else None,
        sa_score_mean if _IMPORT_OK else None,
        "< 2.8 (Wf-139 P6 synthesizable threshold); 2.8-4.0 WARN; >= 4.0 FAIL",
        "sa_score.json",
    ),
    (
        "qed_score",
        "Quantitative estimate of drug-likeness (Bickerton 2012) in [0, 1]; higher = more drug-like; 0.5 = druglikeness threshold per Wf-139 P6",
        qed_score if _IMPORT_OK else None,
        qed_score_mean if _IMPORT_OK else None,
        "> 0.5 (Wf-139 P6 druglikeness threshold)",
        "qed_score.json",
    ),
]


def _load_smiles_from_csv(path: Path) -> List[str]:
    """Load SMILES from a CSV with a `smiles` column (default)."""
    if not path.exists():
        return []
    out: List[str] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row.get("smiles", "").strip()
            if s:
                out.append(s)
    return out


def _load_smiles_from_text(path: Path) -> List[str]:
    """Load one-SMILES-per-line text file (no header)."""
    if not path.exists():
        return []
    out: List[str] = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _load_smiles(path: Path) -> List[str]:
    """Dispatch on file extension to choose CSV vs plain-text loader."""
    if path.suffix.lower() == ".csv":
        return _load_smiles_from_csv(path)
    return _load_smiles_from_text(path)


def _compute_metric_for_smiles(
    smiles_list: List[str], fn: Callable[[str], Optional[float]]
) -> Tuple[List[float], int, int]:
    """Apply per-SMILES function and collect (parsed values, n_parsed, n_failed)."""
    parsed: List[float] = []
    n_parsed = 0
    n_failed = 0
    for smi in smiles_list:
        v = fn(smi)
        if v is None or not _is_finite(v):
            n_failed += 1
            continue
        parsed.append(float(v))
        n_parsed += 1
    return parsed, n_parsed, n_failed


def _is_finite(x: float) -> bool:
    """NaN / inf check."""
    try:
        return x == x and abs(x) != float("inf")
    except Exception:
        return False


def _emit_degraded_passthrough(
    metric_name: str,
    description: str,
    target_band: str,
    output_path: Path,
    n_total: int,
    import_err: Optional[Exception],
) -> None:
    """Emit a DEGRADED_PASSTHROUGH JSON when RDKit import fails.

    Per party-twenty-spirit §3.5 the schema stays consistent so downstream
    tooling can branch on `status`.
    """
    payload = {
        "metric": metric_name,
        "description": description,
        "unit": _unit_for(metric_name),
        "scope": "ADMET pass-through panel (Wf-303C T1 / Wf-312C R01 5-NEW extension)",
        "source": ("molmetal/molmetal_lam/sbdd_env/metrics_v2.py:" + metric_name),
        "value": None,
        "n_samples": int(n_total),
        "data_source": "metallo_drugs_500_train.csv",
        "timestamp": _timestamp(),
        "status": "DEGRADED_PASSTHROUGH",
        "target_band": target_band,
        "honest_framing_note": (
            f"{metric_name} emission aborted: import failed ({import_err!r}). "
            "Emit a null record so the panel schema stays consistent and "
            "downstream tooling can branch on status."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def _unit_for(metric_name: str) -> str:
    if metric_name == "ames_mutagen":
        return "ratio (0-1, lower = less mutagenic)"
    if metric_name == "cell_permeability_logPapp":
        return "log Papp (cm/s); -6 = Papp 1e-6; higher = better permeability"
    if metric_name == "aqueous_solubility_logS":
        return "log S (mol/L); -4 = 0.1 mM; higher = more soluble"
    if metric_name == "caco2_perm":
        return (
            "log Papp Caco-2 (cm/s); -5.15 = well-permeable threshold; higher = better"
        )
    if metric_name == "bbb":
        return "logBB; >-1 = BBB-penetrating, <-1 = BBB-excluded"
    if metric_name == "clearance":
        return "mL/min/kg (hepatic intrinsic clearance CLint)"
    if metric_name == "half_life":
        return "hours (elimination half-life t1/2)"
    if metric_name == "vdss":
        return "L/kg (volume of distribution at steady state)"
    if metric_name == "sa_score":
        return "SA score (1-10); lower = easier to synthesise; <2.8 = synthesizable"
    if metric_name == "qed_score":
        return "QED (0-1); higher = more drug-like; >0.5 = druglikeness threshold"
    return ""


def _timestamp() -> str:
    import time as _t

    return _t.strftime("%Y-%m-%d")


def _emit_metric_json(
    metric_name: str,
    description: str,
    target_band: str,
    output_path: Path,
    n_total: int,
    n_parsed: int,
    n_failed: int,
    parsed: List[float],
    mean_fn: Callable[[List[str]], Optional[float]],
    smiles_list: List[str],
) -> None:
    """Emit the standard FLAT-schema metric JSON."""
    mean_val = (sum(parsed) / len(parsed)) if parsed else None
    std_val: Optional[float] = None
    if len(parsed) > 1:
        std_val = statistics.pstdev(parsed)

    # Cross-check against the _mean helper from metrics_v2 (sanity guard).
    cross_check: Optional[float] = None
    if mean_fn is not None and parsed:
        try:
            cross_check = float(mean_fn(smiles_list))
        except Exception:
            cross_check = None

    # Pass/fail vs target band (cohort mean).
    # Wf-312C — verdict logic for 5 NEW ADMET extension metrics:
    #   * caco2_perm    — PASS if > -5.15 (well-permeable threshold)
    #   * bbb           — INFO only (no binary PASS/FAIL — logBB is a spectrum)
    #   * clearance     — INFO only (typical range, no binary band)
    #   * half_life     — INFO only (typical range, no binary band)
    #   * vdss          — INFO only (typical range, no binary band)
    verdict = "PENDING"
    if mean_val is not None:
        if metric_name == "ames_mutagen":
            verdict = "PASS" if mean_val < 0.30 else "FAIL"
        elif metric_name == "cell_permeability_logPapp":
            verdict = "PASS" if mean_val > -6.0 else "FAIL"
        elif metric_name == "aqueous_solubility_logS":
            verdict = "PASS" if mean_val > -4.0 else "FAIL"
        elif metric_name == "caco2_perm":
            verdict = "PASS" if mean_val > -5.15 else "FAIL"
        elif metric_name == "sa_score":
            # Wf-315A — 3-band verdict per Wf-139 P6 specs:
            #   SA < 2.8 → PASS (synthesizable)
            #   2.8 <= SA < 4.0 → WARN (synthesizable but not trivial)
            #   SA >= 4.0 → FAIL (not synthesizable)
            if mean_val < 2.8:
                verdict = "PASS"
            elif mean_val < 4.0:
                verdict = "WARN"
            else:
                verdict = "FAIL"
        elif metric_name == "qed_score":
            # Wf-315A — Wf-139 P6 druglikeness threshold:
            #   QED > 0.5 → PASS (drug-like)
            #   QED <= 0.5 → FAIL (not drug-like)
            verdict = "PASS" if mean_val > 0.5 else "FAIL"
        elif metric_name in ("bbb", "clearance", "half_life", "vdss"):
            # INFO-only — no binary PASS/FAIL per source literature; report
            # the band-fit so the panel can be scanned at-a-glance.
            verdict = "INFO"

    payload = {
        "metric": metric_name,
        "description": description,
        "unit": _unit_for(metric_name),
        "scope": "ADMET pass-through panel (Wf-303C T1 / Wf-312C R01 5-NEW extension)",
        "source": ("molmetal/molmetal_lam/sbdd_env/metrics_v2.py:" + metric_name),
        "value": (float(mean_val) if mean_val is not None else None),
        "std": (float(std_val) if std_val is not None else None),
        "n_samples": int(n_total),
        "n_parsed": int(n_parsed),
        "n_failed": int(n_failed),
        "data_source": "metallo_drugs_500_train.csv",
        "timestamp": _timestamp(),
        "status": "MEASURED",
        "target_band": target_band,
        "verdict": verdict,
        "wf_tag": (
            "Wf-303C T1 / R03 ADMET pass-through"
            if metric_name
            in ("ames_mutagen", "cell_permeability_logPapp", "aqueous_solubility_logS")
            else "Wf-312C R01 ADMET 5-NEW extension"
            if metric_name in ("caco2_perm", "bbb", "clearance", "half_life", "vdss")
            else "Wf-315A SA + QED evaluator wiring (WEAK_TO_STRONG §3.6)"
        ),
        "wf_design_ref": (
            "molmetal/reports/wf297d_remediation_plan/remediation_plan.md §R03"
            if metric_name
            in ("ames_mutagen", "cell_permeability_logPapp", "aqueous_solubility_logS")
            else "molmetal/reports/wf312c_admet_extension/design.md §R01"
            if metric_name in ("caco2_perm", "bbb", "clearance", "half_life", "vdss")
            else "TODO/03_plans/WEAK_TO_STRONG_PLAN.md §3.6"
        ),
        "cross_check_via_metrics_v2_mean_helper": cross_check,
        "honest_framing_note": (
            f"{metric_name} computed via metrics_v2 pass-through on the "
            f"449 metallo-drug SMILES cohort. {n_parsed}/{n_total} parsed "
            f"({n_failed} parse failures, gracefully skipped). "
            f"Mean = {mean_val}, std = {std_val}, verdict = {verdict}. "
            "Descriptive physico-chemical proxy — NOT a learned metallodrug "
            "score. For paper §3 cohort panel only."
        ),
    }

    # Wf-308B / Wf-312C — auto-attach HONEST_NEGATIVE=true tag whenever the
    # binary PASS/FAIL verdict resolves to FAIL, so downstream tooling
    # (paper_dashboard, pre_flight_check) can branch on the canonical field
    # rather than re-deriving the FAIL/THRESHOLD comparison at every read
    # site.  INFO metrics (caco2_perm / bbb / clearance / half_life /
    # vdss) carry no HONEST_NEGATIVE tag — they are continuous-spectrum
    # metrics, not binary band-pass/fail.
    if verdict == "FAIL":
        payload["HONEST_NEGATIVE"] = True
        payload["honest_negative_reason"] = (
            f"Mean cohort {metric_name} = {mean_val:.4f} (target band "
            f"{target_band}) — verdict FAIL; auto-attached HONEST_NEGATIVE "
            f"tag per Wf-308B / Wf-312C contract so downstream tooling can "
            f"branch on the canonical field."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-303C T1 / R03 ADMET pass-through — emit 3 ADMET metric JSONs "
            "from a metallo-drug SMILES cohort via RDKit pass-through."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_CSV,
        help="Input SMILES file (CSV with 'smiles' column, or plain-text one-per-line)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Output directory for the 3 metric JSON files",
    )
    args = parser.parse_args(argv)

    smiles_list = _load_smiles(args.input)
    n_total = len(smiles_list)
    if n_total == 0:
        print(
            f"[wf303c-t1] no SMILES loaded from {args.input}; "
            "emitting 3 DEGRADED_PASSTHROUGH records.",
            file=sys.stderr,
            flush=True,
        )

    for spec in METRIC_SPECS:
        metric_name, description, fn, mean_fn, target_band, filename = spec
        output_path = args.output_dir / filename
        if not _IMPORT_OK or fn is None:
            _emit_degraded_passthrough(
                metric_name,
                description,
                target_band,
                output_path,
                n_total,
                _IMPORT_ERR,
            )
            print(
                f"[wf303c-t1] DEGRADED_PASSTHROUGH: {output_path}",
                flush=True,
            )
            continue
        parsed, n_parsed, n_failed = _compute_metric_for_smiles(smiles_list, fn)
        _emit_metric_json(
            metric_name,
            description,
            target_band,
            output_path,
            n_total,
            n_parsed,
            n_failed,
            parsed,
            mean_fn,
            smiles_list,
        )
        mean_val = (sum(parsed) / len(parsed)) if parsed else None
        print(
            f"[wf303c-t1] wrote {output_path} n_parsed={n_parsed}/{n_total} "
            f"mean={mean_val}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
