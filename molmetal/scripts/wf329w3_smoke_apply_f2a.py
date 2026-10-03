#!/usr/bin/env python3
"""Wf-329-W3 — smoke helper: apply F2(a) covalent scoring modifier on top of
Wf-143 / Wf-325B historical per-pocket cisplatin kcal/mol values.

This script demonstrates the conceptual F2(a) lift on the Wf-143
MEASURED_GENERIC_ATOM cohort (1h36, 3eml, 3ny8, 4aaw, 4rlu,
cisplatin).  It does NOT run new Vina docking — Vina's UFF cannot
parameterise Pt+2 (per Wf-325B + Wf-301B), so a fresh Vina run
on the original cisplatin SMILES returns STUB_FALLBACK.  Instead
the script reads the Wf-143 per-pocket cisplatin kcal/mol values
(MEASURED_GENERIC_ATOM, with Pt→Zn substitution) and applies
the F2(a) covalent scoring modifier on top.

Outputs
-------
* ``molmetal/reports/wf329w3_f2a_vina/smoke_per_pocket.json``
  JSON with per-pocket vina_kcal_mol, f2a_kcal_mol, modifier,
  recovery status, and cohort median lift.

Honest framing
--------------
* MEASURED: Wf-143 / Wf-325B per-pocket cisplatin kcal/mol values
  (Pt→Zn substitution via Vina PDBQT atype fallback).
* PROJECTED: per-bond modifier = -1.5 kcal/mol (per Wf-329-W3 task
  brief; conservative midpoint per Lippard 1995 + Jamieson 2016).
* HONEST_NEG: cohort median after F2(a) ≈ -4.78 kcal/mol (still
  above -7 industry cutoff).  Per Wf-325B + WEAK_TO_STRONG §3.5,
  F2(a) alone is INSUFFICIENT — Phase 2 GPU metal-coordination
  term (Wf-246-C) is the next recovery step.

Usage
-----
::

    .venv/bin/python molmetal/scripts/wf329w3_smoke_apply_f2a.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from molmetal.scripts.wf329w3_f2a_metal_ligand import (  # noqa: E402
    F2AMetalLigandScorer,
    DEFAULT_COVALENT_MODIFIER_KCAL_MOL,
    INDUSTRY_CUTOFF_KCAL_MOL,
)

# ---------------------------------------------------------------------------
# Wf-143 / Wf-325B per-pocket cisplatin MEASURED_GENERIC_ATOM values.
# ---------------------------------------------------------------------------
# Per Wf-325B "headline numbers (MEASURED)":
# | Pocket | kcal/mol per seed (Wf-143) | Zn-substituted median |
# | 1h36   | -3.597, -3.575, -3.589     | -3.589                |
# | 3eml   | -2.826, -2.827, -2.827     | -2.827                |
# | 3ny8   | -3.721, -3.716, -3.734     | -3.724                |
# | 4aaw   | -3.287, -3.275, -3.283     | -3.282                |
# | 4rlu   | -2.995, -2.981, -2.974     | -2.983                |
# | Cross-pocket median                                    | -3.2830 |
# ---------------------------------------------------------------------------
WF143_CISPLATIN_KCAL_MOL: Dict[str, float] = {
    "1h36": -3.589,
    "3eml": -2.827,
    "3ny8": -3.724,
    "4aaw": -3.282,
    "4rlu": -2.983,
}

CISPLATIN_SMILES = "Cl[Pt](Cl)(N)(N)"


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    m = n // 2
    return (s[m - 1] + s[m]) / 2.0 if n % 2 == 0 else s[m]


def main() -> int:
    scorer = F2AMetalLigandScorer()
    # First verify F2(a) detects the Pt centre in cisplatin.
    dispatch = scorer.dispatch(CISPLATIN_SMILES)
    print(
        f"[wf329w3-f2a-smoke] scorer: {scorer.n_patterns_loaded} patterns, "
        f"modifier={scorer.modifier_kcal_mol:+.2f} kcal/mol/bond"
    )
    print(
        f"[wf329w3-f2a-smoke] cisplatin dispatch: n_bonds="
        f"{dispatch.n_covalent_bonds} matched={dispatch.matched_patterns}"
    )

    per_pocket: Dict[str, Dict[str, float]] = {}
    vina_scores: List[float] = []
    f2a_scores: List[float] = []
    for pocket_id, vina_kcal in WF143_CISPLATIN_KCAL_MOL.items():
        f2a_kcal, mod_total, n_bonds = scorer.apply_covalent_modifier(
            CISPLATIN_SMILES, vina_kcal
        )
        status, gap = scorer.classify_recovery(f2a_kcal)
        per_pocket[pocket_id] = {
            "vina_kcal_mol": vina_kcal,
            "f2a_kcal_mol": f2a_kcal,
            "modifier_total_kcal_mol": mod_total,
            "n_bonds": n_bonds,
            "f2a_recovery_status": status,
            "f2a_gap_to_industry_kcal_mol": gap,
        }
        vina_scores.append(vina_kcal)
        f2a_scores.append(f2a_kcal)
        print(
            f"  {pocket_id}: vina={vina_kcal:.3f} f2a={f2a_kcal:.3f} "
            f"mod={mod_total:+.2f} n={n_bonds} status={status} "
            f"gap={gap:+.3f}"
        )

    cohort_vina_median = _median(vina_scores)
    cohort_f2a_median = _median(f2a_scores)
    cohort_median_lift = cohort_f2a_median - cohort_vina_median
    cohort_f2a_gap_to_industry = (
        cohort_f2a_median - INDUSTRY_CUTOFF_KCAL_MOL
    )
    overall_status, overall_gap = scorer.classify_recovery(cohort_f2a_median)

    headline = {
        "cohort_vina_median_kcal_mol": round(cohort_vina_median, 4),
        "cohort_f2a_median_kcal_mol": round(cohort_f2a_median, 4),
        "cohort_median_lift_kcal_mol": round(cohort_median_lift, 4),
        "industry_cutoff_kcal_mol": INDUSTRY_CUTOFF_KCAL_MOL,
        "cohort_f2a_gap_to_industry_kcal_mol": round(
            cohort_f2a_gap_to_industry, 4
        ),
        "cohort_overall_recovery_status": overall_status,
        "PROJECTED_cohort_status": (
            "HONEST_NEGATIVE_COVALENT_INSUFFICIENT"
            if overall_gap > 1.7
            else "RECOVERY_BORDERLINE"
        ),
        "baseline_wf325b_median_kcal_mol": -3.283,
        "baseline_wf325b_gap_to_industry_kcal_mol": 3.717,
        "wf329w3_lift_closes_gap_by": round(
            -cohort_median_lift, 4
        ),  # positive = closes gap
        "remaining_gap_after_wf329w3": round(cohort_f2a_gap_to_industry, 4),
    }

    out = {
        "metadata": {
            "workflow": "Wf-329-W3",
            "engine": "wf329w3_smoke_apply_f2a.py",
            "generated_utc": _utc_now_iso(),
            "scope": "Wf-143 / Wf-325B 5-pocket cisplatin cohort + "
                     "F2(a) covalent scoring modifier",
            "modifier_kcal_mol_per_bond": (
                DEFAULT_COVALENT_MODIFIER_KCAL_MOL
            ),
            "n_patterns_loaded": scorer.n_patterns_loaded,
            "smiles": CISPLATIN_SMILES,
        },
        "per_pocket": per_pocket,
        "headline": headline,
        "honest_framing": {
            "MEASURED": (
                "Wf-143 / Wf-325B per-pocket cisplatin kcal/mol "
                "(MEASURED_GENERIC_ATOM, Pt→Zn substitution)"
            ),
            "PROJECTED_modifier": (
                f"per-bond modifier = {DEFAULT_COVALENT_MODIFIER_KCAL_MOL} "
                f"kcal/mol (Lippard 1995 + Jamieson 2016 + industry "
                f"conservative midpoint)"
            ),
            "PROJECTED_lift_per_Pt_center": (
                f"{DEFAULT_COVALENT_MODIFIER_KCAL_MOL} kcal/mol "
                f"(1 Pt center per cisplatin)"
            ),
            "HONEST_NEG": (
                f"Cohort median after F2(a) = {cohort_f2a_median:+.3f} "
                f"kcal/mol — still {cohort_f2a_gap_to_industry:+.3f} "
                f"kcal/mol above industry cutoff {INDUSTRY_CUTOFF_KCAL_MOL}. "
                f"F2(a) alone is INSUFFICIENT to close the m3 Vina "
                f"Dock gap. Next step per WEAK_TO_STRONG §3.5: "
                f"Phase 2 GPU metal-coordination term (Wf-246-C)."
            ),
            "recovery_path": (
                "F2(a) MetalLigandExchange SMARTS rule (this Wf-329-W3) "
                "+ Phase 2 GPU metal-coordination term (Wf-246-C) per "
                "WEAK_TO_STRONG §3.5"
            ),
        },
        "honest_negative_key": (
            "HONEST_NEGATIVE_M3_VINA_COVALENT_INSUFFICIENT"
        ),
        "exit_status": "PASS",
    }

    out_path = (
        ROOT / "molmetal" / "reports" / "wf329w3_f2a_vina"
        / "smoke_per_pocket.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    print()
    print(f"[wf329w3-f2a-smoke] headline:")
    print(f"  cohort_vina_median = {headline['cohort_vina_median_kcal_mol']:+.4f} kcal/mol")
    print(f"  cohort_f2a_median  = {headline['cohort_f2a_median_kcal_mol']:+.4f} kcal/mol")
    print(f"  cohort_median_lift = {headline['cohort_median_lift_kcal_mol']:+.4f} kcal/mol (PROJECTED: -1.5)")
    print(f"  industry_cutoff    = {headline['industry_cutoff_kcal_mol']:+.1f} kcal/mol")
    print(f"  cohort_f2a_gap     = {headline['cohort_f2a_gap_to_industry_kcal_mol']:+.4f} kcal/mol")
    print(f"  overall_status     = {overall_status}")
    print(f"  PROJECTED_status   = {headline['PROJECTED_cohort_status']}")
    print(f"  Wf-329-W3 closes the gap by {-cohort_median_lift:.4f} kcal/mol "
          f"(Wf-325B gap was 3.717 kcal/mol)")
    print()
    print(f"  Output: {out_path}")
    return 0


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
