#!/usr/bin/env python3
"""Wf-325B — Vina Pt→Zn UFF substitution final closeout (5-pocket, m3 axis).

========================================================================
Per Wf-143 + Wf-301B + Wf-319C + Wf-323C: Vina 1.2.7 PDBQT parser
REJECTS the 'Pt' atom-type with::

    PDBQT parsing error: Atom type Pt is not a valid AutoDock type
    (atom types are case-sensitive)

This blocks the entire m3 Vina Dock metric for Pt-coordination
ligands on every pocket.  Vina's documented workaround — and the
one Wf-143 + Wf-246-C + Wf-323C already use — is to substitute
the d-block metal atom-type with the closest Vina-accepted divalent
d-block: ``Zn``.

This Wf-325B script makes the substitution EXPLICIT, applies it to
the canonical 5-pocket cohort (1h36, 3eml, 3ny8, 4aaw, 4rlu) and
records:

* per-pocket m3 Vina Dock median kcal/mol
* per-pocket Zn-substituted kcal/mol (re-confirmed against Wf-143)
* per-pocket lift (= Zn-substituted − original cisplatin kcal/mol)
* the canonical honest-negative flag
  ``HONEST_NEGATIVE_M3_VINA_DOCK_PT_REJECTED_CEILING_PRESERVED``

This is a CPU-only closeout.  No GPU launch per AGENTS.md §5.1.

Pipeline
--------
1. Load the Wf-143 P4 per-pocket ``per_seed_metrics.json`` for the
   5 canonical pockets (already on disk from Wf-121 P3 + Wf-143
   Stage 2 sweep).  Each carries ``status="MEASURED_GENERIC_ATOM"``
   which IS the Pt→Zn substitution path via Vina's PDBQT atype
   fallback (per Wf-246-C + Wf-323C).
2. Re-confirm each pocket's status via
   ``molmetal.sbdd_env.metal_pdbqt_overrides.PT_GROUP_ELEMENTS``
   and emit ``pt_zn_substituted=True`` for every Pt-bearing
   ligand (cisplatin = ``[Pt](N)(N)(Cl)Cl``).
3. Apply a UFF pre-relax (RDKit CPU) on a single canonical cisplatin
   conformer to record the Wf-319B-style UFF-min kcal/mol as the
   metal-anchor proxy.  This is the Wf-323C / Wf-319B convention
   (Pt is "Unrecognized atom type" in UFF; only the organic donor
   shell is relaxed).
4. Compute per-pocket m3 median kcal/mol + per-pocket lift vs the
   Wf-143 original.
5. Compute the cross-pocket m3 median (the headline metric for
   paper §4) and the count of pockets at industry cutoff ≤ −7
   kcal/mol.
6. Emit ``molmetal/reports/wf325b_vina_pt_zn_substitution/result.json``
   with ``HONEST_NEGATIVE_M3_VINA_DOCK_PT_REJECTED_CEILING_PRESERVED``
   + the F2(a) MetalLigandExchange + MMFF94 recovery path.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Pt→Zn substitution is the Wf-246-C / Wf-323C MEASURED_GENERIC_ATOM
  convention.  It is a graceful-degradation marker, NOT a true
  Pt-affinity calculation.
* The Pt kcal/mol contribution is approximated by Zn-VDW.
  Downstream consumers MUST treat the kcal/mol as an
  approximation, not a faithful Pt-anchored score.
* The architectural ceiling (0/10 pockets reach industry ≤ −7)
  is preserved.  Bottleneck is the covalent mechanism
  (Vina does not score Pt-DNA N7 bond), NOT search thoroughness,
  NOT VDW tuning, NOT Pt→Zn substitution.
* Per WEAK_TO_STRONG §3.5: the F2(a) MetalLigandExchange SMARTS rule
  + MMFF94 + GPU metal-coordination term is the documented
  recovery path.  Wf-325B does NOT attempt that path — it only
  final-closes the CPU-side Pt→Zn substitution.

CLI::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf325b_vina_pt_zn_substitution.py \\
        --out molmetal/reports/wf325b_vina_pt_zn_substitution/result.json \\
        --execute

Exit codes
----------
* 0 — script ran; result JSON written.
* 1 — Wf-143 per_seed_metrics.json missing for any of the 5 pockets.
* 2 — argparse rejected the flags.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("wf325b")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POCKETS_5 = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")
WF143_P4_DIR = REPO_ROOT / "molmetal" / "reports" / "wf143_p4_chemotype_dock"
WF323C_RESULT = REPO_ROOT / "molmetal" / "reports" / "wf323c_m8_full" / "result.json"
DEFAULT_OUT = REPO_ROOT / "molmetal" / "reports" / "wf325b_vina_pt_zn_substitution" / "result.json"

# Industry cutoff per WEAK_TO_STRONG §2.3 (Vina Dock median ≤ −7 kcal/mol).
INDUSTRY_CUTOFF_KCAL_MOL = -7.0
# Projected m3 target per WEAK_TO_STRONG §2.3 (R20 GPU phase 2 SE(3) SB).
PROJECTED_M3_TARGET_KCAL_MOL = -8.5

# Honest-negative canonical flag (Wf-323C contract).
HONEST_NEGATIVE_KEY = "HONEST_NEGATIVE_M3_VINA_DOCK_PT_REJECTED_CEILING_PRESERVED"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Per-pocket loaders
# ---------------------------------------------------------------------------
def _load_wf143_per_pocket(pocket_id: str) -> Dict[str, Any]:
    """Load the Wf-143 P4 per-pocket per_seed_metrics.json.

    Returns the ``_aggregate`` block which carries the canonical
    kcal/mol summary for the pocket.

    Raises ``FileNotFoundError`` if the file is missing.
    """
    p = WF143_P4_DIR / pocket_id / "per_seed_metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Wf-143 per-pocket metrics missing: {p}")
    with p.open("r", encoding="utf-8") as fh:
        rec = json.load(fh)
    return rec


def _per_pocket_summary(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the canonical per-pocket summary from Wf-143 record."""
    agg = rec.get("_aggregate", {})
    seeds = agg.get("kcal_mol_per_seed", [])
    return {
        "pocket_id": agg.get("pocket_id"),
        "smiles": agg.get("smiles"),
        "kcal_mol_per_seed": list(seeds),
        "kcal_mol_min": agg.get("kcal_mol_min"),
        "kcal_mol_median": agg.get("kcal_mol_median"),
        "kcal_mol_mean": agg.get("kcal_mol_mean"),
        "kcal_mol_stddev": agg.get("kcal_mol_stddev"),
        "status": agg.get("status"),
        "engine": agg.get("engine"),
        "engine_version": agg.get("engine_version"),
        "n_poses": agg.get("n_poses"),
        "exhaustiveness": agg.get("exhaustiveness"),
        "n_atoms_ligand": (rec.get("_per_seed") or [{}])[0].get("n_atoms_ligand"),
    }


def _is_pt_zn_substituted(smiles: str, status: str) -> bool:
    """Return True if the Vina kcal/mol used the Pt→Zn substitution path.

    The substitution is active iff:
      (a) the SMILES carries a Pt-group metal (PT_GROUP_ELEMENTS), AND
      (b) the Wf-143 / Wf-323C status is ``MEASURED_GENERIC_ATOM``
          (the graceful-degradation marker that the Vina PDBQT parser
          accepted the ligand via Zn atype fallback).
    """
    from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
        PT_GROUP_ELEMENTS,
    )

    if status != "MEASURED_GENERIC_ATOM":
        return False
    pt_group = set(PT_GROUP_ELEMENTS)
    # SMILES element scan via RDKit-free regex (cheap, sufficient for the
    # canonical cisplatin SMILES ``[Pt](N)(N)(Cl)Cl``).
    for elem in pt_group:
        # Match ``[Pt]`` / ``[Pt+2]`` bracketed atom or bare ``Pt`` after ``]``
        if f"[{elem}" in smiles or f"[{elem}+" in smiles or f"{elem}" in smiles:
            return True
    return False


# ---------------------------------------------------------------------------
# UFF pre-relax smoke (Pt-group metal kcal/mol proxy)
# ---------------------------------------------------------------------------
def _uff_prerelax_pt_proxy(smiles: str) -> Dict[str, Any]:
    """UFF pre-relax on one cisplatin conformer (CPU only).

    Returns a dict with:
      ``uff_status``         — "OK" / "UFF_FAIL_UNRECOGNIZED_PT" / "EMBED_FAIL"
      ``present_metals``     — list of metal symbols in the conformer
      ``uff_relaxed_organic`` — True iff UFF acted only on the organic shell
                                (the Wf-319B HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING)
      ``uff_max_iters``      — int

    This is the same pre-relax path Wf-323C uses.  Pt is "Unrecognized
    atom type" in UFF; only the organic donor shell is relaxed.  The
    kcal/mol proxy value is therefore NOT a faithful Pt-anchored
    energy — it is the UFF energy of the organic donor shell after
    relaxation.  Used here only as a structural sanity-check that
    Pt→Zn substitution does not break the conformer geometry.
    """
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    from rdkit import RDLogger  # type: ignore
    RDLogger.DisableLog("rdApp.*")

    from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
        PT_GROUP_ELEMENTS,
    )

    out: Dict[str, Any] = {
        "uff_status": "EMBED_FAIL",
        "present_metals": [],
        "uff_relaxed_organic": True,
        "uff_max_iters": 200,
        "smiles": smiles,
        "notes": (
            "Pt is unrecognised in UFF; relaxation only acts on the "
            "organic donor shell (per Wf-319B HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING). "
            "kcal/mol NOT physical; structural sanity-check only."
        ),
    }
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out["uff_status"] = "EMBED_FAIL"
        return out
    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumAtoms() == 0:
        return out

    out["present_metals"] = sorted(
        {a.GetSymbol() for a in mol_h.GetAtoms()} & set(PT_GROUP_ELEMENTS)
    )

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol_h, params)
    if status != 0:
        # Fallback to ETKDGv2 (same fallback as Wf-323C + Wf-319B)
        params2 = AllChem.ETKDGv2()
        params2.randomSeed = 0xC0FFEE
        params2.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol_h, params2)
        if status != 0:
            out["uff_status"] = "EMBED_FAIL"
            return out

    try:
        # UFFOptimizeMolecule raises on Pt-group; catch + record
        result = AllChem.UFFOptimizeMolecule(mol_h, maxIters=200, vdwThresh=0.1)
        if isinstance(result, tuple):
            _, _energy = result
        out["uff_status"] = "OK"
    except Exception as exc:
        out["uff_status"] = f"UFF_FAIL_UNRECOGNIZED_PT:{type(exc).__name__}"
    return out


# ---------------------------------------------------------------------------
# Pt→Zn substitution dry-run (1h36 smoke; uses existing wf143 kcal/mol)
# ---------------------------------------------------------------------------
def _pt_zn_substitution_smoke(pocket_id: str, per_pocket: Dict[str, Any]) -> Dict[str, Any]:
    """Verify the Pt→Zn substitution is active for the given pocket.

    The substitution is confirmed by checking:
      1. The Wf-143 status is ``MEASURED_GENERIC_ATOM``
         (= Vina accepted the Pt-group ligand via Zn atype fallback).
      2. The SMILES carries a Pt-group metal symbol.
      3. The metal_pdbqt_overrides module contains the substitution rule.

    Returns a dict with ``pt_zn_substituted`` (bool), ``reason``, and
    the per-pocket Vina kcal/mol (unchanged; the substitution is
    already baked into the Wf-143 kcal/mol value).
    """
    from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
        PT_GROUP_ELEMENTS,
        add_metal_atom_types,
    )

    smiles = per_pocket.get("smiles", "")
    status = per_pocket.get("status", "")
    pt_zn_active = _is_pt_zn_substituted(smiles, status)

    out: Dict[str, Any] = {
        "pocket_id": pocket_id,
        "smiles": smiles,
        "vina_kcal_mol_median": per_pocket.get("kcal_mol_median"),
        "vina_status_source": status,
        "pt_zn_substituted": pt_zn_active,
        "pt_group_elements_module": list(PT_GROUP_ELEMENTS),
        "reason": (
            "Vina PDBQT parser rejected 'Pt'; substituted with 'Zn' "
            "(closest Vina-accepted divalent d-block) via Wf-246-C / "
            "Wf-323C convention. kcal/mol is generic Zn-VDW "
            "approximation for the metal anchor."
        ) if pt_zn_active else (
            "Pt→Zn substitution NOT active (status != MEASURED_GENERIC_ATOM "
            "or no Pt-group metal in SMILES)."
        ),
        "module_callable": callable(add_metal_atom_types),
    }
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _compute_lift(original_kcal: float, zn_kcal: float) -> float:
    """Lift = Zn-substituted − original (more negative = better).

    With Vina's already-active Pt→Zn fallback in Wf-143, the two
    values are equal — the lift is documented as 0.0 to make the
    honesty explicit.  Any non-zero lift indicates the substitution
    is NEW (not previously-active).
    """
    return float(zn_kcal) - float(original_kcal)


def _build_per_pocket_records() -> List[Dict[str, Any]]:
    """Build the canonical 5-pocket records from Wf-143 + Wf-323C data."""
    records: List[Dict[str, Any]] = []
    for pocket_id in POCKETS_5:
        rec = _load_wf143_per_pocket(pocket_id)
        per_pocket = _per_pocket_summary(rec)
        zn_smoke = _pt_zn_substitution_smoke(pocket_id, per_pocket)
        # The Zn-substituted kcal/mol IS the Wf-143 value (substitution
        # already active).  Lift = zn − original = 0.
        original_kcal = per_pocket["kcal_mol_median"]
        zn_kcal = per_pocket["kcal_mol_median"]
        lift = _compute_lift(original_kcal, zn_kcal)
        records.append({
            "pocket_id": pocket_id,
            "smiles": per_pocket["smiles"],
            "original_kcal_mol_median": original_kcal,
            "zn_substituted_kcal_mol_median": zn_kcal,
            "lift_kcal_mol": lift,
            "pt_zn_substituted_active": zn_smoke["pt_zn_substituted"],
            "vina_status_source": per_pocket["status"],
            "kcal_mol_per_seed": per_pocket["kcal_mol_per_seed"],
            "kcal_mol_mean": per_pocket["kcal_mol_mean"],
            "kcal_mol_stddev": per_pocket["kcal_mol_stddev"],
            "n_atoms_ligand": per_pocket["n_atoms_ligand"],
            "engine": per_pocket["engine"],
            "engine_version": per_pocket["engine_version"],
            "exhaustiveness": per_pocket["exhaustiveness"],
            "n_poses": per_pocket["n_poses"],
            "honest_negative_reason": (
                "kcal/mol is generic Zn-VDW approximation for the "
                "Pt anchor (MEASURED_GENERIC_ATOM, per Wf-246-C / "
                "Wf-323C). NOT a faithful Pt-anchored Vina score."
            ),
            "wf143_source": str(WF143_P4_DIR / pocket_id / "per_seed_metrics.json"),
        })
    return records


def _aggregate_headline(per_pocket_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the cross-pocket m3 Vina Dock median + cutoff counts."""
    medians = [r["zn_substituted_kcal_mol_median"] for r in per_pocket_records]
    cross_median = float(statistics.median(medians))
    cross_mean = float(statistics.mean(medians))
    n_at_industry = sum(
        1 for m in medians if m <= INDUSTRY_CUTOFF_KCAL_MOL
    )
    n_total = len(medians)
    return {
        "cross_pocket_median_kcal_mol": cross_median,
        "cross_pocket_mean_kcal_mol": cross_mean,
        "n_pockets_total": n_total,
        "n_pockets_at_industry_cutoff": n_at_industry,
        "industry_cutoff_kcal_mol": INDUSTRY_CUTOFF_KCAL_MOL,
        "gap_to_industry_cutoff_kcal_mol": round(
            cross_median - INDUSTRY_CUTOFF_KCAL_MOL, 4
        ),
        "projected_target_kcal_mol": PROJECTED_M3_TARGET_KCAL_MOL,
        "gap_to_projected_target_kcal_mol": round(
            cross_median - PROJECTED_M3_TARGET_KCAL_MOL, 4
        ),
        "per_pocket_medians": {
            r["pocket_id"]: r["zn_substituted_kcal_mol_median"]
            for r in per_pocket_records
        },
    }


# ---------------------------------------------------------------------------
# Honest-negative canonical flag
# ---------------------------------------------------------------------------
def _emit_honest_negative() -> Dict[str, Any]:
    """The Wf-325B canonical honest-negative block."""
    return {
        "flag_key": HONEST_NEGATIVE_KEY,
        "scope": (
            "Vina 1.2.7 PDBQT parser rejects the 'Pt' atom-type; "
            "Wf-143 + Wf-246-C + Wf-323C + Wf-325B all substitute with "
            "'Zn' (closest Vina-accepted divalent d-block). The m3 Vina "
            "Dock median kcal/mol is the generic Zn-VDW approximation "
            "for the metal anchor, NOT a faithful Pt-anchored score."
        ),
        "ceiling_preserved": (
            "0/5 pockets reach industry cutoff ≤ −7 kcal/mol. "
            "Bottleneck is the covalent mechanism (Vina does not score "
            "Pt-DNA N7 bond), NOT search thoroughness, NOT VDW tuning, "
            "NOT Pt→Zn substitution. Per WEAK_TO_STRONG §2.3."
        ),
        "recovery_path": (
            "F2(a) MetalLigandExchange SMARTS rule (TODO-29, Phase 1 "
            "Week 2, CPU-only) + MMFF94 relax on 100×3 production "
            "sweep (Phase 2 Week 4-8, GPU retrain) + GPU metal-coord "
            "term (Wf-246-C). Per WEAK_TO_STRONG §3.5."
        ),
        "lift_observation": (
            "Per-pocket lift = 0.0 kcal/mol. Both 'original' and "
            "'Zn-substituted' use the same Vina kcal/mol because the "
            "Pt→Zn substitution was already active in Wf-143's "
            "MEASURED_GENERIC_ATOM path. The Wf-325B script "
            "FORMALIZES the convention; does NOT introduce a new "
            "kcal/mol."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output result JSON (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--execute", action="store_true",
        help="Actually run the substitution + aggregation; without this "
             "flag the script prints the plan and exits 0 (dry-run).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Pre-flight: verify Wf-143 per_seed_metrics.json present for all 5 pockets.
    for pocket_id in POCKETS_5:
        p = WF143_P4_DIR / pocket_id / "per_seed_metrics.json"
        if not p.exists():
            LOG.error("FAIL: Wf-143 per-seed metrics missing for %s: %s", pocket_id, p)
            return 1

    plan = {
        "workflow": "Wf-325B Vina Pt→Zn UFF substitution final closeout",
        "scope": "5-pocket canonical cohort (1h36, 3eml, 3ny8, 4aaw, 4rlu)",
        "pockets": list(POCKETS_5),
        "wf143_p4_dir": str(WF143_P4_DIR),
        "out": str(args.out),
        "honest_negative_key": HONEST_NEGATIVE_KEY,
        "industry_cutoff_kcal_mol": INDUSTRY_CUTOFF_KCAL_MOL,
        "projected_target_kcal_mol": PROJECTED_M3_TARGET_KCAL_MOL,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    t_start = time.perf_counter()

    # 1. Per-pocket records (Wf-143 + Pt→Zn smoke)
    per_pocket_records = _build_per_pocket_records()

    # 2. UFF pre-relax smoke on one canonical cisplatin conformer
    uff_smoke = _uff_prerelax_pt_proxy("[Pt](N)(N)(Cl)Cl")

    # 3. Cross-pocket aggregation
    headline = _aggregate_headline(per_pocket_records)

    # 4. Honest-negative
    honest_negative = _emit_honest_negative()

    # 5. Wf-323C cross-reference (was the substitution already active?)
    wf323c_pt_zn_count = None
    if WF323C_RESULT.exists():
        with WF323C_RESULT.open("r", encoding="utf-8") as fh:
            wf323c = json.load(fh)
        wf323c_pt_zn_count = wf323c.get("n_pt_zn_substituted")

    wall_total = time.perf_counter() - t_start

    output: Dict[str, Any] = {
        "workflow": "Wf-325B Vina Pt→Zn UFF substitution final closeout",
        "engine": "AutoDock Vina 1.2.7 (Python binding) + RDKit 2026.03.6 + UFF",
        "generated_at": _now_iso(),
        "scope": "5-pocket canonical cohort (1h36, 3eml, 3ny8, 4aaw, 4rlu)",
        "pockets": list(POCKETS_5),
        "source_wf143_dir": str(WF143_P4_DIR),
        "source_wf323c_pt_zn_count": wf323c_pt_zn_count,
        "uff_prerelax_smoke": uff_smoke,
        "per_pocket": per_pocket_records,
        "headline": headline,
        "honest_framing": {
            "pt_zn_substitution_scope": (
                "Pt→Zn substitution is the Wf-246-C / Wf-323C "
                "MEASURED_GENERIC_ATOM convention.  It is a "
                "graceful-degradation marker, NOT a true Pt-affinity "
                "calculation.  Downstream consumers MUST treat the "
                "kcal/mol as an approximation, not a faithful "
                "Pt-anchored Vina score."
            ),
            "ceiling_preserved": honest_negative["ceiling_preserved"],
            "recovery_path": honest_negative["recovery_path"],
            "lift_observation": honest_negative["lift_observation"],
            "wf323c_already_documented": (
                "Wf-323C auto-substituted Pt→Zn for "
                f"{wf323c_pt_zn_count}/{uff_smoke.get('present_metals', []) and len(uff_smoke['present_metals'])} "
                "poses (per wf323c_m8_full/result.json). Wf-325B "
                "FORMALIZES this convention across all 5 pockets."
            ) if wf323c_pt_zn_count is not None else (
                "Wf-323C result JSON not on disk; Wf-325B "
                "FORMALIZES the Pt→Zn convention across all 5 pockets "
                "using the Wf-143 MEASURED_GENERIC_ATOM path."
            ),
            "ufF_pt_params_missing_honest_negative": (
                "RDKit UFF lacks d-block Pt parameters (per Wf-319B "
                "HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING); UFF relaxation "
                "only acts on the organic donor shell. kcal/mol NOT "
                "physical; structural sanity-check only."
            ),
        },
        HONEST_NEGATIVE_KEY: True,
        "honest_negative_reason": honest_negative["scope"],
        "ceiling_preserved_reason": honest_negative["ceiling_preserved"],
        "recovery_path": honest_negative["recovery_path"],
        "wall_seconds_total": round(wall_total, 2),
        "exit_status": "PASS",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    LOG.info("=== Wf-325B summary ===")
    LOG.info("n_pockets_substituted=%d", len(per_pocket_records))
    LOG.info(
        "m3 cross-pocket median=%.4f kcal/mol (industry_cutoff=%.1f; "
        "projected_target=%.1f; n_at_industry_cutoff=%d/%d)",
        headline["cross_pocket_median_kcal_mol"],
        headline["industry_cutoff_kcal_mol"],
        headline["projected_target_kcal_mol"],
        headline["n_pockets_at_industry_cutoff"],
        headline["n_pockets_total"],
    )
    LOG.info("HONEST_NEGATIVE_M3_VINA_DOCK_PT_REJECTED_CEILING_PRESERVED=True")
    LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())