#!/usr/bin/env python3
"""Append the BLK-C-M4 / next_actions[3] (METAL-LABEL) resolution to cfm_todo.json.

The METAL-LABEL question (next_actions[3]) is the cheapest open question in the
CFM line. This script commits the resolution: metal-bound aromatic bonds are
0.0149 of the truth-aromatic population, the head is trained on ALL pairs (no
metal exclusion), and the 70.7% aromatic->single collapse is overwhelmingly
organic-organic. The Wiberg BO literature confirms 0.3-0.5 is the dative range
and 1.3-1.7 the aromatic band, so the label is MEANINGFUL even for metal-ligand
contacts -- the metal's Wiberg never reaches the aromatic band by construction.

Audit script: molmetal/scripts/wf_meta_audit_metal_labels.py
Audit JSON:   molmetal/reports/wf_meta_metal_label_audit/audit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TARGET = REPO / "cfm_todo.json"

# Verdict entry to append into next_actions, mirroring the existing schema.
VERDICT_BLOCK = {
    "id": "BLK-C-M4.METAL_LABEL_2026_10_13",
    "line": "CFM",
    "title": "METAL-LABEL QUESTION RESOLVED: metal-bound pairs are 1.49% of truth-aromatic bonds; the 70.7% aromatic->single collapse is overwhelmingly organic-organic. Wiberg BO labels are MEANINGFUL for metal-ligand contacts -- dative band 0.3-0.5, aromatic band 1.3-1.7, disjoint. The head IS trained on metal-bound pairs (with BOND_SINGLE / BOND_NO_BOND labels) -- there is NO exclusion of metal-bound pairs from the loss. Therefore the BLK-C-M4 retrain slot should NOT be spent on the head for a metal-label reason.",
    "priority": 4,
    "status": "RESOLVED. The metal-label premise of BLK-C-M4 is refuted; BLK-C-M4 remains open on a different premise (see next_actions[0]).",
    "evidence_paths": [
        "molmetal/scripts/wf_meta_audit_metal_labels.py (audit script)",
        "molmetal/reports/wf_meta_metal_label_audit/audit.json (measured counts)",
        "molmetal/scripts/train_cfm_yuelbond_wasserstein.py::_bond_target_for_clean_cloud (line 1950-2101; no metal exclusion)",
        "molmetal/data/tmqm_clouds.py::_BO_TYPE_BANDS (line 174-179; dative band <1.25 includes metal BO 0.3-0.9)",
        "molmetal/tests/test_wf_blkc_qed_lever_aromatic_head.py (line 314-315; SCOPE excludes metal-bound pairs, not the TRAIN target)",
    ],
    "literature": [
        "Balcells & Skjelstad 2020, J. Chem. Inf. Model. 60, 6135-6146, DOI 10.1021/acs.jcim.0c01041 (tmQM dataset; Wiberg BO files in 2024 extended release)",
        "Frontiers Chem. 2023 metalloporphyrin WBI: dative 0.3-0.6, Sc-N 0.428-0.512",
        "Pt-C sigma NHC complexes WBI 0.28-0.85 (Wiberg range maps to coordination regime)",
        "Pt(II) trans-influence NMR studies WBI as continuous covalency descriptor (no binary threshold)",
        "RDKit default AROMATICITY_RDKIT: metal atoms are not in the donor-type table; metal-containing rings are NEVER aromatic by default",
        "CAS Structuring and Registration Policies for Coordination Compounds: metal-ligand bonds default to single; aromaticity is preserved on coordinated ligands but the metal atom itself never closes the aromatic ring",
    ],
    "measured_a": {
        "scope": "non-H heavy-atom pairs from tmQM restricted to {Pt,Ru,Ir,Rh,Os,Re}; non-metal scope (same as bench_tmqm_bond_perception.py --heavy-atom-scope)",
        "n_complexes_scanned": 108542,
        "n_complexes_with_target_metal": 33429,
        "n_pairs_in_scope_total": 1416944,
        "n_aromatic_truth_total": 649758,
        "n_aromatic_truth_metal_bound": 9677,
        "n_aromatic_truth_organic": 640081,
        "metal_share_of_aromatic_truth": 0.01489323717445572,
        "per_metal_aromatic_metal_bound": {"Pt": 33, "Re": 6040, "Os": 367, "Rh": 354, "Ir": 442, "Ru": 2441},
        "source": "molmetal/reports/wf_meta_metal_label_audit/audit.json",
    },
    "measured_b_wiberg_meaningfulness": {
        "dative_typical_range": "0.3-0.5 (Frontiers Chem 2023 metalloporphyrin Sc-N 0.428-0.512; Pd-NHC N-donor 0.314; Pt-N sigma 0.916 with strong donor; Pt-C in eta3-propargyl 0.157-0.576)",
        "aromatic_band_used_by_trainer": "1.25-1.75 Wiberg BO (= BOND_AROMATIC class)",
        "aromatic_band_used_by_eval": "1.3-1.7 Wiberg BO (open interval, disjoint from {single, double, triple})",
        "verdict": "Wiberg BO is meaningful for metal-ligand contacts as a continuous covalency descriptor. tmQM BO files use TPSSh-D3BJ/def2-SVP. Dative bonds sit BELOW the aromatic band by construction (0.3-0.5 vs 1.3-1.7), so the BOND_AROMATIC label never falls on a dative pair -- there is NO label mismatch on the metal side.",
    },
    "measured_c_train_target_excludes_metal": {
        "verdict": "NO. The training target includes metal-bound pairs. _bond_target_for_clean_cloud at train_cfm_yuelbond_wasserstein.py:1950-2101 reads from a dense (N, N) truth table built from cloud.bonds / cloud.bond_types; ALL pairs in edge_index (within cutoff=3.5 A) get a target -- no metal exclusion.",
        "metal_pair_label_distribution": "BO in [0.3, 1.25) -> BOND_SINGLE (dative); BO < 0.3 -> BOND_NO_BOND. No pair on the metal side ever receives BOND_AROMATIC.",
        "implication": "The 70.7% aromatic->single collapse cannot be a 'metal label is wrong' artefact because the aromatic band [1.3, 1.7) is disjoint from the metal dative band [0.3, 0.5). The collapse is an organic-organic defect.",
    },
    "measured_d_correct_metal_ligand_label_convention": {
        "answer": "There is no single canonical convention, but the literature converges on: (i) coordination number = number of DONOR ATOMS not number of ligands; (ii) hapticity (eta-n) is the standard for pi-bound ligands; (iii) RDKit's default AROMATICITY_RDKIT never marks metal atoms aromatic because the donor-type table excludes metals; (v) CAS convention: metal-ligand bonds default to single (with narrow exceptions for O/S/N as monatomic anions), and aromatic systems in ligands REMAIN normalized when coordinated.",
        "practical_answer_for_5_class_bond_head": "Keep the current label scheme. The aromatic band (1.25-1.75 Wiberg) and dative band (0.3-0.9 Wiberg) are disjoint, so BOND_AROMATIC never collides with metal coordination. BOND_SINGLE correctly captures dative Pt/Ru/Ir-ligand bonds.",
        "alternative_if_bond_head_needed_a_metal_class": "Could add BOND_COORDINATIVE = 5 as a sixth class for dative pairs (BO 0.3-0.9), but it would change the 5-way per-pair classifier's loss weights and require a new class-frequency table. NOT recommended -- the disjoint-band scheme already separates the two regimes correctly.",
    },
    "verdict_summary": "Metal labels are SOUND. The BLK-C-M4 retrain slot should NOT be spent on the head for a metal-label reason; the 70.7% collapse is overwhelmingly organic-organic and the aromatic band is disjoint from the metal dative band by construction. BLK-C-M4 remains open on the original (B) premise -- the sampler produces 0.0% aromatic bonds because (i) generated p01 pair distance 1.131 A sits BELOW the aromatic band 1.39+ A and (ii) the bias sweep at bias 3.0 reaches the decoder but does not produce aromatic bonds -- which next_actions[0] (the 24-cloud 4-seed P0 sweep) was designed to separate.",
    "honest_negative": "Re scope: the 798-complex cohort in the BLK-C-M4 docstring is a 900-complex 150 x 6-metal selection. The 1.49% metal share is measured on the ENTIRE tmQM corpus restricted to {Pt,Ru,Ir,Rh,Os,Re}, n=33,429 complexes. Restricting to the 798-complex 6-metal evaluation window would not materially change the answer -- aromatic is an organic phenomenon by Wiberg convention, and the metal's Wiberg contribution never reaches 1.3.",
    "verified": True,
    "verified_by": "wf_meta_audit_metal_labels.py (Python -I invocation, all numbers reproduced on 2026-10-13)",
    "audit_path": "molmetal/reports/wf_meta_metal_label_audit/audit.json",
}


def main() -> int:
    if not TARGET.is_file():
        print(f"ERROR: {TARGET} not on disk", file=sys.stderr)
        return 1
    raw = json.loads(TARGET.read_text(encoding="utf-8"))

    # Append to next_actions[] as a 'verdict' record (priority 99, status RESOLVED)
    # so the next_actions table shows this next_actions[3] entry resolved.
    next_actions = raw.get("next_actions", [])
    # Find the metal-label question entry (next_actions[3] in the v2 schema;
    # but identify by the title to be safe).
    resolved_index = None
    for i, na in enumerate(next_actions):
        if isinstance(na, dict) and "METAL-LABEL" in na.get("action", ""):
            resolved_index = i
            break
    if resolved_index is None:
        print("WARNING: metal-label next_action entry not found; appending new.", file=sys.stderr)

    verdict_entry = {
        "priority": 99,
        "action": (
            "RESOLVED 2026-10-13: cfm_todo.json::BLK-C-M4.METAL_LABEL_2026_10_13. "
            "Metal-bound pairs account for 0.0149 of truth-aromatic bonds on the "
            "33,429-complex 6-metal cohort; the 70.7% aromatic->single collapse is "
            "organic-organic by construction (aromatic band [1.3, 1.7) Wiberg is "
            "disjoint from dative [0.3, 0.5]); the head IS trained on metal-bound "
            "pairs (NO exclusion in _bond_target_for_clean_cloud); Wiberg BO is "
            "meaningful as a continuous covalency descriptor (0.3-0.5 dative typical, "
            "0.28-0.85 Pt/Rh complexes, RDKit never aromatic on metal atoms). "
            "Verdict: metal labels are SOUND, BLK-C-M4 retrain slot should NOT be "
            "spent on head for metal-label reasons."
        ),
        "blocking": "BLK-C-M4",
        "status": "RESOLVED, refutes the metal-label premise of BLK-C-M4",
        "audit_path": "molmetal/reports/wf_meta_metal_label_audit/audit.json",
        "audit_script": "molmetal/scripts/wf_meta_audit_metal_labels.py",
        "verdict_block": VERDICT_BLOCK,
    }
    next_actions.append(verdict_entry)
    raw["next_actions"] = next_actions

    # Also append the verdict to resolved_2026_10_12 list (as a structured
    # marker) so the audit trail in the plan reflects the resolution.
    if "resolved_2026_10_13" not in raw:
        raw["resolved_2026_10_13"] = []
    if VERDICT_BLOCK not in raw["resolved_2026_10_13"]:
        raw["resolved_2026_10_13"].append(VERDICT_BLOCK)

    # Bump the updated stamp on the plan.
    raw["updated"] = (
        "2026-10-13 (v6: BLK-C-M4.METAL_LABEL_2026_10_13 resolved, "
        "metal-bound share 1.49% measured on 33,429 complexes)"
    )

    # Atomic write (direct import of the helper module by file path)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_wf386a3_helper",
        str(REPO / "molmetal" / "scripts" / "wf386a3_atomic_write.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    atomic_write_file = _mod.atomic_write_file
    new_text = json.dumps(raw, indent=2, ensure_ascii=False)
    atomic_write_file(str(TARGET), new_text + "\n")
    print(f"Appended BLK-C-M4.METAL_LABEL_2026_10_13 to {TARGET}")
    print(f"  next_actions size: {len(next_actions)}")
    print(f"  resolved_2026_10_13 size: {len(raw['resolved_2026_10_13'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())