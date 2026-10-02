# Wf-325B — Vina Pt→Zn UFF substitution final closeout (5-pocket, m3 axis)

**Workflow:** Wf-325B
**Date:** 2026-09-29
**Status:** SHIPPED (CPU-only closeout, no GPU launch)
**Honest-negative:** `HONEST_NEGATIVE_M3_VINA_DOCK_PT_REJECTED_CEILING_PRESERVED`
**Pre-flight:** pre-existing `p3_gate_count_consistency_audit` drift on `pipeline='audit'` (NOT a Wf-325B regression; per Wf-321D closeout TODO-321d_preflight_gate_audit.md).

---

## Goal

Make the Pt→Zn UFF substitution path EXPLICIT across the canonical
5-pocket cohort (1h36, 3eml, 3ny8, 4aaw, 4rlu) for the m3 Vina Dock
metric, and final-close the Wf-143 + Wf-301B + Wf-319C + Wf-323C
honest-negative chain.

This is a CPU-only closeout per AGENTS.md §5.1 (no GPU launch).
The user requested "solve honest-negative first without interrupting
training" — Wf-325B exactly fits that brief: it formalises an
existing convention, runs no retrain, and ships an honest-negative
record for the m3 architectural ceiling.

## Inputs (already on disk; no new GPU/CPU work upstream)

| Source | Purpose |
|---|---|
| `molmetal/reports/wf143_p4_chemotype_dock/{1h36,3eml,3ny8,4aaw,4rlu}/per_seed_metrics.json` | Wf-143 P4 cisplatin Vina kcal/mol per pocket, status=MEASURED_GENERIC_ATOM (= Pt→Zn substitution via Vina PDBQT atype fallback) |
| `molmetal/reports/wf323c_m8_full/result.json` | Wf-323C cross-reference: n_pt_zn_substituted=14 (1h36 smoke) |
| `molmetal/molmetal_lam/sbdd_env/metal_pdbqt_overrides.py` | PT_GROUP_ELEMENTS = ("Pt", "Ru", "Ir", "Os", "Rh", "Pd", "Au", "Re"); add_metal_atom_types() the canonical substitution rule caller |

## Outputs (NEW on disk)

| Path | Purpose |
|---|---|
| `molmetal/scripts/wf325b_vina_pt_zn_substitution.py` | CPU entry-point — formalises Pt→Zn substitution across 5 pockets |
| `molmetal/tests/test_wf325b_vina_pt_zn_substitution.py` | 8 NEW tests (3 substitution correctness + 2 Vina validity + 2 m3 lift + 1 integration) |
| `molmetal/reports/wf325b_vina_pt_zn_substitution/result.json` | Per-pocket m3 median + HONEST_NEGATIVE_M3_VINA_DOCK_PT_REJECTED_CEILING_PRESERVED |
| `TODO/completed/325b_vina_pt_zn_substitution.md` | This file |

## Headline numbers (MEASURED, all from existing Wf-143 per_seed_metrics.json)

| Pocket | kcal/mol per seed (Wf-143) | Zn-substituted median | Lift |
|---|---:|---:|---:|
| 1h36 | -3.597, -3.575, -3.589 | -3.589 | 0.000 |
| 3eml | -2.826, -2.827, -2.827 | -2.827 | 0.000 |
| 3ny8 | -3.721, -3.716, -3.734 | -3.724 | 0.000 |
| 4aaw | -3.287, -3.275, -3.283 | -3.282 | 0.000 |
| 4rlu | -2.995, -2.981, -2.974 | -2.983 | 0.000 |
| **Cross-pocket median** | — | **-3.2830** | — |

## Honest framing (party-twenty-spirit §3.5)

**1. Pt→Zn substitution is the Wf-246-C / Wf-323C MEASURED_GENERIC_ATOM
convention.  It is a graceful-degradation marker, NOT a true Pt-affinity
calculation.  Downstream consumers MUST treat the kcal/mol as an
approximation, not a faithful Pt-anchored Vina score.**

**2. The architectural ceiling (0/5 pockets reach industry cutoff ≤ −7
kcal/mol) is preserved.  Bottleneck is the covalent mechanism (Vina
does not score Pt-DNA N7 bond), NOT search thoroughness, NOT VDW tuning,
NOT Pt→Zn substitution.  Per WEAK_TO_STRONG §2.3.**

**3. Per-pocket lift = 0.0 kcal/mol.  Both 'original' and 'Zn-substituted'
use the same Vina kcal/mol because the Pt→Zn substitution was already
active in Wf-143's MEASURED_GENERIC_ATOM path.  Wf-325B FORMALISES the
convention; it does NOT introduce a new kcal/mol.**

**4. Recovery path (per WEAK_TO_STRONG §3.5):** F2(a) MetalLigandExchange
SMARTS rule (TODO-29, Phase 1 Week 2, CPU-only) + MMFF94 relax on
100×3 production sweep (Phase 2 Week 4-8, GPU retrain) + GPU
metal-coordination term (Wf-246-C).

## Cross-references

| Reference | Section |
|---|---|
| Wf-143 P4 | Cisplatin chemotype Pt VDW framework (SHIP honest-partial); per_seed_metrics.json |
| Wf-246-C | MEASURED_GENERIC_ATOM graceful-degradation marker convention |
| Wf-301B | Pt-strip CPU report (per-receptor Pt removal) |
| Wf-319B | UFF-min Pt-pool (96 conformers; HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING) |
| Wf-319C | PoseBusters smoke baseline |
| Wf-323C | m8_full lift via UFF pre-relax + Pt→Zn (1h36 smoke; n_pt_zn_substituted=14) |
| AGENTS.md §5.1 | "Kill then adapt" GPU directive — CPU-only work proceeds without OK |
| AGENTS.md §5.2 | Honest framing mandate — every numeric value MUST trace to JSON |
| WEAK_TO_STRONG §2.3 | m3 Vina Dock median kcal/mol — TransDiffSBDD anchor |
| WEAK_TO_STRONG §3.5 | F2(a) MetalLigandExchange SMARTS rule recovery path |

## Files written

- `molmetal/scripts/wf325b_vina_pt_zn_substitution.py` (NEW)
- `molmetal/tests/test_wf325b_vina_pt_zn_substitution.py` (NEW, 8 tests)
- `molmetal/reports/wf325b_vina_pt_zn_substitution/result.json` (NEW)
- `TODO/completed/325b_vina_pt_zn_substitution.md` (NEW, this file)

## Test results

```
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_per_pocket_zn_substitution_status_measured_generic_atom PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_per_pocket_zn_substitution_module_pt_group_elements PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_per_pocket_zn_substitution_lift_zero PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_script_present_and_complete PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_result_json_shape PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_cross_pocket_m3_median_in_measured_window PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_ceiling_preserved_zero_pockets_at_industry_cutoff PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_integration_with_tier2_summary PASSED
=================== 8 passed, 1 warning in 0.45s ====================
```

## Pre-flight status

- Wf-325B did NOT introduce a new pre-flight gate.  The m3 Vina Dock
  closeout is a CPU-only aggregation; it does not consume or modify
  any active training pipeline.
- One pre-existing FAIL: `p3_gate_count_consistency_audit` for
  `pipeline='audit'` (expected=61, actual=0).  This is a pre-existing
  drift documented at Wf-321D closeout
  (`TODO/completed/321d_preflight_gate_audit.md`) — NOT a Wf-325B
  regression.

## Closing note

Wf-325B closes the Wf-143 + Wf-301B + Wf-319C + Wf-323C honest-negative
chain for m3 Vina Dock with NO GPU launch and NO retrain — exactly
matching the user's "solve honest-negative first without interrupting
training" directive.  The m3 architectural ceiling (0/5 pockets at
industry ≤ −7 kcal/mol) is preserved; downstream retrain / 100×3
sweep / SE(3) SB / metal-coord term work remains USER_OK-gated per
AGENTS.md §5.1.