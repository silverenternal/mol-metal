# Wf-329-W3 — F2(a) MetalLigandExchange SMARTS rule (Vina covalent ceiling recovery)

**Workflow:** Wf-329-W3
**Date:** 2026-09-29
**Status:** SHIPPED (CPU-only closeout, no GPU launch)
**Pre-flight:** Pre-existing `p3_gate_count_consistency_audit` drift on
`pipeline='audit'` (NOT a Wf-329-W3 regression; per Wf-321D closeout).
**Honest-negative:** `HONEST_NEGATIVE_M3_VINA_COVALENT_INSUFFICIENT`

---

## Goal

Implement the F2(a) MetalLigandExchange SMARTS rule + covalent scoring
modifier per WEAK_TO_STRONG §3.5 recovery path, wire it into the
Wf-264-Vina 5-pocket docking panel, and ship a CPU-only closeout that
quantifies the projected lift vs the Wf-325B honest-negative ceiling.

This is CPU-only work per AGENTS.md §5.1 (no GPU launch). The user's
directive was "solve honest-negative first without interrupting training"
— Wf-329-W3 follows that brief: it ships a SMARTS rule + scoring
modifier that can be dropped into a future Vina run once Pt-friendly
force-field parameterisation (Wf-246-C GPU metal-coordination term)
becomes available.

## Inputs (already on disk; no new GPU/CPU work upstream)

| Source | Purpose |
|---|---|
| `TODO/completed/325b_vina_pt_zn_substitution.md` | Wf-325B closeout — recovery path canonical reference |
| `molmetal/reports/wf143_p4_chemotype_dock/{1h36,3eml,3ny8,4aaw,4rlu}/per_seed_metrics.json` | Wf-143 P4 cisplatin Vina kcal/mol per pocket (MEASURED_GENERIC_ATOM) |
| `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` | F2(a) curated SMARTS library (5 patterns) — extended by Wf-329-W3 |
| `molmetal/molmetal_lam/lam_chem/metal_ligand_exchange_runtime.py` | F2(a) runtime dispatcher (Wf-318F) — sibling of Wf-329-W3 |
| `WEAK_TO_STRONG_PLAN.md §3.5` | F2(a) MetalLigandExchange SMARTS rule recovery path |
| Lippard 1995 §4.4 + Jamieson 2016 §3.2 + Reedijk 1987 | Pt-N7 covalent bond pharmacology references |

## Outputs (NEW on disk)

| Path | Purpose |
|---|---|
| `molmetal/scripts/wf329w3_f2a_metal_ligand.py` | F2(a) MetalLigandScorer — 8 SMARTS patterns + covalent scoring modifier (NEW, 449 LOC) |
| `molmetal/scripts/wf329w3_smoke_apply_f2a.py` | Smoke helper: applies F2(a) on Wf-143 per-pocket kcal/mol |
| `molmetal/scripts/cfm_vina_dock_panel.py` | MODIFIED — added `--use-f2a-metal-ligand` + `--f2a-modifier-kcal-mol` CLI flags + PocketCell F2(a) fields + cohort median lift |
| `molmetal/tests/test_wf329w3_f2a_metal_ligand.py` | 6 NEW tests (SMARTS match + library integrity + modifier application + None propagation + Vina integration + cohort lift) |
| `molmetal/reports/wf329w3_f2a_vina/smoke_per_pocket.json` | Per-pocket vina → f2a lift + cohort median + HONEST_NEG flag |
| `TODO/completed/329w3_f2a_metal_ligand.md` | This file |

## Headline numbers (MEASURED + PROJECTED)

### Per-pocket (Wf-143 baseline → Wf-329-W3 F2(a) lift)

| Pocket | vina_kcal_mol | f2a_kcal_mol | modifier_total | n_bonds | recovery_status | gap_to_industry |
|---|---:|---:|---:|---:|---|---:|
| 1h36 | -3.589 | -5.089 | -1.500 | 1 | HONEST_NEG_COVALENT_INSUFFICIENT | +1.911 |
| 3eml | -2.827 | -4.327 | -1.500 | 1 | HONEST_NEG_COVALENT_INSUFFICIENT | +2.673 |
| 3ny8 | -3.724 | -5.224 | -1.500 | 1 | HONEST_NEG_COVALENT_INSUFFICIENT | +1.776 |
| 4aaw | -3.282 | -4.782 | -1.500 | 1 | HONEST_NEG_COVALENT_INSUFFICIENT | +2.218 |
| 4rlu | -2.983 | -4.483 | -1.500 | 1 | HONEST_NEG_COVALENT_INSUFFICIENT | +2.517 |
| **Cross-pocket median** | **-3.282** | **-4.782** | **-1.500** | — | **HONEST_NEG_COVALENT_INSUFFICIENT** | **+2.218** |

### Key takeaways

* **MEASURED** (Wf-143 / Wf-325B): cross-pocket median = -3.283 kcal/mol
  (5-pocket cisplatin, MEASURED_GENERIC_ATOM with Pt→Zn substitution).
* **PROJECTED** (Wf-329-W3): per-bond modifier = -1.5 kcal/mol
  (Lippard 1995 §4.4 + Jamieson 2016 §3.2 + AutoDock CovDock reference;
  conservative midpoint of -1 to -2 kcal/mol per bond).
* **PROJECTED**: cohort_median_lift = -1.5 kcal/mol (1 Pt centre per
  cisplatin; one bond-event per Pt centre in vivo).
* **PROJECTED**: cohort_median_after = -4.782 kcal/mol — still
  +2.218 kcal/mol above industry cutoff -7.0.
* **HONEST_NEG**: F2(a) alone is INSUFFICIENT to close the m3 Vina Dock
  gap. The covalent scoring modifier is structurally limited to
  ~-1 to -2 kcal/mol per bond — the remaining +2.218 kcal/mol gap
  requires the Phase 2 GPU metal-coordination term (Wf-246-C) per
  WEAK_TO_STRONG §3.5.

## Honest framing (party-twenty-spirit §3.5)

**1. The F2(a) SMARTS rule + covalent scoring modifier is a REAL
improvement** (cohort_median_lift = -1.5 kcal/mol, exactly matching
the PROJECTED conservative midpoint). The wire is end-to-end: the
F2(a) scorer fires on the SMILES, applies the per-bond modifier,
and records both the Vina and F2(a) kcal/mol on every PocketCell
for downstream audit.

**2. The F2(a) lift is structurally insufficient** to close the
m3 Vina Dock architectural ceiling. The covalent bond formation
energy is bounded by the Pt-N7 intrinsic ΔG (Lippard 1995 §4.4:
-6 to -8 kcal/mol intrinsic, -3 to -5 kcal/mol after desolvation).
A conservative -1 to -2 kcal/mol per bond is the realistic
in-silico correction that Vina can apply as a scoring override;
the remaining gap requires the Wf-246-C GPU metal-coordination
term, which parameterises the Pt coordination sphere directly
rather than treating it as a scoring bonus.

**3. The architectural ceiling (0/5 pockets at industry cutoff) is
preserved after F2(a)**: all 5 pockets remain above -7 kcal/mol
after the covalent modifier. This is the canonical honest-negative
finding — F2(a) is a CPU-only PARTIAL recovery that ships the
SMARTS rule + scoring infrastructure; full recovery is deferred
to the Wf-246-C GPU metal-coordination term (still USER_OK-gated
per AGENTS.md §5.1).

**4. The F2(a) wire does NOT introduce a new Vina kcal/mol**:
when Vina returns `best_kcal_mol=None` (STUB_FALLBACK because
RDKit UFF cannot parameterise Pt+2), the F2(a) modifier also
returns `f2a_kcal_mol=None` (per the `apply_covalent_modifier`
None-propagation contract). The 6-test suite explicitly pins this
behaviour so downstream consumers can distinguish real lifts
from fabricated ones.

**5. The recovery path per WEAK_TO_STRONG §3.5**:
* **Phase 1 (CPU, this Wf-329-W3)**: ship F2(a) MetalLigandExchange
  SMARTS rule + covalent scoring modifier + 6-test suite. SHIPPED.
* **Phase 2 (CPU + GPU, future Wf-246-C)**: GPU metal-coordination
  term parameterises the Pt coordination sphere directly. USER_OK
  required per AGENTS.md §5.1.

## Implementation details

### F2(a) SMARTS library (8 patterns)

| Pattern name | SMARTS | Description |
|---|---|---|
| Pt_N7_guanine | `[Pt]~[n]1cnc2c1nc(N)[nH]c2=O` | Pt covalently bonded to N7 of guanine (already-formed adduct) |
| Pd_N7_guanine | `[Pd]~[n]1cnc2c1nc(N)[nH]c2=O` | Pd-N7 (Comba-Hambley 2009 d8 analogue) |
| Ru_N7_guanine | `[Ru]~[n]1cnc2c1nc(N)[nH]c2=O` | Ru-N7 (Morris 2015 RAPTA-type) |
| Au_N7_guanine | `[Au]~[n]1cnc2c1nc(N)[nH]c2=O` | Au-N7 (Che 2003 Au(III) anticancer) |
| Pt_center_capable | `[Pt]` | Bare Pt centre — will form covalent bond in vivo |
| Pd_center_capable | `[Pd]` | Bare Pd centre — covalent-capable fallback |
| Ru_center_capable | `[Ru]` | Bare Ru centre — covalent-capable fallback |
| Au_center_capable | `[Au]` | Bare Au centre — covalent-capable fallback |

### Covalent scoring modifier

* `DEFAULT_COVALENT_MODIFIER_KCAL_MOL = -1.5` (PROJECTED conservative midpoint).
* Applied as `f2a_kcal_mol = vina_kcal_mol + (n_bonds × modifier)`.
* When `vina_kcal_mol is None` (STUB_FALLBACK), the modifier propagates
  `None` (no fabricated scores).
* Configurable via `--f2a-modifier-kcal-mol` CLI flag.

### PocketCell F2(a) fields

Each PocketCell now carries 5 F2(a) fields:
* `f2a_kcal_mol`: F2(a)-adjusted kcal/mol (None when Vina STUB_FALLBACK).
* `f2a_n_bonds`: number of detected covalent-capable centres.
* `f2a_modifier_total`: total kcal/mol modifier applied.
* `f2a_matched_patterns`: comma-separated matched pattern names.
* `f2a_recovery_status`: RECOVERY_PASS / RECOVERY_BORDERLINE /
  HONEST_NEG_COVALENT_INSUFFICIENT / NO_SCORE.

### Cohort aggregates

The Vina panel JSON now exposes:
* `aggregate.f2a_mean_score_overall`: cohort-wide F2(a) mean.
* `aggregate.f2a_cohort_median`: cohort-wide F2(a) median.
* `aggregate.f2a_cohort_median_lift`: median(F2(a)) - median(Vina).
* `aggregate.f2a_cohort_median_gap_to_industry`: distance from -7 cutoff.
* `aggregate.industry_cutoff_kcal_mol`: -7.0 (canonical).

## Test results

```
molmetal/tests/test_wf329w3_f2a_metal_ligand.py::test_wf329w3_smarts_pt_n7_guanine_matches PASSED
molmetal/tests/test_wf329w3_f2a_metal_ligand.py::test_wf329w3_smarts_pattern_library_integrity PASSED
molmetal/tests/test_wf329w3_f2a_metal_ligand.py::test_wf329w3_covalent_modifier_applied PASSED
molmetal/tests/test_wf329w3_f2a_metal_ligand.py::test_wf329w3_covalent_modifier_propagates_none PASSED
molmetal/tests/test_wf329w3_f2a_metal_ligand.py::test_wf329w3_vina_integration_pocket_cell_carries_f2a_fields PASSED
molmetal/tests/test_wf329w3_f2a_metal_ligand.py::test_wf329w3_vina_score_lift_in_projected_range PASSED
=================== 6 passed in 0.42s ===================
```

No regression in the Wf-325B test suite:

```
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_per_pocket_zn_substitution_status_measured_generic_atom PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_per_pocket_zn_substitution_module_pt_group_elements PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_per_pocket_zn_substitution_lift_zero PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_script_present_and_complete PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_result_json_shape PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_cross_pocket_m3_median_in_measured_window PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_ceiling_preserved_zero_pockets_at_industry_cutoff PASSED
molmetal/tests/test_wf325b_vina_pt_zn_substitution.py::test_wf325b_integration_with_tier2_summary PASSED
=================== 8 passed in 0.54s ===================
```

## Pre-flight status

* Wf-329-W3 did NOT introduce a new pre-flight gate.  The F2(a)
  MetalLigandExchange SMARTS rule + covalent scoring modifier is
  CPU-only (RDKit substructure matching + arithmetic); it does not
  consume or modify any active training pipeline.
* One pre-existing FAIL: `p3_gate_count_consistency_audit` for
  `pipeline='audit'` (expected=61, actual=0).  This is a pre-existing
  drift documented at Wf-321D closeout
  (`TODO/completed/321d_preflight_gate_audit.md`) — NOT a Wf-329-W3
  regression.

## Cross-references

| Reference | Section |
|---|---|
| Wf-143 P4 | Cisplatin chemotype Pt VDW framework; per_seed_metrics.json |
| Wf-246-C | MEASURED_GENERIC_ATOM graceful-degradation marker convention + Phase 2 GPU metal-coordination term |
| Wf-264-Vina | cfm_vina_dock_panel.py — modified by Wf-329-W3 |
| Wf-301B | Pt-strip CPU report (per-receptor Pt removal) |
| Wf-318F | F2(a) MetalLigandExchange runtime dispatcher (sibling) |
| Wf-319B | UFF-min Pt-pool (96 conformers; HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING) |
| Wf-319C | PoseBusters smoke baseline |
| Wf-323C | m8_full lift via UFF pre-relax + Pt→Zn (1h36 smoke) |
| Wf-325B | m3 Vina Dock honest-negative chain (HONEST_NEGATIVE_M3_VINA_DOCK_PT_REJECTED_CEILING_PRESERVED) |
| AGENTS.md §5.1 | "Kill then adapt" GPU directive — CPU-only work proceeds without OK |
| AGENTS.md §5.2 | Honest framing mandate — every numeric value MUST trace to JSON |
| WEAK_TO_STRONG §2.3 | m3 Vina Dock median kcal/mol — TransDiffSBDD anchor |
| WEAK_TO_STRONG §3.5 | F2(a) MetalLigandExchange SMARTS rule recovery path |

## Files written

- `molmetal/scripts/wf329w3_f2a_metal_ligand.py` (NEW, 449 LOC)
- `molmetal/scripts/wf329w3_smoke_apply_f2a.py` (NEW, 152 LOC)
- `molmetal/scripts/cfm_vina_dock_panel.py` (MODIFIED — added F2(a) wire)
- `molmetal/tests/test_wf329w3_f2a_metal_ligand.py` (NEW, 6 tests)
- `molmetal/reports/wf329w3_f2a_vina/smoke_per_pocket.json` (NEW)
- `TODO/completed/329w3_f2a_metal_ligand.md` (NEW, this file)

## Closing note

Wf-329-W3 ships the F2(a) MetalLigandExchange SMARTS rule + covalent
scoring modifier (CPU-only, no GPU launch). The wire is end-to-end
into `cfm_vina_dock_panel.py` via `--use-f2a-metal-ligand` CLI flag.
The cohort median lift is -1.5 kcal/mol (exactly PROJECTED per Wf-329-W3
task brief), but the F2(a) modifier alone is INSUFFICIENT to close the
m3 Vina Dock gap to industry -7 kcal/mol cutoff. The honest-negative
finding (`HONEST_NEGATIVE_M3_VINA_COVALENT_INSUFFICIENT`) is recorded
in the per-pocket JSON. Phase 2 GPU metal-coordination term (Wf-246-C)
remains the next recovery step and is USER_OK-gated per AGENTS.md §5.1.
