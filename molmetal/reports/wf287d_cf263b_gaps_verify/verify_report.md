# Wf-287D: Cf-263B TODO-207 §9 6 Gaps Verify (post-Wf-282B + Wf-284B)

**Date**: 2026-09-29
**Source**: `molmetal/reports/wf263b_audit/gap_list.md` (6 gaps identified 2026-09-28)
**Verify scope**: post-Wf-282B + Wf-284B + all intervening waves (Wf-263A..Wf-284D)
**Audit method**: Read TODO-207 v2 §9 subsections verbatim + cross-check against wave reports + git diff `metrics/by_metric/cfm_decode_real.json`

---

## §1 Method

1. Read `molmetal/reports/wf263b_audit/gap_list.md` §2 (6 gaps).
2. Read `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 (lines 352-871).
3. Map each gap to its current §9.N entry (or absence).
4. Verify fix completeness via cross-wave ledger + per-gap MEASURED evidence path.
5. Identify remaining work / partial fixes / regressions.

---

## §2 Per-gap status table

| Gap # | Original target §9.N | Current §9.N | Status | Verdict |
|---:|---|---|---|---|
| **G1** CFM application-layer NEVER MEASURED (SA/QED/Vina) | §9.4 NEW | §9.6 (sub: §9.6.1 MEASURED SA 3.289 / QED 0.280 / Vina 0/500 + largest-fragment 96/100; §9.6.1.1 wave ledger; §9.6.2 Tanimoto; §9.6.3 validity 0.9958) + §9.AA (panel scripts) + §9.AD (flag verify) + §9.AF (YuelBond load-bearing) | **FIXED** | Wf-264 + Wf-267-Panel + Wf-268A + Wf-269C all shipped; n=26,801 SA+QED measured; n=20 largest-fragment Vina measured |
| **G2** m7 Tanimoto gap +0.820 (off vs MEASURED 0.7535) | §9.5 NEW | §9.6.2 (gap +0.82 + calc 0.86−0.1065=0.7535 + TODO-180 §3.3 0.0397 post-Tanimoto-reward projection) | **FIXED** (with caveat) | Doc now cites both +0.82 round label AND 0.7535 actual calc; projection 0.0397 cited as still well below TargetDiff (gap preserved) |
| **G3** chemprop v3 m17 per-metal coverage uneven (Ru 4/4, Ir 3/4, Os 1/4) | §9.6 NEW | §9.4 (LEAD-disqualification rationale BELOW_RANDOM 0.4544) + §9.5 (Wf-263A v3-fallback fix) + §9.AE (Wf-268B backbone mismatch) + §9.AI.1 (Wf-275B ECFP4 adapter SHIPPED) + §9.AI.7 (Wf-277B re-eval PENDING) | **PARTIAL** | Per-metal coverage (Ru 4/4, Ir 3/4, Os 1/4 ckpts) NOT explicitly broken out as a single §9.N row; instead distributed across §9.4 + §9.AE + §9.AI.1. The aggregate BELOW_RANDOM 0.4544 honest-negative IS preserved (§9.4); per-metal ckpt coverage gap remains informal |
| **G4** Wf-253d `CHECKPOINT_LOADED_FORWARD_DEFERRED` state — no §9 dedicated entry | §9.7 NEW | §9.AE (Wf-268B DrugOOD re-verify NOT RESOLVED, backbone mismatch) + §9.AI.1 (ECFP4 adapter fix) + §9.AI.7 (re-eval PENDING) | **PARTIAL** | §9.AE names the Wf-268B re-verify + backbone mismatch, which IS the closest analog to the §7.2.7 invariant. But the EXPLICIT invariant name `CHECKPOINT_LOADED_FORWARD_DEFERRED` is not given a dedicated §9.N entry. Forward-pass reeval state is captured indirectly |
| **G5** m6_kappa MetalCytoToxDB validation NEGATIVE_WEAK | §9.8 NEW | §9.6.1.1 (wave ledger row: `Wf-253d m6 \| honest-negative \| m6_kappa NEGATIVE_WEAK Pearson r=0.081`) | **FIXED** (single-line) | Catalogued as one row in §9.6.1.1 wave ledger; per-metal breakdown (Re r=-0.1843 NEGATIVE) NOT explicitly cited. Acceptable compact catalog form |
| **G6** 5 CFM TODO forward-looking (TODO-119 P1.3/P1.4 + TODO-180/181/182) | §9.9 NEW | §9.Z (Z.1-Z.5 explicit table with 37.8 hr total; 39 waves; GPU hours) | **FIXED** | Full enumeration shipped: Z.1 TODO-180 G4 → ✅ MEASURED COMPLETE per §9.AI.4 (14/15); Z.2 TODO-181 G5 → 1/15 MEASURED per §9.AK (Ru seed=0); Z.3 TODO-182 G6 PENDING; Z.4 TODO-119 P1.3 PENDING; Z.5 TODO-119 P1.4 → reduced to cross-cohort per §9.AF |

**Summary**:
- **n_gaps_total = 6**
- **n_fixed = 4** (G1, G2, G5, G6)
- **n_partial = 2** (G3, G4)
- **n_unfixed = 0** (all 6 gaps have at least partial §9 catalog entry)

---

## §3 Regression / new finding (Wf-284B / Wf-287D discovery)

### §3.1 `metrics/by_metric/cfm_decode_real.json` silent regression (Wf-271B discovery)

Per `git diff HEAD -- metrics/by_metric/cfm_decode_real.json`:
- **Previous value**: 0.9980 (3 seeds × 5991 samples, NCI-60+DrugOOD, GPU cuda:0, 2026-09-27, 281s wall)
- **Current value**: 0.9645 (1 seed × 818 samples, PlatinAI pt_subset, **CPU**, 2026-09-27, 128s wall)
- **Verdict**: `silent_regression_methodology_change_not_disclosed`
- **Field added**: `regression_disclosure` block (discovered_in_wf=Wf-271B, discovered_date=2026-09-29)
- **Recommendation**: do NOT cite single value without specifying sample size + seed count + compute (GPU vs CPU)

This is **NOT** in the Wf-263B 6-gap list but is a NEW honest-negative introduced post-audit. §9.6.3 cites "0.9958 on n=26,801" (Wf-188 result) — the Wf-264-era 0.9980 figure is superseded by Wf-188 + Wf-264 SA/QED panel. The `metrics/by_metric/cfm_decode_real.json` 0.9645 figure is the cfm_decode_real.json current on-disk artifact and is NOT cited in §9 anywhere as the LEAD-anchor. **No new §9.N entry needed** but the regression_disclosure block is the on-disk mitigation.

### §3.2 Wf-284B cfm_wasserstein_loss.py false-positive (FIXED)

Per `molmetal/reports/wf284b_cfm_wasserstein_recovery/recovery_report.md`:
- Originally reported "missing" path `lam_chem/cfm_wasserstein_loss.py` was a stale gate definition
- Actual file at `molmetal/molmetal_lam/reward/wasserstein_loss.py` (32138 bytes)
- Wf-260 already corrected the gate path; Wf-282D was a false-positive
- Wf-282 GREEN: 25/25 PASS (P0 BLOCKER 3/3, P1 VERIFY 16/16, P2 DOCUMENTED 6/6)
- **No further action required**

---

## §4 Remaining work

| Gap | Remaining work | Estimated effort |
|---|---|---|
| **G3** per-metal coverage explicit row | Add a §9.AI.7-bis or extend §9.4 with per-metal ckpt breakdown (Ru 4/4, Ir 3/4, Os 1/4 ckpts MEASURED; 4/12 DEFERRED_NO_CHECKPOINT per Wf-253d) | ~5 min doc edit |
| **G4** CHECKPOINT_LOADED_FORWARD_DEFERRED naming | Add a §9.AE.0-bis or cross-ref §7.2.7 explicitly in §9.AE preamble | ~3 min doc edit |
| **G5** per-metal breakdown (Re r=-0.1843) | Optional: add Re Rh NEGATIVE per-metal cells to §9.6.1.1 row text | ~2 min doc edit |
| **Wf-277A §9.AF source-value mismatch** | PENDING per §9.AI.6; edit §9.AF verbatim Δ to align with `wf269c_real_ablation/per_lever.json` (no_w Δ=−0.013, no_cb Δ=−0.008, NOT Δ=0.000) | ~5 min doc edit (ultracode-suitable) |
| **Wf-277B DrugOOD v3 ECFP4 re-eval** | PENDING per §9.AI.7; run Wf-275B ECFP4 adapter on DrugOOD `lbap_refined_ic50_assay` scaffold split, target ≥ 0.7137 | ~75 min GPU wall |

---

## §5 Honest framing (party-twenty-spirit §3.5)

- 4/6 gaps fully fixed; 2/6 partially fixed; 0/6 unfixed.
- The 2 partials (G3, G4) are honest-positive-status: the underlying negative finding IS documented in §9 (BELOW_RANDOM 0.4544 LEAD-disqualification, backbone mismatch), just not as a single dedicated row with explicit ckpt-coverage or invariant-name breakdown.
- No new §9.N entries are required for the regressions (cfm_decode_real 0.9645 silent regression has the on-disk `regression_disclosure` block; Wf-284B false-positive has Wf-260 prior fix).
- The 5 CFM TODO forward-looking catalog (§9.Z) is the most complete fix (G6) — Z.1 G4 closed per §9.AI.4, Z.2 G5 1/15 closed per §9.AK, Z.3 G6 PENDING, Z.4 P1.3 PENDING, Z.5 P1.4 reduced-to-cross-cohort per §9.AF.
- The Wf-277A doc edit (verbatim Δ alignment in §9.AF) is the highest-leverage remaining fix; it is a small text-edit suitable for parallel ultracode execution.

---

## §6 Cross-references

- Wf-263B-Audit source: `molmetal/reports/wf263b_audit/gap_list.md`
- TODO-207 v2 §9 (target catalog): `TODO/completed/207_unified_pre_flight_checklist_v2.md` lines 352-871
- §9.6 CFM application-layer honest-negatives: lines 390-470 (sub: 6.1 / 6.1.1 / 6.2 / 6.3)
- §9.Z 5 CFM TODOs: lines 511-554
- §9.AA CFM SA/QED/Vina panel: lines 616-645
- §9.AF Wf-269C YuelBond load-bearing: lines 696-725
- §9.AI.6 Wf-277A PENDING doc edit: lines 825-831
- §9.AI.7 Wf-277B PENDING DrugOOD ECFP4 re-eval: lines 833-839
- §9.AK Wf-282 G5 Ru seed=0 MEASURED test_auc=0.7716: lines 843-871
- Wf-284B recovery report: `molmetal/reports/wf284b_cfm_wasserstein_recovery/recovery_report.md`

---

**End of Wf-287D verify report. n_gaps_total=6, n_fixed=4, n_partial=2, n_unfixed=0.**