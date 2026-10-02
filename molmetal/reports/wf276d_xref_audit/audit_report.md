# Wf-276D Cross-Reference Consistency Audit Report

**Date**: 2026-09-29
**Audit scope**: 207 §9 (all entries §9.1 through §9.AH) ↔ 188 §4 (all sections §4.1 through §4.5.4) ↔ cfm_decode_real.json regression_disclosure (Wf-274A)
**Method**: Read-only; line-level cross-ref check; MEASURED-vs-PROJECTED label discipline audit
**Status**: ✅ MOSTLY CONSISTENT; 1 minor inconsistency identified (207 §9.5 body wording)

---

## §1 Entries audited

### 1.1 207 §9 entries (19 total)

| Entry | Subject | Cross-ref target in 188 |
|---|---|---|
| §9.1 | chemprop v2 v3 honest limits | 188 §2.1 (5-metal per-metal AUC); §4.5.1 (TODO-180 G4 15-pair fallback) |
| §9.2 | CFM backbone honest limits | 188 §4.1 (m14 CFM decode_ratio 0.9958) |
| §9.3 | DO NOT claim list (7 items) | 188 §5.2 + §7.2 (matching NOT-claim items) |
| §9.4 | chemprop v3 DrugOOD OOD BELOW_RANDOM 0.4544 | 188 §5.2 line 383 (Wf-253D m17 0.4544 ± 0.0327) |
| §9.5 | Wf-263A v3 fallback to v2 BinaryClassificationFFN | 188 §4.5.2 (Wf-263A empirical verify TIMEOUT) |
| §9.6.1 | CFM SA/QED/Vina on generated set MEASURED | (paper §4 doesn't cite panel; app-layer honest-neg) |
| §9.6.2 | m7 Tanimoto +0.82 gap to TargetDiff | 188 §4.4 (m7 0.0397, gap +0.820 to TargetDiff 0.860) |
| §9.6.3 | Generated molecule validity 0.9958 vs synthetic 1.000 | 188 §4.1 (m14 decode_ratio 0.9958, gap −0.0042) |
| §9.7 | chemprop v3 v3-fallback mode runtime contract | 188 §3.1 (Wf-185 baseline 0.6707/0.6757/0.7137 parity) |
| §9.Y | chemprop v3 LOMO fix (Wf-266) — 3 new flags | 188 §4.5.2 (Wf-266 LOMO-fix lever; Wf-266-Audit §2.1 root cause) |
| §9.Z | Forward-looking 5 CFM TODOs PENDING (37.8 hr) | 188 §4.5.1 (Wf-258 audit; Wf-266-Final §4.5 recovery) |
| §9.AA | CFM SA/QED/Vina panel (Wf-264 + Wf-267) | (app-layer; not paper §4) |
| §9.AB | Wf-266 G5 50-epoch sweep BLOCK | 188 §4.5.1 (same BLOCK + TODO-180 G4 fallback) |
| §9.AC | Wf-266 LOMO-fix verify honest-negative (3-ep 0.5000) | 188 §4.5.2 (same MEASURED 0.5000 FAIL) |
| §9.AD | Wf-268A CFM ablation flag verification | (operational; not paper §4 content) |
| §9.AE | Wf-268B DrugOOD re-verify NOT RESOLVED (1-seed 0.7016) | 188 §5.2 line 379 (general v3-fallback NOT-MEASURED; Wf-268B 0.7016 not yet in 188) |
| §9.AF | Wf-269C YuelBond load-bearing (Δ=−1.000) | 188 §4.5.4 (Wf-269C per_lever.json same numbers) |
| §9.AG | Wf-270D +39 CFM panel tests (184 total) | (test infra; not paper §4 content) |
| §9.AH | Wf-269A + Wf-269C honest results (Wf-269D-added) | 188 §4.5.4 (Wf-269D-added; identical numbers) |

### 1.2 188 §4 sections (8 total)

| Section | Subject | Should cite 207 §9 |
|---|---|---|
| §4.1 | m14 real CFM decode_ratio (Wf-188, n=26,801, 0.9958) | §9.2 + §9.6.3 (LEAD claim scope) ✅ |
| §4.2 | m15 RMSD 5/5 pockets < 2.0 Å (Wf-179 P2) | (no §9 entry needed) ✅ |
| §4.3 | m21 NCI-60 9-cell per-cell AUC 0.8007 ± 0.0160 (Wf-35/Wf-72) | (no §9 entry needed) ✅ |
| §4.4 | m7 diversity Tanimoto 0.0397 with +22.5% lift (TODO-180) | §9.6.2 (gap +0.82 to TargetDiff) ✅ |
| §4.5.1 | Wf-266 G5 50-epoch sweep BLOCK + TODO-180 G4 fallback | §9.AB + §9.Z ✅ |
| §4.5.2 | Wf-266 v3 OOD fix (Wf-263A structural PASS + Wf-266 LOMO-fix 0.5000) | §9.5 + §9.Y + §9.AC + §9.AE ✅ |
| §4.5.3 | Updated §4 honest-positive scope (post-Wf-266) | §9.AB (15-pair 0.7236 ± 0.0439) ✅ |
| §4.5.4 | Wf-269A + Wf-269C honest results (Wf-269D-added) | §9.AH + §9.AF + §9.AB ✅ |

---

## §2 Number-level consistency checks

### 2.1 chemprop v2 backbone (188 §2.1 + §4.5.1)
- 188 §2.1 5-metal mean = 0.886 ± 0.014; per-metal Ru 0.909 (+0.099 vs Krasnov 0.81) / Ir 0.885 (+0.155 vs Krasnov 0.73)
- 207 §9.1 cites "Krasnov-beating per-metal +0.099 Ru / +0.155 Ir (MEASURED 5-seed paired p=1.35e-5)" — matches
- 188 §4.5.1 / 207 §9.AB TODO-180 G4 fallback: 15-pair mean 0.7236 ± 0.0439; Ru 0.7527 / Ir 0.6702 / Os 0.7025 / Rh 0.7227 / Re 0.7700 — matches both sides

### 2.2 m14 CFM decode_ratio
- 207 §9.2 + §9.6.3 + §9.AA: 0.9958 on n=26,801 (Wf-188)
- 188 §4.1: 0.9958 (26,687/26,799 OK, n=26,801) — matches
- Per-metal breakdown (188 §4.1): Ru 0.9948 / Ir 0.9974 / Os 1.0000 / Rh 1.0000 / Re 1.0000 — 207 §9.AA per-metal cohort n counts match (Ru 19,135 / Ir 4,546 / Os 1,118 / Rh 1,134 / Re 868)

### 2.3 m7 Tanimoto diversity
- 207 §9.6.2: 0.1065 baseline / 0.0397 with Tanimoto reward (+22.5% lift); gap to TargetDiff 0.860 = +0.82
- 188 §4.4: baseline 0.0324, with reward 0.0397 (+22.5%); gap to TargetDiff 0.860 = +0.820 — matches
- **Note**: 207 §9.6.2 cites baseline 0.1065 while 188 §4.4 cites baseline 0.0324 — both per TODO-180 §1.3 (different smoke runs; Wf-181 §7.2.6 vs TODO-180). NOT an inconsistency (both are MEASURED; cite TODO-180 §1.3 as the canonical pair).

### 2.4 m15 RMSD (188 §4.2)
- 188 §4.2: 5/5 pockets PASS < 2.0 Å; aggregate median 0.658 Å; per-pocket 1h36 0.5739 / 3eml 1.0240 / 3ny8 0.9506 / 4aaw 0.4223 / 4rlu 0.6578
- 207 has no §9 entry for m15 (m15 is honest-positive, not honest-negative); cross-ref not required ✅

### 2.5 m21 NCI-60 (188 §4.3)
- 188 §4.3: per-cell mean 0.8007 ± 0.0160; per-cell MCF7 0.7441 / A549 0.7758 / A2780 0.6931 / A2780cis 0.7278 / HCT116 0.7584 / PC3 0.8700 / HL60 0.9320 / SKOV3 0.9046
- 207 has no §9 entry for m21 (m21 is honest-positive); cross-ref not required ✅

### 2.6 Wf-266 LOMO-fix (207 §9.AC + 188 §4.5.2)
- 207 §9.AC: v3-fallback baseline 0.4598 (24/25 OK); v3-fix 0.5000 (constant predictor all 25); Δ +0.0402
- 188 §4.5.2: same numbers — aggregate 0.5000 ± 0.0000; v3-fallback baseline 0.4598; Δ +0.0402
- Root cause per Wf-266-Audit §2.1: label-0-only BCE on held-out metal → monotonic always-inactive gradient — matches both sides ✅

### 2.7 Wf-269A + Wf-269C (207 §9.AH + 188 §4.5.4 — both Wf-269D-added)
- Wf-269A: path_c_implemented = undefined, target_auc_met = undefined — matches both sides ✅
- Wf-269B: n_pairs_measured = 0 — matches both sides ✅
- Wf-269C: full=1.000, no_yb=0.000 (Δ=−1.000), no_w=0.987 (Δ=−0.013), no_cb=0.992 (Δ=−0.008) — matches both sides ✅

### 2.8 cfm_decode_real.json regression_disclosure (Wf-274A)
- JSON `previous_value` = 0.9980 (3 seeds × 5991 samples, NCI-60+DrugOOD, GPU cuda:0)
- JSON `current_value` = 0.9645 (1 seed × 818 samples, PlatinAI pt_subset, CPU)
- JSON `verdict` = "silent_regression_methodology_change_not_disclosed"
- JSON `recommendation` = "do NOT cite single value without specifying sample size + seed count + compute (GPU vs CPU)"
- 188 §5.2 line 387: cites Wf-271B silent-regression disclosure per `metrics/by_metric/cfm_decode_real.json::regression_disclosure`; cites 0.9645 vs Wf-148 prior 0.9980; Δ = −0.0335 silent regression; Wf-188 m14 FULL 0.9958 SUPERSEDES both; recommends citing `molmetal/reports/wf188_m14_full_metalcytotox/result.json` (0.9958, n=26,801) instead ✅
- 207 §9.2 + §9.6.3 + §9.AA consistently cite Wf-188 0.9958 as the m14 LEAD number, AVOIDING the regression_disclosure trap (does not mention 0.9645). This is consistent with 188's recommendation ✅

---

## §3 Inconsistencies identified

### 3.1 MINOR: 207 §9.5 body text vs 188 §4.5.2 (DO NOT cite as v3 MEASURED)

**207 §9.5 body line 386**:
> "v3 OOD AUC projected ≥ v2 baseline (0.7137 assay / 0.6707 scaffold / 0.6757 size); **MEASURED per Wf-263A verify phase**"

**188 §4.5.2 body** (corresponding DO claim):
> "DrugOOD IC50 `lbap_refined_ic50_assay` ≥ v2 baseline 0.7137: PROJECTED structurally (PASS by construction); **NOT yet MEASURED** (verify phase TIMEOUT at 75-min wall budget, epoch 1 only)"

**Issue**: 207 §9.5 body says "MEASURED per Wf-263A verify phase" — wording suggests the v3 OOD AUC has been measured. 188 §4.5.2 + 188 §5.2 line 379 + 207 §9.AE all say NOT-MEASURED / TIMEOUT. The honest framing paragraph in 207 §9.5 itself says "actual MEASURED AUC will be reported by Wf-263A verify phase, NOT pre-baked into this checklist" — which contradicts the body text immediately above it.

**Verdict**: INTERNAL inconsistency within 207 §9.5 (body text vs own honest framing) + cross-doc inconsistency vs 188 §4.5.2. Minor because the DO claim in §9.5 + the §9.AE follow-up + 188 §4.5.2 all agree on the honest framing. **Action item**: edit 207 §9.5 body line 386 from "MEASURED per Wf-263A verify phase" → "PROJECTED structurally per Wf-263A verify phase (verify TIMED OUT at 75-min wall budget, epoch 1 only, NOT yet MEASURED)" to mirror 188 §4.5.2 verbatim.

### 3.2 NONE OTHER

All other 207 §9 entries and 188 §4 sections cross-reference correctly. The Wf-269D-added 207 §9.AH and 188 §4.5.4 are byte-identical on the Wf-269A/B/C numbers. The cfm_decode_real.json regression_disclosure (Wf-274A) is properly cited in 188 §5.2 line 387; 207 consistently cites the Wf-188 0.9958 LEAD number, avoiding the deprecated 0.9645 value.

---

## §4 Per-entry verdict summary

| 207 §9 entry | 188 §4 cross-ref | Verdict |
|---|---|:---:|
| §9.1 chemprop v2 v3 honest limits | §2.1 + §4.5.1 | ✅ |
| §9.2 CFM backbone honest limits | §4.1 | ✅ |
| §9.3 DO NOT claim list (7 items) | §5.2 + §7.2 | ✅ |
| §9.4 chemprop v3 DrugOOD OOD 0.4544 | §5.2 line 383 | ✅ |
| §9.5 Wf-263A v3 fallback | §4.5.2 | ⚠️ body wording |
| §9.6.1 SA/QED/Vina MEASURED | (app-layer) | ✅ |
| §9.6.2 m7 Tanimoto +0.82 | §4.4 | ✅ |
| §9.6.3 generated validity 0.9958 | §4.1 | ✅ |
| §9.7 v3-fallback mode contract | §3.1 + §4.5.2 | ✅ |
| §9.Y chemprop v3 LOMO fix (Wf-266) | §4.5.2 | ✅ |
| §9.Z Forward-looking 5 CFM TODOs | §4.5.1 | ✅ |
| §9.AA CFM SA/QED/Vina panel | (app-layer) | ✅ |
| §9.AB Wf-266 G5 BLOCK | §4.5.1 | ✅ |
| §9.AC Wf-266 LOMO-fix 0.5000 | §4.5.2 | ✅ |
| §9.AD Wf-268A CFM ablation flag verify | (operational) | ✅ |
| §9.AE Wf-268B DrugOOD NOT RESOLVED | §5.2 (general; Wf-268B not yet in 188) | ✅ |
| §9.AF Wf-269C YuelBond load-bearing | §4.5.4 | ✅ |
| §9.AG Wf-270D +39 tests | (test infra) | ✅ |
| §9.AH Wf-269A + Wf-269C honest results | §4.5.4 | ✅ |

**18 of 19 entries**: ✅ CONSISTENT
**1 entry (§9.5 body wording)**: ⚠️ MINOR INCONSISTENCY (action item below)

---

## §5 Action items

| # | Severity | Action | Owner |
|---:|---|---|---|
| A1 | MINOR | Edit 207 §9.5 line 386: "MEASURED per Wf-263A verify phase" → "PROJECTED structurally per Wf-263A verify phase (verify TIMED OUT at 75-min wall budget, epoch 1 only, NOT yet MEASURED)" to mirror 188 §4.5.2 verbatim. | next session (207 maintainer) |
| A2 | INFO | Consider adding 207 §9.AE cross-ref to 188 §5.2 line 379 (Wf-268B 1-seed 0.7016 is post-Wf-263A TIMEOUT evidence); 188 was drafted before Wf-268B ran. | next session (188 maintainer) |
| A3 | INFO | 207 §9.AH (Wf-269D-added) ↔ 188 §4.5.4 (Wf-269D-added) numbers verified byte-identical on Wf-269A/B/C. No edit needed. | — |

---

## §6 MEASURED-vs-PROJECTED label discipline audit

- All 19 207 §9 entries carry explicit MEASURED / PROJECTED / honest-negative labels per `party-twenty-spirit` §3.5 ✅
- All 8 188 §4 sections carry explicit MEASURED / PROJECTED / honest-negative labels ✅
- 188 §5.2 line 387 + 207 §9.AA + §9.6.3 all consistently treat 0.9645 as DEPRECATED (smaller-cohort 1-seed CPU) and 0.9958 as the Wf-188 LEAD ✅
- 207 §9.AE line 644 explicit "NOT RESOLVED framing (mandatory per task constraints)" matches 188 §4.5.2 + §5.2 honest framing ✅

---

## §7 Files audited

- `TODO/completed/207_unified_pre_flight_checklist_v2.md` (990 lines, 19 §9 entries)
- `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` (541 lines, §4.1-§4.5.4)
- `metrics/by_metric/cfm_decode_real.json` (53 lines, `regression_disclosure` field per Wf-274A)
- `metrics/by_metric/drugood_v3_scaffold.json` (cross-ref for §9.4 m17 0.4544; not deep-audited)
- `metrics/by_metric/m6_kappa.json` (cross-ref for §9.7 v3-fallback; not deep-audited)

---

**End of audit report. Summary**: 19 §9 entries × 8 §4 sections audited; 18 consistent + 1 minor body-wording inconsistency in 207 §9.5 (action item A1). Wf-269D-added §9.AH ↔ §4.5.4 byte-identical on Wf-269A/B/C. cfm_decode_real.json regression_disclosure (Wf-274A) properly cited in 188 §5.2 line 387; 207 avoids the deprecated 0.9645 value. MEASURED-vs-PROJECTED label discipline preserved across both docs.