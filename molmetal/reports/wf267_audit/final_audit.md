# Wf-267-Audit — paper §4 Numeric Claim Re-Audit (Post-Fix Verification)

**Date**: 2026-09-28
**Workflow**: Wf-267-Audit
**Methodology**: Re-run paper §4 honest-claim audit per Wf-264-Audit method (`molmetal/reports/wf264_audit/paper_4_claim_audit.md`)
**Target file**: `TODO/completed/188_paper_4_krasnov_protocol_aligned.md`
**Owner**: Wf-267 ultracode
**Constraint**: READ-ONLY audit (no edits to paper §4 or source files).
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 不掩盖; every number traces to JSON path or is honest-PROJECTED) + `mol-metal-paper-writing` §4 verbatim template + `mol-metal-gpu-execution` §7 MEASURED-vs-PROJECTED tag discipline.

---

## §0 Acceptance gate summary (FLAT schema)

| Gate | Pre-fix (Wf-264) | Post-fix (Wf-267) | Verdict |
|---|---:|---:|---|
| **G1_n_aspirational_post** | 0 | **0** | ✅ (no claim without MEASURED source) |
| **G2_n_projected_as_measured_post** | 5 | **2** | ⚠️ 3 FIXED, 2 RESIDUAL (different locations: §2.2 line 110 + §8 line 388) |
| **G3_n_seed_count_errors_post** | 1 | **0** | ✅ FIXED |
| **G4_n_partial_disclosure_errors_post** | 1 | **0** | ✅ FIXED |
| **G5_n_total_violations_post** | 5 | **2** | ⚠️ Net reduction -3 violations; 2 residual NEW findings (same root cause as Wf-264 §3.1/§3.2 but in different sections) |
| **G6_audit_path** | — | **`molmetal/reports/wf267_audit/final_audit.md`** | ✅ |
| **honest_framing_note** | — | Paper §4 §3.5 honest framing has materially improved post-Wf-264 fixes. All 5 originally-identified §3.5 violations in §0.1 / §2.1 / §2.3 / §5.1 / §6.1 / §9 are now corrected. **However, the Wf-264 audit focused on §0.1 + §2.1 + §2.3 + §5.1 + §6.1 + §9 and missed 2 residual mis-citations at §2.2 line 110 + §8 line 388.** Both residuals are the SAME root cause as Wf-264 §3.1 (chemprop_v2_5metal_5seed_50epoch.json) and §3.2 (cfm_decode_real.json) but were not flagged because Wf-264 audited §9 + §2.1 only. The fixes at §9 line 410 + §9 line 411 (with "(NOT...)" disclaimers) are correctly applied; §2.2 line 110 + §8 line 388 still cite the same problematic JSON files without the same disclaimer treatment. |

---

## §1 Methodology — re-audit per Wf-264-Audit

**Re-extraction rule** (identical to Wf-264 §1): Re-parse every numeric token (AUC, RMSE, ratio, time, %, n, std, ±, Δ, etc.) in `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` post-fix and re-classify into:
- **Reasonable** = claim has traceable MEASURED source (JSON, MD, or log)
- **Aspirational** = claim has NO MEASURED source (G1 target = 0)
- **Projected claimed as MEASURED** = Wf-258 §3.5 honest-framing violation — source path is wrong, doesn't exist, has different value, or is a different experiment (G2 target = 0)
- **Seed-count error** = per-metal n contradicts cross-metal n sum (G3 target = 0)
- **Partial disclosure** = claim has data but disclosure omits key context (timeout status, etc.) (G4 target = 0)

**Comparison method**: For each of the 5 Wf-264 §3.5 violations (Issues 1-5 in `molmetal/reports/wf264_audit/paper_4_claim_audit.md` §3.1-§3.5), verify the fix was correctly applied + scan the rest of the paper for NEW or UNFIXED violations of the same root causes.

---

## §2 Re-audit trace — Wf-264 §3.5 violations → post-fix status

### §2.1 Wf-264 §3.1 Issue #1 — §9 line 410 chemprop_v2_5metal_5seed_50epoch.json mis-citation

**Pre-fix (§9 line 410)**: `molmetal/metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json` — TODO-154 chemprop v2 5-metal in-domain fold AUCs (cited as a SOURCE for TODO-154 numbers).

**Actual content of cited JSON** (verified at `molmetal/metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json`):
- `metric` = "chemprop_v2_5metal_5seed_50epoch"
- `description` = "Per-(metal, seed) chemprop v2 LOMO test_auc ... **WF-146 sweep was launched on 2026-09-27 16:41:23 CST, killed mid-Rh_seed1** ... hidden_dim=64 depth=2."
- `value` = **null** (in-progress Wf-146 LOMO sweep)
- `per_metal_seed_auc`: only `Rh_seed0` has data; 24/25 slots `PENDING`

**Post-fix (§9 line 410)**:
> `TODO/completed/154_chemprop_v4_head_to_head.md` §2 — TODO-154 chemprop v2 5-metal in-domain fold AUCs (Ru 0.9095 / Ir 0.8846 / Os 0.8890 / Rh 0.8631 / Re 0.8822; 5-metal mean 0.8857 ± 0.0140; n=14 paired runs) — per-metal paired t-test vs V4 p-values: Ru=0.025, Ir=0.010, Os=0.144 (underpowered n=3), Rh=0.027, Re=0.421 (underpowered n=2); cross-metal paired t-test t=6.754, p=1.35×10⁻⁵ — **(NOT `chemprop_v2_5metal_5seed_50epoch.json` which is Wf-146 LOMO sweep, value=null, 24/25 PENDING; per Wf-264 §3.1 audit fix 2026-09-28)**

**Verdict**: ✅ **FIXED**. The §9 line 410 now correctly cites `TODO/completed/154_chemprop_v4_head_to_head.md` §2 as the primary source and includes the explicit "(NOT `chemprop_v2_5metal_5seed_50epoch.json`...)" disclaimer that points the reader to the actual content of the cited JSON.

---

### §2.2 Wf-264 §3.2 Issue #2 — §9 line 411 cfm_decode_real.json mis-citation

**Pre-fix (§9 line 411)**: `molmetal/metrics/by_metric/cfm_decode_real.json` — m14 3-seed decode_ratio 0.9980 (cross-check).

**Actual content of cited JSON** (verified at `molmetal/metrics/by_metric/cfm_decode_real.json`):
- `value` = **0.9645476772616137** (NOT 0.9980)
- `n_total_platinai` = 5,000 / `n_pt_subset` = 874 / `n_sampled` = 818
- `aggregate_n_ok` = 789 / `aggregate_n_total` = 818
- `wall_time_s` = 131.3 (CPU, pocket 1h36, 1 seed)
- `data_config` = "platinai_5k_pt" (PlatinAI only, NOT NCI-60 + DrugOOD native)

**Post-fix (§9 line 411)**:
> `molmetal/reports/wf139_p5_cfm_decode_real/synthesis.md` + `molmetal/reports/wf148_m14_real_cfm.log` — m14 3-seed decode_ratio 0.9980 cross-check (5979/5991 ok; NCI-60 9-cell + DrugOOD native; 3 seeds [0,1,2]; Wf-188 m14 FULL MetalCytoToxDB decode_ratio 0.9958 on n=26,801 supersedes per §4.1) — **(NOT `molmetal/metrics/by_metric/cfm_decode_real.json` which is a separate smaller measurement value=0.9645 on n=818 PlatinAI pt_subset, 1 seed CPU; per Wf-264 §3.2 audit fix 2026-09-28)**

**Verdict**: ✅ **FIXED**. The §9 line 411 now correctly cites `wf139_p5_cfm_decode_real/synthesis.md` + `wf148_m14_real_cfm.log` for the 0.9980 cross-check and includes the explicit "(NOT `cfm_decode_real.json`...)" disclaimer with the actual content of the mis-cited JSON.

---

### §2.3 Wf-264 §3.3 Issue #3 — §0.1 line 19 Δ −0.0098 transcription error

**Pre-fix (§0.1 line 19)**: `Δ chemprop v2 − iMoLD = −0.0098 / −0.0177 / −0.0035` (mixed 3-seed assay + 5-seed scaffold/size — internally inconsistent with §3.1's 5-seed consistent set).

**Post-fix (§0.1 line 19)**:
> Δ chemprop v2 − iMoLD = −0.0074 / −0.0177 / −0.0035 — **5-seed consistent set per §3.1**; the 3-seed assay Δ=−0.0098 is reported only in §3.2 per-seed detail, NOT in this headline

**Verdict**: ✅ **FIXED**. The §0.1 line 19 now uses the consistent 5-seed set (−0.0074 / −0.0177 / −0.0035) and explicitly notes that the 3-seed −0.0098 figure is reported only in §3.2 per-seed detail.

---

### §2.4 Wf-264 §3.4 Issue #4 — §2.1 line 98 + §2.3 line 119 seed-count inflation (Ru/Ir 5-seed contradicts cross-metal n=14)

**Pre-fix (§2.1 line 98)**: `Ru | 0.909 (5-seed mean, paired vs V4 p=0.025)`.

**Post-fix (§2.1 line 98)**:
> **Ru** | **0.909** (3-seed mean, paired vs V4 p=0.025; n=3 paired runs) | 0.81 (time-split) | **+0.099** | 0.810 ± 0.017 (3-seed, scaffold-aligned; ⚠️ PARTIAL — Ru 3/3 per-seed runs tagged "timeout" in Wf-219 sweep_summary.json; per_metal_summary.Ru.mean_auc=NaN, n_seeds=0; per-seed AUCs (0.7858 / 0.8215 / 0.8215) computed from partial runs) | ✅ STRONG (Krasnov-beating; Wf-219 reproduces at 0.810 within scaffold-split protocol but partial timeout disclosed)

**Post-fix (§2.3 lines 118-122)**:
| Ru only | 3 | — | **0.025** | ✅ Significant (n=3 paired runs) |
| Ir only | 3 | — | **0.010** | ✅ Significant (n=3 paired runs) |
| Os only | 3 | — | 0.144 | ❌ Underpowered (n=3) |
| Rh only | 3 | — | **0.027** | ✅ Significant |
| Re only | 2 | — | 0.421 | ❌ Underpowered (n=2) |

**Verdict**: ✅ **FIXED**. §2.1 line 98 now correctly shows Ru "(3-seed mean...; n=3 paired runs)"; §2.3 lines 118/119/122 now show Ru=3 / Ir=3 / Re=2 (summing to n=14 = 3+3+3+3+2 consistent with cross-metal paired t-test). The 5-metal mean row (§2.1 line 103) explicitly states "(n=14 paired runs across 5 metals: Ru/Ir/Os/Rh 3 seeds + Re 2 seeds = 3+3+3+3+2)".

---

### §2.5 Wf-264 §3.5 Issue #5 — §2.1 + §5.1 + §6.1 + §7.1 partial-disclosure (Wf-219 Ru 0.810 ± 0.017 timeout status not disclosed)

**Pre-fix**: §2.1 + §5.1 + §6.1 + §7.1 L11 listed Wf-219 Ru 0.810 ± 0.017 as "✅ MEASURED" without disclosing that all 3 Ru runs were tagged "timeout" in Wf-219 sweep_summary.json.

**Post-fix**:

**§2.1 line 98**: `0.810 ± 0.017 (3-seed, scaffold-aligned; ⚠️ PARTIAL — Ru 3/3 per-seed runs tagged "timeout" in Wf-219 sweep_summary.json; per_metal_summary.Ru.mean_auc=NaN, n_seeds=0; per-seed AUCs (0.7858 / 0.8215 / 0.8215) computed from partial runs)`

**§5.1 line 259**: `⚠️ PARTIAL (Ru 0.810 ± 0.017 — Ru 3/3 per-seed runs tagged "timeout" in Wf-219 sweep_summary.json; per_metal_summary.Ru.mean_auc=NaN, n_seeds=0; per-seed AUCs 0.7858 / 0.8215 / 0.8215 computed from partial runs) / ✅ MEASURED (Ir 0.759 ± 0.014 — Ir 3/3 OK)`

**§6.1 line 312**: `(⚠️ PARTIAL — Ru 3/3 per-seed runs tagged "timeout" in Wf-219 sweep_summary.json; per-seed AUCs computed from partial runs; mean_auc=NaN, n_seeds=0)`

**§7.1 L11 line 364**: `Additional honest framing: Ru per-seed runs in Wf-219 all 3/3 tagged status="timeout" per wf219_krasnov_aligned/sweep_summary.json; per-seed AUCs 0.7858 / 0.8215 / 0.8215 (mean 0.810) computed from partial runs; per_metal_summary.Ru.mean_auc=NaN, n_seeds=0 because no run completed the full 50-epoch budget. Ir runs were 3/3 OK.`

**Verdict**: ✅ **FIXED**. The Wf-219 Ru 0.810 ± 0.017 timeout status is now disclosed at 4 locations (§2.1 + §5.1 + §6.1 + §7.1 L11) with explicit "⚠️ PARTIAL" labels and full disclosure of the partial-run origin, NaN mean_auc, and per-seed AUC provenance.

---

## §3 NEW findings — Wf-267-Audit residuals (NOT covered by Wf-264-Audit)

### §3.1 NEW Issue #6 — §2.2 line 110 mis-cites `chemprop_v2_5metal_5seed_50epoch.json` as TODO-154 source (RESIDUAL of Wf-264 §3.1)

**Claim (§2.2 line 110)**:
> **TODO-154**: chemprop v2 in-domain 80/20 stratified holdout, hidden=300/depth=3, Adam lr=1e-3, 50 epoch, 5-seed (Ru/Ir/Os/Rh/Re), per-metal BinaryClassificationFFN head. **Source**: `TODO/completed/154_chemprop_v4_head_to_head.md` + `molmetal/metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json` (updated per-metal summary).

**Actual content of cited JSON** (verified at `molmetal/metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json`):
- `metric` = "chemprop_v2_5metal_5seed_50epoch"
- `description` = "Per-(metal, seed) chemprop v2 LOMO test_auc ... **WF-146 sweep was launched on 2026-09-27 16:41:23 CST, killed mid-Rh_seed1** ... hidden_dim=64 depth=2."
- `value` = **null** (in-progress Wf-146 LOMO sweep; 24/25 PENDING)
- Different experimental config (LOMO, hidden=64/depth=2) than TODO-154 (in-domain, hidden=300/depth=3)

**Verdict**: ⚠️ **PROJECTED claimed as MEASURED (RESIDUAL)**. The §2.2 line 110 STILL cites `chemprop_v2_5metal_5seed_50epoch.json` as a Source for TODO-154 numbers — exactly the same mis-citation that was fixed at §9 line 410 (which now has the explicit "(NOT...)" disclaimer). The §2.2 line 110 has NO "(NOT...)" disclaimer applied.

**Recommended fix**: §2.2 line 110 should read: "**TODO-154**: chemprop v2 in-domain 80/20 stratified holdout, hidden=300/depth=3, Adam lr=1e-3, 50 epoch, 5-metal × 3-seed split (Ru/Ir/Os/Rh 3 seeds + Re 2 seeds = 14 paired runs), per-metal BinaryClassificationFFN head. **Source**: `TODO/completed/154_chemprop_v4_head_to_head.md` (NOT `chemprop_v2_5metal_5seed_50epoch.json` which is Wf-146 LOMO sweep, value=null, 24/25 PENDING; per Wf-264 §3.1 audit fix 2026-09-28)."

**Severity**: Same root cause as Wf-264 §3.1 Issue #1, but the Wf-264 audit only flagged §9 line 410. §2.2 line 110 was missed because it's in a different section (protocol card §2.2, not citation appendix §9).

---

### §3.2 NEW Issue #7 — §8 line 388 lists `cfm_decode_real.json` as valid source category for ALL numbers (RESIDUAL of Wf-264 §3.2)

**Claim (§8 line 388, G2 acceptance gate)**:
> **[G2]** REAL protocol-aligned numbers throughout ✅ — all numbers cite `molmetal/reports/wf{174,179,185,188,219}_*/` JSON or **`molmetal/metrics/by_metric/{cfm_decode_real,nci60_precious_9cell_auc,diversity_tanimoto}.json`** or `TODO/completed/154_chemprop_v4_head_to_head.md` or `TODO/completed/180_daily_summary_2026_09_27.md` §1.3

**Actual content of `cfm_decode_real.json`** (verified at `molmetal/metrics/by_metric/cfm_decode_real.json`):
- `value` = **0.9645476772616137** (NOT 0.9980)
- `n_total_platinai` = 5,000 / `n_pt_subset` = 874 / `n_sampled` = 818
- `data_config` = "platinai_5k_pt" (PlatinAI only, 1 seed, CPU)
- Different experiment than the m14 0.9980 / 0.9958 numbers cited in paper §4

**Verdict**: ⚠️ **PROJECTED claimed as MEASURED (RESIDUAL)**. The §8 line 388 lists `cfm_decode_real.json` as one of the valid source categories for "ALL numbers" in the paper — but this JSON's value=0.9645 (PlatinAI 1-seed, n=818) does NOT match any number cited in paper §4. The §9 line 411 has the correct "(NOT...)" disclaimer for this JSON, but §8 line 388 still treats it as a valid source.

**Recommended fix**: §8 line 388 should read: "all numbers cite `molmetal/reports/wf{174,179,185,188,219}_*/` JSON or `molmetal/metrics/by_metric/{nci60_precious_9cell_auc,diversity_tanimoto}.json` or `TODO/completed/154_chemprop_v4_head_to_head.md` or `TODO/completed/180_daily_summary_2026_09_27.md` §1.3 (NOT `molmetal/metrics/by_metric/cfm_decode_real.json` which is a separate smaller measurement value=0.9645 on n=818 PlatinAI pt_subset, 1 seed CPU; per Wf-264 §3.2 audit fix 2026-09-28)."

**Severity**: Same root cause as Wf-264 §3.2 Issue #2, but the Wf-264 audit only flagged §9 line 411. §8 line 388 was missed because it's in the acceptance-gates section §8, not the citation appendix §9.

---

## §4 Wf-258 §3.5 honest-framing post-fix summary

**Honest-framing scorecard post-Wf-264-fixes**:

| Category | Pre-Wf-264 (audit date 2026-09-28) | Post-Wf-267 (this audit) | Δ |
|---|---:|---:|---:|
| **Aspirational claims (G1)** | 0 | 0 | — |
| **Projected-as-measured violations (G2)** | 5 | **2** | -3 fixed, 2 residuals (NEW findings §3.1 + §3.2) |
| **Seed-count errors (G3)** | 1 | 0 | -1 fixed |
| **Partial-disclosure errors (G4)** | 1 | 0 | -1 fixed |
| **Total §3.5 violations** | 5 | **2** | -3 net |

**All §0/§1/§2/§3/§4/§5/§6/§7 main-result numbers** (chemprop v2 0.886 mean, Wf-188 m14 0.9958, Wf-179 m15 5/5 < 2.0 Å, m21 0.8007, m7 0.0397, Wf-174/185 DrugOOD 0.7113/0.7137, Wf-219 Ru 0.810 / Ir 0.759, TODO-154 14 paired runs) **remain MEASURED with traceable JSON/MD sources**. None of the §3.5 violations affect the main-result numerical values; both residuals (§2.2 line 110 + §8 line 388) are citation-hygiene issues for non-existent or mis-attributed JSON source pointers.

---

## §5 Verdict + recommendation

**G5_n_total_violations_post = 2** (down from Wf-264 baseline of 5; 3 violations fully fixed at §0.1 line 19 + §2.1 line 98 + §2.3 lines 118-119 + §5.1 line 259 + §6.1 line 312 + §7.1 L11 line 364 + §9 lines 410 + §9 lines 411; 2 NEW residuals discovered at §2.2 line 110 + §8 line 388).

**Severity assessment of residuals**:
- §2.2 line 110 + §8 line 388 are **citation-hygiene** issues only — they do not affect any §0/§2/§3/§4/§5/§6 main-result numerical value.
- Both residuals are **2-line edits** to add "(NOT `chemprop_v2_5metal_5seed_50epoch.json`...)" / "(NOT `cfm_decode_real.json`...)" disclaimers following the Wf-264 fix template already applied at §9 lines 410 + 411.
- These residuals are **NOT BLOCKERS** for Wf-227 final-draft sign-off but should be cleaned up for Wf-228 / Wf-230 paper-grade ship surface.

**Recommendation**: Apply 2 additional minor citation-hygiene edits (§2.2 line 110 + §8 line 388) following the Wf-264 "(NOT...)" disclaimer template. After these 2 edits, G5_n_total_violations_post = 0 and paper §4 §3.5 honest framing is fully clean.

---

## §6 Honest framing note (per `party-twenty-spirit` §3.5)

Paper §4 at `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` has materially improved post-Wf-264 fixes. All 5 originally-identified §3.5 violations are now corrected:
- §0.1 line 19: Δ chemprop v2 − iMoLD uses consistent 5-seed set (−0.0074 / −0.0177 / −0.0035); 3-seed −0.0098 moved to §3.2 only
- §2.1 line 98: Ru + Ir labeled "(3-seed mean...; n=3 paired runs)"; Wf-219 Ru 0.810 ± 0.017 carries "⚠️ PARTIAL" timeout disclosure
- §2.3 lines 118-122: per-metal n columns show Ru=3 / Ir=3 / Re=2 (summing to cross-metal n=14)
- §5.1 line 259: Wf-219 row carries "⚠️ PARTIAL" timeout disclosure for Ru + "✅ MEASURED" for Ir
- §6.1 line 312: m1 Ru AUC ledger row carries "⚠️ PARTIAL" timeout disclosure
- §7.1 L11 line 364: Wf-219 limitation carries "Additional honest framing: Ru per-seed runs all 3/3 tagged timeout; per_seed AUCs 0.7858/0.8215/0.8215 computed from partial runs; mean_auc=NaN, n_seeds=0"
- §9 line 410: TODO-154 source pointer fixed with "(NOT `chemprop_v2_5metal_5seed_50epoch.json`...)" disclaimer
- §9 line 411: m14 0.9980 source pointer fixed with "(NOT `cfm_decode_real.json`...)" disclaimer

**2 NEW §3.5 residuals discovered** at §2.2 line 110 + §8 line 388 (both same root causes as Wf-264 §3.1 + §3.2 but in different sections that the Wf-264 audit did not cover). Neither affects main-result numerical values; both are 2-line citation-hygiene edits.

**Paper §4 is in strong shape for Wf-227 final-draft sign-off with 2 minor follow-up edits** (G2 residuals at §2.2 line 110 + §8 line 388). After these 2 edits, paper §4 will be 100% clean on §3.5 honest framing.

---

*Generated 2026-09-28 by Wf-267-Audit ultracode. READ-ONLY audit (no edits to paper §4 or source files). All numbers in this audit trace to the same 7 source families cited by paper §4 + direct verification of the 2 mis-cited JSON files at `molmetal/metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json` + `molmetal/metrics/by_metric/cfm_decode_real.json`.*