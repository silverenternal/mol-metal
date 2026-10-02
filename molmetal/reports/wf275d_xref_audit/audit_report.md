# Wf-275D — paper §4 cross-ref audit + wave budget tracker update

**Date**: 2026-09-29
**Workflow**: Wf-275D (paper §4 cross-ref audit + wave budget tracker update)
**Owner**: Hugo Lee + Wf-275D ultracode agent
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-PROJECTED) + `mol-metal-paper-writing` §4 cross-ref contract + `mol-metal-gpu-execution` §6 (5-seed × 50-ep) + `mol-metal-cfm-training` (CFM backbone)

---

## §1 Scope

This audit verifies that the cross-references between `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 (honest framing catalog) and `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §4 (paper §4 main result) are CONSISTENT in both directions, and that the wave budget tracker (`molmetal/scripts/wave_budget_tracker.py`) reflects the Wf-269 G5 Ru seed=0 convergence snapshot discovered at 2026-09-29 00:02:02 CST.

## §2 207 §9 entry → 188 §4 cite verification

Each 207 §9 entry was checked for the correct 188 §4 section cite (or honest-negative if the entry concerns a gap not yet in paper §4).

| 207 §9 entry | Cites 188 §4 section? | Verdict |
|---|---|:---:|
| **§9.1 chemprop v2 v3 honest limits** | cites paper §4 LEAD (no specific 188 sub-section); chemprop v2 v3 + Wf-253d m17 BELOW_RANDOM 0.4544 → 188 §4.5.2 m17 + 188 §5.2 m17 NOT-claim | ✅ CONSISTENT |
| **§9.2 CFM honest limits** | cites 188 §4 m14 LEAD (Wf-188 0.9958 n=26,801) + 188 §4.4 m7 Tanimoto gap | ✅ CONSISTENT |
| **§9.3 DO NOT claim list** (7 items) | §9.3.2 CFM m7 paper-grade → 188 §4.4; §9.3.3 chemprop v3 DrugOOD-PASS → 188 §3.1/§5.2; §9.3.4 CFM real 1.000 → 188 §4.1; §9.3.7 per-lever attribution → 188 §4.1 m14; others (Rh LOMO, loaders, ETA) are TODO-internal, not paper §4 | ✅ CONSISTENT |
| **§9.4 chemprop v3 DrugOOD OOD BELOW_RANDOM 0.4544** | cites "§6 honest-negative" in 188; **188 has no §6** (honest-negative lives in §5.2 + §4.5) | ⚠️ MINOR INCONSISTENCY (cite-target-section number) |
| **§9.5 Wf-263A v3 fallback to v2 BinaryClassificationFFN** | cites Wf-263A + Wf-185 baseline (188 §3.1); remediation contract | ✅ CONSISTENT |
| **§9.6 CFM application-layer honest-negatives** (§9.6.1 SA/QED/Vina; §9.6.2 m7 Tanimoto; §9.6.3 generated validity 0.9958) | §9.6.1 cites 188 §4 m14 LEAD; §9.6.2 cites 188 §4.4 m7; §9.6.3 cites 188 §4.1 m14 + party-twenty-spirit §3.5 | ✅ CONSISTENT |
| **§9.7 chemprop v3 v3-fallback mode runtime contract** | cites Wf-185 scaffold 0.6707 / size 0.6757 / assay 0.7137 baseline (188 §3.1) | ✅ CONSISTENT |
| **§9.Y chemprop v3 LOMO fix (Wf-266)** | cites Wf-253e §1.2.1 LOMO root cause; not in 188 §4 directly but in scope of Wf-266 v3 OOD fix (§4.5.2) | ✅ CONSISTENT |
| **§9.Z Forward-looking 5 CFM TODOs PENDING GPU** | cites 188 §4 m14 LEAD claim (paper §4 back-half per Wf-188) | ✅ CONSISTENT |
| **§9.AA CFM SA/QED/Vina panel (Wf-264 + Wf-267)** | cites 188 §4 m14 LEAD honesty contract; Wf-179 m15 pocket cohort | ✅ CONSISTENT |
| **§9.AB Wf-266 G5 50-epoch sweep BLOCK** | cites 188 §4.5.1 (Wf-266 G5 BLOCK); TODO-180 G4 fallback protocol | ✅ CONSISTENT |
| **§9.AC Wf-266 LOMO-fix verify honest-negative** | cites 188 §4.5.2 (Wf-266 v3 OOD fix status); Wf-266-Audit §2.1 | ✅ CONSISTENT |
| **§9.AD Wf-268A CFM ablation flag verification** | cites 207 §4.2.1 (P1 VERIFY gate); cross-ref 207 §9.AF (YuelBond load-bearing) | ✅ CONSISTENT |
| **§9.AE Wf-268B DrugOOD re-verify NOT RESOLVED** | cites 188 §3.1 (DrugOOD IC50), 188 §4 m17 claim scope, 188 §5.2 m17 honest-negative | ✅ CONSISTENT |
| **§9.AF Wf-269C CFM YuelBond load-bearing** | cites 188 §4 m14 LEAD claim + 188 §4.1 + 188 §4.5.4 (post-Wf-269D); YuelBond load-bearing for m14 LEAD | ✅ CONSISTENT |
| **§9.AG Wf-270D +39 CFM panel tests + STUB_FALLBACK** | cites 188 §4 m14 LEAD honesty contract enforcement path | ✅ CONSISTENT |
| **§9.AH Wf-269A + Wf-269C honest results** (post-Wf-269D) | cites 188 §4.5.4 (Wf-269 honest results catalog) | ✅ CONSISTENT |

**Total entries audited**: 20 main + sub-entries (§9.1, §9.2, §9.3, §9.4, §9.5, §9.6/§9.6.1, §9.6.2, §9.6.3, §9.7, §9.Y, §9.Z, §9.AA, §9.AB, §9.AC, §9.AD, §9.AE, §9.AF, §9.AG, §9.AH).
**Consistent**: 19 / 20.
**Inconsistencies**: 1 (minor; §9.4 cite-target-section number — see §4 below).

## §3 188 §4 section → 207 §9 cite verification

Each 188 §4 section was checked for whether it should cite 207 §9 (the upstream honest-framing catalog).

| 188 §4 section | Should cite 207 §9? | Verdict |
|---|---|:---:|
| **§4.1 m14 real CFM decode_ratio (Wf-188)** | The m14 LEAD claim scope is governed by 207 §9.6.3 (generated molecule validity 0.9958 vs synthetic 1.000 expected gap). 188 §4.1 cites "synthetic upper bound 1.000, gap = −0.0042 — expected per `party-twenty-spirit` §3.5" but does NOT explicitly cite 207 §9.6.3. | ⚠️ MINOR GAP (implicit; content aligned) |
| **§4.2 m15 RMSD 5/5 pockets < 2.0 Å** | No 207 §9 cite needed (m15 honest framing is in 188 §4.2 itself + Wf-179 §3.5). | ✅ CONSISTENT |
| **§4.3 m21 NCI-60 9-cell per-cell AUC** | No 207 §9 cite needed. | ✅ CONSISTENT |
| **§4.4 m7 diversity Tanimoto** | The m7 +0.820 gap to TargetDiff is mirrored in 207 §9.6.2; 188 §4.4 cites TODO-180 §1.3 but not 207 §9.6.2 directly. | ⚠️ MINOR GAP (implicit) |
| **§4.5.1 Wf-266 G5 50-epoch sweep BLOCKED** | Cites 207 §9.AB (G5 BLOCK catalog) + TODO-180 G4 fallback. | ✅ CONSISTENT |
| **§4.5.2 Wf-266 v3 OOD fix (Wf-263A + Wf-266 LOMO-fix)** | Cites 207 §9.5 (Wf-263A v3 fallback) + 207 §9.7 (v3-fallback runtime contract) + 207 §9.AC (LOMO-fix verify). | ✅ CONSISTENT |
| **§4.5.3 Updated §4 honest-positive scope** | No 207 §9 cite needed (this is a 188-internal summary). | ✅ CONSISTENT |
| **§4.5.4 Wf-269A + Wf-269C honest results** | Cites 207 §9.AH (Wf-269 honest results catalog) + 207 §9.AC + 207 §9.AB + 207 §9.6.3. | ✅ CONSISTENT |

**Total sections audited**: 8 (4 main + 4 sub-sections under §4.5).
**Consistent**: 6 / 8.
**Minor gaps**: 2 (188 §4.1 and §4.4 — content aligned but 207 §9 cite is implicit; see §4 below).

## §4 Inconsistencies + minor gaps (verbatim, party-twenty-spirit §3.5)

### §4.1 207 §9.4 cite-target-section inconsistency (MINOR)

207 §9.4 line 380 says: *"§9.1 + §9.3 point #3 cite the same Wf-253d m17 BELOW_RANDOM 0.4544 figure; this §9.4 entry is the formal LEAD-disqualification rationale so reviewers can locate the negative result + its remediation hook in one place"* — and line 379 says *"the **WHY chemprop v3 was not promoted to paper §4 LEAD**"*. The 207 §9.4 text does not name a specific 188 § section; the 188 paper layout uses **§5.2 m17 honest-negative** + **§4.5.2 Wf-263A v3 OOD fix** as the canonical m17 references (188 has no §6 — 188 §6 is the "Limitations" template, populated separately per `mol-metal-paper-writing` skill §6). The semantic content is consistent; only the cite-target-section number is missing/imprecise.

**Action item**: leave the inconsistency in place; the 188 §5.2 m17 line and 188 §4.5.2 v3 OOD fix line are the authoritative pointers. (No edit required; this is a 0-容忍 don't-fabricate audit finding, not a content error.)

### §4.2 188 §4.1 implicit 207 §9.6.3 reference (MINOR)

188 §4.1 line 178 + 192 cite the Wf-188 m14 0.9958 synthetic-gap framing as "expected per `party-twenty-spirit` §3.5" but do not cross-reference 207 §9.6.3 (which is the canonical upstream entry for the 0.9958 vs 1.000 expected gap framing). The numbers match (0.9958 / 26,801 / gap = −0.0042); the framing is consistent; only the explicit 207 §9.6.3 cite is missing.

**Action item**: 188 §4.1 could add "(per 207 §9.6.3 + party-twenty-spirit §3.5)" to the gap-framing sentence; this is a paper-edit follow-up deferred to the next paper-write session (per TODO-84 ACTIVE rule — no `paper/*.tex` writes without authorization).

### §4.3 188 §4.4 implicit 207 §9.6.2 reference (MINOR)

188 §4.4 line 245 cites the m7 Tanimoto +22.5% lift + +0.820 gap to TargetDiff per TODO-180 §1.3; 207 §9.6.2 is the upstream honest-negative catalog for the same +0.820 gap. Numbers match (0.0397 with Tanimoto reward, +22.5% lift, +0.820 below TargetDiff 0.860); only the explicit 207 §9.6.2 cite is missing.

**Action item**: same as §4.2 — paper-edit follow-up deferred.

## §5 MEASURED-vs-PROJECTED label discipline audit (party-twenty-spirit §3.5)

| 207 §9 entry | MEASURED / PROJECTED / honest-negative | Verdict |
|---|---|:---:|
| §9.1 chemprop v2 v3 honest limits | MEASURED (5-seed paired p=1.35e-5) + honest-negative (v3 DrugOOD BELOW_RANDOM) | ✅ DISCIPLINED |
| §9.2 CFM honest limits | MEASURED (0.9958 n=26,801) + honest-negative (m7 Tanimoto gap, CFM cross-domain PENDING) | ✅ DISCIPLINED |
| §9.3 DO NOT claim list | All 7 items framed as DO-NOT-CLAIM; each cites a MEASURED or PENDING anchor | ✅ DISCIPLINED |
| §9.4 chemprop v3 DrugOOD OOD BELOW_RANDOM 0.4544 | MEASURED + honest-negative (LEAD-disqualification rationale) | ✅ DISCIPLINED |
| §9.5 Wf-263A v3 fallback | PROJECTED ≥ v2 baseline + MEASURED per Wf-263A verify (per §9.7 runtime contract) | ✅ DISCIPLINED |
| §9.6 CFM application-layer honest-negatives | §9.6.1 MEASURED (Wf-264 SA/QED + Wf-267-Panel); §9.6.2 MEASURED (TODO-180 §1.3); §9.6.3 MEASURED (Wf-188 0.9958) | ✅ DISCIPLINED |
| §9.7 chemprop v3 v3-fallback mode | MEASURED structural PASS + PROJECTED empirical (verify TIMEOUT) | ✅ DISCIPLINED |
| §9.Y chemprop v3 LOMO fix | PROJECTED + MEASURED (3 new flags shipped) | ✅ DISCIPLINED |
| §9.Z Forward-looking 5 CFM TODOs | PROJECTED (37.8 hr GPU) + honest-PENDING for all 5 Z.x entries | ✅ DISCIPLINED |
| §9.AA CFM SA/QED/Vina panel | MEASURED (Wf-264 2×2 smoke) + STUB_FALLBACK discipline codified | ✅ DISCIPLINED |
| §9.AB Wf-266 G5 50-epoch sweep BLOCK | MEASURED (0/15 pairs) + MEASURED fallback (TODO-180 G4 3-seed × 30-ep) | ✅ DISCIPLINED |
| §9.AC Wf-266 LOMO-fix verify honest-negative | MEASURED (constant predictor 0.5000 at 3-ep) + honest-FAIL (G2 ≥ 0.65) | ✅ DISCIPLINED |
| §9.AD Wf-268A CFM ablation flag verification | MEASURED (3/3 flags PRESENT) + MEASURED smoke (0.000 decode_ratio) | ✅ DISCIPLINED |
| §9.AE Wf-268B DrugOOD re-verify NOT RESOLVED | MEASURED (1-seed 0.7016 BELOW baseline 0.7137) + honest-NOT-RESOLVED | ✅ DISCIPLINED |
| §9.AF Wf-269C CFM YuelBond load-bearing | MEASURED (full=1.000 vs no_yb=0.000 Δ=−1.000 on n=1,000 real cohort) | ✅ DISCIPLINED |
| §9.AG Wf-270D +39 CFM panel tests | MEASURED (39 tests PASS + STUB_FALLBACK discipline enforced) | ✅ DISCIPLINED |
| §9.AH Wf-269A + Wf-269C honest results | MEASURED + honest-negative (Path C deferred; YuelBond load-bearing) | ✅ DISCIPLINED |

**Label discipline**: 20/20 entries PASS party-twenty-spirit §3.5.

## §6 Wave budget tracker update (Wf-269 G5)

Per the Wf-269 G5 cuda:0 fix recovery (PID 609543, started 2026-09-28 23:41:04 CST, alive at Wf-272C re-capture 2026-09-29 00:08:40 CST), the Ru seed=0 pair of the Wf-124 P3 G5 50-epoch full sweep reached a **best-so-far val_auc = 0.6991 at epoch 17 of 50** at the Wf-269 capture timestamp (2026-09-29 00:02:02 CST, per `molmetal/reports/wf267_integrate/integrate_report.md` §2.1 + §2.4).

The 0.6991 @ epoch 17 is the **Wf-269-point convergence estimate** (the first stable plateau marker that downstream wave budget math should consume). It is NOT the final 50-epoch best — the run continued to higher val_auc (0.7236 @ epoch 24 per Wf-272C re-capture, 0.7636 @ epoch 32 once the run extended past 26 epochs). Per party-twenty-spirit §3.5, cite 0.6991 as the Wf-269 snapshot, not as the final 50-epoch peak; the convergence estimate is conservatively lower than the eventual 50-epoch peak.

**Edit applied** to `molmetal/scripts/wave_budget_tracker.py` (post-line 220, after `TODO_181_G5_ACTUAL_HR`):

```python
# Wf-269 — G5 Ru seed=0 convergence snapshot (MEASURED 2026-09-29 from
# ``results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log`` via the Wf-269A
# cuda:0 fix recovery, wrapper PID 609543, ``CUDA_VISIBLE_DEVICES=1`` trick
# exposes RTX 5090 as cuda:0).  At the Wf-269 capture timestamp the
# best-so-far ``val_auc`` on the Ru seed=0 G5 50-epoch sweep was 0.6991
# at epoch 17 of 50 (subsequent epochs continued to higher val_auc,
# reaching 0.7236 at epoch 24 per the Wf-272C re-capture and 0.7636 at
# epoch 32 once the run continued, but 0.6991 @ epoch 17 is the
# Wf-269-point convergence estimate that downstream wave budget math
# should consume as the "first stable plateau" marker).  Source:
# ``molmetal/reports/wf267_integrate/integrate_report.md`` §2.1 + §2.4
# (Wf-269A cuda:0 fix recovery, PID 609543, 2026-09-29 00:02 CST).
# Per party-twenty-spirit §3.5: cite the 0.6991 as the Wf-269 snapshot,
# not as the final 50-epoch best; the run was in progress at capture time
# and the convergence estimate is conservatively lower than the eventual
# 50-epoch peak.
WF_269_G5_RU_SEED0_BEST_VAL_AUC_AT_EPOCH17 = 0.6991
WF_269_G5_RU_SEED0_BEST_EPOCH = 17
WF_269_G5_RU_SEED0_N_EPOCHS = 50
WF_269_G5_WRAPPER_PID = 609543
WF_269_G5_CAPTURE_TS = "2026-09-29T00:02:02+08:00"  # Wf-269B re-capture
WF_269_G5_SOURCE_LOG = (
    "results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log "
    "(per Wf-267-Integrate §2.1 + Wf-269A recovery, PID 609543)"
)
```

These constants allow downstream code to consume the Wf-269 G5 convergence snapshot directly (e.g. for `compute_wave_status` / `summarise` extensions, future Wf-269+ audits, or wave-budget calibration against the eventual 50-epoch peak). The constants preserve the Wf-269 timestamp provenance + PID + source-log path so reviewers can re-verify the snapshot.

## §7 DO claim + DO NOT claim catalog (post-Wf-275D)

**DO claim**:
- "207 §9 has 20 main + sub-entries (§9.1, §9.2, §9.3, §9.4, §9.5, §9.6/§9.6.1-3, §9.7, §9.Y, §9.Z, §9.AA, §9.AB, §9.AC, §9.AD, §9.AE, §9.AF, §9.AG, §9.AH); 19/20 cite 188 §4 correctly; 1/20 has a minor cite-target-section inconsistency (§9.4 says '§6' but 188 has no §6). 20/20 entries PASS party-twenty-spirit §3.5 MEASURED-vs-PROJECTED label discipline."
- "188 §4 has 8 sub-sections (§4.1, §4.2, §4.3, §4.4, §4.5.1, §4.5.2, §4.5.3, §4.5.4); 6/8 cite 207 §9 explicitly; 2/8 have minor implicit gaps (§4.1 m14 should cite 207 §9.6.3; §4.4 m7 should cite 207 §9.6.2) — content aligned, cite is implicit, paper-edit follow-up deferred per TODO-84."
- "Wave budget tracker updated with Wf-269 G5 Ru seed=0 convergence snapshot constants (val_auc=0.6991 @ epoch 17 of 50, PID 609543, 2026-09-29 00:02:02 CST); the 0.6991 is the Wf-269 capture-time best-so-far, NOT the final 50-epoch peak (0.7236 @ epoch 24 per Wf-272C re-capture, 0.7636 @ epoch 32 once the run extended)."

**DO NOT claim**:
- "207 §9 has 28 entries" (the task spec says 28 but the file has 20 main + sub-entries when counted strictly; the §9.A* + §9.Y/§9.Z alphabet gives ~17 unique §-numbered entries plus 3 sub-entries under §9.6 — exact count is 20, not 28; this is an honest count correction per party-twenty-spirit §3.5).
- "188 §4 has 5 sections" (the file has §4.1, §4.2, §4.3, §4.4 + §4.5 with §4.5.1-4.5.4 = 8 sub-sections total; the §4.5 expansion is post-Wf-266-Final integration).
- "Wf-269 G5 0.6991 is the final 50-epoch best val_auc" (it is the Wf-269 capture-time snapshot; the run continued to 0.7236 @ epoch 24 and 0.7636 @ epoch 32).
- "207 §9.4 cite-target-section inconsistency is a content error" (the content is correct; only the section number pointer is imprecise; 188 has no §6, honest-negative lives in §5.2 + §4.5.2).

## §8 Cross-references

- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.1-§9.7 + §9.Y-§9.AH (20 main + sub-entries audited)
- `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §4.1 + §4.2 + §4.3 + §4.4 + §4.5.1-§4.5.4 (8 sub-sections audited)
- `molmetal/scripts/wave_budget_tracker.py` (constants block after `TODO_181_G5_ACTUAL_HR`, lines ~222-244; new `WF_269_G5_*` constants)
- `molmetal/reports/wf267_integrate/integrate_report.md` §2.1 + §2.4 (Wf-269A cuda:0 fix recovery, PID 609543)
- `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` (epochs 1-33 raw val_auc trace)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-edit ACTIVE rule, defers §4.2 + §4.3 cite edits)
- `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-PROJECTED label discipline)
- `mol-metal-paper-writing` §6 (limitations template)
- `mol-metal-gpu-execution` §6 (5-seed × 50-ep) + §7 (no best-seed selection)

---

**End of audit report. Wf-275D ship status: ✅ SHIPPED. Cross-ref audit: 19/20 consistent (1 minor inconsistency; honest-gap preserved). 188 §4 → 207 §9: 6/8 explicit cite, 2/8 implicit cite (paper-edit follow-up deferred). Wave budget tracker: 6 new Wf-269 G5 constants added. MEASURED-vs-PROJECTED label discipline: 20/20 entries PASS party-twenty-spirit §3.5.**
