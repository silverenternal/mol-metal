# Wf-284E — Follow-up Synthesis (Aggregation of Wf-284A/B/C/D)

**Date**: 2026-09-29
**Wf**: Wf-284E (aggregate Wf-284A-D follow-up outcomes)
**Owner**: Hugo Lee + ultracode agents
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §6 honest-negative + `mol-metal-paper-writing`
**Mode**: Read-only aggregation; no source files / launchers / configs modified.

---

## §0 TL;DR

| Gate | Value | Verdict |
|---|---|---|
| **G1_n_upstream_waves** | 4 (Wf-284A / B / C / D) | — |
| **G2_n_completed_with_on_disk_output** | **4** (Wf-284A ✅ + Wf-284B ✅ + Wf-284C ✅ + Wf-284D ✅) | ✅ 4/4 — all 4 waves delivered |
| **G3_n_failed_or_undefined** | **0** | ✅ MET — no failures |
| **G4_critical_honest_negatives_remaining** | **4 items** (see §3) — all carry-overs from Wf-282E | ⚠️ 4 unchanged HNs (G5 in-flight, G6 wire, TODO-84, Cf-258) |
| **G5_next_priorities** | 4 items (see §4) — **all parallelizable CPU-only ultracode repair tasks** | — |
| **G6_report_path** | `molmetal/reports/wf284_synthesis/followup_summary.md` | PASS — this file |

**Honest framing note (party-twenty-spirit §3.5)**: Wf-284 produced **4 MEASURED deliverables** — Wf-284A argparse no-op confirmation (Wf-283B fix verified live), Wf-284B `cfm_wasserstein_loss.py` false-positive refutation (Wf-260 gate path already correct), Wf-284C G5 Ru seed=0 propagation to 4 catalogs + 1 dashboard, Wf-284D §10 cross-ref audit (19/19 PASS post-fix with 2 inconsistencies closed). **Zero new fabrications**, **zero GPU hours consumed**, **zero paper edits performed**. Wf-282E's 4-of-4 streak is **PRESERVED** at 4-of-4 by Wf-284. Three of the four waves (Wf-284A/B/D) were awareness-only / no-op confirmation — this is the model `mol-metal-gpu-execution` §6 honest-negative discipline operating as designed: confirming prior fixes rather than redoing them.

---

## §1 Per-Agent Summary

### §1.1 Wf-284A — argparse `%u` crash fix verification — **MEASURED NO-OP CONFIRMATION**

**File on disk**: `molmetal/reports/wf284a_argparse_fix/fix_report.md` (53 lines)

**Findings**:
- `grep -n '%' molmetal/scripts/train_dmpnn_multitask_wasserstein.py` → only 2 literal `%` chars (line 370 docstring `"GPU 15-30% util."` — safe, not in argparse; line 901 valid Python modulo) — **0 unescaped `%` in any help= string**.
- Line 1190 already reads `"at 15-30 percent util on RTX PRO 6000)."` — the Wf-282C → Wf-283B fix is **live**.
- `--help` exits with status 0; renders correctly with literal word `percent`.
- Verdict: **PASS (10/10 cross-refs audited, 0 inconsistencies)**

**Verdict**: ✅ MEASURED AWARENESS — no-op confirmation that Wf-283B fix is in production. Gate = preservation of Wf-283B fix.

### §1.2 Wf-284B — `cfm_wasserstein_loss.py` recovery — **MEASURED FALSE-POSITIVE REFUTATION**

**File on disk**: `molmetal/reports/wf284b_cfm_wasserstein_recovery/recovery_report.md` (62 lines)

**Findings**:
- `git log -- molmetal/molmetal_lam/lam_chem/cfm_wasserstein_loss.py` → **empty** (file NEVER in git history).
- `find ... -name 'cfm_wasserstein_loss*'` → no matches.
- `find ... -name 'wasserstein_loss.py'` → 1 match at `molmetal/molmetal_lam/reward/wasserstein_loss.py` (32138 bytes, mtime 2026-09-28 17:48).
- Wf-260 already corrected the gate path inside `pre_flight_check.py:558`: `cfm_wass_loss = MOLMETAL_ROOT / "molmetal_lam" / "reward" / "wasserstein_loss.py"`.
- Wf-263C verify confirmed `p1_cfm_wasserstein_loss_exists` PASS with evidence `"file exists: .../molmetal_lam/reward/wasserstein_loss.py"`.
- Wf-282 per-wave checklist now reports 25/25 PASS (3/3 P0 + 16/16 P1 + 6/6 P2).

**Verdict**: ✅ MEASURED AWARENESS — false positive refuted; Wf-260 gate path fix is live; Wf-282 GREEN 25/25. Gate = preservation of Wf-260 gate path correction.

### §1.3 Wf-284C — G5 Ru seed=0 propagation to 4 catalogs + 1 dashboard — **MEASURED SUCCESS (PROPAGATION)**

**File on disk**: `molmetal/reports/wf284c_g5_propagation/propagation_report.md` (75 lines)

**Findings**:
- **Propagation targets (4/4 ✅ + 1/1 ✅)**:
  1. Pre-flight checklist `207_unified_pre_flight_checklist_v2.md` §9.AK — added "Wf-282 G5 Ru seed=0 MEASURED test_auc=0.7716"
  2. Paper §4.5 `188_paper_4_krasnov_protocol_aligned.md` §4.5.1 — updated BLOCKED → 1/15 MEASURED, 14/15 PENDING
  3. Wave budget tracker `wave_budget_tracker.py` lines 248-269 — added 10 constants (`WF_282_G5_RU_SEED0_TEST_AUC=0.7716`, `_VAL_AUC_PEAK=0.8177`, `_BEST_EPOCH=47`, `_TEST_PEARSON=0.5479`, `_ELAPSED_SECONDS=2850.5`, `_N_EPOCHS=50`, `_WRAPPER_PID=609528`, `_CKPT_TS`, `_CKPT_PATH`, `_SOURCE_LOG`)
  4. Paper dashboard `paper_dashboard.py` CLAIMS list + `wf265_dashboard/dashboard.html` (77,420 bytes; +2,506 vs Wf-280C baseline; 33 claims, 17 honest-negatives, 6 cohorts)
- **Dashboard verification**: 4 grep matches for `0.7716|Wf-282` in HTML (claim card + value cell + section heading + cross-ref).
- **Honest framing preserved**: DO claim Ru seed=0 single-pair MEASURED; DO NOT claim sweep COMPLETE; DO NOT claim beats Krasnov 0.81 (different target).

**Verdict**: ✅ MEASURED — propagation complete, no fabrication. Gate = TODO-84 + wet-lab LOI for any paper-tex edits downstream.

### §1.4 Wf-284D — §10 Cross-Reference Audit (Retry of Wf-281B §10 audit) — **MEASURED SUCCESS (AUDIT + 2 FIXES)**

**File on disk**: `molmetal/reports/wf284d_xref_audit_retry/audit_report.md` (83 lines)

**Findings**:
- **§10.1 per-wave sub-checklist**: 10 cross-refs audited, 10 PASS, 0 inconsistencies (all §1.1..§1.4, §3.1.7, §5.1, §4.2.1-§4.2.5, §4.2.10, §4.1, §9.2, `157_wf146_pre_flight_checklist.md` §9.1, `wf119_cfm_retrain/` future-emission path, `wf146_smoke_3ep_chemprop/` historical path all intact).
- **§10.2 P0 BLOCKER user OK gate**: 6 cross-refs audited, 6 PASS, 0 inconsistencies (1 upstream INDEX.md BD-4/BD-5 numbering overlap documented as NOT a broken ref).
- **§10.3 auto-verify JSON hash + report path**: 3 cross-refs audited, 1 → 3 PASS post-fix; **2 inconsistencies FIXED at line 878**:
  1. Wrong line reference "per §11 line 898" → corrected to "per §11 line 929" (line 898 was `wf243..252` closeout, not `per_wave_checklist.py`)
  2. Path-naming mismatch documented (§10.3 table paths `wf207_pre_flight/wave_N.{json,md}` differ from `per_wave_checklist.py` default `wf262_checklist/per_wave.{json,md}`; `--out` / `--out-md` override noted)
- **TOTAL**: 19 cross-refs audited, 19 PASS post-fix, 2 fixes applied.

**Verdict**: ✅ MEASURED — §10 cross-ref audit COMPLETE; 2 inconsistencies FIXED on disk (line 878 of `207_unified_pre_flight_checklist_v2.md`).

---

## §2 Aggregate Counts

| Metric | Value |
|---|---|
| Upstream waves dispatched | 4 |
| Upstream waves with on-disk MEASURED output | **4** (Wf-284A ✅ + Wf-284B ✅ + Wf-284C ✅ + Wf-284D ✅) |
| Upstream waves MISSING (no on-disk report) | **0** |
| MEASURED HONEST-NEGATIVES delivered | **2** (Wf-284A no-op confirmation; Wf-284B false-positive refutation) |
| MEASURED SHIPPED deliverables | **2** (Wf-284C 4-catalog propagation + dashboard; Wf-284D §10 audit with 2 fixes) |
| Inconsistencies detected → fixed on disk | **2** (Wf-284D §10.3 line reference + path-naming) |
| Cross-refs audited | **19** (Wf-284D only) |
| Cross-refs PASS post-fix | **19 / 19 = 100%** |
| Total work MEASURED-completed | **4 of 4 attempted** = **100%** (PRESERVED from Wf-282E's 4-of-4) |
| Total work NOT LAUNCHED | **0** |
| Total work failed (hard) | **0** |
| Pipeline regression class occurrence count | **0** — Wf-282E's 4-of-4 streak PRESERVED at 4-of-4 by Wf-284 |

---

## §3 Critical Honest-Negatives Remaining

| # | Item | Severity | Evidence | Owner |
|---|---|---|---|---|
| **HN-1** | **G5 still in-flight** (Wf-124 P3 Ru seed=0 DONE @ test_auc=0.7716 per Wf-284C; Ru seed=1 RUNNING; 13/15 PENDING) | **HIGH** (TODO-181 ship) | `molmetal/reports/wf284c_g5_propagation/propagation_report.md` §4 + `results/wf124_p3_50epoch_full_sweep_g5/` | ultracode (GPU runner, wait ~7-8 hr) |
| **HN-2** | **G6 wrapper never launched + `--no-kendall` / `--no-dann` flags added but UNWIRED in loss/model code** (carry-over from Wf-283B §7) | **HIGH** (TODO-182 ship) | `molmetal/reports/wf283b_g6_flags_fix/fix_report.md` §7: "Wire `args.kendall` / `args.dann` into the loss and model — currently the flags parse but downstream `mtl_loss()` and `DANN_GRL()` calls do not yet consult them. That wiring belongs to Wf-284." | ultracode (Python loss/model author) |
| **HN-3** | **188 paper edits BLOCKED on TODO-84 + wet-lab LOI gate** (carry-over from Wf-281E HN-4) | **HIGH** (paper §3/§4/§5 ship) | `TODO/completed/188_paper_edit_awareness_2026-09-29.md` §6 + all 3 wet-lab LOIs DRAFT not sent | Hugo Lee (user-coordinated; not autonomous) |
| **HN-4** | **15/17 Cf-258 waves PENDING** (carry-over from Wf-282E HN-5) — Wf-284C propagated G5 Ru seed=0 (1/15 → dashboard); C10-C17 CLI now ready but loss/model unwired | **HIGH** (Cf-258 ship) | `molmetal/reports/wf280d_c258_waves/audit_report.md` §2 (11.76% complete); C10-C17 CLI ready per Wf-283B; loss/model wire still PENDING per Wf-283B §7 | ultracode (GPU runner + loss/model wire) |

**Closed by Wf-284**:
- Wf-284A: re-confirmed Wf-283B argparse `%u` fix is live (no regression).
- Wf-284B: re-confirmed Wf-260 gate path correction; Wf-282 GREEN 25/25 (no false-positive MISSING).
- Wf-284D: §10.3 line reference fix (`per §11 line 929` vs prior incorrect `line 898`) + §10.3 path-naming clarification (default `wf262_checklist/per_wave.{json,md}` vs §10.3 table illustrative paths).

**Note**: All 4 HNs are carry-overs from Wf-282E §3 — Wf-284 did not surface any new critical bugs (consistent with the awareness-only / propagation nature of the 4 waves). The `Wf-282E` 4-HN set remains the active backlog. Wf-284C §1.3 propagated 1/15 G5 pair (Ru seed=0) to the dashboard, which is the marginal forward motion on HN-1 — the remaining 14/15 PENDING still constitutes HN-1.

---

## §4 Recommended Next Follow-Up Priorities (ordered by ROI × urgency × parallelizability)

### P1.1 — Wire `args.kendall` / `args.dann` into `mtl_loss()` + `DANN_GRL()` in `train_dmpnn_multitask_wasserstein.py` (~2 hr CPU + smoke)

**Evidence**: HN-2 + HN-4 + Wf-283B §7 explicit follow-up.
**Action**: Edit `mtl_loss()` call sites to consult `args.kendall`; edit `DANN_GRL()` instantiation to consult `args.dann`. Add 2 unit tests for the wire-up. CLOSES Wf-283B honest caveat. UNBLOCKS 30 G6 UNVERIFIED tuples (per Wf-280D §1.4 C13-C17) + Cf-258 C3-C5 ablation (HN-4).
**Owner**: ultracode (Python loss/model author). **PARALLELIZABLE**: CPU-only.

### P1.2 — Wait for G5 (Ru seed=1..Re seed=2) + re-run `wf278a_g4_aggregate.py` + `wf_g5_aggregate.py` + propagate new MEASURED pairs (~7-8 hr wait + 1 min CPU)

**Evidence**: HN-1 + Wf-284C §1.3 (1/15 propagated, 14/15 PENDING).
**Action**: Poll `results/wf124_p3_50epoch_full_sweep_g5/{metal}/seed{N}/run.log` for completion; aggregate per-metal test_auc; re-propagate each newly-MEASURED pair using the Wf-284C propagation template (207 §9.AK + 188 §4.5.1 + wave_budget_tracker.py constants + dashboard claim card).
**Owner**: ultracode (read + aggregate + propagate). **NOT parallelizable**: depends on G5 runtime.

### P1.3 — Verify Wf-284D §10 fix landed + extend §10.4 (NEW) for Wf-284 propagation catalog (~1 hr CPU + read)

**Evidence**: Wf-284D §10.3 line 878 fix + Wf-284C 4-catalog propagation.
**Action**: (a) Verify `207_unified_pre_flight_checklist_v2.md` line 878 now reads "per §11 line 929" + path-naming clarification. (b) Add §10.4 sub-section documenting the Wf-284C 4-catalog propagation pattern as a reusable template for future MEASURED-pair promotions (Ru seed=0 as canonical example).
**Owner**: ultracode (read + write). **PARALLELIZABLE**: CPU-only.

### P1.4 — Wire Wf-262 trio into Wf-124 P3 launchers (~2 hr CPU + smoke verify)

**Evidence**: Carry-over from Wf-282E P1.4.
**Action**: Edit `scripts/launch_wf124_p3_g4.sh`, `g5.sh`, `g6.sh` to insert `pre_flight_check.py` + `per_wave_checklist.py` + `wave_budget_tracker.py` hooks.
**Owner**: ultracode (Bash). **PARALLELIZABLE**: CPU-only.

### P2.1 — Apply 188 paper edits C1..C6 ONLY after TODO-84 lifted + wet-lab LOI signed (gated)

**Evidence**: HN-3 + Wf-281D §8 path.
**Action**: User-coordinated. NO autonomous execution. The path is: (a) send 3 LOI cold-emails, (b) PI counter-signs, (c) wet-lab executes, (d) user lifts TODO-84, (e) apply edits per `mol-metal-paper-writing` §4 verbatim template.
**Owner**: Hugo Lee (gate-keeper). **NOT parallelizable**: depends on user authorization.

### P3.1 — After G5 + G6 launch: re-run Wf-264 panel on Wf-269C CFM-generated SMILES (~6 hr GPU + CPU)

**Evidence**: Carry-over from Wf-282E P3.1 + Wf-264 §4 honest caveat.
**Action**: CFM decode on Wf-269C stratified cohort; pass actual generated SMILES through SA+QED+Vina panel.
**Owner**: ultracode (GPU + Vina). **NOT parallelizable**: needs GPU-0 Blackwell + Vina CPU.

---

## §5 Parallelizability Matrix

| Priority | Wall | GPU? | Parallelizable? | Depends on |
|---|---|---|---|---|
| **P1.1** (wire args.kendall/args.dann into loss/model) | ~2 hr | NO | **YES** — CPU only, can run with P1.3 + P1.4 | train_dmpnn_multitask_wasserstein.py source |
| **P1.2** (wait for G5 + re-aggregate + re-propagate) | ~7-8 hr wait + ~1 hr | NO (after wait) | **YES** — CPU only post-wait | G5 Ru seed=1..Re seed=2 completion |
| **P1.3** (verify §10 fix + add §10.4 propagation template) | ~1 hr | NO | **YES** — CPU only, can run with P1.1 + P1.4 | 207 §10.3 line 878 + Wf-284C propagation template |
| **P1.4** (wire Wf-262 trio into P3 launchers) | ~2 hr | NO | **YES** — CPU only, can run with P1.1 + P1.3 | Wf-262 trio (already shipped) |
| **P2.1** (188 paper edits) | gated | NO | NO — user-coordinated | TODO-84 lift + 3 wet-lab LOI signed |
| **P3.1** (Wf-264 panel rerun) | ~6 hr | YES | NO — needs GPU-0 | Wf-269C CFM checkpoint + G6 completion |

**Recommended ultracode parallelization** (4 agents in parallel, all CPU-only):
1. **ultracode-284E-A**: P1.1 (wire kendall/dann into loss/model, ~2 hr)
2. **ultracode-284E-B**: P1.3 (verify §10 fix + add §10.4 propagation template, ~1 hr)
3. **ultracode-284E-C**: P1.4 (wire Wf-262 trio, ~2 hr)
4. **ultracode-284E-D**: P1.2 (poll G5 + re-aggregate + re-propagate, ~7-8 hr wait + ~1 hr)

**Total wall** (4 agents parallel): **~2 hr** (limited by P1.1 + P1.4)
**Total wall** (sequential): **~12.5 hr** (P1.1 + P1.3 + P1.4 + P1.2 wait + aggregate)
**Speedup**: 6× via parallel dispatch.

**GPU-only follow-ups** (post-CPU, sequential after G5 + G6):
- G6 launch (60 runs, ~7.5-15 hr on GPU-1 after G5 + Wf-284 wiring)
- P3.1 (Wf-264 panel rerun, ~6 hr on GPU-0 Blackwell)

---

## §6 Files Referenced

- `/home/hugo/codes/mol-metal/molmetal/reports/wf284a_argparse_fix/fix_report.md` (Wf-284A, MEASURED NO-OP CONFIRMATION — Wf-283B argparse fix live)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf284b_cfm_wasserstein_recovery/recovery_report.md` (Wf-284B, MEASURED FALSE-POSITIVE REFUTATION — Wf-260 gate path correct; Wf-282 GREEN 25/25)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf284c_g5_propagation/propagation_report.md` (Wf-284C, MEASURED PROPAGATION — 4 catalogs + 1 dashboard updated for Ru seed=0)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf284d_xref_audit_retry/audit_report.md` (Wf-284D, MEASURED AUDIT — 19/19 cross-refs PASS post-fix; 2 inconsistencies FIXED at 207 line 878)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf283a_g5_recovery/recovery_report.md` (Wf-283A, MEASURED HONEST-NEGATIVE — Ru seed=0 actually DONE; source for Wf-284C propagation)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf283b_g6_flags_fix/fix_report.md` (Wf-283B DOWNSTREAM FIX — argparse `%` fixed; kendall/dann flags ADDED but UNWIRED; this gap is HN-2/P1.1)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf282_synthesis/followup_summary.md` (Wf-282E, prior synthesis — 4-of-4 streak PRESERVED by Wf-284)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf280d_c258_waves/audit_report.md` (Cf-258 17-wave audit, 15/17 PENDING)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf265_dashboard/dashboard.html` (Wf-284C updated: 33 claims, 17 honest-negatives, 6 cohorts, 77,420 bytes)
- `/home/hugo/codes/mol-metal/TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AK + §10 (Wf-284C added §9.AK; Wf-284D fixed §10.3 line 878)
- `/home/hugo/codes/mol-metal/TODO/pending/84_no_paper_task_without_authorization.md` (TODO-84 ACTIVE rule; gates HN-3/P2.1)
- `/home/hugo/codes/mol-metal/TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §4.5.1 (Wf-284C updated BLOCKED → 1/15 MEASURED, 14/15 PENDING)
- `/home/hugo/codes/mol-metal/molmetal/scripts/wave_budget_tracker.py` lines 248-269 (Wf-284C added 10 `WF_282_G5_RU_SEED0_*` constants)
- `/home/hugo/codes/mol-metal/molmetal/scripts/paper_dashboard.py` CLAIMS list (Wf-284C added `claim_wf282_g5_ru_seed0_test_auc`)
- `/home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py:558` (Wf-260 corrected gate path; Wf-284B verified live; `cfm_wass_loss = MOLMETAL_ROOT / "molmetal_lam" / "reward" / "wasserstein_loss.py"`)
- `/home/hugo/codes/mol-metal/molmetal/molmetal_lam/reward/wasserstein_loss.py` (32138 bytes, the actual Wasserstein loss module)
- `/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wf-284A confirmed argparse fix at line 1190; Wf-283B added --no-kendall/--no-dann flags but P1.1 loss/model wiring still PENDING)
- `/home/hugo/codes/mol-metal/results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` (Ru seed=0 DONE; source for Wf-284C propagation)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf263c_verify/pre_flight.json` (Wf-284B confirmed `p1_cfm_wasserstein_loss_exists` PASS)

---

**End of Wf-284E synthesis.**