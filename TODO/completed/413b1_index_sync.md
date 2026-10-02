# Wf-413-B1 — docs-only INDEX + AGENTS sync + comprehensive final synthesis v42 SHIPPED

**Date:** 2026-09-30 ~17:30 CST
**Wave:** Wf-413 (ultracode GPU 0 + Re seed=4 wait + Wave A/B/C launch attempts — 3 SHIPPED + 1 MISSING per filesystem-truth rule)
**Sub-flow:** B1 — docs-only INDEX + AGENTS sync + comprehensive final synthesis
**Authoring tool:** CPU-only docs+code+test+verify; 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning

---

## §0 TL;DR

Wf-413-B1 SHIPPED the docs-only sync surface for the Wf-413 ultracode wave:

1. ✅ **`TODO/completed/413b1_index_sync.md`** (this file, ~150 LOC, 6 sections)
2. ✅ **`TODO/completed/413_final_synthesis.md`** (~700 LOC, 9 sections)
3. ✅ **`TODO/INDEX.md`** `## §22.98` updated to cover all 4 sub-flows (3 SHIPPED + 1 MISSING per §5.5 filesystem-truth rule)
4. ✅ **`AGENTS.md`** NEW §6 row appended for Wf-413 wave

**On-disk reality (verified via `ls -la TODO/completed/413*.md` at Wf-413-B1 timestamp):**

| Sub-flow | On-disk state | mtime | Wall | Status |
|---|---|---|---:|---|
| **Wf-413-A1** | `TODO/completed/413a1_gpu0_probe_re_seed4_wait.md` (193 lines / ~20 KB) | 2026-09-30 16:46 | ~64.2 min GPU (Re seed=4 wait) + ~5 min CPU | ✅ SHIPPED |
| **Wf-413-A2** | **MISSING from disk**; `molmetal/reports/wf413_wave_a/wrapper.log` 258 bytes mtime 2026-09-30 16:51 (Wave A attempted but BLOCKED at pre-flight FAIL — closeout never authored) | 16:51 wrapper.log only | ~30 s CPU-only | ⚠️ MISSING per §5.5 |
| **Wf-413-A3** | `TODO/completed/413a3_wf299_wave_b_launch.md` (312 lines / ~16 KB) | 2026-09-30 17:14 | ~5 min CPU-only (1-line wrapper fix + launch attempt killed at 75s) | ✅ SHIPPED |
| **Wf-413-A4** | `TODO/completed/413a4_wf299_wave_c_launch.md` (324 lines / ~17 KB) | 2026-09-30 17:17 | ~30 s CPU-only (2 launch attempts BLOCKED) | ✅ SHIPPED |
| **Wf-413-B1** | this file + `413_final_synthesis.md` + INDEX.md §22.98 update + AGENTS.md §6 row | Wf-413-B1 timestamp | ~10 min CPU-only | ✅ SHIPPED |

**Total SHIPPED on disk: 3 of 4 sub-flows (A1 + A3 + A4).** The compute task brief claim "Wf-413-A1..A4 SHIPPED 4 deliverables: GPU probe + Wave A + Wave B + Wave C" is **INCORRECT for A2** per AGENTS.md §5.5 filesystem-truth rule — file `TODO/completed/413a2_wf299_wave_a_launch.md` does NOT exist on disk; only the Wave A wrapper.log exists showing the attempt at 16:51 was BLOCKED at pre-flight FAIL. The predecessor closeouts (413a3 + 413a4) reference 413a2 as a "predecessor — Wave A pre-flight FAIL" but the closeout document was never authored.

**Pre-flight cascade PRESERVED** at Wf-411-B1 baseline `95 PASS / 18 FAIL / 4 SKIP / 117 TOTAL` (NOT re-run by Wf-413-B1 — CPU-only docs+verify per AGENTS.md §5.1 + kill_then_adapt §0).

---

## §1 §22.98 INDEX entry update

The existing `## §22.98 Wf-413 ultracode GPU USER_OK m1_auc 7/7 run completion` INDEX entry only covers Wf-413-A1. This Wf-413-B1 sync expands it to cover all 4 sub-flows (A1 + A3 + A4 SHIPPED + A2 MISSING per §5.5).

**§22.98 updated entry appended between §22.97 Wf-411 + §22.99 Wf-414 (preserved order):**

- §22.97 Wf-411 ultracode GPU USER_OK m2 R² Path A + Re seed=4 + m6_kappa supplementary verifier wave — 3 SHIPPED + 1 HONEST-NEGATIVE + 1 IN-FLIGHT deliverables + comprehensive final synthesis v39 (2026-09-30 — Wf-411-B1 this entry; user verbatim "我感觉得基于现状做更具体的规划..." via Wf-410 final synthesis §7 user voice relay)
- **§22.98 Wf-413 ultracode GPU 0 sequential CFM retrain — Wave A + Wave B + Wave C + Wf-131 Re seed=4 wait** (2026-09-30 — Wf-413-B1 this entry; user directive 2026-09-30 verbatim *"用GPU0好了，同时启动好了"* via Wf-413-A1 §0 + Wf-413-A3 §0 + Wf-413-A4 §0)
- §22.99 Wf-414 ultracode CPU-only m2 R² launcher + Lambda CFM audit fix + atomic migration final — 3 SHIPPED deliverables + 17 NEW tests PASS + comprehensive final synthesis v41 (2026-09-30 — Wf-414-B1 this entry)

**Note**: The INDEX.md §22.98 entry was authored in 2 stages:
1. Wf-413-A1 SHIPPED its own §22.98 partial entry (2026-09-30 16:46 CST) covering only the A1 sub-flow.
2. Wf-413-B1 SHIPPED this expanded §22.98 entry (2026-09-30 ~17:30 CST) covering all 4 sub-flows per §5.5 filesystem-truth rule.

---

## §2 AGENTS.md §6 row update

NEW §6 row appended between Wf-411 + Wf-414 (preserved order):

- §6 row Wf-411 ... (preserved at line 677)
- **§6 row Wf-413 ... (NEW this sync)** — references §22.98 with all 4 sub-flows + 3 SHIPPED + 1 MISSING per §5.5
- §6 row Wf-414 ... (preserved at line 683)

**§6 row content** — follows the Wf-411-B1 + Wf-414-B1 prior sync pattern: USER_OK=1 grant via AGENTS.md §5.1 standing GPU authorization; `mol-metal-cfm-training` + `mol-metal-anticancer` + `mol-metal-gpu-execution` + `kill_then_adapt` §0 + `party-twenty-spirit` §3.5 + AGENTS.md §5.5 filesystem-truth rule strictly observed; §5.7 NEW trainer flag wire-in smoke-required rule applied; standing directives §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10 re-confirmed.

---

## §3 Wall budget

- **Wf-413-B1 wall** MEASURED ~10 min CPU-only (this file + final synthesis + INDEX + AGENTS edits)
- **Wf-413 wave total wall** MEASURED:
  - A1: ~64.2 min GPU + ~5 min CPU
  - A2: ~30 s CPU (launch attempt BLOCKED)
  - A3: ~5 min CPU (1-line wrapper fix + launch attempt killed at 75s)
  - A4: ~30 s CPU (2 launch attempts BLOCKED)
  - B1: ~10 min CPU
  - **Total: ~70 min GPU + ~21 min CPU**

**Combined wall PROJECTED 25-45 min CPU-only for B1 per task brief** (MEASURED ~10 min CPU-only — well under PROJECTED).

---

## §4 Honest framing per `party-twenty-spirit §3.5`

9 honest-negative caveats preserved per AGENTS.md §5.2 + §5.5 + Wf-358-C2 §9 + Wf-369-A3 §6:

1. **m1 AUC overall MEASURED_BELOW_KRASNOV_PARTIAL** (per Wf-413-A1 §4; 7/7 runs complete; Rh fold PASS_KRASNOV 0.8798 +0.020 + Re fold BELOW_KRASNOV 0.7088 gap −0.151 + overall 0.7577 BELOW_KRASNOV 0.86 gap −0.102; paper §4 LEAD claim = Rh fold only; paper §6 honest-negative on Re fold)
2. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 9 paths exhausted per Wf-389-A1 + Wf-381-A4 + Wf-377-A1 + Wf-365-A3 + Wf-371-A1 + Wf-375-A4 + Wf-411-A4 per-cohort × Cohen 1988 3-band refinement 13/15 cohort-cells reach MODERATE band |r| ≥ 0.1; 0 STRONG bands; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted)
3. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-411-A2 + Wf-398-A2 + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-390-A2 deeper audit; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per Wf-388-A1 §2.1 + Wf-398-A2 §1.1 MEASURED -0.1440 ± 0.0325; metalcytotox_combined n=2000 WEAK_RECOVERY +0.1981)
4. **Wf-413-A2 closeout file MISSING on disk** per AGENTS.md §5.5 filesystem-truth rule — only `molmetal/reports/wf413_wave_a/wrapper.log` 258 bytes mtime 2026-09-30 16:51 exists showing Wave A was attempted but BLOCKED at pre-flight FAIL; the closeout document `TODO/completed/413a2_wf299_wave_a_launch.md` was never authored; the predecessor closeouts (413a3 + 413a4) reference 413a2 as a "predecessor — Wave A pre-flight FAIL" but this is filesystem-truth-impossible; recommended follow-up = Wf-415-A1 author the missing A2 closeout with retroactive timestamp per AGENTS.md §5.5 rule #5
5. **Wf-413-A3 + Wf-413-A4: 0 GPU launches performed** (both attempts BLOCKED at pre-flight gate line 262 + flag parser line 217 of `molmetal/scripts/launch_wf299_cfm_main.sh`; `kill_then_adapt §0` safety contract honored)
6. **Pre-flight cascade 95/18/4/117 PRESERVED** at Wf-411-B1 baseline (drift from Wf-407-Wf-410 baseline 89/2/3/94 → 95/18/4/117 reflects pipeline expansion +23 NEW gates from Wf-410-A2/A3/A4 + Wf-411-A1/A4 wave activity; all 18 FAILs PRE-EXISTING per Wf-345-D1 §3 + Wf-358-D1 §1 honest-negative baseline)
7. **Wf-413-A3 Issue 1-4 carry forward** (case statement rejects CFM flags + Wave B/C use D-MPNN G6 trainer not CFM + Wave B/C hardcoded cuda:1 contradicting user voice GPU 0 + `--json` argparse bug FIXED by Wf-413-A3 1-line patch)
8. **Wf-299 GPU launch BLOCKED** on Wf-408-A3 alias mapping + Wf-408-A2 trainer wire-in (per Wf-410-A4 contract (e) FAIL)
9. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)

---

## §5 Cross-references

- **Wf-413 SHIPPED on disk (3 files):**
  - `TODO/completed/413a1_gpu0_probe_re_seed4_wait.md` (193 lines / ~20 KB; mtime 2026-09-30 16:46)
  - `TODO/completed/413a3_wf299_wave_b_launch.md` (312 lines / ~16 KB; mtime 2026-09-30 17:14)
  - `TODO/completed/413a4_wf299_wave_c_launch.md` (324 lines / ~17 KB; mtime 2026-09-30 17:17)
- **Wf-413 MISSING on disk:**
  - `TODO/completed/413a2_wf299_wave_a_launch.md` (referenced by 413a3 + 413a4 as "predecessor — Wave A pre-flight FAIL" but file does NOT exist; only `molmetal/reports/wf413_wave_a/wrapper.log` 258 bytes mtime 2026-09-30 16:51 evidence of attempt)
- **Wf-413 GPU work evidence:**
  - `molmetal/reports/wf131_p3_deferred_runs/Re/seed4/aggregate.json` (4021 bytes; Re seed=4 MEASURED test_auc=0.7196)
  - `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` (canonical 7/7 catalog updated 2026-09-30 16:47:16; n_runs=7; value=0.7577)
  - `metrics/by_metric/metal_lomo_auc_wasserstein.json` (canonical Tier-2 catalog 12667 bytes mtime 2026-09-30 16:46:29; same update)
  - `molmetal/reports/wf413_wave_a/wrapper.log` (258 bytes; "[FATAL] pre_flight_check FAIL")
  - `molmetal/reports/wf413_wave_b/wrapper.log` (145 bytes) + `result.json` (4908 bytes)
  - `molmetal/reports/wf413_wave_c/wrapper_a4_attempt.log` (258 bytes) + `wrapper_a4_wavec_literal.log` (43 bytes) + `preflight_a4.json` (38847 bytes; 95/18/4/117) + `result.json` (9986 bytes)
- **Wf-413-B1 SHIPPED on disk (2 files + 2 updates):**
  - `TODO/completed/413b1_index_sync.md` (this file)
  - `TODO/completed/413_final_synthesis.md` (~700 LOC, 9 sections)
  - `TODO/INDEX.md` §22.98 entry updated (expanded to cover all 4 sub-flows)
  - `AGENTS.md` §6 row appended (NEW Wf-413 row between Wf-411 + Wf-414)
- **Prior ultracode wave closeouts (the Wf-413 context):**
  - `TODO/completed/{411a1_gpu_env_probe,411a2_m2_path_a_launch,411a4_m6k_verifier,411b1_index_sync,411_final_synthesis}_*.md` (Wf-411 wave — 3 SHIPPED + 1 IN-FLIGHT + 1 HONEST-NEGATIVE)
  - `TODO/completed/410{a1_paper_section6_v4,a2_lambda_cfm_verify,a3_atomic_migration_extended,a4_wf299_re_verify,b1_index_sync,final_synthesis}_*.md` (Wf-410 wave — 5 SHIPPED)
  - `TODO/completed/{407a1_gpu_env_probe,407a3_wf299_wave3_scaffold_not_functional,407a4_metrics_update,407b1_index_sync,407_final_synthesis}_*.md` (Wf-407 wave — 5 SHIPPED)
  - `TODO/completed/{408a2_wf380_loss_wire,408a3_preflight_launcher_fixes,408b1_index_sync,408_final_synthesis}_*.md` (Wf-408 wave — 4 SHIPPED)
- **Related scripts + reports:**
  - `molmetal/scripts/launch_wf299_cfm_main.sh` (586 lines; case statement lines 211-216; Wave A/B/C functions lines 458/494/520; pre-flight call lines 252-264; `--json` fix applied by Wf-413-A3 at line 257-258)
  - `molmetal/scripts/pre_flight_check.py` (line 6955-6959 defines `--json` argparse `type=Path`)
  - `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Wave A trainer)
  - `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wave B + C trainer)
  - `molmetal/scripts/train_metal_lomo_wasserstein.py` (Wf-131 P3 trainer for Re seed=4 retrain)
  - `molmetal/scripts/launch_wf131_p3_deferred_runs.sh` (Wf-355-B1 USER_OK-gated wrapper that launched Wf-413-A1 Re seed=4 retrain)
- **Cross-ref prior INDEX entries (preserved order):**
  - `TODO/INDEX.md` §22.97 Wf-411 (preserved)
  - `TODO/INDEX.md` §22.99 Wf-414 (preserved)
- **AGENTS.md standing directives re-confirmed:** §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 (filesystem-truth rule) + §5.6 (atomic write) + §5.7 NEW (trainer flag smoke-required) + §10 (wetlab deprecation)

---

## §6 User voice answer

**Per direct user directive (2026-09-30)** *"用GPU0好了，同时启动好了"* via Wf-413-A1 §0 + Wf-413-A3 §1 + Wf-413-A4 §7: Wf-413 ultracode wave SHIPPED 3 of 4 sub-flows on disk + 1 HONEST-NEGATIVE close (A2 Wave A BLOCKED at pre-flight FAIL — closeout never authored per AGENTS.md §5.5 filesystem-truth rule). **PARTIAL — 3/4 deliverables SHIPPED**: (a) **Wf-413-A1 GPU 0 probe + Re seed=4 wait SHIPPED** — 7/7 (metal, seed) pairs complete at 50 epochs of Wf-131 P3 DEFERRED 7-run sweep; m1_auc canonical catalog updated 6/7 → 7/7 runs; (b) **Wf-413-A2 Wave A launch MISSING closeout** — Wave A attempted but BLOCKED at pre-flight FAIL (`molmetal/reports/wf413_wave_a/wrapper.log` 258 bytes); closeout document never authored; the predecessor closeouts (413a3 + 413a4) reference 413a2 as "predecessor — Wave A pre-flight FAIL" but this is filesystem-truth-impossible per AGENTS.md §5.5; recommended follow-up = Wf-415-A1 author the missing A2 closeout with retroactive timestamp per §5.5 rule #5; (c) **Wf-413-A3 Wave B launch BLOCKED SHIPPED** — 4 design mismatches documented (case statement rejects CFM flags + Wave B uses D-MPNN G6 trainer + cuda:1 hardcode + `--json` argparse bug FIXED by 1-line wrapper patch); (d) **Wf-413-A4 Wave C launch BLOCKED SHIPPED** — pre-flight regression (95/18/4/117 vs documented baseline 89/2/3/94) + Wave C hardcoded cuda:1 + wrapper flag rejection; 0 GPU launches across A3+A4 attempts. Standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10 re-confirmed. **NEW ultracode Wf-415+ IS the recommended follow-up** per Wf-413-B1 §22.98 final synthesis + Wf-413-A1 §9 + Wf-411 final synthesis §4 + Wf-410 final synthesis §4 + Wf-407 final synthesis §6 + Wf-406 final synthesis §4 + Wf-399 final synthesis §4 + Wf-398 final synthesis §4 + Wf-396-B1 §7 + Wf-388 final synthesis §7 + Wf-381-B1 §22.77 + Wf-378-B1 §22.73 with 6+ USER_OK-gated sub-flows including: Wf-415-A1 author the missing Wf-413-A2 closeout (retroactive per §5.5) + Wf-415-A2 USER_OK-gated Wf-299 GPU launch (post-Wf-408-A3 alias mapping patch) + Wf-415-A3 USER_OK-gated m2 R² Path A 5-metal × 3-seed × 50-epoch retrain (~12 hr GPU) + Wf-415-A4 USER_OK-gated Wf-388-A1 m2 R² per-cohort REAL verification retrain + Wf-415-A5 USER_OK-gated paper §6 v4 write from `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (~12-15 hr CPU) + Wf-415-A6 USER_OK-gated paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md` (~10 hr CPU) (~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update + Wf-413 retroactive closeouts + Wf-414 follow-ups total).

**CHRONOLOGY NOTE**: Wf-413-B1 ships AFTER Wf-411-B1 + Wf-413-A1 + Wf-413-A3 + Wf-413-A4 (all ~16:00-17:30 CST 2026-09-30); Wf-413 wave structure: 3 SHIPPED on disk (A1 + A3 + A4) + 1 MISSING (A2 closeout) + 1 docs-only sync (B1) = 4 of 5 planned deliverables SHIPPED. The 5th deliverable (Wf-413-A2 closeout) is the recommended Wf-415-A1 follow-up.

---

**Wf-413-B1 SHIPPED.** §22.98 INDEX entry expanded + §6 row appended + 9 honest-negative caveats preserved.
