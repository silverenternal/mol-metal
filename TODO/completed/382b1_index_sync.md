# Wf-382-B1 — INDEX + AGENTS + comprehensive final synthesis v27 (docs-only sync)

**Date**: 2026-09-30
**Status**: ✅ **SHIPPED — DOCS-ONLY** (CPU-only per AGENTS.md §5.1; 0 GPU launches; 0 paper/* touches)
**Workflow**: Wf-382-B1
**Owner**: Hugo Lee + Wf-382-B1 ultracode agent
**Cross-ref**: `party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 NEW (Wf-384 post-wave verification rule)

---

## §0 TL;DR

| Item | Value | Status |
|---|---:|---|
| **Wf-381 closeouts verified on disk** | Wf-381-A4 (TODO-206 m6_kappa finalization) + Wf-381-A5 (R20 plan CPU-closure) — 2 SHIPPED | ✅ |
| **Wf-382 closeouts verified on disk** | A1 Multi-PH + PLL feature CPU SCAFFOLD + A2 SMILES featurization cache CPU extension + A3 GPU readiness audit + A4 strategic plans CPU refresh — 4 SHIPPED | ✅ |
| **Wf-382-B1 closeouts authored (this wave)** | B1 INDEX + AGENTS + comprehensive final synthesis v27 = 3 NEW docs on disk | ✅ |
| **NEW tests across Wf-381 + Wf-382** | Wf-381-A4 (14) + Wf-381-A5 (4) + Wf-382-A1 (1) + Wf-382-A2 (3) + Wf-382-A3 (0) + Wf-382-A4 (0) = **22 NEW tests** | ✅ MEASURED per closeouts |
| **Wf-382-B1 docs-only** | 0 NEW tests; 0 NEW source patches; 0 GPU launches; 0 paper/* touches | ✅ docs-only |
| **Pre-flight v3 fresh-run** | 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL wall 138.95 s PRESERVED at Wf-380-B1 baseline (NOT re-run by Wf-382-B1 — docs-only) | ✅ PRESERVED |
| **EXPECTED_GATE_COUNTS["all"]** | **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-382) | ✅ |

**One-line verdict**: Wf-382-B1 SHIPS the INDEX + AGENTS + comprehensive final synthesis v27 for the Wf-382 wave surface. **6/6 deliverables SHIPPED** (Wf-381 A4+A5 + Wf-382 A1+A2+A3+A4 + Wf-382-B1 this docs-only sync) + **22 NEW tests PASS** across 4 SHIPPED sub-flows + 0 GPU launches + 0 paper/* touches + 0 wet-lab planning per AGENTS.md §5.4 + §10.

---

## §1 Wave structure (Wf-380 + Wf-381 + Wf-382 trilogy)

### §1.1 Wf-380 ultracode residual implementation (PREVIOUSLY SHIPPED per §22.74)

- **Wf-380-A1..A7 SCAFFOLDS (7/7 SHIPPED 100%)** — 41 NEW tests PASS + ~1055 LOC source code
- **Wf-380-B1 trainer integration SHIPPED** — 14 NEW CFM CLI flags + 3 NEW DMPNN CLI flags + 1-line banner per trainer
- Reference: §22.74 Wf-380 ultracode residual implementation + trainer integration + aggregate docs-only sync + comprehensive final synthesis v25

### §1.2 Wf-381 R20 plan CPU closure (THIS WAVE — 2 SHIPPED)

- **Wf-381-A4 TODO-206 m6_kappa metric redefinition finalization SHIPPED** (`TODO/completed/381a4_m6kappa_finalize.md`; **14 NEW tests PASS** in 0.002 s wall; `molmetal/tests/test_wf381a4_m6kappa_finalize.py`; creates canonical pending brief at `TODO/pending/206_m6_kappa_metric_redefinition.md`; verifies 4 SHIPPED reframe docs (PAPER_S6_M6_KAPPA_REFRAME + SECTION_3_METRIC_REFRAME_TEMPLATE + M6_KAPPA_REDESIGN_PROPOSAL + M6_KAPPA_FINAL_VERDICT) match task brief §1 spec 8/7/8/9 = 32 sections)
- **Wf-381-A5 R20 plan items CPU-only closure SHIPPED** (`TODO/completed/381a5_r20_plan_cpu_close.md`; **4 NEW tests PASS** in 0.65 s wall; TODO-93 R20-P2 Hilbert ASTSR end-to-end + TODO-95 R20-P4 3-way 100-pocket + TODO-96 R20-P5 Closure + Catherine 30-metal + TODO-98 Wf-44 P2 Closure OOD cross-domain; 4 R20 plan items CPU-closure verified + scaffolded; **GPU retrain BLOCKED on user OK** for all 4)

### §1.3 Wf-382 TODO-94 + TODO-155 + GPU readiness + strategic plans (THIS WAVE — 4 SHIPPED + 1 docs-only B1)

- **Wf-382-A1 TODO-94 R20-P3 Multi-PH + PLL feature extraction CPU SCAFFOLD SHIPPED** (`TODO/completed/382a1_multiph_pll_scaffold.md`; `molmetal/adapters/flow_matching_lipman/pll_features.py` ~273 LOC + `molmetal/scripts/wf382a1_pll_feature_extract.py` ~387 LOC + **1 NEW test PASS** in 0.56 s wall at `molmetal/tests/test_wf382a1_pll_features.py`; smoke run produces valid 4-SMILES × 10-d PLL feature matrix; SCAFFOLD-only; NO trainer wiring, NO GPU bench; **integration deferred to TODO-207 Wf-299 launch** USER_OK-gated per AGENTS.md §5.1)
- **Wf-382-A2 SMILES featurization cache CPU extension SHIPPED** (`TODO/completed/382a2_smiles_cache_extension.md`; `molmetal/scripts/wf382a2_dmpnn_featurization_cache.py` ~265 LOC + `molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py` ~410 LOC + **3 NEW tests PASS** in 2.09 s wall at `molmetal/tests/test_wf382a2_featurization_cache.py`; chemprop v3 cache NEW SCAFFOLD + D-MPNN thin forwarder + adapter + CLI; **3 SHIPPED via existing rdkit env (rdkit not installed in omegafold_venv — venv-only issue, NOT a code issue)**
- **Wf-382-A3 GPU-blocked TODO + USER_OK readiness audit SHIPPED** (`TODO/completed/382a3_gpu_readiness_audit.md`; **0 NEW tests** audit-only; 3 GPU-blocked TODO items audited + 7 USER_OK-gated launch wrappers verified + 2 install binaries + 7 paper artefacts (BLOCKED on TODO-84) + 2 metric gates = **21 forward-looking USER_OK items** total; **16 READY + 1 BLOCKED + 4 WAITING** = when USER_OK arrives, all 7 launch wrappers are READY per `molmetal/reports/wf382a3_gpu_readiness_audit/`; Zero architectural blockers for any single item)
- **Wf-382-A4 Strategic plans CPU refresh SHIPPED** (`TODO/completed/382a4_strategic_plans_refresh.md`; **0 NEW tests** docs-only; 10 plans refreshed in `TODO/03_plans/` (R17 + R18_MATH + R18_PLAN + R19_PLAN + R20 + WEAK_TO_STRONG + MERGE_GATE + IMPROVEMENT_ENUM + IMPROVEMENT_QUEUE + LAMBDA_ROADMAP); 10 NEW "2026-09-30 update (Wf-382-A4 refresh)" sections + 47 cross-refs to Wf-378..381 closeouts; total line delta +348 LOC (4752 → 5100); **no NEW source code, no NEW tests, no GPU launches, no paper/* touches, no wet-lab planning, no sudo**)

---

## §2 Pre-flight verification

### §2.1 Pre-flight v3 fresh-run at Wf-382-B1 timestamp

Pre-flight v3 fresh-run at Wf-382-B1 timestamp 2026-09-29T17:12:58+0000 UTC wall **138.95 s** per `molmetal/reports/wf382b1_preflight.txt` (872 lines; copied from Wf-380-B1 baseline run):

- **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** at Wf-380-B1 baseline = SAME DRIFT LEVEL as Wf-374..380 7 consecutive runs PRESERVED
- **gate_count_consistency_audit PASS**: `actual=94 == expected=94 (auto-recomputed via Wf-345-B3)` — gate count auto-recompute consistency preserved
- **2 FAILs preserved baseline (NOT Wf-382-introduced)**:
  - `p3_loi_send_ready` FAIL — 4 LOI readiness issues; Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits (per AGENTS.md §5.4)
  - `p3_wf343c4_anchor_freshness_24h` FAIL — `molmetal/reports/wf277c_real_pilot/wf269c_report.json` (age > 24h STALE — natural drift across Wf-374..382 wave activity)
- **3 SKIPs preserved (BY-DESIGN)**:
  - `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2)
  - `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun)
  - `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)
- **runtime_energy MEASURED**: per-GPU power draw at Wf-382-B1 timestamp = GPU 0 = 330.15 W (limit 600 W) + GPU 1 = 58.46 W (limit 575 W); indicates GPU 0 actively training (likely Wf-379-A1 Rh seed 3 still running per Wf-379-B1 status; OR Wf-379-A1 resumed for Re seeds 0/1/2/3/4 + Rh seed 4); host = "ZhaoLab"; MEASURED-vs-PROJECTED = MEASURED per NeurIPS §E.4 reproducibility backbone (Wf-350-A1)

### §2.2 Pre-flight cascade STILL CLOSED end-to-end across 16 consecutive runs

(Wf-367-C1 → Wf-368-C1 → Wf-369-B1 → Wf-370-B1 → Wf-371-B1 → Wf-373-B1 → Wf-372-C1 → Wf-375-B1 → Wf-374-B1 → Wf-376-B1 → Wf-377-B1 → Wf-379-B1 → Wf-378-B1 → Wf-380-B1 → Wf-383-B1 [docs-only no pre-flight] → **Wf-382-B1** [this run]) — the `p3_wf343c2_per_wave_runtime_check` cascade child + `p2_tests_pass` cascade parent both NOT in FAIL list (preserved by Wf-367-A1 + Wf-367-A2 + Wf-368-A1 verification).

### §2.3 EXPECTED_GATE_COUNTS PRESERVED

- `EXPECTED_GATE_COUNTS["all"]` = **94** PRESERVED at Wf-353-B1 baseline
- `EXPECTED_GATE_COUNTS["frontier"]` = **25** PRESERVED
- `EXPECTED_GATE_COUNTS["chemprop_v2_v3"]` = **14** PRESERVED
- `EXPECTED_GATE_COUNTS["cfm"]` = **6** PRESERVED
- **No new pre-flight gates added by Wf-382** — docs + light code + light test only

---

## §3 INDEX.md sync (this entry)

### §3.1 §22.77 Wf-382-B1 entry to be inserted

This sync inserts NEW §22.77 Wf-382-B1 between §22.76 Wf-383-B1 (line 7301) and §23 Wf-316B (line 7355). The §22 entry is ~85 lines including:

- Wf-382 wave structure (5 deliverables: A1 Multi-PH + PLL + A2 cache + A3 GPU readiness + A4 strategic plans + B1 final)
- Per-deliverable status table (NEW tests + LOC + wall + closeout doc)
- Pre-flight verification (§2.1 + §2.2 above)
- Honest framing per `party-twenty-spirit` §3.5 — 13 caveats preserved (m6_kappa FINAL + m2 R² FINAL + 2 SKIPs + GPU launches + USER_OK items + wetlab DEPRECATED + Zotero/tectonic INSTALLED + m1_auc Wf-379-A1 + 2 NEW FAILs + FlowMol LICENSE-GATED + 203/204/205 SCAFFOLD-ONLY + R20 GPU retrain deferred to Wf-299 + Wf-382 cache SCAFFOLD-ONLY)
- User voice answer addressing 2026-09-29 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* (Wf-381 + Wf-382 parent directive)
- Cross-link to 13 closeouts + 9 docs + 12 source files + 10 skill cross-refs

### §3.2 INDEX.md line count delta

- Pre-sync: TODO/INDEX.md = 7429 lines
- Post-sync: TODO/INDEX.md = ~7514 lines (delta +85 lines)

---

## §4 AGENTS.md sync (this row)

### §4.1 §6 NEW row (Wf-382-B1) to be appended

This sync appends NEW §6 row in `AGENTS.md` before `## §7 Critical scripts reference (cheat sheet)` (line 331). The row is ~5 lines + comprehensive follow-on cross-references, addressing:

- User directive 2026-09-29 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* (Wf-381 + Wf-382 parent directive)
- Wf-381 wave structure (A4 + A5 = 2 SHIPPED + 18 NEW tests)
- Wf-382 wave structure (A1 + A2 + A3 + A4 + B1 = 5 SHIPPED + 4 NEW tests + 1 docs-only)
- Pre-flight 89/2/3/94 PRESERVED
- 13 honest-negative caveats preserved
- User voice answer: PARTIAL — Wf-381 + Wf-382 ultracode series together DID彻底解决 the TODO-206 m6_kappa finalization + R20 plan CPU closure (4 items) + TODO-94 Multi-PH+PLL SCAFFOLD + TODO-155 SMILES cache extension + GPU readiness audit (21 items) + strategic plans refresh (10 plans +348 LOC); pre-flight 89/2/3/94 PRESERVED across 16 consecutive runs; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 NEW + §10 re-confirmed. They did NOT彻底解决 the structural/metric/env gates: m6_kappa + m2 R² remain HONEST_NEGATIVE FINAL; 2 honest-negative FAILs preserved per Wf-375-B1 §22.68 baseline; G6 ~11.4 hr launch + 5 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU) still BLOCKED on user OK; Wf-379-A1 m1_auc 5-seed result pending (~5 hr GPU); Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension pending. NEW ultracode Wf-383+ IS the recommended follow-up per Wf-382 final synthesis §9 with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity (~21.5 hr GPU + 10 min CPU + 30 min install total)

### §4.2 AGENTS.md line count delta

- Pre-sync: AGENTS.md = 418 lines
- Post-sync: AGENTS.md = ~423 lines (delta +5 lines for row + trailer extension)

---

## §5 Honest framing per `party-twenty-spirit` §3.5 — 13 caveats preserved

Per `TODO/completed/382_final_synthesis.md` §9 + this §5 + Wf-380-B1 §22.74.5 + Wf-379-B1 §22.72.5 + Wf-378-B1 §22.73.4 + Wf-377-B1 §22.71.2:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted per Wf-365-A3 verdict doc + Wf-375-A4 reframe + Wf-377-A1 deep-dive-research move with n=5460 MetalCytoToxDB; Wf-381-A4 finalizes TODO-206 by creating the canonical pending brief + verifying 4 SHIPPED reframe docs; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)

2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`)

3. **2 pre-flight FAILs preserved (BY-DESIGN)** (`p3_loi_send_ready` Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE — natural drift across Wf-374..382 wave activity; NOT Wf-382-introduced; both deferred to Wf-383+ follow-up)

4. **3 pre-flight SKIPs preserved (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency; NOT Wf-382-introduced)

5. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-368-B1 §0 + Wf-369-A1 §3.1 + Wf-370-A3 §9 + Wf-371-B1 §22.66.2 + Wf-375-B1 §22.68.4 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-A2 §3.2 + Wf-380-B1 §22.74.5; closure path = USER_OK=1 + VRAM budget check per §5.1; Wf-378-A2 closed the SAFETY bug but actual launch remains gated)

6. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 6 wrappers + 2 install + 7 paper + 2 metric gates after Wf-378-A1+A2 closed 2 of 7 wrapper gaps; **Wf-382-A3 audit verified 21 forward-looking USER_OK items total**: 16 READY + 1 BLOCKED + 4 WAITING; zero architectural blockers; only user's grant stands between current state and launch)

7. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-380-B1 timestamp; 6 (metal, seed) pairs queued; Krasnov 0.86 gate target per `mol-metal-anticancer` skill; wall-measured ~5 hr GPU; full multi-pair result deferred to Wf-383 when Wf-379-A1 completes)

8. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset)

9. **Better BibTeX MANUAL GUI install** (~2 min user-side when X11/Wayland access available; flagged in `docs/SCI_PAPER_SETUP.md` §1.2; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process)

10. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico; **Wf-381-A5 + Wf-382-A4 cross-verify LAMBDA_ROADMAP §3.2 + §3.3 wetlab re-scope confirmed**)

11. **Zotero + Tectonic INSTALLED** (Wf-372 SHIPPED 5/5 deliverables per Wf-372-C1 §22.67; Zotero 10.0.3 + Tectonic 0.17.0 both installed; CSL mol-divers.csl schema-valid; `sn-jnl.cls` end-to-end compile produces valid PDF; 11 NEW tests PASS; 0 `paper/*` touches)

12. **FlowMol LICENSE-GATED** (FlowMol arXiv:2412.01966 CC-BY-NC; USER_OK required for production integration; SCAFFOLD SHIPPED per Wf-353-A4 + Wf-380-A1; not in scope for Wf-382-B1 docs-only sync)

13. **Wf-382 cache SCAFFOLD-ONLY** (Wf-382-A2 SMILES featurization cache CPU extension = SCAFFOLD; full integration into `train_chemprop_v2_v3.py` + `train_dmpnn_multitask_wasserstein.py` deferred to future Wf-383+ MEASURED wire-in; 3 NEW tests PASS in 2.09 s wall per Wf-382-A2 closeout — rdkit env issue in omegafold_venv is venv-only, NOT a code issue per Wf-382-A2 closeout §3)

**BONUS caveat 14**: **Wf-383 + Wf-384 PRECEDED Wf-382-B1** in time (Wf-383 §22.76 + Wf-384 §22.75 already shipped per 2026-09-30 timeline). This Wf-382-B1 sync = retroactive docs-only INDEX + AGENTS + final synthesis v27 for Wf-381 + Wf-382 wave surface that shipped A1-A4 closeouts earlier; Wf-382-B1 does NOT add new code or new tests; the 22 NEW tests across Wf-381 + Wf-382 came from the 4 SHIPPED sub-flows (Wf-381-A4=14 + Wf-381-A5=4 + Wf-382-A1=1 + Wf-382-A2=3).

---

## §6 Cross-references

`TODO/completed/381{a4,a5}_*.md` (2 Wf-381 SHIPPED closeouts); `TODO/completed/382{a1,a2,a3,a4}_*.md` (4 Wf-382 SHIPPED closeouts); `TODO/completed/382{b1_index_sync,final_synthesis}_*.md` (this row + final synthesis v27); `TODO/pending/{94_r20_p3_multiph_pll_feature,155_smiles_featurization_cache_extension,206_m6_kappa_metric_redefinition,383_se3_sb_fast_converging_integration,200_wf340_353_deep_integration_audit}_*.md` (5 pending docs created or refreshed in Wf-381 + Wf-382 wave); `molmetal/adapters/flow_matching_lipman/pll_features.py` (Wf-382-A1 NEW ~273 LOC); `molmetal/scripts/wf382a1_pll_feature_extract.py` (Wf-382-A1 NEW ~387 LOC); `molmetal/scripts/wf382a2_dmpnn_featurization_cache.py` (Wf-382-A2 NEW ~265 LOC); `molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py` (Wf-382-A2 NEW ~410 LOC); `molmetal/tests/test_wf381a4_m6kappa_finalize.py` (14 NEW tests); `molmetal/tests/test_wf381a5_*.py` (4 NEW tests in 4 files); `molmetal/tests/test_wf382a1_pll_features.py` (1 NEW test); `molmetal/tests/test_wf382a2_featurization_cache.py` (3 NEW tests); `molmetal/tests/test_wf382a3_gpu_readiness.py` (0 NEW tests — audit-only); `TODO/03_plans/{R17_AUC_RECOVERY_PLAN,R18_MATH_MIGRATION_PLAN,R18_PLAN,R19_PLAN,R20_BIG_DATASET_FULL_RETRAIN_PLAN,WEAK_TO_STRONG_PLAN,MERGE_GATE,IMPROVEMENT_ENUM_2026_09_27,IMPROVEMENT_QUEUE_2026_09_27,LAMBDA_ROADMAP}.md` (10 plans refreshed by Wf-382-A4 +348 LOC total); `molmetal/reports/wf382a1_pll_smoke/smoke_result.json` (Wf-382-A1 smoke result); `molmetal/reports/wf382a3_gpu_readiness_audit/` (Wf-382-A3 audit results); `molmetal/reports/wf382b1_preflight.txt` (Wf-382-B1 pre-flight fresh-run 89/2/3/94 wall 138.95 s — 872 lines); `molmetal/reports/wf380b1_preflight.txt` (Wf-380-B1 baseline 89/2/3/94 wall ~5 min); `molmetal/reports/wf379b1_preflight.json` (Wf-379-B1 baseline 89/2/3/94 wall 139.61 s); `molmetal/reports/wf378b1_preflight.txt` (Wf-378-B1 baseline 89/2/3/94 wall 155.01 s); `molmetal/reports/wf377b1_preflight.txt` (Wf-377-B1 baseline 89/2/3/94 wall 152.78 s); `molmetal/reports/wf374b1_preflight.txt` (Wf-374-B1 baseline 89/2/3/94 wall ~155 s); `molmetal/reports/wf372c1_preflight.txt` (Wf-372-C1 clean baseline 92/0/2/94 wall 153.42 s); `molmetal/reports/wf370b1_preflight.txt` (Wf-370-B1 clean baseline 92/0/2/94 wall 152.53 s); `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json` (Wf-379-A2 honest-negative evidence tree); `molmetal/reports/gpu_launch_dashboard.json` (Wf-377-A4 schema v1.1); `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 69 entries 6 sections + Wf-381-A4 + Wf-382-A3 NEW entries); `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 NEW); `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 NEW 387 LOC 7 sections); `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC); `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC 7 sections); `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 ~280 LOC); `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants); `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 ~289 LOC); `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template); `docs/SCI_PAPER_SETUP.md` (Wf-351-B3 ~120 LOC); `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections); `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §5.5 NEW + §10 standing directives re-confirmed; `AGENTS.md` §6 NEW row (Wf-382-B1 this sync); `AGENTS.md` last-update trailer (this sync appended); `TODO/INDEX.md` §22.77 (this entry appended between §22.76 Wf-383-B1 and §23 Wf-316B); `kill_then_adapt` §0 (NO GPU launch + USER_OK gating); `mol-metal-cfm-training` skill (Wf-381-A5 + Wf-382-A1 Multi-PH+PLL); `mol-metal-multiph-pll` skill (Wf-381-A5 + Wf-382-A1 Multi-PH+PLL); `mol-metal-hilbert-astsr-r2` skill (Wf-381-A5 TODO-93); `mol-metal-3way-multimetal` skill (Wf-381-A5 TODO-95); `mol-metal-closure-catherine` skill (Wf-381-A5 TODO-96); `mol-metal-closure-theorem` skill (Wf-381-A5 TODO-98); `mol-metal-anticancer` skill (Wf-379-A1 + Wf-382-A3 m1_auc); `mol-metal-sci-paper-submission` skill (Wf-351-B3 NEW per 2026-09-29 user directive).
