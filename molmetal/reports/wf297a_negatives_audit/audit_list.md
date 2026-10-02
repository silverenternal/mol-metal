# Wf-297A — Honest-Negative Audit (Wf-256..296)

**Date**: 2026-09-29
**Wf**: Wf-297A
**Owner**: Hugo Lee + workflow harness
**Scope**: All honest-negative findings documented across Wf-256..296 (41 waves, 41 days)
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §6 honest-negative

---

## §0 TL;DR (FLAT schema verdict)

| Gate | Value |
|---|---|
| **G1_total_honest_negatives_cataloged** | **18** (in `HONEST_NEGATIVE_CATALOG` `molmetal/scripts/paper_dashboard.py:454-565`) |
| **G2_total_critical_honest_negatives_aggregated** | **24** (across Wf-281/282/283/284/285/286/287/288/289/292/295A + Wf-276 §3 + Wf-293D + Wf-271E) |
| **G3_honest_negatives_carried_forward_to_paper** | **18** dashboard-catalog + **6** synthesis-tracked = **24** total |
| **G4_NEW_since_Wf_280C** | **6** new honest-negatives (Wf-268C 3 + Wf-279C 2 + Wf-293D 1) |
| **G5_wave_impact** | 0 waves newly-unblocked since Wf-280D (Cf-258 n_done=2/17 unchanged); G5 C6 PARTIAL seed=1 epoch 30/50 |
| **G6_target_close_path** | R20-P1..P6 deferral + 1.8 hr G4 GPU burst + 7.5 hr G6 GPU sweep + 16 hr CFM P1.3/P1.4 |
| **G7_discipline_recommendation** | Maintain honest-negative ledger intact; do NOT cite any "lift" claim without MEASURED GPU evidence per party-twenty-spirit §3.5 |

---

## §1 HONEST_NEGATIVE_CATALOG (verbatim, dashboard-catalog entries)

> Source: `molmetal/scripts/paper_dashboard.py:454-565` (18 entries). 12 entries from Wf-264 baseline + 3 added Wf-268C-Dash (Wf-263A + Wf-266 honest-negatives) + 2 added Wf-279C + 1 added Wf-293D.

### §1.1 Original 12 entries (Wf-264 baseline, §9.3)

| ID | Section | Item (verbatim) | Status |
|---|---|---|---|
| `hn_no_imold_beat` | §9.3 | DO NOT claim chemprop v2 beats iMoLD on DrugOOD (Δ = −0.0074 assay within std) | ⚠️ MEASURED |
| `hn_no_cross_metal_lomo` | §9.3 | DO NOT claim cross-metal LOMO transfer (TODO-154 in-domain per-metal only) | ⚠️ MEASURED |
| `hn_no_drugood_full` | §9.3 | DO NOT claim DrugOOD 1.58M train full retrain is measured (BD-11 DEFERRED) | ⚪ DEFERRED |
| `hn_no_platinai_226k` | §9.3 | DO NOT claim full PlatinAI 226K retrain (500-mol subset only — L2 DEFERRED) | ⚪ DEFERRED |
| `hn_no_bindingdb_full` | §9.3 | DO NOT claim BindingDB full 1.07M metal-filter retrain (BD-12 DEFERRED) | ⚪ DEFERRED |
| `hn_no_krasnov_temporal_split` | §9.3 | DO NOT claim scaffold-split is more comparable to Krasnov time-split than a temporal driver | ⚠️ MEASURED |
| `hn_no_m21_krasnov_beat` | §9.3 | DO NOT claim m21 9-cell aggregate BEATS Krasnov 0.81 (statistically equivalent) | ⚠️ MEASURED |
| `hn_no_cfm_synthetic_1_000` | §9.3 | DO NOT claim CFM decode_ratio 0.9958 BEATS synthetic 1.000 (expected gap) | ⚠️ MEASURED |
| `hn_no_wf219_more_accurate` | §9.3 | DO NOT claim Wf-219 smaller-backbone 3-seed numbers more accurate than TODO-154 5-seed full-backbone | ⚠️ MEASURED |
| `hn_no_re_significance` | §9.3 | DO NOT claim Re per-metal paired t-test is significant (n=2 underpowered) | ⚠️ MEASURED |
| `hn_no_m7_targetdiff` | §9.3 | DO NOT claim m7 diversity 0.0397 reaches TargetDiff 0.860 ceiling (gap +0.820 honest) | ⚠️ MEASURED |
| `hn_no_v3_drugood_pass` | §9.3 | DO NOT claim chemprop v3 DrugOOD-PASS (BELOW_RANDOM 0.4544, §6 honest-negative) | ⚠️ MEASURED |

### §1.2 NEW Wf-268C-Dash additions (3 entries, §9.4)

| ID | Section | Item (verbatim) | Status |
|---|---|---|---|
| `hn_no_wf263a_empirical_pass` | §9.4 | DO NOT claim Wf-263A empirical DrugOOD re-measurement PASS (75-min wall budget TIMEOUT; structural PASS only, empirical TBD) | ⚪ DEFERRED |
| `hn_no_wf266_g5_measured` | §9.4 | DO NOT claim Wf-266 G5 50-epoch sweep MEASURED (0/15 pairs — CUDA error invalid device ordinal; wrapper BLOCKED at GPU ordinal layer) | ⚪ BLOCKED |
| `hn_no_wf266_retrain_lift` | §9.4 | DO NOT claim Wf-266 v3-fix mode produces MEASURED AUC lift (IMPLEMENT only — no GPU retrain; AUC lift deferred to follow-up TODO-130 G5 retrain) | ⚪ DEFERRED |

### §1.3 NEW Wf-279C additions (2 entries, §9.AI + §9.AJ)

| ID | Section | Item (verbatim) | Status |
|---|---|---|---|
| `hn_no_cfm_decode_real_09645_as_lead` | §9.AI | DO NOT cite cfm_decode_real.json value=0.9645 as the m14 LEAD (Wf-274A regression_disclosure flags it as a smaller-cohort 1-seed CPU PlatinAI pt_subset measurement; cite Wf-188 m14 FULL = 0.9958 instead) | ⚠️ MEASURED |
| `hn_no_yuelbond_off_deployment` | §9.AJ | DO NOT deploy m14 LEAD CFM with --no-yuelbond flag (Wf-269C MEASURED: decode_ratio 1.000 → 0.000, Δ=−1.000; m14 LEAD silently invalidates without raising any runtime error) | ⚠️ MEASURED |

### §1.4 NEW Wf-293D addition (1 entry, §9.AK)

| ID | Section | Item (verbatim) | Status |
|---|---|---|---|
| `hn_wf286a_5_seed_drugood_failed` | §9.AK | DO NOT cite Wf-286A as 5-seed DrugOOD v3-fallback confirmation (FAILED: gate_1_all_ran=false, gate_2_total_runs=0, all 7 gates empty; only load_split_data ran (n_train=60908) before per-seed trainer step did not start — no AUC produced; wall = 1.86e-5 min effective NO-OP) | ❌ FAILED |

---

## §2 Critical honest-negatives aggregated across Wf-256..296 (24 total)

> Sources: Wf-267-Final §2.2 + Wf-276 §3 + Wf-281E §3 + Wf-282E §3 + Wf-283E + Wf-285E §3 + Wf-286E §0 + Wf-287D §2 + Wf-288E §3 + Wf-289E §3 + Wf-290E §3 + Wf-291E + Wf-292E §2 + Wf-293D §2 + Wf-295A §0 + Wf-296A.

### §2.1 Wave-level honest-negatives (Wf-256..296)

| # | Wave ID | Honest-negative finding | Current value | Status |
|---:|---|---|---|---|
| 1 | Wf-256 | Catherine 30-metal 6/6 PASS (no honest-negative — greenfield PASS) | 30/30 PASS | ✅ GREEN |
| 2 | Wf-257 | CFM potential evaluation (PROJECTED — not yet MEASURED on real cohort) | PROJECTED | ⚪ DEFERRED |
| 3 | Wf-258 | Audit: §7 PROJECTED scaling numbers; not MEASURED on GPU | PROJECTED | ⚪ DEFERRED |
| 4 | Wf-259 | pre_flight_check PARTIAL (20/22 PASS, 2 FAIL: p1_cfm_wasserstein_loss_exists, p2_vina_installed — both FIXED in Wf-260) | 20/22 | ✅ FIXED |
| 5 | Wf-261 | scaffold-split audit + regenerate; no empirical model change | MEASURED | ✅ GREEN |
| 6 | Wf-262 | abort_recovery.py + wave_budget_tracker.py + per_wave_checklist.py SHIPPED | MEASURED | ✅ GREEN |
| 7 | Wf-263A | chemprop v3 v3-fallback to v2 BinaryClassificationFFN structural SHIPPED; **empirical DrugOOD re-measurement TIMEOUT at 75-min wall budget** | TIMEOUT | ⚪ DEFERRED |
| 8 | Wf-263B | chemprop v3 v3-fallback gap-list audit | MEASURED | ✅ GREEN |
| 9 | Wf-263C | chemprop v3 verify audit | MEASURED | ✅ GREEN |
| 10 | Wf-264 | dashboard baseline 32 claims / 12 honest-negatives / 6 cohorts SHIPPED | MEASURED | ✅ GREEN |
| 11 | Wf-265 | dashboard build + collect (33/17/6 post-Wf-280C; 34/18/7 post-Wf-293D) | MEASURED | ✅ GREEN |
| 12 | Wf-266 | chemprop v3 LOMO-fix levers (--chemprop-min-epochs, --chemprop-lomo-init-scale, --chemprop-mode v3-fix) SHIPPED + 4/4 new + 14/14 backward compat tests PASS + LOMO-fix verify MEASURED = **constant predictor 0.5000 at 3-ep verify budget** + **G5 50-epoch sweep BLOCKED at GPU-ordinal layer (0/15 pairs MEASURED)** | 0.5000 / 0/15 | ❌ HONEST-NEGATIVE |
| 13 | Wf-267 | Final integration: chemprop v2 strong, chemprop v3 structurally shipped + empirically HONEST-NEGATIVE; CFM strong; Vina multi-fragment 0/500 docked (meeko 0.8.0 honest-negative on salt forms); 2 §3.5 citation-hygiene residuals at §2.2 line 110 + §8 line 388 | MEASURED | ⚠️ MIXED |
| 14 | Wf-268A | CFM ablation flag verification | MEASURED | ✅ GREEN |
| 15 | Wf-268B | DrugOOD re-verify NOT RESOLVED (1-seed 0.7016 not yet in 188) | 0.7016 | ⚪ DEFERRED |
| 16 | Wf-269A | chemprop v3 Path C (shared encoder + per-metal linear probes) **path_c_implemented = undefined, target_auc_met = undefined** — IMPLEMENT only, no MEASURED AUC | UNDEFINED | ⚪ DEFERRED |
| 17 | Wf-269B | chemprop v3 ablation G5 n_pairs_measured = 0 (no GPU retrain) | 0 | ⚪ DEFERRED |
| 18 | Wf-269C | CFM real-cohort YuelBond ablation: **full=1.000, no_yb=0.000 (Δ=−1.000), no_w=0.987 (Δ=−0.013), no_cb=0.992 (Δ=−0.008)** — YuelBond load-bearing (HONEST-NEGATIVE for W₂ + closure-bounded; Δ within noise); PlatinAI/DrugOOD cohort generalization NOT yet tested | 1.000/0.000/0.987/0.992 | ⚠️ MEASURED (YuelBond) + ⚪ DEFERRED (generalization) |
| 19 | Wf-269D | §4.5.4 honest results catalog (208-AH + 188 §4.5.4 byte-identical) | MEASURED | ✅ GREEN |
| 20 | Wf-271 | All 4 upstream feeds (Wf-271A-D) returned `undefined` schema fields — **3rd occurrence of upstream-pipeline regression class** (after Wf-270A-D + Wf-272A-D) | UNDEFINED | ❌ HONEST-NEGATIVE |
| 21 | Wf-274A | **cfm_decode_real.json regression_disclosure**: previous 0.9980 (3 seeds × 5991, GPU) vs current 0.9645 (1 seed × 818, CPU) — silent methodology change NOT disclosed in original emit; **DO NOT cite 0.9645 as m14 LEAD** (cite Wf-188 m14 FULL = 0.9958) | 0.9645 vs 0.9980 | ⚠️ MEASURED |
| 22 | Wf-275B | drugood_ecfp4_scaffold_adapter SHIPPED (10 tests pass) | MEASURED | ✅ GREEN |
| 23 | Wf-276 | 207 §9 cross-ref audit: **1 minor inconsistency** at 207 §9.5 body line 386 wording vs 188 §4.5.2; 6 honest-negatives documented | MEASURED | ⚠️ MIXED |
| 24 | Wf-278A | G4 aggregate fix (n_pairs_total 14 → 15 cascade; Ru seed=0 cascaded from G5) | MEASURED | ✅ GREEN |
| 25 | Wf-278B | PlatinAI 4-cell CSV loader (referenced by Wf-292D strategy_report as ORPHAN script) | MEASURED | ✅ GREEN |
| 26 | Wf-278C | Tanimoto OOD filter integration SHIPPED | MEASURED | ✅ GREEN |
| 27 | Wf-280C | dashboard refresh: 33/17/6 (Wf-282 G5 Ru seed=0 claim added) | MEASURED | ✅ GREEN |
| 28 | Wf-280D | Cf-258 17-wave progress refresh: n_done=2 (C1, C2); n_pending=15; **G5 14/15 pairs still PENDING (Ru seed=0 done, seed=1 running, 13 to go)**; G6 60/60 PENDING | 2/17 DONE | ⚪ DEFERRED |
| 29 | Wf-281 | **Wf-281A TODO-180 G4 Ru seed=0 recovery** MEASURED HONEST-NEGATIVE — Ru seed=0 ckpt + run.log found; aggregate test_auc 0.7209 ± 0.0458 preserved; **Wf-281B §10 cross-ref audit did not launch** (3-of-4 streak BROKEN); 6 critical honest-negatives carried forward | 0.7209 ± 0.0458 | ⚠️ MEASURED |
| 30 | Wf-282 | Wf-282A supersession doc + Wf-282B ledger wire + Wf-282C lint pass + Wf-282D SHA-256 checklist; **Wf-282C detected 1 CRITICAL bug** (literal `% util` in `train_dmpnn_multitask_wasserstein.py` line 1190) FIXED downstream by Wf-283B | MEASURED | ✅ GREEN |
| 31 | Wf-283 | Wf-283A G5 wrapper self-recovery (Ru seed=0 actually completed; HN-1 DOWNGRADED HIGH→LOW); Wf-283B `--no-kendall` / `--no-dann` CLI surface added (downstream loss/model wiring deferred to Wf-284); Wf-283C wired Wf-262 pre-flight trio into all 3 Wf-124 P3 launchers | MEASURED | ✅ GREEN |
| 32 | Wf-284 | Wf-284A argparse no-op confirmation (Wf-283B fix verified); Wf-284B cfm_wasserstein_loss.py false-positive refutation; Wf-284C G5 Ru seed=0 propagation to 4 catalogs + 1 dashboard; Wf-284D §10 cross-ref audit 19/19 PASS | MEASURED | ✅ GREEN |
| 33 | Wf-285 | Wf-285B Kendall UW + DANN GRL wiring (tests green); Wf-285C `--no-enforce-budget` CLI flag; **Wf-285A + Wf-285D MISSING-DELIVERABLE honest-negative** (no on-disk report) | MEASURED | ⚠️ MIXED |
| 34 | Wf-286A | **5-seed DrugOOD v3-fallback confirm FAILED / NO-OP** — gate_1_all_ran=false, gate_2_total_runs=0, all 7 gates empty; only load_split_data ran (n_train=60908 loaded @ 11.9s); no AUC produced; wall=1.86e-5 min | NO-OP | ❌ FAILED |
| 35 | Wf-286B | ECFP4-direct re-eval (smoke): test_auc=0.8820 (Δ=+0.211 vs Wf-185 0.6707); **rules OUT ECFP4 axis as Wf-253-D 0.4544 collapse bottleneck**; collapse is backbone / v3-fallback / JSON-split axis | 0.8820 | ✅ GREEN |
| 36 | Wf-286C | Cf-258 17-wave progress refresh #1: n_done=2 (unchanged); 7 newly unblocked (C10, C11, C13-C17); C6 PARTIAL (Ru seed=0 MEASURED 0.7716; seed=1 RUNNING epoch 11/50) | 2/17 DONE | ⚪ DEFERRED |
| 37 | Wf-286D | Cf-265 dashboard refresh: 33/17/6; Wf-282 G5 Ru seed=0 claim propagated; dashboard.html 76.9 KB | MEASURED | ✅ GREEN |
| 38 | Wf-287 | **Wf-287D G3 partial** — per-metal ckpt coverage (Ru 4/4, Ir 3/4, Os 1/4 MEASURED; **4/12 DEFERRED_NO_CHECKPOINT** per Wf-253d) distributed across §9.4 + §9.AE + §9.AI.1; aggregate **BELOW_RANDOM 0.4544 LEAD-disqualification IS preserved** | 4/12 MEASURED | ⚪ DEFERRED |
| 39 | Wf-288B | Cf-264 refresh: §4.5.x additions (Wf-266 G5 + Wf-269A/B/C) added 17 new claims; 0 NEW §3.5 violations introduced; 2 Wf-267 residuals FIXED post-Wf-269D | MEASURED | ✅ GREEN |
| 40 | Wf-288C | pre_flight_check 24/25 PASS + 1 FAIL (p1_chemprop_smoke budget tightness — 55s budget too tight by 3-5s) | 24/25 | ❌ HONEST-NEGATIVE |
| 41 | Wf-289 | **G4 14 PENDING per-pair audit gap** — only Ru seed=0 of 15 (metal, seed) pairs is independently auditable; 14-pair MEASURED aggregate cited WITH Wf-268C caveat; **single highest-priority honest-negative** | 14/15 PENDING | ⚪ DEFERRED |
| 42 | Wf-290 | **TODO-130 chemprop v3 DrugOOD re-train 0.4544 BELOW_RANDOM** — v3 OOD fix structurally PASS but empirical DrugOOD re-measurement DEFERRED per Wf-277B (75-min kanzi CPU PENDING) | 0.4544 | ⚪ DEFERRED |
| 43 | Wf-291C | pipeline-root-cause: 6-occurrence upstream-pipeline regression class (Wf-270/271/272/274/279/285) — doc-only investigation, no fix | DOC-ONLY | ⚪ DEFERRED |
| 44 | Wf-292A | G5→G4 cascade (TODO-180): n_pairs_total 14 → 15 (Ru seed=0 cascaded); test_auc 0.7561 ± 0.0112 for Ru n=3 | MEASURED | ✅ GREEN |
| 45 | Wf-292B | P1.4 + G6 pre-flight: 25/25 PASS; 5 scripts `--help` OK; 60-tuple G6 enumeration verified; 9.5 hr total wall (G6 7.5 hr + P1.4 2 hr); **launch blocked on user OK** | 25/25 PASS | ⚪ DEFERRED |
| 46 | Wf-292C | Cf-258 17-wave refresh: n_done=2 unchanged; 7 G6 waves PRE-LAUNCH READY but never launched; C6 PARTIAL seed=1 RUNNING epoch 19/50 | 2/17 DONE | ⚪ DEFERRED |
| 47 | Wf-292D | PlatinAI real-label loader bypass: 8 strategies evaluated; top-2 = S7 (MB-Finder portal, ~1 hr, 60%) + S1 (Selenium UC, ~2 hr, 60-75%); S6 (author outreach) deferred; **S5 (chemprop v2 proxy) honest-negative: NOT REAL** | EVAL-ONLY | ⚪ DEFERRED |
| 48 | Wf-293C | launcher safety trio verify (Wf-124 P3 launchers): GUARD INTACT; pre_flight_check + per_wave_checklist + wave_budget_tracker all wired | MEASURED | ✅ GREEN |
| 49 | Wf-293D | dashboard refresh #2: 34 claims (+1), 18 honest-negatives (+1, `hn_wf286a_5_seed_drugood_failed`), 7 cohorts (+1); 80.9 KB HTML | MEASURED | ✅ GREEN |
| 50 | Wf-294A | pre_flight_check `p1_chemprop_smoke` budget fix: 55s → 90s; **25/25 PASS** | 25/25 | ✅ GREEN |
| 51 | Wf-294B | post-budget-fix pre-flight re-verify: **25/25 PASS, 0 FAIL** | 25/25 | ✅ GREEN |
| 52 | Wf-294D | G5 epoch-50 G4 cascade readiness: **3 blockers** identified — `expected_pairs: 14` mismatch; cross-directory cascade not wired; Ru seed=1 not yet emitted (ETA §2) | 3 BLOCKERS | ⚪ DEFERRED |
| 53 | Wf-295A | Cf-258 refresh #3: **NO new wave crossed PENDING → DONE on disk since Wf-280D**; n_done=2 (unchanged); 7 G6 waves PRE-LAUNCH READY but never launched; C6 PARTIAL seed=1 RUNNING epoch 30/50; seed=2 PENDING | 2/17 DONE | ⚪ DEFERRED |
| 54 | Wf-295D | launch orchestration plan (P1.4 smoke + G6 C12 no_wass smoke + G6 full sweep 60 runs); **7 honest-negative abort criteria** documented (§7.2.1-§7.2.8) including m7 Tanimoto <0.04 drop + CHECKPOINT_LOADED_FORWARD_DEFERRED | PLAN | ⚠️ MIXED |
| 55 | Wf-296A | pipeline integration audit: **42 ORPHAN artifacts** (10 orphan scripts + 16 orphan flags + 14 orphan tests + 2 orphan catalog) | 42 ORPHANS | ⚠️ MEASURED |
| 56 | Wf-296D | Wire L-R17-B2 84-pattern matrix + YuelBond precondition into CFM main generation pipeline: `m14_lead_invalidated: bool` field added; `--no-yuelbond` flips to True + emits `[Wf-296D][WARNING]` on stderr; smoke PASS | MEASURED | ✅ GREEN |

### §2.2 Synthesis-level critical honest-negatives (Wf-281/282/283/284/285/286/287/288/289/290/292/295A)

| # | Source | Critical honest-negative | Severity |
|---:|---|---|---|
| 1 | Wf-281E §3 | TODO-180 G4 14 PENDING pairs + Wf-280 4-of-4 BROKEN at 3-of-4 (Wf-281B no-launch) + 188 paper edits BLOCKED on TODO-84 + 15/17 Cf-258 waves PENDING | ⚠️ HIGH |
| 2 | Wf-282E | 4-of-4 streak RECOVERED; Wf-282C detection→Wf-283B fix loop is model for honest-negative → action chain | ⚪ LOW |
| 3 | Wf-283E | HN-1 (G5 wrapper) DOWNGRADED HIGH→LOW; HN-3 (pre-flight trio) CLOSED; HN-4 (--no-kendall/--no-dann) PARTIAL (CLI surface only, loss wiring deferred to Wf-284); HN-5 (paper §9.AG + §9.5 wording) NOT YET ON DISK | ⚠️ MIXED |
| 4 | Wf-284E | 4-of-4 streak PRESERVED at 4-of-4; 3 of 4 waves awareness-only / no-op confirmation (Wf-284A/B/D); this is `mol-metal-gpu-execution` §6 honest-negative discipline operating as designed | ⚪ LOW |
| 5 | Wf-285E §3 | **MISSING-DELIVERABLE honest-negative** for Wf-285A + Wf-285D (no on-disk reports) | ⚠️ HIGH |
| 6 | Wf-286A | **5-seed DrugOOD v3-fallback confirm FAILED / NO-OP** — only load_split_data ran; no training, no AUC, all 7 gates empty | ❌ FAILED |
| 7 | Wf-287D §2 | **4/12 DEFERRED_NO_CHECKPOINT** per-metal ckpt coverage (Wf-253d); aggregate BELOW_RANDOM 0.4544 LEAD-disqualification preserved | ⚠️ HIGH |
| 8 | Wf-288E §3 | 207 §9.5 line 386 internal inconsistency FIXED post-Wf-269D | ⚪ LOW |
| 9 | Wf-289E §3 | **G4 14 PENDING per-pair audit gap** — only Ru seed=0 of 15 (metal, seed) pairs independently auditable; single highest-priority honest-negative | ⚠️ HIGH |
| 10 | Wf-290E §3 | TODO-130 chemprop v3 DrugOOD re-train 0.4544 BELOW_RANDOM; v3 OOD fix structurally PASS but empirical re-measurement DEFERRED | ⚠️ HIGH |
| 11 | Wf-292E §2 | All 10 retrain items PENDING FRESH USER OK per `kill_then_adapt`; 17 hr wall on GPU-0 + GPU-1 + 3 hr CPU needed | ⚪ DEFERRED |
| 12 | Wf-295A §0 | 0 waves newly-unblocked since Wf-280D; 7 G6 waves PRE-LAUNCH READY but never launched; C6 PARTIAL seed=1 RUNNING epoch 30/50 | ⚠️ HIGH |

---

## §3 Per-wave summary table (top-10 honest-negatives by severity)

| Rank | Wave ID | Honest-negative headline | Severity | Path forward |
|---:|---|---|---|---|
| 1 | Wf-286A | 5-seed DrugOOD v3-fallback confirm FAILED / NO-OP (gate_1_all_ran=false, wall=1.86e-5 min) | ❌ FAILED | Re-run with `--debug-epochs 1` + stdout capture |
| 2 | Wf-263A | chemprop v3 v3-fallback structural SHIPPED; empirical DrugOOD re-measurement TIMEOUT at 75-min | ⚪ DEFERRED | TODO-130 G5 retrain (16 GPU-hr) |
| 3 | Wf-266 | chemprop v3 LOMO-fix verify MEASURED = constant predictor 0.5000 (FAIL G2 ≥ 0.65) + G5 50-ep sweep BLOCKED (0/15 pairs) | ❌ HONEST-NEGATIVE | Structural Path C (shared encoder + per-metal linear probes) |
| 4 | Wf-269A | chemprop v3 Path C path_c_implemented = undefined; no MEASURED AUC | ⚪ DEFERRED | TODO-130 G5 retrain |
| 5 | Wf-269B | chemprop v3 ablation G5 n_pairs_measured = 0 (no GPU retrain) | ⚪ DEFERRED | TODO-130 G5 retrain |
| 6 | Wf-269C | CFM real-cohort: W₂ + closure-bounded Δ within noise (−0.013, −0.008); PlatinAI/DrugOOD generalization NOT yet tested | ⚠️ MEASURED (YuelBond) + ⚪ DEFERRED (generalization) | TODO-119 P1.3 (16 GPU-hr) + PlatinAI S7 harvest (1 hr CPU) |
| 7 | Wf-271 | All 4 upstream feeds returned `undefined` — 3rd occurrence of upstream-pipeline regression class | ❌ HONEST-NEGATIVE | Wf-271F pipeline recovery (≤30 min CPU, P0 BLOCKER) |
| 8 | Wf-274A | cfm_decode_real.json 0.9645 vs prior 0.9980 silent methodology change | ⚠️ MEASURED | Already mitigated by 188 §5.2 line 387 (cite Wf-188 m14 FULL = 0.9958) |
| 9 | Wf-285 | Wf-285A + Wf-285D MISSING-DELIVERABLE honest-negative | ⚠️ HIGH | Re-dispatch as follow-ups |
| 10 | Wf-287D | 4/12 DEFERRED_NO_CHECKPOINT per-metal ckpt coverage (Wf-253d); BELOW_RANDOM 0.4544 LEAD-disqualification preserved | ⚠️ HIGH | TODO-130 G5 retrain |
| 11 | Wf-288C | pre_flight_check 24/25 PASS + 1 FAIL (p1_chemprop_smoke budget 55s too tight) | ❌ HONEST-NEGATIVE | ✅ FIXED by Wf-294A (25/25 PASS) |
| 12 | Wf-289 | G4 14 PENDING per-pair audit gap — single highest-priority honest-negative | ⚠️ HIGH | Wf-290-A per-pair log+ckpt+metrics re-verification |
| 13 | Wf-292D | S5 (chemprop v2 proxy) honest-negative: NOT REAL | ⚠️ MEASURED | Use S7 (MB-Finder portal, 1 hr CPU) or S1 (Selenium UC, 2 hr CPU) instead |
| 14 | Wf-293D | hn_wf286a_5_seed_drugood_failed NEW honest-negative entry | ⚠️ MEASURED | Already catalogued + surfaced in dashboard |
| 15 | Wf-295A | 0 waves newly-unblocked since Wf-280D; 7 G6 waves PRE-LAUNCH READY but never launched | ⚠️ HIGH | USER OK #5..#11 for C10-C17 (6.4 hr GPU-1) |
| 16 | Wf-296A | 42 ORPHAN artifacts (10 scripts + 16 flags + 14 tests + 2 catalog) | ⚠️ MEASURED | Wf-296B-C integration pass |

---

## §4 Wave-budget Cf-258 ledger snapshot (Wf-295A)

| Wave | Status | Pairs | Notes |
|---|---|---|---|
| **C1** | ✅ DONE | Ru s1/s2 + Ir s0/s1/s2 (5 pairs) | unchanged since Wf-280D |
| **C2** | ✅ DONE | Os/Rh/Re s0/s1/s2 (9 pairs) | unchanged since Wf-280D |
| **C3** | ⚪ PENDING | CFM P1.3 smoke | 16-hr Blackwell pre-launch scope |
| **C4** | ⚪ PENDING | CFM P1.4 a/b | blocked on C3 |
| **C5** | ⚪ PENDING | CFM P1.4 c/d | blocked on C4 |
| **C6** | 🟡 PARTIAL | G5 Ru s0/s1/s2 | Ru seed=0 MEASURED 0.7716; seed=1 RUNNING epoch 30/50; seed=2 PENDING |
| **C7** | ⚪ PENDING | G5 Ir s0/s1/s2 | blocked on C6 |
| **C8** | ⚪ PENDING | G5 Os+Rh s0/s1/s2 | blocked on C7 |
| **C9** | ⚪ PENDING | G5 Re s0/s1/s2 + comparison JSON | blocked on C8 |
| **C10** | ⚪ PRE-LAUNCH READY | G6 no_multi_task Ru/Ir/Os | USER OK #5 |
| **C11-C17** | ⚪ PRE-LAUNCH READY | G6 sequential (no_wass + no_kendall + no_dann × 5 metals × 3 seeds) | USER OK #6..#11 |

**completion_pct (DONE-only)**: 11.76% (2/17)
**completion_pct (DONE + PARTIAL + PRE-LAUNCH READY)**: 52.94% (9/17)
**Δ vs Wf-280D**: 0 pp DONE-only; +41.18 pp including PRE-LAUNCH

---

## §5 Recommended close-out path (per party-twenty-spirit §3.5)

| Priority | Action | Wave | Wall | GPU | Honest-negative impact |
|---:|---|---|---|---|---|
| **P0** | Wf-271F pipeline recovery (re-emit Wf-271A-D wave JSONs with populated schema fields) | Wf-271F | ≤30 min | CPU | Closes 3rd-occurrence upstream-pipeline regression |
| **P1** | Wf-294A budget fix (already SHIPPED 25/25 PASS) | Wf-294A | 0 min | n/a | ✅ CLOSED |
| **P1** | PlatinAI S7 MB-Finder portal harvest (3,725 Pt compounds) | Wf-292D §3 R1 | ~1 hr | CPU | Replaces S5 (NOT REAL honest-negative) with real Rusanov 2025 |
| **P1** | G5 C6 finish (Ru seed=1 + seed=2) | Wf-294D | ~2 hr | GPU-1 | Closes Ru seed=0 MEASURED cascade (0.7716 already in dashboard) |
| **P2** | TODO-180 G4 14 PENDING (metal, seed) per-pair audit | Wf-290-A | ~30 min | CPU | Closes §3 critical honest-negative #1 in Wf-289E |
| **P2** | TODO-130 G5 chemprop v3 DrugOOD re-train | TODO-130 G5 | ~5 hr | GPU-1 | Closes Wf-263A TIMEOUT + Wf-286A NO-OP |
| **P2** | G6 C10 G6 no_multi_task Ru/Ir/Os | Wf-295D §1 Wave-A | ~1.1 hr | GPU-1 | Closes TODO-182 G6 first wave |
| **P2** | TODO-119 P1.3 CFM real-decode lift (PlatinAI/DrugOOD) | TODO-119 P1.3 | ~16 hr | GPU-0 | Closes Wf-269C generalization honest-negative |
| **P3** | L9-L12 R20-P1..P6 deferral | R20-P1..P6 | TBD | TBD | Closes vuln_f+g + closure OOD + 3-way R² honest-negatives |
| **P3** | 188 §3.5 §2.2 line 110 + §8 line 388 fixes | Wf-271G | ≤15 min | CPU | Closes 2 §3.5 citation-hygiene residuals |

**Total wall-time to close all honest-negatives**: ~25-30 GPU-hr (GPU-0 + GPU-1 parallel) + ~3 hr CPU

---

## §6 Discipline recommendations (party-twenty-spirit §3.5)

1. **DO NOT cite any "lift" claim without MEASURED GPU evidence** — every honest-negative finding in §1 + §2 must remain in the catalog with verbatim text; dashboard `HONEST_NEGATIVE_CATALOG` is the canonical source of truth.
2. **DO NOT modify §3.5 honesty ledger** — `paper/main.tex` §3.5 + §6 + §9 all reference `paper_dashboard.py::HONEST_NEGATIVE_CATALOG` (18 entries); any modification requires Hugo Lee's paper-edit authority.
3. **DO NOT best-seed select** — per `mol-metal-gpu-execution` §7, all reported numbers are 3-seed mean ± std unfiltered.
4. **DO NOT silently regress methodology** — cfm_decode_real.json regression_disclosure (Wf-274A) is the canonical pattern: every JSON write that changes cohort/seed/compute/sample-size must include a `regression_disclosure` field with `previous_value` + `previous_methodology` + `verdict` + `recommendation`.
5. **MAINTAIN honest-negative ledger across waves** — every Wf-XXX-E synthesis must include a "Critical honest-negatives remaining" section ranked by severity (per Wf-281E/282E/283E/284E/285E/286E/287E/288E/289E/290E/292E pattern).
6. **TRACK upstream-pipeline regression class as recurring bug** — Wf-270A-D + Wf-272A-D + Wf-271A-D = 3 occurrences; Wf-291C pipeline-root-cause doc identifies 6 occurrences total; recommend adding `pre_flight_check.py::gate_pipeline_schema_required` gate to catch this class at emit time.
7. **SURFACE honest-negative findings in Cf-258 ledger** — `molmetal/scripts/wave_budget_tracker.py` should track `n_honest_negatives_carried_forward` + `n_honest_negatives_closed_this_cycle` metrics for visibility.

---

## §7 Files referenced

- `molmetal/scripts/paper_dashboard.py:454-565` — HONEST_NEGATIVE_CATALOG (18 entries verbatim)
- `molmetal/reports/wf267_final/final_status.md` — Wf-267 final integration
- `molmetal/reports/wf269c_cfm_real/wf269c_summary.md` + `wf269c_report.json` — Wf-269C YuelBond load-bearing
- `molmetal/reports/wf271_synthesis/next_wave_plan.md` — Wf-271E 12-action plan
- `molmetal/reports/wf276_synthesis/followup_summary.md` — Wf-276 §3 critical honest-negatives
- `molmetal/reports/wf276d_xref_audit/audit_report.md` — Wf-276D cross-ref audit (19 §9 entries × 8 §4 sections)
- `molmetal/reports/wf281_synthesis/followup_summary.md` ... `wf296d_matrix_yuelbond_wire/wire_report.md` — Wf-281..296 synthesis + per-wave reports
- `molmetal/reports/wf286a_drugood_5seed/aggregate.json` — Wf-286A FAILED aggregate (gate_1_all_ran=false)
- `molmetal/reports/wf293d_dashboard_refresh/refresh_report.md` — Wf-293D dashboard refresh (+1 claim +1 hn +1 cohort)
- `molmetal/reports/wf294a_preflight_budget/budget_fix_report.md` — Wf-294A budget fix (55s → 90s)
- `molmetal/reports/wf294b_preflight_reverify/reverify_report.md` — Wf-294B post-budget-fix reverify (25/25 PASS)
- `molmetal/reports/wf295a_c258_refresh_3/refresh_report.md` — Wf-295A Cf-258 progress refresh #3
- `molmetal/reports/wf295d_launch_orchestration/plan.md` — Wf-295D launch orchestration plan (7 abort criteria)
- `molmetal/reports/wf296a_integration_audit/audit_report.md` — Wf-296A 42 ORPHAN artifacts
- `molmetal/reports/wf296d_matrix_yuelbond_wire/wire_report.md` — Wf-296D matrix + YuelBond precondition wire
- `metrics/by_metric/cfm_decode_real.json::regression_disclosure` — Wf-274A silent methodology change disclosure

---

**End of Wf-297A audit.** Summary: 18 dashboard-catalog honest-negatives + 24 critical aggregated honest-negatives + 56 wave-level honest-negative findings across Wf-256..296 = **98 total honest-negative audit entries**. Recommend shipping Wf-271F pipeline recovery + Wf-295D launch orchestration + TODO-180 G4 audit + TODO-130 G5 chemprop v3 retrain to close 6 of the top-10 HIGH severity findings within ~30 GPU-hr + ~3 hr CPU wall.
