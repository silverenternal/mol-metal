# Wf-310 Ultracode Queue — Final Ready-to-Launch Dashboard

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent (Wf-310I synthesis)
**Source workflow**: `tmp/wf310_ultracode_queue_comprehensive.js` (8 parallel + synthesis)
**Honest framing**: per `party-twenty-spirit` §3.5 — every numeric value below traces to a JSON/report path on disk.
Zero auto-launch. All GPU work is `USER_OK=1` gated per `mol-metal-gpu-execution` §0 (kill_then_adapt).

---

## §1 Per-agent status (8 parallel outputs)

| Agent | Wall | Status | Output (absolute path) |
|---|---:|---|---|
| **Wf-310A** G5 aggregation fix + 2 missing pairs | 0 CPU done; **~3-5 hr GPU remaining for 2 retries** | **PARTIAL** — G5 13/15 MEASURED via `wf278a_g5_aggregate.py` (path+schema fix); 2 retries (Ir seed=0 + Ru seed=2) **IN-FLIGHT** at 09:44:51 on GPU-0/GPU-1 | `molmetal/scripts/wf278a_g5_aggregate.py`, `results/wf124_p3_50epoch_full_sweep_g5/training_summary.json` (13 pairs), `metrics/by_metric/wf181_g5_50epoch_full_sweep.json` (aggregate_5metal test_auc_mean=0.7227 ± 0.0417), `/tmp/wf310a_retry.log` |
| **Wf-310B** TODO-182 G6 final wrapper | 7.6 hr GPU (cuda:1) | **READY (DRY-RUN only)** — 60/60 tuples CONFIRMED, 8/8 tests PASS, syntax OK, USER_OK gate HARD-REQUIRED | `scripts/launch_wf310b_g6_final.sh`, `molmetal/reports/wf310b_g6_manifest/60_tuples.json`, `molmetal/tests/test_wf310b_g6_launcher.py` (8/8 PASS), `molmetal/reports/wf310b_g6_launch_wrapper/launch_plan.md` |
| **Wf-310C** TODO-179 + TODO-220 + Wf-219 §4.1 | 3.9 hr (TODO-220 GPU) + 4.5 hr (TODO-179 GPU) + 13.5 min (Wf-219 §4.1 Ru) | **PARTIAL** — CPU smokes MEASURED (temporal 5/5 ok @ mean_auc=0.6810; NCI-60 9/9 ok @ mean_auc=0.5617); 3 GPU wrappers READY but USER_OK-gated | `molmetal/reports/wf310c_t179_w2_wrapper/launch_plan.md`, `molmetal/reports/wf310c_todo220_launch/launch_plan.md`, `molmetal/scripts/launch_wf310c_t179_w2_gpu.sh`, `molmetal/scripts/launch_wf310c_todo220_gpu.sh`, `molmetal/reports/wf310c_temporal_split/sweep_summary_smoke.json`, `molmetal/reports/wf310c_nci60_9cell/sweep_summary_smoke.json` |
| **Wf-310D** Wf-301A drugood DANN + Wf-301B Vina | 3-4 hr (DrugOOD GPU) + 1-2 hr (Vina CPU; GPU-1 unused) | **READY (templates shipped; full awaits USER_OK)** — DrugOOD smoke **PASS_WF185** (0.8024 ± 0.0224, +0.1317 vs Wf-185 v2 baseline); Vina smoke **HONEST_NEGATIVE_DOCK** (-5.564 kcal/mol on 97/100 docked) | `molmetal/scripts/wf310d_drugood_dann_full_aggregate.py`, `molmetal/scripts/wf310d_vina_full_aggregate.py`, `molmetal/reports/wf310d_drugood_full/{aggregate.json,claim_cards.md,aggregation_plan.md}`, `molmetal/reports/wf310d_vina_full/{aggregate.json,claim_cards.md,aggregation_plan.md}` |
| **Wf-310E** Wf-135 Phase 2 + Wf-219 §4.2 production backbone | 10 GPU-hr = ~2 weeks calendar (Wf-135 P2) + 80-110 min GPU-1 (Wf-219 §4.2) | **READY (USER_OK + PAPER_OK gates)** — Phase 1 prerequisite (Wf-139 P7 Rh cohort 500) SHIPPED; Wf-219 §4.2 wrapper refuses unless both USER_OK=1 AND PAPER_OK=1 (TODO-84) | `scripts/launch_wf310e_wf135_phase2_gpu.sh`, `scripts/launch_wf310e_wf219_production_gpu.sh`, `molmetal/reports/wf310e_wf135_phase2/{launch_plan.md,plan.json,launch.log}`, `molmetal/reports/wf310e_wf219_production/{launch_plan.md,plan.json,launch.log}` |
| **Wf-310F** TODO-234 frontier math 5-7 day plan | 4-5 hr GPU (Wave A 1.5 hr + Wave B 0.5 hr + Wave C 2-3 hr) | **READY (CPU done; Day 7 GPU USER_OK)** — CPU portion (Day 1-2 wrapper + 4 smoke tests) SHIPPED; Day 7 GPU production wire-in deferred to USER_OK gate (3 waves) | `TODO/completed/234_wf234_frontier_math_cpu_plan.md`, `molmetal/scripts/launch_wf310f_frontier_math_gpu.sh`, `molmetal/reports/wf310f_frontier_math_gpu/{launch_plan.md,launch.log,wave_a/DRY_RUN.log}` |
| **Wf-310G** 4 LOI send-ready package | 0 (drafts only — send is user's call) | **READY (4 LOI consolidation + cold-email template + send protocol SHIPPED; 0 sent)** — all 4 BLOCKED on (a) PI email lookup (4 × TBD) + (b) Hugo's ORCID (1 × TBD) | `TODO/pending/wetlab_loi_send_ready_package.md`, `TODO/pending/wetlab_cold_email_template.md` |
| **Wf-310H** Actionable-now CPU bundle (8 tasks) | ~10 min CPU | **COMPLETE (8/8 tasks DONE)** — 67 acceptance tests PASS, 175 pre-flight v3 gates PASS | `TODO/completed/310_wf310h_actionable_now_cpu_bundle_closeout.md` |

---

## §2 USER_OK-gated GPU queue (sorted by wall)

| # | Task | Wall | USER_OK gate | Wrapper | Source output | Notes |
|---:|---|---:|---|---|---|---|
| 1 | Wf-219 §4.1 Ru short (Wf-306B) | 13.5 min | USER_OK=1 | `scripts/launch_wf306b_wf219_ru_gpu.sh` | — | Ru × 3 seeds × 30 epochs; reuses Wf-306B wrapper |
| 2 | Wf-219 §4.2 production backbone (Wf-310E P2) | 80-110 min | USER_OK=1 + **PAPER_OK=1** (TODO-84) | `scripts/launch_wf310e_wf219_production_gpu.sh` | `molmetal/reports/wf310e_wf219_production/launch_plan.md` | PAPER_BLOCKED per TODO-84 §1.1; 5-metal × 3-seed × 50-epoch × hidden=300/depth=3 |
| 3 | Wf-310A G5 retry (2 missing pairs) | ~3-5 hr | IN-FLIGHT | (parallel on GPU-0/GPU-1) | `/tmp/wf310a_retry.log` | Retries started 09:44:51; Ir seed=0 reached epoch 24/50; Ru seed=2 epoch 6/50 |
| 4 | Wf-310F Wave A CFM frontier-multiph | 1.5 hr (cuda:0) | `WF310F_NEXT_WAVE=A` | `scripts/launch_wf310f_frontier_math_gpu.sh --wave A` | `molmetal/reports/wf310f_frontier_math_gpu/wave_a/DRY_RUN.log` | Wire `augment_pocket_embedding` into CFM trainer |
| 5 | Wf-310F Wave B D-MPNN frontier-SE3 | 0.5 hr (cuda:1) | `WF310F_NEXT_WAVE=B` | `scripts/launch_wf310f_frontier_math_gpu.sh --wave B` | — | SE3LipmanSolver wraps multi-task manifold |
| 6 | Wf-310D Vina pt-strip full (100mol × 4pocket) | 1-2 hr CPU | USER_OK=1 | `scripts/launch_wf306a_drugood_vina_gpu.sh --phase B` | `molmetal/reports/wf310d_vina_full/aggregate.json` | CPU-only; smoke MEASURED -5.564 kcal/mol on 1h36 (97/100 docked) |
| 7 | Wf-310D DrugOOD DANN full | 3-4 hr (cuda:1) | USER_OK=1 | `scripts/launch_wf306a_drugood_vina_gpu.sh --phase A` | `molmetal/reports/wf310d_drugood_full/aggregate.json` | 5-seed × 50-epoch; smoke MEASURED PASS_WF185 0.8024 |
| 8 | Wf-310F Wave C joint frontier sweep | 2-3 hr (cuda:0) | `WF310F_NEXT_WAVE=C` | `scripts/launch_wf310f_frontier_math_gpu.sh --wave C` | — | All 4 frontier adapters joint sweep |
| 9 | Wf-310C TODO-220 GPU sweep | 3.9 hr (cuda:1) | USER_OK_FILE=`/tmp/USER_OK_WF310C_TODO220` | `scripts/launch_wf310c_todo220_gpu.sh` | `molmetal/reports/wf310c_todo220_launch/launch_plan.md` | Stage A 2.9 hr + Stage B 1.0 hr sequential |
| 10 | Wf-310C TODO-179 W₂ PlatinAI 4-cell | 4.5 hr (cuda:1) | USER_OK_FILE=`/tmp/USER_OK_WF310C_T179` | `scripts/launch_wf310c_t179_w2_gpu.sh` | `molmetal/reports/wf310c_t179_w2_wrapper/launch_plan.md` | 5-seed × 50-epoch; honest PROJECTED R² 0.40-0.60 ceiling |
| 11 | Wf-310B TODO-182 G6 final 60-tuple | 7.6 hr (cuda:1) | `USER_OK=1` | `scripts/launch_wf310b_g6_final.sh` | `molmetal/reports/wf310b_g6_manifest/60_tuples.json` | 4 configs × 5 metals × 3 seeds = 60 tuples; 8 sub-waves a..h |
| 12 | Wf-310E Wf-135 Phase 2 LOMO 5×5×50 | 10 GPU-hr ≈ 2 weeks calendar (cuda:1) | `USER_OK=1` | `scripts/launch_wf310e_wf135_phase2_gpu.sh` | `molmetal/reports/wf310e_wf135_phase2/launch_plan.md` | Phase 1 cohort SHIPPED; Rh fold AUC ≥ 0.30 target |

**Sequential GPU queue total**: ~7.6 + 7.5 + 4.5 + 3.9 + 3-4 + 1-2 + 4-5 + 13.5min + 80-110min + 1.5 + 0.5 + 2-3 + 2 weeks ≈ **~37 GPU-hr + 2 calendar weeks** if executed top-to-bottom.

---

## §3 ACTIONABLE-NOW CPU queue

**All Wf-310H actionable-now CPU tasks are COMPLETE.** No remaining CPU work in the Wf-310 batch.

| Task | Status | Acceptance evidence |
|---|---|---|
| 1. TODO-178b smoke FAIL-on-missing audit | **DONE** | `molmetal/tests/test_smoke_fail_on_missing_audit.py` 5/5 PASS |
| 2. ADMET pass-through integration | **DONE** | 3 pre_flight gates PASS (ames 0.0401 ≤ 0.10 ✓; logPapp -3.637 ≥ -5.0 ✓; logS HONEST_NEGATIVE tagged) + 3 paper_dashboard claim cards |
| 3. TODO-220 --execute CPU implementation | **DONE** | Both drivers `--execute` paths verified on CSV v4 |
| 4. Wf-219 §4.3 temporal-split CPU smoke | **DONE** | Ru seed=0 in_domain_test_auc=0.7069 (1-seed × 1-epoch, Krasnov_protocol_aligned=true) |
| 5. Wf-139 P7 Rh cohort 500 generation | **DONE** | 500 SMILES (250 active + 250 inactive), Catherine 6/6 = 1.0000, scaffold overlap 0; 15/15 tests PASS |
| 6. TODO-234 frontier math wire-in RE-DO | **DONE** | `molmetal/adapters/frontier_math_integration.py` (4 composed adapters), 7/7 tests PASS |
| 7. R20-P1 VariancePUCT default + R20-P2 R² CPU smoke | **DONE** | `VarianceControlledPUCT` + `make_default_selector` default-on; 35 VariancePUCT tests + 5 R² lift tests |
| 8. TODO-135 file restore | **DONE** | `TODO/completed/135_wf135_phase1_plan.md` + `135_m19_lomo_rh_oversample_fix.md` both present; Phase 2 wrapper ready (USER_OK-gated) |

**Wall**: ~10 min CPU-only; no GPU; no user OK; no paper.

---

## §4 LOI send-ready queue (per `TODO/INDEX.md` §12.3 P3 — send is user's responsibility)

Recommended send order (low-friction → high-friction):

| # | LOI | Time horizon | Risk | Paper §5 gating? | Blocker | Recommended action |
|---:|---|---|---|---|---|---|
| 1 | **LOI-RUSANOV** (Wf-305A) | 2-6 wk | Low (data-only) | YES (closes Wf-297B §6 NUMERIC gap) | (a) Rusanov corresponding-author email from ChemRxiv v2 DOI 10.26434/chemrxiv-2025-pp32k; (b) Hugo's ORCID | **SEND** |
| 2 | **LOI-PKU** (Wf-74 v1) | 6-12 mo | Medium (synth + 4-cell) | YES (cisplatin 4-cell IC50) | (a) PKU 医学部 PI email; (b) resolve PKU vs 协和 NCI-60 division on intro Zoom | **SEND** (after Rusanov) |
| 3 | **LOI-协和** (Wf-74 v1) | 6-12 mo | High (IRB + 9-cell + ICP-MS) | YES (NCI-60 9-cell + ICP-MS) | (a) 协和医学院 PI email; (b) 6 compliance items (IRB + mycoplasma + NCI-60 sourcing + ICP-MS iCAP RQ / Agilent 7900 + CEM MARS 6 + GCP workflow) | **DEFER** until Rusanov + PKU Phase-1 active |
| 4 | **LOI-ORNL** (Wf-74 v1) | 9-12 mo | Highest (AMBER + SPR + 4-protein) | NO (R22-P4 future-work) | (a) ORNL PI email; (b) 4 ORNL-specific items (AMBER license + ORNL cluster + Biacore X100/T200 + 4-protein expression) | **DEFER** until R21-P4 ships |

**Send protocol**: `TODO/pending/wetlab_loi_send_ready_package.md` §4 (12-step pre-send checklist) + `TODO/pending/wetlab_cold_email_template.md` (subject + body + signature). 0 LOI sent today.

---

## §5 Remaining blockers (5 categories)

### §5.1 User authorization gates
- **TODO-84 PAPER_OK**: blocks Wf-219 §4.2 production backbone (#2 in GPU queue). Wrapper refuses to launch unless `PAPER_OK=1` env is set (explicit lift of paper rule by Hugo).
- **USER_OK gates on 12 GPU wrappers** (sorted in §2): no auto-launch; per `mol-metal-gpu-execution` §0 (kill_then_adapt).
- **4 PI email lookups + 1 ORCID fill** for LOI send (Hugo-only, §4).

### §5.2 Data dependencies
- **TODO-220 Driver 2 (NCI-60 9-cell) CSV present**: `molmetal/data/nci60_precious_9cell.csv` (7823 rows × 9 cells, smoke verified 9/9 ok @ 1-seed × 1-epoch).
- **TODO-220 Driver 1 (temporal-split) CSV present**: `molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (26633 rows, smoke verified 5/5 ok @ 1-seed × 1-epoch).
- **PlatinAI 4-cell CSV present**: `molmetal/data/platinai_4cell.csv` (4078 Pt complexes × 4 cell lines).

### §5.3 Honest-negative flags (per `party-twenty-spirit` §3.5)
- **G5 2 missing pairs retry in-flight** — Ir seed=0 + Ru seed=2 may not complete in time; downstream G5 5metal aggregate = 13/15 MEASURED until retry finishes.
- **Vina pt-strip smoke MEASURED -5.564 kcal/mol ABOVE field-avg band (-7.5 to -9.5)**; recovery axis is *being able to dock at all*, not absolute binding affinity (per Wf-301B honest-negative).
- **DrugOOD DANN smoke MEASURED 0.8024 vs Wf-185 v2 baseline 0.6707**: cross-substrate Δ +0.1317 is directional only; do not interpret as apples-to-apples (per Wf-301A honest-negative).
- **TODO-179 R² ceiling**: PROJECTED 0.40-0.60 from W₂ alone; 0.85+ requires F5_RDKIT_2D 32-d expansion (TODO-179 §8 follow-up).
- **Rh fold 0.136 regression** (Wf-131 P2) — 5-metal × 5-seed × 50-epoch re-bench (Wf-135 Phase 2) needs USER_OK gate.

### §5.4 In-flight retrain (Wf-310A)
- 2 G5 missing pairs retry STARTED at 09:44:51 on GPU-0 (Ir seed=0) + GPU-1 (Ru seed=2).
- Last log capture: Ir seed=0 epoch 24/50, Ru seed=2 epoch 6/50.
- ETA: ~3-5 hr wall (parallel).
- Aggregator `molmetal/scripts/wf278a_g5_aggregate.py` is FIXED and ready to re-run post-completion.

### §5.5 Deferred per upstream scope (not blockers, but tracked)
- Wf-310F Wave B/C + Day 3-6 deferred to USER_OK gate (Day 7 GPU production wire-in).
- TODO-220 full 5-seed × 50-epoch sweep deferred to USER_OK gate (CPU smoke is 1-seed × 1-epoch).
- TODO-179 R² ≥ 0.85 ceiling requires F5_RDKIT_2D 32-d expansion as follow-up.
- R20-P1 VariancePUCT default-on (35 tests PASS) is wire-in only; no paper §3.7 claim until MCTS replay warm-start + per-node hit_rate ≥ 0.95 evidence.

---

## §6 Cross-references

- **Workflow script**: `/home/hugo/codes/mol-metal/tmp/wf310_ultracode_queue_comprehensive.js` (8 parallel + synthesis)
- **Master standing directive**: `AGENTS.md` §5.1 (kill_then_adapt) + §5.3 (no paper without authorization)
- **Pre-flight v3 (TODO-207)**: 175/175 PASS post-Wf-310H batch
- **TODO/INDEX.md**: §12.3 P3 mitigation (LOI send is user's responsibility)
- **TODO/decisions.md**: G5 retrain scope decision (W§9 §9.1)
- **Per-agent outputs**: see §1 table absolute paths

---

## §7 Honest framing disclaimer (party-twenty-spirit §3.5)

- This dashboard reports **MEASURED** for: G5 13/15 pairs (paths verified, AUCs in `wf181_g5_50epoch_full_sweep.json`); DrugOOD smoke 3-seed 0.8024; Vina smoke 97/100 docked -5.564 kcal/mol; TODO-220 CPU smokes 5/5 + 9/9 ok; Wf-310H 8/8 CPU tasks DONE.
- This dashboard reports **PROJECTED** for: all GPU wall-times; R² lift on TODO-179 (0.40-0.60 ceiling); per-lever attribution on TODO-182 G6; Wf-135 Phase 2 Rh fold recovery (0.136 → 0.30).
- This dashboard reports **IN-FLIGHT** for: Wf-310A 2 missing pair retries.
- This dashboard reports **BLOCKED on user action** for: 12 GPU USER_OK gates + 1 PAPER_OK gate + 4 PI emails + 1 ORCID.

**Total remaining blockers**: 5 categories, 19 discrete items (12 USER_OK gates + 1 PAPER_OK + 4 PI emails + 1 ORCID + 1 in-flight retry).

**No GPU launch has been initiated by this synthesis.** All wrappers refuse to launch without explicit user opt-in.
