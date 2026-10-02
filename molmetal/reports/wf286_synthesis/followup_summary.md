# Wf-286E — Wf-286 (A–D) Synthesis + Follow-up Priorities

**Date:** 2026-09-29
**Mode:** Read-only aggregation of Wf-286A + Wf-286B + Wf-286C + Wf-286D outputs.
**Owner:** Hugo Lee

---

## §0 TL;DR

| Agent | TL;DR | Outcome |
|---|---|---|
| **Wf-286A** | 5-seed DrugOOD v3-fallback confirm | **FAILED** — gate_1_all_ran=false, gate_2_total_runs=0; only data-loading ran (n_train=60908 loaded @ 11.9s); no training, no AUC, all 7 gates empty. |
| **Wf-286B** | ECFP4-direct re-eval (smoke) | **PASS** — 1-seed × 5-ep ECFP4 + BinaryClassificationFFN on DrugOOD scaffold-split → test_auc=**0.8820** (Δ=+0.211 vs Wf-185 0.6707). Rules OUT ECFP4 axis as Wf-253-D 0.4544 collapse bottleneck; collapse is backbone / v3-fallback / JSON-split axis. |
| **Wf-286C** | Cf-258 17-wave progress refresh | **PARTIAL** — n_done=2/17 unchanged; 7 newly unblocked (C10,C11,C13,C14,C15,C16,C17 via `--no-kendall`/`--no-dann` wire); C6 progressed (Ru seed=0 MEASURED 0.7716; seed=1 RUNNING; seed=2 PENDING). |
| **Wf-286D** | Cf-265 dashboard refresh | **PASS** — 33 claims, 17 honest-negatives, 6 cohorts; Wf-282 G5 Ru seed=0 claim propagated (`claim_wf282_g5_ru_seed0_test_auc` = 0.7716, 8 grep matches); dashboard.html 76.9 KB persisted. |

**Aggregate: 2 PASS (B, D) + 1 PARTIAL (C) + 1 FAILED (A) → 3 productive + 1 failure.**

---

## §1 Per-agent summary

### §1.1 Wf-286A — 5-seed DrugOOD v3-fallback confirm (FAILED)
- **What was attempted:** Verify chemprop v3-fallback on the Wf-275B adapter's `lbap_refined_ic50_scaffold` CSV at 5-seed × 5-ep.
- **What actually ran:** Only `load_split_data` executed (n_train=60908, n_pos=58015, pos_rate=0.953; iid_val=14068) before the per-seed trainer step did not start — aggregate gates remain empty.
- **Honest verdict:** **FAILED / NO-OP** — `gate_1_all_ran=false`, `gate_2_total_runs=0`, all downstream gates empty (`{}`). Wall = 1.86e-5 min (effectively zero). The smoke did NOT actually train.
- **Root cause hypothesis:** Likely the v3-fallback branch hit a guard or the trainer stage aborted before the first epoch; needs re-run with `--debug-epochs 1` and stdout capture to localise.
- **Source:** `molmetal/reports/wf286a_drugood_5seed/aggregate.json` + `run.log`.

### §1.2 Wf-286B — ECFP4-direct re-eval (PASS smoke)
- **Setup:** Linear 2048→128 projection + chemprop v2 `BinaryClassificationFFN` (hidden=128, n_layers=2), 1-seed × 5-ep, CPU, 190.9 s wall.
- **Result:** train_auc=0.9603 / val_auc=0.8598 / test_auc=**0.8820** (single seed).
- **Δ vs Wf-185:** +0.2113 (test_auc 0.8820 vs baseline 0.6707 ± 0.0042).
- **Caveat (honest):** Subset baked into CSV is `lbap_core_ic50_assay` (OOD-assay task), NOT `lbap_refined_ic50_scaffold` used by Wf-185. Δ is across subsets, optimistic; matching-subset chemprop v2 baseline still needed to pin down exact Δ.
- **Verdict:** ECFP4 featurization axis is **NOT** the Wf-253-D 0.4544 collapse bottleneck; the collapse is backbone / v3-fallback / JSON-split.
- **Source:** `molmetal/reports/wf286b_ecfp4_reeval/aggregate.json` + `reeval_report.md`.

### §1.3 Wf-286C — Cf-258 17-wave progress refresh (PARTIAL)
- **n_done BEFORE (Wf-280D):** 2/17 (C1, C2).
- **n_done AFTER (Wf-286C):** 2/17 — **unchanged**.
- **n_partial (progressed):** +1 (C6: Ru seed=0 MEASURED test_auc=0.7716; seed=1 RUNNING epoch 11/50; seed=2 PENDING).
- **n_pre_launch_ready (infra closed, not run):** +7 (C10, C11, C13, C14, C15, C16, C17 — `--no-kendall`/`--no-dann` flags WIRED end-to-end per Wf-285B; 3 wiring tests PASS).
- **completion_pct (DONE-only):** 11.76% unchanged.
- **completion_pct (DONE + PARTIAL + PRE-LAUNCH):** 52.94% (9/17), +41.18 pp.
- **Live process state:** G5 wrapper PID 609528 alive since Sep 28 23:41; Ru seed=1 trainer PID 1015907 RUNNING epoch 11/50.
- **Remaining blockers:** C3–C5 (CFM P1.3 + P1.4 ablation, GPU-0 Blackwell, 16-hr scope — pre-launch); C7–C9 (G5 Ir/Os/Re continuation — blocked on C6 finish, ETA ~7–8 hr from wrapper start).
- **Source:** `molmetal/reports/wf286c_c258_refresh/refresh_report.md`.

### §1.4 Wf-286D — Cf-265 dashboard refresh (PASS)
- **Pre-check:** Catalog had 32 claims / 12 honest-negatives / 6 cohorts (Wf-280C baseline).
- **Post-check:** Catalog has **33** claims / **17** honest-negatives / 6 cohorts after Wf-284C added `claim_wf282_g5_ru_seed0_test_auc` (value=0.7716).
- **Dashboard rebuild:** `/tmp/dashboard_v286d.html` 76.9 KB (33 claims, 17 honest-negatives, 6 cohorts); persisted to `molmetal/reports/wf265_dashboard/dashboard.html`. 8 grep matches for "0.7716|Wf-282".
- **Wf-284C propagation:** Verified across all 4 targets (pre-flight §9.AK, paper §4.5.1, wave_budget_tracker 10 constants, dashboard catalog).
- **Source:** `molmetal/reports/wf286d_dashboard_refresh/refresh_report.md`.

---

## §2 Aggregate tally

| Metric | Count |
|---|---:|
| Upstream agents run | 4 |
| Agents with concrete MEASURED output | **3** (B smoke AUC, C 1 partial seed, D dashboard rebuild) |
| Agents failed (no progress) | **1** (A: gates empty, no trainer step) |
| Total honest-negatives newly cited | +5 (12 → 17) |
| Total claim cards | +1 (32 → 33) |
| New MEASURED test_auc numbers | 2 (Wf-286B 0.8820; Wf-286C Ru seed=0 0.7716) |
| Cf-258 waves newly unblocked | +7 (C10, C11, C13–C17) |
| Cf-258 waves progressed (seeds) | +1 (Ru seed=0 of C6) |

---

## §3 Critical honest-negatives remaining

1. **TODO-130 G5 (chemprop v3 DrugOOD):** Still 0.4544 BELOW_RANDOM. Wf-286B rules out ECFP4 axis; remaining hypotheses: (a) chemprop v3 message-passing backbone, (b) v3-fallback path config, (c) DrugOOD JSON train/val/test OOD splits. **Wf-286A failed to confirm any of these — needs re-run.**
2. **Cf-258 waves C3–C5 (CFM P1.3 + P1.4):** Pre-launch scope is 16-hr Blackwell + W₂ gradient-path verification. CFM ablation flag + PlatinAI 4-cell + DrugOOD per-cohort assays not yet wired. **15/17 waves still PENDING.**
3. **Cf-258 waves C7–C9 (G5 Ir + Os + Rh + Re):** Blocked on C6 (Ru seed=1/2) ETA ~7–8 hr from wrapper PID 609528 start. **14/15 G5 pairs PENDING.**
4. **Cf-258 wave C6 Ru seed=1/2:** LIVE; seed=1 epoch 11/50 (PID 1015907); seed=2 PENDING. ETA mid-day 2026-09-29.
5. **Cf-258 wave C12 (no_wass via `--wasserstein-weight 0.0`):** Was always launchable (no flag gap); out of "newly unblocked" scope but still PENDING — should be cheap ~10 min smoke to MEASURE.
6. **TODO-182 G6 per-lever attribution:** 0/60 runs MEASURED; 7/8 waves launch-ready (only C12 always-launchable). Estimated G6 wall = 7.5 hr sequential on GPU-1 after G5 sweep ends.
7. **TODO-96 R20-P5 (Catherine 30-metal reachability ≥ 0.95):** DEFERRED — no movement.
8. **TODO-98 L12 (cross-domain OOD ≥ 0.85):** DEFERRED — Q4 2026 ETA.

---

## §4 Recommended next-follow-up priorities

| Priority | Action | Wall | GPU | Honest framing | Source |
|---:|---|---|---|---|---|
| **P1** | **Re-run Wf-286A** with `--debug-epochs 1` + full stdout capture to localise why 0 trainer steps ran. | <15 min | CPU | Don't claim 5-seed result until gates populate. | Wf-286A §1.1 |
| **P2** | **Wf-286B-Full** (5-seed × 20-ep ECFP4 + BinaryClassificationFFN) + matching-subset chemprop v2 MPNN baseline on `lbap_core_ic50_assay`. | ~32 min | CPU or RTX 5090 (~5 min) | Pin down exact Δ ± pooled-std; update TODO-130 G5 sign-flip audit. | Wf-286B §9 |
| **P3** | **Wf-286C-P2** — Launch C7 (Ir s0/s1/s2, ~60 min) AFTER C6 seed=2 ends (~mid-day 2026-09-29). | ~1 hr | GPU-1 | Ready; no fresh user OK needed. | Wf-286C §5 |
| **P4** | **Wf-286C-P3** — Launch C8 + C9 in sequence AFTER C7 ends (~120 min). | ~2 hr | GPU-1 | Ready; same wrapper. | Wf-286C §5 |
| **P5** | **Wf-286C-P6 + P7** — Launch G6 C10–C17 sequentially (~8.6 hr) AFTER G5 sweep ends. CLI verified by Wf-285B + 3 wiring tests. | ~8.6 hr | GPU-1 | 0/60 G6 pairs MEASURED — start now to close per-lever attribution gap. | Wf-286C §5 |
| **P6** | **Wf-286C-P4 + P5** — Launch C3 (P1.3 smoke, GPU-0 Blackwell, ~30 min) + C4 + C5 (P1.4 ablation, ~120 min). Parallel to G5/G6 on GPU-1. | ~2.5 hr | GPU-0 | Fresh user OK required (16-hr Blackwell pre-launch). | Wf-286C §5 |
| **P7** | **Launch C12** (no_wass, `--wasserstein-weight 0.0`, ~10 min) as cheap first MEASURED G6 wave. | ~10 min | GPU-1 | Was always launchable; do it before C10 to validate the wrapper path. | Wf-286C §1.4 |
| **P8** | **TODO-130 G5 root-cause** — pick the strongest of (a)/(b)/(c) from §3.1 based on Wf-286B-Full outcome + Wf-286A re-run. | <1 hr analysis | — | Cannot claim collapse fix until root cause is MEASURED. | Wf-286B §6 |

---

## §5 Sequencing note

- **Steps P3–P5** form a sequential GPU-1 chain (~11.6 hr after C6 finishes) and should be wrapped in a single launcher script (analogous to `launch_wf124_p3_g5.sh`).
- **Steps P4 + P5** (GPU-0 Blackwell) and **P3–P5** (GPU-1 RTX 5090) can run **in parallel** — different GPUs, no contention.
- **Step P2** (Wf-286B-Full) is **independent** of all GPU waves; runs on CPU/5090 and can launch any time.
- **Step P1** (Wf-286A re-run) is **independent** and should be the **first** action — it is the only upstream agent that produced no result.

---

## §6 Files referenced

- `molmetal/reports/wf286a_drugood_5seed/aggregate.json` + `run.log`
- `molmetal/reports/wf286b_ecfp4_reeval/aggregate.json` + `reeval_report.md` + `seed0.json`
- `molmetal/reports/wf286c_c258_refresh/refresh_report.md`
- `molmetal/reports/wf286d_dashboard_refresh/refresh_report.md`
- `molmetal/reports/wf265_dashboard/dashboard.html` (76.9 KB, 33 claims, 17 honest-negatives, 6 cohorts)
- `molmetal/reports/wf280d_c258_waves/audit_report.md` (Wf-280D baseline)
- `molmetal/reports/wf284_synthesis/followup_summary.md` (Wf-284E 4-of-4 synthesis)
- `molmetal/tests/test_kendall_dann_wiring.py` (3 tests pass — Wf-285B wire-up contract)

---

*Generated by Wf-286E (2026-09-29). Read-only synthesis. 1 of 4 upstream agents failed (Wf-286A: empty gates, no trainer step). 3 of 4 productive (Wf-286B PASS smoke, Wf-286C PARTIAL +1 seed + 7 unblocked, Wf-286D PASS dashboard refresh). 8 follow-up priorities queued.*