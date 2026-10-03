# Wf-327-F13 — Rh LOMO 0.136 regression investigation (CPU-only)

**Date**: 2026-09-29
**Workflow**: Wf-327-F13 (this doc) — task ID `wf327-f13`
**Status**: ✅ **VERIFIED — root cause already documented; no new investigation needed**
**Owner**: Hugo Lee + ultracode agent
**Parent**: Wf-121 + Wf-125 P2 + Wf-131 P2 + Wf-176 + Wf-322F + TODO-135
**Investigation type**: CPU-only root-cause analysis (no GPU launch; no paper work; per AGENTS.md §1.2 kill_then_adapt + §5.3 paper-gate)

---

## §0 TL;DR — the regression is REAL, root cause is DOCUMENTED

The Rh LOMO 0.136 regression has been **fully investigated** by Wf-322F (2026-09-29) and the prior Wf-125 P2 + Wf-131 P2 + Wf-176 work. Wf-327-F13 is therefore a **verification + synthesis** step (per the computed task spec "If Wf-325C already investigated: confirm + skip; otherwise execute"), not a fresh investigation.

| Item | Value | Source |
|---|---|---|
| **WF325C_STATUS** | DOES NOT EXIST (no Wf-325C file; numbering gap in Wf-325 series is A/B/D/E/F — 325C was never written) | `ls TODO/completed/325*.md` |
| **WF322F_STATUS** | **SHIPPED 2026-09-29** — full root-cause analysis + Phase 1/2 scaffold/wrapper state | `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` |
| **WF125_P2_STATUS** | **SHIPPED 2026-09-27** — first discovery of the regression in the W₂-stacked run | `TODO/completed/125_wf125_p2_lomo_wasserstein_rerun.md` |
| **VERDICT** | **VERIFIED — root cause = MODEL_CHANGE_REGRESSION (constant-inactive label bias on synthetic Rh cohort)** | this doc |

---

## §1 What is Rh LOMO measuring?

**Definition**: m19 = Multi-metal LOMO (Leave-One-Metal-Out) transfer — train on N-1 metals, hold out the test metal, evaluate test ROC-AUC. Per `metrics/by_metric/metal_lomo_auc.json`:

```
metal_lomo_auc.json (R17 baseline, 5-fold × 3-seed × 30-epoch)
  per_fold_auc_mean:
    Ru: 0.5706
    Ir: 0.6176
    Os: 0.6113
    Rh: 0.3677   <- bottleneck fold
    Re: 0.5059
  value: 0.5346 ± 0.110
  verdict: BELOW_GATE (gate ≥ 0.70)
  source: molmetal/scripts/train_metal_lomo.py
```

**The Rh fold is the bottleneck** — at 0.3677 it is well below the 0.5 random baseline (huh? actually 0.5 is the random baseline for AUC; 0.3677 is below-random). This is the Rh leave-out fold's known weakness: when Rh is held out, the model cannot transfer knowledge from the other 4 metals because Rh has a distinct chemotype profile (Rh(III) octahedral + π-bonding vs the square-planar Pt/Ru/Pd base).

**Aggregation scope**: mean of per-fold AUCs across 5 held-out metals × N seeds.

---

## §2 What is the BASELINE (pre-regression)?

| Timepoint | Rh fold AUC | Source | n_runs | Epochs | Status |
|---|---:|---|---:|---:|---|
| Wf-32 R17 baseline (m19 ship) | **0.3677** | `metrics/by_metric/metal_lomo_auc.json` | 3 seeds × 5 folds = 15 | 30 | MEASURED (pre-regression) |
| Wf-125 P1 (synthetic Rh cohort, no oversample) | **0.368** | `molmetal/reports/wf125_p1_rh_oversample/aggregate.json` | n/a | n/a | MEASURED (pre-bug) |

**Baseline = 0.368** (consistent across two independent runs).

---

## §3 What is the MEASURED 0.136 value?

| Timepoint | Rh fold AUC | Source | n_runs | Epochs | Status |
|---|---:|---|---:|---:|---|
| **Wf-131 P2 (constant-inactive label bug)** | **0.1362 ± 0.0649** | `TODO/completed/125_wf125_p2_lomo_wasserstein_rerun.md` §2.2 | 2 seeds × 1 fold = 2 | 6 | MEASURED (regression) |
| Wf-176 (Re fold only; Rh fold blocked) | 0.34375 (Re only) | `molmetal/reports/wf176_m19_oversample_lomo/aggregate.json` | n/a | 50 | MEASURED_BELOW_GATE |

The **0.1362 value is the 2-seed mean** from Wf-131 P2 (also Wf-125 P2 in the same run family). Per-seed breakdown:

| Seed | test_AUC | val_AUC | Notes |
|---:|---:|---:|---|
| 0 | 0.071 | 0.67 | model predicts all inactives; below random |
| 1 | 0.202 | 0.59 | below baseline; but train-pool val generalises |

---

## §4 Regression magnitude

```
baseline (Wf-125 P1)        : 0.368
regressed (Wf-131 P2 / Wf-125 P2) : 0.136
Δ                              : -0.232  (absolute)
Δ                              : -63.0%  (relative drop)
```

This is a **-0.232 / -63% regression** in the Rh fold AUC. The fold went from weak-but-functional (0.368) to near-random (0.136 — worse than random guessing). This is **catastrophic** in ML terms.

---

## §5 Root cause — MODEL_CHANGE_REGRESSION (label-bias bug)

Per `TODO/completed/125_wf125_p2_lomo_wasserstein_rerun.md` §2.4 (Wf-125 P2 discovery) + `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` §0 (Wf-322F codification):

**Root cause = constant-inactive label on the 500 synthetic Rh oversample compounds.**

Mechanism (per Wf-125 P2 §2.4 honest-negative narrative):

1. **Test-set imbalance dominates AUC**: Rh held-out test set has 608 rows with 5.8% actives (35 actives). With only 6 epochs (vs 30 in baseline), the model doesn't converge well on this small minority class.

2. **Synthetic oversampling biases the inactive class**: The 500 synthetic Rh compounds appended to the Rh training pool carry `label=0` (inactive) by default. This pushes the train pool's prior further toward "inactive" (already 70% inactive), making the held-out Rh test predictions even more biased toward "inactive" → test AUC collapses below random.

3. **6 epochs is too few** for any LOMO fold (vs 30 used in baseline); the under-trained model amplifies the label-bias effect.

The **constant-inactive label** is the load-bearing honest-negative lesson:
> *"an oversample with a constant inactive label reinforces the prior and is not the right knob for an imbalanced held-out test fold."* (Wf-125 P2 §2.4)

### §5.1 Root cause category (per task spec)

| Category | Applies? | Rationale |
|---|:---:|---|
| **MODEL_CHANGE_REGRESSION** | ✅ YES | The model architecture was *not* changed, but the **data augmentation strategy** was changed (500 synthetic Rh compounds appended with constant inactive label). The "change" is on the training data side. |
| DATA_SHIFT | ❌ NO | The training data distribution was deliberately shifted by the oversample, but the test set was unchanged. The shift is the *cause*, not a *passive* drift. |
| METRIC_BUG | ❌ NO | The AUC calculation is correct; the regression is a real test-AUC collapse, not a numerical artefact. |
| HONEST_NEGATIVE_BLOCKED | ✅ partial | The fold is reproducible but the fix is blocked on Phase 2 GPU (USER_OK-gated). |

**Primary category = MODEL_CHANGE_REGRESSION** (specifically a **data-augmentation regression** — adding 500 inactive-labeled compounds to the train pool).

---

## §6 CPU-fixable? — YES, the fix is already wired

The fix path is well-documented and **CPU-only work has already shipped**:

| Phase | Deliverable | Status | Path |
|---|---|---|---|
| **Phase 1 (CPU)** | 500-row balanced Rh cohort (250 active + 250 inactive), Catherine 6/6 PASS | ✅ **SHIPPED** (Wf-309A / Wf-139 P7) | `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` |
| **Phase 2 CPU scaffold** | Trainer with `--balanced-cohort-json`, `--oversample-positives`, `--pos-weight` flags + 12 tests | ✅ **SHIPPED** (Wf-321A) | `molmetal/tests/test_wf321a_wf135_phase2_cpu_scaffold.py` |
| **Phase 2 GPU wrapper** | 5-metal × 5-seed × 50-epoch = 25 runs, ~10 GPU-hr | ⚠️ **READY, USER_OK-BLOCKED** (Wf-310E) | `scripts/launch_wf310e_wf135_phase2_gpu.sh` |

The fix is **CPU-architecturally complete**; only the GPU re-bench is pending. Per user instruction 2026-09-29 ("retraining is held until honest-negative issues resolved first"), the Phase 2 GPU launch is **deliberately blocked**, not a defect.

**CPU-fixable verdict: YES** (Phase 1 + Phase 2 CPU scaffold are SHIPPED; GPU re-bench awaits USER_OK).

---

## §7 What Wf-327-F13 confirms (and does NOT do)

**Confirms (CPU-only, on-disk evidence)**:

1. ✅ The 0.136 regression is **REAL** (per Wf-131 P2 / Wf-125 P2 MEASURED JSON).
2. ✅ The baseline is 0.368 (Wf-125 P1 MEASURED).
3. ✅ The magnitude is -0.232 / -63% (catastrophic).
4. ✅ The root cause is the **constant-inactive label bug** on the 500 synthetic Rh oversample.
5. ✅ The fix path is **CPU-architecturally complete** (Phase 1 cohort SHIPPED + Phase 2 CPU scaffold VERIFIED).
6. ✅ The Phase 2 GPU re-bench is **deliberately USER_OK-BLOCKED** per user instruction.
7. ✅ No paper work touched (TODO-84 honored — READ-only on `paper/*`).

**Does NOT do (per task spec scope)**:

1. ❌ Does NOT launch any GPU re-bench (USER_OK-blocked + per AGENTS.md §1.2 kill_then_adapt).
2. ❌ Does NOT touch `paper/*` (TODO-84 paper-gate honored).
3. ❌ Does NOT modify any code (this is a verification + synthesis step, not a fix).
4. ❌ Does NOT contradict any prior Wf's finding — this doc is consistent with Wf-322F §0 (the comprehensive closeout) + Wf-125 P2 §2.4 (the discovery doc).

---

## §8 Cross-references

### §8.1 Direct ship surface (this doc)

* `TODO/completed/327f13_rh_lomo_investigation.md` — this file (NEW, 2026-09-29)

### §8.2 Prior work (MEASURED on disk)

* `TODO/completed/125_wf125_p2_lomo_wasserstein_rerun.md` — Wf-125 P2 discovery (2026-09-27)
* `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` — Wf-322F final closeout (2026-09-29)
* `TODO/completed/309a_*` — Wf-309A Phase 1 cohort SHIP (referenced via Wf-322F §1)
* `TODO/completed/321a_*` — Wf-321A Phase 2 CPU scaffold VERIFY (referenced via Wf-322F §2)
* `TODO/completed/310e_*` — Wf-310E Phase 2 GPU wrapper READY (referenced via Wf-322F §3)
* `molmetal/reports/wf125_p1_rh_oversample/synthetic_rh_cohort.json` — buggy cohort (constant-inactive)
* `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` — fixed cohort (balanced)

### §8.3 Metrics

* `metrics/by_metric/metal_lomo_auc.json` — R17 baseline (Rh fold 0.3677)
* `metrics/by_metric/metal_lomo_auc_wasserstein.json` — Tier-2 catalog (overall 0.6220, but Rh fold not recovered)
* `metrics/by_metric/metal_lomo_auc_v2.json` — v2 trainer placeholder (currently None per_fold)

### §8.4 Plans / risk / INDEX

* `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` §1.4 — acceptance gates (Rh ≥ 0.30, LOMO ≥ 0.60)
* `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` row 4 — TODO-135 P0 PENDING
* `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §2.9 — m19 multi-metal transfer (note: §3.5 reference in computed task is unrelated — it's about MetalLigandExchange SMARTS, not Rh LOMO)
* `TODO/INDEX.md` §0.10 / §17 — Rh LOMO regression rows

---

## §9 Honest framing (party-twenty-spirit §3.5)

**What this investigation claims (MEASURED, on-disk evidence)**:

1. ✅ **wf325c_status**: Wf-325C does NOT exist; the regression investigation has been done under Wf-322F (2026-09-29 final closeout) and Wf-125 P2 (2026-09-27 discovery).
2. ✅ **rh_lomo_baseline**: 0.368 (Wf-125 P1, MEASURED).
3. ✅ **rh_lomo_measured**: 0.1362 ± 0.0649 (Wf-131 P2 / Wf-125 P2, MEASURED, 2 seeds × 1 fold).
4. ✅ **regression_magnitude**: -0.232 / -63% (catastrophic, below-random).
5. ✅ **root_cause**: MODEL_CHANGE_REGRESSION — constant-inactive label bug on the 500 synthetic Rh oversample compounds. Data-augmentation strategy change, not a model architecture change.
6. ✅ **cpu_fixable**: YES — Phase 1 cohort SHIPPED (Wf-309A), Phase 2 CPU scaffold VERIFIED (Wf-321A), Phase 2 GPU wrapper READY (Wf-310E). Only the GPU re-bench is USER_OK-BLOCKED.

**What this investigation does NOT claim (per AGENTS.md §5.2)**:

1. ❌ Does NOT claim Rh fold AUC is recovered — it is **NOT** recovered (still 0.136 honest-negative per Tier-2 catalog).
2. ❌ Does NOT claim Phase 2 GPU launch is imminent — USER_OK-BLOCKED per user instruction.
3. ❌ Does NOT contradict any prior Wf's finding — fully consistent with Wf-322F.
4. ❌ Does NOT touch paper/* (TODO-84 honored).
5. ❌ Does NOT promote the PROJECTED Rh AUC +0.164 recovery to MEASURED.

**Standing directive alignment**:

* AGENTS.md §1 rule 1 (MEASURED-vs-PROJECTED tag discipline): respected.
* AGENTS.md §1 rule 2 (kill_then_adapt GPU directive): respected — this is a CPU-only investigation.
* AGENTS.md §1 rule 3 (no paper work without authorization): respected.
* AGENTS.md §1 rule 4 (pre-flight v2 gate): not triggered (no GPU/panel/ablation launch).
* AGENTS.md §1 rule 5 (honest-negative framing): respected — no aspirational numbers; 0.136 honest-negative preserved.

---

**END OF Wf-327-F13** — Owner Hugo Lee; review party-twenty-spirit §3.5; this is a verification + synthesis doc (the underlying investigation work was done by Wf-125 P2 + Wf-322F).
