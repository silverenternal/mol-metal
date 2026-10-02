# Wf-310E Part 1 — Wf-135 Phase 2 LOMO re-bench launch plan

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Wrapper**: `scripts/launch_wf310e_wf135_phase2_gpu.sh`
**Status**: 🚧 **READY on user OK gate** (per `mol-metal-gpu-execution` §0 kill_then_adapt)

---

## §0 TL;DR

| Gate | Description | Status |
|---|---|---|
| **G1** | Wrapper script ready on disk | ✅ `scripts/launch_wf310e_wf135_phase2_gpu.sh` (175 LOC, `bash -n` PASS) |
| **G2** | USER_OK gate enforced (per Wf-306B pattern) | ✅ `USER_OK=1` env required; `--dry-run` does NOT bypass |
| **G3** | Phase 1 prerequisite wired | ✅ Checks `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` |
| **G4** | Acceptance gates documented | ✅ Rh fold AUC ≥ 0.30 + LOMO mean ≥ 0.60 + ≥16 tests PASS |
| **G5** | Wall budget honest (MEASURED vs PROJECTED) | ✅ ~10 GPU-hr compute = ~2 calendar weeks (best 7, worst 14) |

**One-line verdict**: Wf-135 Phase 2 GPU wrapper is READY. Launch blocked on explicit user OK (per `kill_then_adapt` directive, 2-week wall).

---

## §1 Phase 2 specification (from `TODO/completed/135_wf135_phase1_plan.md` §2.1)

### §1.1 Scope

5-metal × 5-seed × 50-epoch = **1250 runs** total.

| Dimension | Value | Notes |
|---|---|---|
| Metals | `Rh Ru Ir Os Re` | LOMO 5-fold (hold one out per fold) |
| Seeds | `0 1 2 3 4` | 5 seeds (vs Wf-131 P3 baseline 3) |
| Epochs | 50 | Krasnov-2026-aligned |
| Backbone | Multi-task D-MPNN | per Wf-137 ship |
| L2 guard | ON | per Wf-127 P2 (grad-norm clip ≤ 5.0) |
| ASTSR scale clip | ON | per Wf-127 P3 (clip predicted pIC50 to [-10, 10]) |
| Wasserstein target | Rh **real active** distribution | NOT synthetic cohort (per TODO-135 §3.2) |
| Per-metal head ensemble | K=3 heads, majority vote | per TODO-135 §3.1 |
| Rh oversample | Wf-139 P7 balanced cohort (250 active + 250 inactive) | diff-based ranking |

### §1.2 Trainer

`molmetal/scripts/train_metal_lomo_v2.py` (~280 LOC) — already exists from Wf-145 P2.
Has flags:
- `--metals` (default `[Rh, Ru, Ir, Os, Re]`)
- `--seeds` (default `[0, 1, 2, 3, 4]`)
- `--epochs` (default 50)
- `--wasserstein-weight` (default 0.05)
- `--balanced-cohort-json` (default `wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json`)
- `--lomo-csv` (default `molmetal/data/metal_lomo.csv`)
- `--use-l2-guard` (default ON)
- `--out-dir`
- `--device`

### §1.3 Why 5-metal × 5-seed × 50-epoch (not 3-seed as Wf-219 used)

- Wf-131 P3 baseline used 3 seeds; the user noted 3 is "enough" for Krasnov protocol comparison
- Phase 2 needs 5 seeds for **per-seed std ≤ 0.05** stability gate (TODO-135 §2.3)
- Per TODO-135 §2.2 hard gate: `5/5 metals × 5 seeds × 50 epochs = 1250 runs completed`

---

## §2 Phase 1 prerequisite (Wf-309A Rh cohort 500)

Per IMPROVEMENT_ENUM §1.2, Phase 1 = balanced Rh cohort generation. **Status: ✅ SHIPPED 2026-09-27 via Wf-139 P7** (per `TODO/completed/143_wf139_p7_rh_active_cohort.md`).

| Path | Status |
|---|---|
| `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` | ✅ 500 balanced (250 active + 250 inactive) |
| `molmetal/scripts/rh_active_label_generation.py` | ✅ 340 LOC |
| `molmetal/tests/test_rh_active_label_generation.py` | ✅ 15/15 PASS |
| Catherine 6/6 PASS | ✅ 1.0000 (500/500) |
| Scaffold leak | ✅ 0 overlap |

The wrapper checks this file exists at launch time. If missing, abort with `[FATAL]` exit code 64.

---

## §3 Acceptance gates (per IMPROVEMENT_ENUM §1.4)

### §3.1 Hard gates (HALT if violated)

| Gate | Standard | Source |
|---|---|---|
| All 1250 runs completed | 5/5 metals × 5 seeds × 50 epochs | TODO-135 §2.2 |
| **Rh fold AUC ≥ 0.30** | vs Wf-131 P2 regression 0.136 (baseline 0.368) | TODO-135 §2.3 |
| **Overall LOMO mean ≥ 0.60** | vs baseline 0.5346 | TODO-135 §2.3 |
| ≥16 tests PASS | `molmetal/tests/test_metal_lomo_v2.py` | TODO-135 §2.2 |

### §3.2 Soft gates (PARTIAL if violated)

- Rh 0.30–0.50 → honest frame as PARTIAL (vs target ≥ 0.30)
- LOMO < 0.60 → honest frame as PARTIAL
- Multi-task backbone signal leak observed → PARTIAL (per-metal head ensemble is the structural fix; if still leaking, deeper issue)

### §3.3 Risk fallback

If Phase 2 does not lift Rh ≥ 0.30:
- Paper §4.5.6 lead candidate forgone (per Wf-135 §2.5 honest framing)
- §6 honest-negative framing applies
- NCI-60 9-cell 0.8007 + single-cell SOTA in §4 unaffected

---

## §4 Wall-time breakdown (honest estimate)

| Activity | Wall-time | Source |
|---|---|---|
| `check_blackwell_env.py` probe + GPU-1 lock | 0.5 day | Wf-135 Phase 1 plan §2.4 |
| Cohort + metal-cohort ingestion prep | 0.5 day | Wf-135 Phase 1 plan §2.4 |
| `train_metal_lomo_v2.py` scaffold | (already exists, 0 day) | Wf-145 P2 ship |
| 5-metal × 5-seed × 50-epoch training | **10 GPU-hours = ~2 calendar days** | Wf-135 Phase 1 plan §2.4 |
| Metric JSON + synthesis.md | 1 day | Wf-135 Phase 1 plan §2.4 |
| Regression vs Wf-131 P2 baseline | 1 day | Wf-135 Phase 1 plan §2.4 |
| Tests + close-out doc | 0.5 day | Wf-135 Phase 1 plan §2.4 |
| **Total wall** | **~7 days** (best), **~14 days** (worst) | Wf-135 Phase 1 plan §2.4 honest upper bound |

PROJECTED compute is **10 GPU-hours**. Calendar wall is **2 weeks** worst case (setup + retries + reporting).

---

## §5 USER_OK gate

USER must explicitly approve (per `mol-metal-gpu-execution` §0 + `TODO-pending/207_unified_pre_flight_checklist_v2.md` §1.1):

1. **Compute**: ~10 GPU-hours on GPU-1 RTX 5090 (Blackwell sm_120). Confirm env probe via `check_blackwell_env.py` first.
2. **Schedule**: 2 calendar weeks wall budget. If GPU-1 contention with kanzi means longer, USER_OK to either (a) wait, (b) drop to 3-seed × 30-epoch (acceptable scope shrink), or (c) cancel Phase 2 and ship honest-negative close-out.
3. **Acceptance criteria**: Rh fold AUC ≥ 0.30 + LOMO mean ≥ 0.60. If Rh stays flat (0.10–0.20), USER_OK to ship honest-negative framing similar to TODO-131 (Rh sample size n=60 is structural ceiling).
4. **Reporting**: LABELING-STRICT (MEASURED-vs-PROJECTED). Every result must label whether it is MEASURED from GPU run or PROJECTED from prior work. No mixing.
5. **Risk fallback**: If Phase 2 does not lift Rh ≥ 0.30, paper §4.5.6 lead candidate is forgone; §6 honest-negative framing applies.

---

## §6 Pre-launch checks (BLOCKER before any GPU work)

1. **Phase 1 cohort on disk**: `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` exists with 250 active + 250 inactive.
2. **`pre_flight_check.py --pipeline all`**: 24/24 PASS target.
3. **`wave_budget_tracker.py --budget-hr 14 --chemprop-not-required`**: confirms wall budget fits.
4. **`USER_OK=1` env gate**: kill_then_adapt enforced.
5. **GPU-1 contention check**: confirm kanzi not blocking RTX 5090 cuda:1.

---

## §7 Abort criteria (per TODO-207 v2 §7.2)

- **per-phase timeout**: ckpt older than 2× projected wall-time
- **failure pattern**: 3+ consecutive NaN/OOM/IndexError
- **GPU util < 70%**: for 5+ minutes
- **collapse**: Rh fold AUC < 0.10 on 2+ seeds (binding regression alarm)

---

## §8 Honest framing per `party-twenty-spirit` §3.5

**承认 (Acknowledge)**:

1. **Wf-131 P2 + Wf-125 P1 both used constant-inactive label** → Rh regression is real bug (Rh fold 0.368 → 0.136).
2. **Catherine 6/6 PASS guarantees structure, not label** — closure ≠ active.
3. **Rh cohort sample size n=60 train is structural ceiling** — oversample is best-effort.
4. **Phase 2 GPU retrain is deferred** until USER_OK gate lifts per `kill_then_adapt`.
5. **Per-metal head ensemble + Wasserstein target fix are documented but unexercised** at MEASURED level.

**PROJECTED lift**:
- Rh fold AUC: 0.136 → 0.30 (target) = **+0.164 recovery** (Δ +120%)
- Overall LOMO mean: 0.5346 → 0.60 (target) = **+0.065** (Δ +12%)

These projections are Wf-139 P7 + Wf-145 P2 + TODO-135 §3.1-§3.2 root-cause analysis informed, NOT MEASURED on the Phase 2 driver until runs complete.

---

## §9 Cross-references

- `TODO/completed/135_wf135_phase1_plan.md` §2.1-§2.4 (full spec + wall-time breakdown)
- `TODO/completed/135_m19_lomo_rh_oversample_fix.md` §0 (root-cause recap)
- `TODO/completed/143_wf139_p7_rh_active_cohort.md` (Phase 1 ship)
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` §1.4 (acceptance gates)
- `molmetal/reports/wf176_m19_oversample_lomo/synthesis.md` (current Rh fold state: 0.3438 ± 0.0 on Re only — needs full 5-metal re-bench)
- `molmetal/scripts/train_metal_lomo_v2.py` (target trainer)
- `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` (Phase 1 cohort)
- `scripts/launch_wf306b_wf219_ru_gpu.sh` (template pattern)
- `mol-metal-gpu-execution` skill §0 (kill_then_adapt directive)

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: Wf-310E Part 1 wrapper is READY on disk. Phase 1 prerequisite is SHIPPED. Acceptance gates are documented. Wall budget is honestly PROJECTED (2 weeks worst case). USER_OK gate is HARD-REQUIRED per `kill_then_adapt`; the wrapper refuses to launch without `USER_OK=1`. No GPU work has been performed; this is a wrapper + plan doc ship, NOT a measurement ship.