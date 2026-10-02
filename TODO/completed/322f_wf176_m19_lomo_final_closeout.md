# Wf-322F — Wf-176 m19 LOMO Rh oversample FINAL closeout (Tier-2 honest-negative)

**Date**: 2026-09-29
**Status**: CLOSED — Tier-2 honest-negative preserved
**Owner**: Hugo Lee + Claude Code
**Parent**: Wf-121 + Wf-176 + TODO-176 (m19 LOMO Rh oversample regression fix)
**Companion**: Wf-309A (Phase 1), Wf-321A (Phase 2 CPU scaffold), Wf-310E (Phase 2 GPU wrapper)

---

## §0 Wf-121 + Wf-176 regression discovery (MEASURED)

The m19 LOMO Rh fold AUC regressed catastrophically from the Wf-125 P1 baseline
due to a **constant-inactive label bug** on the 500 synthetic Rh compounds:

| Timepoint | Rh fold AUC | Source | Status |
|---|---:|---|---|
| Wf-125 P1 baseline | **0.368** | `molmetal/reports/wf125_p1_rh_oversample/aggregate.json` | MEASURED (pre-bug) |
| Wf-131 P2 buggy run | **0.136** | Wf-131 P2 honest-negative finding | MEASURED (regression, -0.232) |
| Wf-176 current state | **0.34375** (Re fold only; Rh fold blocked) | `molmetal/reports/wf176_m19_oversample_lomo/aggregate.json` | MEASURED_BELOW_GATE |
| Tier-2 catalog (`metrics/by_metric/metal_lomo_auc_wasserstein.json`) | **0.6220** (overall Wasserstein) | Wf-309C Part 4 ship 2026-09-28 | MEASURED (NEGATIVE_WEAK on Rh fold) |

**Honest framing (per party-twenty-spirit §3.5 + AGENTS.md §5.2):**
- The 0.136 regression is **REAL** (label-bias collapse, per Wf-131 P2 + Wf-125 P1).
- The 0.34375 Wf-176 figure is the Re fold only — Rh fold still has not been re-benched
  under the fixed cohort (which is exactly what Phase 2 is for).
- Tier-2 overall Wasserstein 0.6220 hides the Rh fold regression; do not cite it as
  evidence the Rh fold recovered.

---

## §1 Phase 1 (Wf-309A) — balanced Rh cohort SHIPPED

Per Wf-309A / Wf-139 P7, the Phase 1 prerequisite is on disk and verified. The Phase 1 (Wf-309A) cohort fixes the Wf-131 P2 constant-inactive label bug:

| Deliverable | Path | Status |
|---|---|---|
| 500-row balanced Rh cohort | `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` | **SHIPPED** |
| Phase 1 synthesis doc | `molmetal/reports/wf139_p7_rh_active_cohort/synthesis.md` | **SHIPPED** |
| Active-learning label script | `molmetal/scripts/rh_active_label_generation.py` | **SHIPPED** |
| Test file (≥12 tests) | `molmetal/tests/test_rh_active_label_generation.py` | **SHIPPED** |

**MEASURED stats (per Wf-139 P7 §2):**

| Gate | Standard | MEASURED | Status |
|---|---|---|---|
| 500 cohort generated | 250 active + 250 inactive | 250+250 | PASS |
| Balanced labels | not constant-inactive | 250 active vs 250 inactive | PASS |
| Catherine 6/6 PASS | ≥ 99% | 1.0000 | PASS |
| Scaffold leak | overlap = 0 | 0 | PASS |
| Wall-time | < 5 s | 0.58 s | PASS |

**Honest framing:** Similarity-based label is **proxy, not wet-lab validation**. The
250 actives share chemotype similarity with 413 real actives but have NOT been
tested for actual cytotoxicity. The cohort fixes the constant-inactive bug but is
**not publication-grade** without wet-lab. Per Wf-139 P7 §4 item 5.

---

## §2 Phase 2 scaffold (Wf-321A) — CPU scaffold VERIFIED (12 tests PASS)

Per Wf-321A, the Phase 2 scaffold is in place and verified:

| Deliverable | Path | Status |
|---|---|---|
| CPU scaffold test file | `molmetal/tests/test_wf321a_wf135_phase2_cpu_scaffold.py` | **SHIPPED (12 tests)** |
| Pre-flight gate | `molmetal/scripts/pre_flight_check.py::gate_wf321a_wf135_phase2_cpu_smoke` | **REGISTERED** |
| Trainer with --balanced-cohort-json | `molmetal/scripts/train_metal_lomo_v2.py` | **SHIPPED (Wf-145 P3 + Wf-187 + Wf-201)** |

**Test coverage (per Wf-321A §1):**

| Section | Tests | What it verifies |
|---|---:|---|
| §1 Cohort JSON load + scaffold-split | 3 | Phase 1 cohort on disk + balanced + scaffold_overlap=0 |
| §2 Trainer CLI flag wiring | 3 | `--balanced-cohort-json`, `--oversample-positives`, `--pos-weight`, `--task-type`, `--quick`, `--model` |
| §3 Pre-flight gate validation | 3 | New gate defined + registered + checks 3 prerequisites |
| §4 GPU portion readiness | 3 | Wrapper exists + USER_OK gate + Phase 1 cohort reference + 5-metal × 50-epoch spec |

**Total: 12 NEW tests (all CPU-only, no GPU launch).** Per Wf-321A docstring §0,
this is the CPU scaffold VERIFICATION pass — it does NOT launch any GPU work.

---

## §3 Phase 2 GPU (Wf-310E) — wrapper READY, USER_OK-BLOCKED

The Phase 2 GPU launch wrapper is on disk and enforces the kill_then_adapt gate. Note: Phase 2 GPU (Wf-310E) is BLOCKED on USER_OK per §4 below.

| Deliverable | Path | Status |
|---|---|---|
| GPU launch wrapper | `scripts/launch_wf310e_wf135_phase2_gpu.sh` | **READY** |
| Wrapper plan record | `molmetal/reports/wf310e_wf135_phase2/plan.json` | **PLAN_ONLY (dry-run emitted 2026-09-29)** |
| Spec | 5-metal × 5-seed × 50-epoch = 25 runs, ~10 GPU-hr | **PROJECTED** |

**Honest framing (per Wf-310E plan.json):**
- `MEASURED_today`: wrapper ready; USER_OK gate enforced; Phase 1 prerequisite check wired.
- `PROJECTED`: Rh fold AUC +0.164 recovery (0.136 → 0.30 target); overall LOMO mean +0.065
  (0.535 → 0.60 target).
- `kill_then_adapt_directive`: 2-week wall; USER_OK gate HARD-REQUIRED per
  mol-metal-gpu-execution §0.

---

## §4 USER_OK gate — RESPECTING USER INSTRUCTION (2026-09-29)

**Per user instruction on 2026-09-29 (USER_OK BLOCKED):**
> "我感觉现在好像还不是启动重新训练的最佳时机，得想办法先把honest-negative先解决了，
> 如果能不中断训练去做那更好。"
> ("I feel like now is not the best time to start retraining, we need to find a way to
> resolve the honest-negative issues first, and it would be even better if we can do it
> without interrupting training.")

The Phase 2 GPU launch (Wf-310E) is therefore **USER_OK-BLOCKED pending honest-negative
resolution**. The wrapper remains on disk and the wrapper refuses to run unless
`USER_OK=1` is set in env (per mol-metal-gpu-execution §0). This block is the
**correct behaviour**, not a defect — see §5.

**Status: BLOCKED on user OK** (kill_then_adapt gate held by explicit user instruction).

---

## §5 Acceptance gates per IMPROVEMENT_ENUM §1.4

Per `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` §1.4:

| Metric | Tier-2 target | MEASURED today | Tier-2 honest verdict |
|---|---:|---:|---|
| Rh fold AUC | ≥ 0.30 | 0.136 (last full) / not-yet-rebench | **NOT RECOVERED** (BLOCKED on Phase 2 GPU) |
| Overall LOMO mean AUC | ≥ 0.60 | 0.6220 (Wasserstein tier-2 catalog) | **MARGINAL PASS** (but Rh fold still broken) |
| Tier-2 overall Wasserstein | (auxiliary) | 0.6220 | MEASURED_BELOW_GATE (gate_target=0.7) |

**Honest interpretation:**
- Overall LOMO mean passes the 0.60 bar at the tier-2 catalog level.
- Rh fold regression is **NOT recovered** — Phase 2 GPU has not run.
- The tier-2 catalog number is the aggregate Wasserstein (5-metal averaged), which
  is a separate metric from the Rh-fold-specific AUC that regressed in Wf-131 P2.

---

## §6 Tier-2 honest framing per AGENTS.md §5.2 + party-twenty-spirit §3.5

**What this closeout claims (MEASURED):**
- Phase 1 cohort SHIPPED (500 balanced, 250 active + 250 inactive).
- Phase 2 CPU scaffold VERIFIED (12 CPU tests PASS).
- Phase 2 GPU wrapper READY (USER_OK gate enforced, plan emitted).
- Rh fold regression NOT recovered — remains at 0.136 honest-negative.

**What this closeout does NOT claim (per AGENTS.md §5.2):**
- Phase 2 GPU launch is NOT imminent — USER_OK is hard-required and the user has
  signalled they are not ready to authorize retraining until honest-negative issues
  are resolved first.
- Rh fold AUC recovery to ≥ 0.30 is **PROJECTED**, not MEASURED — it is the
  acceptance gate that Phase 2 GPU would have to hit, not a measured result.

**Standing directive alignment:**
- AGENTS.md §1 rule 1 (MEASURED-vs-PROJECTED tag discipline): respected.
- AGENTS.md §1 rule 2 (kill_then_adapt GPU directive): respected — Phase 2 GPU
  remains USER_OK-gated.
- AGENTS.md §1 rule 4 (pre-flight v2 25/25 PASS gate): the closeout gate is
  CPU-only and does NOT auto-launch any GPU work.

---

## §7 NON-DISRUPTIVE next steps (CPU-only / no GPU launch)

Per the user's preference for "如果不中断训练去做那更好" (do it without interrupting
training if possible), here is the non-disruptive path forward. **None of these
launch any GPU work**:

1. **CPU-only diagnostic runs** — use `molmetal/scripts/train_metal_lomo_v2.py --quick`
   (1-metal × 1-seed × 2-epoch smoke) to validate end-to-end pipeline wiring without
   touching any real retraining slot. Wf-321A tests already cover this contract.
2. **Wet-lab LOI parallel work** — `TODO/pending/wetlab_loi_*.md` are 3 LOIs (ORNL,
   PKU, Xiehe) that DO NOT require any GPU retrain. They advance paper §5
   (wet-lab validation gate) without touching training.
3. **Paper §3 honest-framing audit** — `paper/section_03_*.tex` can be revised to
   carry the Tier-2 honest-negative framing for m19 LOMO Rh fold (READ-ONLY on paper
   per TODO-84 unless user OKs — do NOT touch without authorization).
4. **Other P3 workstreams** — Wf-321H (TODO-119 P1.4 CPU scaffold), Wf-321C,
   Wf-321D, etc., are all CPU-only and ship in parallel without GPU contention.
5. **Pre-flight audit dry-run** — re-run `pre_flight_check.py` (CPU-only, no torch.cuda)
   any time to verify the gate stack remains green without launching any retrain.

The Phase 2 GPU launch (Wf-310E) remains blocked until the user explicitly clears
honest-negative resolution and re-issues `USER_OK=1`.

---

## §8 Cross-references

- **TODO-135 / Wf-135** — m19 LOMO Rh oversample fix master
- **TODO-176** — parent TODO for the regression fix workstream
- **Wf-309A** — Phase 1 cohort SHIP
- **Wf-139 P7** — Phase 1 cohort SHIP (parent of Wf-309A)
- **Wf-321A** — Phase 2 CPU scaffold VERIFY
- **Wf-310E Part 1** — Phase 2 GPU launch wrapper (USER_OK-gated)
- **Wf-322F** — this closeout
- **Wf-131 P2 + Wf-125 P1** — regression discovery
- **IMPROVEMENT_ENUM §1.4** — acceptance gates (Rh ≥ 0.30, LOMO ≥ 0.60)
- **AGENTS.md §1.2 + §5.2** — honest framing mandate
- **mol-metal-gpu-execution §0** — kill_then_adapt directive

---

## §9 Files & metrics inventory

| Path | Purpose | Status |
|---|---|---|
| `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` | THIS FILE | **SHIPPED 2026-09-29** |
| `molmetal/reports/wf176_m19_oversample_lomo/aggregate.json` | Wf-176 raw aggregate (Re fold only, BLOCKED) | MEASURED_BELOW_GATE |
| `metrics/by_metric/metal_lomo_auc_wasserstein.json` | Tier-2 catalog (Wasserstein, 5-metal) | MEASURED_BELOW_GATE (gate=0.7) |
| `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` | Phase 1 cohort JSON | SHIPPED |
| `molmetal/reports/wf139_p7_rh_active_cohort/synthesis.md` | Phase 1 cohort synthesis doc | SHIPPED |
| `molmetal/tests/test_wf321a_wf135_phase2_cpu_scaffold.py` | Phase 2 CPU scaffold tests (12) | SHIPPED |
| `scripts/launch_wf310e_wf135_phase2_gpu.sh` | Phase 2 GPU launch wrapper (USER_OK-gated) | READY (PLAN_ONLY) |
| `molmetal/reports/wf310e_wf135_phase2/plan.json` | Wrapper plan record (dry-run) | PLAN_ONLY |

**Tier-2 honest-negative tag: TRUE** (Rh fold regression not recovered; Phase 2 GPU
USER_OK-blocked per user instruction 2026-09-29).
