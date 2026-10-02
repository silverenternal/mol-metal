# Wf-310F Day 7 — GPU Production Wire-in Launch Plan

**Date**: 2026-09-29
**Workflow**: Wf-310F Day 7 (TODO-234 frontier-math GPU production wire-in)
**Owner**: Hugo Lee (ultracode agent)
**Parent**: `TODO/completed/234_wf234_frontier_math_cpu_plan.md` §1.7 + §2.1
**Launch wrapper**: `molmetal/scripts/launch_wf310f_frontier_math_gpu.sh`
**Status**: PLAN SHIPPED — runtime DEFERRED to USER_OK gate (`WF310F_NEXT_WAVE={A,B,C}` or `--yes` for dry-run).

---

## §0 TL;DR

Three sequential GPU waves that wire `FrontierMathIntegration` (Wf-309B ship) into the R18 trainer surfaces. Each wave is gated on USER_OK (`kill_then_adapt` semantics, per-wave < 2 hr).

| Wave | GPU | Wall | Purpose | Status |
|---|---|---|---|---|
| A | cuda:0 (PRO 6000 Blackwell) | ~1.5 hr | Wires `augment_pocket_embedding` (W2.3) into CFM trainer | READY (USER_OK gate) |
| B | cuda:1 (RTX 5090) | ~0.5 hr | Wires `SE3LipmanSolver` (Integration 1) into D-MPNN multi-task | READY (USER_OK gate) |
| C | cuda:0 (PRO 6000 Blackwell) | ~2-3 hr | Joint sweep of all 4 frontier adapters | READY (USER_OK gate, 2 sub-waves) |

**One-line summary**: The CPU portion (Day 1-2 wrapper + smoke tests) shipped in Wf-310F; the GPU production wire-in (Day 3-7) is wrapped in `launch_wf310f_frontier_math_gpu.sh` and gated on USER_OK per `kill_then_adapt`.

---

## §1 Pre-launch checks (BLOCKER)

Per `TODO/completed/207_unified_pre_flight_checklist_v2.md` §1 + Wf-310F §1.7:

1. **`pre_flight_check.py --pipeline all`** — target 36/36 PASS / 0 FAIL / 0 SKIP. Verified at 36/36 PASS in Wf-310F CPU run.
2. **Trainer surface check** — `train_cfm_yuelbond_wasserstein.py` must expose `--frontier-multi-ph` flag (Day 3-4 deliverable, deferred); `train_dmpnn_multitask_wasserstein.py` must expose `--frontier-se3` flag (Day 5-6 deliverable, deferred).
3. **USER_OK gate** — `WF310F_NEXT_WAVE={A,B,C}` must be set before each wave.

---

## §2 Wave A — CFM frontier-multi-ph wire-in (~1.5 hr)

**Goal**: Run the P1.4 CFM smoke with `augment_pocket_embedding` (W2.3 helper) feeding the augmented `(B, H + n_bins²)` pocket embedding into the velocity field.

**Trainer**: `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
**Config**: `configs/cfm_per_innovation_ablation_p1.4.yaml`
**Flags**: `--epochs 3 --seed 0 --pocket-id pocket_a --frontier-multi-ph --frontier-multi-ph-bins 8`
**Output dir**: `molmetal/reports/wf310f_frontier_math_gpu/wave_a/frontier_multiph/`
**Expected metrics** (PROJECTED):
* `decode_ratio ≥ 0.05` (cfm frontier track acceptance floor)
* `test_auc ≥ 0.7` (P1.4 baseline anchor)
* Wall-time < 1.5 hr

**Abort criteria** (per TODO-207 v2 §7.2):
* Wall > 1.5 hr → kill, abort_recovery
* `decode_ratio < 0.05` for 3+ consecutive epochs → kill
* `test_auc < 0.51` on Ru/Ir → kill (cfm frontier collapse)

---

## §3 Wave B — D-MPNN multi-task frontier-SE3 wire-in (~0.5 hr)

**Goal**: Run the G6 no_wass smoke with `SE3LipmanSolver` (Integration 1) wrapping the multi-task manifold.

**Trainer**: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
**Config**: `configs/wf124_p3_per_innovation_ablation_g6.yaml`
**Flags**: `--metals Ru --seeds 0 --epochs 10 --batch-size 64 --lr 1e-3 --alpha 0.5 --wasserstein-weight 0.0 --device cuda:1 --frontier-se3`
**Output dir**: `molmetal/reports/wf310f_frontier_math_gpu/wave_b/frontier_se3_ru/`
**Expected metrics** (PROJECTED):
* `test_auc ≥ 0.7` on Ru (G6 no_wass baseline = 0.7613, Wf-124 P3)
* Wall-time < 0.5 hr (10 epochs × Ru × 1 seed smoke)

**Abort criteria**:
* Wall > 0.5 hr → kill
* `test_auc < 0.51` on Ru → kill

---

## §4 Wave C — Joint FrontierMathIntegration paper-grade sweep (~2-3 hr)

**Goal**: Joint sweep of all 4 frontier adapters (`FrontierMathIntegration.get_multi_ph_features + get_se3_lipman_solver + get_se3_sb_decoder + get_transport_relaxation`) on a paper-grade subset.

**Trainer**: joint (`train_cfm_yuelbond_wasserstein.py` + `train_dmpnn_multitask_wasserstein.py`)
**Sub-wave 1** (~1.5 hr): `Multi-PH + Transport Relaxation` augmented training.
**Sub-wave 2** (~1.0 hr): `SE(3) Lipman + SE(3) SB` augmented training.

**Honest framing**: PROJECTED wall-times only; MEASURED after each sub-wave runs.

---

## §5 USER_OK gate semantics

The launch wrapper enforces a USER_OK gate between waves (per `kill_then_adapt` standing directive):

```bash
# Default behavior: USER_OK required between waves
bash molmetal/scripts/launch_wf310f_frontier_math_gpu.sh --wave A
# → blocks, waits for: export WF310F_NEXT_WAVE=A

# Dry-run (skip GPU live verify)
bash molmetal/scripts/launch_wf310f_frontier_math_gpu.sh --wave A --dry-run

# CI / smoke (auto-OK all gates)
bash molmetal/scripts/launch_wf310f_frontier_math_gpu.sh --wave all --yes
```

The default invocation `bash launch_wf310f_frontier_math_gpu.sh --wave all` will:
1. Run `pre_flight_check.py` (BLOCKER).
2. Wave A (waits for `WF310F_NEXT_WAVE=A`).
3. Wave B (waits for `WF310F_NEXT_WAVE=B`).
4. Wave C (waits for `WF310F_NEXT_WAVE=C`).

---

## §6 Deliverables shipped in Wf-310F

| Artifact | Path | Status |
|---|---|---|
| Plan doc | `TODO/completed/234_wf234_frontier_math_cpu_plan.md` | SHIPPED |
| Wrapper class | `molmetal/adapters/frontier_math_integration.py` | PRE-EXISTING (Wf-309B) |
| Smoke tests | `molmetal/tests/test_wf310f_frontier_math_integration.py` | SHIPPED (4/4 PASS, 0.13 s) |
| Pre-flight | 36/36 PASS, 0 FAIL, 0 SKIP | VERIFIED |
| GPU launch wrapper | `molmetal/scripts/launch_wf310f_frontier_math_gpu.sh` | SHIPPED |
| GPU launch plan doc | `molmetal/reports/wf310f_frontier_math_gpu/launch_plan.md` (this file) | SHIPPED |
| Day 3-7 runtime | (TBD) | DEFERRED to USER_OK |

---

## §7 Cross-references

* Wf-310F plan doc: `TODO/completed/234_wf234_frontier_math_cpu_plan.md`
* Wf-307D CPU close-out: `TODO/completed/234_wf234_frontier_math_cpu_portion.md`
* Wf-309B verification: `molmetal/tests/test_wf309b_frontier_math_integration.py`
* TODO-234 audit: `TODO/pending/234_wf234_frontier_math_audit_integration.md`
* R18 plan: `TODO/03_plans/R18_PLAN.md`
* kill_then_adapt directive: `TODO/INDEX.md` BD-5
* Pre-flight: `molmetal/scripts/pre_flight_check.py` + `TODO/completed/207_unified_pre_flight_checklist_v2.md`
* Wf-299C sibling launch wrapper: `molmetal/scripts/launch_wf299_cfm_main.sh` (template)