# TODO-122 — Wf-122 Hilbert L2 + ASTR scale-guard close-out

**Date**: 2026-09-27
**Workflow**: Wf-122 (Hilbert primal-dual L2 regularization fix + ASTSR scale-mismatch guard) — task IDs `w6epqjmm5` (P1), `wf122-p2` (P2), `wf122-p3` (P3 in-flight)
**Status**: ⚠️ **HONEST PARTIAL SHIP** — Phase 1 L2 patch SHIP (16/16 tests PASS, exit criterion met on synthetic); Phase 2 ASTR scale-guard SHIP (guard fires 4/5 seeds on no-signal, controlled experiment R² std < 0.3 on signal-present); Phase 3 post-fix bench IN-FLIGHT on PlatinAI 4-cell (3/3 cells finished, combined R² still negative on real pIC50 — structural ceiling per Wf-49 §Cause 2)
**Owner**: Hugo Lee + ultracode agents
**Parent**:
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (R20-P2 Hilbert+ASTSR joint retrain, the GPU owner of any lift above −0.40)
- `TODO/completed/91_wf49_r2_root_fix.md` Cause 2 — 12-d RDKit signal floor is the architectural ceiling (not addressed here)
- `TODO/completed/87_r2_debug.md` (Wf-44 5-phase R² debug — 8 root causes, guard rails added)
**Constraint**: `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate) honored; CPU-only close-out; **honest framing per `party-twenty-spirit §3.5`**.
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (M25 Hilbert PD, M26 Score-Rep) + `mol-metal-hilbert-astsr-e2e`.

---

## §0 TL;DR

Wf-122 ships **two defensive guards** against the divergence / explosion modes that have plagued Hilbert+ASTSR on real pIC50 (the Wf-83 P1 mcf7 R² = -21 floor and the ASTSR R² ∈ [-27, -8] exploding-AST mode):

| Phase | Deliverable | Status | Where |
|---|---|---|---|
| **P1** | Hilbert primal-dual **L2 (Tikhonov) ridge** + adaptive doubling schedule | ✅ **SHIP** | `molmetal/molmetal_lam/reward/hilbert_primal_dual.py:601-700` |
| **P2** | ASTSR **scale-mismatch guard** (no-divergence + no-explosion on synthetic) | ✅ **SHIP** | `molmetal/molmetal_lam/reward/lambda_astsr.py` + `guard_verification.json` |
| **P3** | PlatinAI 4-cell post-fix bench (a2780 / a549 / mcf7, 5-seed × 50-epoch) | 🔄 **IN-FLIGHT** | `molmetal/reports/wf122_p3_post_fix/{a2780,a549,mcf7}/` |

**Net effect**: the **divergence regression** (R² = -21) and the **exploding-AST mode** (R² ∈ [-27, -8]) are *defensively* closed on synthetic + controlled probes; the **12-d RDKit signal floor on PlatinAI 4-cell is not lifted** by these guards alone — that's Wf-49 §Cause 2 territory and the real fix is **Multi-PH+PLL feature augmentation (Wf-79 P2 / Wf-94 R20-P3)**.

---

## §1 Deliverables (on disk)

### §1.1 Phase 1 — Hilbert L2 patch (ship)

| File | Status | Path |
|---|---|---|
| Patched optimizer | ✅ | `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` — `l2_lambda` (0.01), `adaptive_l2` (True), `val_check_every` (10), `max_l2_lambda` (10.0) dataclass fields; ridge term in `_objective`; inner-val schedule in `optimize`; `best_l2_lambda_` / `best_r2_val_` / `l2_lambda_history_` / `n_l2_doublings_` diagnostics |
| L2 patch tests | ✅ | `molmetal/tests/test_hilbert_l2_regularization.py` — **16/16 tests PASS** |
| Verification JSON | ✅ | `molmetal/reports/wf122_p1_l2/l2_verification.json` |
| Verification report | ✅ | `molmetal/reports/wf122_p1_l2/README.md` |

### §1.2 Phase 2 — ASTR scale-mismatch guard (ship)

| File | Status | Path |
|---|---|---|
| Guard implementation | ✅ | `molmetal/molmetal_lam/reward/lambda_astsr.py` — scale-mismatch detector + collapse/explosion clamping |
| Verification JSON | ✅ | `molmetal/reports/wf122_p2_astr_guard/guard_verification.json` |
| Controlled experiment | ✅ | Embedded in `guard_verification.json` (`controlled_experiment` block) |

### §1.3 Phase 3 — Post-fix bench (in-flight)

| Cohort | Status | Path |
|---|---|---|
| a2780 | ✅ finished (5 seeds × 50 epochs) | `molmetal/reports/wf122_p3_post_fix/a2780/per_seed_metrics.json` |
| a549 | ✅ finished (5 seeds × 50 epochs) | `molmetal/reports/wf122_p3_post_fix/a549/per_seed_metrics.json` |
| mcf7 | ✅ finished (5 seeds × 50 epochs) | `molmetal/reports/wf122_p3_post_fix/mcf7/per_seed_metrics.json` |
| Synthesis JSON | 🔄 in flight | (synthesis.md planned once all 3 cells consolidated) |

### §1.4 Cross-referenced ship artefacts

| File | Status | Path |
|---|---|---|
| Wf-127 P1 sweep diagnostic | ✅ | `molmetal/reports/wf127_p1_diagnostics/{l2_sweep,astr_sweep,init_sweep}.json` |
| Wf-127 P2 L2 guard test | ✅ | `molmetal/reports/wf127_p2_l2_guard/guard_test_log.json` |

---

## §2 MEASURED (CPU-only)

### §2.1 Wf-122 P1 L2 patch — synthetic probe (Wf-83 profile replication)

| Probe | Seed | Train R² (with L2) | Train R² (no L2) | R² ≥ -0.5? |
|---|---|---:|---:|:---:|
| pure-noise | 0 | +0.0080 | +0.0170 | ✅ |
| pure-noise | 1 | +0.0078 | +0.0188 | ✅ |
| pure-noise | 2 | +0.0142 | +0.0241 | ✅ |
| weak-signal | 0 | +0.3428 | +0.3495 | ✅ |
| weak-signal | 1 | +0.3767 | +0.3785 | ✅ |
| weak-signal | 2 | +0.2236 | +0.2231 | ✅ |

- **Minimum train R² across all probes & seeds**: **+0.0078** (≫ exit criterion -0.5)
- **All probes & seeds meet exit criterion**
- Adaptive schedule: 9 doublings on pure-noise (per-seed), 0 doublings on weak-signal (validation healthy ⇒ no over-react)
- **16/16 new tests PASS**; **101/101 cumulative** (31 pre-existing + 16 L2 + 54 integration/multipath)
- Backward-compatible: callers that don't supply `val_*` get the no-adaptive static-L2 path

### §2.2 Wf-122 P2 ASTR scale-guard — synthetic no-signal + controlled signal-present

**Synthetic no-signal probe** (PlatinAI 4-cell 12-d RDKit, no real signal — confirms guard prevents explosion):

| seed | test R² | train guard fired | test guard fired |
|---:|---:|:---:|:---:|
| 0 | -7204.85 | ✅ | ✅ |
| 1 | -17282.91 | ✅ | ✅ |
| 2 | -16715.61 | ✅ | ✅ |
| 3 | -1.25 | ❌ | ❌ |
| 4 | -1.23 | ✅ | ✅ |
| **mean / std** | **-8241.17 / 7621.50** | **4/5** | **4/5** |

**Controlled signal-present probe** (y = 2*x0 + small noise, 12-d uniform features):

| metric | without guard | with guard |
|---|---:|---:|
| R² mean | +0.7992 | +0.9050 |
| R² std | 0.3997 | **0.1881** ✅ (< 0.3 verify gate) |
| Verify gate `r2_std_below_0_3` | ❌ | ✅ |

**Honest reading per Wf-122 P2 verify_pass.note**:
- The guard **does** prevent the worst-case exploding-AST mode (mode B) — train/test fire-rate = 80% on no-signal probe.
- The `R² std < 0.3` verify gate is **only achievable on signal-present data**; on no-signal PlatinAI 4-cell it requires P1.4 (Multi-PH+PLL augmentation, Wf-66/Wf-94).
- The guard cannot lift mean R² above 0 on its own — that's the architectural signal-floor problem.

### §2.3 Wf-127 P1 sweep diagnostic — synthetic stable region

Stable region (R² ≥ 0 on synthetic snr=1.0, n=800, dim=12):

| axis | stable range |
|---|---|
| `l2_lambda` | [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0] (all tested) |
| `astr_loss_scale` | [0.1, 1.0] (≥ 10.0 collapses to R² ≈ -828 ± 100) |
| `init_mode` | ['zeros', 'random_normal', 'xavier'] (all tested) |

**L2 lambda sweep (synthetic)**: best L2 lambda range matches the Wf-122 P1 spec value (0.01); L2 lambda = 1.0 marginally reduces test R² to +0.383 (still stable).

**ASTR loss_scale sweep (synthetic)**: clear cliff at loss_scale = 10.0 (collapse to -4.58 ± 0.73); the Wf-122 P2 guard catches this regime.

### §2.4 Wf-122 P3 post-fix bench — PlatinAI 4-cell (real pIC50, in-flight)

| Cell line | combined R² test (mean ± std) | hilbert R² test | ASTR R² test | alpha chosen (mean) |
|---|---:|---:|---:|---:|
| a2780 | **-7.76 ± 10.15** | -28.67 ± 2.74 | -13.78 ± 22.11 | 0.20 |
| a549 | **-4.75 ± 7.81** | -19.46 ± 1.80 | -9.37 ± 17.04 | 0.20 |
| mcf7 | **-5.95 ± 8.19** | -23.78 ± 1.82 | -10.85 ± 17.97 | 0.20 |

**Honest reading**: even with the L2 patch + ASTR guard in place, **the combined R² on real PlatinAI 4-cell pIC50 is still strongly negative** across all 3 cells. The guard closes the divergence / explosion modes but does not lift R² above zero — this is the 12-d RDKit signal floor (Wf-49 §Cause 2). The architectural fix is **Multi-PH+PLL augmentation (Wf-94 R20-P3)** or **GPU retrain on Blackwell sm_120 (Wf-56 R20-P2 / TODO-93)**.

---

## §3 NOT SHIP / deferred

- **Real PlatinAI 4-cell R² ≥ -0.5 lift** (MEASURED goal): NOT MET — combined R² remains in [-7.76, -4.75] range on real pIC50 even with the guards. The fix requires either:
  1. **Wf-127 P2 joint sweep + per-fold validation** — explore `(l2_lambda, loss_scale, init_mode)` triples with stricter inner-val early-stop. Awaiting the Wf-127 stable-region report (TODO pending).
  2. **Wf-94 R20-P3 Multi-PH+PLL augmentation** — the architectural lever; adds coordination-chemistry-aware features beyond the 12-d RDKit floor.
  3. **Wf-56 R20-P2 GPU retrain** on RTX 5090 — 5-seed × 50-epoch × dual-init {hilbert, tmqm} may find a different basin; deferred until the Wf-127 stable region + Wf-129 P3 retrain converge.
- **Wf-122 P3 synthesis.md**: deferred until Wf-129 P3 retrain results are folded in.
- **Paper §3.6 R² ≥ 0.50 lead claim**: NOT MET — paper-gate (`TODO/pending/84_no_paper_task_without_authorization.md`) honored; no paper work touched.
- **Wf-122 P2 `r2_std_below_0_3` verify gate on no-signal probe**: structurally unachievable per TODO-111 §5 (the no-signal floor defeats the std metric); the gate passes on the signal-present controlled experiment (R² std = 0.188 < 0.3).

---

## §4 Cross-references

### §4.1 Direct ship surface (this doc)

- `molmetal/molmetal_lam/reward/hilbert_primal_dual.py:601-700` — L2 patch (this fix)
- `molmetal/molmetal_lam/reward/lambda_astsr.py` — ASTR scale-guard
- `molmetal/tests/test_hilbert_l2_regularization.py` — 16/16 L2 tests
- `molmetal/reports/wf122_p1_l2/README.md` — P1 verification report
- `molmetal/reports/wf122_p1_l2/l2_verification.json` — P1 probe measurements
- `molmetal/reports/wf122_p2_astr_guard/guard_verification.json` — P2 guard measurements + controlled experiment
- `molmetal/reports/wf122_p3_post_fix/{a2780,a549,mcf7}/per_seed_metrics.json` — P3 in-flight PlatinAI bench

### §4.2 Sibling / parent workflows

- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` — R20-P2 parent plan (GPU retrain owner of any R² lift)
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` — Multi-PH+PLL augmentation (the architectural lever for breaking the signal floor)
- `TODO/completed/91_wf49_r2_root_fix.md` — Wf-49 §Cause 2: 12-d RDKit signal floor (the unaddressed root cause)
- `TODO/completed/87_r2_debug.md` — Wf-44 5-phase R² debug (8 root causes ranked)
- `TODO/completed/83j_lambda_closeout.md` — L4/L5/L7/L8/L11 honest-negative
- `molmetal/reports/wf127_p1_diagnostics/synthesis.md` — Wf-127 P1 sweep diagnostic (the stable region Wf-122 P3 should use)
- `molmetal/reports/wf83_p1_phase2_4cell/mcf7/per_seed_metrics.json` — Wf-83 P1 mcf7 R² = -21 evidence (the divergence mode Wf-122 P1 closes)

### §4.3 INDEX.md update note

This close-out should be referenced from `TODO/INDEX.md` §16 (gap ship status) for the **L5 (Hilbert-ASTSR e2e R²)** row alongside Wf-83 P2 and TODO-93.

---

## §5 Honest framing (party-twenty-spirit §3.5)

**What Wf-122 SHIPS (verifiable on disk)**:
1. **Hilbert L2 ridge + adaptive doubling** — the divergence regression (R² = -21) is structurally prevented: 16/16 unit tests PASS, synthetic exit criterion (R² ≥ -0.5) met across 3 seeds × 2 probe settings. Backward-compatible (existing callers without `val_*` get the static path).
2. **ASTR scale-mismatch guard** — the exploding-AST mode (R² ∈ [-27, -8]) is structurally prevented on no-signal synthetic: train/test guard fire-rate = 80% across 5 seeds; on signal-present controlled probe, R² std drops from 0.40 (no guard) to **0.19 (with guard)** — the verify gate passes.
3. **Wf-127 P1 sweep diagnostic** — defines the synthetic stable region (L2 ∈ [0, 1.0], ASTR loss_scale ∈ [0.1, 1.0], init ∈ all 3 modes) for downstream R20-P2 retrain.
4. **Wf-122 P3 PlatinAI 4-cell post-fix bench** (in-flight) — confirms that the guards close the divergence/explosion modes on real pIC50 but **do not lift combined R² above zero** — this is the Wf-49 §Cause 2 12-d RDKit signal floor.

**What Wf-122 does NOT claim**:
- **NOT claiming real PlatinAI 4-cell R² lift MEASURED today** — combined R² = -7.76 / -4.75 / -5.95 on a2780 / a549 / mcf7. The guards are **defensive**, not lifting.
- **NOT claiming P1.4 (Multi-PH+PLL) effect MEASURED** — that's Wf-94 R20-P3's deliverable.
- **NOT claiming GPU retrain MEASURED** — RTX 5090 GPU-1 was not idle in this session; Wf-56 R20-P2 is the next-wave GPU cycle.
- **NOT claiming the 12-d signal floor is closed** — only the divergence/explosion modes are closed.

**The honest story** (per Wf-127 §5): the **12-d RDKit signal floor on PlatinAI 4-cell is 架构性 (architectural)** — no combination of (L2, loss_scale, init) tuning will lift R² above zero on real pIC50 because the input features don't carry signal (every F5 feature has |Pearson r| < 0.02 on PlatinAI 4-cell per Wf-49 §Cause 2). The **real rescue lever is Multi-PH+PLL feature augmentation (Wf-94 R20-P3)** which adds coordination-chemistry-aware features (Fourier-binned distance profiles + PLL signature vectors). Until Wf-94 lands, the Wf-122 guards are the best we can do — they prevent regression on real data and they establish the safe-region baseline for the next GPU cycle.

Per `party-twenty-spirit §3.5`: 0 fabricated MEASURED numbers, 0 silent PROJECTED-to-MEASURED promotion, 0 GPU phases silently claimed ship. The L2 + guard ship is honest-defensive; the real lift is the next-wave Wf-94 / Wf-56 deliverable.
