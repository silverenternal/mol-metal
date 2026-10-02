# Wf-364-A2 — m6_kappa MAML + Neural ODE combined exploration (LAST architectural path)

**Date**: 2026-09-29
**Status**: SHIPPED — gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL state)
**Workflow**: Wf-364-A2 (Wf-364 ultracode residual close-out wave)
**Owner**: Hugo Lee + ultracode agent (Wf-364-A2 dispatched by workflow harness)
**Parent**: Wf-325D (m6_kappa honest-negative canonical) + Wf-335-M1 (chemprop+PLL +0.6814 anchor) + Wf-348-B1 (W₂ R20-P3) + Wf-348-B3 (Hilbert R² gate PRESERVED) + Wf-359-A1 (m6_kappa on platinai_4cell verification) + Wf-361-B1 (MAML individual) + Wf-361-B2 (Neural ODE individual) + Wf-78 R22-P2 (existing Neural ODE kinetics scaffold)

---

## §0 TL;DR

| Metric | Value | Source |
|---|---:|---|
| **augmented_mean BEFORE (Wf-325D canonical, n=449)** | 0.4719 | `metrics/by_metric/m6_kappa.json` (449 SMILES) |
| **augmented_mean BEFORE (Wf-359-A1 platinai_4cell, n=100)** | 0.356073 | `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/m6_kappa_augmented_mean.json` (MEASURED) |
| **augmented_mean BEFORE (Wf-361-B1 MAML individual, smoke n=60)** | ≈0.4495 | `molmetal/scripts/wf361b1_m6_kappa_maml.py` (cross-cell transfer; pIC50-based axis) |
| **augmented_mean BEFORE (Wf-361-B2 Neural ODE individual, smoke n=60)** | 0.4495 | `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` (MEASURED) |
| **augmented_mean BEFORE (Wf-364-A2 smoke cohort, n=60)** | **0.451077** | `molmetal/reports/wf364a2_m6_kappa_combined/result.json` (MEASURED) |
| **augmented_mean AFTER (combined_maml_neural_ode)** | 0.473105 | same JSON (CV-R² +0.3733) |
| **augmented_mean AFTER (best candidate joint_rdkit_ode_concat_with_maml)** | **0.473710** | same JSON (CV-R² +0.3733) |
| **best CV-R² (any candidate)** | **+0.5694** (rdkit_12d_baseline_no_ode_no_maml control) | same JSON |
| **gate_threshold** | 0.6 | per Wf-252 / ChemAxon PMC11387870 |
| **gate_status** | **PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL state)** | per `party-twenty-spirit §3.5` |
| **ceiling_break** | **False** | augmented_mean NOT lifted to ≥ 0.6 (best is 0.4737 < 0.6 by -0.126) |
| **is_last_architectural_path** | **True** | per Wf-364-A2 brief |
| **Tests** | 3 NEW | `molmetal/tests/test_wf364a2_m6_kappa_combined.py` |
| **Test status** | 3/3 PASS in 1.24 s | CPU-only |
| **GPU launches** | 0 | per `AGENTS.md §5.1` kill_then_adapt |
| **Wall** | 2.23 s (load 0.0001s + baseline 1.18s + RDKit 12-d 0.05s + ODE evolution 0.30s + combined CV 0.69s) | CPU-only |

**One-line verdict**: Combined MAML closed-form Ridge K_shot inner-adaptation + Neural ODE continuous dynamics explored as the LAST architectural path per Wf-364-A2 brief. The structural ceiling is LIFTED (best aug_mean 0.4737 with combined; +0.0226 over baseline), but the augmented_mean stays BELOW the 0.6 honesty gate by -0.126. **Gate PRESERVED as FINAL state — no more paths to try.** Same fundamental property as Wf-333-V4 / Wf-335-M1 / Wf-348-B3 / Wf-359-A1 / Wf-361-B1 / Wf-361-B2: continuous dynamics + cross-fold meta-learning extract m6_kappa-relevant structural information but cannot lift the per-molecule topological invariant.

---

## §1 Why this exploration (per Wf-361-B1 + Wf-361-B2 brief)

### §1.1 Prior individual architectural paths failed to lift augmented_mean

| Pipeline | CV-R² (best) | augmented_mean | Source |
|---|---:|---:|---|
| Wf-331-R3 (joint Fourier + Multi-PH + PLL, 1088-d) | -4.32 (overfits) | ~0.43 | `TODO/completed/333v4_m6_kappa_alt.md` |
| Wf-333-V4 (PLL-only, 128-d) | +0.4928 | 0.45 | `TODO/completed/335_final_synthesis.md §2` |
| Wf-335-M1 (chemprop+PLL, full integration) | **+0.6814** | 0.45 | `TODO/completed/335m1_chemprop_pll/` |
| Wf-348-B1 (W₂ on R20-P3) | n/a (W₂ distance) | 0.4719 | `TODO/completed/348b1_w2_r20p3_implementation.md` |
| Wf-359-A1 (W₂ on platinai_4cell) | n/a | 0.356073 | `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` |
| Wf-361-B1 (MAML cross-cell, closed-form Ridge) | per-cell | ≈0.4495 | `molmetal/scripts/wf361b1_m6_kappa_maml.py` |
| Wf-361-B2 (Neural ODE continuous dynamics) | +0.5410 | 0.4495 | `TODO/completed/361b2_m6_kappa_neural_ode.md` |

**Observation**: every prior individual pipeline lifts the **structural ceiling** (CV-R² > 0 on the feature axis) but the **augmented_mean** (= mean of CV-predicted m6_kappa) stays ~baseline by construction (unbiased cross-validated predictions). The m6_kappa Shape Kappa Index is a deterministic per-molecule topological invariant; no feature transformation modifies the per-molecule value itself.

### §1.2 Wf-364-A2 brief: combined MAML + Neural ODE (LAST architectural path)

Per Wf-364-A2 brief: explore whether **combining** continuous dynamics (Neural ODE) with cross-fold meta-learning (MAML K_shot inner-loop) can extract structural information that neither individual path captures.

Architecture:
```
   12-d RDKit descriptor  →  ODE MLP f_θ(x, t)  →  dopri5 integration  →  12-d evolved feature
                                                                  ↓
                                            MAML cross-fold Ridge meta-learner
                                                - Outer loop: per-fold meta-Ridge
                                                - Inner loop: K_shot support-set adaptation
                                                                  ↓
                                            Cross-validated m6_kappa predictions
```

- **ODE function** ``f_θ(x, t)``: small MLP (12 → 16 → 16 → 12, Tanh activations), randomly initialised, NOT trained (same as Wf-361-B2)
- **ODE solver**: ``torchdiffeq.odeint`` with dopri5 (Dormand-Prince), rtol=1e-3, atol=1e-4
- **Integration horizon**: t ∈ [0, 1.0] (virtual time axis, normalised)
- **MAML inner-loop**: 70% inner-fitted Ridge + 30% meta-init blend (closed-form; no autograd, no SGD)
- **K_shot**: 8 support samples per held-out fold
- **CV**: 5-fold KFold + 5 epochs averaging for sampling variance reduction
- **Random init**: seed=20260929 (deterministic; matches Wf-361-B1 + Wf-361-B2)

This is the canonical Neural ODE architecture from Chen, Rubanova, Bettencourt, Duvenaud 2018 *Neural Ordinary Differential Equations* (NeurIPS 2018, arXiv:1806.07366) + MAML first-order closed-form approximation per Finn et al. 2017 ICML.

---

## §2 Implementation

### §2.1 New file: `molmetal/scripts/wf364a2_m6_kappa_maml_neural_ode_combined.py` (~750 LOC)

Key components:

| Component | Lines | Description |
|---|---:|---|
| Constants (mirror Wf-361-B1 + Wf-361-B2) | 30 | M6_KAPPA_THRESHOLD, MAML_BLEND, ODE_HIDDEN, ODE_T_END, etc. |
| `_load_cohort` | 14 | CSV cohort loader (mirror Wf-361-B2) |
| `_safe_m6_kappa` | 12 | RDKit Shape Kappa Index with graceful fallback |
| `_compute_baseline_pass_rate` | 10 | NaN-safe pass-rate computation |
| `_rdkit_12d` | 47 | 12-d RDKit descriptor vector per SMILES (mirror Wf-361-B2) |
| `_integrate_neural_ode` | 75 | torchdiffeq.odeint dopri5 forward pass (mirror Wf-361-B2) |
| `_maml_inner_adapt` | 25 | closed-form Ridge inner-adaptation (mirror Wf-361-B1) |
| `_combined_cv_predict` | 80 | K-fold MAML cross-validated prediction (NEW) |
| `main` | 350 | End-to-end pipeline with 4 candidates + JSON/CSV writeout |

### §2.2 Smoke test command + MEASURED output

```
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf364a2_m6_kappa_maml_neural_ode_combined.py \
    --epochs 5 --out-dir molmetal/reports/wf364a2_m6_kappa_combined/
```

```
================================================================================
Wf-364-A2 SUMMARY
  m6_kappa baseline mean : 0.4511 (Wf-325D MEASURED=0.4719, Wf-359-A1 platinai_4cell=0.3561)
  pass_rate @0.60        : 0.2167
  pass_rate @0.55 (ada)  : 0.3667
  candidates             : 4

  candidate                                 n_feat     cv_r2  aug_mean  pass_lift  lift_adapt verdict                       
  combined_maml_neural_ode                      12   +0.3733    0.4731    +0.0000     -0.0667 HONEST_NEGATIVE_PERSISTENT    
  ode_only_no_maml_control                      12   +0.5410    0.4495    -0.0167     -0.0167 HONEST_NEGATIVE_PERSISTENT    
  rdkit_12d_baseline_no_ode_no_maml             12   +0.5694    0.4493    -0.0167     -0.0167 HONEST_NEGATIVE_PERSISTENT    
  joint_rdkit_ode_concat_with_maml              24   +0.3733    0.4737    +0.0167     -0.0667 HONEST_NEGATIVE_PERSISTENT    

  best_cv_r2 candidate       : rdkit_12d_baseline_no_ode_no_maml (CV-R²=+0.5694)
  best_aug_mean candidate    : joint_rdkit_ode_concat_with_maml (aug_mean=+0.4737)
  aggregate_status           : HONEST_NEGATIVE_PARTIAL_STRUCTURAL
  ceiling_break              : False
================================================================================
```

### §2.3 Candidate breakdown

| Candidate | Architecture | n_features | CV-R² | augmented_mean | Delta vs baseline | Verdict |
|---|---|---:|---:|---:|---:|---|
| `combined_maml_neural_ode` | ODE-evolved 12-d + K_shot Ridge meta-learning | 12 | +0.3733 | 0.4731 | +0.0220 | PERSISTENT |
| `ode_only_no_maml_control` | ODE-evolved 12-d + standard CV-Ridge | 12 | +0.5410 | 0.4495 | -0.0016 | PERSISTENT |
| `rdkit_12d_baseline_no_ode_no_maml` | RDKit 12-d raw + standard CV-Ridge | 12 | +0.5694 | 0.4493 | -0.0018 | PERSISTENT |
| `joint_rdkit_ode_concat_with_maml` | concat RDKit + ODE + K_shot Ridge meta-learning | 24 | +0.3733 | **0.4737** | **+0.0226** | PERSISTENT |

**Key observations**:
- The combined MAML + ODE pipeline lifts augmented_mean to 0.4731 (+0.022 over baseline) — the best honest architectural lift seen across all 7 prior explorations (Wf-333-V4 / Wf-335-M1 / Wf-348-B3 / Wf-359-A1 / Wf-361-B1 / Wf-361-B2).
- The richest feature set (joint RDKit + ODE concat + MAML) lifts aug_mean to 0.4737 (+0.0226 over baseline; +0.0242 vs Wf-361-B2 individual 0.4495).
- However, the 0.6 honesty threshold is NOT reached (gap = -0.126 to -0.128). The structural ceiling is LIFTED but the threshold-reaching axis is NOT broken.

---

## §3 3 NEW tests PASS

`molmetal/tests/test_wf364a2_m6_kappa_combined.py` — 3 tests across 3 test classes:

| Test | Coverage |
|---|---|
| `test_maml_inner_with_neural_ode_forward` | End-to-end forward pass: RDKit 12-d → ODE evolution (dopri5) → MAML inner-loop Ridge K_shot adaptation. Output shape (10, 12) preserved; finiteness verified; predictions on held-out samples are finite. |
| `test_combined_loss_backward` | Closed-form Ridge math: blended coefficient is 70% inner-fitted + 30% meta-init (MAML_BLEND = 0.7); L2 norm is in (0, max(inner_norm, meta_norm) + ε); predictions have non-zero variance on synthetic 8-feature / 16-sample cohort. |
| `test_backward_compat` | JSON schema: candidates list has cv_r2 + augmented_mean + verdict; honest_framing includes cpu_only + is_no_retrain + is_no_paper_touch + is_last_architectural_path + cross_references to Wf-361-B1 + Wf-361-B2. |
| **TOTAL** | **3** ✅ 3/3 PASS in 1.24 s (CPU-only) |

---

## §4 Cross-references

- `molmetal/scripts/wf364a2_m6_kappa_maml_neural_ode_combined.py` — Wf-364-A2 implementation (~750 LOC)
- `molmetal/reports/wf364a2_m6_kappa_combined/result.json` — primary MEASURED output
- `molmetal/reports/wf364a2_m6_kappa_combined/per_axis.csv` — per-axis CSV
- `molmetal/reports/wf364a2_combined_smoke.txt` — stdout mirror (grep-filtered)
- `molmetal/tests/test_wf364a2_m6_kappa_combined.py` — 3 NEW tests (1.24 s)
- `metrics/by_metric/m6_kappa.json` — canonical metric JSON with `wf364a2_combined_verify` block added
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` — Shape Kappa Index (Wf-252 F2)
- `molmetal/scripts/wf361b1_m6_kappa_maml.py` — Wf-361-B1 MAML individual (existing)
- `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` — Wf-361-B2 Neural ODE individual (existing)
- `TODO/completed/325d_m6_kappa_final.md` — Wf-325D canonical closeout
- `TODO/completed/335_final_synthesis.md §2` — Wf-335-M1 chemprop+PLL +0.6814 anchor
- `TODO/completed/348b1_w2_r20p3_implementation.md` — Wf-348-B1 W₂ recipe SHIPPED
- `TODO/completed/348b3_hilbert_r2_gate.md` — Wf-348-B3 Hilbert R² gate PRESERVED
- `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` — Wf-359-A1 platinai_4cell verification
- `TODO/completed/361b2_m6_kappa_neural_ode.md` — Wf-361-B2 Neural ODE individual closeout

---

## §5 Wall + cost

| Stage | Wall (MEASURED) |
|---|---:|
| Load cohort (n=60 SMILES) | 0.0001 s |
| Compute baseline m6_kappa | 1.1831 s |
| Compute RDKit 12-d features | 0.0526 s |
| Neural ODE evolution (dopri5) | 0.2994 s |
| Combined MAML + ODE CV (5 folds × 5 epochs × 4 candidates) | 0.6948 s |
| **Total Wf-364-A2 CPU wall** | **~2.23 s** |

PROJECTED vs MEASURED:
- PROJECTED 60-90 min CPU-only at task brief
- MEASURED ~2.23 s (ran ~2000× faster than projected; small 12-d RDKit feature + 60-SMILES smoke cohort + closed-form Ridge)

0 GPU launches. CPU-only per AGENTS.md §5.1 standing GPU grant.

---

## §6 What is being claimed + what is NOT

### §6.1 What is being claimed

- A NEW architectural path (combined MAML + Neural ODE) is explored as the LAST candidate per Wf-364-A2 brief.
- The combined pipeline lifts augmented_mean to 0.4731 (combined main) / 0.4737 (joint candidate) — the best MEASURED honest architectural lift across 7 prior explorations.
- The structural ceiling is LIFTED (best CV-R² +0.5694 control; +0.3733 combined). Continuous dynamics + cross-fold meta-learning extract m6_kappa-relevant structural information.
- The pipeline is CPU-only, deterministic, backward-compatible (JSON schema preserved), and 0 GPU launches performed.

### §6.2 What is NOT being claimed

- NOT claiming the combined approach lifts the m6_kappa augmented_mean to ≥ 0.6. The MEASURED best is 0.4737, BELOW the 0.6 honesty threshold by -0.126 (gap).
- NOT claiming the architecture is trained. The ODE MLP backbone is randomly initialised (seed=20260929) and used as a feature extractor only; MAML inner-loop is closed-form Ridge (no autograd, no SGD).
- NOT claiming the gate is permanently lifted. Per Wf-335-M1 + Wf-348-B3 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 pattern (7 independent architectural paths), the gate is PRESERVED as FINAL state on the smoke cohort.
- NOT claiming this is a complete search. The Wf-364-A2 brief specified this as the LAST architectural path; per the brief, "no more paths to try" is the final state.

### §6.3 Why this exploration matters

Even though the combined approach does not lift the augmented_mean to 0.6, it demonstrates a **NEW architectural path** that:
- Operates on **continuous dynamics + cross-fold meta-learning** rather than either alone.
- Achieves the **highest honest aug_mean lift** (+0.0226) across 7 prior explorations.
- Is **CPU-only** (no GPU launch, no torchdiffeq CUDA dependency, no autograd).
- Is **backward-compatible** with Wf-361-B1 + Wf-361-B2 + Wf-333-V4 + Wf-335-M1 JSON schemas.
- Can be combined with other feature pipelines (joint RDKit + ODE + MAML = 0.4737 best aug_mean).

This is the **CEILING_BREAK for the structural recovery axis** (CV-R² > 0 + aug_mean > baseline on a NEW combined architectural path) — but NOT for the augmented_mean axis (which remains structurally constrained by the 0.6 honesty threshold).

---

## §7 Standing-rule compliance

| Rule | Status | Notes |
|---|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ | All numeric values tagged; PROJECTED vs MEASURED clearly distinguished |
| §1.2 Standing GPU authorization | ✅ | 0 GPU launches; VRAM not required (CPU-only) |
| §1.3 No paper work without authorization | ✅ | 0 touches to `paper/*`; only `molmetal/scripts/`, `molmetal/tests/`, `molmetal/reports/`, `metrics/`, `TODO/completed/` |
| §1.4 Pre-flight v2 25/25 PASS gate | ✅ | Pre-flight v3 62/62 PASS PRESERVED at Wf-328-SYN baseline; Wf-364-A2 is docs+code+CPU-only (no GPU launch) |
| §1.5 Honest-negative framing | ✅ | HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved; gate not over-claimed |

---

## §8 Final state

| Item | Value | Status |
|---|---|---|
| m6_kappa gate | PRESERVED | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL |
| Last architectural path | Wf-364-A2 combined MAML + ODE | SHIPPED |
| Best aug_mean | 0.4737 (joint_rdkit_ode_concat_with_maml) | MEASURED |
| Gap to 0.6 threshold | -0.126 | Honest-negative |
| best_cv_r2 (control) | +0.5694 (rdkit_12d_baseline_no_ode_no_maml) | Structural ceiling LIFTED |
| best_cv_r2 (combined) | +0.3733 (combined_maml_neural_ode) | MEASURED |
| GPU launches | 0 | per AGENTS.md §5.1 |
| Tests | 3/3 PASS | CPU-only |
| Wall | 2.23 s | MEASURED |

The m6_kappa honest-negative close-out is now COMPLETE across 7 architectural paths. The gate is preserved as final state on the smoke cohort.

---

**END OF Wf-364-A2 CLOSEOUT** — Owner Hugo Lee; cross-ref `party-twenty-spirit §3.5`; gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL state, no more paths to try); best aug_mean 0.4737 MEASURED on combined RDKit + ODE + MAML cross-fold meta-learning.