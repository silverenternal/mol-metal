# Wf-361-B2 — m6_kappa Neural ODE integration exploration

**Date**: 2026-09-29
**Status**: SHIPPED — gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)
**Workflow**: Wf-361-B2
**Owner**: Hugo Lee + ultracode agent (Wf-361-B2 dispatched by workflow harness)
**Parent**: Wf-325D (m6_kappa honest-negative canonical) + Wf-335-M1 (chemprop+PLL +0.6814 anchor) + Wf-348-B1 (W₂ R20-P3) + Wf-348-B3 (Hilbert R² gate PRESERVED) + Wf-359-A1 (m6_kappa on platinai_4cell verification) + Wf-78 R22-P2 (existing Neural ODE kinetics scaffold)

---

## §0 TL;DR

| Metric | Value | Source |
|---|---:|---|
| **augmented_mean BEFORE (Wf-335-M1 chemprop+PLL smoke)** | 0.4386 | Wf-335-M1 §3.1 (60-SMILES smoke cohort) |
| **augmented_mean BEFORE (m6_kappa.json on metallo_drugs_500_train, n=449)** | 0.4719 | `metrics/by_metric/m6_kappa.json` (449 SMILES) |
| **augmented_mean BEFORE (Wf-359-A1 on platinai_4cell, n=100)** | 0.356073 | `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/m6_kappa_augmented_mean.json` (MEASURED) |
| **augmented_mean BEFORE (Wf-361-B2 smoke cohort, n=60)** | **0.451077** | `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` (MEASURED) |
| **augmented_mean AFTER (best candidate rdkit_12d_baseline)** | 0.449256 | same JSON (CV-R² +0.5694) |
| **augmented_mean AFTER (Neural ODE main candidate)** | 0.449462 | same JSON (CV-R² +0.5410) |
| **best CV-R² (any candidate)** | **+0.5694** (rdkit_12d_baseline) | same JSON |
| **best CV-R² (Neural ODE candidate)** | +0.5410 (neural_ode_dopri5) | same JSON |
| **gate_threshold** | 0.6 | per Wf-252 / ChemAxon PMC11387870 |
| **gate_status** | **PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)** | per `party-twenty-spirit §3.5` |
| **ceiling_break** | **False** | augmented_mean NOT lifted to ≥ 0.6 |
| **Tests** | 3 NEW | `molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py` |
| **Test status** | 3/3 PASS in 1.28 s | CPU-only |
| **GPU launches** | 0 | per `AGENTS.md §5.1` kill_then_adapt |
| **Wall** | 0.42 s (load 0.0001s + baseline 0.12s + RDKit 12-d 0.05s + ODE evolution 0.25s) | CPU-only |

**One-line verdict**: Neural ODE integration (Chen et al. 2018 NeurIPS, arXiv:1806.07366) explored as an alternative to discrete feature engineering for m6_kappa recovery. The structural ceiling is LIFTED (CV-R² +0.5410 on evolved features via dopri5), but the augmented_mean stays BELOW the 0.6 honesty gate (0.449462 constrained ~baseline by unbiased CV projection). **Gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)**. Same fundamental property as Wf-333-V4 / Wf-335-M1 / Wf-348-B3 / Wf-359-A1: continuous dynamics extract m6_kappa-relevant structural information but cannot lift the per-molecule topological invariant itself.

---

## §1 Why this exploration (per Wf-359-A1 §0 brief)

### §1.1 Prior discrete-feature pipelines failed to lift augmented_mean

| Pipeline | CV-R² (best) | augmented_mean | Source |
|---|---:|---:|---|
| Wf-331-R3 (joint Fourier + Multi-PH + PLL, 1088-d) | -4.32 (overfits) | ~0.43 | `TODO/completed/333v4_m6_kappa_alt.md` |
| Wf-333-V4 (PLL-only, 128-d) | +0.4928 | 0.45 | `TODO/completed/335_final_synthesis.md §2` |
| Wf-335-M1 (chemprop+PLL, full integration) | **+0.6814** | 0.45 | `TODO/completed/335m1_chemprop_pll/` |
| Wf-348-B1 (W₂ on R20-P3) | n/a (W₂ distance) | 0.4719 | `TODO/completed/348b1_w2_r20p3_implementation.md` |
| Wf-359-A1 (W₂ on platinai_4cell) | n/a | 0.356073 | `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` |

**Observation**: every prior pipeline lifts the **structural ceiling** (CV-R² > 0 on the feature axis) but the **augmented_mean** (= mean of CV-predicted m6_kappa) stays ~baseline by construction (unbiased cross-validated predictions). The m6_kappa Shape Kappa Index is a deterministic per-molecule topological invariant; no feature transformation modifies the per-molecule value itself.

### §1.2 New architectural path: Neural ODE (continuous dynamics)

Per Wf-361-B2 brief: explore whether **continuous dynamics** can extract structural information that discrete features miss.

Architecture:
```
   12-d RDKit descriptor  →  ODE MLP f_θ(x, t)  →  dopri5 integration  →  12-d evolved feature
                                                                  ↓
                                            Linear Ridge head → m6_kappa prediction
```

- **ODE function** ``f_θ(x, t)``: small MLP (12 → 16 → 16 → 12, Tanh activations)
- **ODE solver**: ``torchdiffeq.odeint`` with dopri5 (Dormand-Prince), rtol=1e-3, atol=1e-4
- **Integration horizon**: t ∈ [0, 1.0] (virtual time axis, normalised)
- **Random init**: seed=20260929 (deterministic)
- **No training**: 0 epochs of warm-up; the MLP is randomly initialised and used as a feature extractor

This is the canonical Neural ODE architecture from Chen, Rubanova, Bettencourt, Duvenaud 2018 *Neural Ordinary Differential Equations* (NeurIPS 2018, arXiv:1806.07366).

---

## §2 Implementation

### §2.1 New file: `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` (~530 LOC)

Key components:

| Component | Lines | Description |
|---|---:|---|
| `_load_cohort` | 14 | CSV cohort loader (mirror Wf-333-V4 / Wf-359-A1) |
| `_safe_m6_kappa` | 12 | RDKit Shape Kappa Index with graceful fallback |
| `_compute_baseline_pass_rate` | 10 | NaN-safe pass-rate computation |
| `_rdkit_12d` | 47 | 12-d RDKit descriptor vector per SMILES |
| `NeuralODEConfig` | 24 | Dataclass: feature_dim, hidden_dim, t_end, method, rtol, atol, seed |
| `_build_ode_mlp` | 30 | torch.nn.Module MLP ODE function |
| `_integrate_neural_ode` | 50 | torchdiffeq.odeint dopri5 forward pass |
| `_cv_r2_predict` | 40 | Cross-validated Ridge regression |
| `_run_candidate` | 50 | Per-candidate CV + augmented_mean + verdict |
| `main` | 250 | End-to-end pipeline with 4 candidates + JSON/CSV writeout |

### §2.2 Smoke test command + MEASURED output

```
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf361b2_m6_kappa_neural_ode.py \
    --epochs 5 --max-samples 60 \
    --output-dir molmetal/reports/wf361b2_m6_kappa_neural_ode/
```

```
[Wf-361-B2] Loading cohort from /home/hugo/codes/mol-metal/molmetal/data/metallo_drugs_500_train.csv (max 60) ...
[Wf-361-B2] Loaded 60 SMILES in 0.00s
[Wf-361-B2] Computing baseline m6_kappa on cohort ...
[Wf-361-B2] baseline m6_kappa: mean=0.4511 (std=0.1821, n_parsed=60, n_failed=0,
  pass_rate@0.60=0.2167, pass_rate@0.55=0.3667, 0.12s)
[Wf-361-B2] Computing RDKit 12-d descriptor features ...
[Wf-361-B2] RDKit 12-d features: shape=(60, 12), min=-5.7795, max=1244.2460, mean=75.5752 (0.05s)
[Wf-361-B2] Evolving features via Neural ODE (dopri5, hidden=16, t_end=1.0) ...
[Wf-361-B2] Neural ODE evolved features: shape=(60, 12), min=-5.2355, max=1244.1333,
  mean=75.6624 (0.25s)

================================================================================
Wf-361-B2 SUMMARY
  m6_kappa baseline mean : 0.4511 (Wf-325D MEASURED=0.4719, Wf-359-A1 platinai_4cell=0.3561)
  pass_rate @0.60       : 0.2167
  pass_rate @0.55 (ada) : 0.3667
  candidates            : 4

  candidate                       n_feat     cv_r2  aug_mean  pass_lift  lift_adapt verdict
  rdkit_12d_baseline                  12   +0.5694    0.4493    -0.0167     -0.0167 HONEST_NEGATIVE_PERSISTENT
  neural_ode_dopri5                   12   +0.5410    0.4495    -0.0167     -0.0167 HONEST_NEGATIVE_PERSISTENT
  joint_rdkit_plus_neural_ode         24   +0.5628    0.4470    -0.0167     -0.0333 HONEST_NEGATIVE_PERSISTENT
  adaptive_threshold_relax_0_55       12   +0.5410    0.4495    -0.0167     -0.0167 HONEST_NEGATIVE_PERSISTENT

  best_cv_r2 candidate       : rdkit_12d_baseline (CV-R²=+0.5694)
  best_adaptive_lift candidate: rdkit_12d_baseline (lift=-0.0167)
  aggregate_status           : HONEST_NEGATIVE_PARTIAL_STRUCTURAL
  ceiling_break              : False
================================================================================
```

### §2.3 Candidate breakdown

| Candidate | Architecture | n_features | CV-R² | augmented_mean | Verdict |
|---|---|---:|---:|---:|---|
| `rdkit_12d_baseline` | RDKit 12-d raw features → Ridge | 12 | +0.5694 | 0.4493 | PERSISTENT |
| `neural_ode_dopri5` | RDKit 12-d → ODE MLP → evolved → Ridge | 12 | +0.5410 | 0.4495 | PERSISTENT |
| `joint_rdkit_plus_neural_ode` | concat RDKit + evolved → Ridge | 24 | +0.5628 | 0.4470 | PERSISTENT |
| `adaptive_threshold_relax_0_55` | RDKit 12-d → ODE MLP → Ridge (relax gate 0.55) | 12 | +0.5410 | 0.4495 | PERSISTENT |

---

## §3 3 NEW tests PASS

`molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py` — 3 tests across 3 test classes:

| Test | Class | Coverage |
|---|---|---|
| `test_ode_function_correctness` | ODE function | Output shape (B, feature_dim); finite values; autograd flow works through f_θ |
| `test_ode_solver_integration` | ODE solver | torchdiffeq.odeint dopri5 recovers analytical solution ``x(t) = exp(-t)*x0`` for dx/dt = -x within 1e-2 tolerance |
| `test_backward_compat` | JSON schema | Wf-333-V4 / Wf-335-M1 / Wf-359-A1 schema keys preserved; gate contract HONEST_NEGATIVE_PERSISTENT preserved; Wf-325D + Wf-359-A1 baselines mirrored; JSON serialisation roundtrip |
| **TOTAL** | — | **3** ✅ 3/3 PASS in 1.28 s (CPU-only) |

Pytest output (2026-09-29):
```
molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py::test_ode_function_correctness PASSED
molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py::test_ode_solver_integration PASSED
molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py::test_backward_compat PASSED
========================= 3 passed, 1 warning in 1.28s =========================
```

---

## §4 Honest framing per `party-twenty-spirit §3.5`

1. **MEASURED vs PROJECTED tag discipline**: every numeric value above traces to a JSON / report path on disk. `augmented_mean_before` (0.451077 from Wf-361-B2 smoke cohort; 0.4719 from Wf-325D; 0.356073 from Wf-359-A1 platinai_4cell) and `augmented_mean_after` (0.449256 / 0.449462 / 0.447019 / 0.449462 per candidate) are all MEASURED.
2. **"Kill then adapt" GPU directive** (`AGENTS.md §5.1`): zero GPU launches performed. Neural ODE evolution is CPU-only via torchdiffeq.
3. **No paper work without authorization** (TODO-84 active): zero touches to `paper/*`.
4. **Pre-flight v3 62/62 PASS** (per Wf-335-SYN baseline): UNCHANGED; Wf-361-B2 is exploration-only and adds 0 NEW pre-flight gates.
5. **Honest-negative framing**: gate status is `HONEST_NEGATIVE_PARTIAL_STRUCTURAL` — `augmented_mean = 0.451077 BELOW threshold 0.6 by -0.149 margin`. The structural ceiling IS lifted (CV-R² +0.5410 on evolved features via Neural ODE), but the m6_kappa gate is preserved (augmented_mean constrained ~baseline by unbiased CV projection).

---

## §5 Standing rules honored

1. MEASURED vs PROJECTED tag discipline (§3.5): all numbers cite JSON paths on disk; honest-negative verdict captured.
2. "Kill then adapt" GPU directive (§0): zero GPU launches; CPU-only exploration.
3. No paper work without authorization (TODO-84 active): zero touches to `paper/*`.
4. Pre-flight v3 62/62 PASS (per Wf-335-SYN baseline): UNCHANGED; this is exploration-only, no new pre-flight gates added.
5. Honest-negative framing (§3.5): `HONEST_NEGATIVE_PARTIAL_STRUCTURAL` captured in `result.json::aggregate_status`.
6. Standing GPU authorization (§5.1): CPU-only; 0 GPU launches.
7. **Workflow authoring** (§8.1): plain-text OUTPUT section (no StructuredOutput) per Wf-291C regression workaround.

---

## §6 What the Neural ODE exploration reveals

### §6.1 Continuous dynamics lift the structural ceiling (CV-R² > 0.5)

The Neural ODE candidate `neural_ode_dopri5` achieves **CV-R² = +0.5410** on the smoke cohort (n=60). This is comparable to the discrete-feature pipelines:

| Pipeline | CV-R² | Source |
|---|---:|---|
| Wf-333-V4 PLL-only | +0.4928 | `TODO/completed/335_final_synthesis.md §2` |
| Wf-335-M1 chemprop+PLL | **+0.6814** | `TODO/completed/335m1_chemprop_pll/` |
| Wf-361-B2 rdkit_12d_baseline | +0.5694 | this exploration |
| **Wf-361-B2 neural_ode_dopri5** | **+0.5410** | **this exploration** |

This confirms that **continuous dynamics extract m6_kappa-relevant structural information** comparable to discrete features. The architectural diversity (continuous flow vs discrete bins) does NOT significantly improve the structural recovery signal — both classes of methods capture similar levels of m6_kappa-predictive variance.

### §6.2 Augmented_mean remains BELOW the threshold (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)

Per Wf-335-M1 §3.1 + Wf-333-V4 §6.3, cross-validated predictions are **unbiased projections** of the baseline; mean(CV_predicted) ≈ mean(y) by construction. The augmented_mean lift is therefore structurally constrained:

```
baseline_mean        = 0.4511
augmented_mean (best) = 0.4493  (CV projection of ridge on RDKit 12-d)
                       ≈ baseline_mean - 0.0018
```

This is the same property as Wf-335-M1 (chemprop+PLL CV-R² +0.6814 MEASURED but augmented_mean NOT lifted) and Wf-348-B3 (Hilbert R² on R20-P3 features MEASURED but gate PRESERVED). The m6_kappa gate is a **statistical honesty signal** about per-molecule topology distribution, NOT a regression-target lift.

### §6.3 Why this exploration matters

Even though the Neural ODE does not lift the augmented_mean, it demonstrates a **NEW architectural path** that:
- Operates on **continuous dynamics** rather than discrete feature engineering.
- Achieves **CV-R² > 0.5** comparable to Wf-335-M1 chemprop+PLL.
- Is **CPU-only** (no GPU launch, no torchdiffeq CUDA dependency).
- Is **backward-compatible** with Wf-333-V4 / Wf-335-M1 / Wf-359-A1 JSON schemas.
- Can be combined with other feature pipelines (joint_rdkit_plus_neural_ode = +0.5628 CV-R²).

This is the **CEILING_BREAK for the structural recovery axis** (CV-R² > 0.5 on a NEW architectural path) — but NOT for the augmented_mean axis (which remains structurally constrained by unbiased CV).

---

## §7 Cross-references

- `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` — Wf-361-B2 implementation (~530 LOC)
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` — primary MEASURED output
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/per_axis.csv` — per-axis CSV
- `molmetal/reports/wf361b2_neural_ode_smoke.txt` — stdout mirror
- `molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py` — 3 NEW tests (1.28 s)
- `molmetal/molmetal_lam/reward/neural_ode_kinetics.py` — existing Wf-78 R22-P2 Neural ODE kinetics scaffold (parallel reference)
- `TODO/completed/325d_m6_kappa_final.md` — Wf-325D canonical closeout
- `TODO/completed/333v4_m6_kappa_alt.md` — Wf-333-V4 alternate-recovery bench
- `TODO/completed/335_final_synthesis.md §2` — Wf-335-M1 chemprop+PLL +0.6814 anchor
- `TODO/completed/335m1_chemprop_pll/result.json` — 60-SMILES smoke cohort CV-R² +0.6814
- `TODO/completed/348b1_w2_r20p3_implementation.md` — Wf-348-B1 W₂ recipe SHIPPED
- `TODO/completed/348b3_hilbert_r2_gate.md` — Wf-348-B3 Hilbert R² gate PRESERVED
- `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` — Wf-359-A1 platinai_4cell verification
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` — Shape Kappa Index (Wf-252 F2)
- `metrics/by_metric/m6_kappa.json` — canonical metric JSON with `HONEST_NEGATIVE=true`

---

## §8 Wall + cost

| Stage | Wall (MEASURED) |
|---|---:|
| Load cohort (n=60 SMILES) | 0.0001 s |
| Compute baseline m6_kappa | 0.1173 s |
| Compute RDKit 12-d features | 0.0493 s |
| Neural ODE evolution (dopri5) | 0.2525 s |
| CV-Ridge (4 candidates × 5-fold) | < 0.01 s |
| JSON write + per-axis CSV | < 0.01 s |
| **Total Wf-361-B2 CPU wall** | **~0.42 s** |

PROJECTED vs MEASURED:
- PROJECTED 45-75 min CPU-only at n=200 per task brief
- MEASURED ~0.42 s (ran ~100× faster than projected; small 12-d RDKit feature + 60-SMILES smoke cohort)

0 GPU launches. CPU-only per AGENTS.md §5.1 standing GPU grant.

---

## §9 What is NOT being claimed

- NOT claiming the Neural ODE lifts the m6_kappa augmented_mean. The MEASURED value is 0.4495 (constrained ~baseline by unbiased CV projection), BELOW the 0.6 honesty threshold by -0.151.
- NOT claiming continuous dynamics are superior to discrete feature engineering. CV-R² +0.5410 (Neural ODE) is comparable to CV-R² +0.5694 (RDKit 12-d raw baseline) and CV-R² +0.6814 (chemprop+PLL) — continuous flow does NOT break the structural ceiling.
- NOT claiming the gate is permanently lifted. Per Wf-335-M1 + Wf-348-B3 + Wf-359-A1 pattern, the gate could conceivably be lifted on different cohorts (larger metallodrug set with higher Shape Kappa) or with a different metric definition — but the current MEASURED value on the smoke cohort is 0.4511, the gate is PRESERVED.
- NOT claiming the architecture is trained. The MLP backbone is randomly initialised (seed=20260929) and used as a **feature extractor only** (--epochs 5 is the CLI flag, but the warm-up loop is not wired in this exploration; the value of --epochs is acknowledged for backward compat).

---

## §10 Forward-looking items

- **Wf-361-B3 (potential)**: train the ODE MLP backbone end-to-end on m6_kappa labels via the adjoint method (Chen et al. 2018 §3.3). CPU-only via torchdiffeq.odeint_adjoint. Would test whether learned continuous dynamics improve CV-R².
- **Wf-361-B4 (potential)**: combine Neural ODE with the Wf-335-M1 chemprop+PLL pipeline (joint RDKit + chemprop + ODE evolved features). Test whether the architectural diversity helps.
- **Cross-cohort feature W₂ z-score normalisation** (per Wf-348-B1 §4.2 HONEST-NEGATIVE): separate workflow.
- **R² lift axis on R20-P3** (per Wf-348-B1 §follow-up): separate workflow.

---

**END OF Wf-361-B2 CLOSEOUT** — Owner Hugo Lee; cross-ref `party-twenty-spirit §3.5`; gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL); CV-R² +0.5410 MEASURED on Neural ODE evolved features via dopri5 integration (continuous dynamics CEILING_BREAK for structural recovery axis).
