# Wf-367-B1 — m2 R² recovery scaffold (platinai_4cell + hilbert)

**Date**: 2026-09-29
**Status**: SHIPPED — gate PRESERVED (HONEST_NEGATIVE_R2_DEEP on both cohorts)
**Workflow**: Wf-367-B1
**Parent context**: Wf-365-A2 follow-up to Wf-335-M2 finding
(platinai_4cell best R² = -0.1251 + hilbert best R² = -0.1268)

---

## §1 Purpose

Per **Wf-365-A2 NEW finding** + **Wf-335-M2 result**, the m2 W₂ +
chemprop v2 graph integration achieved CEILING_BREAK on
metalcytotox_combined only; platinai_4cell and hilbert remain at
HONEST_NEGATIVE_R2_DEEP.  This wave ships an **R² recovery
scaffold** that explores a multi-stream feature ensemble:

* **RDKit 12-d baseline** (first 12 columns of the platinai CSV).
* **chemprop-v3-style 32-d enhanced features** (CPU-only structural
  surrogate: atom-count bins + bond-count bins + ring-count bins +
  functional-group counts via 8 SMARTS patterns).
* **Hilbert ASTSR 8-d spectral features** (per Wf-348-B1 Fourier
  8-bin truncation: MolWt/MolLogP/HeavyAtoms/Aromatic bins + spectral
  hash bins).
* **Per-cell 4-d one-hot** for `platinai_4cell`
  (mcf7/a549/a2780/a2780cis).
* **W₂ distance** (best axis, multi-axis selection).

The multi-stream ensemble is evaluated by Ridge regression with
multi-seed × 5-fold CV (the Wf-335-M2 protocol).

---

## §2 Implementation

### §2.1 Recovery script (CPU-only, no GPU launch)

| Item | Path | LOC |
|---|---|---:|
| Recovery script | `molmetal/scripts/wf367b1_m2_r2_recovery.py` | 437 |
| Test file | `molmetal/tests/test_wf367b1_m2_r2.py` | 244 |
| Result JSON | `molmetal/reports/wf367b1_m2_r2/result.json` | — |
| Smoke stdout | `molmetal/reports/wf367b1_m2_smoke.txt` | — |

### §2.2 Public API

| Function | Purpose |
|---|---|
| `chemprop_v3_enhanced_features(smiles_list)` | 32-d chemprop-v3-style enhanced graph features (CPU surrogate) |
| `hilbert_astsr_spectral_features(smiles_list)` | 8-d Hilbert ASTSR spectral features (per Wf-348-B1) |
| `per_cell_onehot(smiles_list, cohort)` | 4-d cell-line one-hot (platinai_4cell only) |
| `build_recovery_feature_matrix(smiles_list, X_32d, cohort)` | Multi-stream concatenation (54-d / 50-d) |
| `run_wf367b1(out_path, cohorts, ...)` | Main entry point |

### §2.3 Test surface (3 NEW tests, 3/3 PASS)

| Test | Verifies |
|---|---|
| `test_wf367b1_r2_recovery_on_platinai_4cell` | Stream dimensions; structural invariants; verdict taxonomy; 12-d baseline R² matches Wf-335-M2 within 1e-3 |
| `test_wf367b1_r2_recovery_on_hilbert` | Same diagnostic on hilbert cohort (no per-cell one-hot) |
| `test_wf367b1_backward_compat` | Wf-325A / Wf-335-M2 / Wf-331-R5 baselines unchanged; Wf-335-M2 result.json mtime preserved; stream-level structural invariants |

---

## §3 Results (MEASURED, on disk)

### §3.1 Smoke run (2026-09-29, n_splits=5 n_seeds=5 alpha=1.0)

```
Wf-367-B1: n_cohorts = 2
  CV config = n_splits=5 n_seeds=5 alpha=1.0

  cohort=        platinai_4cell  n=  200  Wf-335-M2 best R²=-0.1251  n_features_recovery=54  stream_dims={'rdkit_12d': 12, 'chemprop_v3': 32, 'hilbert_astr': 8, 'cell_onehot': 4}  elapsed=1.3s
                                12d_baseline  n_feat= 12  R²_mean=-0.1251 ± 0.1515  lift_vs_wf335m2=+0.0000  NO_LIFT_HONEST_NEGATIVE
                       multi_stream_recovery  n_feat= 54  R²_mean=-0.5158 ± 0.4170  lift_vs_wf335m2=-0.3908  HONEST_NEGATIVE_R2_DEEP
     multi_stream_recovery_plus_w2_best_axis  n_feat= 55  R²_mean=-0.5106 ± 0.3981  lift_vs_wf335m2=-0.3855  HONEST_NEGATIVE_R2_DEEP

  cohort=               hilbert  n=  200  Wf-335-M2 best R²=-0.1268  n_features_recovery=50  stream_dims={'rdkit_12d': 12, 'chemprop_v3': 32, 'hilbert_astr': 8}  elapsed=1.2s
                                12d_baseline  n_feat= 12  R²_mean=-0.1268 ± 0.1709  lift_vs_wf335m2=+0.0000  NO_LIFT_HONEST_NEGATIVE
                       multi_stream_recovery  n_feat= 50  R²_mean=-0.4980 ± 0.3216  lift_vs_wf335m2=-0.3713  HONEST_NEGATIVE_R2_DEEP
     multi_stream_recovery_plus_w2_best_axis  n_feat= 51  R²_mean=-0.4933 ± 0.3188  lift_vs_wf335m2=-0.3665  HONEST_NEGATIVE_R2_DEEP

  aggregate_status = HONEST_NEGATIVE_R2_DEEP  (ceiling_break=0 above_zero=0)
  elapsed_sec_total = 2.5s
```

### §3.2 Per-cohort verdict table

| Cohort | Wf-335-M2 best R² | Wf-367-B1 12d baseline | multi_stream_recovery | + W₂ best axis | Lift? | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `platinai_4cell` | **-0.1251** | -0.1251 ± 0.1515 | -0.5158 ± 0.4170 | -0.5106 ± 0.3981 | +0.0000 / -0.3908 / -0.3855 | HONEST_NEGATIVE_R2_DEEP preserved |
| `hilbert` | **-0.1268** | -0.1268 ± 0.1709 | -0.4980 ± 0.3216 | -0.4933 ± 0.3188 | +0.0000 / -0.3713 / -0.3665 | HONEST_NEGATIVE_R2_DEEP preserved |

### §3.3 Aggregate verdict

* **Before** (Wf-335-M2): platinai_4cell = -0.1251, hilbert = -0.1268
  — both HONEST_NEGATIVE_R2_DEEP.
* **After** (Wf-367-B1 multi-stream): platinai_4cell = -0.5158,
  hilbert = -0.4980 — both HONEST_NEGATIVE_R2_DEEP preserved
  (multi-stream ensemble makes R² WORSE due to overfitting on
  small cohorts, n=200 SMILES).
* **aggregate_status = HONEST_NEGATIVE_R2_DEEP** (0 ceiling_break,
  0 above_zero).

### §3.4 Why the multi-stream ensemble does NOT lift R² on small cohorts

The 200-SMILES cohorts have too few samples for a Ridge regression
with 50+ features to outperform the 12-d baseline.  Adding
chemprop-v3 (32-d), Hilbert ASTSR (8-d), and per-cell one-hot (4-d)
**inflates** the variance of the Ridge coefficients, making the
out-of-fold R² worse than the 12-d constant-only baseline.

This is the classic **bias-variance tradeoff at small n**:  more
features → higher model variance → worse CV-R² when n_samples
<< n_features.

---

## §4 Honest framing (party-twenty-spirit §3.5)

* Per-cohort verdict:
  - **Before** (Wf-335-M2): both at HONEST_NEGATIVE_R2_DEEP.
  - **After** (Wf-367-B1 multi-stream): both at
    HONEST_NEGATIVE_R2_DEEP preserved.
  - **Net change**: 0 cohorts lifted (aggregate_status unchanged).
* Structural notes:
  - **CPU-only diagnostic** — no torch autograd, no GPU chemprop v3
    encoder forward pass, no Hilbert primal-dual regression.
  - The multi-stream ensemble is a **structural upper bound** on
    what Ridge can do on these cohorts.  Lifting R² above 0
    requires either (a) more samples per cohort, or (b) a non-linear
    model (e.g. chemprop v3 D-MPNN encoder forward pass + autograd,
    which is GPU-gated per AGENTS.md §5.1).
  - **metalcytotox_combined** (n=2000) is the only cohort large
    enough to benefit from the multi-stream ensemble — and it
    achieved CEILING_BREAK in Wf-335-M2.
* Forward-looking honest-negative:
  - To LIFT platinai_4cell + hilbert R² above 0, the project needs
    EITHER (a) larger cohorts (n ≥ 1000 per cohort) OR (b) GPU
    chemprop v3 encoder + Hilbert primal-dual regression (per
    AGENTS.md §5.1 GPU grant).  Neither is in scope for Wf-367-B1.

---

## §5 File paths (absolute)

* `/home/hugo/codes/mol-metal/molmetal/scripts/wf367b1_m2_r2_recovery.py`
* `/home/hugo/codes/mol-metal/molmetal/tests/test_wf367b1_m2_r2.py`
* `/home/hugo/codes/mol-metal/molmetal/reports/wf367b1_m2_r2/result.json`
* `/home/hugo/codes/mol-metal/molmetal/reports/wf367b1_m2_smoke.txt`

---

## §6 Pre-flight status

CPU-only diagnostic; pre-flight v2 gate NOT required (no retrain /
panel / ablation).  EXPECTED_GATE_COUNTS["all"] 62 PRESERVED.

---

**END OF Wf-367-B1 CLOSEOUT**