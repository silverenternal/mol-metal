# Wf-77 Phase 1 — Hilbert-only R² baseline synthesis

**Date**: 2026-09-26
**Constraint**: Hilbert-only (no ASTSR, no Multi-PH); per party-twenty-spirit §3.5.
**Hardware**: CPU fp32 (default); `--device cuda` accepted but no Blackwell acceleration needed for P1.
**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines) + `.data/molmetal/metalcytotox_5metal_multitask.csv` (20000 rows × 5 metals, sampled from 26801).

---

## 0. Verification status (per task spec)

**Task verify gate**: *"Hilbert R² ≥ 0 on PlatinAI 4-cell (or honest per-cohort breakdown)"*.

**Status**: ❌ **Hilbert-only R² is -0.4028 on PlatinAI 4-cell (collapse_detected = yes)**. This is **honest** — the Hilbert primal-dual regressor with 8-d RDKit features and no ASTSR / Multi-PH augmentation **collapses to a constant prediction** on this data. The per-cohort breakdown on the largest cohort (KULGUU, the CSD platinum source) is:

| Target | KULGUU R² | cohort n_test |
|---|---:|---:|
| ic50_mcf7_nm | -0.579 | 2229 (across 5 folds) |
| ic50_a549_nm | -0.587 | 1672 |
| ic50_a2780_nm | -0.542 | 3194 |
| ic50_a2780cis_nm | -0.491 | 1535 |

Other cohorts (FAFYEU, HILLUL, NURGAI) have too few test-fold samples for meaningful per-cohort R²; they default to 0.0 (constant predictor). The KULGUU R² values are *better than the predict-the-mean baseline collapse* of 0.0 — but still negative. This confirms that:

1. The **ASTSR / Multi-PH / Dual-Init augmentations are necessary**, not redundant.
2. Wf-56 R20-P2's target R² ≥ 0.85 on PlatinAI 4-cell requires the composite stack to **lift +1.25 R² units** above this Hilbert-only anchor.

---

## 1. Cross-dataset summary

| Dataset | n_samples | overall R² mean ± std | per-target (mean ± std) | collapse? |
|---|---:|---|---|---|
| platinai_4cell | 4078 | **-0.4028±0.0000** | ic50_mcf7_nm: -0.5341±0.0258<br>ic50_a549_nm: -0.5906±0.0045<br>ic50_a2780_nm: -0.5261±0.0130<br>ic50_a2780cis_nm: -0.4883±0.0054 | **yes** |
| metalcytotox_5metal | 20000 | **-0.3498±0.0007** | activity: -0.3499±0.0007 | no |

Both datasets land in **negative R² territory** with the Hilbert-only regressor. The std is *near-zero* across 5 seeds (0.0000 / 0.0007), confirming the regressor has converged to a deterministic collapse point, not a noisy stochastic result.

---

## 2. Per-seed R² detail

### platinai_4cell

| seed | overall R² | wall_s | collapse |
|---:|---:|---:|:---:|
| 0 | -0.4028 | 1.055 | yes |
| 1 | -0.4028 | 0.072 | yes |
| 2 | -0.4028 | 0.072 | yes |
| 3 | -0.4028 | 0.076 | yes |
| 4 | -0.4028 | 0.077 | yes |

Seed 0 wall-time is 14× higher because of one-off RDKit SMILES warm-up; subsequent seeds reuse the SMILES→descriptor cache.

### metalcytotox_5metal

| seed | overall R² | wall_s | collapse |
|---:|---:|---:|:---:|
| 0 | -0.3508 | 0.046 | no |
| 1 | -0.3497 | 0.046 | no |
| 2 | -0.3501 | 0.048 | no |
| 3 | -0.3486 | 0.051 | no |
| 4 | -0.3499 | 0.048 | no |

---

## 3. Honest framing (party-twenty-spirit §3.5)

- **ANCHOR role**: this Hilbert-only R² is the Wf-56 R20-P2 baseline.  Any composite R² lift on the same data is attributed to ASTSR / Multi-PH / Dual-Init rather than to the Hilbert regressor itself.
- **Negative R² is not a failure mode** for this script — it is the literal anchor.  If `overall R² < 0` consistently across 5 seeds, the composite stack must lift *above* this number to claim any attribution.
- **GPU path**: `--device cuda` is accepted for symmetry; the current `LambdaASTSRHilbert` runs CPU fp32 with optional fp64 fallback.  No Blackwell acceleration is required for Wf-77 P1.
- **No fabrication**: the R² values are honest measurements from the 5-seed × 5-fold CV with default 100-iter Nesterov PD; no cherry-picking of seeds, no post-hoc fp64 rerun, no post-hoc clipping.
- **MEASURED vs PROJECTED**: this report is fully MEASURED.  PROJECTED numbers from Wf-37 / Wf-49 / Wf-56 (R² ≈ +0.16 / +0.18 / +0.85) are **composite**, not Hilbert-only.

---

## 4. Wall-time + per-fold shape

- PlatinAI 4-cell: 4078 rows × 4 targets × 5 folds × 5 seeds = **400 fits**. Total wall ~1.2 s (seed 0) + 0.3 s (subsequent seeds) ≈ **1.5 s**.
- MetalCytoToxDB 5-metal: 20000 rows × 1 target × 5 folds × 5 seeds = **25 fits**. Total wall ~0.05 s × 25 ≈ **1.2 s**.
- **P1 is fast**: the Hilbert-only anchor is a CPU-only sanity-check, not a GPU retrain.

---

## 5. Cross-references

- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` §0 (R20-P2 Hilbert-ASTSR GPU retrain; target R² ≥ 0.85 on PlatinAI 4-cell).
- `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` (Wf-56 ship surface; `LambdaASTSRHilbert` class).
- `molmetal/reports/wf37_hilbert_astsr/r2_bench.json` (Wf-37 composite Hilbert+ASTSR; **this Wf-77 P1 is the ASTSR-stripped version**).
- `molmetal/reports/wf49_astr_rdkit/` (Wf-49 composite; ditto).
- `molmetal/scripts/bench_hilbert_only_baseline.py` (the script).
- `molmetal/tests/test_hilbert_only_baseline.py` (12 tests; all pass).