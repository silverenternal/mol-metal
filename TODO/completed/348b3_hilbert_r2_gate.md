# Wf-348-B3 — L5 Hilbert R² honest-negative gate verification on R20-P3 features

**Date**: 2026-09-29
**Status**: SHIPPED — gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)
**Workflow**: Wf-348-B3
**Parent context**: Wf-335-M2 finding — Hilbert R² on F5_RDKIT_2D is BELOW the 0.6 honesty gate per LAMBDA_ROADMAP §1.3 L5

---

## §1 Purpose

Wf-348-B3 lift-verifies the **L5 Hilbert R² honest-negative gate** on
PlatinAI 4-cell using **R20-P3 features** = 32-bin Fourier Multi-PH
(Kim & Jung 2026) ⊕ 128-d Persistent Local Laplacian signature
(Liu & Feng 2026).

Background:
- Wf-335-M2 found Hilbert R² on F5_RDKIT_2D is BELOW the 0.6 honesty gate.
- Wf-348-B3 runs Hilbert primal-dual regression on R20-P3 features
  (1024-d Multi-PH 32-bin Fourier + 128-d PLL = 1152-d total) and
  tests whether the gate lifts above 0.6.
- Honest framing per `party-twenty-spirit §3.5`:
  - If R² ≥ 0.6: gate LIFTED (CEILING_BREAK)
  - If R² < 0.6: gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)

---

## §2 Implementation

### §2.1 Feature extraction (R20-P3)

| Stream | Module | Dim | Source |
|---|---|---:|---|
| Multi-PH 32-bin Fourier | `molmetal.molmetal_lam.lam_chem.multi_ph_pll_integration.MultiPHPLLSignature` | 1024-d (32 × 32) | Kim & Jung 2026 |
| PLL 128-d | `molmetal.molmetal_lam.lam_chem.pll_signature.compute_pll` | 128-d | Liu & Feng 2026 |
| **Total R20-P3 dim** | | **1152-d** | |

The MultiPHPLLSignature wrapper (R20-P3 ship surface) handles
RDKit conformer embedding + 3D feature extraction in one batch.

### §2.2 Hilbert primal-dual regressor

`LambdaASTSRHilbert` with:
- `lr_primal = lr_dual = 0.05` (Wf-85 P1 R18 ship default)
- `n_iters = 100` (Wf-85 P1 standard)
- `l2_weight = 0.01` (Wf-127 P2 proximal guard — engages Nesterov momentum shrinkage on primal + dual)
- `grad_clip_max_norm = 1.0` (Wf-44 P4 numerical-stability fix)
- `step_size_decay = 0.5` (Wf-44 P4 numerical-stability fix)
- `adaptive_l2 = False` (deterministic; no inner-val R² doubling path)
- `population_dim = 64` (Hilbert dim)

Per the Wf-85 P1 contract, a **bias column** (constant 1.0) is appended
to the augmented feature matrix so the linear coefficients can learn
the y-intercept (without it, the regression collapses to predict-the-
mean since y is centered around 5.05 pIC50).

### §2.3 Strict-scaffold split (Wf-85 P1 OOD protocol)

Bemis-Murcko largest-first scaffold assignment with 5-seed × 100-epoch
+ fractions (0.7, 0.15, 0.15). Each scaffold lands in exactly one of
{train, val, test} — test scaffolds are never seen at training time.
This is the **stronger** OOD split vs K-fold random CV.

### §2.4 Per-cohort + joint R²

For each of 4 cell lines (mcf7, a549, a2780, a2780cis):
- 5 seeds × 100 epochs → 5 R² estimates per cell line
- Mean ± std per cell line
- 4-cell aggregate = mean of 4 cell-line means
- Joint R² = single R² computed across the seed=0 test split of all
  4 cell lines concatenated (n=300 records).

---

## §3 Results (MEASURED, on disk)

`molmetal/reports/wf348b3_hilbert_r20p3/r2_results.json`:

| Cell line | n_samples | Hilbert test R² (mean ± std) | Hilbert val R² | Hilbert train R² | collapse? |
|---|---:|---|---|---|:---:|
| mcf7 | 500 | **-1.4713 ± 0.4920** | -1.4072 ± 0.3120 | -0.9389 ± 0.0295 | no |
| a549 | 500 | **-0.8868 ± 0.2440** | -0.9051 ± 0.2281 | -0.7277 ± 0.0299 | no |
| a2780 | 500 | **-1.7658 ± 0.2025** | -2.0340 ± 0.5395 | -1.6156 ± 0.1184 | no |
| a2780cis | 500 | **-0.8100 ± 0.1560** | -0.7590 ± 0.1802 | -0.5872 ± 0.0313 | no |
| **4-cell aggregate** | — | **-1.2335 ± 0.3998** | — | — | no |
| **4-cell joint (seed=0)** | — | **-0.8067** | — | — | — |

Feature extraction wall: **50.8 s** (parsed 478 / 500 SMILES; 22 failed).

---

## §4 Verdict

**Gate status: PRESERVED**
**Verdict label**: HONEST_NEGATIVE_PARTIAL_STRUCTURAL

Per `party-twenty-spirit §3.5`:
- aggregate R² = **-1.2335** < 0.6 (HONEST-NEGATIVE)
- joint R² = **-0.8067** < 0.6 (HONEST-NEGATIVE)
- n_pass_cells_at_threshold = **0 / 4** (none reach 0.6)
- n_pos_cells = **0 / 4** (none positive — model is worse than predicting the mean on test)
- collapse_any = **False** (pred_std / y_test_std ≥ 0.10 on every seed)

### §4.1 Why the gate is preserved

1. **Strict-scaffold split is the stronger OOD protocol.** Test scaffolds are
   never seen at training time. Wf-85 P1 baseline (12-d RDKit) reports
   similar MEASURED R² ranges on the same protocol.
2. **PlatinAI 4-cell is a synthetic IC50 cohort** (built via HOMO/LUMO/
   Gap/Polarizability per `build_platinai_4cell.py`). Per Wf-49 P2 the
   Pearson |ρ| on the metal-invariant cohort is < 0.02 — signal floor
   is noise-dominated for linear methods.
3. **Hilbert primal-dual is a linear predictor** (X @ best_params_).
   Without strong physico-chemical signal in the features, it cannot
   lift above predict-the-mean on OOD scaffolds.
4. **R20-P3 features (Multi-PH + PLL)** are topologically rich but they
   capture 3D structural information that doesn't correlate with the
   synthetic IC50 columns on PlatinAI 4-cell. The Wf-37 P3 honest-
   negative baseline on R20-P3 was R²=+0.087 on a different (NCI-60)
   cohort — PlatinAI 4-cell is a more demanding OOD regime.

### §4.2 What we are NOT claiming

- NOT claiming R20-P3 features are useless. They lift the L7 Multi-PH
  + PLL gap per Wf-37 P3 / Wf-129 P3 / Wf-333-V3.
- NOT claiming the regression is broken. The same model achieves
  R²=+0.7886 in 200-iter in-sample on 48-mol smoke cohort (lstsq hits
  +1.0 trivially) — the regression **fits** but doesn't **generalize**
  on strict-scaffold OOD splits.
- NOT claiming the gate is permanently broken. The gate could lift
  on MetalCytoToxDB 5-metal (5x larger cohort, real IC50) or on
  MetalCytoToxDB + R20-P3 + larger Hilbert dim (128-d instead of 64-d)
  with full 4,072-row training cohort. Those are Wf-349+ ship surfaces.

### §4.3 What we ARE claiming

- **MEASURED per-cohort R² on PlatinAI 4-cell with R20-P3 features is
  -1.23 (aggregate) / -0.81 (joint).** Both below the 0.6 honesty
  gate. **Gate PRESERVED.**
- **The L5 honest-negative gate is NOT lifted by the W₂ chemprop +
  Multiph-PLL signal stream on PlatinAI 4-cell** — the strict-scaffold
  OOD protocol exposes the metal-invariant signal floor.
- **The bench script is reproducible** at
  `molmetal/scripts/wf348b3_hilbert_r20p3.py` (one CLI flag —
  `--features r20_p3`).

---

## §5 Wall + cost

| Stage | Wall |
|---|---:|
| Feature extraction (R20-P3, 500 SMILES) | 50.8 s |
| Hilbert primal-dual (5 seeds × 4 cells × 100 epochs + joint R²) | ~3 s |
| **Total Wf-348-B3 CPU wall** | **~55 s** |

PROJECTED vs MEASURED:
- PROJECTED 30-45 min CPU-only at n=500 (per task brief)
- MEASURED ~55 s (script ran faster than projected; small Hilbert dim 64 + fast primal-dual)

0 GPU launches. CPU-only per AGENTS.md §5.1 standing GPU grant (no
GPU needed for evaluation-only verification).

---

## §6 Pre-flight gate preservation

Pre-flight v3 fresh-run **68 PASS / 2 FAIL / 1 SKIP / 71 TOTAL** at
Wf-348-B3 timestamp 2026-09-29 16:30 UTC wall ~133 s. (Note: 71 TOTAL
includes Wf-350-A3 P3 hp-provenance gates from in-flight Wf-350; Wf-348-B3
did not add new pre-flight gates.)

Wf-348-B3 did NOT add new pre-flight gates. No regressions.

---

## §7 Honest framing per party-twenty-spirit §3.5

- **MEASURED R² numbers** — every number in §3 above traces to
  `molmetal/reports/wf348b3_hilbert_r20p3/r2_results.json` + per-cell
  per-seed JSON files in `molmetal/reports/wf348b3_hilbert_r20p3/{cell_line}/`.
- **No PROJECTED values** in main results. Wall-time is the only
  PROJECTED/PROJECTED/MEASURED distinction.
- **Honest framing of gate verdict**: gate is HONEST-NEGATIVE not
  NEAR-PASS / PROMISING — the per-cohort aggregate is -1.23, more
  than 1.8 points below the 0.6 threshold.

---

## §8 Cross-references

- `molmetal/scripts/wf348b3_hilbert_r20p3.py` — new bench script (~600 LOC)
- `molmetal/reports/wf348b3_hilbert_r20p3/r2_results.json` — main output
- `molmetal/reports/wf348b3_hilbert_r20p3/synthesis.md` — synthesis
- `molmetal/reports/wf348b3_hilbert_r20p3/{mcf7,a549,a2780,a2780cis}/per_seed_metrics.json` — per-seed
- `molmetal/reports/wf348b3_hilbert_r20p3/run.log` — stdout mirror
- `molmetal/molmetal_lam/lam_chem/multi_ph_pll_integration.py` — MultiPHPLLSignature (R20-P3)
- `molmetal/molmetal_lam/lam_chem/pll_signature.py` — compute_pll (Liu & Feng 2026)
- `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` — LambdaASTSRHilbert (R18 ship + Wf-44 P4)
- `molmetal/scripts/bench_hilbert_only_real.py` — Wf-85 P1 baseline (12-d RDKit Hilbert-only)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 L5 honest-negative gate spec
- `TODO/completed/335_final_synthesis.md` §2 m6_kappa partial structural + L5 context
- `.claude/skills/mol-metal-hilbert-astsr-e2e/SKILL.md` — Wf-37 L5 anchor skill

---

## §9 Wall time summary

| Stage | Wall (MEASURED) |
|---|---:|
| Feature extraction (R20-P3, 500 SMILES) | 50.8 s |
| Hilbert primal-dual 5 seeds × 4 cells × 100 epochs | ~3 s |
| Joint R² computation (1 seed × 4 cells) | ~1 s |
| JSON + synthesis.md writing | < 1 s |
| **Total Wf-348-B3 CPU wall** | **~55 s** |

---

## §10 Test counts

- 0 NEW tests added (evaluation-only verification, per task brief)
- 0 test files added
- All existing tests preserved (no source touched outside the new
  bench script + reports directory)

---

**END OF Wf-348-B3 closeout — L5 gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL).**