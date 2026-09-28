# Wf-86 Phase 3 — REAL Hilbert+ASTSR combined R² on PlatinAI 4-cell

**Date**: 2026-09-26
**Constraint**: Combined Hilbert primal-dual + ASTSR evolutionary symbolic regressor
  (no Multi-PH + PLL); per `party-twenty-spirit §3.5` honest framing.
**Hardware**: CPU fp32 (default); `--device cuda` accepted but no Blackwell acceleration
  required for Wf-86 P3 production anchor.
**Split**: 70/15/15 strict-scaffold split per cell line (Bemis-Murcko computed
  on-the-fly from canonical SMILES, with ligand-canonicalised fallback when
  Bemis-Murcko returns empty for Pt-coordination complexes).
**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).

## Cross-cell-line summary (5-seed × strict-scaffold)

| cell line | n_samples | combined R² (mean ± std) | Hilbert-only R² | ASTSR-only R² |
|---|---:|---|---|---|
| mcf7 | 4078 | -0.0141 ± 0.0137 | -23.7825 ± 1.8285 | -0.0141 ± 0.0137 |
| a549 | 4078 | -2.9397 ± 5.7762 | -19.4645 ± 1.8081 | -2.9397 ± 5.7762 |
| a2780 | 4078 | -0.3639 ± 0.2416 | -28.6781 ± 2.7473 | -0.3639 ± 0.2416 |
| a2780cis | 4078 | -2.6941 ± 5.1968 | -18.9316 ± 1.6164 | -2.8054 ± 5.4194 |
| **4-cell aggregate** | — | **-1.5030 ± 1.3226** | — | — |

## Verify gate (target ≥ +0.30 mean across 4 cell lines, vs Wf-85 P3 anchor)

**Aggregate combined R² (mean of 4 cell-line means) = -1.5030** — ❌ **FAIL** (combined R² < 0.0; honest per-cell-line breakdown below)

## Interpretation

Per `party-twenty-spirit §3.5`: aggregate combined R² = -1.5030; this is the **MEASURED** Wf-86 Phase 3 production anchor for PlatinAI 4-cell.  Negative R² values are reported without hiding (the Wf-85 P3 anchor already established that the Hilbert+ASTSR combination is at-baseline for this synthetic cohort).

The two worst seeds (a549 seed=3 and a2780cis seed=3) show large negative R² values (-14.5 and -13.1), likely due to ASTSR converging to a near-constant predictor on those particular train/val/test splits.  This is the expected failure mode for the 12-d RDKit 2D descriptor on **Pt-coordination pIC50** — the IC50 signal lives in the metal-coordination geometry (number of dative bonds, leaving-group identity, trans effect), not in the 12-d ligand-only descriptor envelope.

## Per-cell-line per-seed R² detail

### mcf7 (ic50_mcf7_nm)

| seed | combined R² (test) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0161 | -22.5622 | -0.0161 | 0.000 | 22.50 |
| 1 | -0.0377 | -22.4993 | -0.0377 | 0.000 | 19.18 |
| 2 | -0.0157 | -24.5784 | -0.0157 | 0.000 | 19.89 |
| 3 | -0.0017 | -22.2392 | -0.0017 | 0.000 | 20.42 |
| 4 | +0.0008 | -27.0331 | +0.0008 | 0.000 | 20.30 |

### a549 (ic50_a549_nm)

| seed | combined R² (test) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0373 | -17.7317 | -0.0373 | 0.000 | 32.63 |
| 1 | -0.0382 | -17.0044 | -0.0382 | 0.000 | 18.68 |
| 2 | -0.0803 | -21.8369 | -0.0803 | 0.000 | 18.25 |
| 3 | -14.4922 | -20.3176 | -14.4922 | 0.000 | 36.40 |
| 4 | -0.0507 | -20.4318 | -0.0507 | 0.000 | 19.29 |

### a2780 (ic50_a2780_nm)

| seed | combined R² (test) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0363 | -29.6043 | -0.0363 | 0.000 | 23.63 |
| 1 | -0.4987 | -24.5205 | -0.4987 | 0.000 | 19.53 |
| 2 | -0.6007 | -28.5085 | -0.6007 | 0.000 | 23.96 |
| 3 | -0.1081 | -27.7533 | -0.1081 | 0.000 | 25.09 |
| 4 | -0.5757 | -33.0037 | -0.5757 | 0.000 | 19.80 |

### a2780cis (ic50_a2780cis_nm)

| seed | combined R² (test) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0811 | -16.8811 | -0.0811 | 0.000 | 34.17 |
| 1 | -0.1165 | -17.8487 | -0.1165 | 0.000 | 20.28 |
| 2 | -0.0937 | -20.2312 | -0.0937 | 0.000 | 21.54 |
| 3 | -13.0877 | -18.3795 | -13.6442 | 0.250 | 34.30 |
| 4 | -0.0915 | -21.3177 | -0.0915 | 0.000 | 19.31 |

## Honest framing

- **MEASURED today**: 5 seeds × strict-scaffold × 4 cell lines on PlatinAI 4-cell
  synthetic pIC50 data; Hilbert+ASTSR convex blend with per-seed alpha selection.
  Source script: `molmetal/scripts/bench_hilbert_astr_combined_real.py`
  (Wf-85 P3 anchor).  This Wf-86 P3 report is a **schema transformation**
  of the Wf-85 P3 MEASURED outputs (5-fold style with test/val/train
  per-fold entries).
- **Strict-scaffold split**: Bemis-Murcko scaffold is computed
  on-the-fly from canonical SMILES, with a two-tier fallback to
  canonicalised ligand SMILES when Bemis-Murcko returns empty
  (PlatinAI 4-cell has many Pt-coordination complexes with open-chain
  or saturated ligands that lack ring systems).
  Each scaffold is assigned to exactly one split (train/val/test); this
  eliminates scaffold leakage between train and test.
- **Per-cell-line cohort sizes**: mcf7=4078, a549=4078,
  a2780=4078, a2780cis=4078 (PlatinAI 4-cell has the same
  rows × 4 cell-line IC50 columns).
- **Alpha selection**: per-seed 90/10 inner validation; grid over
  alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}; reported as `alpha_chosen` per seed.
- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged in `per_seed`.
- **No Multi-PH + PLL**: this is the **combined Hilbert+ASTSR** anchor;
  Multi-PH/PLL augmentation is a separate workstream (Wf-79 P2 3-way + PLL).
- **CPU smoke**: per-cell-line fits in < 5 min on a single CPU core for
  the 4 078-row cohort (full 4-cell × 5-seed × strict-scaffold benchmark
  runs in < 8 min total).  GPU-0 Blackwell optional but not required.

Per `party-twenty-spirit §3.5`: aggregate combined R² = -1.5030; this is the **MEASURED** Wf-86 Phase 3 production anchor.
