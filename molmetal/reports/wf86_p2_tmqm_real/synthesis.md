# Wf-86 Phase 2 — REAL Hilbert+ASTSR combined R² on tmQM d-block 5-element

**Date**: 2026-09-26
**Target**: `coord_number` (regression; default `coord_number`)
**Constraint**: Combined Hilbert primal-dual + ASTSR evolutionary symbolic regressor
  (no Multi-PH + PLL); per `party-twenty-spirit §3.5` honest framing.
**Hardware**: CPU fp32 (default); `--device cuda` accepted but no Blackwell acceleration
  required for Wf-86 P2 production anchor.
**Split**: 5-fold CV with **strict-scaffold disjoint** scaffold assignment per element (Bemis-Murcko computed on-the-fly from canonical SMILES, with ligand-canonicalised fallback when Bemis-Murcko returns empty for open-chain ligands).
**Inputs**: `molmetal/data/tmqm_dblock_subset.csv`
  (5 000 mononuclear d-block complexes × 30 d-block metals).

## Cross-element summary (5-seed × 5-fold CV)

| element | n_samples | combined R² (mean ± std) | Hilbert-only R² | ASTSR-only R² |
|---|---:|---|---|---|
| Cu | 174 | -0.1101 ± 0.0273 | -1.8838 ± 0.0824 | -0.0811 ± 0.0538 |
| Fe | 261 | -0.4446 ± 0.5048 | -4.5172 ± 0.1165 | -0.4437 ± 0.5216 |
| Ni | 522 | -0.0874 ± 0.0169 | -3.0773 ± 0.0665 | -0.0853 ± 0.0363 |
| Zn | 365 | -0.1156 ± 0.0353 | -4.4639 ± 0.1313 | -0.1081 ± 0.0896 |
| Co | 179 | -0.2293 ± 0.1519 | -10.1855 ± 1.0542 | -0.1653 ± 0.1910 |
| **5-element aggregate** | — | **-0.1974 ± 0.1331** | — | — |

## Verify gate (target ≥ 0.0 mean across 5 elements)

**Aggregate combined R² (mean of 5 element means) = -0.1974** — ❌ **FAIL** (combined R² < 0.0; honest per-element breakdown below)

## Interpretation

The aggregate combined R² is **negative** (−0.20) — the Hilbert+ASTSR blend **does not beat the predict-the-mean baseline** for the `coord_number` regression task on tmQM d-block 5-element.

Three observations explain the negative result:

1. **ASTSR dominates the blend** — `alpha_mean ≈ 0.10` across all 25 (5 elements × 5 seeds) fits.  The inner-validation alpha grid consistently down-weights the Hilbert branch.
2. **Hilbert overfits dramatically** — Hilbert-only R² is **−1.9 to −10.2** (way worse than baseline).  This is the **N >> F failure mode** of the primal-dual least-squares solver: 64-d Hilbert feature map on 100-500 train rows leaves the system strongly underdetermined.
3. **ASTSR is at-baseline** — ASTSR-only R² ranges from **−0.08 to −0.44**, clustered very close to 0.  This is the expected behaviour for the 12-d RDKit 2D descriptor when no physico-chemical feature strongly correlates with coordination number — `coord_number` is a **structural property of the metal centre**, not a property of the ligand envelope.

**Bottom line**: the negative result is the **MEASURED honest anchor** for the Hilbert+ASTSR combined approach on the tmQM d-block `coord_number` regression task.  Path to R² ≥ 0 requires either (a) per-element structural features (e.g. dative-bond topology), (b) D-MPNN embeddings of the full metal-coordination complex, or (c) Multi-PH + PLL augmentation (Wf-79 P2 workstream).

## Per-element per-seed R² detail

### Cu (coord_number)

| seed | combined R² (overall) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0885 | -1.7595 | -0.0220 | 0.150 | 2.70 |
| 1 | -0.0815 | -1.9199 | -0.0141 | 0.150 | 1.48 |
| 2 | -0.1548 | -1.9027 | -0.1700 | 0.100 | 1.07 |
| 3 | -0.0625 | -1.6935 | -0.0220 | 0.100 | 1.54 |
| 4 | -0.0940 | -1.7136 | -0.0940 | 0.000 | 1.33 |

### Fe (coord_number)

| seed | combined R² (overall) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0003 | -4.2286 | -0.0003 | 0.000 | 2.33 |
| 1 | -0.7825 | -4.2614 | -1.1002 | 0.250 | 1.74 |
| 2 | -1.1863 | -4.5693 | -0.9205 | 0.100 | 1.68 |
| 3 | -0.0079 | -4.2581 | -0.0079 | 0.000 | 1.76 |
| 4 | -0.0447 | -4.2485 | -0.0003 | 0.050 | 1.74 |

### Ni (coord_number)

| seed | combined R² (overall) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0791 | -3.0235 | -0.0462 | 0.100 | 4.34 |
| 1 | -0.0825 | -2.9565 | -0.0825 | 0.000 | 3.44 |
| 2 | -0.0935 | -3.0101 | -0.0935 | 0.000 | 3.40 |
| 3 | -0.0960 | -3.0696 | -0.1240 | 0.200 | 3.30 |
| 4 | -0.0568 | -3.1349 | -0.0421 | 0.050 | 3.46 |

### Zn (coord_number)

| seed | combined R² (overall) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.1198 | -4.5275 | -0.0235 | 0.050 | 2.48 |
| 1 | -0.0709 | -4.3248 | -0.0235 | 0.050 | 2.35 |
| 2 | -0.0876 | -4.3932 | -0.0235 | 0.050 | 3.01 |
| 3 | -0.1457 | -4.2113 | -0.1946 | 0.150 | 2.36 |
| 4 | -0.0556 | -4.2784 | -0.1832 | 0.150 | 3.16 |

### Co (coord_number)

| seed | combined R² (overall) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0001 | -8.6481 | -0.0001 | 0.000 | 1.21 |
| 1 | -0.3259 | -8.7793 | -0.3259 | 0.000 | 1.47 |
| 2 | -0.3458 | -8.7670 | -0.3458 | 0.000 | 1.22 |
| 3 | -0.0983 | -8.9725 | -0.0001 | 0.050 | 1.24 |
| 4 | -0.1482 | -8.5854 | -0.0001 | 0.050 | 1.30 |

## Honest framing

- **MEASURED today**: 5 seeds × 5-fold × 5 d-block elements on tmQM
  d-block 5 000-row subset (regression on `coord_number`);
  Hilbert+ASTSR convex blend with per-fold alpha selection.
- **Strict-scaffold split**: Bemis-Murcko scaffold is computed
  on-the-fly from canonical SMILES, with a two-tier fallback to
  canonicalised ligand SMILES when Bemis-Murcko returns empty
  (tmQM d-block has many open-chain ligands that lack ring systems).
  Each scaffold is assigned to exactly one fold's test set; this
  eliminates scaffold leakage between train and test.
- **Per-element cohort sizes**: Cu=174, Fe=261, Ni=522, Zn=365,
  Co=179 (5 000-row tmQM d-block subset).  Most rows have a
  unique ligand, so the strict-scaffold split acts near-identically
  to random split — the strict-scaffold invariant (no scaffold in
  both train and test) is preserved trivially when every row has a
  unique scaffold key.
- **Alpha selection**: per-fold 90/10 inner validation; grid over
  alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}; reported as `alpha_chosen` per fold.
- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged in `per_fold`.
- **No Multi-PH + PLL**: this is the **combined Hilbert+ASTSR** anchor;
  Multi-PH/PLL augmentation is a separate workstream (Wf-79 P2 3-way + PLL).
- **CPU smoke**: per-element fits in < 5 s on a single CPU core for
  the typical 100-500 row per-element cohort (full 5-element × 5-seed
  × 5-fold benchmark runs in < 60 s total).  GPU-1 RTX 5090 optional.

Per `party-twenty-spirit §3.5`: aggregate combined R² = -0.1974; this is the **MEASURED** Wf-86 Phase 2 production anchor.
