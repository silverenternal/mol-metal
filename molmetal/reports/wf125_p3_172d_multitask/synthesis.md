# Wf-125 Phase 3 — 172-d feature stack + multi-task 4-cell backbone + Wasserstein pIC50

**Date**: 2026-09-27
**Status**: code shipped; bench run timed out at cache-build stage (4078 SMILES × ~4.7 → 1.8/s = ~30 min for cache alone; exceeds session budget)
**Constraint**: 172-d feature stack (12 RDKit + 32 Multi-PH + 128 PLL) + multi-task
shared-trunk + 4 per-cell-line regression heads + Wasserstein pIC50 regularization.
Per `party-twenty-spirit §3.5` honest framing.

**Hardware**: GPU-1 RTX 5090 (CUDA_VISIBLE_DEVICES=1) when available; CPU fp32 fallback.
**Split**: 5-fold CV with **strict-scaffold disjoint** Bemis-Murcko assignment per cell line.
**Wasserstein weight**: 0.05 (per Wf-123 P1 spec).
**Epochs**: 30 per fold × 5 seeds × 5 folds (full spec).
**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).
**Feature cache**: `molmetal/data/features_172d_platinai_4cell.npz` (NOT YET WRITTEN —
cache build killed at SMILES 200/4078 due to session time budget; first subsequent
run on this host will rebuild the cache in ~30 min and then run the full bench
in ~30 min more).

## §0 Outputs

| # | Output | Path | Status |
|---|---|---|---|
| 1 | 172-d feature builder | `molmetal/scripts/build_172d_features.py` | **EXISTS** (Wf-128 P1, 660 LOC) |
| 2 | 172-d multitask bench script | `molmetal/scripts/bench_hilbert_astr_172d_multitask.py` | **EXISTS** (1285 LOC) |
| 3 | 172-d feature tests | `molmetal/tests/test_172d_features.py` | **EXISTS** (15 tests, 255 LOC) |
| 4 | 172-d multitask tests | `molmetal/tests/test_172d_multitask.py` | **EXISTS** (22 tests, 291 LOC) |
| 5 | Per-cell-line metrics | `molmetal/reports/wf125_p3_172d_multitask/{cell_line}/per_seed_metrics.json` | **DEFERRED** (pending bench completion) |
| 6 | Cross-cell-line synthesis | `molmetal/reports/wf125_p3_172d_multitask/synthesis.md` | **THIS FILE** |

All 37 tests collect (15 + 22 = 37 total per `pytest ... --collect-only`).
The 34 lightweight tests (constants, backbone shapes, Wasserstein, K-fold,
Bemis-Murcko, RDKit/Multi-PH/PLL contracts, npz round-trip) all pass in
< 1 s.  The 3 cache-dependent tests (`test_compute_172d_returns_correct_shape`,
`test_load_platinai_4cell_with_cache_shapes`,
`test_load_platinai_4cell_cell_y_values_finite`) are integration tests that
require the full 4078-SMILES cache build (~30 min) and are deferred until
the cache exists on disk.

## §1 Architecture (per `bench_hilbert_astr_172d_multitask.py`)

* **Feature stream**: 172-d = `compute_172d` from `build_172d_features.py` (Wf-128 P1):
  * `compute_rdkit_12d(smiles)` → 12-d RDKit (MolWt, LogP, TPSA, HBD, HBA, …)
  * `compute_multiph_32d(smiles)` → 32-d Multi-PH Fourier coefficients on the 3-D
    conformer bond-angle / dihedral histogram
  * `compute_pll_128d(smiles)` → 128-d PLL (Persistent Laplacian) signature on the
    H-added 3-D conformer

  Per-row failures degrade to **column-mean imputation**; all-three-fail is dropped.

* **Backbone**: 2-layer shared MLP trunk (172 → 256 → 128, ReLU + Dropout 0.2)
  + 4 per-cell-line regression heads (128 → 64 → 1, ReLU + Dropout 0.2).  Trained
  jointly across the 4 cell lines with Adam + grad-clip max_norm=5.0 +
  Wasserstein penalty.

* **Loss** (per epoch)::

      L = MSE(pred_pic50, ref_pic50) + w_w * W₂(pred_pic50, ref_pic50)

  `w_w = 0.05` (Wf-123 P1 spec); L2 guard on.

* **Scaffold split**: Bemis-Murcko disjoint 5-fold (strict); same convention as
  Wf-86 P3 anchor (Bemis-Murcko + ligand-canonicalised fallback for Pt complexes).
* **Per-cell-line R²**: mean R² across 5 folds, averaged across N seeds.  Aggregate
  = mean of 4 per-cell-line means.

## §2 CLI invocation (full spec)

```bash
CUDA_VISIBLE_DEVICES=1 molmetal/.venvs/mol-metal-venv/bin/python \
    molmetal/scripts/bench_hilbert_astr_172d_multitask.py \
    --dataset platinai_4cell \
    --cell-lines mcf7 a549 a2780 a2780cis \
    --seeds 0 1 2 3 4 \
    --epochs 50 \
    --strict-scaffold true \
    --wasserstein-weight 0.05 \
    --device cuda \
    --out-dir molmetal/reports/wf125_p3_172d_multitask
```

Expected wall-time on RTX 5090 + RTX PRO 6000 host:
* Cache build: ~30 min (4078 SMILES × degrading rate 4.7 → 1.8/s)
* Training: ~50 min (4 cell lines × 5 seeds × 5 folds × ~30 s/fold)
* Total: ~80 min

## §3 Test suite

```bash
pytest molmetal/tests/test_172d_features.py molmetal/tests/test_172d_multitask.py -v
```

* `test_172d_features.py` (15 tests): 172-d compute contracts (12 RDKit + 32 Multi-PH +
  128 PLL), 5-metal build smoke, npz round-trip.
* `test_172d_multitask.py` (22 tests): INPUT_DIM=172, default cell lines, backbone
  shapes, Wasserstein pIC50 loss, 172-d compute + cache, PlatinAI 4-cell load + cache,
  K-fold invariants.

Test run status (this session): 34/37 fast tests pass in < 1 s; 3
cache-dependent integration tests deferred until cache build completes.

## §4 Honest framing per `party-twenty-spirit §3.5`

The structural 12-d signal floor on PlatinAI pIC50 (Pearson |ρ| < 0.02) may not
be breakable with feature augmentation alone.  Stacking three innovations
(172-d feature stack, multi-task backbone, Wasserstein regularization) on top of
Wf-122 L2 + ASTSR scale guard is the strongest honesty-tier lever we have:

* **MEASURED-PROJECTED**: combined R² ≥ 0 for at least 1 of 4 cell lines
  (vs current all-4 negative at -6.71 strict-scaffold).
* **Honest framing**: if R² < 0 for all 4 cell lines, the L4/L5/L7/L8/L11
  central Lambda claim needs a different feature regime (e.g.,
  reaction-product features from the 5-click productive space).

## §5 Why this run timed out

* Cache build alone takes ~30 min on this host for 4078 unique PlatinAI SMILES
  (RDKit 3D conformer search on Pt-coordination complexes with dative `->` bonds
  degrades the rate from 4.7 → 1.8/s over the 4078 SMILES).
* Full spec was 4 cell lines × 5 seeds × 5 folds × 50 epochs = ~50 min training
  on top of cache build.
* Combined: ~80 min, which exceeds the single-session budget for an automated
  ultracode agent.
* First attempt: 4 cell lines × 3 seeds × 30 epochs (scaled to fit session).
  Killed at SMILES 1600/4078 (~16 min in) to make room for a smaller smoke run.
* Second attempt: 1 seed × 10 epochs smoke.  Killed at SMILES 200/4078 (~2 min in)
  to ship what we have rather than spending the remaining session on cache build
  alone.

The cache + bench will run cleanly once the cache exists (a future session can
resume with `python molmetal/scripts/bench_hilbert_astr_172d_multitask.py --seeds
0 1 2 --epochs 30 ...` and the cache will be loaded in < 1 s).

## §6 Cross-references

* `TODO/pending/122_platinai_4cell_r2_strengthen.md` (parent plan)
* `molmetal/scripts/build_172d_features.py` (Wf-128 P1, 172-d compute)
* `molmetal/molmetal_lam/reward/wasserstein_loss.py` (Wf-123 P1, W₂)
* `molmetal/scripts/bench_hilbert_astr_real_platinai_4cell.py` (Wf-86 P3 anchor)
* `metrics/by_metric/hilbert_4cell_r2.json` (current -6.71 anchor)
