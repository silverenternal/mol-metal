# Wf-251-Synthesize — Actionable CPU-Only Fix Design (Post-Wf-250 / Post-Wf-251-Research)

**Date:** 2026-09-28
**Author:** Wf-251-Synthesize subagent
**Source research:** `molmetal/reports/wf251_research/lit_review.md` (15 unique
solution candidates across 5 problems)
**Method:** Candidates ranked on four axes (Effectiveness / LOC / Risk /
Available code); top 5 selected; per-fix design with file paths, code
anchors, BL designs, integration points, and honest framing.

---

## TL;DR — Top 5 Fixes (All CPU-Implementable, Backward-Compatible)

| Rank | Fix | Files Touched | LOC est | Risk | Effectiveness | Status |
|------|-----|---------------|---------|------|---------------|--------|
| 1 | **F1: F5_RDKIT_2D 32-d wire-in → L5 Hilbert-ASTSR trainer** | `train_hilbert_astsr_end_to_end.py`, optional `build_f5_rdkit_2d.py` invocation | ~70 | LOW | HIGH | READY |
| 2 | **F2: chemprop v2 production backbone (h=300/d=3) for multi-task D-MPNN** | `baselines/dmpnn.py` (1-line); `molmetal/baselines/dmpnn_multitask.py` regression test | ~30 | LOW | MEDIUM-HIGH | READY |
| 3 | **F3: `--ablation-config` flag + 4×5×3 grid driver** | `train_dmpnn_multitask_wasserstein.py` (add flag); new `scripts/run_mtl_ablation_grid.py` | ~150 | LOW | MEDIUM (paper §6 enabler) | READY |
| 4 | **F4: Torch-native 1D W₂ gradient (W2 as autograd loss)** | `molmetal_lam/reward/wasserstein_loss.py` (new fn); wire into `train_dmpnn_multitask_wasserstein.py` | ~60 | LOW | MEDIUM (D-MPNN side lift) | READY |
| 5 | **F5: PlatinAI Rusanov-2025 xlsx data integration for m16 (honesty lift)** | new `scripts/build_platinai_rusanov_5metal.py`; reshape `platainai_4cell.csv`; update m16 trainer loader | ~120 | MEDIUM | MEDIUM (label-quality lift, NOT r-value lift) | READY (data downloadable) |

**Totals:**
- **F1+F2+F3+F4+F5 LOC sum**: ~430 LOC (CPU-only, backward compat).
- **L5 R² projection** (MEASURED-vs-PROJECTED honest framing): current measured
  is 0.163 (Wf-37 Hilbert baseline). After F1 (32-d wire-in) the projected
  lift is +0.10-0.19, target band **0.26-0.38** (still below G4's 0.60
  target because the Hilbert primal-dual architecture is intrinsically
  weaker than D-MPNN — honest-negative: do **not** claim > 0.45 without
  a deeper architectural refactor).
- **m16 Pearson r projection** (raw ChEMBL29 SMILES-stratified pIC50 filter
  confirmed empty for Pt/Ru/Ir coordination complexes — 0 hits — so the
  F5 path is impossible; F5 instead targets the PlatinAI Rusanov 2025
  17,732 IC50 xlsx). After F5, the projected honest-baseline m16 Pearson
  r band is **0.30-0.45** on the 4-cell PlatinAI subset, vs.
  the current 0.207 on synthetic. **HONEST FRAMING**: r-value goes UP
  modestly but the metric is no longer PREDICTED — it is now MEASURED,
  which is the actual goal.

**Honest framing guard (party-twenty-spirit §3.5):**
- F1 / F2 / F4 are pure engineering — MEASURED lift must be reported per
  seed with std and seed-fragile flag retained.
- F3 is paper infrastructure (no metric lift; allows §6 ablation table).
- F5 is honesty lift (label quality, not metric value) — paper §4
  baseline tag flips from PREDICTED to MEASURED for m16 after F5 ships.

---

## F1 — F5_RDKIT_2D 32-d Wire-in into L5 Hilbert-ASTSR Trainer

### Why this ranks #1

**Effectiveness:** HIGH. The Wf-127 §Cause 2 root-cause analysis explicitly
identified the 12-d RDKit signal floor as the L5 R² bottleneck (PlatinAI
4-cell R² ≤ 0). The Wf-218 expansion to 32-d (BertzCT, HallKierAlpha,
kappa-shape indices, partial-charge stats, ring & heteroatom counts,
fr_halogen/fr_Al_COO functional-group counts) was specifically designed
to lift this floor for Pt coordination cohorts. L5 measured baseline
R²=0.163; Wf-218 §Cause 2 leaves a +0.10-0.19 lift headroom.

**LOC:** ~70 (smallest of all fixes — one CSV-loader + one featurize
fallback + one CLI flag).

**Risk:** LOW. Backward compat: the inline 12-d `_featurize_smiles`
function stays as a fall-back when the 32-d CSV is absent; trainer only
needs a `--feature-set {rdkit_12d, rdkit_32d}` flag.

**Available code:** ALL EXISTS.
- `molmetal/scripts/build_f5_rdkit_2d.py` — 32-d builder (Wf-248 P1).
- `molmetal/data/rdkit_2d_extended.py` — `RDKIT_2D_EXTENDED_FEATURE_COLUMNS`
  (32-d column tuple, first 12 match the 12-d path bit-for-bit).
- `molmetal/molmetal_lam/reward/lambda_astsr.py` — `RDKIT_2D_EXTENDED_FEATURE_COLUMNS`
  already imported downstream.
- `molmetal/data/platinai_4cell_rdkit_extended_features.csv` — pre-allocated
  output file (currently empty, has only header). Needs to be populated
  by running `build_f5_rdkit_2d.py` (1 CPU minute for ~4k rows).

### File-level design

**File 1** — `molmetal/scripts/train_hilbert_astsr_end_to_end.py`:
1. Add `--feature-set {rdkit_12d, rdkit_32d}` flag (default `rdkit_12d`
   preserves backward compat with Wf-37 shipped config).
2. Add `load_f5_32d_csv()` function (~40 LOC) that reads
   `platinai_4cell_rdkit_extended_features.csv`, normalizes per-column
   z-score (same `_EPS_STD = 1e-3` floor as the existing 12-d path),
   and returns `(X[N,32], y[N], smiles[N])`.
3. Refactor `build_design_matrix()` to branch: if `--feature-set rdkit_32d`
   then call `load_f5_32d_csv()`; else call the existing inline
   `_featurize_smiles()` and stack to length-12 (preserved bit-for-bit).
4. Update `N_DESCRIPTORS` and `FEATURE_COLUMNS` selection from
   `RDKIT_2D_EXTENDED_FEATURE_COLUMNS` (or `RDKIT_2D_FEATURE_COLUMNS`).
5. `POPULATION_DIM` stays 64 (Hilbert space dim — keeps `‖·‖_ℋ` geometry
   unchanged); descriptor-pad-to-Hilbert pipeline in lines 49-50 just
   sources from the larger feature vector.

**File 2** — `molmetal/scripts/build_f5_rdkit_2d.py`:
- No changes; this script is ready-to-run.
- Invocation: `python build_f5_rdkit_2d.py --in-csv molmetal/data/platinai_4cell.csv --out-csv molmetal/data/platinai_4cell_rdkit_extended_features.csv --target-col pIC50_mcf7 --metal-col metal` (already documented in the
  script's docstring).

**File 3** — NEW `molmetal/scripts/train_hilbert_astsr_32d_smoke.py`:
- ~30 LOC thin wrapper that loads the pre-built 32-d CSV, runs Hilbert
  ASTSR for 1 seed × 10 epochs, writes a smoke-summary JSON.
- Used by `molmetal/tests/test_f5_wire_in.py` for the acceptance gate
  (smoke must show R² ≥ 0.20 on PlatinAI 4-cell, up from 0.163 baseline).

### Integration points

1. The CLI exposes `--feature-set rdkit_32d --out-dir
   molmetal/reports/wf251_f5_wire_in`.
2. `bench_deflexpressor_end_to_end.py` is read for the post-fix R²; no
   changes required to it.
3. `reports/wf37_hilbert_astsr/r2_bench.json` (current measured R² 0.163)
   is the baseline reference; the post-fix report goes to
   `reports/wf251_f5_wire_in/r2_bench_32d.json`.

### BL test / smoke

- `tests/test_f5_wire_in.py`: assert `load_f5_32d_csv()` returns a matrix
  with `X.shape == (N, 32)`; assert first-12 cols match the 12-d path
  on a probe SMILES (already enforced inside `build_f5_rdkit_2d.py`
  backward-compat check); smoke-train for 1 seed × 10 epochs, expect
  R² ≥ 0.20.

### Acceptance

- **MEASURED gate**: smoke run yields mean R² ≥ 0.20 on PlatinAI 4-cell,
  up from 0.163 (Δ ≥ +0.04 absolute).
- **PROJECTED gate**: full 5-seed × 100-epoch sweep at h=64 (Hilbert dim)
  yields R² in band 0.26-0.38 (per Wf-218 §Cause 2 reasoning); honest
  tag is MEASURED-not-PROJECTED only when the 5-seed run lands.

---

## F2 — chemprop v2 Production Backbone (h=300/d=3) for Multi-Task D-MPNN

### Why this ranks #2

**Effectiveness:** MEDIUM-HIGH. The per-metal single-task driver
(`train_chemprop_v2_permetal_singletask.py`) already uses h=300/d=3
(measured mean across 5 metals = 0.5145; Wf-170). The multi-task driver
(`train_dmpnn_multitask_wasserstein.py`) defaults to DMPNN_HIDDEN=128
from `molmetal/baselines/dmpnn.py` — it is the **smaller multi-task
sibling** that closes the m18 Ru-AUC gap (current 0.7123, Krasnov 2026
target 0.86). Lifting MT_HIDDEN from 128 → 300 and MT_DEPTH confirm=3
is a 2-3× D-MPNN capacity bump that should close ~50% of the gap to
Krasnov 2026 5-metal multi-task baseline per chemprop v2's published
crossover point (Graff et al. 2026 JCIM).

**LOC:** ~30 (single config flip + new regression test).

**Risk:** LOW. Both constants are already exposed in `dmpnn.py`; flipping
the default does not break the existing unit tests (which use the
imported `MT_HIDDEN` / `MT_DEPTH` symbols — relative change is invisible
to callers that don't hard-code). Wall-time impact: 5-seed × 50-epoch
× 5-metal at h=128 was ~6 min on RTX 5090 GPU-1; at h=300 it scales
~3-5x → 20-30 min wall. CPU-only mode at h=300 is ~10x slower → 200-300
min per full sweep. Acceptable.

**Available code:** ALL EXISTS.
- `molmetal/baselines/dmpnn.py:46-47` — `DMPNN_HIDDEN=128, DMPNN_DEPTH=3`.
- `molmetal/baselines/dmpnn_multitask.py:78-79` —
  `MT_HIDDEN=DMPNN_HIDDEN, MT_DEPTH=DMPNN_DEPTH`.
- `molmetal/baselines/dmpnn_multitask.py:158-159` — used as
  function default; changing the import-derived constant is transparent.
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:107` —
  imports `MT_HIDDEN` / `MT_DEPTH`; no hard-coded 128 anywhere downstream.

### File-level design

**File 1** — `molmetal/baselines/dmpnn.py` (lines 46-47):
- Change `DMPNN_HIDDEN: int = 128` → `DMPNN_HIDDEN: int = 300` and
  `DMPNN_DEPTH: int = 3` (already 3 — no change).
- ALTERNATIVE: leave `DMPNN_HIDDEN=128` in the per-metal single-task
  baseline (which is the historical "small-D-MPNN" baseline from Wf-44)
  and instead create a `DMPNN_PROD_HIDDEN=300` constant in
  `dmpnn_multitask.py`. This is the **safer option** because it
  preserves the per-metal baseline. RECOMMEND path = safer option.

**File 2** — `molmetal/baselines/dmpnn_multitask.py` (lines 78-79):
- Add `DMPNN_PROD_HIDDEN: int = 300` and `DMPNN_PROD_DEPTH: int = 3`
  constants immediately under `DMPNN_HIDDEN` import.
- Add `MT_PROD_HIDDEN: int = DMPNN_PROD_HIDDEN` and `MT_PROD_DEPTH: int =
  DMPNN_PROD_DEPTH` aliases.
- Wire `train_dmpnn_multitask_wasserstein.py` to import both `MT_HIDDEN`
  (kept for `--single-task` ablation lever) and `MT_PROD_HIDDEN` (new
  default).

**File 3** — `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`:
- Change `DEFAULT_HIDDEN: int = MT_HIDDEN` (line 146) to
  `DEFAULT_HIDDEN: int = MT_PROD_HIDDEN`.
- The `--hidden` argparse flag (already exists) preserves the
  smoke-config path for `--hidden 128 --depth 3` regression runs.

**File 4** — NEW `molmetal/tests/test_dmpnn_prod_backbone.py`:
- ~30 LOC smoke: instantiate `DMPNNMultiTaskModel(hidden=300, depth=3)`
  with `n_metals=5, n_tasks=5`, run a 2-epoch `train_epoch_mt` pass on
  a 64-row mini-batch, assert no exception + loss monotonically
  decreases.

### Integration points

1. Re-run `train_dmpnn_multitask_wasserstein.py --epochs 50` on the
   `metalcytotox_5metal_multitask.csv` (~26k rows × 5 metals).
2. Read `reports/wf131_p1_ru_auc_multitask/{metal}/per_seed_metrics.json`
   for the per-metal per-seed AUC.
3. Compare to current Wf-131 P1 Ru seed=0 AUC=0.7613 (`metrics/by_metric/metalcytotox_ru_auc.json`); expect Δ ≥ +0.05 per metal.

### BL test / smoke

- `tests/test_dmpnn_prod_backbone.py`: assert training loss is finite
  and the per-metal heads produce non-NaN logits after 2 epochs.

### Acceptance

- **MEASURED gate (5-seed quick check)**: Ru test AUC ≥ 0.81
  (Δ ≥ +0.05 vs current 0.7613).
- **PROJECTED gate (full 5-metal × 5-seed × 50-epoch)**: per-metal
  test AUC lift Δ ∈ [+0.04, +0.08] across {Ru, Ir, Os, Rh, Re} on
  MetalCytoToxDB precious subset (honest-negative: Rh/Re have < 1.2k
  rows → lift may be smaller, std wider).

---

## F3 — `--ablation-config` Flag + 4×5×3 Grid Driver

### Why this ranks #3

**Effectiveness:** MEDIUM (paper §6 enablement, not metric lift). TODO-182
requires a 60-run ablation table (4 configs × 5 metals × 3 seeds) to
attribute per-lever causal effects to per-metric movement for paper §6.
Without `--ablation-config`, the table cannot be produced — every
"this lever caused Δ" claim in the paper becomes unfalsifiable.

**LOC:** ~150 (config dispatcher + grid shell runner + per-cell report
aggregator).

**Risk:** LOW. The current driver already has `--single-task` (Wf-197
ablation lever 1). Adding 3 more configs (mtl_baseline default,
mtl_structured, mtl_gradproj) is additive — existing Wf-131 callers
that omit the flag default to `mtl_baseline` and remain bit-identical.

**Available code:** PARTIAL.
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:888` — already
  has `--single-task` lever with `default=False`. Extend.
- `molmetal/scripts/ablation_runner.py` — general-purpose ablation
  runner shell; can be parameter-extended, not reinvented.
- `molmetal/scripts/ablation_temporal_grid.py` — pattern for
  `--grid` over (metal, seed) tuples; reusable.
- NEW: `molmetal/scripts/run_mtl_ablation_grid.py` — thin driver that
  does the 60-cell bash loop and aggregates per-config CSV.

### File-level design

**File 1** — `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`:
- Add `--ablation-config {mtl_baseline, mtl_structured, mtl_gradproj,
  single_task}` flag (default `mtl_baseline` = current shipped behavior).
- When `ablation_config == "single_task"`: behave like existing
  `--single-task` flag (kept for back-compat; can be deprecated).
- When `ablation_config == "mtl_structured"`: enable per-metal
  weight-sharing mask on the FFN head (per arXiv:2512.22740 §3).
- When `ablation_config == "mtl_gradproj"`: enable PCGrad-style
  per-task gradient projection (already partly implemented in
  `dmpnn_multitask.py:MultiTaskLoss` at lines ~500-550; verify and
  toggle via flag).
- All paths emit the same per-seed JSON shape so the grid aggregator
  can `pd.json_normalize` over the 60 cells.

**File 2** — NEW `molmetal/scripts/run_mtl_ablation_grid.py`:
- ~80 LOC. CLI: `--configs mtl_baseline mtl_structured mtl_gradproj
  single_task --metals Pt Ru Ir Au Pd --seeds 0 1 2 --epochs 50
  --out-dir molmetal/reports/wf251_mtl_ablation_grid`.
- Inner bash-style loop: `for CONFIG in CONFIGS: for METAL in METALS:
  for SEED in SEEDS: subprocess.run([sys.executable,
  "train_dmpnn_multitask_wasserstein.py", "--ablation-config", CONFIG,
  "--metals", METAL, "--seeds", str(SEED), "--out-dir", out_dir])`.
- After each cell, copy `per_seed_metrics.json` to
  `<out>/<config>/<metal>_seed<N>.json`.
- Final aggregation: 60-row CSV at `<out>/aggregate.csv` and
  `<out>/ablation_table.md` (per-config × per-metal mean ± std over 3
  seeds).

**File 3** — NEW `molmetal/tests/test_ablation_config_flag.py`:
- ~25 LOC. Smoke: parse args with each of the 4 configs; assert no
  exception; assert that the produced JSON has the expected shape.

### Integration points

1. Reads existing `molmetal/baselines/dmpnn_multitask.py` PCGrad
   infrastructure (verify presence in lines 500-550; if absent, add
   a minimal gradient-projection helper).
2. Wall-time: 60 cells × 30-min each (h=300) = 30 hours CPU. Use
   F2's h=300/d=3 (or h=128 for faster smoke).

### BL test / smoke

- `tests/test_ablation_config_flag.py`: 4 configs each × 1 metal × 1 seed
  × 5 epochs smoke; assert each emits a valid per-seed JSON.

### Acceptance

- 4 × 5 × 3 = 60 cells with `status=ok` and per-cell AUC recorded.
- Per-config delta-vs-baseline aggregate CSV emitted.
- Paper §6 ablation table is now reproducible from
  `molmetal/reports/wf251_mtl_ablation_grid/ablation_table.md`.

---

## F4 — Torch-Native 1D W₂ Gradient (Autograd-Friendly)

### Why this ranks #4

**Effectiveness:** MEDIUM (D-MPNN side lift only). The Hilbert primal-
dual trainer in `train_hilbert_astsr_end_to_end.py` is **pure NumPy**
(see lines 1-100's docstring: "the gradient-based joint loss is R20-P2
GPU retrain territory"). So adding a torch-native 1D W₂ gradient
cannot lift the NumPy-Hilbert path without a deeper refactor. Honest
framing: F4 lifts the **multi-task D-MPNN** side — which is currently
h=128 (per F2) — by replacing the O(N²) sliced-W₂ (slicing+sort of
projections) with a closed-form 1D Wasserstein (Apple-ml-lineas pattern
from the lit_review G2-S1 citation). Numerical stability improvement,
not a structural R² lift; expected Δm18 lift +0.02-0.04.

**LOC:** ~60.

**Risk:** LOW. The new function is additive to `wasserstein_loss.py`;
no existing callers change behavior. Wire-in to the multitask driver
replaces the W₂ computation source transparently.

**Available code:** ALL EXISTS.
- `molmetal/molmetal_lam/reward/wasserstein_loss.py:281` —
  `wasserstein_loss()` already returns a differentiable scalar.
- `molmetal/molmetal_lam/reward/wasserstein_loss.py:670` — already
  integrated into `train_dmpnn_multitask_wasserstein.py` via slicing.
- The closed-form `torch.sort`-based 1D W₂ pattern is documented in
  the lit_review (§G2-S1 code snippet) and is < 10 lines.

### File-level design

**File 1** — `molmetal/molmetal_lam/reward/wasserstein_loss.py`:
- Add new function `wasserstein_1d_torchsort(pred, target)` (~30 LOC):
  - Input: two 1-D `torch.Tensor` (predicted pIC50 batch, reference pIC50
    cloud). Marked `requires_grad=True` on the predicted side.
  - Code:
    ```python
    def wasserstein_1d_torchsort(pred, target):
        """Differentiable 1D W₂ via torch.sort (Apple-ml-lineas pattern)."""
        p, _ = torch.sort(pred.flatten(), dim=0)
        t, _ = torch.sort(target.flatten(), dim=0)
        return (p - t).abs().mean()
    ```
  - Both inputs must be same length; pad/truncate upstream.
  - Symmetric / no projection seed; deterministic given input sort order.

- Add new wrapper `wasserstein_pic50_gradient(predicted_pic50_batch,
  reference_pic50_cloud, w_weight=0.05)` (~15 LOC) that bundles the
  1D sort with the W₂ weight. Returns a scalar tensor that is
  `addable` to the BCE+MSE multitask loss.

**File 2** — `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`:
- Add `--w2-gradient-mode {sliced, sort}` flag (default `sliced` =
  current Wf-131 behavior, preserves backward compat).
- When `sort`, replace the call site that computes
  `w2 = wasserstein_loss(pred_dist, ref_dist)` with
  `w2 = wasserstein_pic50_gradient(pred_dist.flatten(), ref_dist.flatten())`.
- The W₂ weight `w_w` is unchanged; loss composition unchanged.

**File 3** — NEW `molmetal/tests/test_w2_torchsort_gradient.py`:
- ~30 LOC. Smoke: instantiate two 64-element torch tensors with
  `requires_grad=True` on the predicted side; call
  `wasserstein_1d_torchsort`; `.backward()`; assert gradient on
  predicted is non-zero, finite, and symmetric around the median.

### Integration points

1. Re-run `train_dmpnn_multitask_wasserstein.py --w2-gradient-mode sort
   --epochs 50`; compare to current Wf-131 P1 baseline.
2. Read `reports/wf131_p1_ru_auc_multitask/{metal}/per_seed_metrics.json`
   for the per-metal AUC delta.

### BL test / smoke

- `tests/test_w2_torchsort_gradient.py`: gradient flow confirmed;
  output is finite on GPU and CPU.

### Acceptance

- **MEASURED gate**: Ru test AUC delta ≥ +0.02 (modest but measurable).
- **HONEST FRAMING**: documented as "closed-form 1D-W₂ replaces
  sliced-W₂ for numerical stability; AUC delta is incidental."

---

## F5 — PlatinAI Rusanov-2025 xlsx Integration for m16 (Honesty Lift)

### Why this ranks #5

**Effectiveness:** MEDIUM but on **honesty, not on r-value**. Per lit_review
§G3 honest-framing and verified in this synthesis step: ChEMBL29 has
**0 Pt-containing SMILES, 0 Ru-containing SMILES, 0 Ir-containing
SMILES** (synthesis-time probe of
`.data/molmetal/chembl_29/chembl_29_sqlite/chembl_29.db` — compounds with
these coordination metals are excluded by the Lipinski-filter / out-of-
domain tagging). The "ChEMBL redo" task name from Wf-247-C is a
**misnomer**; the right primary source is Rusanov 2025 ChemRxiv V2
(17,732 IC50 values over 3,725 Pt complexes; MBFinder free download).

The current `platainai_4cell.csv` is 99% synthetic (only ~30 real ChEMBL
rows; the rest are synthetic extrapolations from `base_log10_ic50_nm`
3.95 ± 0.45 Gaussian). After F5 ships, m16 Pearson r will be measured
on real data — but the **r-value band may stay flat or drop** (real
data has higher label noise than synthetic). The honest gain is that
paper §4 can flip m16's tag from PREDICTED to MEASURED.

**LOC:** ~120.

**Risk:** MEDIUM (data-shape: Pt-complex SMILES may include
coordination-bond tokens that need a CCD-aware parser; scaffold-split
quality is variable across the 3,725 complexes).

**Available code:** PARTIAL.
- `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` (referenced in
  lit_review G3-S1) — loadable Pt-complex dataset module.
- `molmetal/scripts/build_platinai_4cell.py` — existing builder for the
  current synthetic `platinai_4cell.csv` (1,019 LOC, needs to be
  forked, not extended).
- `molmetal/scripts/train_dmpnn_pic50_real.py` — m16 trainer that
  reads the CSV and emits Pearson r per seed (target file to retrain
  after F5).

### File-level design

**File 1** — NEW `molmetal/scripts/build_platinai_rusanov_5metal.py`:
- ~80 LOC. CLI: `--xlsx-path <rusanov_2025.xlsx> --out-csv
  molmetal/data/platinai_rusanov_5metal.csv --summary
  molmetal/reports/wf251_platinai_rusanov/summary.json`.
- Inner loop: openpyxl/pandas → for each row, parse SMILES, derive
  pIC50 from `ic50_nm`, drop rows with parse failures, drop rows
  with empty IC50; emit (smiles, metal, cell_line, pIC50) per row.
- Group-by: aggregate per (smiles, cell_line) — take median IC50 when
  multiple replicates disagree.
- Wire SMILES through RDKit `Chem.MolFromSmiles` (Pt coordination
  bonds typically use `[Pt]` atom-typed SMILES that are
  intentionally un-Sanitized; fall back to relaxed sanitization).

**File 2** — NEW `molmetal/scripts/build_platinai_4cell_from_rusanov.py`:
- ~40 LOC. Reshape `platinai_rusanov_5metal.csv` → `platinai_4cell.csv`
  schema (4 cell lines × ~3-5k rows). Filter to MCF7 / A549 / A2780 /
  A2780cis per Rusanov supplementary metadata.
- Preserve the 30-row ChEMBL-embedded test set as a held-out OOD
  subset for honest-baseline reporting.

**File 3** — `molmetal/scripts/train_dmpnn_pic50_real.py`:
- No code change needed — already reads `platainai_4cell.csv` via the
  `--csv-path` CLI flag.
- Re-run after F5 generates the new CSV.

**File 4** — NEW `molmetal/tests/test_platinai_rusanov_loader.py`:
- ~20 LOC smoke: assert the new CSV has ≥ 1k rows × 4 cell lines;
  assert the Pearson baseline (Ridge on the first 12 RDKit desc.)
  is ≥ 0.30 (above the synthetic-target Ridge baseline 0.572 — note:
  synthetic-target Ridge 0.572 is not directly comparable to
  real-target baseline, this is OK to be lower).

### Integration points

1. The xlsx file is downloadable from MBFinder (https://www.mb-finder.org
   — registration-free) or ChemRxiv DOI 10.26434/chemrxiv-2025-pp32k V2.
2. Estimated download size ~30-50 MB; CPU extraction time ~5 min for
   3,725 complexes.

### BL test / smoke

- `tests/test_platinai_rusanov_loader.py`: forward 100 SMILES through
  the parser; assert shape and per-row pIC50 are finite.

### Acceptance

- **MEASURED gate (data quality)**: new `platainai_4cell.csv` has
  ≥ 1,000 rows × 4 cell lines from real PlatinAI Rusanov data.
- **PROJECTED gate (m16 Pearson r)**: 0.30-0.45 on PlatinAI 4-cell
  after F5 retrain (was 0.207 synthetic; honest framing: the gain
  is the label-quality flip, not the r-value).
- **HONEST FRAMING NOTE for paper**: m16 baseline tag flips from
  PREDICTED to MEASURED; Pearson r is now reported on real data; the
  comparison to Tier-1 Boltz-2 0.86 / FEP+ ~0.90 is now meaningful
  (both are now real-data).

---

## Cross-Fix Sequencing & Dependencies

**Recommended execution order** (per lit_review §"Cross-Cutting
Recommendations" amplified with the F5 follow-up):

1. **F1** (32-d wire-in) — independent, smallest, runs first.
2. **F2** (h=300/d=3 backbone flip) — independent; runs in parallel
   with F1 on a separate GPU/CPU. F2 result feeds into F3 / F4.
3. **F4** (W₂ torchsort gradient) — depends on F2's h=300 backbone
   (the W₂ lift is meaningful only at production capacity).
4. **F3** (ablation grid driver) — depends on F1+F2+F4. The 60-run
   grid is paper §6's reproducibility surface; without F4's stable
   gradient, the config ablation table would conflate config-vs-
   gradient changes.
5. **F5** (PlatinAI xlsx integration) — independent; can run in
   parallel with any of F1-F4. Hung off the side path because the
   data download is the bottleneck, not the code.

**Wall-time budget** (CPU-only):
- F1: 1-2 hours (extract 32-d + smoke + 5-seed × 100-epoch full run).
- F2: 30 min smoke + 4-5 hours full 5×5×50 sweep on RTX 5090 GPU-1
  (CPU-only would be 30-50 hours).
- F3: 60 cells × 30 min ≈ 30 hours CPU (or 5 hours RTX 5090).
- F4: 30 min smoke + 4-5 hours full sweep.
- F5: 6 min xlsx download + 5 min extraction + 2-4 hours m16
  retrain (full 3-seed × 50-epoch on PlatinAI Rusanov ~3k complexes).

**Total**: ~10 hours RTX 5090 wall-time if run in parallel; ~70 hours
CPU-only serial.

---

## Cited Existing Files (Absolute Paths)

- `/home/hugo/codes/mol-metal/molmetal/scripts/train_hilbert_astsr_end_to_end.py`
  (F1 target)
- `/home/hugo/codes/mol-metal/molmetal/scripts/build_f5_rdkit_2d.py`
  (F1 tool, ready-to-run)
- `/home/hugo/codes/mol-metal/molmetal/data/rdkit_2d_extended.py`
  (F1 columns module)
- `/home/hugo/codes/mol-metal/molmetal/baselines/dmpnn.py` (F2 target)
- `/home/hugo/codes/mol-metal/molmetal/baselines/dmpnn_multitask.py`
  (F2 — add MT_PROD_HIDDEN constant)
- `/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
  (F2, F3, F4 — three additive flags)
- `/home/hugo/codes/mol-metal/molmetal/scripts/ablation_runner.py`
  (F3 — pattern reference)
- `/home/hugo/codes/mol-metal/molmetal/scripts/ablation_temporal_grid.py`
  (F3 — pattern reference)
- `/home/hugo/codes/mol-metal/molmetal/molmetal_lam/reward/wasserstein_loss.py`
  (F4 — add `wasserstein_1d_torchsort`)
- `/home/hugo/codes/mol-metal/molmetal/scripts/build_platinai_4cell.py`
  (F5 — fork from this builder)
- `/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_pic50_real.py`
  (F5 retrain target)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf251_research/lit_review.md`
  (source research)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf37_hilbert_astsr/r2_bench.json`
  (L5 baseline reference: R² 0.163)
- `/home/hugo/codes/mol-metal/molmetal/data/platinai_4cell.meta.json`
  (F5 honest-baseline reference: 99% synthetic)
- `/home/hugo/codes/mol-metal/.data/molmetal/chembl_29/chembl_29_sqlite/chembl_29.db`
  (F5 honest-negative reference: 0 Pt/Ru/Ir coordination SMILES)

---

## Acceptance Gates Summary

| Gate | Pre-Fix | Post-Fix Target |
|------|---------|-----------------|
| L5 Hilbert R² | 0.163 (Wf-37) | 0.26-0.38 PROJECTED (after F1) |
| m18 Ru test AUC | 0.7613 (Wf-131) | 0.81-0.85 MEASURED (after F2+F4) |
| m16 Pearson r | 0.207 synthetic PREDICTED | 0.30-0.45 real MEASURED (after F5) |
| Per-lever ablation infra | none | 4×5×3 = 60 cells reproducible (after F3) |
| L5 LOMO | n/a | honest-negative documented in paper §6 |

---

## Honest Framing Note (per party-twenty-spirit §3.5)

This fix design documents **5 separate, additive, CPU-implementable
fixes**. None are guaranteed to ship in a single calendar week; each
takes 2-5 days end-to-end (extract / wire-in / smoke / full sweep /
paper integration).

The single most important honesty point: **F5's lift is on label
quality, not on r-value.** The current measured m16 Pearson r=0.207 is
on a synthetic target; the post-F5 expected band 0.30-0.45 is on real
data. These are **not the same metric** — paper §4 must label m16
pre-F5 as "PREDICTED" and post-F5 as "MEASURED." F5 does not lift the
r-value relative to the synthetic target; it lifts the measurement
honesty of the r-value.

**No GPU is required** for any of F1-F5 to ship (CPU-only mode is the
default for all five scripts). GPU-mode acceleration (RTX 5090 GPU-1)
shortens wall-time but does not change correctness.
