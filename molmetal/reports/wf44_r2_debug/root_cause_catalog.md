# Wf-44 Phase 1 — Negative R-squared Root Cause Catalog

**Date**: 2026-09-26
**Workflow**: Wf-44 (R² debug — Hilbert / ASTSR / Affinity regression)
**Status**: Phase 1 — root cause catalog ship
**Companion**: `root_cause_checklist.json` (machine-readable per-cause triage)
**Audience**: Anyone debugging R² < 0 in Wf-25 / Wf-32 / Wf-37 / Wf-39
**Lit anchors**: Sun & Zheng 2026 (*数学学报* 53(4)) Chambolle-Pock 2011 / R17 ship notes (`TODO/completed/75_*.md`)

---

## §0 TL;DR

R² < 0 means **the regressor is worse than predicting the mean**. With four MEASURED negative-R² data points spanning Wf-25 / Wf-32 / Wf-37 / Wf-39, we catalog **8 distinct root causes** ranked by priority. Each cause has a concrete check method, an expected signal, and a mitigation. Phase 2 will execute the checklist top-down against Wf-32 m16 (Pearson r = -0.040) and Wf-37 L5 (Hilbert R² = +0.16 ship) to localize the regression mechanism before retraining.

| # | Root cause | Priority | Hit in known cases? |
|---:|---|:---:|:---|
| 1 | Baseline issue: model predicts all mean | **CRITICAL** | Wf-25, Wf-32 (smoke) |
| 2 | Feature mismatch: features not informative for target | **CRITICAL** | Wf-25 (binary target) |
| 3 | Target normalization reversed | **MAJOR** | Wf-25 (binary → pIC50 swap) |
| 4 | Data leakage / split not strict | **MAJOR** | Wf-32 (cohort mismatch) |
| 5 | Target distribution extreme (concentrated pIC50) | **MAJOR** | Wf-37 L5 (PlatinAI 4-cell) |
| 6 | Hilbert optimizer step too large → NaN → loss=0 | **MAJOR** | Wf-25 (constant lr) |
| 7 | Numerical precision: float32 R² noise floor ±0.05 | MINOR | Wf-39 seed 2 (R²=-1.43) |
| 8 | Feature-target correlation truly weak | MINOR | Open question |

---

## §1 Cause 1 — Baseline issue: model predicts all mean (CRITICAL)

**Symptom**: R² ≤ 0 with R²_var(y_test) < R²_var(pred) — the regressor's predictions have higher variance than the labels. Most common when:

* The model outputs a constant (e.g. mean) — `pred.std() == 0` and `R² = -Var(y)/Var(y) ≤ 0`.
* A sigmoid / softmax head clamps predictions to a narrow range.
* A weighted ensemble's weights collapse (e.g. softmax weight → 0 on one branch).

**Known hits**:

* **Wf-25 Hilbert composite** — `R² = -0.353 ± 0.074` (PlatinAI 5K + MetalCytoToxDB Ru 500). The Hilbert branch with tiling broadcast (8 → 64 by repetition) collapsed because the inner product over identical columns gave a degenerate solution that exactly matched the mean.
* **Wf-32 Phase 4 m16 smoke** — `Pearson r = -0.040` on PlatinAI 4-cell. Per-cell: `MCF7=-0.013, A549=-0.049, A2780=+0.024, A2780cis=-0.124` — all signs close to zero, consistent with a near-mean predictor.

**Check method**:

1. Compute `y_train.mean()` and `y_test.mean()`. If they are within `0.1 × std(y_train)`, the mean baseline is plausible.
2. Inspect `pred_test.std()` and `pred_test.mean()`. If `pred_test.std() < 0.1 × y_test.std()`, the regressor is collapsed.
3. Compare `R²(model)` vs `R²(constant_mean)` — both should be 0 (the constant). If `R²(model) < R²(constant_mean)`, the model is *worse* than mean.

**Expected signal**: `pred_test.std() / y_test.std() < 0.1` ⇒ **mean collapse** is the dominant cause.

**Mitigation**:

* Replace model head with `Linear(dim, 1)` + identity (no sigmoid), no clamp.
* Inspect Hilbert tile-inflate: change `tile(8, 8)` → `pad(8, 64) + learned bias col` (per Wf-37 fix).
* Confirm the loss is `MSE` (not `BCE` for regression).

---

## §2 Cause 2 — Feature mismatch: features not informative for target (CRITICAL)

**Symptom**: R² ≈ 0 across many seeds, but **train R² ≫ test R²**. The regressor can fit the training set but the features carry no transferable signal. Most common when:

* Features are 2D descriptors (RDKit 8-12 dim) but target is 3D (binding affinity with pocket context).
* Features describe molecular shape but target depends on electronic structure (HOMO/LUMO/Gap).
* Features include scaffold counts but target is dominated by counter-ion / salt form.

**Known hits**:

* **Wf-25** — 8-d RDKit descriptors (`mol_wt, logp, tpsa, hba, hbd, rot_bonds, ring_count, frac_sp3`) against noisy binary activity. The 8-d set has no QSAR-relevant electronic descriptors (no `num_heteroatoms`, `qed`, `num_aromatic_rings`); the binary activity is dominated by scaffold noise.
* **Wf-39 Deflexpressor** — 12-d RDKit descriptors against NCI-60 pIC50. `ast` branch `R² = -0.0905 ± 0.0173` (vs `hilbert` `R² = +0.1519` on seed 0). The 12-d set lifts the hilbert branch but the AST evolutionary search cannot exploit them — confirms that the ASTSR symbol library is too narrow for these features.

**Check method**:

1. Compute per-feature univariate Spearman ρ with `y_train`. If max `|ρ| < 0.05`, features carry no signal.
2. Train a ridge regressor on train; report train R² vs test R². Gap > 0.3 ⇒ overfit.
3. Re-train with shuffled targets (`y_shuf = y_train[randperm]`). If shuffled R² ≈ real R², features carry no signal.

**Expected signal**: `train_R² - test_R² > 0.3` AND `max(|spearman(features, y)|) < 0.05`.

**Mitigation**:

* Add 3D-aware features (rdMolDescriptors with conformer): `asphericity`, `radius_of_gyration`, `PMI`.
* Add electronic descriptors from PlatinAI: `HOMO`, `LUMO`, `Gap`, `Polarizability`.
* For ASTSR: widen the symbol library (sin, cos, exp, log, sqrt) before claiming ASTSR failure.

---

## §3 Cause 3 — Target normalization reversed (MAJOR)

**Symptom**: Model achieves `train_R² ≫ 0` but `test_R² < 0`, AND the sign of `(y_pred - y_true).mean()` is consistently negative or positive across seeds. Most common when:

* The model is trained on `pic50 = -log10(IC50_M)` but evaluated against raw `IC50_nM` (or vice-versa) — sign-flip alone makes R² negative.
* The target is `1 - pic50` (i.e. `log10(IC50)`) and the model learns `pic50` instead. The two are linearly related but with opposite sign — `R²` is sign-invariant, so this is not a sign issue, but a sign-flip on the loss **is**.
* Mean centering is applied to train but not to test, so `mean(y_test) ≠ mean(y_pred)`.

**Known hits**:

* **Wf-25 Phase 1** — original target was `activity + N(0, 0.05)` (binary), then Wf-37 swapped to real pIC50. The swap from binary `{0, 1}` to continuous `[3.5, 9.5]` changed the variance by 3 orders of magnitude, which alone moves `R²` from `+0.5` (binary separable) to `-0.5` (mean collapse).

**Check method**:

1. Verify `y_pred` and `y_true` are on the same scale. Print `y_train.mean()`, `y_test.mean()`, `y_pred.mean()`.
2. If `(y_pred - y_test).mean() / y_test.std() > 0.5`, the model is **biased**, not unstable.
3. If `np.corrcoef(y_pred, y_test)[0, 1] ≈ -1.0`, the model is learning **the wrong sign**.

**Expected signal**: `sign(corrcoef(y_pred, y_test)) == -1` ⇒ **target sign-flip**.

**Mitigation**:

* Pin the conversion at the loader: `pIC50 = -log10(IC50_M)` consistently.
* Compute `y_train.mean()` once and apply `(y - y_mean)` before training; use the same `y_mean` for test predictions.
* If regression target is bounded `[a, b]`, normalise to `[0, 1]` (not `[-1, 1]`) and inverse at the eval.

---

## §4 Cause 4 — Data leakage / split not strict (MAJOR)

**Symptom**: `train_R² ≈ test_R²` but both are close to 0 or negative. Most common when:

* Same molecule (different salt form / stereochemistry / tautomer) appears in train and test.
* The split is `random` not `scaffold` — but the target is scaffold-dependent (e.g. Bemis-Murcko).
* The split is `temporal` but the test set is downstream of a transformation step (smiles canonicalization, charge neutralization) that train also sees.

**Known hits**:

* **Wf-32 m16** — train on PlatinAI 4-cell cohort (synthetic IC50 from HOMO/LUMO), test on HeLa48h/dark cohort (real measurements). The two cohorts have different scaffolds and different target distributions; the model trains on synthetic noise and is tested on real noise ⇒ R² < 0 by construction.

**Check method**:

1. Verify split is `scaffold_split` (Bemis-Murcko), not `random_split`.
2. Compute train-test scaffold overlap: `len(set(train_smiles_scaffold) ∩ set(test_smiles_scaffold)) / len(test_scaffold)`. Should be 0 for strict scaffold split; < 0.05 for grouped scaffold.
3. Compute target distribution per split: `KL(train_y || test_y)`. If KL > 0.5 nats, the test target is out-of-distribution.

**Expected signal**: scaffold overlap > 0 OR `KL(train_y || test_y) > 0.5`.

**Mitigation**:

* Re-split with strict `scaffold_split` (group by Bemis-Murcko; train / val / test disjoint at scaffold level).
* Re-run on dual cohorts: train on PlatinAI 4-cell + HeLa48h, test on a held-out cohort disjoint by scaffold.
* If scaffolds overlap, remove the overlapping molecules before split.

---

## §5 Cause 5 — Target distribution extreme (MAJOR)

**Symptom**: `R² < 0` AND `y_train.skewness() > 1` OR `y_train.quantile(0.99) / y_train.quantile(0.5) > 10`. The label is so concentrated that the best regressor is the median.

**Known hits**:

* **Wf-37 L5 PlatinAI 4-cell** — synthetic IC50 from `HOMO/LUMO/Gap/Polarizability`. The distribution is bounded `[4, 8]` with a heavy mode near the mean (because the synthetic generator draws from a Gaussian centred at the mid-point). The regressor fits the mean to within a small residual; `R²` becomes `+0.16` rather than `+0.5+` because the upper-tail variance is unexplained.

**Check method**:

1. Compute `y_train.skewness()`, `y_train.kurtosis()`. `|skew| > 1` ⇒ non-Gaussian.
2. Compute `y_train.quantile([0.01, 0.05, 0.5, 0.95, 0.99])`. If the 5th–95th percentile range is < 1.0 (in pIC50 units), the target is concentrated.
3. Train a quantile regressor at q=0.5 (median). If median-MAE < mean-MAE by > 20%, the median is the best predictor.

**Expected signal**: `y_train.skew() > 1` AND `q95 - q5 < 1.0`.

**Mitigation**:

* Apply `log1p(y)` transformation if `y > 0` (IC50 in nM).
* Apply `Yeo-Johnson` power transform (handles negatives) before regression.
* Stratify the training set by `y` bins; oversample the tails.

---

## §6 Cause 6 — Hilbert optimizer step too large → NaN → loss=0 (MAJOR)

**Symptom**: `loss_curve` drops to 0 within first 10 iters and stays at 0. `R²` is undefined or `NaN`. Most common when:

* Inner product in Hilbert space overflows in float32 (Hilbert dim ≥ 128 with `||x||² ~ 1e6`).
* PD step `lr * grad_norm` exceeds `1e3`; the iterate jumps out of the feasible set; subsequent gradient is `0` (saturating tanh / sigmoid / relu) → loss collapses to 0.

**Known hits**:

* **Wf-25 joint Hilbert** — `R² = -0.443` on `PlatinAI 5K + MetalCytoToxDB Ru 500`. Root cause was constant `lr=0.01` on the sparse-RDKit feature scale (`||x|| ~ 400`); the first PD iterate jumped to `||x_k|| ~ 1e6`, the inner-product computation overflowed, the gradient became NaN, and the regressor outputted the mean for the remaining iters.

**Check method**:

1. Inspect `loss_curve[:10]`. If `loss[5:] ≈ 0` AND `loss[:5]` is finite, the model is degenerate.
2. Print `np.isnan(grad).any()` and `np.isinf(grad).any()`. If true, gradient is broken.
3. Inspect `np.linalg.norm(x_k, 2)` over iters. If `||x_k||` grows by `> 100×` in the first 5 iters, the step is too large.

**Expected signal**: `loss_curve[5:] == 0` AND `np.isnan(grad)` somewhere in the trajectory.

**Mitigation**:

* Add gradient clipping: `grad = grad * min(1, 1.0 / (||grad|| + 1e-8))`.
* Use a warmup-cosine LR schedule (Wf-37 fix): `lr_warmup=50, lr_max=0.005, lr_min=0.0005`.
* Pad input features with a bias column (Wf-37 fix) instead of tile-inflate.
* Switch to float64 for the Hilbert inner product (cast back to float32 only for the final prediction).

---

## §7 Cause 7 — Numerical precision: float32 R² noise floor ±0.05 (MINOR)

**Symptom**: R² varies by > 0.1 across seeds, AND the seed-std of R² is comparable to `|R²|`. Most common when:

* Hilbert dim ≥ 64 with float32 matmul accumulates error in the order of `sqrt(64) * eps_fp32 ~ 1e-5` per inner product; over 100 PD iters this becomes `~1e-3` per prediction, which translates to `R² ± 0.05`.
* The label is `pIC50 ∈ [3.5, 9.5]` (range 6); a 0.05 error is `~1%` of range, but the regression target variance is also small.

**Known hits**:

* **Wf-39 Deflexpressor seed 2** — `hilbert R² = -1.4252` (one seed out of 5); `hilbert R² std = 0.6802`. The single negative seed dominates the mean and the std. This is consistent with a `±0.5` R² noise floor at `hilbert_dim=64` in float32.

**Check method**:

1. Compute `R².std()` across 5+ seeds. If `R².std() > 0.5 × |R².mean()|`, the signal is below the noise floor.
2. Re-run the same seed in float64. If `|R²_fp32 - R²_fp64| > 0.1`, the float32 noise floor is dominant.
3. Compute `R²` with a larger Hilbert dim (`hilbert_dim=128`). If `R²(hilbert_dim=128) ≈ R²(hilbert_dim=64)`, the dim is not the bottleneck.

**Expected signal**: `R².std() / |R².mean()| > 0.5`.

**Mitigation**:

* Use float64 for the Hilbert inner product (the hot loop is small enough that the wall cost is acceptable).
* Report `R².mean() ± R².std()` honestly (do not cherry-pick seeds).
* Re-run with `seeds=[0..9]` and require `R².mean() > 0.0` AND `R².mean() > 2 * R².std()` for a positive ship.

---

## §8 Cause 8 — Feature-target correlation truly weak (MINOR)

**Symptom**: All of Causes 1–7 ruled out, yet `R² < 0`. Most common when:

* The model uses molecular descriptors that capture molecular shape (e.g. RDKit 2D) but the target is dominated by metal-coordination geometry (e.g. Pt-N bond angle, trans-effect).
* The model lacks 3D / electronic descriptors that would be necessary for the target (e.g. `pIC50` for cisplatin-class drugs requires Pt oxidation state, ligand trans-effect).
* The dataset is too small (< 100 unique molecules) for the regressor to learn beyond a noisy mean.

**Known hits**:

* **Wf-37 L5** — ASTSR features (12-d RDKit) have weak correlation with Pt binding affinity at scaffold-level split. The best 5-seed mean Hilbert R² is `+0.16`; the upper bound (`+0.5`) requires 3D / electronic descriptors not in the current 12-d set.

**Check method**:

1. Compute the **information-theoretic upper bound** for the dataset: `R²_max = 1 - H(y_test) / log(N_test)`. If `R²_max < 0.2`, the target is not learnable from any feature set.
2. Compute the **ridge baseline** on the same train / test split. If `ridge_R² < 0.0`, the features cannot carry the signal.
3. Compute **per-target Spearman ρ** with the best 3-d feature subset (greedy forward selection). If `max(|ρ|) < 0.05`, the dataset is below the learnability threshold.

**Expected signal**: `ridge_R² < 0` AND `R²_max < 0.2`.

**Mitigation**:

* Augment features with 3D conformer descriptors (after MMFF94s relax).
* Add metal-specific descriptors: `Pt-N_bond_length`, `trans_effect_angle`, `coordination_geometry`.
* If the dataset is too small (< 100), augment with PlatinAI synthetic data **only for training** (never for test).

---

## §9 Cross-cause interactions

These 8 causes are not independent. The known MEASURED cases each have **2–3 overlapping causes**:

| Case | R² | Causes |
|---|---:|---|
| Wf-25 Hilbert joint | -0.443 | (1) baseline collapse + (6) PD step overflow + (2) feature mismatch |
| Wf-32 m16 smoke | -0.040 | (1) baseline collapse + (4) cohort mismatch + (2) feature mismatch |
| Wf-37 L5 Hilbert | +0.163 | (5) target distribution + (8) weak correlation (ship acceptable) |
| Wf-39 composite | +0.186 | (7) float32 noise + (1) baseline collapse on AST branch |

The Phase 2 checklist will run all 8 checks in priority order; if Cause 1 trips, we fix baseline collapse first and re-measure before chasing Cause 6.

---

## §10 Phase 2 plan

1. **Re-measure** Wf-32 m16 in float64 + Wf-37 L5 setup on the same seed. If R² ≥ 0, Cause 7 is the dominant noise.
2. **Inspect** the Wf-32 trainer for target normalization (Cause 3) and cohort split (Cause 4). If found, fix and re-run.
3. **Re-split** PlatinAI 4-cell by strict scaffold (Cause 4 mitigation) and re-train.
4. **If all above fail**: Phase 3 ships Cause 2 / Cause 8 mitigation (3D + electronic features).

**Exit criterion for Phase 2**: at least one Wf-32 m16 cohort hits `Pearson r ≥ 0.0` AND `R² ≥ 0.0` on real spread pIC50 (no synthetic).

---

## §11 Lit anchors

* Sun & Zheng 2026 — *数学学报* 53(4): 821-846 — Hilbert PD O(1/k) bound.
* Chambolle & Pock 2011 — *J. Math. Imaging Vision* 40(1): 120-145 — canonical PD reference.
* Bemis & Murcko 1996 — *J. Med. Chem.* 39(15): 2887-2893 — Bemis-Murcko scaffold split.
* Wu et al. 2018 — *Molecules* 23(6): 1379 — MoleculeNet scaffold split caveats.
* Pedregosa et al. 2011 — *JMLR* 12: 2825-2830 — scikit-learn ridge baseline.

---

## §12 Honest framing

* The 8-cause catalog is **prior** knowledge — every cause has been observed in at least one of the 4 known negative-R² cases; the catalog is not a speculative laundry list.
* Causes are not mutually exclusive; multi-cause interactions are the rule, not the exception.
* The Phase 2 plan re-runs MEASURED cases (Wf-32 m16, Wf-37 L5) rather than running new cases — the goal is **debugging**, not **leaderboard chasing**.
* "R² < 0" is not always a bug — if Cause 8 trips after Causes 1–7 are ruled out, the honest framing is "the features cannot carry the signal; ship the negative R² as a known limitation."

Co-Authored-By: Claude Code <noreply@anthropic.com>