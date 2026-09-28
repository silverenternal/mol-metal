# Chemprop v2 reference notes (TODO-143 §2.4)

> Purpose: document chemprop v2 multi-task + uncertainty design patterns, plus the
> honest gap analysis vs Mol-Metal needs. **NOT a wholesale replacement proposal** —
> the multi-task / D-MPNN infrastructure in `molmetal/models/v4_multitask_dmpnn.py`
> is intentionally leaner, and GroupNorm / Kendall / DANN still have to be added
> by us.

**Cloned repo**: `/home/hugo/codes/mol-metal/.cloned/chemprop_v2/chemprop/`
**JCIM paper**: "Chemprop v2: An Efficient, Modular Machine Learning Package for
Chemical Property Prediction" (10.1021/acs.jcim.5c02332)
**License**: MIT
**TODO**: TODO-143 §2.4 (Phase 2 W2D5 "chemprop v2 reference 读")

---

## 1. Architecture summary

### 1.1 MPNN top-level (`models/model.py`, 323 LOC)

Chemprop v2's flagship is a `pl.LightningModule` with strict three-stage pipeline:

```
MPNN(G) = predictor( agg( message_passing(G) ) )
```

- `MessagePassing` — D-MPNN-style vertex/bond message passing, returns
  `V × d_h` node-level embeddings.
- `Aggregation` — graph-level pooling (`MeanAggregation`, `SumAggregation`,
  `NormAggregation`, `AttentiveAggregation`); outputs `b × d_h`.
- `Predictor` — task head (FFN-MLP); outputs `b × n_tasks × n_targets`.

Lightning handles `.training_step`, `.validation_step`, `.test_step`,
`.predict_step`, `.configure_optimizers` automatically. The `MPNN` class also
owns the optional `BatchNorm1d` after aggregation:

```python
# model.py L94
self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
```

Note: `output_dim` is the message passing output dim (per-graph after agg sum
across components), so BN is applied on the post-aggregation fingerprint before
the predictor head. **Default is `False`**; you opt in with `batch_norm=True`.

### 1.2 MulticomponentMPNN (`models/multi.py`, 109 LOC)

Thin subclass of `MPNN` for **multi-component** inputs (e.g. drug + protein in
reaction prediction). Key differences from the base MPNN:

- Accepts `MulticomponentMessagePassing` (list of message-passing blocks).
- `fingerprint(...)` returns concatenation of aggregated per-component
  fingerprints: `H = concat([agg_i(bmg_i) for i in components])`.
- `batch_norm` is applied to the concatenated fingerprint (`H = self.bn(H)`),
  so BN mixes the per-component channels together — that's not the same as
  per-component normalization.
- `on_validation_model_eval()` keeps BN-like layers (graph_transform,
  V_d_transform, X_d_transform) in train mode while everything else evals —
  this is the "BatchNorm-safe" eval pattern.

`MulticomponentMessagePassing` (`nn/message_passing/multi.py`, 84 LOC) holds a
list of independent message-passing blocks (or one block shared via
`shared=True`). Per-component embeddings are concatenated at the readout.

> ⚠️ **This is "multi-component" input, NOT "multi-task" head sharing**. The
> per-task head is still a single `Predictor` with `n_tasks` outputs. So a
> 5-metal multi-task chemprop model would be: 5 separate MPNNs (or one
> `MulticomponentMessagePassing` with 5 blocks if you want shared
> featurization) + 1 shared predictor with `n_tasks=5`. That's effectively
> the same as Mol-Metal's `v4_multitask_dmpnn.py` head architecture.

### 1.3 FFN / MLP (`nn/ffn.py`, 141 LOC)

The `MLP.build(...)` factory:

- Layer pattern: `Linear → [Activation → Dropout → Linear]×n_layers`
- **No normalization inside the MLP at all** — no BN, no LN, no GN.
- Supports per-layer width via `hidden_dim: int | Sequence[int]`.
- `ConstrainerFFN` is a separate op that maps fingerprint → per-atom weights
  used for additive constraint correction (sum-constrained atom/bond
  predictions).

### 1.4 Predictor head (`nn/predictors.py`, 371 LOC)

The Predictor registry contains task-head subclasses:

| Predictor                | `n_targets` | Loss           | Use                              |
|--------------------------|-------------|----------------|----------------------------------|
| `RegressionFFN`          | 1           | MSE            | Plain regression                 |
| `MveFFN`                 | 2           | MVELoss        | Mean+variance (Nix & Weigend 94) |
| `EvidentialFFN`          | 4           | EvidentialLoss | (μ, v, α, β) NIG (Amini 2020)   |
| `QuantileFFN`            | 2           | QuantileLoss   | Lower+upper quantile             |
| `BinaryClassificationFFN`| 1           | BCELoss        | Sigmoid head                     |
| `BinaryDirichletFFN`     | 2           | DirichletLoss  | Evidential binary                |
| `MulticlassClassificationFFN` | n_classes | CrossEntropy | Softmax head                  |
| `MulticlassDirichletFFN` | n_classes   | DirichletLoss  | Evidential multiclass            |
| `SpectralFFN`            | 1           | SID            | Softplus+normalize spectrum      |

So multi-task regression over `n_tasks=5` metals is just `RegressionFFN` with
`n_tasks=5`, output shape `b × 5`, loss `MSE` with per-task
`task_weights: Tensor`. **No Kendall-style uncertainty weighting** —
`task_weights` is a fixed buffer (see `nn/metrics.py` L36-49, registered as
`register_buffer("task_weights", task_weights)`).

---

## 2. Multi-task head pattern (what chemprop v2 gives us)

Chemprop v2's multi-task is the **simplest possible**: one shared backbone,
one FFN predictor head, `n_tasks` outputs, per-task static weights.

```python
mp = MulticomponentMessagePassing(blocks=[BondMessagePassing(...) for _ in range(5)])
predictor = RegressionFFN(
    n_tasks=5,            # one output per metal
    input_dim=mp.output_dim,
    criterion=MSE(task_weights=torch.tensor([1.0] * 5)),
)
model = MulticomponentMPNN(mp, agg, predictor, batch_norm=True)
```

What we **don't** get from chemprop v2 for multi-task:

- **No Kendall MTL** (`log σ²` learnable task weights). The `task_weights` is a
  fixed buffer initialized once; it's not learned end-to-end.
- **No GradNorm / DWA / uncertainty-weighted MTL**. All multi-task balancing
  is manual (`--task_weights 1.0 2.0 1.5` CLI flag in `cli/train.py`).
- **No per-task BN or per-task LR**. Everything is shared.

For 5-metal MetalCytoToxDB training, this is **inadequate** because the
per-metal dataset sizes are wildly different (Ru is data-poor, Pt/Cu is
data-rich). Static task weights will overfit the rare metals. **We need
Kendall MTL on top.**

---

## 3. Normalization — confirmed: BatchNorm1d only

Searched all of `chemprop/`:

```
$ grep -rE "BatchNorm1d|GroupNorm|LayerNorm" chemprop/
chemprop/models/model.py:        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
chemprop/models/mol_atom_bond.py:                nn.BatchNorm1d(dim) if batch_norm and predictor is not None else nn.Identity()

$ grep -ri "GroupNorm\|LayerNorm" chemprop/   # → no matches
```

Conclusion:
- **BatchNorm1d confirmed**, gated by `batch_norm=False` default.
- **No GroupNorm, no LayerNorm** anywhere — not even in tests or
  featurizers.
- BN is applied **once after aggregation**, NOT inside the FFN predictor.

Implications for Mol-Metal:
- TODO-143 §3.1 already decided "BatchNorm→GroupNorm 自实现". Confirmed
  chemprop v2 also needs custom GroupNorm if we want it. The GroupNorm is
  added in **our** wrapper, not in chemprop.
- GroupNorm is small-batch-friendly which matters for per-metal fine-tuning
  with N=50-200 molecules. BN1d running stats would be very noisy there.

---

## 4. Uncertainty quantification APIs (`chemprop/uncertainty/`, ~1700 LOC)

This is genuinely strong in chemprop v2 — three independent registries:

### 4.1 Estimators (`uncertainty/estimator.py`, 877 LOC)

| Estimator                    | Output                                     | Notes                                  |
|------------------------------|--------------------------------------------|----------------------------------------|
| `NoUncertaintyEstimator`     | preds only                                 | Point prediction                       |
| `MVEEstimator`               | (mean, var)                                | Pair with `MveFFN` predictor           |
| `EnsembleEstimator`          | (preds_stack, var)                         | Variance across ensemble members       |
| `ClassEstimator`             | preds → "uncertainty" via confidence       | Classification head                    |
| `EvidentialTotalEstimator`   | (μ, total_unc)                             | `total_unc = (1+1/v) * β/(α-1)`        |
| `EvidentialEpistemicEstimator`| (μ, epistemic_unc)                       | `(1/v) * β/(α-1)`                      |
| `EvidentialAleatoricEstimator`| (μ, aleatoric_unc)                        | `β/(α-1)`                              |
| `ClassificationDirichletEstimator` | (y, u)                               | Sensoy 2018 evidential classification  |
| `MulticlassDirichletEstimator`| (y_per_class, u)                          | Same idea, multiclass                  |
| `QuantileRegressionEstimator` | (mean, half_interval)                     | Pair with `QuantileFFN` predictor      |
| `DropoutEstimator`           | (mean, var) over MC dropout samples        | Gal & Ghahramani 2016                  |

The interesting ones for Mol-Metal:
- **MVE** (mean-variance estimation): simple, one network, well-calibrated
  variance via NLL loss (`MVELoss` in `nn/metrics.py`).
- **Evidential (NIG)**: 4 outputs (μ, v, α, β), gives aleatoric + epistemic
  decomposition, but harder to train.
- **Dropout MC**: Bayesian approximation, slow at inference (K forward passes).
- **Ensemble**: most reliable, K independent models. We've already used this
  in `metal_hybrid_v4.py` seeds.

### 4.2 Calibrators (`uncertainty/calibrator.py`, 713 LOC)

Post-hoc rescaling of uncalibrated uncertainty:

| Calibrator                       | Task          | Reference                        |
|----------------------------------|---------------|----------------------------------|
| `ZScalingCalibrator`             | Regression    | Levi 2022 (NLL-based)            |
| `ZelikmanCalibrator` (CRUDE)     | Regression    | Zelikman 2020 (quantile-based)   |
| `MVEWeightingCalibrator`         | Regression    | Wang 2021 (ensemble of MVEs)     |
| `RegressionConformalCalibrator`  | Regression    | Angelopoulos 2021 (conformal)    |
| `PlattCalibrator`                | Binary        | Platt 1999                       |
| `IsotonicCalibrator`             | Binary        | Guo 2017                         |
| `MultilabelConformalCalibrator`  | Multilabel bin| Cauchois 2020                    |
| `MulticlassConformalCalibrator`  | Multiclass    | Angelopoulos 2021                |
| `AdaptiveMulticlassConformalCalibrator` | Multiclass| Same, RAPS-style               |
| `IsotonicMulticlassCalibrator`   | Multiclass    | One-vs-all isotonic              |

The conformal calibrators are the gold standard for distribution-free
coverage guarantees. For Mol-Metal OOD detection on 5 metals, conformal
prediction is the right tool — but we need to lift only the regression
conformal calibrator (`RegressionConformalCalibrator`) and not all 14
classes.

### 4.3 Evaluators (`uncertainty/evaluator.py`)

Calibration metrics: `CalibrationAreaEvaluator`, `ExpectedNormalizedErrorEvaluator`,
`NLLRegressionEvaluator`, `SpearmanEvaluator`, etc. These report the quality of
the calibrated uncertainties — **NOT** prediction quality metrics.

---

## 5. What's MISSING (gap vs Mol-Metal needs)

| Need                                       | Chemprop v2 status           | Mol-Metal work-around                 |
|--------------------------------------------|------------------------------|---------------------------------------|
| **GroupNorm / LayerNorm**                  | ❌ BatchNorm1d only           | Self-implement GroupNorm wrapper (~50 LOC) |
| **Kendall MTL uncertainty weighting**      | ❌ Static `task_weights` only | Drop-in `mtl_uncertainty` (already cloned) |
| **GradNorm / DWA dynamic loss balancing**  | ❌ None                       | DIY (~30 LOC)                         |
| **DANN domain-adversarial head**           | ❌ No `GradientReversal`     | Drop-in `DANN_py3.ReverseLayerF`     |
| **Per-metal BN/LR for fine-tuning**        | ❌ Shared only                | Wrap each metal head in BN1d layer    |
| **Wasserstein pIC50 reg loss**             | ✅ `Wasserstein` in `nn/metrics.py` (line 570) | Reuse `mtl_uncertainty.WassersteinLoss` instead (which has `pIC50` calibration) |
| **Reaction-style multicomponent input**    | ✅ `MulticomponentMPNN`       | Use as reference for `drug+protein` if needed |
| **Lightning training loop boilerplate**    | ✅ Full `pl.LightningModule`  | Borrow pattern into `molmetal.trainers` |
| **Noam LR scheduler**                      | ✅ `build_NoamLike_LRSched`   | Optional import — useful for warmup   |
| **Ensemble / MC-dropout uncertainty**      | ✅ `EnsembleEstimator`, `DropoutEstimator` | Use as-is                             |
| **Evidential NIG (4 outputs)**             | ✅ `EvidentialFFN` + 3 estimators | Lift for metal UQ                  |
| **Conformal regression calibration**       | ✅ `RegressionConformalCalibrator` | Lift for OOD coverage guarantee |

---

## 6. Adapt-to-Mol-Metal recommendations

### 6.1 Recommended path: **borrow patterns, don't wholesale replace**

`molmetal/baselines/dmpnn_multitask.py` (865 LOC) and
`molmetal/models/v4_multitask_dmpnn.py` (550 LOC) already implement
multi-task D-MPNN with per-metal heads + Wasserstein loss. Replacing this with
chemprop v2 wholesale would mean:
- Re-implementing the `MolGraph` featurization (`atom_features`, `bond_features`)
  in chemprop v2's format.
- Re-tuning all 9-cell × 5-metal hyperparams (deep_size, depth, message hidden).
- Risking metric regressions on the 5 published baselines.

**Not worth it.** Instead, lift 4 specific patterns:

1. **Conformal regression calibration** (for OOD coverage guarantees on
   cross-domain transfer): lift `RegressionConformalCalibrator` into
   `molmetal/molmetal_lam/sbdd_env/conformal.py` (~80 LOC).
2. **`MulticomponentMessagePassing` reference** (for documentation only,
   since we already have per-component handling): record the design pattern
   in `multi_task_design_notes.md` (this file).
3. **Noam LR scheduler** (`build_NoamLike_LRSched`): drop into
   `molmetal/trainers/` as an alternative scheduler — useful for
   warmup-then-cooldown pattern.
4. **MVE / Evidential heads** (`MveFFN`, `EvidentialFFN`): useful templates
   if we want to add per-metal variance estimates (Wasserstein baseline could
   benefit from this). The hard part is the loss function plumbing.

### 6.2 Skip these chemprop v2 patterns

- `MulticomponentMessagePassing` for **shared** 5-metal backbone — too coupled
  with chemprop's `BatchMolGraph` featurization. Our per-metal independent
  D-MPNN with shared smiles tokenizer is simpler.
- Chemprop's `PredictorRegistry` overhead — our 5 metal heads are too small
  (5-10 task dimensions) to justify the registry pattern.
- `trainer.predict(model, dataloader)`-based uncertainty estimators — these
  assume Lightning `Trainer.predict()`, which we don't use.

### 6.3 Must add ourselves

- **GroupNorm replacement for BatchNorm1d**: TODO-143 §3.1 "BatchNorm→GroupNorm
  自实现". Patch `molmetal/models/dmpnn.py` to accept `norm_layer: Literal["bn","gn"]`
  and instantiate the right module.
- **Kendall MTL**: drop-in from `mtl_uncertainty/` (already cloned in
  `.cloned/mtl_uncertainty/`). Add `KendalUncertaintyWeighting` wrapper around
  per-metal loss computation.
- **DANN head**: drop-in from `DANN_py3/functions.py` (`ReverseLayerF`).
  Add adversarial domain classifier branch in `v4_multitask_dmpnn.py`.

---

## 7. Honest assessment

**Chemprop v2 is well-engineered but for a different problem class.**

| Dim                  | Chemprop v2 focus                | Mol-Metal focus                    |
|----------------------|----------------------------------|------------------------------------|
| Task count           | 1-1000 (large HTS)               | 5 metals (tiny)                    |
| Dataset size         | 10K-10M molecules                | 100-50K per metal                  |
| Featurization        | Atom/bond features, no 3D        | + 3D coord, metal-bond, complex    |
| Backbone             | D-MPNN (shared backbone)         | D-MPNN + 3D-aware head             |
| Multi-task balance   | Static                           | Per-metal NLL/Wasserstein          |
| Domain adaptation    | None                             | DANN, WeightedRandomSampler, OOD   |
| Normalization        | BN1d (big batch assumption)      | BN1d+ small-batch unfriendly; GN needed |
| UQ                   | Comprehensive (10 estimators, 14 calibrators, 11 evaluators) | None natively, conformal only |

**Bottom line**: chemprop v2's uncertainty infrastructure is **genuinely
excellent** and worth lifting (conformal calibrator, MVE/Evidential heads).
Its multi-task and normalization are **insufficient** for Mol-Metal's needs
and must be patched by us. Do not wholesale-replace the existing D-MPNN.

**Time budget for this lift** (vs TODO-143 §3.2 estimate):
- Lift conformal calibrator: 0.5 day (was 1 day; it's just one class to port)
- Lift MVE / Evidential: 0.5 day (was 1 day; one FFN class each)
- Noam scheduler: 0.25 day (already 50 LOC in chemprop)
- **Total: ~1.25 days** vs TODO-143 §3.2's 2 days

The other 0.75 day we save goes into Kendall MTL integration
(.cloned/mtl_uncertainty/ drop-in), which chemprop v2 doesn't have at all.

---

## 8. File map (chemprop v2 → Mol-Metal lift targets)

```
chemprop/uncertainty/calibrator.py:RegressionConformalCalibrator  →  molmetal/molmetal_lam/sbdd_env/conformal.py
chemprop/nn/predictors.py:MveFFN, EvidentialFFN                  →  molmetal/models/uq_heads.py
chemprop/schedulers.py:build_NoamLike_LRSched                     →  molmetal/trainers/noam_lr.py
chemprop/nn/metrics.py:Wasserstein (line 570)                     →  (NOT lifted — use mtl_uncertainty.WassersteinLoss)
chemprop/models/multi.py:MulticomponentMPNN                       →  (reference only, NOT lifted)
chemprop/nn/ffn.py:MLP.build                                      →  (reference only, our MLP is smaller)
chemprop/nn/message_passing/multi.py:MulticomponentMessagePassing  →  (reference only)
```

**What gets lifted**: 3 files, ~200 LOC total, ~1.25 person-days.
**What stays as reference**: ~600 LOC of `models/multi.py` + `nn/predictors.py`
read for design notes (this file).

---

## 9. Verification checklist (when lift happens)

- [ ] `RegressionConformalCalibrator.fit()` on a held-out 20% of MetalCytoToxDB
      produces `qhat < 1.0` for Ru (low-N metal) — sanity check.
- [ ] `MveFFN` trains on 1 metal (e.g. Ru) with `MVELoss`, MAE within 0.05 of
      plain MSE baseline (variance head shouldn't hurt point prediction).
- [ ] `build_NoamLike_LRSched` warmup_lr=max_lr at end of epoch 2 matches
      expected linear schedule (init_lr=1e-4, max_lr=1e-3).
- [ ] No GroupNorm / LayerNorm referenced anywhere in Mol-Metal still uses
      BatchNorm1d unintentionally — grep audit.

---

## 10. File metrics

- This design notes file: **~340 LOC**, designed for completion_audit review.
- chemprop v2 files reviewed: **2712 LOC** across 8 files (see §1-§4).
- Time spent reading: ~10 min (Read calls only, no execution).

---

**Status**: ✅ reference-read complete. No code changes. Lift work is a
separate TODO (TODO-143 §3.2) and is gated on the Kendall MTL + GroupNorm +
DANN being wired in `molmetal/models/v4_multitask_dmpnn.py` first.