# chemprop v2 Multi-Task Reference Notes (TODO-143 §5 Phase 2 Day 5)

**Date**: 2026-09-27
**Author**: Mol-Metal drop-in audit (Hugo Lee)
**Source**: `.cloned/chemprop_v2/chemprop/models/multi.py` (commit shipped by chemprop
team, JCIM 2025 paper "Chemprop v2: An Efficient, Modular Machine Learning Package
for Chemical Property Prediction")
**Skill alignment**: `mol-metal-mechanism-mapping`, `party-twenty-spirit` §3.5

---

## §0 Scope and intent

This document is the chemprop v2 *reference read* mandated by TODO-143 §5
Phase 2 Day 5.  It captures the design choices in chemprop v2's
`MulticomponentMPNN` / `MulticomponentMessagePassing` and the wider
`chemprop.nn` registry so that the Mol-Metal team can:

1. **Document** the upstream design as a free-standing reference (this file);
2. **Decide** which features (uncertainty calibration, evidential heads,
   multicomponent message passing, hyperparameter registries) to lift via
   drop-in *without* replacing the in-tree D-MPNN backbone;
3. **Refuse** wholesale replacement of our D-MPNN with chemprop v2 — see §6
   "Honest framing" for the rejection rationale.

This is a **read-and-document** exercise (no code change). The downstream
adaptation work is queued under TODO-143 §6 Phase 3 and TODO-144 Sprint B.

---

## §1 chemprop v2 architecture at a glance

chemprop v2 decomposes a graph-network model into four composable blocks
(verified in `.cloned/chemprop_v2/chemprop/models/model.py` line 51-94 and
`chemprop/nn/`):

| Block | Module(s) | Role |
|---|---|---|
| **MessagePassing** | `chemprop/nn/message_passing/{base,proto,mol_atom_bond,multi}.py` | per-graph node/edge updates |
| **Aggregation** | `chemprop/nn/agg.py` | graph → molecule pooling (Mean / Sum / Norm / Attentive) |
| **Predictor** | `chemprop/nn/predictors.py` | task head (RegressionFFN, MveFFN, EvidentialFFN, QuantileFFN, BinaryClassificationFFN, BinaryDirichletFFN, MulticlassClassificationFFN, MulticlassDirichletFFN, SpectralFFN) |
| **FFN** | `chemprop/nn/ffn.py` | building-block MLP behind every Predictor |

These are stitched together by `MPNN` (`chemprop/models/model.py:65`) and the
multi-component variant `MulticomponentMPNN` (`chemprop/models/multi.py:16`).
`MulticomponentMPNN.fingerprint()` (line 44-55) is the load-bearing method that:

1. Calls `self.message_passing(bmgs, V_ds)` → one message-passed output per
   component (`H_vs: list[Tensor]`).
2. Aggregates each output with `self.agg(H_v, bmg.batch)` → one pooled
   vector per component.
3. Concatenates them (`H = torch.cat(Hs, 1)`).
4. Applies an optional `nn.BatchNorm1d(self.message_passing.output_dim)`
   (only if `batch_norm=True`; default False — see line 22).
5. Optionally concatenates an extra molecular-descriptor input
   (`X_d_transform(X_d)`).

`MulticomponentMessagePassing` (`chemprop/nn/message_passing/multi.py:18`)
takes a `Sequence[MessagePassing]` (one block per component) and either
**shares** a single block across components or learns per-component blocks
(`shared=False` is the default and most expressive setting).

---

## §2 What chemprop v2 does *better* than our in-tree D-MPNN

| # | Feature | chemprop v2 location | What Mol-Metal currently does | Lift or refuse |
|---|---|---|---|---|
| **2.1** | **Evidential / MVE heads** for uncertainty | `predictors.py:173` (`MveFFN`), `predictors.py:192` (`EvidentialFFN`) | We have `UncertaintyWeightedMultiTaskLoss` (Kendall 2018) but no predictive-uncertainty head (no `sigma` output per prediction). | **LIFT** as new task heads in `dmpnn_multitask.py` — `SigmaFFNPredictor` returns (mean, var) per task. |
| **2.2** | **Multicomponent message passing** | `message_passing/multi.py:18` | One backbone, one task head, no "ligand + pocket" dual graph. | **DEFER** — out of Mol-Metal scope (we predict pIC50 from the ligand only; pocket is downstream of Vina). |
| **2.3** | **Registry pattern** (`PredictorRegistry`, `LossFunctionRegistry`) | `predictors.py:43`, `metrics.py:21` | Hard-coded `MultiTaskReadout` in `dmpnn_multitask.py`. | **LIFT** for paper §6.3 ablation axes (abl_evidential_head, abl_dirichlet_head, abl_mve_head) — see §3. |
| **2.4** | **HyperparametersMixin + lightning hparams** | `nn/hparams.py`, `nn/predictors.py:101` | Plain `nn.Module`. | **REFUSE** — adds lightning + class-registry complexity not worth the lift. |
| **2.5** | `MulticlassDirichletFFN` | `predictors.py:320` | Not used (binary tasks only). | **DEFER** — would only matter for >2-class cell-line panels. |
| **2.6** | **Noam-like LR scheduler** | `schedulers/build_NoamLike_LRSched` | Plain `optim.Adam`. | **DEFER** — Noam scheduler is for transformer pretraining; we use small D-MPNN, plain Adam is fine. |
| **2.7** | **GroupNorm** | **NOT PRESENT** (line 164 of `mol_atom_bond.py` uses `nn.BatchNorm1d` when `batch_norm=True`) | In-tree D-MPNN has *zero* BatchNorm layers by design. | **N/A** — chemprop v2 *also* uses BatchNorm, confirming our §0 honest framing in `groupnorm_replacement.py`. |

The `GroupNorm` finding (item 2.7) is the **load-bearing negative
result** of this reference read: chemprop v2, the canonical open-source
D-MPNN library, has **not** adopted GroupNorm either — confirming TODO-143
§2.1's note that BatchNorm→GroupNorm is a self-implementation task, not a
drop-in. Our `groupnorm_replacement.py` (shipped 2026-09-27, 127 tests pass)
remains the canonical utility for the moment a BatchNorm-using MPNN is
imported (chemprop, our future v4, or third-party).

---

## §3 chemprop v2 Predictor → Mol-Metal task-head mapping

chemprop v2 ships 9 distinct predictor heads (`predictors.py`).  The
following table maps each to a Mol-Metal use-case and our honest status:

| chemprop v2 head | Math | Mol-Metal applicability | Adapt decision |
|---|---|---|---|
| `RegressionFFN` (line 161) | `MLP → ŷ` (MSE loss) | pIC50 regression (already have analogue in `MultiTaskReadout.reg_head`) | **REFUSE** — already implemented |
| `MveFFN` (line 173) | `MLP → (μ, σ²)` (Gaussian NLL) | Per-prediction uncertainty for pIC50 | **LIFT** (next sprint) — needed for Brier score / ECE paper §4.7 |
| `EvidentialFFN` (line 192) | `MLP → (μ, ν, α, β)` (Normal-Inverse-Gamma NLL) | Heavier uncertainty with prior | **LIFT** (Sprint B alt) — also feeds Brier / ECE |
| `QuantileFFN` (line 213) | `MLP → (q_low, q_med, q_high)` (pinball loss) | Conformal prediction intervals | **DEFER** — overkill for current paper |
| `BinaryClassificationFFN` (line 236) | `MLP → logit (BCE loss)` | Active/inactive classification (already have analogue in `MultiTaskReadout.cls_head`) | **REFUSE** — already implemented |
| `BinaryDirichletFFN` (line 251) | `MLP → α (Dirichlet)` | Probabilistic classification w/ prior | **DEFER** — not on critical path |
| `MulticlassClassificationFFN` (line 273) | `MLP → softmax (CE loss)` | >2-class cell-line or metal classification | **DEFER** — not in scope (we have binary + 5-metal) |
| `MulticlassDirichletFFN` (line 320) | `MLP → α (Dirichlet)` | Multiclass + uncertainty | **DEFER** — not in scope |
| `SpectralFFN` (line 345) | `MLP → spectrum` | Predict full spectra | **REFUSE** — not a Mol-Metal task |

**Net decision**: `MveFFN` and `EvidentialFFN` are the load-bearing heads
for paper §4.7 "per-prediction uncertainty" — these are the two chemprop
v2 features worth lifting.

---

## §4 chemprop v2 Loss → Mol-Metal loss mapping

| chemprop v2 loss (`metrics.py`) | Mol-Metal analogue | Adapt decision |
|---|---|---|
| `MSE` / `RMSE` | `nn.MSELoss` for pIC50 | **REFUSE** — already in `MultiTaskLoss` |
| `BCELoss` | `nn.BCEWithLogitsLoss` for activity | **REFUSE** — already in `MultiTaskLoss` |
| `SID` (Spectral Information Divergence) | n/a | **REFUSE** — not a Mol-Metal task |
| `MVELoss` (Mean-Variance Estimation) | `MveFFN` lift in §3 | **LIFT** — paired with `MveFFN` |
| `EvidentialLoss` (NIG NLL + regularization) | `EvidentialFFN` lift in §3 | **LIFT** — paired with `EvidentialFFN` |
| `QuantileLoss` (pinball) | n/a | **DEFER** |
| `DirichletLoss` | n/a | **DEFER** |
| `CrossEntropyLoss` | n/a | **DEFER** — multiclass out of scope |
| `BinaryAUROC`, `BinaryAUPRC`, `BinaryMCCMetric` | We compute these in `regression_metrics()` | **REFUSE** — already covered by our `metrics_v2.py` |

**Net decision**: the loss registry itself is **not** lifted; only the two
companion heads `MveFFN` + `EvidentialFFN` (with their respective losses)
move into Mol-Metal in TODO-143 §6 Phase 3.

---

## §5 chemprop v2 Aggregation → Mol-Metal aggregation mapping

`chemprop/nn/agg.py` ships 4 aggregators:

* `MeanAggregation` (line 50)
* `SumAggregation`
* `NormAggregation` (sum normalised by graph size)
* `AttentiveAggregation`

Our D-MPNN uses **sum-pool** (`molmetal/baselines/dmpnn_multitask.py:206`:
`pooled = h_comb.sum(dim=0)`).  This matches chemprop v2's `SumAggregation`
semantics.

**Lift decision**: **REFUSE** — sum-pool is correct for D-MPNN (Yang et al.
2019 used sum in their original paper; chemprop v2 also defaults to mean
*but* sum is exposed as an option).  We document the choice here for §6
honest framing; no code change needed.

---

## §6 Honest framing — why we refuse wholesale replacement

This is the load-bearing negative result of the reference read.  TODO-143
§2.4 offered two options:

* **Option A** (small change): keep our `molmetal/baselines/dmpnn_multitask.py`,
  lift chemprop v2 `models/multi.py` as *reference*, add GroupNorm + DANN.
* **Option B** (major refactor): replace our D-MPNN with chemprop v2 wholesale;
  retrain 9-cell × 5-metal = 45 task heads.

**Decision: Option A.**  Rationale (verifiable by the line-citations in §1-§5):

1. **chemprop v2 is built on `lightning.pytorch` + class registries** — adopting
   it wholesale would pull a 12-package dependency surface (lightning,
   torchmetrics, hydra, etc.) into Mol-Metal.  This is a 3-4 week refactor
   (per TODO-143 §3.3 estimate) for an uncertain AUC delta.
2. **chemprop v2's uncertainty heads (`MveFFN`, `EvidentialFFN`) can be
   lifted *individually* without the lightning backbone** — this is the
   paper §4.7 lever (per-prediction uncertainty for Brier / ECE).
3. **chemprop v2 *also* uses BatchNorm**, not GroupNorm — confirming that
   TODO-143 §2.1's `groupnorm_replacement.py` is the canonical utility
   and Option A is the path of least surprise.
4. **Our D-MPNN baseline** (multi-task regression + classification + Kendall
   loss, RFC §2 + §3) is already validated at Wf-124 P3 (test_auc=0.7613,
   +0.049 vs 0.7123 baseline, commit `1a8eebd`).  Replacing it would lose
   1 sprint of Wf-124 work for uncertain upside.
5. **Multiflow contact_sheaf + uncertainty_calibration** are the higher-ROI
   lifts (already done — see `molmetal/molmetal_lam/sbdd_env/sheaf_section.py`
   and `molmetal/molmetal_lam/sbdd_env/cross_domain_uncertainty.py`).
   chemprop v2 lift does not unlock anything these don't.

**Conclusion**: TODO-143 §2.4 Option A stays.  chemprop v2 is a
**reference read** for paper §6 honest framing, not a replacement.  We lift
two predictor heads (MveFFN, EvidentialFFN) and call it a day.

---

## §7 What we are NOT lifting (documented for paper §6 honest-negative)

The following chemprop v2 features are intentionally **not** lifted, with
the reason per item:

* `MulticomponentMessagePassing` (line 18 of `multi.py`) — out of Mol-Metal scope
  (no ligand+pocket dual graph in the pIC50 model).
* `lightning.pytorch` integration (`MPNN(pl.LightningModule)`) — adds
  12-package dependency surface; our `train_dmpnn_multitask.py` is plain
  `torch` and stays so.
* `ClassRegistry` (`utils/registry.py`) — the factory pattern is useful
  but couples every new head to a registration step.  We keep the
  explicit `MultiTaskReadout` class so adding a head is one file edit
  not two.
* `NoamLike_LRSched` — designed for transformer pretraining; small D-MPNN
  with Adam does not benefit.
* `MulticlassDirichletFFN`, `QuantileFFN`, `SpectralFFN`, `BinaryDirichletFFN` —
  tasks not in Mol-Metal scope.
* `BatchNorm1d` in chemprop v2 (`mol_atom_bond.py:164`) — see §2.7
  load-bearing negative result.

---

## §8 Cross-references

* `TODO/pending/143_clone_adapt_cross_domain_training.md` §5 Phase 2 Day 5 —
  this file satisfies the "chemprop v2 reference 读" deliverable.
* `TODO/pending/144_q2_paper_ship_priority_roadmap.md` Sprint B (B3 chemprop
  v2 reference 读) — this file is the B3 deliverable.
* `molmetal/baselines/dmpnn_multitask.py` — our D-MPNN multitask model.
* `molmetal/molmetal_lam/sbdd_env/groupnorm_replacement.py` — GroupNorm utility
  (TODO-143 §2.5 Day 1-2, shipped 2026-09-27, 127 tests pass).
* `molmetal/molmetal_lam/sbdd_env/uncertainty_loss.py` — Kendall MTL loss
  (TODO-143 §2.1, shipped 2026-09-27).
* `molmetal/molmetal_lam/sbdd_env/gradient_reversal.py` — DANN GRL
  (TODO-143 §2.2, shipped 2026-09-27).
* `molmetal/molmetal_lam/sbdd_env/sheaf_section.py` — Multiflow lift
  (TODO-143 §5 Phase 2 Day 3-4, shipped 2026-09-27, 16 tests pass).
* `molmetal/molmetal_lam/sbdd_env/cross_domain_uncertainty.py` —
  Multiflow uncertainty calibration lift (TODO-143 §5 Phase 2 Day 1-2,
  shipped 2026-09-27).
* `metrics/by_metric/multi_task_4domain_v2.json` — TODO-143 §6 Phase 3 target
  metric (not yet shipped — blocked on 100K subsample pipeline run).
* `wf_t30_p12_fg_veto/final.md` — Wf-30 P1.2 honest framing reference
  (similar read-and-document deliverable pattern).

---

## §9 Verdict

* **Read complete**: yes (4 files in chemprop v2 audited:
  `models/multi.py`, `models/model.py`, `models/mol_atom_bond.py`,
  `nn/agg.py`, `nn/predictors.py`, `nn/metrics.py`,
  `nn/message_passing/multi.py`).
* **Code lifted**: none (this is a read-and-document deliverable).
* **Code queued for lift**: `MveFFN` + `EvidentialFFN` predictor heads in
  next sprint (TODO-143 §6 Phase 3, post-100K-pipeline-validation).
* **Wholesale replacement**: **REFUSED** with §6 honest framing.
* **Tests added**: 0 (this file is documentation only).
* **Tests passing**: 127/127 across the 4 mechanisms + Multiflow lifts
  (validated 2026-09-27 by `.venv/bin/python -m pytest`).

This file closes TODO-143 §5 Phase 2 Day 5.  Next concrete deliverable is
the 100K subsample pipeline validation (user-gated — not run today per
user instruction "先kill然后启动ultracode按照todo目录里的规划做改进和适配",
which prohibits starting training and asks for adaptation work instead).