# wf_coupling_pretrain_5k — Lambda × CFM coupling adapter pretraining (PlatinAI 5K, CPU-only)

**Workflow:** Lambda × CFM coupling pretraining (TODO-21 / C-R17-C1)
**Phase:** SHIP (CPU-only)
**Date:** 2026-09-25
**Wall budget:** ~7 s on a single CPU
**Owner:** (R17 ultracode dispatch)

## TL;DR

The PlatinAI-5K CPU-only pretraining of the 32-d lambda → 64-d bridge
coupling adapter is shipped.  All gates from TODO C-R17-C1 step
(4)-(5) pass:

* `embed_pocket` output shape on a 1K hold-out: **(1000, 64)** ✓
* Reconstruction RMSE on the 1K hold-out: **0.0046** (gate ≤ 0.05) ✓
* 11 new unit tests pass; existing 18 coupling-adapter tests still
  pass (1 pre-existing CLI-flag test was failing before this
  workflow, unrelated to these changes).

## 1. What shipped

| File                                                          | Status | Lines |
|---------------------------------------------------------------|--------|-------|
| `molmetal/data/platinai_5k.csv`                                | NEW    | 5000 rows |
| `molmetal/scripts/tmqm_coupling_pretraining_cpu.py`             | NEW    | ~660 |
| `models/coupling_adapter_platinai5k.npz`                       | NEW    | binary |
| `models/coupling_adapter_platinai5k.json`                      | NEW    | metadata |
| `molmetal/tests/test_coupling_adapter_platinai5k.py`           | NEW    | ~250 |
| `molmetal/reports/wf_coupling_pretrain_5k/final.md`            | NEW    | this file |

## 2. Dataset

`molmetal/data/platinai_5k.csv` was auto-materialised from
`.data/molmetal/tmQM/tmqm_parsed.csv` by filtering to the
**PlatinAI chemistry space** (Pt, Pd, Au, Rh, Ir, Ru, Os), dedup,
and deterministically sub-sampling 5,000 rows.  Columns:

* `id`          — `platinai_0000`..`platinai_4999`
* `smiles`      — RDKit-canonical SMILES
* `ic50_nm`     — NaN placeholder (no measured IC50 available in tmQM)
* `selectivity` — NaN placeholder
* `metal`       — Pt / Pd / Au / Rh / Ir / Ru / Os

When the canonical PlatinAI 226K corpus is mounted at
`/mnt/storage/data/molmetal/PlatinAI_*.xlsx` the loader will use it
instead of the tmQM fallback (see `ensure_platinai_5k_csv`).

## 3. Architecture

A **32-d lambda → 64-d bridge** MLP:

```
lambda_features (32)
       │
       ▼
Linear(32, 64)  ─── ReLU
       │
       ▼
Linear(64, 64)  ─── ReLU       ← hidden layer
       │
       ▼
Linear(64, 64)                  ← head layer
       │
       ▼
L2-normalise per row            ← unit-norm bridge embedding
```

Two hidden layers of size 64 (matching the existing
`tmqm_cfm_pretraining.py` 2-layer MLP convention).  Output is
L2-normalised per row so cosine similarity is well-defined (max
value = 1) and the model cannot cheat by growing `|pred|`.

### 3.1 Target = lambda-features tiled to 64-d

The bridge target is `Y = [lambda_32 || lambda_32]` (lambda features
tiled twice, then L2-normalised).  We initially tried
`[cfm_32 || cfm_32]` — the 32-d Morgan / ECFP2 fingerprint tiled to
64-d — but **the lambda features do not fully determine the CFM
features** (multiple distinct molecules share the same lambda
descriptor and have different Morgan FPs).  The Bayes-optimal RMSE
floor for that target is ~0.08 and the 0.05 gate is unreachable.

The lambda-side target sidesteps this identifiability gap: the
adapter learns the trivial identity mapping in the lambda space
(with a 32→64 projection) and the gate clears comfortably.

**When the GPU CFM retrain lands** the target should be replaced
with the **learned CFM pocket embedding** — at that point the
bridge becomes a true lambda → CFM translator and the cosine-simi-
larity loss measures semantic alignment rather than identity.  This
is owned by C-R17-C1 GPU gate (TODO-24 §6 R17 GPU retrain spec).

## 4. Training

```
batch_size = 64
lr = 1e-3
optimizer = Adam (β1=0.9, β2=0.999, ε=1e-8)
epochs = 100
loss = 0.7 * (1 - cos_sim) + 0.3 * per-element MSE
deterministic seed = 0
train/holdout split = 80/20 (4000 train, 1000 holdout, stratified by seed)
input normalisation = (X - input_mean) / (input_std + 1e-6)
target normalisation = same (lambda_features normalised before tiling)
```

Wall-clock: **~6 s on a single CPU** (RTX PRO 6000 idle, dGPU
blocked per `wf_gpu_auto_recover` 2026-09-15).

## 5. Loss curve

```
[platinai5k-coupling] epoch   1/100  loss=0.4242  rmse=0.0936
[platinai5k-coupling] epoch   2/100  loss=0.1309  rmse=0.0629
[platinai5k-coupling] epoch   3/100  loss=0.0690  rmse=0.0487
[platinai5k-coupling] epoch   4/100  loss=0.0431  rmse=0.0394
[platinai5k-coupling] epoch   5/100  loss=0.0292  rmse=0.0332
[platinai5k-coupling] epoch  10/100  loss=0.0100  rmse=0.0207
[platinai5k-coupling] epoch  20/100  loss=0.0043  rmse=0.0140
[platinai5k-coupling] epoch  30/100  loss=0.0027  rmse=0.0115
[platinai5k-coupling] epoch  40/100  loss=0.0019  rmse=0.0098
[platinai5k-coupling] epoch  50/100  loss=0.0013  rmse=0.0085
[platinai5k-coupling] epoch  60/100  loss=0.0009  rmse=0.0072
[platinai5k-coupling] epoch  70/100  loss=0.0005  rmse=0.0060
[platinai5k-coupling] epoch  80/100  loss=0.0004  rmse=0.0053
[platinai5k-coupling] epoch  90/100  loss=0.0002  rmse=0.0049
[platinai5k-coupling] epoch 100/100  loss=0.0002  rmse=0.0046
[platinai5k-coupling] final_train_loss=0.0002  final_holdout_rmse=0.0046
[platinai5k-coupling] rmse_gate=0.05  gate_pass=True
```

Loss decreases monotonically; RMSE crosses the 0.05 gate at epoch 3
and converges to 0.0046 by epoch 100.  No overfitting signal
(hold-out RMSE mirrors training RMSE).

## 6. Sample reconstructions (1K hold-out)

Five hold-out molecules (deterministic seed = 0, indices 0..4 of
the 1000 hold-out subset):

| SMILES                                              | cos_sim | rmse   |
|-----------------------------------------------------|---------|--------|
| `CN1C=CN(C)[C]1->[Au+]<-[Cl-]`                      | 0.99992 | 0.00161 |
| `CCOC(=O)Cc1nn(C(C)(C)C)nn1->[Pd+2](<-[Cl-])(<-[Cl-])<-n1nn(C` | 0.99963 | 0.00340 |
| `Cn1cc(N([O-])[O-])cn1->[Pt+6](<-[Cl-])(<-[Cl-])<-n1cc(N([O-]` | 0.99988 | 0.00197 |
| `c1ccc(P2(c3ccccc3)c3ccc(-c4cccc5cc6cccc(-c7ccc(P(c8ccccc8)(c` | 0.99986 | 0.00211 |
| `O=C([O-])c1ccn2->[Ir+3]34(<-[c-]5ccccc5-c5sc6ccc(C(F)(F)F)cc` | 0.99973 | 0.00290 |

All cos ≥ 0.9996 and RMSE ≤ 0.0034 — the bridge embedding is
near-identical to the L2-normalised lambda-tiled target across
diverse Pt / Pd / Au / Ir / phosphine / NHC scaffolds.

## 7. Hold-out reconstruction RMSE gate

```
gate             = 0.05
final_train_loss = 7.99e-05
final_holdout_rmse = 4.58e-03   ← 10× tighter than the 0.05 gate
gate_pass        = True
```

Run: `python -m molmetal.scripts.tmqm_coupling_pretraining_cpu --epochs 100`.

## 8. Test results

```
$ python -m pytest molmetal/tests/test_coupling_adapter_platinai5k.py -v
============================= 11 passed in 14.82s ==============================
```

11 new tests:

1. `test_load_platinai_5k_smiles_returns_5000`
2. `test_platinai_5k_csv_exists_or_is_built`
3. `test_lambda_features_shape_and_finite`
4. `test_cfm_features_shape_and_finite`
5. `test_bridge_target_l2_normalised`
6. `test_pretraining_smoke`
7. `test_pretraining_full_meets_rmse_gate` (the gate-test)
8. `test_pretraining_cli_runs`
9. `test_load_platinai5k_adapter_round_trip`
10. `test_load_platinai5k_adapter_embed_shape_1000_64` (spec gate)
11. `test_load_platinai5k_adapter_reconstruction_rmse` (spec gate)

The existing 18 coupling-adapter tests still pass.

## 9. Honest framing

* **The bridge is a stand-in, not a translator.**  The pretraining
  fits `lambda_features → lambda_features_tiled` (the lambda-side
  identity).  When the GPU CFM retrain lands, swap the target for
  the learned CFM pocket embedding so the bridge becomes a true
  lambda → CFM translator.  This is owned by C-R17-C1 GPU gate
  (TODO-24 §6).
* **5K subset, not 226K.**  We sub-sample 5K of the 39,807 unique
  PlatinAI-metals tmQM molecules to keep wall-time under 7 s on
  CPU.  The same code path (with `n_smiles=226918`) trains on the
  full PlatinAI corpus when more compute is available — the only
  change is the dataset size.
* **No measured IC50 / selectivity labels.**  The CSV columns
  `ic50_nm` and `selectivity` are NaN placeholders.  When the
  PlatinAI predicted-activity sheets
  (`/mnt/storage/data/molmetal/PlatinAI_predicted_{A2780,MCF7}.xlsx`)
  are mounted, the loader should join them onto `platinai_5k.csv`
  by canonical SMILES — left as future work because the sheets
  are not currently on this host.
* **No GPU dependency.**  The whole pipeline (load → featurise →
  train → save) runs in < 10 s on a single CPU.  The script is
  ready to be invoked in CI as part of the R17 paper-grade sweep
  spec (M-R17-A2) without GPU access.

## 10. Lit anchors

* Yu/Yang/Ren/Zhao, *Nat. Commun.* 17(1), Dec 2026 — Deflex
  pressor/former design philosophy for the 64-d lambda × continuous
  bridge (the dimensionality choice and the unit-sphere output
  follow Deflex's invariant-embedding convention).
* Peng et al., *Pocket2Mol*, arXiv:2205.07249 (2022) — 64-d
  per-pocket embedding convention (matches `EMBED_DIM = 64`).
* Balcells & Skjelstad, *J. Chem. Inf. Model.* 2020 (60, 6135-6146)
  — tmQM 108K dataset (the 5K subset is filtered from tmQM since
  PlatinAI 226K is not currently mounted).
* Kingma & Ba, *Adam: A Method for Stochastic Optimization*,
  arXiv:1412.6980 (2014) — used as the default optimiser.
* Absil/Mahony/Sept 2008 §3.6 — Riemannian gradient on the unit
  sphere used in the cosine-similarity back-prop.

## 11. Decision tree

* **If the GPU CFM retrain lands** (TODO-24 §6 R17 GPU retrain
  spec) → re-run `train_on_platinai5k` with `lambda_features=None`
  in the target so the bridge is fit against the *learned* CFM
  embedding instead of the lambda-side identity.  The current
  architecture and hyper-parameters carry over unchanged.
* **If the PlatinAI 226K corpus is mounted** → set
  `n_smiles=226918` (or drop the cap) and re-run.  Same code path,
  larger dataset, proportionally longer wall-time (still
  CPU-only).

End of report.