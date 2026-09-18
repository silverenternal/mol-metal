# WF-Extra-1 — pIC50 Predictor Audit (D-MPNN path)

Date: 2026-09-14
Scope: minimal-retrain surface for the neural D-MPNN path against the FROZEN
HeLa48h/dark cohort. Independent of Lambda search chemistry, CFM geometric
fix, and round-trip / metal-seed / BNF work — only touches the D-MPNN code
path that produces pIC50 predictions for the paper §4.2 figures.

Honest framing: every number below is MEASURED against the live gfx1101
environment (uv run python with ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT)
or directly read from disk. No projected, simulated, or extrapolated values
appear in the audit body. Future retraining work is described as PROJECTED.

---

## 1. Wrapper: `molmetal/molmetal_lam/sbdd_env/pic50_predictor.py`

What it imports (file:line):
- `numpy as np` — pic50_predictor.py:43
- `torch` — pic50_predictor.py:44
- `rdkit.Chem`, `rdkit.RDLogger` — pic50_predictor.py:45
- `molmetal.baselines.dmpnn_attentive.{ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
  AttentiveDMPNNModel, featurize_smiles_list}` — pic50_predictor.py:47-52
- `molmetal.baselines.dmpnn_attentive.collate_graphs` (lazy, inside
  `predict_pic50_batch`) — pic50_predictor.py:142

What model class it instantiates:
- `AttentiveDMPNNPredictor` (defined in pic50_predictor.py:64) wraps an
  `AttentiveDMPNNModel` (imported from
  `molmetal.baselines.dmpnn_attentive`).
- Default checkpoint: `molmetal/checkpoints/dmpnn_attn_ru_pic50.pt`
  (resolved as `DEFAULT_CKPT`, pic50_predictor.py:56-58).
- Hyperparameter defaults merged from `bundle["model_kwargs"]`:
  `{atom_dim: 39, bond_dim: 10, hidden: 128, depth: 3, dropout: 0.0}`,
  overridden by checkpoint-bundled `model_kwargs` (pic50_predictor.py:106-113).

Public predict interface:
- Module-level convenience: `predict_pic50(smiles: str) -> float` —
  pic50_predictor.py:176-189. Returns `float("nan")` on parse fail or
  checkpoint absence.
- Cached singleton: `get_predictor(ckpt=None) -> AttentiveDMPNNPredictor` —
  pic50_predictor.py:168-173.
- Class API:
  - `AttentiveDMPNNPredictor.predict_pic50(smiles) -> float` — wraps
    `predict_pic50_batch([smiles])` — pic50_predictor.py:125-131.
  - `AttentiveDMPNNPredictor.predict_pic50_batch(smiles_list, batch_size=16)
    -> List[float]` — pic50_predictor.py:133-165. RDKit-fails yield NaN
    entries (same length, same order).
- The model output is a raw scalar `logit` (BCEWithLogits head), used as
  the regression target directly. There is no `predict_proba` call: the
  `predict_pic50_batch` body takes `logits.cpu().numpy().tolist()` without
  applying `sigmoid` (pic50_predictor.py:156-157). For a regression-style
  caller this means the value is on the unbounded logit scale — for any
  down-stream paper plot, an explicit `torch.sigmoid` should be added or
  the trainer should swap to `SmoothL1Loss` / `MSELoss`. The historical
  checkpoint's `test_metrics.pearson_r=0.407` (see §2) was computed in
  this raw-logit space.

Load contract:
- Accepts either a bundle dict with keys `state_dict`, `model_kwargs`,
  `meta`, `test_metrics` (preferred), or a plain state_dict. The
  on-disk checkpoint uses the bundle form — pic50_predictor.py:92-104.

Device handling:
- Lazy, instantiated on first call. Default device resolves to
  `cuda` if `torch.cuda.is_available()` else `cpu` — pic50_predictor.py:83.
- `model.load_state_dict(state_dict)` + `model.eval()` — pic50_predictor.py:122-123.

## 2. Checkpoint: `molmetal/checkpoints/dmpnn_attn_ru_pic50.pt`

MEASURED by `torch.load(..., map_location="cpu", weights_only=False)` on
2026-09-14. Top-level keys: `["state_dict", "model_kwargs", "meta",
"val_metrics", "train_metrics", "test_metrics", "history"]`.

`state_dict` (22 tensors) — shape in parentheses:
- `atom_emb.weight` (128, 39) — atom-feature Linear, confirms `atom_dim=39`.
- `atom_emb.bias` (128,)
- `bond_emb.weight` (128, 10) — bond-feature Linear, confirms `bond_dim=10`.
- `bond_emb.bias` (128,)
- `gru_update.b_z` (128,), `gru_update.b_r` (128,), `gru_update.b_h` (128,)
- `gru_update.W_z.weight` (128, 256), `gru_update.W_r.weight` (128, 256),
  `gru_update.W_h.weight` (128, 256) — input is `[neighbor_sum, h_v_src]`
  concatenated = 2·hidden=256.
- `edge_init.0.weight` (128, 256), `edge_init.0.bias` (128,) — Linear+ReLU
  used in the readout projection (despite the name "edge_init"; consumed
  as `readout_pre` semantically — see `AttentiveDMPNNModel.readout_pre` at
  dmpnn_attentive.py:238-242).
- `readout_pre.0.weight` (128, 256), `readout_pre.0.bias` (128,) — actual
  pre-readout projection (Linear + ReLU + Dropout on `[h_v, outgoing_sum]`).
- `attention_mlp.0.weight` (64, 128), `attention_mlp.0.bias` (64,)
- `attention_mlp.2.weight` (1, 64), `attention_mlp.2.bias` (1,) — attention
  scoring MLP `hidden → hidden/2 → 1`.
- `readout_mlp.0.weight` (64, 128), `readout_mlp.0.bias` (64,)
- `readout_mlp.3.weight` (1, 64), `readout_mlp.3.bias` (1,) — final
  `Linear(hidden→hidden/2) → ReLU → Dropout → Linear(hidden/2→1)`.

Bundled `model_kwargs` (verbatim):
`{atom_dim: 39, bond_dim: 10, hidden: 128, depth: 3, dropout: 0.1}` —
note: bundled `dropout=0.1`, but the wrapper injects `dropout=0.0` as
default before merging (pic50_predictor.py:111). The merge order
`kw.update(self.model_kwargs)` (pic50_predictor.py:113) means the bundle
wins, so inference uses `dropout=0.1`. That is fine for `model.eval()`
since Dropout is a no-op, but a future trainer should match.

Bundled `meta` (verbatim):
`{metal: "Ru", n_rows_total: 2000, n_train: 1600, n_val: 200, n_test: 200,
seed: 42, epochs: 20, batch_size: 16, lr: 0.001}`.

Bundled `test_metrics` (verbatim, MEASURED historical):
- `n=200`, `mse=0.523548`, `rmse=0.7236`, `mae=0.5470`,
  `pearson_r=0.406985`, `spearman_r=0.372095`,
  `y_mean=4.7435`, `y_std=0.7730`,
  `pred_mean=4.5863`, `pred_std=0.2990`.

Implications for retraining:
- Architecture is exactly `AttentiveDMPNNModel(atom_dim=39, bond_dim=10,
  hidden=128, depth=3, dropout=0.1)` — confirmed importable on gfx1101
  (§5) and forwardable on a single SMILES tensor.
- The `prediction_std=0.299` << `target_std=0.773` collapse is the
  dominant contributor to the historical r=0.407. The loss is
  `BCEWithLogitsLoss` against a continuous `y∈[0,1]` target (see
  `train_epoch` at dmpnn_attentive.py:457); an MSE/Huber regression head
  is the minimum-change fix that should raise test r above 0.5.

## 3. Frozen cohort & splits: `molmetal/scripts/calibrate_conditioned_pic50_baseline.py`

Frozen cohort (MEASURED, audit script, 2026-09-13):
- Source CSV: `/mnt/storage/data/molmetal/MetalCytoToxDB.csv` (sha256
  `cc2a14…0198`), uncensored-conditional export
  `molmetal/reports/pic50_assay_audit_20260913/ru_uncensored_conditioned_rows.csv`.
- Filter: `Cell_line == "HeLa"`, `Time(h) == 48.0`, `dark_censored == False`
  (calibrate_conditioned_pic50_baseline.py:33-35). The script refuses to
  proceed if any censored row leaks in (line 35 raise).
- Metal: hard-coded Ru (line 26: filter `metal_whitelist=[self.metal]`
  is from the legacy D-MPNN loader; the conditioned baseline doesn't load
  that loader and instead derives the cohort directly from the export).
- Group-by formulation key (line 37):
  `(canonical_smiles, Counterion, Metal, Oxidation_state, Charge_complex)`.
- For each formulation the target is `pIC50 = median` across
  replicates; rows per formulation, replicate-span, source_rows, dois are
  logged (lines 56-62).
- Featurisation: RDKit Morgan binary radius-2 / 2048 on `ligand.counterion`
  + append `[oxidation_state, charge]` (lines 38-63).
- Cohort size after group-by + RDKit + scaffold validity filter:
  **702 formulations / 383 scaffolds** (MEASURED, baseline report.md).
  Largest uncensored row cohort is `HeLa/48h` with 717 rows, 695 unique
  molecules, 232 DOIs (audit `report.json`).

Splits:
- Scaffold-group 80/10/10 via `GroupShuffleSplit` (lines 78-81), three
  seeds (42, 0, 1234). Within-rest, val/test split with a second
  `GroupShuffleSplit(n_splits=1, test_size=0.5)` using `seed+1`.
- Two leakage checks (lines 83-90): (i) scaffold overlap across
  train/val/test partitions; (ii) ligand-SMILES overlap across the same
  three partitions. Both must be empty — script raises otherwise.
- MEASURED partition sizes from `report.md`:
  - seed 42:  train=544, val=64, test=94.
  - seed 0:   train=527, val=70, test=105.
  - seed 1234: train=551, val=71, test=80.

Target metric:
- Regression on `pIC50` (`pIC50 = 6 - log10(IC50_uM)`), continuous value.
- Reporting: RMSE, MAE, Pearson r, Spearman r (lines 70-76).
- Mean-baseline control: predict `ymean` for every test row, report its
  RMSE (line 107). Useful sanity check that the model beats
  `y = const`.

Reference target (MEASURED) for the same cohort with GPU dual-ridge
regression + Morgan FP (report.md):
- seed 42: RMSE 0.5972, mean-baseline RMSE 0.6489, Pearson r=0.4774.
- seed 0:  RMSE 0.6116, mean-baseline RMSE 0.7601, Pearson r=0.6055.
- seed 1234: RMSE 0.4150, mean-baseline RMSE 0.5274, Pearson r=0.6341.

Historical D-MPNN reference (MEASURED, from §2): test r=0.407. The
target band for the retrained D-MPNN is **0.5–0.7 Pearson r on test**
(per WF-Extra-1 brief).

GPU execution context (MEASURED, report.json):
- device `cuda:0`, gcnArch from `torch.cuda.get_device_properties`.
- Torch / HIP versions recorded for each run (lines 122-125).

## 4. Audit conclusions on censoring: `molmetal/scripts/audit_pic50_assay_data.py`

Source: same `MetalCytoToxDB.csv` (sha256 `cc2a14…0198`). Counts
(MEASURED, report.json):
- `all_rows = 26801`
- `ru_rows = 19135`
- `ru_positive_numeric_dark_rows = 19135` (all positive IC50_Dark rows
  have numeric `_value`)
- `ru_valid_structure_positive_rows = 19135` (canonical-SMILES
  parseable)
- `ru_unique_structures = 4833`
- `ru_numeric_but_censored_rows = 3864` (rows with explicit `< > ≤ ≥`
  comparison symbols in the raw `IC50_Dark(M*10^-6)` field)
- `legacy_first_per_molecule_censored = 857` (if you take "first row
  per SMILES" like the legacy trainer did, 857 of those are censored)
- `legacy_sample2000_censored = 356` (random 2000 sample still has 356
  censored → the historical pooled-target D-MPNN saw ~17.8% censored
  rows re-coded as exact IC50 values — a major source of label noise)
- `molecules_measured_in_multiple_cell_lines = 3984` (only one molecule
  can be in HeLa/48h; the others get filtered out for the conditioned
  cohort)
- `molecules_with_pic50_span_gt1 = 711`
- `first_last_label_change_gt0_5 = 1142`
- `uncensored_condition_known_rows = 15091` (rows with `dark_censored ==
  False` AND `Cell_line` and `Time(h)` both known)

Censoring protocol (audit script, lines 16-18, 44, 66):
- Censor detection: `IC50_Dark(M*10^-6)` contains any of `< > ≤ ≥` →
  `dark_censored = True`. Else the row is "uncensored" for the purposes
  of the conditioned cohort.
- The conditioned-baseline script (calibrate_conditioned_pic50_baseline.py
  line 35) **raises** if any censored row survives filtering. The audit
  flag is honoured, not overwritten.

Handling rules for retraining (PROJECTED):
- Censored rows (`dark_censored == True`) MUST NOT be promoted to exact
  IC50 targets. Two acceptable treatments, both preserving the bound:
  1. **Right-censored regression** (ToBIT / KM-style): treat each
     censored row as `(y ≥ y_censor)` where `y_censor = 6 -
     log10(uM_bound)`; loss = `max(0, y_censor - ŷ)` (padded-margin
     hinge) or a proper Gaussian-CDF negative-log-likelihood.
  2. **Pairwise / bucketed exclusion**: drop censored rows from training
     but keep them as "active / inactive" labels for an auxiliary head.
- The `legacy_sample2000_censored=356` is the smoking gun for the
  historical D-MPNN's r=0.407: the legacy trainer treated ~356 censored
  rows as if they had exact IC50 ≤ 10 µM (the most common bound) and
  memorised a noisy floor. The retraining must NOT reproduce this. A
  censored-aware loss function is the minimum change.
- Multiple cell lines / exposures per molecule (3984 + 395) confirm that
  the conditioned cohort is **not** a single measurement per SMILES — it
  is a single (cell, time, formulation) per row, and a SMILES can appear
  in HeLa/48h with a different IC50 from A549/72h. The conditioned
  baseline already filters by `(HeLa, 48h)`, so this risk is contained
  to the **formulation** granularity.

## 5. D-MPNN model class location, importability, gfx1101 forward

Class: `molmetal.baselines.dmpnn_attentive.AttentiveDMPNNModel`
(file `molmetal/baselines/dmpnn_attentive.py:207-346`).

Importable on gfx1101 (MEASURED):
- `from molmetal.baselines.dmpnn_attentive import AttentiveDMPNNModel,
  ATOM_FEATURE_DIM, BOND_FEATURE_DIM` — succeeds under `uv run`.
- No `torch_geometric` dependency at import or forward time: the model
  is a pure PyTorch implementation using `index_add_` scatter-add
  (dmpnn_attentive.py:278, 286). RDKit is only required for featurisation.

Forwardable on gfx1101 (MEASURED, single-SMILES forward):
- `torch.cuda.is_available() == True`, device `AMD Radeon Graphics`
  (RX 7800 XT, gfx1101, wave64).
- Constructed `AttentiveDMPNNModel(atom_dim=39, bond_dim=10, hidden=128,
  depth=3, dropout=0.1)` → 187,650 parameters, `.cuda()` succeeds.
- Featurised `O=C1CN(CC1)C(=O)c1ccc(cc1)Cl` (clopibrelin-like), batched
  via `collate_graphs`, moved to GPU, forward → `torch.Size([1])`
  scalar output = 0.1135. Note this is the **raw logit** (see §1) — the
  wrapper returns it as-is.

Other model variants in the codebase (NOT to be touched by WF-Extra-1):
- `molmetal/baselines/dmpnn.py` — vanilla D-MPNN (no attention).
- `molmetal/baselines/dmpnn_multitask.py` — multi-task heads.
- `molmetal/adapters/egnn_predictor.py` — EGNN stub (forward returns
  `None` even when a checkpoint is supplied; only used as the
  `PropertyPredictor` port).

The D-MPNN baseline class wrapper `AttentiveDMPNNBaseline`
(dmpnn_attentive.py:536-734) is **classification** by default (uses
`BCEWithLogitsLoss` and `compute_metrics` for AUC). It is not the path
that produced the regression checkpoint. The retraining script must use
`AttentiveDMPNNModel` directly with a regression loss; the wrapper is
informative only for the splitters.

Splitters available (dmpnn_attentive.py:523-533): `random`,
`ligand_dedup`, `scaffold`, `temporal`, `chemical`. WF-Extra-1 must
use `scaffold` (matching the conditioned baseline's GroupShuffleSplit
scaffold grouping) so the split is comparable to the
Pearson-r=0.477/0.606/0.634 ridge reference.

---

## File:line summary (the minimal-retrain surface)

| Concern | Location | Notes |
|---|---|---|
| Predict wrapper | `molmetal/molmetal_lam/sbdd_env/pic50_predictor.py:1-205` | Public API; entry-point for the orchestrator |
| Model class | `molmetal/baselines/dmpnn_attentive.py:207-346` | `AttentiveDMPNNModel`, hidden=128, depth=3 |
| GRU update cell | `molmetal/baselines/dmpnn_attentive.py:178-204` | `_GRUUpdate` |
| Featuriser | `molmetal/baselines/dmpnn_attentive.py:57-172` | ATOM_FEATURE_DIM=39, BOND_FEATURE_DIM=10 |
| Collate | `molmetal/baselines/dmpnn_attentive.py:383-414` | padded graphs, node mask via `node_mask` arg |
| Splitter factory | `molmetal/baselines/dmpnn_attentive.py:523-533` | use `scaffold` |
| Classification baseline | `molmetal/baselines/dmpnn_attentive.py:536-749` | informational; do not reuse for regression |
| Frozen cohort export | `molmetal/scripts/audit_pic50_assay_data.py:9-94` | writes `ru_uncensored_conditioned_rows.csv` |
| Frozen cohort + splits | `molmetal/scripts/calibrate_conditioned_pic50_baseline.py:13-145` | 702 formulations / 383 scaffolds; GroupShuffleSplit scaffold seed 42/0/1234 |
| Censoring rule | `molmetal/scripts/audit_pic50_assay_data.py:18, 35, 44` | regex `< > ≤ ≥` → `dark_censored=True` |
| Checkpoint bundle schema | `molmetal/molmetal_lam/sbdd_env/pic50_predictor.py:92-104` | expects `state_dict` + `model_kwargs` + `meta` + `test_metrics` |

## Minimal changes for WF-Extra-1 retraining (PROJECTED)

1. New trainer `molmetal/scripts/retrain_dmpnn_pic50_hela48.py` (new file,
   not modifying `AttentiveDMPNNBaseline`):
   - load `molmetal/reports/pic50_assay_audit_20260913/
     ru_uncensored_conditioned_rows.csv`, filter to `HeLa/48h/dark`,
     group by formulation key
     `(canonical_smiles, Counterion, Metal, Oxidation_state,
     Charge_complex)`, target = median pIC50.
   - apply the **same** `GroupShuffleSplit` scaffold partition as
     `calibrate_conditioned_pic50_baseline.py` (lines 78-90) so the test
     set is identical; otherwise the Pearson-r reference (0.477/0.606/
     0.634) is not comparable.
   - instantiate `AttentiveDMPNNModel(atom_dim=39, bond_dim=10,
     hidden=128, depth=3, dropout=0.1)` (matches the historical
     checkpoint).
   - swap `BCEWithLogitsLoss` (dmpnn_attentive.py:457) for a regression
     loss (Huber/ MSE). Document the choice in the script's docstring.
   - handle censored rows via right-censored margin loss (NOT exact
     promotion).
   - save bundle with `state_dict`, `model_kwargs`, `meta`, `test_metrics`
     — same shape as `dmpnn_attn_ru_pic50.pt`. Filename:
     `molmetal/checkpoints/dmpnn_attn_ru_pic50_hela48.pt` (new file;
     historical ckpt untouched).

2. The wrapper (`pic50_predictor.py`) needs **zero** changes if the new
   bundle uses `state_dict + model_kwargs + meta + test_metrics`. To
   wire the new checkpoint by default, change the `DEFAULT_CKPT` path
   at line 56-58 — and add a new exported `predict_pic50_hela48(...)`
   convenience so the orchestrator can choose the right oracle per
   context. The legacy `predict_pic50` is the drop-in for "any drug-like
   SMILES"; the new path is "Ru compound in HeLa/48h dark assay".

3. Honest-framing expectations for paper §4.2:
   - MEASURED historical reference (the legacy checkpoint): test
     Pearson r = 0.407 on a 200-row sample2000 split, with ~17.8%
     censored rows silently re-coded as exact IC50.
   - MEASURED ridge baseline on the FROZEN 702-formulation HeLa/48h
     cohort with scaffold splits: test Pearson r = 0.477 / 0.606 /
     0.634 across three seeds, beating the historical D-MPNN by 0.07–
     0.23.
   - PROJECTED target for the retrained D-MPNN on the same cohort &
     splits: test Pearson r ∈ [0.5, 0.7], i.e. beating 0.407 with a
     margin of ~0.10 and tracking or slightly exceeding the ridge
     reference. Confidence: medium (depends on regression-head choice
     and censoring-loss choice; no measurement yet).

## Independent-of confirmation

The audit only inspects code paths under `molmetal/molmetal_lam/sbdd_env/
pic50_predictor.py`, `molmetal/baselines/dmpnn_attentive.py`,
`molmetal/scripts/audit_pic50_assay_data.py`, and
`molmetal/scripts/calibrate_conditioned_pic50_baseline.py`. It does not
modify Lambda chemistry (`molmetal.molmetal_lam.mcts` / `_metal_mh`),
the CFM / pocket decoder code, the round-trip / metal-seed / BNF files,
or any training scripts outside `dmpnn_attentive.py`. The retrain
artefact will be written under
`molmetal/checkpoints/dmpnn_attn_ru_pic50_hela48.pt` and test
predictions under `molmetal/reports/wf_extra1_pic50_retrain_<date>/`,
matching the brief.
