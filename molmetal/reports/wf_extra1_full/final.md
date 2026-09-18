# WF-Extra-1 — D-MPNN pIC50 retrain (HeLa/48h/dark, censor-aware) — FINAL REPORT

Date: 2026-09-14
Script: `molmetal/scripts/retrain_pic50_neural.py`
Mode: full retrain (NOT smoke) — 50 epochs, 3 seeds (42, 0, 1234), all 1451 formulations
Output dir: `molmetal/reports/wf_extra1_full/`

---

## 1. Method

**Cohort** — HeLa / 48 h / dark, MetalCytoToxDB.csv (FROZEN, identical loader to
`calibrate_conditioned_pic50_baseline.py`).
- **1451 formulations / 715 scaffolds**.
- **251 censored** (17.3 %) preserved as upper bounds (">X uM"), NEVER re-coded to
  exact targets.
- Within a formulation group, censored rows aggregate to a single censored row
  with the most restrictive (smallest) bound.

**Splits** — 80/10/10 scaffold-group `GroupShuffleSplit`, identical seed schedule
to the conditioned baseline (seeds 42, 0, 1234). Ligand- and scaffold-leakage
checks both pass at construction time.

| Seed | n_train | n_val | n_test | censored_train | censored_test |
|---:|---:|---:|---:|---:|---:|
| 42  | 1173 | 152 | 126 | 203 | 18 |
| 0   | 1150 | 162 | 139 | 200 | 33 |
| 1234| 1181 | 138 | 132 | 202 | 16 |

**Model** — `AttentiveDMPNNModel(atom_dim=39, bond_dim=10, hidden=128, depth=3,
dropout=0.1)` (dims match historical `dmpnn_attn_ru_pic50.pt`).

**Loss** — `MSE(exact) + hinge-margin(censored, margin=0.25)` on pIC50 units.
- Exact rows: standard MSE on `pIC50 = 6 - log10(IC50_uM)`.
- Right-censored rows (censor_dir=+1, ">X uM"): hinge
  `max(0, margin - (pIC50_bound - pred))`, penalising
  `pred > pIC50_bound + margin` (model claims more potency than data allows).
  `pred ≤ pIC50_bound` is free.
- Left-censored (censor_dir=-1, "<X uM"): symmetric.

**Optim** — Adam, lr=1e-4, batch_size=64, grad-clip 5.0, 50 epochs, best
val-RMSE state selected per seed.

**Hardware note (honest)** — the subagent Python process did not see a CUDA
device (`torch.cuda.is_available() == False`); training ran on CPU. Wall-clock
per epoch ≈ 3 s on 1173 graphs; total 3-seed sweep ≈ 7 min. A re-run with a
CUDA-visible Python would execute the same code on gfx1101 with no edits.

---

## 2. Results (MEASURED, 3 seeds × 50 epochs)

### 2.1 Per-seed test metrics

| Seed | n_test | n_exact_test | n_censored_test | RMSE (pIC50) | Pearson r | Spearman ρ | MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42   | 126 | 108 | 18 | 0.6801 | 0.0044 | 0.0115 | 0.5556 |
| 0    | 139 | 106 | 33 | 0.6397 | 0.5189 | 0.4718 | 0.4909 |
| 1234 | 132 | 116 | 16 | 0.7409 | 0.0709 | 0.1199 | 0.6047 |

### 2.2 Aggregate across 3 seeds

| Metric | Mean ± Std | Source |
|---|---:|---|
| **test_rmse** (pIC50) | **0.687 ± 0.042** | this run |
| **test_pearson_r** | **0.198 ± 0.228** | this run |
| **test_spearman_rho** | **0.201 ± 0.197** | this run |
| **censored_coverage** | **0.169** (67/397) | this run |
| **bound_aware_accuracy** | **0.000** (0/67) | this run (see §3) |

### 2.3 Conditioned ridge baseline (DIRECT comparison — same cohort, same splits)

From `molmetal/reports/pic50_conditioned_baseline_20260913/report.json` —
real gfx1101 GPU dual-ridge baseline on the FROZEN HeLa48h/dark cohort:

| Seed | ridge RMSE | ridge Pearson r |
|---:|---:|---:|
| 42   | 0.5972 | 0.4774 |
| 0    | 0.6116 | 0.6055 |
| 1234 | 0.4150 | 0.6341 |
| **mean ± std** | **0.541 ± 0.089** | **0.572 ± 0.068** |

### 2.4 Headline comparison

| Model | RMSE (pIC50) | Pearson r | n_seeds | cohort | comparable? |
|---|---:|---:|---:|---|---|
| **D-MPNN retrain (this run)** | 0.687 ± 0.042 | **0.198 ± 0.228** | 3 | HeLa48h/dark, 1451 form., censor-aware | YES — same cohort |
| Conditioned ridge baseline | 0.541 ± 0.089 | **0.572 ± 0.068** | 3 | HeLa48h/dark, 702 form., exact-only | YES — same cohort |
| Historical D-MPNN (legacy) | n/a | 0.407 | n/a | mixed assay, censored-as-exact, random split | NO — incomparable cohort |

---

## 3. Verdict — does the neural D-MPNN beat the ridge baseline?

**No.** On the FROZEN HeLa48h/dark cohort (the only fair comparison):

- **RMSE**: neural 0.687 vs ridge 0.541 → ridge wins by 0.146 pIC50 units
  (~21% relative reduction).
- **Pearson r**: neural 0.198 vs ridge 0.572 → ridge wins by 0.374, and the
  neural 3-seed std (±0.228) shows the result is **seed-fragile**: one seed
  (0) hit r=0.519, two seeds (42, 1234) collapsed to near-zero correlation.
  The aggregate mean is dragged down by those collapses, not lifted by the
  one good seed.
- **Historical D-MPNN (r=0.407)**: the retrain is **worse** than the legacy
  model on this metric — 0.198 vs 0.407 — but this is **incomparable across
  cohorts**. The legacy model trained on a mixed-assay / censored-as-exact
  / random-split cohort; the retrain cohort is narrower (one cell line /
  one time point) but **honest**. The legacy number is **not** a useful
  ceiling for the new retrain.

**Why the neural model regresses toward the mean**
- D-MPNN with `lr=1e-4`, `batch_size=64`, `hidden=128`, `depth=3` on 1173
  training graphs with a hinge-margin (margin=0.25) regulariser is
  under-regularised for a 1451-row task. Predictions collapse to a near-
  constant (per-seed `pred_std` ∈ [0.077, 0.160] vs `y_std` ≈ 0.70).
- The censored-loss hinge (margin=0.25 pIC50 ≈ half a log unit) is **smaller
  than the model's typical prediction error** (~0.7 RMSE on test). The
  hinge never activates in practice.

---

## 4. Verdict — is the bound-aware loss effective?

**No, not in this configuration.** The censoring handling has a concrete,
measurable defect:

- **bound_aware_accuracy = 0.000** (0/67 censored test rows).
  Every censored test row gets a predicted pIC50 **higher** than the
  bound pIC50 — i.e. the model systematically claims more potency than
  the censoring allows.
- Mean (pred − bound) over censored test rows: **+0.797 pIC50 units**
  (range [0.29, 1.22], all positive).
- Censored test rows have bound pIC50 in [3.52, 4.30] (these are the
  *weak* binders — ">X uM" labels). Exact-row targets have mean 4.83 and
  range [2.85, 6.82]. The model regresses toward the exact-row mean (~4.72),
  which is **above** the censored-row bound (3.93). So the model literally
  cannot predict compliantly for censored rows without an explicit
  directional penalty larger than the regression pull.

**Diagnosis**: a `margin=0.25` hinge on pIC50 only penalises
`pred > bound + 0.25`, but the model's regression-to-mean pull is ~0.80
pIC50. The hinge never fires. To make the bound-aware loss effective,
the margin would need to be raised to ≥1.0 (or the loss replaced with
a `max(0, -dir*(pred-bound))` zero-margin penalty that activates on
*any* violation).

**What is preserved correctly** (the architectural goal of WF-Extra-1):
- Censored rows are NEVER silently re-coded to exact targets — the bound
  stays as a constraint in the loss.
- The hinge loss term does run end-to-end on every censored batch row.
- The defect is one of *strength*, not one of *plumbing*.

---

## 5. Honest framing — MEASURED vs PROJECTED

**MEASURED in this run (3 seeds × 50 epochs, HeLa48h/dark cohort)**:
- test RMSE = 0.687 ± 0.042 pIC50 units
- test Pearson r = 0.198 ± 0.228
- test Spearman ρ = 0.201 ± 0.197
- censored coverage of test = 16.9 %
- bound-aware accuracy = 0.0 % (0/67)
- 3-seed seed-fragility: σ/μ = 1.15 on Pearson r (highly seed-dependent)
- scaffold+ligand leakage checks: pass
- GPU wiring: smoke-equivalent forward/backward proven on CPU; gfx1101
  re-run with same code expected to match (not measured here)

**PROJECTED (not measured in this run)**:
- Generalisation to other cell/time cohorts (HeLa24h, HeLa72h, MCF7,
  A549, …) — the censored-aware loss is a cohort-portable design but
  has not been validated off-cohort.
- Behaviour on full MetalCytoToxDB coverage (no Cell/Time filter) —
  the cohort filter is **required** for the current script; pooled
  cohorts would still violate the assay heterogeneity finding from
  TODO-18.
- A larger / regularised variant (deeper model, dropout↑, weight decay,
  or attention-pool + auxiliary tasks) could plausibly reach the target
  band [0.5, 0.7] — this run **does not** demonstrate that. The current
  run **does not** justify citing the target band as achieved.
- Calibration of the bound-aware loss to a margin ≥ 1.0 pIC50 (or a
  zero-margin penalty). Required before claiming the loss is "effective".

**NOT MEASURED & OUT OF SCOPE**:
- Wet-lab activity prediction. No computational metric establishes
  biological efficacy. The retrained model remains an in-silico estimator
  on one (cell, time, dark) cohort only.

---

## 6. Verdict (paper section 4.2 framing)

On the FROZEN HeLa48h/dark cohort, with scaffold-group splits and a
censor-aware loss, the Attentive D-MPNN retrain (3 seeds × 50 epochs)
**does not match** the conditioned dual-ridge baseline:

- Ridge wins on RMSE (0.541 vs 0.687) and on Pearson r (0.572 vs 0.198).
- Neural D-MPNN is **seed-fragile** (Pearson r ranges from 0.004 to
  0.519 across the three seeds), and predictions collapse toward the
  cohort mean.
- The bound-aware loss as configured (margin=0.25) is **not effective**:
  0/67 censored test rows are predicted compliantly. The hinge is
  weaker than the regression-to-mean pull. Stronger directional
  regularisation is required.

**Recommendations (out of scope here, recorded for TODO-19 / paper §4.2):**
- Raise censor margin to ≥ 1.0 pIC50, OR replace hinge with a
  zero-margin `max(0, -dir*(pred-bound))` penalty that activates on any
  violation.
- Add weight decay + larger dropout, OR a multi-seed hyperparameter
  sweep with early stopping on val-RMSE per seed.
- Treat the **conditioned ridge baseline as the credible ceiling** for
  this cohort until the neural model demonstrates robust ≥ 0.5 Pearson r
  across ≥ 3 seeds with low σ.

**Files produced** (absolute paths):
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_extra1_full/final.md` (this file)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_extra1_full/report.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_extra1_full/report.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_extra1_full/aggregate.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_extra1_full/test_predictions.parquet` (397 rows)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_extra1_full/run.log`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/checkpoints/dmpnn_attn_heLa48h_dark_retrained.pt` (last seed wins; seed=1234)

**Files NOT touched** (WF-Extra-1 invariant preserved):
- `molmetal/baselines/dmpnn_attentive.py`
- `molmetal/molmetal_lam/sbdd_env/pic50_predictor.py`
- `molmetal/molmetal_lam/training/pic50_censored_loader.py`
- `molmetal/scripts/calibrate_conditioned_pic50_baseline.py`
- Any historical checkpoint (`dmpnn_attn_ru_pic50.pt`,
  `dmpnn_tmqm_pretrained.pt`, etc.) other than the per-seed retrain
  bundle written by this script.
- All Lambda / CFM / round-trip / metal-seed / BNF code paths.
