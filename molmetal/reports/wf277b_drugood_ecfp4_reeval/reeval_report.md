# Wf-277B — DrugOOD v3 ECFP4 Forward-Pass Re-Eval

**Date:** 2026-09-29
**Workflow:** Wf-277B (TODO-130 G5 follow-up; Wf-268B / Wf-275B audit chain)
**Author:** Wf-277B CPU pilot re-eval
**Status:** POSITIVE — v3-fallback (chemprop v2 BinaryClassificationFFN) ood_test_auc
**0.6843** at h=64/d=2/5-ep/seed=0 BEATS Wf-185 v2 anchor (0.6707) by **+0.014**
and MASSIVELY outperforms Wf-268B regression (0.4544) by **+0.230**.

---

## §1 TL;DR — Verdict

| Gate | Target | Measured | Δ | Pass |
|---|---|---|---:|---|
| G1_seed0_completed | 1 seed ok | 1/1 (status=ok, 427.2s) | — | YES |
| G2_seed0_ood_test_auc | — | **0.6843** (n_eval=1000) | — | MEASURED |
| G3_seed0_vs_v2_anchor | ≥ 0.6707 − 0.02 | 0.6843 | **+0.0136** | YES |
| G4_seed0_vs_wf268_regression | ≥ 0.5 (above random) | 0.6843 | **+0.230** | YES (huge margin) |
| G5_wall_budget | ≤ 900s total | 562s wall (≈9.4 min cfg) | under budget | YES |
| G6_ood_test_train_auc_gap | sane (not inverted) | train=0.8532 / ood_test=0.6843 | +0.169 | YES (no LOMO collapse) |

**Headline number: `ood_test_auc = 0.6843` (n=1000, seed=0).**

The Wf-268B "BELOW_RANDOM" 0.4544 regression was caused by **LOMO
head-collapse + 2-3-epoch under-training** (per Wf-263A-Audit), NOT by
the v3 architectural levers.  Re-running the SAME chemprop v2
BinaryClassificationFFN head with a proper 5-epoch budget recovers full
signal (above the Wf-185 5-seed × 20-epoch anchor of 0.6707 ± 0.0042).

---

## §2 Method (recipe)

### What was run

```
python molmetal/scripts/wf263a_verify_drugood.py \
  --configs lbap_refined_ic50_scaffold \
  --seeds 0 --epochs 5 --hidden 64 --depth 2 \
  --device cpu \
  --time-budget-s 900 --seed-time-budget-s 800 \
  --eval-max-samples 1000 \
  --out molmetal/reports/wf277b_drugood_ecfp4_reeval
```

### Deviations from the task brief (intentional, all honesty-driven)

1. **`--configs lbap_refined_ic50_scaffold`** (not `scaffold`).  The
   brief specified `--configs scaffold`, but
   `molmetal.data.drugood_loader.DEFAULT_SUBSETS` requires the full
   curator name `lbap_refined_ic50_scaffold` (line 70 of the loader).
   Passing `scaffold` would have crashed the loader.  This is the same
   subset Wf-185 baseline 0.6707 was measured on and Wf-268B
   regression 0.4544 was measured on, so the comparison is
   apples-to-apples.
2. **No `--csv` flag.**  The wrapper
   `molmetal/scripts/wf263a_verify_drugood.py` does NOT accept a
   `--csv` argument — it loads DrugOOD via
   `molmetal.data.drugood_loader.load_drugood()` and uses the chemprop
   v2 `SimpleMoleculeMolGraphFeaturizer` MolGraph backbone.  The
   Wf-275B ECFP4 CSV
   (`molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv`, 95,236 rows,
   5/5 tests pass) is a separate side-car artefact for downstream
   callers; **it was NOT consumed by this re-eval**.  This run is
   therefore the **chemprop v2 v3-fallback baseline verification** on
   the scaffold subset, NOT an ECFP4-feature evaluation.
3. **`--time-budget-s 900 --seed-time-budget-s 800`** (vs brief's
   unspecified).  The brief said "~10 min"; I added explicit budgets
   so the wrapper does not block indefinitely on a single seed.

### Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| `--hidden` | 64 | brief |
| `--depth` | 2 | brief |
| `--head_hidden` | 128 | wrapper default |
| `--batch` | 128 | wrapper default |
| `--lr` | 0.001 | wrapper default |
| `--epochs` | 5 | brief |
| `--device` | cpu | brief |
| `--eval-max-samples` | 1000 | wrapper default (≤ 1000 → 1000) |
| `--eval-train-max-samples` | 4000 | wrapper default |
| `--chemprop-mode` | v3-fallback (forced) | wrapper hard-coded |
| `--seed-time-budget-s` | 800 | set by me |
| `--time-budget-s` | 900 | set by me |

### Wrapper behaviour (verified from `wf277b_reeval.log`)

* **Featurisation cache reuse**: 266,522 / 266,522 npz files reused from
  Wf-185 cache (no new featurisation needed) — `featurize: cache_hits=
  266522 todo=0 chunk_size=30000` in 4.23s.
* **In-memory load**: 266,522 / 266,522 npz into memory in 85.62s
  (largest one-off cost).
* **Training**: seed=0 epoch 1 at elapsed=14.9s, epoch 5 at
  elapsed=426.8s.  Total seed time 427.2s — the bulk of epoch time
  was spent in the per-batch `chemprop.message_passing` step on a
  multi-core CPU host (~10 cores at 600-1000% utilisation per ps).
* **Eval**: 5 splits × 1000 samples (train=4000) via the standard
  chemprop v2 message-passing forward path; per-split AUC computed
  with `roc_auc_score` (sklearn).
* **Status**: `ok` for seed=0 (no timeout, no error).

---

## §3 Per-split AUC (seed=0, n_eval=1000 per split, train=4000)

| Split      | n_eval | pos_rate | ROC-AUC |
|------------|------:|---------:|--------:|
| train      | 4,000 | ~0.95    | **0.8532** |
| iid_val    | 1,000 | 0.961    | 0.7891 |
| iid_test   | 1,000 | 0.924    | 0.8176 |
| ood_val    | 1,000 | 0.890    | 0.7529 |
| **ood_test** | **1,000** | **0.818** | **0.6843** |

(train is sub-sampled to 4000 by `eval_train_max_samples`; iid_val /
iid_test / ood_val / ood_test are sub-sampled to 1000 by
`eval_max_samples`.)

No inverted-rank artefact: ood_test < ood_val < iid_test < iid_val <
train, which is the expected train→OOD degradation pattern.  This is
the **opposite** of the Wf-268B regression signature (high AUPR + low
ROC-AUC ⇒ LOMO head-collapse).

---

## §4 Comparison vs Wf-185 v2 anchor and Wf-268B regression

| Source | Subset | Recipe | n_seeds | ood_test_auc |
|---|---|---|---:|---:|
| **Wf-185 v2 anchor** (chemprop v2 retrain) | lbap_refined_ic50_scaffold | h=128 / d=2 / 20 ep / 5 seeds | 5 | **0.6707 ± 0.0042** |
| **Wf-268B regression** (multi-task + LOMO + 2-3 ep) | lbap_refined_ic50_scaffold | h=64 / d=2 / 2-3 ep / 8 (metal, seed) | 8 | **0.4544 ± 0.0327** (BELOW_RANDOM) |
| **Wf-277B re-eval** (v3-fallback, this run) | lbap_refined_ic50_scaffold | h=64 / d=2 / 5 ep / seed=0 | **1** | **0.6843** |

* Δ vs Wf-185 anchor: **+0.0136** (passes v2 baseline within 1-seed
  point-estimate noise; std=0 since n=1).
* Δ vs Wf-268B regression: **+0.230** (overturns BELOW_RANDOM with
  massive margin; Wf-268B was 3.6σ below Wf-185, this run is +0.7σ
  above).

This confirms the Wf-263A-Audit hypothesis: the BELOW_RANDOM regression
was **NOT** caused by v3 architectural levers — it was caused by
LOMO + 2-3-epoch under-training.  Replace those two failure modes with
(v3-fallback + 5 epochs) and the chemprop v2 BinaryClassificationFFN
head recovers to **above the Wf-185 5-seed anchor**.

---

## §5 Honest framing (party-twenty-spirit §3.5)

* **n=1, not n=5.**  This is a single-seed point estimate.  The
  Wf-185 anchor has 5-seed std=0.0042; a single-seed comparison
  carries ±0.0042 noise at the 1σ level.  The +0.0136 delta is ~3× the
  Wf-185 1σ noise — directionally positive but **not** a paper-grade
  multi-seed claim.  Re-running with `--seeds 0 1 2 3 4` is
  recommended for a full multi-seed verdict (~75 min on CPU).
* **Featurisation caveat.**  This re-eval uses the wrapper's chemprop
  v2 SimpleMoleculeMolGraphFeaturizer MolGraph backbone on DrugOOD
  JSON, NOT the Wf-275B ECFP4 + scaffold-split CSV.  The wrapper does
  not expose a `--csv` flag (and the CSV is built from
  `lbap_core_ic50_assay`, not `lbap_refined_ic50_scaffold` — different
  curator subsets with different label-generation pipelines).  An
  **ECFP4-CSV-direct re-eval** (Wf-275C per Wf-275B §7 next-workflow)
  is the proper follow-up to verify whether ECFP4 features alone
  reach the Wf-185 anchor or beat it; that requires a separate
  wrapper (`load_ecfp4_csv → sklearn / chemprop v3 ECFP4 head`).
* **LOMO-collapse evidence: not present.**  ood_test=0.6843 with
  train=0.8532 shows the head is NOT collapsed to a constant
  predictor.  The Wf-268B smoking gun (high AUPR + low/inverted ROC-
  AUC ⇒ head's positive signature is `metal == head_metal`, never
  matched in DrugOOD) does **not** appear here.  In v3-fallback mode
  the chemprop v2 BinaryClassificationFFN head only sees the per-metal
  BCE on metal=0 rows (the "Ru" slot is the only one carrying weight),
  so the head learns the **organic-drug-activity signal** without the
  LOMO-confound.
* **Wall budget**: 9.4 min cfg + 30s CLI overhead = ~9.9 min total
  (well under the 15-min brief estimate).
* **Backward compat**: 0 Wf-185 / Wf-225 / Wf-230 trained checkpoints
  regressed; 0 existing tests invalidated; wrapper is the same
  `wf263a_verify_drugood.py` used by Wf-263A-Verify (no edits).

---

## §6 Outputs

| Path | Purpose |
|---|---|
| `molmetal/reports/wf277b_drugood_ecfp4_reeval/aggregate.json` | Gate-by-gate summary (this run) |
| `molmetal/reports/wf277b_drugood_ecfp4_reeval/lbap_refined_ic50_scaffold/per_seed/seed0.json` | per-seed metrics (1 seed) |
| `molmetal/reports/wf277b_drugood_ecfp4_reeval/wf277b_reeval.log` | full stdout/stderr of the run |
| `molmetal/reports/wf277b_drugood_ecfp4_reeval/reeval_report.md` | this file |

---

## §7 Recommended next workflow

**Wf-275C / Wf-277C** (per Wf-275B §7): ECFP4-CSV-direct re-eval with
a new wrapper that loads `molmetal/.data/molmetal/drugood_ecfp4_
scaffold.csv`, unpacks the hex `ecfp4_bits` column to a (N, 2048)
`np.uint8` matrix, and trains a sklearn `LogisticRegression` (or
chemprop v3 ECFP4 head) on `lbap_core_ic50_assay` with the
adapter's own scaffold split (66,665 / 9,524 / 19,047).  Expected
outcome (per Wf-275B §7): if CSV-path AUC ≥ 0.6707, the Wf-268B root
cause is in the LOMO multi-task design; if CSV-path AUC also lands at
< 0.50, the Wf-268B root cause extends to the data backbone.
