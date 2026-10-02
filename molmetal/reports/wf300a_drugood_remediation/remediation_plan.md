# Wf-300A: drugood_v3_scaffold BELOW-Field-Avg Remediation Plan

**Date**: 2026-09-29
**Owner**: Hugo Lee
**Branch**: master
**Parent insights**:
- Wf-253D `molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.md` (forward-pass honest-negative, AUC 0.4544 BELOW_RANDOM)
- Wf-286B `molmetal/reports/wf286b_ecfp4_reeval/aggregate.json` (ECFP4 axis ruled OUT as bottleneck — chemprop v2 + ECFP4 lifts to 0.8820, ruling out featurisation)
- Wf-297B `molmetal/reports/wf297b_clone_benchmark_map/benchmark_map.md` (DANN clone present at `.cloned/DANN/` Ganin ICML 2015, field-avg target 0.55–0.65)

---

## 1. Problem statement

`drugood_v3_scaffold` OOD AUC is **0.4574 (Ru seed 0)**, well BELOW the
field-avg band **0.55–0.65** reported in:
- DrugOOD (ICML 2022) Tab. 5 / Tab. 7 scaffold-OOD mean
- chemprop_v2 README baseline ≈ 0.6707 (Wf-185 scaffold split, MEASURED)

Wf-286B ruled out the ECFP4 featurisation axis (chemprop v2 + ECFP4 hits
0.8820 test AUC — recovering Wf-185 baseline within slack). Wf-285B
already wired Kendall UW + DANN GRL into
`MultiTaskWassersteinModel` (3/3 wiring tests PASS, smoke OK).
The remaining gap is therefore the **backbone + multi-task pretrain
axis** (chemprop v3 + DANN adversarial head on DrugOOD scaffold split).

---

## 2. Root-cause diagnosis

| Layer | Component | Current | Diagnosis |
|---|---|---|---|
| Featurisation | ECFP4 (Morgan r=2, 2048 bits) | OK (Wf-286B 0.8820) | not the bottleneck |
| Backbone | chemprop v2 (Wf-185 retrain) | 0.6707 | field-avg baseline |
| Backbone | chemprop v3 (Wf-253D reeval, no DrugOOD retrain) | **0.4544** | below random; v3 was trained on MetalCytoToxDB not DrugOOD |
| Multi-task pretrain | none | n/a | not implemented on DrugOOD yet (Wf-124 P3 in-flight for MetalCytoToxDB) |
| OOD adversarial head | ReverseLayerF GRL present (Wf-285B) | OFF by default; no DrugOOD domain split | wired but not used on DrugOOD |

**Root cause**: chemprop v3 weights were trained on MetalCytoToxDB; when
forwarded on DrugOOD scaffold-OOD split, the embedding distribution
shifts and the per-metal heads (Ru/Ir/Os) over-fit to MetalCytoToxDB
chemistry. Two compounding gaps:
1. **No multi-task pretrain on DrugOOD** (the in-distribution train
   split is not seen by v3 at all).
2. **No OOD-adversarial pressure** to push v3 features toward
   scaffold-invariance.

---

## 3. Remediation recipe

**Recipe**: chemprop v3 backbone + multi-task DrugOOD pretrain +
DANN GRL adversarial head + scaffold split.

### 3.1 Multi-task DrugOOD pretrain (cfm_v3 multitask, chemprop v3 weights)

- Source dataset: DrugOOD `lbap_refined_ic50` scaffold-OOD split
  (already cached in `molmetal/.data/drugood/`).
- Multi-task labels: primary `Y` (pIC50, regression) + auxiliary
  `assay` (multi-class, n≈10 assay panels).
- Pretrain: 5 seeds × 50 epoch × chemprop v3 backbone on
  `lbap_refined_ic50` train split, Wasserstein pIC50 regression
  primary head, Kendall UW task-weighting.
- Output: `checkpoints/cfm_v3_drugood_multitask/{metal}/seed{seed}.pt`
  (5-metal × 5-seed = 25 ckpts).
- Driver: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
  extended with `--dataset drugood --pretrain-domain drugood` CLI flag.

### 3.2 DANN adversarial head (Ganin ICML 2015)

- Re-use `molmetal/molmetal_lam/sbdd_env/gradient_reversal.py`
  `ReverseLayerF` (already Wf-285B wired).
- Re-use `molmetal.models.dmpnn.DomainAdversarialHead` (already Wf-285B
  wired, ON by default in `train_dmpnn_multitask_wasserstein.py`).
- Domain split: DrugOOD has implicit domain = scaffold cluster id
  (computed by Butina clustering on train molecules, k≈500 clusters,
  bucketed into 4 domain bins by quartile). Per-row `domain_id`
  threading deferred (TODO-143 §2.2 — same blocker as Wf-285B §7
  follow-up #1; for Wf-300A we use **constant domain_id=0** to match
  Wf-285B §4.2 smoke contract).
- Hyperparameters: `grl_alpha=0.1`, `lambda_grl=0.1` (Wf-285B §2.3
  defaults; suitable for D-MPNN which is already a stable embedding).
- Optional CLI knobs from Wf-285B: `--grl-alpha`, `--lambda-grl`,
  `--num-domains`, `--domain-id`.

### 3.3 Train / eval protocol

- Train: chemprop v3 + DrugOOD multi-task pretrain + DANN ON,
  5 seeds × 50 epoch, batch=128, lr=1e-3, Wasserstein pIC50 reg,
  Kendall UW task weighting.
- Eval: forward pass on DrugOOD scaffold OOD test split,
  per-(metal, seed) ROC-AUC + AUPR + n_samples + wall_clock.
- Aggregation: `aggregate_drugood_v3_scaffold.py` → per-metal mean ± std
  + PROJECTED band [0.55, 0.65] (field-avg target).
- Pass criterion: per-metal mean AUC ≥ 0.55 (lower bound of field-avg).

### 3.4 Code surface (no NEW files; re-uses Wf-285B + Wf-124 P3)

| Component | File | Status |
|---|---|---|
| Multi-task D-MPNN trainer | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | extended with `--dataset drugood` flag (Wf-124 P3 base) |
| DANN GRL block | same file | already wired (Wf-285B §2.3) |
| DomainAdversarialHead | `molmetal/models/dmpnn.py` | already shipped (Wf-285B §2.1) |
| ReverseLayerF | `molmetal/molmetal_lam/sbdd_env/gradient_reversal.py` | already shipped (TODO-143 §2.2) |
| Kendall UW | `molmetal/baselines/dmpnn_multitask.MultiTaskLoss` | already shipped (Wf-285B §2.2) |
| Eval driver | `molmetal/scripts/eval_drugood_v3_scaffold.py` (NEW) | 60 LOC, calls forward pass + sklearn.metrics.roc_auc_score |
| Aggregation | `molmetal/scripts/aggregate_drugood_v3_scaffold.py` (NEW) | 30 LOC, JSON output to `metrics/by_metric/drugood_v3_scaffold.json` |
| Config | `configs/cfm_drugood_v3_dann.yaml` (NEW) | mirrors `cfm_full_retrain.yaml` + `--dann --use-dann` flags |

### 3.5 Honest framing

- **ASPIRATIONAL TARGETS, not projection.** The 0.55–0.65 band is from
  DrugOOD Tab. 7 / Ganin ICML 2015 — those are the **field-average
  bands** other groups reach with this combo. We have not run Wf-300A
  end-to-end yet, so we cannot claim a "projection." When we measure,
  we will cite MEASURED only and label PROJECTED-only as such.
- **No leakage.** v3 weights will be retrained on DrugOOD train split
  only; scaffold test split will be held out.
- **1-seed-only risk.** Wf-253D had uneven coverage (Ru 4 seeds, Ir 3,
  Os 1). Wf-300A will run **5 seeds × 5 metals** for full coverage
  (25 ckpts × ~5 min/ckpt = ~2 hr pretrain).
- **Constant domain_id caveat.** Per-row domain labelling is TODO-143
  §2.2 blocker; constant-0 DANN reduces to a weaker adversarial
  regulariser. We will note this as a known limitation in the report.
- **Below-random current value (0.4574).** Achieving 0.55 may require
  more than the planned wall time. We will not inflate; if MEASURED
  lands at 0.50–0.55 we will cite honestly as "partial lift, still
  below field-avg band."

---

## 4. Wall-time estimate

| Step | Wall |
|---|---|
| 3.1 Multi-task DrugOOD pretrain (5×5×50ep, RTX 5090) | ~2.5 hr |
| 3.2 DANN adversarial head wire | already shipped (Wf-285B) |
| 3.3 Eval forward pass (5×5 = 25 ckpts, CPU OK) | ~0.25 hr |
| 3.4 Aggregation + JSON write | ~0.05 hr |
| Tests (3/3 re-use Wf-285B test_kendall_dann_wiring.py) | ~0.05 hr |
| Report write-up | ~0.15 hr |
| **TOTAL** | **~3–4 hr wall (GPU-1 RTX 5090)** |

---

## 5. Pass / fail criteria

- **PASS**: per-metal mean AUC ≥ 0.55 (lower bound of field-avg band) for at least 3/5 metals.
- **PARTIAL**: per-metal mean AUC in [0.50, 0.55) for ≥ 3/5 metals → honest-negative disclosed with PARTIAL verdict.
- **FAIL**: per-metal mean AUC < 0.50 for ≥ 3/5 metals → honest-negative REMAINS, next-step is TODO-21 CFM retrain + per-row domain_id.

---

## 6. Bottom line (output for the calling agent)

- **current_value**: 0.4574 (Wf-253D, Ru seed 0; aggregate 0.4544 ± 0.0327 across 8 (metal, seed) pairs — BELOW_RANDOM)
- **field_avg**: 0.55–0.65 (DrugOOD Tab. 7 scaffold-OOD mean / Ganin ICML 2015 Tab. 2)
- **target**: ≥ 0.55 per-metal mean (lower bound of field-avg band); ≥ 0.65 = honest-positive lift
- **wall**: ~3–4 hr GPU-1 RTX 5090 (5 metal × 5 seed × 50 epoch multi-task pretrain + DANN adversarial head + scaffold-OOD eval + aggregation)
- **honest_framing**: ASPIRATIONAL TARGETS, not projection. Wf-285B already wired Kendall+DANN into the trainer; Wf-124 P3 already runs chemprop v3 multi-task on MetalCytoToxDB. Wf-300A re-uses both — extends `--dataset drugood` and adds DrugOOD scaffold-OOD eval driver. Per-row domain_id is a known limitation (TODO-143 §2.2 blocker; constant-0 used). Will not claim projection; will cite MEASURED once run.
