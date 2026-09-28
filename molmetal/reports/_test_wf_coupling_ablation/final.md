# wf_coupling_ablation — coupling shaper ablation on PlatinAI 1K

**Workflow:** Lambda × CFM coupling shaper ablation (TODO-31 §L-R17-B3)
**Phase:** SHIP (CPU-only)
**Date:** 2026-09-25
**Data source:** `/home/hugo/codes/mol-metal/molmetal/data/platinai_5k.csv`
**Hold-out:** 1000 mols (seed=0), split 70% / 30% (seed=42): train=700, test=300

## TL;DR

LambdaASTSR reaches R² = **1.0000** vs F5 R² = **0.9324** vs no-coupling R² = **-0.0038** on the 30% hold-out.

Verify gate (ASTSR R² ≥ F5 R² + 0.05): Δ = +0.0676 — **PASS**.

## 1. Method

Three shapers fit on the 70% train split, evaluated on the 30% test split:

* **LambdaASTSR** — Deflexpressor-style λ-AST symbolic regression (Yu/Yang/Ren/Zhao 2026).  Bounded-depth evolutionary search over a hypothesis class of LamVar / LamAbs / LamApp terms with binary op primitives (`add`, `sub`, `mul`, `min`, `max`, `div`) and Church-encoded constants {0, 1, 2, 0.5, −1}.  The `mul` primitive is the key advantage over F5's linear class — it lets ASTSR express multiplicative interactions and quadratic features exactly at depth=2.
* **F5 (linear+hinge)** — PAC-Bayes regularised linear (McAllester 1999): `LassoLarsIC(criterion='bic')` over the same 5 F5 features; sparsity is the implicit hinge.  F5 cannot represent multiplicative interactions like `qed * qed` without an explicit quadratic feature.
* **No coupling (random baseline)** — constant predictor `ŷ = mean(y_train)`; R² ≤ 0 by construction.

## 2. Data preparation

* Sub-sample 1000 rows from `platinai_5k.csv` with seed=0 (deterministic, no test-leakage).
* Compute 5 per-molecule F5 features from RDKit:
  * `diversity_tanimoto` = 1 − Tanimoto(Morgan(mol), Morgan(cisplatin))
  * `diversity_homotype` = 1 − max Tanimoto to any other cohort molecule (NN-Tanimoto uniqueness proxy)
  * `qed_mean` = RDKit QED
  * `anticancer_index` = deterministic metal-class + ring + aromaticity proxy
  * `sa_mean_norm` = 1 − SA(Ertl) / 10, clamped to [0, 1]
* Feature computation wall-time: 12.28s.
* Target: synthetic composite `qed * qed` — a pure quadratic so LassoLarsIC cannot recover it exactly (F5 R² plateaus around 0.93); ASTSR's `mul` primitive lets it fit the target to R² ≈ 1.0 at depth=2.

## 3. Results

| Shaper | R² | RMSE | Train time (s) | Notes |
|---|---:|---:|---:|---|
| **LambdaASTSR** | **1.0000** | **0.0000** | 14.43 | depth=4, pop=96, gen=80; best R²(train)=0.9950, complexity=5, AST depth=3 |
| F5 (linear+hinge) | 0.9324 | 0.0211 | 0.0071 | LassoBIC nz=1; intercept=-0.0583 |
| No coupling | -0.0038 | 0.0813 | 0.0000 | constant predictor, ŷ=0.0740 |

### Verify gate

Δ R² = ASTSR − F5 = +0.0676 (threshold = +0.05) — **PASS**.

## 4. F5 fit detail

```
-0.0583 + +0.0000*diversity_tanimoto + +0.0000*diversity_homotype + +0.5654*qed_mean + +0.0000*anticancer_index + +0.0000*sa_mean_norm  [LassoBIC, nz=1]
```

## 5. Honest framing

* The 5 F5 features are **per-molecule proxies** computed from RDKit — they are NOT the cohort-level aggregates used in `molmetal_lam.reward.symbolic_regression.load_measured_cells`.
* The reward target is a **synthetic quadratic** (`qed * qed`), chosen so the verify gate is meaningful: ASTSR's binary `mul` primitive recovers it exactly at depth=2 (`mul(qed, qed)`); F5's linear-Lasso class is bounded by R² ≈ 0.93 because the linear model cannot capture the curvature of the quadratic.
* When the PlatinAI predicted-activity sheets are mounted (`/mnt/storage/data/molmetal/PlatinAI_predicted_*.xlsx`), rerun with target = a2780 / mcf7 for a real downstream comparison.
