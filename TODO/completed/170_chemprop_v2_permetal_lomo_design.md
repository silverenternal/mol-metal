# TODO-170 — chemprop v2 PER-METAL single-task design for honest LOMO

**Date**: 2026-09-27
**Status**: 📐 DESIGN — code NOT yet written; this is the architectural recipe to replace the structurally-broken `ChempropV2MTModel + LeaveOneGroupOut` (Wf-146 v1) that locked 78.9% of MEASURED AUCs below random 0.5.
**Owner**: Hugo Lee + ultracode agents
**Parent**: TODO-148 (Wf-146 prep DEFERRED) + TODO-155 (chemprop v2 pivot) + TODO-164 (chemprop v2 PARTIAL ship) + TODO-168 (Wf-146 REAL ship PARTIAL — 1/25 MEASURED) + `molmetal/reports/wf146_lomo_v1_broken_design.json` (the canonical honest-negative for the broken design).
**Skill alignment**: `party-twenty-spirit` §3.5 (every claim traces to a MEASURED datapoint or a code path; AUC ceiling is structural, not tuned)

---

## §0 Verdict — why this design exists

The Wf-146 v1 design (`ChempropV2MTModel` per-metal head + sklearn `LeaveOneGroupOut`) is **structurally broken** for LOMO. Per `molmetal/reports/wf146_lomo_v1_broken_design.json`:

> "During LOMO training the held-out metal's training rows are 100% in the test fold, so that metal's head receives ZERO positive labels during training — its `y_true_all[held_out]` array is identically zero. The head's bias defaults to the prior (all-inactive), and its weights never move toward the positive class. ... MEASURED AUCs (19 runs) confirm: overall_mean=0.4458 < 0.5 random; 15/19 runs (78.9%) fall BELOW 0.5; ... Os (the rarest metal, 1118 test rows) collapses to 0.3260."

| Design | Verdict | Source |
|--------|---------|--------|
| v1 multi-task head + LOMO split | ⛔ BROKEN — structural, not a bug | `wf146_lomo_v1_broken_design.json` |
| v2 (this doc) per-metal single-task + LOMO ensemble | ✅ DESIGN — no head/metal identity leakage | this doc |
| v3 (future) shared head + metal-conditioned embedding | ⏸️ DEFERRED — needs feature-level transfer math | TODO-170 §5 forward-looking |

The v2 design replaces per-metal heads with **5 independent single-task models**, one per metal, each trained on **only that metal's data**. For honest LOMO cross-metal transfer, each held-out-metal fold evaluates on an **ensemble of the 4 trained non-held-out per-metal models** aggregated (mean of sigmoid scores) over the held-out metal's rows. Because every per-metal model is single-task and has never seen the held-out metal's labels during training, every prediction on held-out M rows is a true generalization test — no head-skew leakage.

---

## §1 Architecture — chemprop v2 single-task per metal

### §1.1 Model class — `ChempropV2PerMetalModel` (new)

**File (new)**: `molmetal/models/chemprop_v2_permetal_model.py`

**Reuses** from `ChempropV2MTModel` (TODO-164 §1.1):
- `ChempropV2Encoder` — chemprop v2 `AtomMessagePassing` + `SumAggregation`, pre-featurizes SMILES via `SimpleMoleculeMolGraphFeaturizer`, returns `(B, hidden_dim)` graph embeddings.
- `_coord_label_for(metal_name)` — deterministic metal → coordination-geometry class id.

**NEW components**:
- `ChempropV2PerMetalModel(metal_name, hidden_dim=64, depth=2, dropout=0.2, n_layers_ffn=2)` — wraps `ChempropV2Encoder` + one `BinaryClassificationFFN(n_tasks=1, n_layers=2)` + optional coord `MLP`. **No metal embedding**; the metal identity IS the model (one model per metal).
- `forward(smiles_list) -> (B, 2)` returns `(B, 2)` logits where column 0 is the chemprop BCE logit and column 1 is a zero placeholder (matches V4 2-class shim).
- `train_step(smiles_list, targets) -> loss` uses chemprop v2 canonical `BCELoss` + `BinaryClassificationFFN.train_step` route (NOT multi-task averaging).
- Constrained to a single `metal_name ∈ SUPPORTED_METALS = ("Ru", "Ir", "Os", "Rh", "Re")`.

**Why this is **honest** in-domain**: each model only sees its own metal's positives (per-metal pos rate from `molmetal/data/metal_cytotox.py`:

| Metal | n | pos_rate | MEASURED AUC (TODO-154 §0 in-domain) |
|-------|--:|---------:|--------------------------------------:|
| Ru | 19,135 | 0.2956 | 0.91 (in-domain, NOT LOMO) |
| Ir | 4,546 | 0.4916 | TBD |
| Os | 1,118 | 0.4893 | TBD |
| Rh | 1,134 | 0.4683 | TBD |
| Re | 868 | 0.2800 | TBD |

In-domain chemprop v2 Ru AUC = 0.91 was MEASURED (TODO-154 §0, paired 14 runs vs V4, hidden=300/depth=3). The per-metal single-task design should reproduce this per metal given enough epochs and backbone capacity.

### §1.2 Training loop — `ChempropV2PerMetalTrainer`

**File (new)**: `molmetal/scripts/train_chemprop_v2_permetal.py`

A single CLI entry-point that, given `--metal X --seed N`:
1. Loads `molmetal/data/metal_cytotox.py::load_metal_cytotox()`.
2. Filters to `df.metal == X` (e.g., X = "Ru" → 19,135 rows).
3. Splits 80/20 (stratified by `active`) for in-domain validation; the 20% holdout lets us verify the in-domain AUC ceiling.
4. Pre-featurizes all SMILES via `SimpleMoleculeMolGraphFeaturizer` (one-shot startup, ~22s for the full 26,801-SMILES MetalCytoToxDB subset; for a single metal it's proportional to `n_metal / 26801` × 22s).
5. Builds `ChempropV2PerMetalModel(metal_name=X, hidden_dim=64, depth=2)`.
6. Trains `epochs=50` with `AdamW(lr=1e-3)`, `batch_size=64`, BF16 autocast on the encoder forward only (heads stay fp32 for BCE stability).
7. Logs in-domain AUC per epoch; saves `molmetal/checkpoints/wf170_permetal/{metal}_seed{N}.pt` and `molmetal/reports/wf170_permetal/{metal}_seed{N}.json` with `{metal, seed, n_train, n_val, in_domain_test_auc, elapsed_s, epoch_count, status, ckpt_path}`.

**No coordination-geometry auxiliary head** in the per-metal single-task model — coord CE was a multi-task scaffold assumption; the per-metal model collapses to a single binary head with no per-metal label mixing.

### §1.3 LOMO evaluation — `ChempropV2PerMetalLOMOEvaluator`

**File (new)**: `molmetal/scripts/eval_chemprop_v2_permetal_lomo.py`

For a given held-out metal `M` and seed `N`:
1. Load the **4 trained per-metal ckpts** at `molmetal/checkpoints/wf170_permetal/{X}_seed{N}.pt` for `X ∈ SUPPORTED_METALS \ {M}`.
2. Pre-featurize held-out metal M's SMILES (~22s × `1,134 / 26,801` ≈ 1s for Rh).
3. For each row of held-out M's data:
   - Run all 4 per-metal models (one per non-held-out metal) on the SMILES → 4 sigmoid scores.
   - Aggregate: **mean of 4 sigmoid scores** as the LOMO ensemble prediction (the cheapest aggregation; alternative is max / median / weighted-by-active-rate, but mean is the natural honest default since each model contributes equally).
4. Compute test ROC-AUC on held-out M: `roc_auc_score(y_true_held_out, mean_ensemble_pred)`.
5. Save per-fold JSON: `molmetal/reports/wf170_permetal_lomo/{M}_seed{N}.json` with `{held_out: M, seed: N, n_train_per_model: dict, n_test: int, ensembled_test_auc: float, per_model_test_auc: dict, aggregation: "mean"}`.

**Why mean aggregation**:
- Each per-metal model is trained on disjoint data (different metal's positives + negatives).
- The held-out metal's row (a complex with metal `M`) is OOD for all 4 trained models — the ensemble averages out per-metal-specific over-confidence, giving a more conservative cross-metal generalization estimate.
- Per-model AUCs are also saved so we can decompose: e.g., "Ru-trained-model on Re data" vs "Ir-trained-model on Re data" — these are honest single-source generalization tests.

---

## §2 Honest framing — what this design PROVES and what it DOESN'T

### §2.1 What this design proves (MEASURED-OUTPUT)

For each metal `M`, the per-metal single-task model gives:
1. **In-domain test AUC** — clean benchmark of chemprop v2 on metal M's own train/val split. No inter-metal contamination.
2. **Cross-metal single-source AUC** — for `M != X`, the per-metal model trained on metal X evaluated on metal M's data: how well does "trained on Pt/Ru/Ir" generalize to "tested on Re"?
3. **Cross-metal ensemble LOMO AUC** — mean of 4 single-source AUCs on held-out M's data: honest cross-metal generalization score per LOMO fold per seed.
4. **Per-(metal, seed) test_auc** — full 5 metals × 5 seeds × 50 epochs = 25 in-domain + 25 LOMO entries, all derived from independent per-metal model training runs.

### §2.2 What this design DOES NOT prove (honest limitations)

| Claim | Status | Why |
|-------|--------|-----|
| "chemprop v2 beats V4 D-MPNN on Ru" | ⚠️ PARTIAL — only TODO-154 §0 MEASURED (paired 14 runs, in-domain Ru) | Per-metal in-domain benchmark will replicate TODO-154 on all 5 metals; LOMO is a new metric, NOT directly comparable to the V4 D-MPNN LOMO MEASURED in `metal_lomo_auc.json = 0.5346 ± 0.1096` (DMPNN, per-metal heads × 3 seeds × 5 folds @ 30 epochs). |
| "Cross-metal generalization is good" | ⏸️ TBD — depends on the AUC ceiling | If LOMO AUC ≥ 0.6, cross-metal signal exists; <0.55 means cross-metal transfer is near-random — a MEASURED honest-negative that maps to "metal identity is a strong inductive bias" (no shared chemistry between Ru(II) and Re(VII), for example). |
| "Coord geometry matters" | ⛔ NOT MEASURED in this design | Removed coord head to keep per-metal training interpretable as pure binary classification; coord info lives in the per-metal model's training data (every Ru row IS square-planar or octahedral depending on oxidation state, but the label we predict is binary active/inactive). |
| "Larger backbone helps" | ⏸️ NOT TESTED | Default hidden_dim=64 / depth=2 keeps the sweep under 60 min wall (matching TODO-168 §1.3 measured throughput); production hidden_dim=300/depth=3 would inflate wall time 4-8× per the Wf-124 P3 0.193 min/run baseline × 50 epoch projection. **A separate production-grade retrain would be a follow-up.** |

### §2.3 Limitations relative to the v1 broken design

The v1 (broken) design with per-metal heads CANNOT produce any of the 3 honest-evaluation artifacts in §2.1 — only the 4th (per-(metal, seed) test_auc, all capped below random). The v2 design produces all 4. The trade-off is compute cost: v2 trains **5×5 = 25 separate per-metal models per seed = 125 models per 5-seed sweep** vs v1's 25 multi-task model ckpts. Net compute is dominated by the largest per-metal training run (Ru with 19,135 rows, est. 2-3 min at hidden=64/depth=2); other per-metal runs scale linearly with `n_metal`.

---

## §3 Compute cost — MEASURED-driven estimates

### §3.1 Per-metal single-task model training (hidden_dim=64/depth=2, 50 epochs, RTX 5090)

Calibration datapoints (MEASURED from `molmetal/reports/wf146_chemprop_lomo/{metal}_seed*.json` elapsed_s; not a per-metal training time, but it has the same chemprop v2 backbone cost):

| Fold (held-out metal) | n_train | n_test | elapsed/seed (range) | Source |
|-----------------------|--------:|-------:|---------------------:|--------|
| Ru (uses 4 non-Ru) | 7,666 | 19,135 | 41.88 – 46.95 s | `Ru_seed{0..4}.json` |
| Ir (uses 4 non-Ir) | 22,255 | 4,546 | 120.49 – 190.63 s | `Ir_seed{0..4}.json` |
| Os (uses 4 non-Os) | 25,683 | 1,118 | 175.74 – 224.73 s | `Os_seed{0..3}.json` |
| Rh (uses 4 non-Rh) | 25,667 | 1,134 | 145.66 – 157.69 s | `Rh_seed{0..4}.json` |

**Key observation**: throughput per chemprop v2 epoch roughly proportional to `(n_train × 50 epochs)`. The v2 per-metal design trains on **only that metal's rows** (not 4-metal-combined), so per-metal training time is roughly:

```
per_metal_train_time (s) ≈ measured_per_epoch_cost × 50 × (n_metal / 26801) × 50  # ×50 for v1 n_metal vs n_train comparison
```

But more directly, the v2 per-metal training time at hidden=64/depth/2 with 50 epochs ~ the v1 elapsed time × (n_metal / n_train_v1). Using the Rh fold (n_train=25,667, elapsed=150s) as the backbone-cost-per-sample unit:

| Metal | n_metal | Projected per-metal training (50 epoch) | Source of n_metal |
|-------|--------:|----------------------------------------:|-------------------|
| Ru | 19,135 | 150s × (19,135/25,667) ≈ **112 s ≈ 1.9 min** | `molmetal/data/metal_cytotox.py` |
| Ir | 4,546 | 150s × (4,546/25,667) ≈ **27 s ≈ 0.4 min** | same |
| Os | 1,118 | 150s × (1,118/25,667) ≈ **7 s ≈ 0.1 min** | same |
| Rh | 1,134 | 150s × (1,134/25,667) ≈ **7 s ≈ 0.1 min** | same |
| Re | 868 | 150s × (868/25,667) ≈ **5 s ≈ 0.1 min** | same |
| **Per seed sum (5 models)** | — | **~158 s = 2.6 min** | — |

**5 seeds × 5 models = 25 per-metal training runs**:

- **Per-seed total ≈ 2.6 min** (the bulk is Ru's 19K rows; other metals are tiny).
- **5 seeds in series ≈ 13 min wall** (5 × 2.6).
- **5 seeds in parallel (1 GPU per model — RTX 5090 has 32 GB, can host at least 2-3 in parallel) ≈ 3-5 min**.

### §3.2 LOMO evaluation cost

Per-seed, after the 5 per-metal models are trained:
- For each held-out metal M ∈ 5: load 4 ckpts, featurize M's SMILES (~1s for Re, ~22s for Ru), batch through 4 models → ~5-10s per fold on GPU.
- 5 folds × ~10s ≈ **50 s per seed for LOMO eval**.
- 5 seeds × 50 s ≈ **4 min wall for full LOMO eval**.

### §3.3 Total wall-time for full honest-LOMO sweep (5 × 5 × 50, hidden=64/depth=2)

| Phase | Wall time (serial) | Wall time (parallel, 3-way GPU) |
|-------|-------------------:|---------------------------------:|
| Per-metal training (25 runs) | ~13 min | ~5 min |
| LOMO evaluation (5×5 = 25 fold-eval) | ~4 min | ~2 min |
| Aggregation, JSON write, report gen | ~30 s | ~30 s |
| **TOTAL** | **~17 min** | **~7 min** |

### §3.4 Production-grade upgrade (hidden_dim=300/depth=3)

Per TODO-154 §3 MEASURED baseline: production chemprop v2 at hidden=300/depth/3 is **5-7× slower per epoch** than hidden=64/depth/2. Apply ×7:

| Configuration | Per seed total | 5 seeds serial | Notes |
|---------------|---------------:|---------------:|-------|
| hidden=64/depth=2 (smoke-LOMO) | 2.6 min | 13 min | MEASURED-driven projection |
| hidden=300/depth=3 (production) | 18 min | **90 min ≈ 1.5 hr** | ×7 throughput factor from §3.1 → estimate |
| hidden=300/depth=3 + coord head | 22 min | **110 min ≈ 1.8 hr** | coord MLP +0.5ms/batch overhead |
| hidden=300/depth=3 + 50 epoch + coord + Noam scheduler | 25 min | **125 min ≈ 2 hr** | matches the Wf-146 budget |

**Recommendation**: ship the LOMO sweep at **hidden=64/depth/2 (smoke-LOMO config first)** to get honest AUC numbers, then optionally upgrade to hidden=300/depth/3 (production config) in a follow-up sweep if §4 results warrant the extra 5× wall-time spend.

### §3.5 LOMO 5×5×50 ETA at smoke config

- **Total wall time for 5×5×50 honest LOMO sweep (hidden=64/depth/2) = ~17 min serial, ~7 min 3-way parallel on RTX 5090**.
- A v1 Wf-146 partial sweep took ~65 min for 19 runs of the broken design; v2 takes ~17 min for 25 honest runs — **faster** AND honest-positive.

---

## §4 Expected outcomes (modeled predictions, not measurement)

### §4.1 In-domain per-metal AUC predictions

Based on TODO-154 §0 14 paired runs where in-domain chemprop v2 Ru AUC = 0.91 (mean across paired runs vs V4, MEASURED), and the smaller backbone (`hidden=64 vs 300`, `depth=2 vs 3`) reducing capacity ~5×, my honest **MODEL** prediction (NOT a measurement) for the v2 per-metal in-domain AUC ceiling:

| Metal | n | pos_rate | Predicted in-domain AUC ceiling | Reasoning |
|-------|--:|---------:|------------------------------:|-----------|
| Ru | 19,135 | 0.296 | 0.85-0.91 (matches TODO-154) | Large n, large backbone lift |
| Ir | 4,546 | 0.492 | 0.65-0.75 | Medium n, near-balanced; chemprop v2 should hit this |
| Os | 1,118 | 0.489 | 0.55-0.65 | Small n, near-random may dominate |
| Rh | 1,134 | 0.468 | 0.55-0.70 | Small n, but Wf-146 v1 showed Rh had the highest *LOMO* AUC |
| Re | 868 | 0.280 | 0.55-0.65 | Smallest n, hardest |

### §4.2 Cross-metal LOMO AUC predictions

The 4-metal-ensemble aggregation (§1.3) is honest cross-metal generalization. My MODEL prediction for the ensemble LOMO AUC per held-out metal:

| Held-out metal | Predicted ensemble LOMO AUC | Reasoning |
|----------------|----------------------------:|-----------|
| Re | 0.55-0.65 | Smallest held-out, easiest for ensemble; Re chemistry is similar to Ru(II) octahedral |
| Rh | 0.50-0.60 | Wf-146 v1 partial showed Rh had highest per-metal LOMO AUC (0.5220 mean across 5 seeds) |
| Os | 0.45-0.55 | Mid-frequency |
| Ir | 0.50-0.60 | Mid-frequency |
| Ru | 0.50-0.65 | Held-out metal with MOST data; v1 partial showed Ru LOMO around 0.45-0.47 |

These are predictions, not measurements. If measured values come in ±0.05 of these ranges, the per-metal single-task + ensemble LOMO design is HONEST. If measured values are systematically lower (e.g., Re LOMO AUC < 0.5), the honest-negative conclusion is "cross-metal transfer does not generalize beyond the shared chemprop features" — that's a publishable honest-negative finding for paper §6 (limitations).

### §4.3 Verdict mapping

| Measurement | Verdict | Reason |
|-------------|---------|--------|
| Per-metal in-domain AUC ≥ 0.7 for ≥3 of 5 metals | ✅ Positive — chemprop v2 single-task works in-domain | Hypothesis confirmed |
| Per-metal in-domain AUC ≥ 0.7 for ALL 5 metals | ✅ Strong-positive | Even the small-n metals work |
| Cross-metal ensemble LOMO AUC ≥ 0.6 for ≥3 of 5 held-out | ✅ Positive — some cross-metal signal exists | Hypothesis confirmed |
| Cross-metal ensemble LOMO AUC ≈ 0.5 for all 5 held-out | ⚠️ Honest-negative — cross-metal transfer is near-random | Maps to "metal identity is a strong inductive bias, shared chemprop features don't transfer"; paper §6 limitation |
| Cross-metal ensemble LOMO AUC < 0.5 for ≥3 of 5 held-out | ⛔ Strong-negative — ensemble prediction is anti-correlated | DEEP honest-negative: even the ensemble of "best per-metal models" can't recover the held-out metal's class signal; paper §6 + supplementary honest-negative section |

The published artifact is `metrics/by_metric/chemprop_v2_permetal_lomo.json` with the structure of TODO-154 §0 (mean ± std per metal, per (metal, seed), per LOMO fold) but **with honest framing as described in §2.2**.

---

## §5 Implementation effort — concrete recipe

### §5.1 File-by-file plan (4 new files, 1 modified, 1 new metric)

| # | Path | LOC est | Role |
|---|------|--------:|------|
| 1 | `molmetal/models/chemprop_v2_permetal_model.py` | ~120 | NEW `ChempropV2PerMetalModel(metal_name, hidden_dim, depth)`: encoder + single BinaryClassificationFFN + (B, 2) V4-shim forward + `train_step` using chemprop `BCELoss`. No multi-task scaffolding, no metal embedding. |
| 2 | `molmetal/scripts/train_chemprop_v2_permetal.py` | ~250 | NEW CLI: `--metal {Ru,Ir,Os,Rh,Re} --seed N --epochs 50 --hidden 64 --depth 2 --batch 64 --lr 1e-3 --bf16`. Pre-featurize, train, save ckpt + JSON. |
| 3 | `molmetal/scripts/eval_chemprop_v2_permetal_lomo.py` | ~200 | NEW CLI: `--held_out M --seed N`. Loads 4 trained per-metal ckpts, evaluates ensemble (mean of 4 sigmoid scores) on held-out M's test set, saves per-fold JSON. |
| 4 | `molmetal/scripts/wf170_permetal_sweep.py` | ~150 | NEW orchestrator: launches 25 per-metal train runs (5 metals × 5 seeds) then 25 LOMO eval folds (5 held-out × 5 seeds); aggregates to `metrics/by_metric/chemprop_v2_permetal_lomo.json`. |
| 5 | `molmetal/tests/test_chemprop_v2_permetal_model.py` | ~150 | NEW: 8 unit tests (instantiation per metal, forward shape, backward, single-task BCE matches chemprop canonical, V4 shim column 1 = 0, no metal embedding present, hidden_dim override, dropout config). |
| 6 | `metrics/by_metric/chemprop_v2_permetal_lomo.json` | n/a | NEW schema: `{metric, description, scope, source, gate_target, per_metal_in_domain: {Ru, Ir, Os, Rh, Re}, per_held_out_lomo_ensembled: {Ru, Ir, Os, Rh, Re}, n_runs: 25, value, value_std, verdict, honest_framing_note}` |
| 7 | `TODO/INDEX.md` | +15 lines | MODIFY: add TODO-170 cross-reference. |

**Total new LOC ≈ 870 + 1 metric schema + 1 INDEX update**.

### §5.2 Implementation order (3 phases, all gated by existing TODO-170 ↔ party-twenty-spirit)

**Phase 0 (smoke)** — ~1-2 hr coding, ~5 min wall:
- Create `molmetal/models/chemprop_v2_permetal_model.py` (file 1).
- Create `molmetal/tests/test_chemprop_v2_permetal_model.py` (file 5).
- Run `pytest molmetal/tests/test_chemprop_v2_permetal_model.py -v` — must pass.

**Phase 1 (per-metal training driver)** — ~2-3 hr coding, ~3 min wall per per-metal run on RTX 5090:
- Create `molmetal/scripts/train_chemprop_v2_permetal.py` (file 2).
- Smoke: `python ... --metal Ru --seed 0 --epochs 5` — verify ckpt + JSON save.
- Verify in-domain AUC plot: should rise from ~0.5 → 0.7+ by epoch 5 (smoke).

**Phase 2 (LOMO eval + sweep + metric)** — ~3-4 hr coding, ~17 min wall for the full 5×5×50 sweep:
- Create `molmetal/scripts/eval_chemprop_v2_permetal_lomo.py` (file 3).
- Create `molmetal/scripts/wf170_permetal_sweep.py` (file 4).
- Launch the full sweep (5 seeds in series or 3-way GPU parallel).
- Aggregate to `metrics/by_metric/chemprop_v2_permetal_lomo.json` (file 6).
- Update `TODO/INDEX.md` (file 7).

**Total implementation effort**: ~6-9 hr coding + ~17 min wall = **~7-10 hr end-to-end**.

### §5.3 Honest checkpoints

| Checkpoint | MEASURED or fabricated? |
|------------|-------------------------|
| After Phase 0: 8 unit tests pass | MEASURED (`pytest -v` exit 0) |
| After Phase 1 smoke (1 metal, 1 seed, 5 epochs): in-domain test AUC reported | MEASURED (`{metal}_seed0.json`) |
| After Phase 2 (25 per-metal trains done): per-(metal, seed) AUC matrix complete | MEASURED (`{metal}_seed*.json` × 25) |
| After Phase 2 LOMO eval: 25 (fold, seed) AUCs complete | MEASURED (`{held_out}_seed*.json` × 25) |
| After Phase 2 metric JSON write: `chemprop_v2_permetal_lomo.json` complete | MEASURED (file exists, parsed) |
| All AUC values trace to either per-metal JSON or LOMO-fold JSON | MEASURED — no extrapolation, no projection, no fabrication |

---

## §6 Recommendation

**Recommendation**: ✅ **DEFER code-writing until user fresh-authorizes a non-deferred launch** per TODO-148 §0 "kill then adapt" directive. This design doc is the architectural recipe to replace the structurally-broken Wf-146 v1; the code, tests, and sweep can be staged when the user lifts the deferral. Estimated total wall cost to ship the full honest LOMO 5×5×50 sweep at hidden=64/depth=2 is **~17 min serial or ~7 min 3-way GPU parallel** on RTX 5090 — well within a 30-min subagent budget if launched fresh.

**Concrete next steps** (gated on user fresh authorization):
1. Implement `ChempropV2PerMetalModel` (`molmetal/models/chemprop_v2_permetal_model.py`) — file 1.
2. Verify with 8 unit tests — file 5.
3. Implement per-metal training driver (`molmetal/scripts/train_chemprop_v2_permetal.py`) — file 2.
4. Smoke 1 metal × 1 seed × 5 epochs in-domain; verify AUC rise.
5. Implement LOMO evaluator (`molmetal/scripts/eval_chemprop_v2_permetal_lomo.py`) — file 3.
6. Implement sweep orchestrator (`molmetal/scripts/wf170_permetal_sweep.py`) — file 4.
7. Launch 5×5×50 sweep (3-way GPU parallel, ~7 min wall).
8. Aggregate to `metrics/by_metric/chemprop_v2_permetal_lomo.json` (file 6).
9. Update `TODO/INDEX.md` cross-reference (file 7).
10. Honest report: compare MEASURED per-metal in-domain + LOMO ensemble AUCs to the v1 broken design's per-metal-head AUCs; use the v2 design's numbers for paper §4 + §6 honest framing.

**Deferral rationale** (matches TODO-148 §0 standing "kill then adapt"):
- The current Wf-146 v1 sweep is RUNNING (PID 1382182, 17/25 done per partial snapshot) and will produce honest-negative numbers already frozen at `wf146_lomo_v1_broken_design.json`.
- Starting a parallel v2 sweep inside the Wf-146 window risks GPU contention and corrpts both sweeps' timing measurements.
- A v2 launch in the same subagent turn violates the standing "kill then adapt" directive that protects GPU work from being silently restarted without fresh user authorization.
- The user will see this design doc + the Wf-146 v1 final numbers when both complete; from there they decide whether to authorize v2.

**Honest-framing note (REQUIRED per party-twenty-spirit §3.5)**:
All predicted AUC ranges in §4 are **MODEL predictions, NOT measurements**. They are based on the existing TODO-154 §0 14-run paired head-to-head (chemprop v2 vs V4, in-domain Ru AUC = 0.91 MEASURED) and the v1 Wf-146 partial-snapshot MEASURED data (overall mean AUC = 0.4458 MEASURED for the broken design). The actual MEASURED v2 numbers will be reported only after Phase 2 completes. Until then, any number cited from this doc other than the explicitly-MEASURED ones (TODO-154 §0, wf146_lomo_v1_broken_design.json) is a PREDICTION with explicit uncertainty. This doc is honest-negative by construction — the v2 design could ALSO cap below 0.5 if cross-metal transfer truly doesn't exist beyond chemprop's generic features; the published artifact will reflect whatever is MEASURED.
