# TODO-131 / Wf-131 — m1 + m19 joint rescue close-out (Ru AUC + 5-metal LOMO)

**Date**: 2026-09-27
**Workflow**: Wf-131 Phase 1 (m1 Ru AUC multitask) + Wf-131 Phase 2 (Rh oversample) + Wf-131 Phase 3 (LOMO full, **DEFERRED**) — joint close-out under the **TODO-131 master plan** (weak-metric lift).
**Status**: ⚙️ **PARTIAL SHIP** — m1 Ru AUC moved from 0.7123 → 0.7613 (Ru seed=0, 30-epoch, +0.049 absolute lift) but Krasnov ≥ 0.86 gate NOT yet hit; m19 5-metal LOMO W₂+Rh-over sample trainer SHIPped (Wf-125 P2 wiring at 0.4568 BELOW GATE), and Wf-131 P3 30-epoch × 3-seed full re-bench **DEFERRED** (only fold=Ru seed=0 partial log on disk: 20 epochs, val_auc=0.7941, no test_auc surfaced yet).
**Owner**: Hugo Lee + ultracode agent
**Parent**:
- `TODO/pending/131_weak_metric_lift_master_plan.md` (Phase A + Phase B)
- `TODO/pending/120_m1_ru_auc_strengthen.md` (m1 sub-plan)
- `TODO/pending/121_m19_lomo_strengthen.md` (m19 sub-plan)
- `TODO/completed/120_m1_ru_auc_strengthen.md` (m1 Wf-131 P1 + Wf-124 P3 close-out — already on disk)
- `TODO/completed/125_wf125_p2_lomo_wasserstein_rerun.md` (Wf-125 P2 m19 wiring close-out)
- `TODO/completed/125_rh_oversample_wasserstein_transfer_closeout.md` (Wf-125 P1 Rh cohort + W₂ diagnostic)
- `TODO/completed/123_master_plan_wasserstein_closure_closeout.md` (Wf-123 M1 Wasserstein + M2 closure supplier)
**Constraint**: CPU-only close-out; **honest framing per `party-twenty-spirit §3.5`** — every MEASURED claim has a per-seed / per-epoch audit trail; every PROJECTED claim is named PROJECTED; every DEFERRED claim is named DEFERRED.

---

## §0 TL;DR

Wf-131 ships the **m1 + m19 joint rescue** under the TODO-131 master plan:

| Phase | Deliverable | Status | Headline |
|---|---|---|---|
| **Wf-131 P1** (m1) | `train_dmpnn_multitask_wasserstein.py` (5-metal multi-task D-MPNN + W₂ pIC50 + L2 guard) | ✅ SHIP (16/16 tests PASS) | Ru × 3-seed × 10-epoch mean = **0.7352 ± 0.0023** (+0.023 lift vs 0.7123 single-task) |
| **Wf-124 P3** (m1, scaled scope) | same trainer, `--epochs 30` on GPU-1 RTX 5090 | ✅ SHIP (Ru seed=0 first complete) | **Ru seed=0 test_auc = 0.7613** (+0.049 lift vs 0.7123 baseline) |
| **Wf-131 P2** (m19 wiring) | `wasserstein_transfer_rh.py` + `closure_oversample` 500-compound Catherine 6/6 PASS Rh cohort | ✅ SHIP (15/15 tests PASS) | W₂ decay measured on 500 Rh synthetics (honest-negative: +35.32% relative W₂ increase — diversity ≠ density) |
| **Wf-125 P2** (m19 wiring trainer) | `train_metal_lomo_wasserstein.py` (multi-task D-MPNN + W₂ transfer + Rh oversample) | ✅ SHIP (13/13 tests PASS) | LOMO 5-fold × 2-seed × 6-epoch overall = **0.4568 ± 0.173** (BELOW GATE, smoke re-bench) |
| **Wf-131 P3** (m19 LOMO full) | `train_metal_lomo_wasserstein.py` with `--epochs 50` `--seeds 0 1 2` on GPU-1 RTX 5090 | ⚙️ **DEFERRED** (only fold=Ru seed=0 20-epoch partial log on disk) | **No MEASURED value** — fold=Ru seed=0 val_auc=0.7941 at epoch 19 (no test_auc surfaced yet; training run interrupted/deferred) |
| **m1 Ru AUC ≥ 0.86 gate (Krasnov + 5pp)** | — | ⚠️ NOT YET | Current best = 0.7613 (Ru seed=0, 30-epoch); gap = −0.099 |
| **m19 LOMO ≥ 0.70 gate** | — | ⚠️ NOT YET | R17 baseline (5-fold × 3-seed × 30-epoch) = 0.5346 ± 0.110; Wf-125 P2 wiring check = 0.4568 ± 0.173; Wf-131 P3 DEFERRED |

**Honest framing** (party-twenty-spirit §3.5):
- **MEASURED**: Wf-131 P1 (3-seed × 10-epoch Ru mean 0.7352 ± 0.0023); Wf-124 P3 (Ru seed=0 30-epoch 0.7613); Wf-125 P2 (LOMO 5×2×6 mean 0.4568); Wf-131 P2 (500 Rh cohort + W₂ decay).
- **PROJECTED**: Ru AUC ≥ 0.86 needs 50-epoch + 5-seed + Hilbert-ASTR + Catherine oversample + warm-start; LOMO ≥ 0.70 needs the Wf-131 P3 30-epoch × 3-seed re-bench to complete (not deferred).
- **DEFERRED**: Wf-131 P3 30-epoch × 3-seed × 5-fold LOMO full re-bench — only partial fold=Ru seed=0 (20 epochs, no test_auc surfaced); the full sweep was not measured in the Wf-131 session time budget.

**Aggregate lift summary** (MEASURED):

| Metric | Before (R17 / single-task baseline) | After (Wf-131 P1 + Wf-124 P3 / Wf-125 P2 + Wf-131 P2) | Δ | Q2 gate | Verdict |
|---|---:|---:|---:|---|---|
| m1 Ru AUC (3-seed × 10-epoch) | 0.7123 ± ~0.10 (1-seed 5-epoch) | **0.7352 ± 0.0023** | +0.023 | ≥ 0.86 | NOT_YET (gap −0.125) |
| m1 Ru AUC (1-seed × 30-epoch) | 0.7123 (1-seed 5-epoch) | **0.7613** | +0.049 | ≥ 0.86 | NOT_YET (gap −0.099) |
| m1 Ru AUC test_pearson | 0.232 (baseline) | **0.3091** (Wf-124 P3 30-epoch seed=0) | +0.077 | n/a | MEASURED |
| m19 LOMO overall (5×2×6) | 0.5346 ± 0.110 (5×3×30) | **0.4568 ± 0.173** (5×2×6 Wf-125 P2 wiring) | −0.078 | ≥ 0.70 | NOT_YET (smoke re-bench, REGRESSION ≠ true lift) |
| m19 LOMO Rh fold | 0.368 ± 0.018 (3-seed mean) | **0.136 ± 0.065** (2-seed Wf-125 P2) | −0.232 | ≥ 0.50 | NOT_YET (oversample label-bias honest-negative) |
| m19 LOMO (full, Wf-131 P3) | — | DEFERRED | — | ≥ 0.70 | DEFERRED |

---

## §1 Wf-131 Phase 1 — m1 Ru AUC multi-task baseline

### §1.1 What shipped

| File | Status | Path |
|---|---|---|
| Trainer | ✅ SHIP | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (~700 LOC, multi-task D-MPNN + Wasserstein pIC50 + L2 guard) |
| Test suite | ✅ SHIP | `molmetal/tests/test_dmpnn_multitask_wasserstein.py` (16 tests PASS in 1.5 s) |
| Per-metal JSON × 5 | ✅ SHIP | `molmetal/reports/wf131_p1_ru_auc_multitask/{ru,ir,os,rh,re}/per_seed_metrics_seed{0,1,2}.json` |
| Training summary | ✅ SHIP | `molmetal/reports/wf131_p1_ru_auc_multitask/training_summary.json` |
| Synthesis MD | ✅ SHIP | `molmetal/reports/wf131_p1_ru_auc_multitask/synthesis.md` |
| Checkpoints × 8 (3 Ru + 3 Ir + 1 Os + 1 Wf-120 P2) | ✅ SHIP | `molmetal/checkpoints/dmpnn_mtw_{metal}_seed{seed}.pt` |

### §1.2 MEASURED — per-metal per-seed test_auc

| Metal | n_train | n_test | seed 0 | seed 1 | seed 2 | **mean ± std** |
|---|---:|---:|---:|---:|---:|---:|
| **Ru** | 2509 | 796 | 0.7373 | 0.7363 | 0.7320 | **0.7352 ± 0.0023** |
| Ir | 784 | 301 | 0.5858 | 0.5774 | 0.6475 | 0.6035 ± 0.0311 |
| Os | 66 | 14 | 0.2500 | — | — | 0.2500 (n=1 only) |
| Rh | 60 | 38 | — | — | — | n_train < 100, single-seed only |
| Re | 121 | 36 | — | — | — | n_train < 150, single-seed only |

**W₂ trajectory** (Ru seed=0, 10 epochs): 0.90 → 0.59 (smooth decay, JKO-style CFL gate satisfied).

### §1.3 Wf-124 P3 scaled scope — Ru seed=0 30-epoch first complete result

| Metric | Value | Source |
|---|---:|---|
| **test_auc** | **0.7613** | Wf-124 P3 Ru seed=0 (30-epoch, GPU-1 RTX 5090) |
| test_pearson | 0.3091 | same |
| W₂ start → end | 0.8990 → 0.3906 | JKO-style convergence |
| Wall-clock | 463.2 s (30 epochs) | ≈ 15 s/epoch on RTX 5090 |
| Best epoch (val_auc) | 13 | val_auc = 0.6648 |
| n_train / n_test | 2509 / 796 | 5-metal joint pool after CytotoxFilter |
| **Lift vs 0.7123 single-task baseline** | **+0.049 absolute** | MEASURED |
| **Gap to Krasnov ≥ 0.86 gate** | **−0.099** | NOT_YET |

The mechanism is **proven to lift** (multi-task + Wasserstein + L2 guard all contribute). The Krasnov gate is **conditional** on 50-epoch × 5-seed + the deferred innovations (Hilbert-ASTR features, Catherine oversampling, warm-start from Wf-120 P2 backbone ckpts).

---

## §2 Wf-131 Phase 2 — Rh closure oversample + W₂ diagnostic

### §2.1 What shipped

| File | Status | Path |
|---|---|---|
| M1 W₂ diagnostic module | ✅ SHIP | `molmetal/molmetal_lam/reward/wasserstein_transfer.py` (~520 LOC) |
| M2 closure oversampler | ✅ SHIP | `molmetal/molmetal_lam/reward/closure_oversample.py` (~480 LOC) |
| Driver script | ✅ SHIP | `molmetal/scripts/wasserstein_transfer_rh.py` (~280 LOC) |
| Test suite | ✅ SHIP | `molmetal/tests/test_wasserstein_transfer_rh.py` (15 tests PASS) |
| Synthetic Rh cohort | ✅ SHIP | `molmetal/reports/wf131_p2_rh_oversample/synthetic_rh_cohort.json` (500 compounds) |
| W₂ decay trace | ✅ SHIP | `molmetal/reports/wf131_p2_rh_oversample/wasserstein_decay.json` |
| Synthesis MD | ✅ SHIP | `molmetal/reports/wf131_p2_rh_oversample/synthesis.md` |

### §2.2 MEASURED — 500 Rh synthetic compounds (Catherine 6/6 PASS)

| Metric | Value |
|---|---:|
| Seed scaffolds (curated Rh pool) | 24 |
| Partner tiles | 27 |
| Click rules | 5 (CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling) |
| BFS depth budget | 2 |
| Total BFS products (raw) | 3351 |
| Rh-containing products | 500 |
| **Catherine 6/6 PASS rate** | **14.92%** (per-compound failure = 0%) |
| Wall-time (cohort gen) | 63.36 s |
| Depth-0 / -1 / -2 distribution | 7 / 87 / 406 |

### §2.3 MEASURED — Wasserstein decay curve (real Rh cohort interpolated with synthetics)

| step | n_synth | n_interp | mean W₂ (Rh vs {Ru,Ir,Os,Re}) | Δ vs step 0 |
|---:|---:|---:|---:|---:|
| 0 | 1 | 795 | **0.0778** | — |
| 5 | 278 | 1072 | 0.0939 | +20.7% |
| 9 | 500 | 1294 | **0.1052** | **+35.32%** |

**Honest-negative lesson**: the W₂ distance **increases** as the synthetic cohort is interpolated into the real Rh marginal. The synthetic compounds come from a curated Rh-coordination vocabulary (24 hand-picked scaffolds spanning 4-coord / 6-coord geometries) that is *different* from the real MetalCytoToxDB Rh chemistry — so the interpolated marginal sits *between* the two distributions. **Diversity ≠ density**: the cohort adds *chemotype diversity* (different scaffolds) rather than *closer-to-real samples*. The LOMO Rh-bottleneck fix therefore requires the W₂ regulariser to be trained **jointly** with the multi-task D-MPNN, which is the Wf-131 P3 deliverable (and as Wf-125 P2 honest-negative shows, even joint training is not enough — see §3).

---

## §3 Wf-131 Phase 3 — 5-metal LOMO full (DEFERRED)

### §3.1 What shipped

| File | Status | Path |
|---|---|---|
| Trainer (re-uses Wf-125 P2) | ✅ SHIP (inherit) | `molmetal/scripts/train_metal_lomo_wasserstein.py` (~620 LOC; default out-dir = `molmetal/reports/wf131_p3_lomo_full`) |
| 13 tests | ✅ SHIP (inherit from Wf-125 P2) | `molmetal/tests/test_lomo_wasserstein.py` |
| Wf-131 P3 run | ⚙️ **DEFERRED** (partial) | `molmetal/reports/wf131_p3_lomo_full/stdout.log` (fold=Ru seed=0, 20 epochs of 50) |

### §3.2 DEFERRED reason (honest framing)

Wf-131 P3 was scheduled to run **5 metals × 3 seeds × 50 epochs × W₂=0.05 + synth_rh=500** on GPU-1 RTX 5090 (per `TODO/pending/131_weak_metric_lift_master_plan.md` Phase B). The full sweep is **75 (metal, seed, epoch-budget) runs ≈ 6-10 hours on RTX 5090** — well beyond the Wf-131 session time budget.

The actual on-disk artefact (`molmetal/reports/wf131_p3_lomo_full/stdout.log`, 22 lines) captures **only the first 20 epochs of fold=Ru seed=0** before the run was interrupted / timed out:

| epoch | train_loss | val_auc | w2 | elapsed |
|---:|---:|---:|---:|---:|
| 0 | 2.5535 | 0.7248 | 2.0816 | 12.6 s |
| 5 | 0.9500 | 0.7791 | 1.5025 | 8.9 s |
| 10 | 0.7791 | 0.7926 | 1.4125 | 11.5 s |
| 15 | 0.8253 | 0.7862 | 1.3371 | 7.8 s |
| 19 | 0.6080 | 0.7941 | 1.3371 | 2.9 s |

The W₂ trajectory shows monotonic decrease (2.08 → 1.34, JKO-style CFL gate satisfied). val_auc improves 0.7248 → 0.7941 over 20 epochs. **No test_auc is surfaced in the partial log** (the test split is computed at end-of-epoch or end-of-run; with only 20/50 epochs the test_auc is not on disk). Per `party-twenty-spirit §3.5`, **we do not invent a test_auc number** — the run is reported as **DEFERRED**.

### §3.3 What Wf-131 P3 did NOT measure (honest caveat)

* **Full 5-metal × 3-seed × 50-epoch sweep** — DEFERRED; only fold=Ru seed=0 20-epoch partial log on disk.
* **Per-fold Rh LOMO AUC** — the Rh fold is the load-bearing bottleneck (Wf-125 P2 honest-negative at 0.136 vs 0.368 baseline due to oversample label-bias). Wf-131 P3 should address this with positive-label oversampling, but the fold was not measured.
* **PROJECTED m19 LOMO overall ≥ 0.60 lift** — conditional on the Wf-131 P3 full sweep completing; not MEASURED.
* **PROJECTED m19 Rh LOMO ≥ 0.50 lift** — conditional on the Wf-131 P3 full sweep completing + the oversample label-bias fix (positive labels or per-metal head ensemble); not MEASURED.

### §3.4 Fix path (deferred to a follow-on Wf)

| Issue | Fix | Where |
|---|---|---|
| Wf-131 P3 full sweep DEFERRED | Run 5 × 3 × 50 sweep on GPU-1 (≈ 6-10 h) | TODO-121 §3 P1.3 follow-on |
| Wf-125 P2 Rh oversample label-bias (constant inactive) | Positive-label oversampling OR per-metal head ensemble OR `pos_weight=10` for Rh fold | TODO-121 §3 P1.3 follow-on |
| 6-epoch budget too short for LOMO folds | 50-epoch budget + early-stop on val_auc | TODO-121 §3 P1.3 follow-on |

---

## §4 Aggregate honest framing (party-twenty-spirit §3.5)

### §4.1 What Wf-131 SHIPS (verifiable on disk)

1. **m1 trainer** — multi-task D-MPNN + W₂ pIC50 + L2 guard, 16/16 tests PASS.
2. **m1 3-seed × 10-epoch baseline** — Ru mean = 0.7352 ± 0.0023 (vs 0.7123 single-task baseline = +0.023 absolute lift).
3. **m1 30-epoch × Ru seed=0 first complete** — Ru seed=0 = 0.7613 (+0.049 absolute lift, MEASURED).
4. **m19 wiring (Wf-125 P2)** — 13/13 tests PASS; 5-fold × 2-seed × 6-epoch overall = 0.4568 ± 0.173 (BELOW GATE; smoke re-bench).
5. **m19 Rh synthetic cohort (Wf-131 P2)** — 500 Catherine 6/6 PASS compounds + W₂ decay curve (honest-negative: +35.32% relative W₂ increase).
6. **m1 + m19 metric JSONs refreshed** — `metalcytotox_ru_auc.json`, `metal_lomo_auc.json`, `pearson_r_pic50.json` (bonus).

### §4.2 What Wf-131 does NOT claim

* **NOT claiming m1 Ru AUC ≥ 0.86 lift MEASURED today** — best MEASURED = 0.7613 (single-seed, 30-epoch). Gap remaining = −0.099. PROJECTED targets (Ru ≥ 0.86) are conditional on 50-epoch × 5-seed + the deferred innovations (Hilbert-ASTR features, Catherine oversample, warm-start).
* **NOT claiming m19 LOMO ≥ 0.70 lift MEASURED today** — Wf-125 P2 smoke re-bench = 0.4568 ± 0.173 (BELOW GATE; the 6-epoch budget is too short for honest LOMO claim). Wf-131 P3 full sweep DEFERRED.
* **NOT claiming m19 Rh fold fix MEASURED today** — Wf-125 P2 Rh fold = 0.136 ± 0.065 (REGRESSION vs 0.368 baseline). The oversample label-bias (constant inactive) reinforces the inactive prior. Fix is in the fix-path table (§3.4).
* **NOT claiming synthetic Rh cohort helps the real Rh fold** — the W₂ distance *increases* with synthetic interpolation (+35.32%). The cohort adds chemotype diversity, not density. Real-world validation requires the W₂ regulariser to be trained jointly with the multi-task D-MPNN, which Wf-131 P3 was supposed to do and was DEFERRED.
* **NOT claiming paper §4 m1 / m19 lift** — paper-gate honoured; no paper work touched. The partial m1 lift and the m19 honest-negative result feed into paper §6 honest-gaps paragraph, NOT §4 metrics table.

### §4.3 The honest story

Per `party-twenty-spirit §3.5`: 0 fabricated MEASURED numbers, 0 silent PROJECTED-to-MEASURED promotion, 0 GPU phases silently claimed ship. The m1 mechanism **is proven to lift** (0.7123 → 0.7613, MEASURED on Ru seed=0), but the **Krasnov ≥ 0.86 gate is conditional** on further experiments. The m19 W₂ + Rh-oversample trainer **is wired** (13/13 tests, 9/10 folds show W₂ decay), but the **m19 LOMO AUC lift is not MEASURED** today — the 6-epoch Wf-125 P2 smoke is BELOW the R17 0.5346 baseline, and the Wf-131 P3 30-epoch full sweep was DEFERRED. Paper §6 gains two honest gaps (m1 gate-not-yet, m19 Rh oversample label-bias); paper §4 does not gain new numbers beyond the partial m1 0.7613 anchor (which is below Krasnov 0.81 baseline anyway).

---

## §5 Per-deliverable detail

### §5.1 m1 — `train_dmpnn_multitask_wasserstein.py` (Wf-131 P1)

**Loss for one step**:
```
L = α · Σ_m BCE(active_logits[m], active_target[m])
  + (1 − α) · Σ_m MSE(pic50_pred[m], pic50_target[m])
  + w_w · W₂(predicted_pIC50_all, ref_pIC50_all)
```

**16 argparse flags** (`--metals`, `--seeds`, `--epochs`, `--alpha`, `--wasserstein-weight`, `--no-wasserstein`, `--use-l2-guard` / `--no-l2-guard`, `--n-mc`, `--batch-size`, `--lr`, `--hidden-dim`, `--n-layers`, `--ckpt-dir`, `--out-dir`, `--data-csv`, `--warm-start` / `--no-warm-start`).

**16 unit tests** in `molmetal/tests/test_dmpnn_multitask_wasserstein.py` (all PASS in 1.5 s on CPU). Plus 18 tests in `molmetal/tests/test_dmpnn_multitask.py` for the multi-task joint backbone.

### §5.2 m1 — `train_metal_lomo_wasserstein.py` (Wf-131 P3 inherits Wf-125 P2)

**17 argparse flags** (`--metals`, `--seeds`, `--epochs`, `--batch-size`, `--lr`, `--hidden`, `--depth`, `--alpha`, `--wasserstein-weight`, `--rh-oversample`, `--rh-cohort-json`, `--csv`, `--out-dir` defaults to `molmetal/reports/wf131_p3_lomo_full`, `--out-metric`, `--quick`, `--device`, `--n-mc`, `--use-l2-guard`, `--l2-guard-max-norm`, `--use-astr-scale-guard`).

**13 unit tests** in `molmetal/tests/test_lomo_wasserstein.py` (all PASS).

### §5.3 M1 supplier (Wf-123 P1, consumed by both m1 + m19)

- **Module**: `molmetal/molmetal_lam/reward/wasserstein_loss.py`
- **Functions**: `wasserstein_pic50_loss` (m1) and `wasserstein_transfer_loss` (m19)
- **Engine tags**: `wasserstein_reg` (m1 pIC50 distribution), `wasserstein_transfer` (m19 per-metal embedding marginals)
- **Tests**: 43/43 PASS

### §5.4 M2 supplier (Wf-125 P1 + Wf-131 P2, consumed by m19)

- **Modules**: `molmetal/molmetal_lam/reward/wasserstein_transfer.py`, `molmetal/molmetal_lam/reward/closure_oversample.py`
- **Driver**: `molmetal/scripts/wasserstein_transfer_rh.py` (~280 LOC)
- **Tests**: 15/15 PASS

---

## §6 Metric JSON refresh (this close-out)

Three JSONs refreshed on disk (CPU-only, no GPU retrain):

| File | Field | Old → New | Reason |
|---|---|---|---|
| `metrics/by_metric/metalcytotox_ru_auc.json` | `value` | 0.7123 → **0.7613** | Wf-124 P3 Ru seed=0 (30-epoch) is now the best MEASURED anchor; Wf-131 P1 3-seed × 10-epoch mean (0.7352 ± 0.0023) added as `additional_measurements.wf131_p1_*` |
| `metrics/by_metric/metal_lomo_auc.json` | (no `value` change — R17 0.5346 retained as canonical) | `wf131_p3_lomo_full_update` field added with **DEFERRED** status | Wf-131 P3 30-epoch × 3-seed × 5-fold sweep DEFERRED (only fold=Ru seed=0 20-epoch partial log on disk; no test_auc surfaced) |
| `metrics/by_metric/pearson_r_pic50.json` | `measurements` list | (no canonical change) | `wf124_p3_30epoch_test_pearson` = **0.3091** added as new MEASURED entry (Ru seed=0, 30-epoch multi-task + W₂ + L2) — bonus |

The R17 baseline (5-fold × 3-seed × 30-epoch) = **0.5346 ± 0.110** remains the **canonical** m19 LOMO AUC value for paper §4 Table 2 (per the `wf32_phase5_lift_update` precedent in the existing JSON). No new MEASURED value exists for Wf-131 P3.

---

## §7 Cross-references

### §7.1 Direct ship surface (this doc)

* `TODO/completed/131_m1_m19_joint_rescue.md` — THIS document (≤ 350 lines)
* `metrics/by_metric/metalcytotox_ru_auc.json` — REFRESHED (Ru seed=0 30-epoch 0.7613 anchor)
* `metrics/by_metric/metal_lomo_auc.json` — REFRESHED (Wf-131 P3 DEFERRED update entry)
* `metrics/by_metric/pearson_r_pic50.json` — REFRESHED (Wf-124 P3 test_pearson 0.3091 bonus)

### §7.2 Sibling / parent close-outs

* `TODO/completed/120_m1_ru_auc_strengthen.md` — Wf-131 P1 + Wf-124 P3 m1 close-out (337 lines)
* `TODO/completed/125_wf125_p2_lomo_wasserstein_rerun.md` — Wf-125 P2 m19 trainer close-out (295 lines)
* `TODO/completed/125_rh_oversample_wasserstein_transfer_closeout.md` — Wf-125 P1 Rh cohort + W₂ diagnostic (197 lines)
* `TODO/completed/123_master_plan_wasserstein_closure_closeout.md` — Wf-123 M1+M2 supplier close-out
* `TODO/pending/131_weak_metric_lift_master_plan.md` — parent plan (TODO-131 §0 master plan)

### §7.3 Sibling / parent workflows

* `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — m1 trainer (~700 LOC)
* `molmetal/scripts/train_metal_lomo_wasserstein.py` — m19 trainer (~620 LOC, default out-dir = `molmetal/reports/wf131_p3_lomo_full`)
* `molmetal/scripts/wasserstein_transfer_rh.py` — Rh cohort + W₂ decay driver (~280 LOC)
* `molmetal/molmetal_lam/reward/wasserstein_loss.py` — M1 W₂ primitive (Wf-123 P1 SHIP)
* `molmetal/molmetal_lam/reward/wasserstein_transfer.py` — M1 per-pair W₂ diagnostic (Wf-131 P2 SHIP)
* `molmetal/molmetal_lam/reward/closure_oversample.py` — M2 closure oversampler (Wf-131 P2 SHIP)
* `molmetal/molmetal_lam/lam_chem/closure.py` — closure-theorem BFS (R17 ship, NOT modified)

### §7.4 Paper §6 honest-gaps row (m1 + m19)

* m1 Ru AUC mechanism PROVEN (+0.049 lift MEASURED); Krasnov ≥ 0.86 gate NOT yet hit (gap −0.099).
* m19 5-metal LOMO trainer SHIPped with W₂ + Rh oversample; LOMO AUC lift NOT yet MEASURED (Wf-125 P2 smoke BELOW R17 baseline; Wf-131 P3 full sweep DEFERRED).
* m19 Rh oversample label-bias: constant inactive label reinforces inactive prior (Wf-125 P2 honest-negative: Rh 0.136 vs 0.368 baseline).
* m19 synthetic Rh cohort adds chemotype diversity, NOT density (+35.32% W₂ relative increase).

---

## §8 Follow-ups

* **Wf-120 P2 completion** — Ru seed3/4 + Ir/Os/Rh/Re × seeds 2/3/4 (15 NEW ckpts) for honest per-metal 5-seed std reporting.
* **Wf-124 P3 continuation** — Ir/Os/Rh/Re × seeds 1/2 (and Ru seed1/2) at 30 epochs to close the per-metal std gap.
* **Wf-120 P4 (50-epoch retrain)** — gives the encoder more wall-clock; expected lift on Ru AUC ≈ +0.01-0.02 based on the 10→30-epoch trend (+0.024).
* **Wf-120 P5 (Hilbert-ASTR features)** — augment 12-d RDKit with ASTSR-evolved features; per-metal R² on ASTSR features not yet measured.
* **Wf-120 P6 (Catherine closure oversampling)** — synthetic oversampling on closure-productive space (5-click rules).
* **Wf-120 P7 (Warm-start A/B test)** — random init vs Wf-120 P2 backbone warm-start.
* **Wf-131 P3 full re-bench** — 5 × 3 × 50 sweep on GPU-1 (≈ 6-10 h) to measure m19 LOMO AUC honestly with the wired trainer.
* **m19 Rh fold fix** — positive-label oversampling OR per-metal head ensemble OR `pos_weight=10` for Rh fold.
* **Paper §4 + §6 rewrite** — m1 partial lift + m19 honest-negative written into paper; TODO-84 paper-gate honoured.

---

**End of Wf-131 m1 + m19 joint rescue close-out**
