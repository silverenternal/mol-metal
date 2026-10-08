# TODO 180 — Wf-124 P3 Gap: 14/15 (metal, seed) pairs PENDING (only Ru seed=0 complete)

**Date**: 2026-09-27 (Wf-124 P3 gap analysis per `party-twenty-spirit` §3.5)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ CLOSED (2026-10-08 disk re-verification) — G4 14 pairs MEASURED; aggregator path-mismatch fixed in `molmetal/reports/wf278a_g4_aggregation_fix/fix_report.md` (0 → 14 pairs). This file is the original gap analysis, kept as a historical artifact. See `TODO/completed/laneD_todo_180_181_182_disposition.md` §1.
**Parent**: `TODO/pending/120_m1_ru_auc_strengthen.md` §3 P1.2 + `molmetal/reports/wf124_p3_multitask/synthesis.md` §4
**Sibling**: `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` (G5) + `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (G6)

---

## §1 Gap description

**Wf-124 P3 §4 honest framing** explicitly states:

> "Further (metal, seed) results will populate `ru/per_seed_metrics_seed{1,2}.json`, `ir/...`, `os/...`, `rh/...`, `re/...` as the run progresses (each ≈ 6-8 min wall-clock). See `training_summary.json` for the full per-metal aggregate."

**Currently complete**: 1 / 15 (metal, seed) pairs:
- ✅ Ru seed=0 (30 epochs, L2 guard ON): test_auc = **0.7613**, val_auc = 0.6648, W₂ decay 0.8990 → 0.3906

**PENDING 14 / 15**:
- Ru seed=1, Ru seed=2 (2)
- Ir seed=0, Ir seed=1, Ir seed=2 (3)
- Os seed=0, Os seed=1, Os seed=2 (3)
- Rh seed=0, Rh seed=1, Rh seed=2 (3)
- Re seed=0, Re seed=1, Re seed=2 (3)

**Total wall-time**: 14 × ~7.7 min (mean) ≈ 1.8 hr on RTX 5090 GPU-1

---

## §2 Current MEASURED state

| (metal, seed) | test_auc | val_auc | W₂ decay | W₂ final | grad-norm clip | Status |
|---|---:|---:|---:|---:|---|---|
| (Ru, 0) | **0.7613** | 0.6648 | 0.8990 → 0.3906 | 0.3906 | OK | ✅ DONE |
| (Ru, 1) | — | — | — | — | — | ⏳ PENDING |
| (Ru, 2) | — | — | — | — | — | ⏳ PENDING |
| (Ir, 0) | — | — | — | — | — | ⏳ PENDING |
| (Ir, 1) | — | — | — | — | — | ⏳ PENDING |
| (Ir, 2) | — | — | — | — | — | ⏳ PENDING |
| (Os, 0) | — | — | — | — | — | ⏳ PENDING |
| (Os, 1) | — | — | — | — | — | ⏳ PENDING |
| (Os, 2) | — | — | — | — | — | ⏳ PENDING |
| (Rh, 0) | — | — | — | — | — | ⏳ PENDING |
| (Rh, 1) | — | — | — | — | — | ⏳ PENDING |
| (Rh, 2) | — | — | — | — | — | ⏳ PENDING |
| (Re, 0) | — | — | — | — | — | ⏳ PENDING |
| (Re, 1) | — | — | — | — | — | ⏳ PENDING |
| (Re, 2) | — | — | — | — | — | ⏳ PENDING |

**Per-seed Wf-124 P3 wall-time**: ~6-8 min per (metal, seed) on RTX 5090 GPU-1 → 14 × 7 min ≈ 1.6-1.8 hr

**Smaller-metal caveat** (per Wf-124 P3 §5):
- Rh/Os/Re cohorts < 1.2k rows → noisier per-seed AUC
- Per-seed std likely ±0.10 or more
- Honest framing: report per-seed mean ± std unfiltered, no best-seed selection

---

## §3 What's needed

### §3.1 Run the 14 PENDING (metal, seed) pairs

For each of the 14 PENDING pairs:

1. **Launch trainer**: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py --metal {M} --seed {S} --epochs 30 --batch-size 64 --lr 1e-3 --alpha 0.5 --wass-weight 0.05 --use-l2-guard --device cuda:1`
2. **Verify checkpoint written**: `molmetal/checkpoints/dmpnn_mtw_{metal}_seed{seed}.pt`
3. **Verify per-seed metrics**: `molmetal/reports/wf124_p3_multitask/{metal}/per_seed_metrics_seed{seed}.json`
4. **Verify W₂ decay**: `wasserstein_decay.json` per (metal, seed)
5. **Verify no grad-norm blow-up**: grad-norm trajectory stays bounded (L2 clip at 5.0)

### §3.2 Update training_summary.json

After all 15 (metal, seed) pairs complete, regenerate `molmetal/reports/wf124_p3_multitask/training_summary.json` with:

```json
{
  "metals": ["Ru", "Ir", "Os", "Rh", "Re"],
  "seeds": [0, 1, 2],
  "per_metal_seed_aggregate": {
    "Ru": {"mean_test_auc": ..., "std_test_auc": ..., "mean_w2_final": ..., "n_pairs": 3},
    "Ir": {...},
    "Os": {...},
    "Rh": {...},
    "Re": {...}
  },
  "aggregate_5_metal_3_seed": {
    "mean_test_auc": ...,
    "std_test_auc": ...,
    "n_pairs": 15
  },
  "krasnov_2026_target": 0.86,
  "ship_target_met": true|false,
  "audit_date": "2026-09-27"
}
```

### §3.3 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: Ru seed=0 test_auc = 0.7613 (+0.049 vs 0.7123 single-task baseline)
- **PROJECTED @ R20**: 5-metal × 3-seed mean test_auc ≥ 0.86 (Krasnov 2026 baseline + 5pp headroom)
- **Smaller-metal caveat**: Rh/Os/Re < 1.2k rows → per-seed std likely ±0.10; report unfiltered, no best-seed selection
- **No overclaim**: if 5-metal × 3-seed mean < 0.86, paper §6 honest-negative entry, NOT §4 LEAD

---

## §4 Acceptance gate

- [G4.1] All 14 PENDING (metal, seed) pairs complete (Ru s1/s2 + Ir s0/s1/s2 + Os s0/s1/s2 + Rh s0/s1/s2 + Re s0/s1/s2)
- [G4.2] Each pair has its own per-seed metrics JSON + W₂ decay JSON + checkpoint
- [G4.3] `training_summary.json` regenerated with per-metal 3-seed mean ± std
- [G4.4] 5-metal × 3-seed mean test_auc reported (MEASURED, no best-seed selection)
- [G4.5] If mean ≥ 0.86 → paper §4 LEAD-claim candidate; else §6 honest-negative
- [G4.6] Honest framing per `party-twenty-spirit` §3.5: smaller-metal caveat documented; no fabrication

---

## §5 ETA + dependencies

**ETA**: 1 - 2 days (1.8 hr wall-time + 0.5 day overhead for run + audit + summary)

**Dependencies**:
- RTX 5090 GPU-1 (CUDA_VISIBLE_DEVICES=1) — already used for Ru seed=0
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (~700 LOC) — Wf-124 P3 ship
- `molmetal/baselines/dmpnn_multitask.py` (MultiTaskLoss + DMPNNMultiTaskModel) — Wf-124 P3 ship
- `molmetal/data/cytotox.py` (CytotoxFilter — Wf-120 P2 ship)
- `molmetal/molmetal_lam/reward/wasserstein_loss.py` (~530 LOC — M1 module)
- 5-metal cohort (Ru, Ir, Os, Rh, Re) at `molmetal/data/cytotox_5metal_multitask.csv` (Wf-120 P2 ship)

**No new infrastructure required** — just run the existing trainer for the 14 PENDING (metal, seed) pairs.

---

## §6 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: Ru seed=0 test_auc = 0.7613 (+0.049 vs 0.7123 single-task baseline); W₂ decay 0.8990 → 0.3906 (JKO-style); L2 guard grad-norm OK
- **PROJECTED**: 5-metal × 3-seed mean test_auc ≥ 0.86 — but this is a strong claim; the Ru seed=0 lift is +0.049 absolute, and the Krasnov target is +0.098 absolute further → PROBABLY will not reach target at 30 epochs
- **Smaller-metal caveat**: Rh/Os/Re < 1.2k rows will have noisier per-seed AUC; per-seed std likely ±0.10 or more; the per-metal aggregate for Rh/Os/Re may be below gate even if Ru is at gate
- **W₂ regularization may slow early convergence** (per Wf-124 P3 §5): JKO-style CFL gate satisfied but encoder has to learn something before W₂ term kicks in
- **No best-seed selection** (per `mol-metal-gpu-execution` §7): if 3-seed mean is below gate, the per-seed std IS the answer; do not pick the highest seed and call it MEASURED
- **30-epoch undertraining concern** (per TODO-181 G5): Wf-124 P3 only ran 30 epochs due to session time budget; 50-epoch run may be needed for convergence — see G5 sibling

---

## §7 Cross-references

- `TODO/pending/120_m1_ru_auc_strengthen.md` §3 P1.2 (parent strengthen TODO — m1 Ru AUC ≥ 0.86)
- `molmetal/reports/wf124_p3_multitask/synthesis.md` §4 (Ru seed=0 ship + 14 PENDING list)
- `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` (Ru seed=0 MEASURED)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (~700 LOC — Wf-124 P3 ship)
- `molmetal/tests/test_dmpnn_multitask.py` (18 tests) + `molmetal/tests/test_dmpnn_multitask_wasserstein.py` (16 tests) = 34 tests
- `molmetal/checkpoints/dmpnn_mtw_{metal}_seed{seed}.pt` (15 ckpts target: 5 metals × 3 seeds)
- `TODO/completed/120_dmpnn_seed_integrity.md` (Wf-120 P2 close-out — 5-metal joint cohort)
- `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` (G5 sibling — 50-epoch full sweep)
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (G6 sibling — per-innovation ablation)

---

## §8 Follow-ups

- Once G4 closed: per-metal 3-seed aggregates enable §17.4 row 1 (m1 Ru AUC) honest re-reporting
- If 5-metal × 3-seed mean ≥ 0.86: paper §4 LEAD-claim candidate (Krasnov 2026 + 5pp headroom)
- If < 0.86: paper §6 honest-negative entry; G5 (TODO-181) 50-epoch sweep is the next escalation
- If per-metal std is too high (Rh/Os/Re): consider per-metal hyperparameter tuning (Wf-120 P3 follow-up)

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: Ru seed=0 test_auc=0.7613 MEASURED is +0.049 above the 0.7123 single-task baseline. The Krasnov 2026 0.86 target is +0.098 above the baseline. The +0.049 → +0.098 gap remaining requires the other 14 (metal, seed) pairs to also show similar or larger lifts, which is uncertain — do NOT project "5-metal × 3-seed mean ≥ 0.86" as MEASURED until all 15 pairs complete and the aggregate is computed honestly.
