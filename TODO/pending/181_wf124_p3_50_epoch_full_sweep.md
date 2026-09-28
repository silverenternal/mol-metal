# TODO 181 — Wf-124 P3 Gap: 50-epoch full sweep (current 30-epoch undertraining risk)

**Date**: 2026-09-27 (Wf-124 P3 gap analysis per `party-twenty-spirit` §3.5)
**Owner**: Hugo Lee + ultracode agent
**Status**: 🚧 P1 — IN-FLIGHT (Wf-124 P3 ran 30 epochs due to session time budget; 50-epoch full sweep needed for convergence)
**Parent**: `molmetal/reports/wf124_p3_multitask/synthesis.md` §0 + `TODO/pending/120_m1_ru_auc_strengthen.md` §3 P1.2
**Sibling**: `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (G4) + `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (G6)

---

## §1 Gap description

**Wf-124 P3 §0 honest framing** explicitly states:

> "Hardware: GPU-1 RTX 5090 (CUDA_VISIBLE_DEVICES=1). Wall-time (target): **~1.5 h for 5 metals × 3 seeds × 30 epochs (scaled from 5×5×50 to fit session time budget; full 5×5×50 would take ~5h)**"

The Wf-124 P3 run was **scaled from 50 epochs to 30 epochs** to fit session time budget. This is a known undertraining risk — the recipe may not have converged at 30 epochs, particularly given the W₂ regularization caveat ("predicted pIC50 forced to track reference before encoder has learned anything", per Wf-124 P3 §5).

**The Ru seed=0 result is +0.049 absolute lift at 30 epochs**:
- Baseline: 0.7123 (1-seed × 5-epoch, single-task)
- Wf-124 P3: 0.7613 (1-seed × 30-epoch, multi-task + W₂)
- Krasnov 2026 target: 0.86 (+0.098 absolute above baseline)

The +0.049 → +0.098 gap may close with more epochs (W₂ may need 40-50 epochs to fully kick in), OR it may plateau (the encoder has learned what it can with this architecture + data). Without the 50-epoch run, we cannot distinguish these two scenarios.

---

## §2 Current MEASURED state

| Item | Value | Source |
|---|---|---|
| Wf-124 P3 epochs | 30 | `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` |
| Per-epoch wall time | ~15s (RTX 5090 GPU-1) | Wf-124 P3 §4 |
| Total wall time (1 (metal, seed) pair) | 30 × 15s = 450s ≈ 7.5 min | Wf-124 P3 §4 |
| W₂ trajectory (Ru seed=0, 30 epochs) | 0.8990 → 0.3906 (clear JKO-style convergence, still decreasing at epoch 30) | `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` |
| Wasserstein decay (Ru seed=0, 30 epochs) | non-increasing on average | per Wf-124 P3 §5 honest caveat |
| Best epoch (Ru seed=0) | epoch 13 (val_auc-based selection) | Wf-124 P3 §4 |
| Target epochs | 50 | per `mol-metal-gpu-execution` §6 protocol |
| Full 5-metal × 3-seed × 50-epoch | ~5 hr on RTX 5090 (per Wf-124 P3 §0) | PROJECTED |

**Key observation**: the W₂ trajectory is still decreasing at epoch 30 (final 0.3906), suggesting the W₂ term has not fully converged. The 50-epoch run may produce a meaningfully different W₂ final and therefore a meaningfully different test_auc.

---

## §3 What's needed

### §3.1 50-epoch full sweep

For each of the 15 (metal, seed) pairs (per TODO-180 G4), re-run with `--epochs 50`:

1. **Launch trainer**: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py --metal {M} --seed {S} --epochs 50 --batch-size 64 --lr 1e-3 --alpha 0.5 --wass-weight 0.05 --use-l2-guard --device cuda:1`
2. **Verify checkpoint written**: `molmetal/checkpoints/dmpnn_mtw_{metal}_seed{seed}_ep50.pt` (separate from 30-epoch ckpt)
3. **Verify per-seed metrics**: `molmetal/reports/wf124_p3_multitask_50ep/{metal}/per_seed_metrics_seed{seed}.json`
4. **Compare 30-epoch vs 50-epoch**: per (metal, seed) diff in test_auc, val_auc, W₂ final, best_epoch

### §3.2 Convergence check

Verify per `party-twenty-spirit` §3.5 + `mol-metal-gpu-execution` §6:

- **W₂ trajectory monotonic decreasing** (JKO-style) over all 50 epochs
- **L2 guard grad-norm bounded** (clip at 5.0)
- **val_auc plateaus** within last 10 epochs (early-stopping signal)
- **test_auc mean ± std across 3 seeds per metal** (no best-seed selection)

### §3.3 Comparison report

Write `molmetal/reports/wf124_p3_50ep/30ep_vs_50ep_comparison.json`:

```json
{
  "comparison_date": "2026-09-27",
  "n_pairs": 15,
  "per_pair_comparison": {
    "(Ru, 0)": {
      "30ep_test_auc": 0.7613,
      "50ep_test_auc": ...,
      "delta_test_auc": ...,
      "30ep_w2_final": 0.3906,
      "50ep_w2_final": ...
    },
    ...
  },
  "aggregate_30ep_mean_auc": ...,
  "aggregate_50ep_mean_auc": ...,
  "aggregate_delta": ...,
  "krasnov_target_0.86_reached_30ep": true|false,
  "krasnov_target_0.86_reached_50ep": true|false,
  "conclusion": "50-epoch sweep MEASURED shows X lift over 30-epoch / no meaningful lift"
}
```

### §3.4 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: 30-epoch Ru seed=0 test_auc=0.7613, W₂ still decreasing at epoch 30
- **PROJECTED**: 50-epoch may add +0.01-0.03 absolute lift, OR may plateau
- **W₂ may need 40-50 epochs** to fully kick in (per the still-decreasing trajectory at epoch 30)
- **No fabrication**: do NOT cite 50-epoch numbers as MEASURED until the 50-epoch sweep actually completes

---

## §4 Acceptance gate

- [G5.1] All 15 (metal, seed) pairs re-run with 50 epochs (per TODO-180 G4 + this gap)
- [G5.2] 30-epoch vs 50-epoch comparison report written
- [G5.3] W₂ trajectory monotonic over 50 epochs (JKO-style verified)
- [G5.4] Per-metal 3-seed mean test_auc reported (50-epoch, no best-seed selection)
- [G5.5] If 50-epoch mean ≥ 0.86 → paper §4 LEAD-claim candidate (Krasnov 2026 + 5pp headroom); else §6 honest-negative
- [G5.6] Honest framing per `party-twenty-spirit` §3.5: undertraining risk + W₂ still decreasing caveats documented

---

## §5 ETA + dependencies

**ETA**: 1 - 2 days (5 hr wall-time on RTX 5090 + 0.5 day overhead for run + audit + comparison)

**Dependencies**:
- RTX 5090 GPU-1 (CUDA_VISIBLE_DEVICES=1)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (~700 LOC — Wf-124 P3 ship)
- `molmetal/baselines/dmpnn_multitask.py` (MultiTaskLoss + DMPNNMultiTaskModel — Wf-124 P3 ship)
- `molmetal/molmetal_lam/reward/wasserstein_loss.py` (~530 LOC — M1 module)
- TODO-180 G4 (14 PENDING (metal, seed) pairs complete first — this gap is 50-epoch extension of G4)
- `mol-metal-gpu-execution` §6 5-seed × 50-epoch protocol (waiver for 3-seed per session time budget; 50-epoch adds back some of the rigour)

**No new infrastructure required** — just re-run the existing trainer for 50 epochs vs 30 epochs.

---

## §6 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: Ru seed=0 30-epoch test_auc=0.7613; W₂ still decreasing at epoch 30 (final 0.3906); val_auc best at epoch 13 (early-stop signal)
- **PROJECTED**: 50-epoch may add +0.01-0.03 absolute test_auc lift, OR may plateau at 0.7613 — honest framing is "uncertain, needs measurement"
- **W₂ regularization caveat** (per Wf-124 P3 §5): predicted pIC50 forced to track reference before encoder has learned anything; JKO-style CFL gate satisfied but W₂ needs ~40-50 epochs to fully exercise the predicted distribution toward the reference
- **30-epoch undertraining risk** (per Wf-124 P3 §0): the session time budget forced 30-epoch scaling; the 50-epoch run is the proper scientific measurement
- **No fabrication**: do NOT cite 50-epoch numbers as MEASURED until the 50-epoch sweep actually completes
- **Per-metal std caveat**: Rh/Os/Re < 1.2k rows → per-seed std likely ±0.10; the 50-epoch run does not fix this

---

## §7 Cross-references

- `molmetal/reports/wf124_p3_multitask/synthesis.md` §0 (30-epoch scaling rationale)
- `molmetal/reports/wf124_p3_multitask/synthesis.md` §4 (Ru seed=0 MEASURED)
- `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` (30-epoch per-epoch metrics)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (~700 LOC — Wf-124 P3 ship)
- `TODO/pending/120_m1_ru_auc_strengthen.md` §3 P1.2 (parent strengthen TODO — m1 Ru AUC ≥ 0.86)
- `mol-metal-gpu-execution` §6 (5-seed × 50-epoch protocol)
- `mol-metal-gpu-execution` §7 (MEASURED vs PROJECTED tag discipline)
- `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (G4 sibling — 14 PENDING (metal, seed) pairs)
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (G6 sibling — per-innovation ablation)

---

## §8 Follow-ups

- Once G5 closed: per-metal 50-epoch aggregates enable §17.4 row 1 (m1 Ru AUC) honest re-reporting
- If 50-epoch mean ≥ 0.86: paper §4 LEAD-claim candidate
- If 50-epoch mean < 0.86: paper §6 honest-negative entry; per-innovation ablation (TODO-182 G6) is the next escalation to attribute any lift to multi-task vs W₂
- If W₂ trajectory is NOT monotonic over 50 epochs: indicates divergence or instability; L2 guard may need re-tuning

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: The 30-epoch Wf-124 P3 was scaled down from 50-epoch to fit session time budget (per Wf-124 P3 §0). The 50-epoch sweep is the proper scientific measurement — the W₂ trajectory is still decreasing at epoch 30, suggesting undertraining. Do NOT cite 30-epoch numbers as the final answer; the 50-epoch sweep may meaningfully change the m1 Ru AUC outcome.
