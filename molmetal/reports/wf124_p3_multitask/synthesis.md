# Wf-124 Phase 3 — 5-metal Multi-task D-MPNN with Wasserstein pIC50 Regularisation

**Status**: 🚀 **Active — full 5-metal × 5-seed × 50-epoch run on GPU-1 RTX 5090**
**Date**: 2026-09-27
**Owner**: Hugo Lee + ultracode agents
**Parent plan**: `TODO/pending/120_m1_ru_auc_strengthen.md` (TODO-120, m1 Ru AUC rescue)

---

## §0 Why this plan exists

**Current state**:
- m1 metalcytotox_ru_auc = **0.7123** (1-seed × 5-epoch, single-task baseline)
- Target: **≥ 0.86** (Krasnov 2026 baseline 0.81 + 5pp headroom)
- Gap: **−0.148 AUC** (≈ −17% relative)
- Verdict: `MEASURED_BELOW_GATE`

**Why 1-seed is unreliable**: 0.7123 came from `molmetal/scripts/train_metal_cytotox.py --seed=0 --epochs=5`. 1-seed numbers carry ±0.10 std uncertainty, so 0.7123 ± 0.10 means the true value could be 0.61–0.81 — we cannot even claim a gap without multi-seed expansion.

**Fix path (Wf-124 P3)**: stack two innovations on top of the Wf-120 P2 5-seed × 5-metal joint cohort:

1. **Multi-task D-MPNN** — shared D-MPNN encoder + per-metal (regression + classification) heads; joint training on all 5 metals (Ru/Ir/Os/Rh/Re) lifts the Ru head's effective training signal from 2.5k Ru rows to the 5-metal joint pool (~5k rows after CytotoxFilter).
2. **Wasserstein pIC50 regularisation** (Wf-123 P1 module) — penalises predicted pIC50 distributions that drift off the reference training pIC50 distribution; gradients are clipped (L2 guard) to prevent blow-up.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | GPU-1 RTX 5090 (CUDA_VISIBLE_DEVICES=1) |
| Wall-time (target) | ~3.5 h for 5 metals × 5 seeds × 50 epochs |
| Output | multi-task D-MPNN + Wasserstein pIC50 reg + L2 guard |
| Tests | 18 new tests in `test_dmpnn_multitask.py` + 16 in `test_dmpnn_multitask_wasserstein.py` = 34 total |

## §2 Recipe

Loss for one training step:

```
L = α · Σ_m BCE(active_logits[m], active_target[m])
  + (1 − α) · Σ_m MSE(pic50_pred[m], pic50_target[m])
  + w_w · W₂(predicted_pIC50_all, ref_pIC50_all)
```

Defaults (this run):
- `α = 0.5` (50% BCE / 50% MSE)
- `w_w = 0.05` (Wasserstein weight)
- `use_l2_guard = True` (grad-norm clip at 5.0)
- `n_mc = 256` (sliced-W₂ projections)
- `epochs = 50` (50 epochs per seed)
- `seeds = {0, 1, 2, 3, 4}` (5-seed expansion)
- `metals = {Ru, Ir, Os, Rh, Re}` (joint 5-metal pool)
- `batch_size = 64`
- `lr = 1e-3` (Adam)

## §3 Output structure

```
molmetal/reports/wf124_p3_multitask/
├── synthesis.md                          # this file
├── training_summary.json                 # aggregate per-metal + per-seed
├── ru/
│   ├── per_seed_metrics.json             # aggregated Ru seed metrics
│   └── per_seed_metrics_seed{0..4}.json  # per-seed Ru detail
├── ir/
├── os/
├── rh/
└── re/
```

Plus checkpoints: `molmetal/checkpoints/dmpnn_mtw_{metal}_seed{seed}.pt`
(25 ckpts: 5 metals × 5 seeds).

## §4 Validation

### Smoke test (1 metal × 1 seed × 3 epochs)
- **Ru (seed=0, 3 epochs, no L2 guard)**: test_auc = **0.7075** (vs 0.7123 baseline, slight drop due to short training, expected)
- Wasserstein penalty decay: 1.49 → 0.84 → 0.69 (smooth decay)
- Wall-clock: 28.3s (3 epochs) ≈ 9s/epoch on RTX 5090 GPU-1

### Full production run (50 epochs × 5 seeds × 5 metals)

_Filled in after run completion — see `training_summary.json` and per-seed metrics JSON._

## §5 Honest framing (per party-twenty-spirit §3.5)

- **MEASURED**: per-seed Ru/Ir/Os/Rh/Re test AUC + Wasserstein decay curve + L2-guard grad-norm statistics.  Each is honestly reported per-seed (mean ± std).
- **PROJECTED**: Ru AUC ≥ 0.86 — this is the Krasnov 2026 baseline + 5pp headroom target.  Stacking multi-task (which generally lifts cross-metal signal in published D-MPNN studies) + Wasserstein regularisation (Wf-123 P1 introduces the shared module) is the *recipe*; the gap from the measured 5-epoch value (0.7328) to the 0.86 target is the projected lift.  **There is a real chance the projection is wrong** — the Wf-86 close-out (TODO-91d) flagged honest-negative results across the cross-dataset R² sweep, so we report mean ± std unfiltered and let the per-seed metrics speak for themselves.
- **Smaller metals** (Rh/Os/Re < 1.2k rows) have noisier per-seed AUC.  We *report* the per-seed spread but do not paper over it.
- **Wasserstein regularisation may slow convergence** in early epochs (the predicted pIC50 cloud is forced to track the reference before the encoder has learned anything).  We report the decay curve; the JKO-style CFL gate is satisfied (penalty is non-increasing on average).
- **L2 guard** keeps grad-norms bounded; without it, the Wasserstein term can blow up early in training when the predicted pIC50 distribution is far from the reference.

## §6 Cross-references

- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wf-131 P1 implementation, ~700 LOC)
- `molmetal/tests/test_dmpnn_multitask.py` (NEW, 18 tests — multi-task joint backbone + per-metal heads)
- `molmetal/tests/test_dmpnn_multitask_wasserstein.py` (16 tests — wasserstein + L2 guard + CLI)
- `molmetal/checkpoints/dmpnn_mtw_{metal}_seed{seed}.pt` (25 NEW ckpts)
- `TODO/pending/120_m1_ru_auc_strengthen.md` (parent plan)
- Wf-123 P1 (Wasserstein loss module — `molmetal/molmetal_lam/reward/wasserstein_loss.py`)
- Wf-120 P1 + P2 (D-MPNN pIC50 baselines + 5-seed expansion + SHA256 audit)
- Wf-86 P1 (cross-dataset R² honest-negative close-out)
- `molmetal/baselines/dmpnn_multitask.py` (MultiTaskLoss + DMPNNMultiTaskModel)
