# TODO-120 — m1 Ru AUC strengthen via multi-task D-MPNN + Wasserstein regularization

**Date**: 2026-09-27
**Status**: 🚀 **Active — closes m1 Ru AUC 0.7123 → ≥ 0.86 (Krasnov baseline) gap**
**Priority**: **P0 CRITICAL** (m1 is the canonical precious-metal-anticancer claim)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Honest model-state assessment (2026-09-27)
**Skill alignment**: `mol-metal-cfm-training` + `mol-metal-r18-math`

---

## §0 Why this plan exists

**Current state**:
- m1 metalcytotox_ru_auc = **0.7123** (1-seed only, 5 epochs)
- Target: ≥ 0.86 (Krasnov 2026 baseline 0.81 + 5 pp headroom)
- Gap: **−0.148 AUC** (≈ −17% relative)
- Verdict: `MEASURED_BELOW_GATE`

**Why 1-seed is unreliable**: 0.7123 came from `molmetal/scripts/train_metal_cytotox.py --seed=0 --epochs=5`. The 1-seed number carries ±0.10 std uncertainty, so 0.7123 ± 0.10 means the true value could be anywhere from 0.61 to 0.81. **We can't even claim a gap** without 5-seed expansion (which TODO-120 / Wf-120 is in flight on).

**Why below 0.86**: single-task D-MPNN on Ru subset doesn't leverage:
- Joint signal across Ir/Os/Rh/Re (multi-task)
- Wasserstein gradient flow regularization (theorem T1)
- Hilbert-ASTR combined feature stack
- ASTSR-evolved features beyond raw RDKit 12-d

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | GPU-1 RTX 5090 |
| Wall-time | 1 week GPU-1 |
| Output | multi-task D-MPNN pIC50 + Wasserstein regularization |
| Tests | ~25 new tests |

## §2 Solution path (4 innovations stacked)

1. **Multi-task D-MPNN** — joint Ru/Ir/Os/Rh/Re training with shared backbone + per-metal heads
2. **Wasserstein gradient flow regularization** (`wasserstein_gradient_flow.py`) — penalize off-manifold pIC50 predictions
3. **Hilbert-ASTR combined feature stack** — augment 12-d RDKit with ASTSR-evolved features
4. **Catherine closure-bounded augmentation** — synthetic oversampling on closure-productive space (5-click rules)

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Multi-task D-MPNN (Ru/Ir/Os/Rh/Re joint) | 2 days GPU-1 | `molmetal/scripts/train_dmpnn_multitask.py` (NEW, ~300 LOC) + 12 tests | per-metal AUC mean + std across 5 seeds |
| **P1.2** Wasserstein regularization | 1 day GPU-1 | `molmetal/molmetal_lam/reward/wasserstein_pic50_loss.py` (NEW, ~200 LOC) + 6 tests | finite loss + per-epoch Wasserstein penalty decay |
| **P1.3** Hilbert-ASTR feature augmentation (12-d RDKit + ASTSR features) | 1 day GPU-1 | `molmetal/scripts/bench_astr_features.py` (NEW, ~200 LOC) + 6 tests | per-metal R² on ASTSR features |
| **P1.4** Catherine closure oversampling | 1 day CPU | `molmetal/scripts/closure_oversample.py` (NEW, ~150 LOC) | synthetic count per metal |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: per-metal multi-task AUC + Wasserstein penalty decay + ASTSR feature R²
- **PROJECTED**: Ru AUC ≥ 0.86 with multi-task + Wasserstein + ASTSR features (vs current 0.7123 single-task)

## §5 Honest framing

The 0.7123 number is **1-seed, 5-epoch, single-task** — far from paper-grade. The fix path is **stacking 4 innovations** (multi-task + Wasserstein + ASTSR features + closure oversampling). Each individually may lift 0.05 AUC; stacked, the projection is ≥ 0.86. **There is real chance the projection is wrong** — the Krasnov 0.81 baseline is itself multi-task + larger training set, and we may not match it with our current dataset size.

## §6 Cross-references

- `metrics/by_metric/metalcytotox_ru_auc.json` (current 0.7123)
- `metrics/by_metric/dmpnn_ckpt_status.json` (current 10 ckpts, Rh seed0=seed1 bug)
- `molmetal/scripts/train_metal_cytotox.py` (current single-task trainer)
- `molmetal/math_skills/wasserstein_gradient_flow.py` (theorem)
- TODO-120 D-MPNN seed integrity (in flight for 5-seed expansion)
- TODO-119 decode_ratio strengthen (Wasserstein regularization reuse)
- TODO-84 paper rule (paper §4 not blocked)
- LAMBDA_ROADMAP §4.2 m1 (Ru AUC) column
