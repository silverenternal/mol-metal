# TODO-121 — m19 LOMO strengthen via Wasserstein transfer + Rh bottleneck fix

**Date**: 2026-09-27
**Status**: 🚀 **Active — closes m19 LOMO 0.5346 → ≥ 0.55 (Rh bottleneck fix)**
**Priority**: **P1 HIGH** (LOMO is the canonical cross-metal generalization claim)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Honest model-state assessment (2026-09-27)
**Skill alignment**: `mol-metal-cfm-training` + `mol-metal-r18-math`

---

## §0 Why this plan exists

**Current state**:
- m19 metal_lomo_auc = **0.5346** (5-fold LOMO, mean across 5 metals)
- Target: ≥ 0.55 (5 pp headroom over Krasnov 0.81 baseline — though LOMO is harder than in-domain)
- Gap: **−0.015 AUC** (~3% relative, smaller than m1)
- Verdict: `MEASURED_BELOW_GATE`

**Why Rh is the bottleneck**:
- Per Wf-32 P2: Rh LOMO AUC = 0.368 (vs Ru 0.7+ for the other 4 metals)
- Rh has the smallest training cohort (n=60 train, n=38 test)
- Rh's reaction chemistry is most distinct from Cu/Ni/Co

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | GPU-1 RTX 5090 |
| Wall-time | 1 week GPU-1 |
| Output | Wasserstein transfer + Rh synthetic oversampling + multi-task backbone |
| Tests | ~20 new tests |

## §2 Solution path (3 innovations stacked)

1. **Wasserstein transfer regularization** (`wasserstein_gradient_flow.py`)
   - Penalize the L2 distance between per-metal embedding distributions in shared latent space
   - Forces the model to learn metal-invariant chemistry
2. **Rh synthetic oversampling via closure theorem**
   - Generate ~500 synthetic Rh-compounds using the 5-click productive space (Catherine 6/6 verified)
   - Oversample Rh training cohort n=60 → n=560 (10x)
3. **Multi-task backbone with per-metal heads**
   - Share Ru/Ir/Os layers, separate Rh head
   - Leverages signal from larger metals to lift Rh specifically

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Wasserstein transfer regularization | 1 day GPU-1 | `molmetal/molmetal_lam/reward/wasserstein_transfer.py` (NEW, ~200 LOC) + 6 tests | Wasserstein distance decay across folds |
| **P1.2** Rh closure-theorem oversampling | 1 day CPU + GPU | `molmetal/scripts/closure_oversample_rh.py` (NEW, ~250 LOC) + 8 tests | synthetic Rh cohort count ≥ 500 + Catherine 6/6 PASS per compound |
| **P1.3** LOMO full re-bench with multi-task + Wasserstein + Rh oversample | 1 week GPU-1 | `molmetal/reports/wf121_lomo_rerun/` (NEW) | per-metal AUC + 5-fold mean ≥ 0.55 |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: Wasserstein distance decay + Rh synthetic count + LOMO per-metal AUC
- **PROJECTED**: Rh LOMO AUC ≥ 0.50 (vs current 0.368, +0.13); overall LOMO mean ≥ 0.60 (vs current 0.535)

## §5 Honest framing

m19 gap is small (−0.015), but the **Rh bottleneck** is the real story — it's a 0.368 → 0.55 gap on the hardest metal. The fix path needs **3 innovations stacked** (Wasserstein + oversampling + multi-task). Each individually may lift 0.05-0.10; stacked, the projection is Rh ≥ 0.50 + overall ≥ 0.60.

## §6 Cross-references

- `metrics/by_metric/metal_lomo_auc.json` (current 0.5346)
- `molmetal/scripts/train_metal_lomo.py` (current LOMO trainer)
- `molmetal/math_skills/wasserstein_gradient_flow.py` (theorem)
- `molmetal/math_skills/catherine_properties.py` (closure theorem)
- TODO-119 / TODO-120 Wasserstein regularization (reuse)
- TODO-84 paper rule (paper §4 not blocked)
- LAMBDA_ROADMAP §4.2 m19 (LOMO) column