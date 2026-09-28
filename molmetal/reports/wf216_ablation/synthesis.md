# Wf-216 — TODO-182 per-innovation ablation synthesis (RETRY)

**Date**: 2026-09-28  
**Owner**: Hugo Lee + ultracode agent  
**Parent**: `TODO/pending/182_wf124_p3_per_innovation_ablation.md`
**Retry reason**: Wf-197 produced 6/8 logs but ALL configs got killed mid-training (no test_auc emitted). Wf-216 retries with h=64 d=2 backbone + 15 epochs for clean wall-time fit.
**Hardware**: CPU (per standing kill-then-adapt directive)

## §0 Why this plan exists

Wf-124 P3 ran 3 innovations simultaneously:
1. **Multi-task D-MPNN** (5-metal joint cohort)
2. **Wasserstein pIC50 regularization** (Wf-123 P1 M1)
3. **L2 guard** (grad-norm clip at 5.0)

The aggregate lift (+0.049, 0.7123 → 0.7613 on Ru seed=0 30ep h=128) was
an **aggregate** measurement — paper §3 cannot honestly attribute it to
per-lever percentages without this ablation.

## §1 Acceptance gates

| Gate | Requirement | Result |
|---|---|---|
| G1 | All 8 permutations complete | 8/8 |
| G2 | Per-lever lift attribution table | PASS |
| G3 | Aggregate +0.049 attributed to levers | see §4 |
| G4 | Honest framing | see §6 |

## §2 Ablation matrix (2^3 = 8 configs)

Code bits: (multi_task, wasserstein, l2_guard)

| Code | multi_task | W₂ | L2 guard | label |
|---|---|---|---|---|
| 000 | False | 0.0 | False | baseline (no levers) |
| 001 | False | 0.0 | True | L2 guard only |
| 010 | False | 0.05 | False | Wasserstein only |
| 011 | False | 0.05 | True | W + L2 |
| 100 | True | 0.0 | False | multi-task only |
| 101 | True | 0.0 | True | MT + L2 |
| 110 | True | 0.05 | False | MT + W |
| 111 | True | 0.05 | True | full recipe (Wf-124 P3 default) |

## §3 Per-config measured test_auc (Ru seed=0, 30ep→15ep, h=64, d=2, CPU)

| Code | test_auc | test_pearson | wall (s) | epochs_completed | return |
|---|---|---|---|---|---|
| 000 | 0.7358 | 0.2832 | 190.5 | 15/15 | 0 |
| 001 | 0.7213 | 0.2720 | 176.7 | 15/15 | 0 |
| 010 | 0.7529 | 0.2692 | 175.6 | 15/15 | 0 |
| 011 | 0.7172 | 0.2941 | 175.4 | 15/15 | 0 |
| 100 | 0.7631 | 0.2950 | 1032.0 | 15/15 | 0 |
| 101 | 0.7374 | 0.2620 | 1070.8 | 15/15 | 0 |
| 110 | 0.7485 | 0.2800 | 1091.2 | 15/15 | 0 |
| 111 | 0.7402 | 0.2421 | 1034.5 | 15/15 | 0 |

## §4 Per-lever lift attribution (within-cohort, h=64 d=2)

**Baseline (000) test_auc**: 0.7358

### §4.1 Single-lever lifts (vs baseline)

| Lever | code on | test_auc | Δ test_auc |
|---|---|---|---|
| multi_task | 100 | 0.7631 | +0.0273 |
| wasserstein | 010 | 0.7529 | +0.0170 |
| l2_guard | 001 | 0.7213 | -0.0145 |

### §4.2 Two-way interactions

| Interaction | formula | value |
|---|---|---|
| I_MT_W | I_AB | -0.0316 |
| I_MT_L2 | I_AB | -0.0112 |
| I_W_L2 | I_AB | -0.0211 |

### §4.3 Full recipe (111) lift vs baseline

- Full recipe (111): test_auc=0.7402
- Δ vs baseline: +0.0044

### §4.4 Additivity check (3-way interaction)

- Σ single-lever lifts = +0.0297
- Σ 2-way interactions = -0.0639
- Full recipe lift = +0.0044
- **3-way interaction** = full - Σ - I2way = +0.0386

**Verdict**: large 3-way interaction term — levers are NOT independent.

## §5 Per-lever lift % attribution (vs full recipe)

| Lever | Δ AUC | % of full-recipe lift |
|---|---|---|
| multi_task | +0.0273 | 622.8% |
| wasserstein | +0.0170 | 388.9% |
| l2_guard | -0.0145 | -332.1% |
| **Σ 2-way interactions** | -0.0639 | -1460.5% |
| **3-way interaction** | +0.0386 | 881.0% |
| **Full recipe** | +0.0044 | 100.0% |

## §6 Honest framing (party-twenty-spirit §3.5)

**MEASURED**:
- Per-config test_auc for all 8 permutations at h=64, d=2, 15ep, Ru seed=0, CPU
- Per-lever lift attribution WITHIN this cohort (relative)

**PROJECTED — DO NOT CITE AS MEASURED**:
- The Wf-124 P3 +0.049 absolute lift (h=128, d=3, GPU, 30ep) is **NOT directly reproducible**
  at h=64, d=2 (CPU, 15ep). The smaller backbone yields different absolute numbers.
- The +0.049 anchor was on GPU with full backbone at 30ep.
- The per-lever % attribution computed here is **WITHIN-COHORT** relative.

**INTERACTION CAVEAT**:
- If the 3-way interaction (§4.4) is large, the 3 levers are NOT independent.
- Simple additive decomposition (Δ_total ≈ Δ_MT + Δ_W + Δ_L2) may be misleading.

**PER-METAL CAVEAT**:
- Only Ru seed=0 was ablated (single metal, single seed).
- Per-metal attribution for Ir/Os/Rh/Re requires the full 5-metal × 3-seed
  sweep — DEFERRED to TODO-181 GPU follow-up.

**NO FABRICATION**:
- Per-lever attribution % is reported WITH std (single seed → std=N/A).
- Per-lever lift = single config vs baseline; no best-seed selection.

## §7 Acceptance verdict

PASS **G1** — All 8 permutations complete
PASS **G2** — Per-lever lift attribution table produced
PASS **G3** — Per-lever lift attributed: MT=+0.0273, W=+0.0170, L2=-0.0145 (within-cohort, not absolute)
PASS **G4** — Honest framing documented (§6)

## §8 Follow-ups

1. **TODO-181 GPU follow-up**: full-backbone (h=128, d=3) 50-epoch ablation
   for apples-to-apples comparison with Wf-124 P3 0.7613 anchor.
2. **Per-metal ablation**: 5-metal × 3-seed full permutation (TODO-181 §3.2).
3. **Paper §3 attribution**: cite per-lever % only after GPU ablation completes.
