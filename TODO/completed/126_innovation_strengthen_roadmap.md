# TODO-126 — Innovation strengthen roadmap (master plan)

**Date**: 2026-09-27
**Status**: 🚀 **Active — master plan for innovation-driven metric strengthen**
**Priority**: **P0 CRITICAL** (this is the umbrella plan for all 6 strengthen TODOs)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Honest model-state assessment (2026-09-27)
**Skill alignment**: all 6 strengthen skills

---

## §0 Why this plan exists

The user asked: "weak points can be solved by strengthening our innovations?". **Yes** — every weak metric has a documented innovation-led fix path. This master plan coordinates **6 strengthen TODOs** that share common machinery:

| Shared machinery | Used by | Status |
|---|---|---|
| **Wasserstein gradient flow regularization** (`wasserstein_gradient_flow.py`) | TODO-119/120/121/122/123/125 | NEW module |
| **Catherine closure theorem** (`catherine_properties.py`) | TODO-121/122/124/125 | EXISTING, used for generation bias |
| **Multi-PH + PLL feature augmentation** (Wf-66 R20-P3) | TODO-122 | EXISTING, already shipped |
| **Hilbert-ASTR combined regression** (`hilbert_astsr_integration.py`) | TODO-120/122 | EXISTING, needs L2 fix |
| **Multi-task learning** (multi-task D-MPNN) | TODO-120/121/122 | NEW backbone |

**Single-master Wasserstein module** + **single-master Catherine bias module** = 2 NEW modules + reuse = DRY across 6 strengthen TODOs.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU + GPU-0 + GPU-1 |
| Wall-time | 4 weeks GPU (all 6 strengthen TODOs in parallel) |
| Output | 6 strengthen TODOs (119-125) + 2 master modules (Wasserstein + Catherine-bias) |
| Tests | ~150 new tests across all TODOs |

## §2 The 6 strengthen TODOs

| # | Weak metric | Current | Target | Primary innovation | Secondary innovation |
|---:|---|---:|---:|---|---|
| **TODO-119** | m14 decode_ratio | 0.0/8 (4-fix bundle) | ≥ 0.8 RDKit-strict | YuelBond decoder swap | Wasserstein + Catherine closure |
| **TODO-120** | m1 Ru AUC | 0.7123 (1-seed) | ≥ 0.86 | Multi-task D-MPNN | Wasserstein + Hilbert-ASTR features |
| **TODO-121** | m19 LOMO | 0.5346 | ≥ 0.60 (Rh lift) | Wasserstein transfer | Rh closure oversample |
| **TODO-122** | PlatinAI 4-cell R² | -6.71 | ≥ 0 (any cell) | Multi-task 4-cell + 172-d feature | Wasserstein on pIC50 |
| **TODO-123** | m15 RMSD | 0.428 (partial staging) | ≤ 0.35 | SE(3) Schrödinger Bridge | Wasserstein pose reg + full staging |
| **TODO-124** | m5 SA | 3.099 | ≤ 3.0 (or ≤ 2.8 on closure cohort) | Catherine closure-bias | AiZynth smarts validate |
| **TODO-125** | per-pocket Vina | 0.51 (0.05-0.95) | ≥ 0.30 all pockets | pocket-specific closure bias | Wasserstein per-pocket |

## §3 Shared master modules

### M1. Wasserstein gradient flow regularization module (NEW)

`molmetal/molmetal_lam/reward/wasserstein_loss.py` (~250 LOC):
- `wasserstein_loss(pred_dist, ref_dist, n_mc=1000)` — generic W₂ penalty
- `wasserstein_transfer_loss(per_metal_embeds)` — for cross-metal transfer (TODO-121)
- `wasserstein_pic50_loss(pred_pic50, ref_pic50_dist)` — for pIC50 manifold (TODO-120/122)
- `wasserstein_pose_loss(pred_pose, ref_pose)` — for pose alignment (TODO-123)
- `wasserstein_pocket_loss(candidate_dist, pocket_ligand_dist)` — per-pocket bias (TODO-125)
- `wasserstein_decay_check(loss_history, window=10)` — decay test

### M2. pocket_specific_closure_bias module (NEW)

`molmetal/molmetal_lam/reward/pocket_closure_bias.py` (~200 LOC):
- `pocket_specific_closure_bias(pocket_id)` → dict of bias weights per 5-click rule
- `closure_bounded_smiles_generation(pocket_id, n=100)` → list of candidates
- `closure_oversample(target_metal, n=500)` → synthetic cohort (for TODO-121)
- Per-pocket pharmacophore lookup (1h36/3eml/3ny8/4aaw/4rlu)

## §4 Cross-TODO dependency graph

```
M1 (Wasserstein) ─┬─→ TODO-119 (decode_ratio)
                  ├─→ TODO-120 (Ru AUC)
                  ├─→ TODO-121 (LOMO)
                  ├─→ TODO-122 (PlatinAI R²)
                  ├─→ TODO-123 (RMSD)
                  └─→ TODO-125 (per-pocket Vina)

M2 (closure-bias) ─┬─→ TODO-121 (LOMO)
                   ├─→ TODO-124 (SA)
                   ├─→ TODO-125 (per-pocket Vina)
                   └─→ TODO-119 (decode_ratio — Catherine-bounded generation)

Wf-122 (L2/guard) ───→ TODO-122 (PlatinAI R²)

Wf-66 (Multi-PH+PLL) ─→ TODO-122 (PlatinAI R²)
```

## §5 Phases (parallel where possible)

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Build M1 (Wasserstein) | 4 hr CPU | `wasserstein_loss.py` (NEW, ~250 LOC) + 8 tests | pytest PASS + decay test |
| **P1.2** Build M2 (closure-bias) | 4 hr CPU | `pocket_closure_bias.py` (NEW, ~200 LOC) + 6 tests | per-pocket bias output |
| **P2.x** Run 6 strengthen TODOs in parallel | 3 weeks GPU | 6 close-out docs + new metrics | per-target MEASURED |
| **P3.1** Final cross-comparison | 2 days CPU | synthesis.md | per-metric before/after |

## §6 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: M1 + M2 module PASS + per-TODO target hit/miss
- **PROJECTED**: ≥ 4 of 6 weak metrics lift to gate; ≥ 2 may not lift (signal floor structural)

## §7 Honest framing + scoring-related concerns

**Honest projection**: with 6 strengthen TODOs × 4-6 weeks GPU, we expect:
- **Likely lift**: TODO-119 decode_ratio (YuelBond unproven but available), TODO-120 m1 Ru AUC (multi-task is well-known to help), TODO-123 m15 RMSD (full staging alone may solve it)
- **Plausible lift**: TODO-121 m19 LOMO, TODO-122 PlatinAI R² (signal floor binding), TODO-124 m5 SA
- **May not lift**: TODO-122 if 12-d signal floor is structural

**Scoring-related**: Wasserstein regularization is a NEW scoring component — must NOT silently substitute. Add explicit `engine="wasserstein_reg"` tag to all per-sample outputs.

## §8 Cross-references

- TODO-119 decode_ratio strengthen
- TODO-120 m1 Ru AUC strengthen
- TODO-121 m19 LOMO strengthen
- TODO-122 PlatinAI 4-cell R² strengthen
- TODO-123 m15 RMSD strengthen
- TODO-124 m5 SA strengthen
- TODO-125 per-pocket Vina strengthen
- Wf-122 Hilbert L2 + ASTSR scale guard (in flight)
- Wf-66 Multi-PH + PLL (already shipped)
- Wf-120 D-MPNN seed integrity (in flight)
- Wf-119 honest reclassification (in flight)
- Wf-121 meko metal extension (in flight)
- TODO-84 paper rule (paper §3.4 + §4 + §6 tables not blocked)
- LAMBDA_ROADMAP §4.2 (all metric columns)
- party-twenty-spirit §3.5 (honest-negative framing)
