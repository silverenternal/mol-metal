# TODO-98: Wf-44 P2 Closure OOD Phase 2

**Status**: ACTIVE (in flight Q4 2026 per Wf-44 P2 + LAMBDA_ROADMAP §4.2 row L12; CPU SCAFFOLD SHIPPED per Wf-381-A5 §2.4; HONEST_NEGATIVE_PARTIAL: 1/3 domains lifted)
**Lambda gap**: L12 Phase 2 (per LAMBDA_ROADMAP §4.2)
**Plan anchor**: `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §7 + Wf-44 in-flight

## §1 What

Wf-44 Phase 2 numerical fix lift: `grad_clip + step_decay + fp64` retrain of Wf-41 P2 closure OOD bench on PlatinAI 5K + MetalCytoTox + tmQM d-block. Goal: per-domain reachability ≥ 0.85 on all 3 domains (vs current 0.5581 / 0.0300 / n/a).

## §2 Why

L12 Phase 1 MEASURED 0.5581 (PlatinAI 5K OOD) + 0.0300 (MetalCytoTox OOD) per Wf-41 P2. Phase 2 fix targets the numerical instability (Tanimoto<0.4 filter + grad_clip + step_decay + fp64 should lift to ≥ 0.85 per Wf-44 debug findings).

## §3 How

1. Wf-44 P1 catalog SHIPPED (✅) — closure-theorem-bench-catalog-30-metals.json
2. Wf-44 P2 numerical fix retrain (~2 wk CPU + 1 wk GPU-0 USER_OK-gated)
3. Per-domain reachability lift validation against threshold ≥ 0.85
4. Paper §3.4 cross-domain OOD evidence compilation

## §4 Pre-conditions

- Wf-41 P2 closure OOD (✅ SHIP)
- Wf-44 P1 catalog (✅ SHIP)
- Wf-381-A5 §2.4 closure (✅ SHIP)
- USER_OK=1 + VRAM budget check on cuda:0 Blackwell

## §5 MEASURED-vs-PROJECTED ledger

- **MEASURED**: L12 Phase 1 PlatinAI 5K OOD 0.5581; MetalCytoTox OOD 0.0300; Wf-44 P1 catalog 30 metals
- **PROJECTED**: per-domain reachability ≥ 0.85 on all 3 domains (PlatinAI 5K + MetalCytoTox + tmQM d-block); Tanimoto<0.4 fraction_below_threshold ≥ 0.99 (vs current 0.9978)

## §6 Cross-references

- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §7 (R20-P5 spec)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 row L12 (TODO-98 anchor)
- `molmetal/scripts/bench_closure_ood.py` (OOD bench)
- `.claude/skills/mol-metal-closure-catherine/SKILL.md` (Catherine + OOD skill anchor)
- `TODO/pending/96_r20_p5_closure_catherine.md` (sister R20-P5 wave)
