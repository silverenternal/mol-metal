# TODO-96: R20-P5 Closure theorem + Catherine 6 properties

**Status**: ACTIVE (CPU SCAFFOLD SHIPPED per Wf-381-A5 §2.3; HONEST_NEGATIVE_STRUCTURAL on GPU retrain; Catherine 6/6 partial bench)
**Lambda gap**: L1 + L6 + L12 (per LAMBDA_ROADMAP §4.2)
**Plan anchor**: `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §7

## §1 What

Closure theorem (paper §3.4 main result) + Catherine 6 properties (soundness + completeness + termination + confluence + type preservation + local confluence per J. Algebra 2024 §3) PASS on ≥ 25 metals × 1000 samples. CPU SCAFFOLD SHIPPED via `molmetal/scripts/bench_catherine_6_props_25metals.py` + `molmetal/scripts/bench_closure_ood.py`.

## §2 Why

L1 reachability 0.5108 MEASURED; L6 d-block reachability 0.5891 MEASURED; L12 OOD 0.5581 MEASURED (PlatinAI 5K) / 0.0300 (MetalCytoToxDB) per Wf-36 + Wf-37 + Wf-41. Paper §3.4 main theorem claim requires 6/6 Catherine PASS on ≥ 25 metals, NOT just bench layer ship.

## §3 How

1. Wf-57 CPU bench (✅ SHIP) — 20/25 metals × 6/6 properties
2. R20-P5 GPU-0 25-metal × 1000-sample sweep (~1 wk GPU-0 USER_OK-gated)
3. Wf-44 P2 cross-domain OOD (PlatinAI 5K + MetalCytoTox + tmQM d-block) (~2 wk CPU + 1 wk GPU-0)
4. Wf-54 close-out: `TODO/completed/96_r20_p5_closure_catherine.md` (TODO-94 in old numbering) update with per-metal 6/6 PASS counts

## §4 Pre-conditions

- Wf-381-A5 §2.3 closure SHIPPED (✅)
- Wf-57 Catherine 25-metals partial SHIPPED (✅)
- Wf-255 Catherine 30-metals audit SHIPPED (✅)
- USER_OK=1 + VRAM budget check on cuda:0 Blackwell

## §5 MEASURED-vs-PROJECTED ledger

- **MEASURED**: L1 0.5108; L6 0.5891; L12 OOD 0.5581/0.0300; Tanimoto<0.4 fraction 0.9978 (PlatinAI 5K); Catherine 6/6 partial 20/25 metals
- **PROJECTED**: 6/6 PASS on 25 metals × 1000 samples `verdict: "6/6_PASS_25metals_1000samples"`; per-metal reachability ≥ 0.95; per-domain OOD reachability ≥ 0.85; Tanimoto ≥ 0.99

## §6 Cross-references

- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §7 (R20-P5 spec)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 rows L1+L6+L12 (TODO-96 anchor)
- `molmetal/scripts/bench_catherine_6_props_25metals.py` (Catherine bench)
- `molmetal/scripts/bench_closure_ood.py` (OOD bench)
- `.claude/skills/mol-metal-closure-catherine/SKILL.md` (Catherine skill anchor)
- `.claude/skills/mol-metal-catherine-dblock-bench/SKILL.md` (30-metal d-block anchor)
