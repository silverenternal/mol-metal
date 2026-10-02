# TODO-95: R20-P4 3-way on MetalCytoToxDB + 100-pocket × 10-seed

**Status**: ACTIVE (CPU SCAFFOLD SHIPPED per Wf-381-A5 §2.2; HONEST_NEGATIVE_PARTIAL: CPU bench only)
**Lambda gap**: L6 + L10 + 100-pocket × 10-seed sweep (per LAMBDA_ROADMAP §4.2)
**Plan anchor**: `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §6

## §1 What

3-way (Deflexpressor + Multi-PH + ASTSR) joint retrain on MetalCytoToxDB real pIC50 (CPU bench-only ship: Wf-41 P1 `R² -0.32` honest-neg synthetic + Wf-46 T5 triangle `R² +0.58`) + 100-pocket × 10-seed Vina sweep (Wf-431-A2 wrapper authoring for `cfm_vina_dock_panel.py` per 8 d-block metals).

## §2 Why

L10 3-way triangle is MEASURED `R² -0.32` on synthetic (honest-neg) and is structurally OOD-bound on small cohorts per Wf-377-A2 §3. The 100-pocket × 10-seed Vina sweep is the paper §4 LEAD claim anchor for cross-dataset generalization.

## §3 How

1. Wf-41 P1 3-way closure (MEASURED synthetic R² -0.32) + Wf-46 T5 triangle (MEASURED R² +0.58)
2. Real pIC50 retrain on `metalcytotox_combined` n=2000 + per-cell-line cohorts n=200
3. Wf-431-A2 8 d-block × 5 pockets × 5 seeds = 200 tuples × 0.10 hr = **20 hr GPU PROJECTED** on cuda:0 Blackwell
4. USER_OK-gated actual launch per AGENTS.md §5.1 + kill_then_adapt §0

## §4 Pre-conditions

- Wf-381-A5 §2.2 closure SHIPPED (✅)
- Wf-431-A2 wrapper SHIPPED (✅)
- Wf-3 + Wf-31 + Wf-32 prior partial ships
- USER_OK=1 + VRAM budget check on cuda:0 Blackwell

## §5 MEASURED-vs-PROJECTED ledger

- **MEASURED**: L10 3-way synthetic R² -0.32 (honest-neg); Wf-46 T5 triangle R² +0.58; 30-metal d-block reachability 0.5891 (Wf-37 P2)
- **PROJECTED**: 3-way real-cohort lift +0.05 to +0.30 R² on n=2000; 100-pocket × 10-seed Vina: median -7.5 to -8.5 kcal/mol (per Wf-3)
- **HONEST-NEGATIVE**: small-cohort ceiling (n<300) holds per Wf-377-A2

## §6 Cross-references

- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §6 (R20-P4 spec)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 rows L6+L10 (TODO-95 anchor)
- `molmetal/scripts/launch_wf3_vina_8dblock_sweep.sh` (Wf-431-A2 wrapper)
- `molmetal/scripts/bench_catherine_30metals.py` (30-metal d-block bench)
