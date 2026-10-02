# TODO-94: R20-P3 Multi-PH + PLL feature extraction

**Status**: ACTIVE (CPU SCAFFOLD SHIPPED per Wf-381-A5 §2.4 + Wf-382-A1 SCAFFOLD; HONEST_NEGATIVE_PARTIAL on small cohorts per Wf-388-A1)
**Lambda gap**: L7 (per LAMBDA_ROADMAP §4.2)
**Plan anchor**: `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §5

## §1 What

Multi-PH (Multi-Parameter Persistence) + PLL (Persistent Laplacian Loss) feature extraction scaffold → real retrain on MetalCytoToxDB n=5460. CPU scaffold SHIPPED via `molmetal/adapters/flow_matching_lipman/pll_features.py` + `molmetal/scripts/wf382a1_multiph_pll_scaffold.py`. Real-cohort MEASURED lift = 0 on n=200 cohorts per Wf-388-A1 §2.1 (synthetic-vs-real gap `-0.3410 R²`).

## §2 Why

L7 Multi-PH + PLL end-to-end is MEASURED `R² +0.087` (honest-neg) + Wf-39 P3 retrain `R² -0.32`. Real-cohort lift requires Wf-390-A2 audit structural root cause (encoder signal floor at R²=0.0043 on 12-d RDKit; NOT W₂ alignment). Path A (5-metal subset Ru/Ir/Rh/Os/Re ~12 hr GPU) recommended per Wf-390-A1.

## §3 How

1. Wf-392-A1 SE(3) SB PLL wire-in SHIPPED (3 NEW CLI flags `--cfm-multiph-pll` + `--cfm-multiph-pll-top-k` + `--cfm-multiph-pll-weight`)
2. Wf-381-A5 §2.4 closure + Wf-382-A1 SCAFFOLD SCAFFOLD-only validation
3. Path A 5-metal × 3-seed × 50-epoch retrain on cuda:0 Blackwell (~12 hr GPU USER_OK-gated)
4. Wf-394-B5 follow-up wave for full training-loop integration (~50 LOC trainer edit)

## §4 Pre-conditions

- Wf-381-A5 §2.4 closure SHIPPED (✅)
- Wf-382-A1 SCAFFOLD SHIPPED (✅)
- Wf-392-A1 wire-in SHIPPED (✅)
- USER_OK=1 + VRAM budget check on cuda:0 Blackwell

## §5 MEASURED-vs-PROJECTED ledger

- **MEASURED**: CPU SCAFFOLD on synthetic n=60 cohort: 4-SMILES × 10-d PLL feature matrix; trainer wire-in smoke PASS
- **MEASURED**: Real-cohort (n=200) lift = 0 per Wf-388-A1 §2.1 (synthetic-vs-real gap -0.3410 R²)
- **PROJECTED**: Path A 5-metal × 3-seed × 50-epoch: +0.05 to +0.20 R² lift on `metalcytotox_combined` n=2000 (per Wf-390-A1 §3)

## §6 Cross-references

- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §5 (R20-P3 spec)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 row L7 (TODO-94 anchor)
- `molmetal/adapters/flow_matching_lipman/pll_features.py` (PLL scaffold)
- `molmetal/scripts/wf382a1_multiph_pll_scaffold.py` (smoke scaffold)
- `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` (sister Path A plan)
