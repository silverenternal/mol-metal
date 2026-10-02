# TODO-93: R20-P2 Hilbert ASTSR GPU joint retrain

**Status**: ACTIVE (CPU SCAFFOLD SHIPPED per Wf-381-A5 §2.1 + Wf-377-A1 redesign progress; HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-365-A3 verdict)
**Lambda gap**: L4 + L5 + L7 + L8 (per LAMBDA_ROADMAP §4.2 — multi-gap joint close)
**Plan anchor**: `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §4

## §1 What

GPU retrain of the Hilbert primal-dual optimizer (M25) + ASTSR regression joint pipeline on MetalCytoToxDB real pIC50, targeting R² ≥ 0.85 (vs current honest-negative PARTIAL best CV-R² +0.6814 CEILING_BREAK + aug_mean 0.472).

## §2 Why

5 architectural attempts exhausted per Wf-365-A3 §5 + Wf-371-A1 degenerate chain + Wf-377-A1 deep-dive-research move. The m6_kappa metric itself is structurally un-liftable on n=5460 MetalCytoToxDB (per Wf-377-A1 §2.3 empirical sweep). Forward-looking recommendation: drop m6_kappa from §3 main-result, promote top-3 WEAK-band correlates to §3 supplementary sub-table, retain m6_kappa as §6 honest-limitations complementary heuristic.

## §3 How

1. Run Wf-37 P2 + Wf-39 P2 composite (MEASURED +0.186 R²) on full MetalCytoToxDB n=5460 (CPU)
2. Launch Hilbert ASTSR end-to-end retrain on cuda:0 Blackwell (~3 wk GPU-0 USER_OK-gated)
3. Write paper §6 limitations paragraph per Wf-377-A1 §4.4 + Wf-424-A14 §6 v4 FINAL
4. Author USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (~12-15 hr CPU)

## §4 Pre-conditions

- Wf-377-A1 m6_kappa drop-promote decision SHIPPED (✅)
- Wf-424-A14 paper §3 v4 FINAL SHIPPED (✅)
- Wf-424-A15 m6_kappa drop-promote paper-conversion SHIPPED (✅)
- USER_OK=1 + VRAM budget check on cuda:0 Blackwell

## §5 MEASURED-vs-PROJECTED ledger

- **MEASURED**: R² +0.6814 CV (CEILING_BREAK); aug_mean 0.472 < 0.6 (gate FAIL); 5 architectural attempts exhausted per Wf-365-A3
- **PROJECTED**: drop-from-§3 recommendation implemented; top-3 WEAK correlates (MolLogP +0.1999, MorganDensity_r2 +0.1644, Chi3v +0.1541) to §3 supplementary sub-table
- **HONEST-NEGATIVE**: gate unreachable on this host (5 attempts exhausted; 9 total paths per Wf-389-A1 §3); structural finding preserved per Wf-377-A1 §2.3

## §6 Cross-references

- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §4 (R20-P2 spec)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 rows L4+L5+L7+L8 (TODO-93 anchor)
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 verdict doc)
- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 forward-looking)
- `docs/M6_KAPPA_DROP_DECISION.md` (Wf-424-A15 paper-conversion draft)
