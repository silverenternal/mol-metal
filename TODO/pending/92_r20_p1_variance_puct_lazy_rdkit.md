# TODO-92: R20-P1 VariancePUCT default + Lazy RDKit import

**Status**: ACTIVE (CPU-only scaffold; R20-P1 ship per R20_BIG_DATASET_FULL_RETRAIN_PLAN §3)
**Lambda gap**: L2 (per LAMBDA_ROADMAP §4.2)
**Plan anchor**: `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §3

## §1 What

Wire `variance_puct_default=True` as the MCTS default selection policy in `molmetal/molmetal_lam/lam_chem/mcts.py` (currently default = UCB1) + add lazy RDKit imports in `molmetal/scripts/` paths so RDKit is only required when actually needed (not at module import time).

## §2 Why

L2 Variance MCTS ↔ proof_search is MEASURED `-0.185` (honest-neg) per Wf-36 P2. Switching default to VariancePUCT is the next bench-layer lift that aligns the canonical PUCT variant with the paper §3 + PUCT analysis. Lazy RDKit avoids hard import-time RDKit dependency for CPU-only smoke tests that don't touch RDKit descriptors.

## §3 How

1. Patch `mcts.py::select_branch` to default to variance_puct when `variance_puct_default=True` flag is set
2. Add `try: from rdkit import Chem except ImportError: Chem = None` to `molmetal/scripts/extract_rdkit_2d_descriptors.py` + `molmetal/scripts/cfm_property_panel.py`
3. Run `molmetal/tests/test_mcts_variance_puct.py` (target ≥ 25/25 metals success_rate=1.0 + n_errors=0)

## §4 Pre-conditions

- Wf-36 P2 closure theorem (✅ SHIP) + Wf-39 P1 vuln_a CRITICAL flag (✅ SHIP)
- `mol-metal-mechanism-mapping` skill §5 ablation design table cross-ref
- 0.5 wk GPU-0 for full retrain

## §5 MEASURED-vs-PROJECTED ledger

- **MEASURED**: L2 honest-negative `-0.185` (Wf-36 P2 ship); variance_puct smoke PASS on 3 metals (CPU)
- **PROJECTED**: variance_puct default lift on bench layer ≥ +0.05 over UCB1 baseline; lazy RDKit defers RDKit requirement to first use (no module-load crash)
- **HONEST-NEGATIVE**: L2 cannot fully close without R20-P1 GPU retrain (per Wf-381-A5 §2.1)

## §6 Cross-references

- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §3 (R20-P1 spec)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 row L2 (TODO-92 anchor)
- `TODO/completed/96_r20_p5_closure_catherine.md` (sister R20-P5 wave)
- `molmetal/tests/test_mcts_variance_puct.py` (regression guard)
