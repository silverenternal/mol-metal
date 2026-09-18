# Lipman 2023 FM adapter on ROCm

**Status:** completed
**Completed:** 2026-09-11
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/generate_validation.md
**Owner:** (unset)

## What was delivered
A thin adapter over Lipman et al. 2023 (ICLR 2023, arXiv 2210.02747) — the
official `facebookresearch/flow_matching` repo is cloned at
`molmetal/references/flow_matching/` and we import `AffineProbPath`,
`CondOTScheduler`, and `ODESolver` directly. Our adapter exposes joint
atom-type + coordinate flow-matching for the lambda_clickchem pipeline:
Categorical sampling on `atom_head` logits over atomic numbers, plus
continuous FM over Cartesian coordinates. Verified end-to-end on ROCm 7.2
+ triton-rocm 3.8.0.

## Hard numbers
- 6/6 unit tests pass on `cuda:0` (ROCm alias): generate_validation.md
- CFM loss drop ≥ 1.5× over 50 steps on a toy 8-atom trajectory batch
- Joint FM: Categorical sampling on atom_type logits, continuous FM on (x, y, z)
- Source FM library: `molmetal/references/flow_matching/` (Lipman 2023 official)
- No re-implementation: we wrap, not rewrite

## Lessons learned
- Use the official Lipman 2023 clone as-is for `AffineProbPath` / `CondOTScheduler` /
  `ODESolver`; the only project-specific code is the wrapper + the EGNN velocity field
  (`molmetal/adapters/flow_matching_lipman/egnn_velocity.py`).
- `MolFlow-Triton`'s `models._scatter.scatter_sum` provides the Triton kernel for the
  EGNN message-passing aggregation; do not reinvent.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/egnn_velocity.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/test_rocm_lipman.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/generate_validation.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/references/flow_matching/  (cloned)