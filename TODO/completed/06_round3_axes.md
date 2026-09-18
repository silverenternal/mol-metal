# Round-3 algorithmic axes (L-1, L-2, L-4, M-1, M-2, M-3)

**Status:** completed
**Completed:** 2026-09-12
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_combined_report.md
**Owner:** (unset)

## What was delivered
Six algorithmic axes from the round-3 plan
([[lambda-mh-algorithmic-axes]]) are now wired end-to-end in the project:

- **L-1** — DiffDock + FlowDock binding-oracle channels in the MCTS leaf typecheck
  (CPU preference for FlowDock, fallback verified)
- **L-2** — persistent AlphaZero/MuZero-style MCTS tree across iterations with
  Dirichlet reinjection at the root; `n_pairs_seen_by_prior` trajectory is
  monotonically increasing: [264, 534, 816]
- **L-4** — REINVENT4 subprocess-RPC adapter + `r_reinvent4` reward channel
  (**OBS** until the `reinvent` binary is installed; see TODO/pending/05_reinvent4_install.md)
- **M-1** — Drop M-B2 hooks (EMA + ensemble + mixup + SWA) from combined V4 path;
  M-B1 alone gets 0.5327 mean (above baseline 0.5135)
- **M-2** — Loosen `pic50_min` clamp 4.0 → 3.0 with 10-epoch warmup at 4.0
- **M-3** — Register `LossV4MB1.log_s_*` in combined V4 path optimizer (σ was
  stuck at +0.0000 before this fix)

## Hard numbers
- **544 / 550** tests pass; **6 pre-existing failures** carried over (not new regressions)
- New tests added in round-3: 7 (L-1) + 8 (L-2) + 5 (L-4) + 7 (M-1/M-2/M-3) = **27**
- `n_pairs_seen_by_prior` trajectory: [264, 534, 816] (monotone → persistence works)
- L-1 oracle fallback verified: FlowDock preferred over DiffDock when CPU is the constraint
- L-4 reward aggregator treats `None` as 0.0; r_reinvent4 silently OFF until binary is on PATH

## Lessons learned
- L-1 + L-4 are larger integration work; L-2 requires Dirichlet reinjection to
  avoid overfit to early high-reward regions.
- M-B2 hooks stacked amplify ensemble distribution shift on OOD test set;
  removing them (M-1) closes the gap to M-B1 alone.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_combined_report.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_L1_diffdock_oracle.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_L2_persistent_tree.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_L4_reinvent4_scorer.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py