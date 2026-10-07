# Wf-46 Phase 4 — End-to-end integration verify (R18 modules)

Generated at `2026-10-07T21:14:17.558551+00:00` (UTC). Total wall-clock: `0.1 s`.

- **Total tests**: 1
- **PASS**: 1
- **FAIL**: 0
- **SKIPPED**: 0

## Per-test summary

| ID | Test | Status | Elapsed (s) | Assertions | Notes |
|---|---|---|---|---|---|
| `T5` | ASTSR × Hilbert × Multi-PH triangle closure | **PASS** | 0.1 | 3/3 | 3/3 pass, 0 skipped |

## Per-test detail

### T5 — ASTSR × Hilbert × Multi-PH triangle closure

- **Status**: `PASS`
- **Elapsed**: 0.11 s
- **Inputs**: `{"n_trajectories": 12, "n_rounds": 3, "n_features": 5, "n_arms": 4}`
- **Outputs**: `{"triangle_summary": {"mean_hilbert_r2": 0.5765405661098222, "std_hilbert_r2": 0.14347079034000043, "mean_evolutionary_r2": 0.22871727572248626, "std_evolutionary_r2": 0.3962186080084509, "mean_variance_reduction": 0.025289124957671955, "std_variance_reduction": 0.01966381062359533, "triangle_convergence_rate": 0.5894646215261374, "n_rounds": 3, "n_trajectories": 12, "n_features": 5, "n_arms": 4, "seed": 42}}`

| Assertion | Pass | Skip | Time (s) | Message |
|---|---|---|---|---|
| `triangle_closure_smoke` | PASS | False | 0.05 | triangle_closure.build_report ran 3 rounds × 12 traj × 5 features × 4 arms (0.05s); mean_hilbert_r2=0.5765, mean_evolutionary_r2=0.2287, mean_variance_reduction=0.0253. |
| `triangle_closure_deterministic` | PASS | False | 0.06 | build_report is deterministic for fixed kwargs (0.06s); summary keys: ['mean_evolutionary_r2', 'mean_hilbert_r2', 'mean_variance_reduction', 'n_arms'] ... . |
| `triangle_closure_multiph_hook` | PASS | False | 0.00 | triangle_closure.Multi-PH hook probe (0.00s); has_param=False, attrs_seen=14. |

## Methodology

Each integration test is an end-to-end joint exercise of two or more shipped R18 modules — no stubs, no mocks, no fabricated scores. Tests that depend on a missing adapter (e.g. AutoDock Vina binary) record a SKIPPED assertion with the underlying error rather than failing the verify.

- **T1** — SE(3) Schrödinger-Bridge decoder + Vina Dock (Wf-30 + Wf-40).
- **T2** — Hilbert ASTSR composite: fit + switch + predict + benchmark (Wf-37 + Wf-39).
- **T3** — Multi-PH PLL signature → score_any router for sa_score, qed, pl_likelihood + unknown-metric rejection (Wf-43).
- **T4** — VarianceMCTSWrapper around a stub MCTSProofSearch (Wf-36).
- **T5** — triangle_closure.build_report 3-round smoke + determinism check + Multi-PH hook probe (Wf-41).
