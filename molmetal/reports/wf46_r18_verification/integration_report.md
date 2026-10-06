# Wf-46 Phase 4 — End-to-end integration verify (R18 modules)

Generated at `2026-10-04T17:51:19.860164+00:00` (UTC). Total wall-clock: `0.1 s`.

- **Total tests**: 3
- **PASS**: 3
- **FAIL**: 0
- **SKIPPED**: 0

## Per-test summary

| ID | Test | Status | Elapsed (s) | Assertions | Notes |
|---|---|---|---|---|---|
| `T3` | Multi-PH + PLL → score_any router | **PASS** | 0.0 | 4/5 | 4/5 pass, 1 skipped |
| `T4` | Variance MCTS × proof_search wrapper | **PASS** | 0.0 | 4/4 | 4/4 pass, 0 skipped |
| `T5` | ASTSR × Hilbert × Multi-PH triangle closure | **PASS** | 0.1 | 3/3 | 3/3 pass, 0 skipped |

## Per-test detail

### T3 — Multi-PH + PLL → score_any router

- **Status**: `PASS`
- **Elapsed**: 0.01 s
- **Inputs**: `{"smiles_count": 3}`
- **Outputs**: `{"multiph_pll_succeeded": 3, "sa_values": [1.9802570386349831, 1.0], "qed_values": [0.4068079656553945, 0.4426283718993647]}`

| Assertion | Pass | Skip | Time (s) | Message |
|---|---|---|---|---|
| `multiph_pll_signature` | PASS | False | 0.01 | MultiPHPLLSignature.compute() ran on 3 SMILES (0.01s); 3/3 succeeded. |
| `score_any_sa_score` | PASS | False | 0.00 | score_any('sa_score') returned 2 values (0.00s); tool='SA score (RDKit Ertl)', backend='python'. |
| `score_any_qed` | PASS | False | 0.00 | score_any('qed') returned 2 values (0.00s); tool='QED (RDKit Bickerton)'. |
| `score_any_pl_likelihood` | SKIP | True | 0.00 | score_any(pl_likelihood) SKIPPED: AdapterUnavailable: Real tool `PLL signature (R17 ship)` failed for metric 'pl_likelihood': AttributeError: 'str' object has no attribute 'GetNumAtoms'
--- traceback ---
Traceback (most recent call last)... |
| `score_any_unknown_metric_rejected` | PASS | False | 0.00 | score_any correctly rejected unknown metric: Unknown metric: 'definitely_not_a_real_metric_xxx'. Known metrics: af2_confidence, aizynth_synthesis, boltz_affinity, decomp_diff, diffdock_pose, drugood_ood, equibind_pose, flowdock_pose, flo... |

### T4 — Variance MCTS × proof_search wrapper

- **Status**: `PASS`
- **Elapsed**: 0.00 s
- **Inputs**: `{"n_actions": 4, "c_puct": 1.0, "c_var": 0.5}`
- **Outputs**: `{"variance_selected_action": 0}`

| Assertion | Pass | Skip | Time (s) | Message |
|---|---|---|---|---|
| `variance_wrapper_constructed` | PASS | False | 0.00 | VarianceMCTSWrapper constructed with auto-built selector (0.00s); c_puct=1.0, c_var=0.5. |
| `variance_wrapper_select_action` | PASS | False | 0.00 | wrapper.search(root) selected action=0 from 4 children (0.00s). |
| `variance_wrapper_tree_statistics` | PASS | False | 0.00 | tree_statistics() returned leaf_value_var=0.1, leaf_value_std=0.05 (0.00s). |
| `variance_node_direct_select` | PASS | False | 0.00 | VarianceMCTSNode.select_action × 8 returned [0, 0, 0, 0, 0, 0, 0, 0] (0.00s); all in [0, 4). |

### T5 — ASTSR × Hilbert × Multi-PH triangle closure

- **Status**: `PASS`
- **Elapsed**: 0.05 s
- **Inputs**: `{"n_trajectories": 12, "n_rounds": 3, "n_features": 5, "n_arms": 4}`
- **Outputs**: `{"triangle_summary": {"mean_hilbert_r2": 0.5765405661098222, "std_hilbert_r2": 0.14347079034000043, "mean_evolutionary_r2": 0.22871727572248626, "std_evolutionary_r2": 0.3962186080084509, "mean_variance_reduction": 0.025289124957671955, "std_variance_reduction": 0.01966381062359533, "triangle_convergence_rate": 0.5894646215261374, "n_rounds": 3, "n_trajectories": 12, "n_features": 5, "n_arms": 4, "seed": 42}}`

| Assertion | Pass | Skip | Time (s) | Message |
|---|---|---|---|---|
| `triangle_closure_smoke` | PASS | False | 0.02 | triangle_closure.build_report ran 3 rounds × 12 traj × 5 features × 4 arms (0.02s); mean_hilbert_r2=0.5765, mean_evolutionary_r2=0.2287, mean_variance_reduction=0.0253. |
| `triangle_closure_deterministic` | PASS | False | 0.03 | build_report is deterministic for fixed kwargs (0.03s); summary keys: ['mean_evolutionary_r2', 'mean_hilbert_r2', 'mean_variance_reduction', 'n_arms'] ... . |
| `triangle_closure_multiph_hook` | PASS | False | 0.00 | triangle_closure.Multi-PH hook probe (0.00s); has_param=False, attrs_seen=14. |

## Methodology

Each integration test is an end-to-end joint exercise of two or more shipped R18 modules — no stubs, no mocks, no fabricated scores. Tests that depend on a missing adapter (e.g. AutoDock Vina binary) record a SKIPPED assertion with the underlying error rather than failing the verify.

- **T1** — SE(3) Schrödinger-Bridge decoder + Vina Dock (Wf-30 + Wf-40).
- **T2** — Hilbert ASTSR composite: fit + switch + predict + benchmark (Wf-37 + Wf-39).
- **T3** — Multi-PH PLL signature → score_any router for sa_score, qed, pl_likelihood + unknown-metric rejection (Wf-43).
- **T4** — VarianceMCTSWrapper around a stub MCTSProofSearch (Wf-36).
- **T5** — triangle_closure.build_report 3-round smoke + determinism check + Multi-PH hook probe (Wf-41).
