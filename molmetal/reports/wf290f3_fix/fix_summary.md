# Wf-290F3 — Fix `test_joint_3stage_mcts.py`

## Error before

```
ImportError while importing test module 'molmetal/tests/test_joint_3stage_mcts.py'.
Traceback:
molmetal/tests/test_joint_3stage_mcts.py:72: in <module>
    from molmetal_lam.search_alg.proof_search import (
E   ImportError: cannot import name 'EXPLORATION_STRATEGIES' from
   'molmetal_lam.search_alg.proof_search'
   (/home/hugo/codes/mol-metal/molmetal/molmetal_lam/search_alg/proof_search.py)
```

The test module — plus the joint script it imports from
(`molmetal/scripts/joint_score_repellent_variance_puct_mcts.py`) — referenced
two surface-level contracts that were missing from
`molmetal/molmetal_lam/search_alg/proof_search.py` after the Wf-36
refactor:

1. A module-level constant `EXPLORATION_STRATEGIES` listing the three
   strategies (`"dirichlet"`, `"score_repellent"`, `"variance"`).
2. `MCTSProofSearch` dataclass fields so callers could pass
   `exploration_strategy=...`, `c_var=...`, `n_min_visits=...` and
   `debug=...` to the constructor — every `_make_mcts` fixture in
   the joint suite does this.

Without those, the test collection failed before any assertion could run,
and the joint harness script crashed during
`MCTSProofSearch(**kwargs)` with `TypeError: __init__() got an unexpected
keyword argument`.

## Fix applied

Edited
`/home/hugo/codes/mol-metal/molmetal/molmetal_lam/search_alg/proof_search.py`:

1. Added the module-level constant after `ScorerFn` so it is exported
   at the same place as `heuristic` / `SymbolicPrior`:

   ```python
   EXPLORATION_STRATEGIES: Tuple[str, ...] = (
       "dirichlet", "score_repellent", "variance",
   )
   ```

2. Added four backward-compatible dataclass fields to
   `MCTSProofSearch` (defaults preserve the existing R17/R18 search
   behaviour):

   ```python
   exploration_strategy: str = "dirichlet"
   c_var: float = 0.5
   n_min_visits: int = 2
   debug: bool = False
   ```

No existing logic in `_simulate` / `_select` / `_expand` /
`_backprop` was modified — these are *new opt-in knobs* that the joint
harness and the Wf-36 Phase-2/3 ship surface pass through, with safe
defaults that leave the legacy path bit-for-bit identical.

## Tests after

```
$ uv run --no-sync python -m pytest molmetal/tests/test_joint_3stage_mcts.py -v
============================= test session starts ==============================
collected 15 items

molmetal/tests/test_joint_3stage_mcts.py::TestJointPipelineImports::test_joint_imports PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestJointPipelineImports::test_joint_defaults PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestJointPipelineImports::test_joint_exploration_strategies_contract PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestStage1ScoreRepellent::test_stage1_score_repellent PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestStage1ScoreRepellent::test_stage1_score_repellent_deterministic PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestStage2VariancePuct::test_stage2_variance_puct PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestStage2VariancePuct::test_stage2_variance_puct_handles_zero_prior PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestStage3MctsTrajectory::test_stage3_mcts_trajectory PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestStage3MctsTrajectory::test_stage3_mcts_trajectory_handles_failure PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestStage4ReplayWrite::test_stage4_replay_write PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestStage4ReplayWrite::test_stage4_replay_write_empty_candidates PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestFullPipeline::test_full_pipeline_smoke PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestFullPipeline::test_full_pipeline_replay_buffer_populated PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestFullPipeline::test_full_pipeline_deterministic PASSED
molmetal/tests/test_joint_3stage_mcts.py::TestFullPipeline::test_full_pipeline_stage_wall_time_pct_sums_to_one PASSED

======================= 15 passed, 2 warnings in 50.25s =======================
```

## Cross-check (non-blocking observation)

The sibling module
`molmetal/tests/test_variance_mcts_proof_search.py` was *already broken*
in this same refactor (3 failures on `_build_argparser` import — a
separate, pre-existing issue). My edit unblocks 17 of its 20 tests
(`EXPLORATION_STRATEGIES`, `exploration_strategy`, `c_var`,
`n_min_visits` are now resolvable); the remaining 3 still need
`_build_argparser` to be re-introduced (out of scope for Wf-290F3).

## Files touched

- `/home/hugo/codes/mol-metal/molmetal/molmetal_lam/search_alg/proof_search.py`
  (added `EXPLORATION_STRATEGIES` constant + 4 dataclass fields on
  `MCTSProofSearch`)

## Report path

`/home/hugo/codes/mol-metal/molmetal/reports/wf290f3_fix/fix_summary.md`