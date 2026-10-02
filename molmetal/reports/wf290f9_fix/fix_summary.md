# Wf-290F9 — Fix `test_variance_mcts_proof_search`

## §1 — Error before fix

**Original error** (per Wf-268C §4, TODO/completed/160_p0_blockers_fixed.md §Pre-flight):

```
ImportError: cannot import name 'EXPLORATION_STRATEGIES' from
'molmetal_lam.search_alg.proof_search'
(/home/hugo/codes/mol-metal/molmetal/molmetal_lam/search_alg/proof_search.py)
```

Triggered at `molmetal/tests/test_variance_mcts_proof_search.py:34` and
the fallback path at `:49`.

## §2 — Diagnosis

Two root causes — both pre-existing upstream breakage:

1. **`EXPLORATION_STRATEGIES` constant missing at module top-level.**
   Test file imports it at lines 35 and 50 (`from ... import (
   EXPLORATION_STRATEGIES, MCTSProofSearch,)`); the constant was not
   defined in `proof_search.py` despite being referenced from
   `molmetal/scripts/joint_score_repellent_variance_puct_mcts.py`,
   `molmetal/scripts/verify_variance_mcts_proof_search.py`,
   `molmetal/tests/test_variance_mcts_proof_search.py`, and
   `molmetal/tests/test_joint_3stage_mcts.py`. Symbol not exposed
   (TODO-160 P0 classification).

2. **`_build_argparser` function missing.**
   Three tests in
   `TestProofSearchArgparseStrategy` (`test_argparse_exploration_strategy`,
   `test_argparse_c_var_flag`, `test_argparse_other_flags`) all
   `from molmetal_lam.search_alg.proof_search import _build_argparser`.
   The function did not exist in `proof_search.py`; a docstring at
   line 3317 referenced `_build_argparser` for the `--debug` CLI but
   no implementation was present.

## §3 — Fix applied

File: `/home/hugo/codes/mol-metal/molmetal/molmetal_lam/search_alg/proof_search.py`

1. Added `import argparse` (alphabetised near `logging`/`math`).
2. Added module-level `EXPLORATION_STRATEGIES: Tuple[str, ...] =
   ("dirichlet", "score_repellent", "variance",)` with the Wf-36
   Phase-2 docstring (3 strategies, default `"dirichlet"`,
   `c_var = 0.5`, see verify harness).
3. Added private constants `_DEFAULT_SEED=2025`, `_DEFAULT_SIMS=30`,
   `_DEFAULT_DEPTH=3`, `_DEFAULT_N_TRAJ=100`, `_DEFAULT_C_VAR=0.5`.
4. Added `_build_argparser() -> argparse.ArgumentParser` covering the
   module-level CLI contract used by the test suite:

   | flag                  | type  | default                  | choices                |
   |-----------------------|-------|--------------------------|------------------------|
   | `--exploration-strategy` | str | `dirichlet`              | `EXPLORATION_STRATEGIES` |
   | `--c-var`             | float | `0.5`                    | —                      |
   | `--seed`              | int   | `2025`                   | —                      |
   | `--sims`              | int   | `30`                     | —                      |
   | `--depth`             | int   | `3`                      | —                      |
   | `--n-trajectories`    | int   | `100`                    | —                      |

   Default values match `verify_variance_mcts_proof_search.py`'s
   `_build_parser` so the joint / verify harnesses remain consistent.

## §4 — Tests after fix

```
molmetal/tests/test_variance_mcts_proof_search.py ........ 20 PASSED in 52.73s
```

Breakdown:

- `TestProofSearchStrategyImport` — 5/5 PASS
  (`test_proof_search_imports_with_variance`,
  `test_exploration_strategies_constant`,
  `test_default_strategy_is_dirichlet`, `test_c_var_default`,
  `test_n_min_visits_default`)
- `TestProofSearchArgparseStrategy` — 3/3 PASS
  (`test_argparse_exploration_strategy`, `test_argparse_c_var_flag`,
  `test_argparse_other_flags`)
- `TestProofSearchScoreRepellentRun` — 3/3 PASS
- `TestProofSearchVariancePuctRun` — 5/5 PASS
- `TestProofSearchDirichletBaseline` — 4/4 PASS
  (incl. `test_all_three_strategies_run`)

Side-effects verified:
- `test_joint_3stage_mcts.py` collects 15 tests with no
  `ImportError` on `EXPLORATION_STRATEGIES` (was also broken
  pre-fix).
- `python -c "from molmetal_lam.search_alg.proof_search import
  _build_argparser; _build_argparser().parse_args(['--exploration-strategy',
  'variance', '--c-var', '0.7'])"` → `Namespace(exploration_strategy='variance',
  c_var=0.7, seed=2025, sims=30, depth=3, n_trajectories=100)`.

## §5 — Files touched

- `/home/hugo/codes/mol-metal/molmetal/molmetal_lam/search_alg/proof_search.py`
  — added `import argparse`, defined `EXPLORATION_STRATEGIES`, default
  constants, and `_build_argparser()`.

## §6 — Honest framing

- This is a **pre-existing collection error** noted by Wf-268C §4 and
  TODO-160 ("Symbol not exposed at module top-level").  No
  regressions introduced.
- Test runtime is `52.73s` for 20 tests; the bulk is end-to-end
  `MCTSProofSearch.search()` runs on the mock chemistry fixture.
- No source changes to `MCTSProofSearch` semantics — purely
  surface-level exposure of `EXPLORATION_STRATEGIES` + the
  previously-undefined `_build_argparser`.

## §7 — Artifact paths

- This report:
  `/home/hugo/codes/mol-metal/molmetal/reports/wf290f9_fix/fix_summary.md`
- Source file modified:
  `/home/hugo/codes/mol-metal/molmetal/molmetal_lam/search_alg/proof_search.py`
- Test file under fix:
  `/home/hugo/codes/mol-metal/molmetal/tests/test_variance_mcts_proof_search.py`
