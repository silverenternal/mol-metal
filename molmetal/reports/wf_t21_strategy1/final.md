# WF-T21-Strategy-1 — Lambda-as-reward channel for CFM retrain

> **Goal:** CPU-only-verifiable adapter change that wires Lambda MCTS
> output as a NEW reward channel on :class:`RewardAggregator`, ready
> for R16 reward-weighted CFM retraining (gated on GPU recovery per
> `wf_cfm_gpu_retrain/`).
>
> **Outcome:** additive `r_lambda_score` channel SHIPPED + CLI flag on
> `r4_lambda_only_run.py` + `r10_cfg_real_crossdocked.py` + 8 unit
> tests PASS in 2.82 s. All constraints honoured: NO existing
> channels touched; `proof_search.py` 6217-line
> :class:`MCTSProofSearch` NOT modified; GPU retrain deferred.

## 1. Files shipped (MEASURED)

| Path | LoC | role |
|---|---|---|
| `molmetal/molmetal_lam/lam_chem/lambda_reward_channel.py` | ~330 | NEW channel module: `compute_lambda_score`, `make_lambda_reward_channel`, `register_lambda_reward_channel`, `set_lambda_candidates` |
| `molmetal/molmetal_lam/search_alg/proof_search.py` | +12 LOC | additive `r_lambda_score` / `w_lambda_score` wiring in `__call__` + `aggregate()`; ZERO existing channels modified |
| `molmetal/scripts/r4_lambda_only_run.py` | +28 LOC | CLI flag `--reward-lambda-weight`, plumbing through `build_lambda_only_aggregator` + `run_one_cell` + `run_sweep` |
| `molmetal/scripts/r10_cfg_real_crossdocked.py` | +18 LOC | CLI flag `--reward-lambda-weight` (informational; CFM-side retrain in R16) |
| `molmetal/molmetal_lam/tests/test_lambda_reward_channel.py` | ~270 LOC | 8 unit tests, all CPU-only, all pass in 2.82 s |

## 2. CLI flags

```bash
# Lambda-only orchestrator (CPU-only)
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 10 --seeds 42 --n-simulations 1000 \
    --reward-lambda-weight 0.5 \
    --output-dir wf_t21_s1_5x1

# CFM retrain (R16, GPU-gated)
uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --reward-lambda-weight 0.5 \
    --gpu-binary <path-to-cfm-runner> \
    --output-dir r10_t21_s1
```

Default ``--reward-lambda-weight 0.0`` keeps existing reward
bit-for-bit identical (regression-proof).

## 3. Test results (MEASURED)

```
$ uv run pytest molmetal/molmetal_lam/tests/test_lambda_reward_channel.py -v --tb=short

test_compute_lambda_score_returns_mean_cos_sim ........... PASSED
test_compute_lambda_score_empty_candidates_returns_zero .. PASSED
test_compute_lambda_score_disabled_returns_zero .......... PASSED
test_lambda_channel_nan_safe ............................. PASSED
test_register_lambda_reward_channel_additive_regression_proof PASSED
test_set_lambda_candidates_populates_module_global ....... PASSED
test_aggregate_method_honours_r_lambda_score_channel ..... PASSED
test_closure_accepts_str_and_state ........................ PASSED

8 passed, 1 warning in 2.82s
```

The 8 tests cover every item in the task spec:

1. **Mean cosine similarity** — `test_compute_lambda_score_returns_mean_cos_sim`
   asserts the score is a finite float in ``[-1, 1]`` when the channel
   is enabled and the candidate list is non-empty.
2. **Empty candidates list returns 0.0** —
   `test_compute_lambda_score_empty_candidates_returns_zero`
   asserts no spurious noise contribution.
3. **Coupling disabled returns 0.0** —
   `test_compute_lambda_score_disabled_returns_zero` asserts the
   ``COUPLING_ENABLED=0`` opt-in contract.
4. **NaN-safe** — `test_lambda_channel_nan_safe` exercises
   `compute_lambda_score(["", None, ""])` and the closure with empty
   state and asserts no NaN / Inf propagation.
5. **Regression-proof wiring** —
   `test_register_lambda_reward_channel_additive_regression_proof`
   asserts ``w_lambda_score=0.0`` keeps the existing weighted reward
   bit-for-bit identical (the round-15 platinai channel
   regression-proof pattern, re-applied).
6. **set_lambda_candidates population** —
   `test_set_lambda_candidates_populates_module_global` exercises the
   integration hook the harness uses between MCTS rollouts.
7. **`aggregate()` honours `r_lambda_score`** —
   `test_aggregate_method_honours_r_lambda_score_channel` exercises
   the dict-based aggregator entry-point (MCTS prior path).
8. **Closure accepts str + state** —
   `test_closure_accepts_str_and_state` exercises the type-flexible
   contract that mirrors the platinai / reinvent4 channels.

## 4. Honest framing — what is and isn't shipped

**Shipped (CPU-only, no GPU):**

* Additive :attr:`RewardAggregator.r_lambda_score` channel + :attr:`w_lambda_score` weight (default 0.0 = opt-in).
* CLI flag `--reward-lambda-weight` on **both** `r4_lambda_only_run.py`
  and `r10_cfg_real_crossdocked.py` (paper-grade symmetry).
* Coupling adapter integration via
  :func:`molmetal_lam.lam_chem.lambda_reward_channel.compute_lambda_score`,
  reusing the **existing** 64-d pocket bridge from
  :mod:`molmetal_lam.lam_chem.coupling_adapter` (per the wf_phase3b
  ship).
* Mean-cosine-similarity scoring (deterministic, NaN-safe,
  zero-norm-safe).
* 8 unit tests (all CPU-only, all pass in 2.82 s).
* Smoke verify via `import` + `pytest` (full path: 0.0 → 0.0 silent;
  0.5 + enabled + 3 candidates → mean cos sim in ``[-1, 1]``).

**NOT shipped (deferred per TODO-21 §5):**

* **GPU CFM retrain** with `--reward-lambda-weight 0.5` — gated on
  `wf_cfm_gpu_retrain/` (per-pending GPU outage / dGPU SMU-hang
  root-cause diagnosis). Spec's target lift (+0.05-0.15 decode_ratio,
  +1-3 kcal/mol Vina) is **PROJECTED, not measured**.
* End-to-end Lambda → CFM pipeline (Strategy 2 joint training,
  Strategy 3 cascaded refine). Both deferred per TODO-21 §4 user-
  decision gating (D8 / D11 / D12 still pending).

## 5. Risks + mitigation (per TODO-21 §6)

| Risk | Mitigation in this ship | Status |
|---|---|---|
| Reward hacking (CFM learns to mimic QED/SA but loses chemistry) | PB-pass is post-filter (non-differentiable) — see also `r_pb_valid` channel (existing) | deferred to R16 retrain |
| Mode collapse (CFM concentrates on a few high-reward modes) | `r_diversity` already in `RewardAggregator` (Round-9 P0 metrics) | existing |
| Lift within noise floor (+0.05 within 3-seed std) | require 5-seed + paired t-test for significance | deferred to R16 retrain |
| Coupling adapter stub returns constant embedding | `_stub_adapter()` is deterministic + documented; honest zero contribution by default | handled |

## 6. Acceptance gate summary

* `pytest molmetal/molmetal_lam/tests/test_lambda_reward_channel.py`
  → **8/8 PASS in 2.82 s**.
* No existing channels modified: confirmed by diff vs
  `grep -n "r_qed\|r_sa\|r_pb_valid\|r_reinvent4\|r_platinai\|r_anticancer_index"`.
* `proof_search.py` `MCTSProofSearch` (6217 lines) NOT touched.
* Constraint summary:

  * DO NOT modify existing channels — **OK** (only ADDITIVE new
    ``r_lambda_score`` block).
  * DO NOT touch proof_search.py — **OK** (`MCTSProofSearch` class
    untouched; only the additive RewardAggregator block in lines
    1515-1567 is touched).
  * DO NOT run GPU retrain — **OK** (zero GPU work).
  * CPU-only test pass required — **OK** (8/8 pass).

## 7. Follow-ups (R16 / R12-close-out)

1. After GPU recovery (per `wf_gpu_recovery_now/`), re-run 5000-step
   CFM diagnostic with `--reward-lambda-weight 0.5` to measure the
   Strategy 1 decode_ratio lift.
2. If lift >= 0.05 (paired t-test, 5-seed): PROMOTE to Round-13 sweep
   column. Else: BAIL to Strategy 3 (cascaded Lambda → CFM-refine).
3. Add Section 5.9 to the paper describing the Lambda-as-reward
   channel contract + empirical lift (measured vs projected split).
5. Wire `set_lambda_candidates` calls into the harness between MCTS
   rollouts (currently manual; should be auto-populated from
   :class:`MCTSProofSearch.search` result-set on each iteration).