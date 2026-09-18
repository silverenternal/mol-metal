# WF-R16-BUG-1 Fix Verdict — coupling_adapter reshape 64→5 silent-fail

Date: 2026-09-17
Workflow: r16-gpu-execution-2026-09-17 → BUG-1 (CPU pre-flight)
Author: R16 GPU ultracode pre-flight

## Scope

Per R15 verifier (TODO/INDEX:211 / 2026-09-16): the learned_prior.py
coupling wiring applied `arr.reshape(5, -1).mean(axis=1)` unconditionally
to the 64-d pocket vector returned by `CouplingAdapter.embed_pocket`.
Since 64 is not a multiple of 5, the reshape silently raised
`ValueError: cannot reshape array`; the surrounding `except Exception`
swallowed the error and left `_coupling_bias = None`, masking the
coupling channel on every pocket.

## Pre-existing fix state

The fix is **already merged** in `molmetal/molmetal_lam/search_alg/learned_prior.py`:

1. `reduce_coupling_bias(arr, n_out=…)` defined at lines 142-233 —
   total over all sizes ≥ n_out.  Handles L=64 via block-mean with
   zero-pad-then-mean (block_size=ceil(64/5)=13, n_used=65, 1 zero
   pad, reshape (5,13) and mean).
2. `LearnedPolicyPrior.__post_init__` (lines 506-536) routes through
   the reducer and narrows the catch to `(TypeError, ValueError)`
   with `_logger.warning` on fallback — no silent failure.
3. `LearnedPolicyPrior.set_coupling_pocket` (lines 593-614) mirrors
   the same pattern for the per-pocket refresh path.

## R16 work performed

The fix and test scaffold were already in place from the R15
verifier.  R16 added one small cleanup:

* Patched `tests/test_learned_prior_reshape.py::test_non_finite_adapter_output_falls_back_to_no_bias`
  to drop a buggy `_logging.captureWarnings(True)` context manager
  that broke on this Python build (`'NoneType' object does not
  support the context manager protocol`).  The test's intent — verify
  the wrapper catches non-finite input without crashing — is preserved
  unchanged.

Constraints honoured: NO GPU.  Additive only.  No edits to BUG-2
(import) or unrelated code paths.

## Test results

```
$ uv run pytest -q molmetal/molmetal_lam/tests/test_learned_prior_reshape.py \
                 molmetal/tests/test_coupling_adapter.py --tb=short
...........................................                              [100%]
43 passed, 1 warning in 2.14s
```

Breakdown:
* `test_learned_prior_reshape.py` — 25 tests pass (12 BUG-1-specific +
  the rest covering L=5/7/12/32/63/64/65/100/128/256 sizes, identity,
  zero-mean, non-finite raises, n_out parametrisation, integration smoke).
* `test_coupling_adapter.py` — 18 tests pass (tmqm dry-run checkpoint,
  embed_pocket 64-d output, env-gated wiring, LearnedPolicyPrior
  end-to-end with stub adapter).

## File:line references

* Source fix: `molmetal/molmetal_lam/search_alg/learned_prior.py:142-233`
  (`reduce_coupling_bias` total reducer) and `:506-536` (init wiring)
  and `:593-614` (`set_coupling_pocket` mirror).
* Coupling adapter: `molmetal/molmetal_lam/lam_chem/coupling_adapter.py:195-244`
  (`embed_pocket` 64-d emission; the `padded[:arr.size] = arr` fallback
  in line 244 motivates the dimension-safe reducer).
* Tests: `molmetal/molmetal_lam/tests/test_learned_prior_reshape.py`
  and `molmetal/tests/test_coupling_adapter.py`.

## Verdict

BUG-1 fix is **complete and verified**:

* Behaviour is total over all sizes ≥ n_out (no silent ValueError).
* Wrapper narrows the catch to `(TypeError, ValueError)` and logs a
  warning on fallback — observable, not silent.
* Zero-mean by construction preserves the "uniform-untrained" prior
  contract.
* 43/43 tests pass; CPU-only; backward-compatible.

R16 may now proceed to GPU-bound work (this was the pre-flight gate).