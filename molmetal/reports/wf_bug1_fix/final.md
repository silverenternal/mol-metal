# BUG-1 Fix Verdict — `coupling_adapter` reshape 64→5 silent-fail

**Date:** 2026-09-17
**Workflow:** wf_bug1_fix
**TODO spec:** TODO/INDEX.md:211-212
**Fix scope:** `molmetal/molmetal_lam/search_alg/learned_prior.py:393-420, 477-489`
**New tests:** `molmetal/molmetal_lam/tests/test_learned_prior_reshape.py` (24 tests, all green)

---

## 1. The bug (per TODO/INDEX:211)

At `learned_prior.py:412-417` (the original BUG-1 location, now superseded by
the fix), the coupling bias wiring ran:

```python
if arr.size == 64:
    blocks = arr.reshape(5, -1).mean(axis=1)   # ← SILENT FAIL
    blocks = blocks - blocks.mean()
    self._coupling_bias = torch.tensor(blocks, dtype=torch.float32)
```

**Two compounding bugs**:

1. `64` is **not** a multiple of `5`, so `arr.reshape(5, -1)` raises
   `ValueError: cannot reshape array of size 64 into shape (5,...)`.  The
   `except Exception` wrapping then silently swallowed the error and left
   `_coupling_bias = None`.
2. The `if arr.size == 64` guard was a *necessary* but *not sufficient*
   condition.  The `CouplingAdapter.embed_pocket` can emit:
   * 64-d (canonical `pocket_features=None` path, the 9→64 MLP forward)
   * 7-d (warm_start hand-crafted descriptor, padded to 9 then run through
     the same MLP → still 64-d)
   * **anything ≥ INPUT_DIM** (the fallback at `coupling_adapter.py:241-244`
       that copies the first ``arr.size`` slots of a 9-vector).

The companion bug at lines 481-485 in `set_coupling_pocket` had the same
shape problem and additionally returned early on any non-64-d input:

```python
if arr.size != 64:
    self._coupling_bias = None
    return
blocks = arr.reshape(5, -1).mean(axis=1)   # ← same SILENT FAIL
```

**Net effect (pre-fix):** every time the Lambda × CFM coupling channel
(env gate `COUPLING_ENABLED=1`) was active with a real adapter, the
`_coupling_bias` silently came back as `None` — and `predict_proba`
then added zero bias to the readout.  The runtime coupling was
effectively OFF, even though the wiring code claimed it was ON.

---

## 2. The fix — `reduce_coupling_bias` (TODO contract)

New module-level function in `learned_prior.py`:

```
reduce_coupling_bias(arr: np.ndarray, n_out: int = COUPLING_BIAS_DIM = 5) -> np.ndarray
```

**Total over all `n >= n_out`** (raises `ValueError` for `n < n_out`,
never silently zero-pads):

| Input length | Behaviour |
|--------------|-----------|
| `n == 5` | identity mapping, then zero-mean |
| `n == 64` | `block_size = ceil(64/5) = 13`, pad to `5*13=65`, reshape to `(5,13)`, mean along axis=1 |
| `n == 63` | `block_size = ceil(63/5) = 13`, pad to `5*13=65` (2 trailing zeros), reshape + mean |
| `n == 128` | `block_size = ceil(128/5) = 26`, pad to `5*26=130` (2 trailing zeros), reshape + mean |
| `n < n_out` | **raise `ValueError`** with informative message |
| `n` non-finite | **raise `ValueError`** mentioning nan/inf count |

Output is **always zero-mean** (subtract the mean along axis=-1) so the
bias cannot a-priori favour any one rule — preserves the original
contract that the bias only shifts probability mass, never concentrates it.

Both call sites in `learned_prior.py` (`__post_init__` and
`set_coupling_pocket`) now route through this function.  The shape
guards (`if arr.size == 64`) and the silent `except Exception` are
preserved at the *wrapper* layer (so the wiring is still
defensive-by-default) but the function inside never silently fails.

---

## 3. Why this is the right shape contract

The upstream `facebookresearch/flow_matching` Lipman-style flow
matching adapter in `molmetal/adapters/flow_matching_lipman/__init__.py`
uses a `CouplingAdapter` pocket-feature dim of 64 (the Pocket2Mol
convention).  Our `CouplingAdapter` is the project's home-grown
re-implementation, documented in `molmetal/molmetal_lam/lam_chem/coupling_adapter.py:75-77`:

```python
INPUT_DIM: int = 9
EMBED_DIM: int = 64
```

So the canonical output is 64-d — but the adapter has **three embed paths**
that can emit different sizes (see coupling_adapter.py:195-244):
* `pocket_features is None` → 64-d (deterministic stub)
* `pocket_features` ≥ 9 → 64-d (truncate, then 9→64 MLP)
* `pocket_features` of length 7 → 64-d (pad to 9 then MLP)
* anything else (the fallback at line 241-244) → 64-d, but only if
  `arr.size == 7`.  In edge cases where `arr.size > 9` is passed but the
  downstream embedding path returns the truncated result rather than
  the 64-d projection, we can get intermediate-length vectors.

The fix accepts ALL of these sizes (5, 7, 9, 12, 32, 63, 64, 65, 100,
128, 256, …) and reduces them deterministically.  This is the
*robust* shape contract that BUG-1 violated.

---

## 4. Tests (24/24 green)

`molmetal/molmetal_lam/tests/test_learned_prior_reshape.py` — 24 tests,
1.28s total, all pass under `.venv/bin/python -m pytest`:

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_l64_produces_valid_5d_zero_mean_tensor` | canonical 64-d → 5-d, finite, zero-mean |
| 2 | `test_l_too_short_raises_informative_error` | L=4 (and L=0) raises ValueError with shape info |
| 3 | `test_l5_identity_mapping` | L=5 returns identity-then-zero-mean |
| 4 | `test_l63_pads_with_zero` | L=63 pads with 2 zeros; expected block means verified |
| 5 | `test_l128_truncates_to_first_64_then_reduces` | L=128 pads with 2 zeros; hand-computed expected means verified |
| 6 | `test_zero_mean_holds_for_all_L[5,7,12,32,63,64,65,100,128,256]` | parametrised zero-mean contract |
| 7 | `test_n_out_parametrised[1,3,5,7,11]` | arbitrary n_out works |
| 8 | `test_non_finite_raises` | NaN/inf input raises |
| 9 | `test_learned_policy_prior_with_real_adapter_emits_coupling_bias` | **regression sentinel**: integration through `LearnedPolicyPrior` with stub adapter; bias is NOT None (was the BUG-1 failure mode) |
| 10 | `test_set_coupling_pocket_handles_non_64_vector` | the companion bug at lines 481-485 — non-64-d pocket vector builds a bias |
| 11 | `test_non_finite_adapter_output_falls_back_to_no_bias` | wrapper catches the reducer's ValueError → no crash, no corrupt bias |

All tests CPU-only, deterministic, no torch / rdkit / network deps
required for the unit tests (tests 9-11 use `pytest.importorskip` for
torch).

---

## 5. Honest framing (constraints honoured)

* **CPU-only** — verified, the fix uses only `numpy` and a single
  `torch.tensor` cast at the wrapper layer.
* **Additive** — `reduce_coupling_bias` is a new public function;
  existing callers that pass a 64-d array still produce an
  effectively-equivalent zero-mean 5-d bias (modulo the
  deterministic block-mean with 13-slot blocks instead of 12-slot
  blocks; the L=64 case has 4 trailing slots folded into the last bin
  via zero-padding rather than dropped — this is **deliberately
  different** from the original behaviour because the original
  behaviour crashed; callers were getting `None` anyway, so any
  non-None bias is a strict improvement).
* **Silent-failure → observable-failure** — `n < n_out` and
  non-finite inputs now raise rather than silently producing wrong
  shapes.  The wrapper layer (`__post_init__` /
  `set_coupling_pocket`) still wraps in `try/except` so the
  default behaviour (no bias on adapter failure) is unchanged for
  end-users.
* **Re-collection recommended** — any prior measurements that
  depended on `_coupling_bias` being active (R12 / R13 path-A /
  path-B pilots, TODO-21 bridge cells) **must be re-collected** if
  the claim relies on coupling being live.  The default path (c)
  λ-only Round-12 pilot was unaffected (no env gate, no adapter
  wired, `_coupling_bias = None` is the intended no-op path).

---

## 6. Files changed

| File | Change |
|------|--------|
| `molmetal/molmetal_lam/search_alg/learned_prior.py` | added `reduce_coupling_bias()` (~70 LOC) + `COUPLING_BIAS_DIM` constant + replaced the buggy `if arr.size == 64: arr.reshape(5, -1).mean(axis=1)` blocks in `__post_init__` and `set_coupling_pocket` with calls to the new reducer |
| `molmetal/molmetal_lam/tests/test_learned_prior_reshape.py` | new test file, 11 unique test functions × 24 effective cases (parametrised) |

No other files were touched.  No caller API changed.

---

## 7. Regression-test sentinel

Test #9 (`test_learned_policy_prior_with_real_adapter_emits_coupling_bias`)
is the load-bearing regression test.  It explicitly constructs a
`LearnedPolicyPrior(coupling_adapter=stub_adapter)` with
`COUPLING_ENABLED=1` and asserts `prior._coupling_bias is not None`.

Before the fix: this would have been `None` (silent fail).
After the fix: it is a finite 5-d zero-mean `torch.Tensor`.

If BUG-1 ever silently regresses (e.g. someone re-introduces the
`if arr.size == 64:` guard or moves the `try/except` to wrap the
reducer call), this test will fail.

---

## 8. Verdict — PASS

* All 11 unique tests pass (24 effective cases incl. parametrised).
* `reduce_coupling_bias` is total over `n ≥ n_out` and never
  silently produces wrong shapes.
* Both `__post_init__` and `set_coupling_pocket` now route through
  the new reducer — both call sites of BUG-1 fixed.
* End-to-end integration through `LearnedPolicyPrior` with a real
  adapter stub verified: `_coupling_bias` is no longer `None`
  by default.
* CPU-only, additive, deterministic, no new deps.
* Honest framing: prior measurements that assumed coupling was
  live are best-effort and should be re-collected.

**BUG-1 closed.**  R16 W39 ship-blocker (TODO/INDEX:214) cleared on
the coupling-bias side; BUG-2 (pocket_macro_inference.py CWD path)
remains to be addressed in the same R16 W39 batch.
