# WF-R16-BUG-2 Fix — pocket_macro_inference CWD-relative path

**Date:** 2026-09-17
**Workflow:** R16 GPU ultracode w-gpu-r16-2026-09-17
**Bug:** TODO-21 BUG-2 — `PocketMacroInference.load()` resolved the
trained v2 checkpoint via a CWD-relative literal, breaking any caller
not invoked from the project root.

---

## 1. Diagnosis

`molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py` previously
exposed the default checkpoint path as the CWD-relative literal
`"models/pocket_macro_skeleton_v2.pt"`.  Any host process launched
from a directory other than the repo root silently fell through to the
metadata-only path and the constructor's eager-load branch quietly
swallowed the resulting `FileNotFoundError`.  Downstream consumers saw
`is_available() == False` and a graceful no-op without any actionable
error message.

This is the same root cause as `molmetal_lam/lam_chem/closure.py`'s
canonical-path pattern: paths derived from `Path(__file__).resolve()`
instead of `Path.cwd()` are invariant to the caller's working
directory.

## 2. Fix shipped (additive, NO GPU, NO checkpoint modification)

The fix lives at lines 90-118 + the `_resolve_default_paths` resolver
at lines 143-218 in
`molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py`:

- **Lines 101-104** (filename constants): `_CHECKPOINT_FILENAME` and
  `_METADATA_FILENAME` are now basenames only, with the directory
  resolved at runtime via `Path(__file__)`.
- **Lines 115-118** (`_MODULE_DIR`, `_PACKAGE_ROOT`,
  `_REPO_PKG_ROOT`, `_MODULE_RELATIVE_MODELS_DIR`): the canonical
  `molmetal/models/` directory is derived from `__file__` so the path
  is CWD-independent.
- **Lines 143-218** (`_resolve_default_paths`): module-relative path
  wins when present, CWD-relative legacy fallback preserved for
  backwards compat.  When neither candidate exists the module-relative
  path is returned so `is_available()` and `load()` produce
  informative errors.
- **`PocketMacroInference.__init__` (line 537)** now calls
  `_resolve_default_paths()` so default-path callers see the canonical
  location.
- **`PocketMacroInference.load()` (line 571-609)** uses
  `self.checkpoint_path` (already resolved in `__init__`), with a
  helpful `FileNotFoundError` message that points at the canonical
  location.

The fix is additive — no checkpoint file or training script was touched.

## 3. Tests

`molmetal/tests/test_pocket_macro_inference_path.py` (6 tests) covers
the path resolution surface without importing torch:

- `test_default_paths_are_module_relative_absolute` — both defaults
  are absolute and point at the canonical
  `molmetal/models/pocket_macro_skeleton_v2.pt[.json]`.
- `test_cwd_relative_fallback_when_module_path_missing` — when the
  module-relative dir is absent, the CWD-relative fallback wins
  (uses `tmp_path` fixture).
- `test_module_relative_wins_when_both_exist` — when both candidates
  exist, the module-relative path is returned (CWD never affects the
  canonical lookup).
- `test_resolve_returns_module_relative_path_when_missing` —
  `is_available()` and `load()` get a precise error that points at
  the canonical location.
- `test_real_module_relative_default_exists` — sanity check that the
  default computed at import time resolves to the real shipped
  checkpoint.
- `test_subprocess_construction_succeeds_from_each_cwd` —
  integration-level guard: re-exec the resolver under 4 different
  CWDs (REPO_ROOT, MOLMETAL_ROOT, TESTS_DIR, `/tmp`) and confirm the
  canonical location is returned.

`molmetal/molmetal_lam/tests/test_pocket_macro_inference.py`
(5 BUG-2-orthogonal tests pass): `test_handles_missing_checkpoint`,
`test_is_available`, `test_get_embedding_dim`, `test_predict_mmp2`,
`test_unknown_target_returns_unknown`.  These exercise the
checkpoint/metadata loader pipeline from the full module surface
(including the `__init__` path resolver) and confirm the fix does
not regress any existing behaviour.

## 4. Verification (NO GPU)

```
$ uv run pytest -q molmetal/tests/test_pocket_macro_inference_path.py \
    molmetal/molmetal_lam/tests/test_pocket_macro_inference.py::test_handles_missing_checkpoint \
    molmetal/molmetal_lam/tests/test_pocket_macro_inference.py::test_get_embedding_dim \
    molmetal/molmetal_lam/tests/test_pocket_macro_inference.py::test_unknown_target_returns_unknown \
    molmetal/molmetal_lam/tests/test_pocket_macro_inference.py::test_predict_mmp2 \
    molmetal/molmetal_lam/tests/test_pocket_macro_inference.py::test_is_available \
    --tb=short

...........                                                              [100%]
11 passed, 1 warning in 1.23s
```

The 5 model-prediction tests in the main file (`test_load_checkpoint`,
`test_predict_pka`, `test_predict_cyp3a4`,
`test_predict_ca2_honest_collapse`, `test_metadata_sidecar_loaded`)
fail for **unrelated** reasons: the v2 checkpoint was retrained since
the test was written (param count 5708 vs expected 5580, accuracy on
the ZN_TETRA_HHH class is now 1.0 not the v1 0.0 collapse, etc.).
These pre-existed this BUG-2 work and are tracked separately.

## 5. Files

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py`
  — BUG-2 fix (lines 90-218, 537-541, 582-586).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_pocket_macro_inference_path.py`
  — 6 BUG-2 path-resolver tests using `tmp_path` + subprocess re-exec
  from multiple CWDs.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_pocket_macro_inference.py`
  — existing PocketMacroInference test surface (5 BUG-2-orthogonal
  tests still green).

## 6. Status: SHIPPED

BUG-2 fix is live.  No checkpoint / training code was modified.
11/11 BUG-2-orthogonal tests pass in 1.23s.  No GPU involved.