# WF-BUG-2 fix verdict — pocket_macro_inference CWD-relative path

## TL;DR

**Fixed.** `pocket_macro_inference.py` no longer relies on the
CWD-relative literal `"models/pocket_macro_skeleton_v2.pt"`.  Defaults
now resolve via `Path(__file__).resolve().parent.parent / "models" /
pocket_macro_skeleton_v2.pt` (module-relative), with a graceful
fallback to the legacy CWD-relative path for backwards compatibility.
6/6 new tests pass (4+ required), existing behaviour preserved.

## Bug

Per `TODO/INDEX.md:212`:

> **BUG-2**: `pocket_macro_inference.py:101,104` Deflex v2 checkpoint
> CWD-relative path (5-line fix)

The v2 checkpoint paths were hard-coded as:

```python
DEFAULT_CHECKPOINT_PATH: str = "models/pocket_macro_skeleton_v2.pt"
DEFAULT_METADATA_PATH: str = "models/pocket_macro_skeleton_v2.pt.json"
```

These break whenever the host script is run from any directory other
than the project root — `is_available()` silently returns `False`,
the model weights never load, and downstream consumers fall through
to the metadata-only path.  R15 cross-verifier caught this as a 5-line
ship-blocker for R16 W39.

## Fix

Replaced the CWD-relative literals with a module-relative
`Path(__file__)` resolution that walks `lam_chem → molmetal_lam →
molmetal → models`:

```python
_MODULE_DIR: Path = Path(__file__).resolve().parent
_PACKAGE_ROOT: Path = _MODULE_DIR.parent                # molmetal_lam/
_REPO_PKG_ROOT: Path = _MODULE_DIR.parent.parent         # molmetal/
_MODULE_RELATIVE_MODELS_DIR: Path = _REPO_PKG_ROOT / "models"

DEFAULT_CHECKPOINT_PATH: str = str(
    _MODULE_RELATIVE_MODELS_DIR / _CHECKPOINT_FILENAME
)
DEFAULT_METADATA_PATH: str = str(
    _MODULE_RELATIVE_MODELS_DIR / _METADATA_FILENAME
)
```

The `PocketMacroInference.__init__` now invokes
`_resolve_default_paths()` at construction time to find an existing
checkpoint in this order:

1. **Module-relative** (preferred): `<molmetal>/models/pocket_macro_skeleton_v2.pt`
2. **CWD-relative legacy fallback**: `models/pocket_macro_skeleton_v2.pt`
   (preserved for backwards compatibility — older scripts that cd
   into the project root continue to work)
3. If neither exists, the module-relative path is returned so
   `load()` raises a `FileNotFoundError` pointing at the canonical
   location (informative error message, not an opaque `False`).

## Files changed

| File | Change |
|---|---|
| `molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py` | Replaced CWD-relative defaults with `Path(__file__)`-resolved absolute paths; added `_resolve_default_paths()` helper; `__init__` resolves at construction.  No checkpoint moved, no callers broken. |
| `molmetal/tests/test_pocket_macro_inference_path.py` | NEW.  6 unit tests (4+ required). |

## Tests added (6, all passing)

```
tests/test_pocket_macro_inference_path.py::test_default_paths_are_module_relative_absolute           PASSED
tests/test_pocket_macro_inference_path.py::test_cwd_relative_fallback_when_module_path_missing       PASSED
tests/test_pocket_macro_inference_path.py::test_module_relative_wins_when_both_exist                  PASSED
tests/test_pocket_macro_inference_path.py::test_resolve_returns_module_relative_path_when_missing    PASSED
tests/test_pocket_macro_inference_path.py::test_real_module_relative_default_exists                   PASSED
tests/test_pocket_macro_inference_path.py::test_subprocess_construction_succeeds_from_each_cwd       PASSED

============================== 6 passed in 0.14s ===============================
```

Coverage:

1. **`test_default_paths_are_module_relative_absolute`** — both
   `DEFAULT_CHECKPOINT_PATH` and `DEFAULT_METADATA_PATH` are absolute
   paths (not the CWD-relative literal) and resolve to the canonical
   `molmetal/models/` files.
2. **`test_cwd_relative_fallback_when_module_path_missing`** — when
   the module-relative path is absent (e.g. a slim install), a
   CWD-relative `models/pocket_macro_skeleton_v2.pt` is found
   correctly.  Uses the ``test_inject=`` parameters of
   `_resolve_default_paths` so it doesn't mutate the real
   filesystem.
3. **`test_module_relative_wins_when_both_exist`** — when both
   candidates exist, the module-relative path ALWAYS wins (so CWD
   never affects the canonical lookup).
4. **`test_resolve_returns_module_relative_path_when_missing`** —
   when neither candidate exists, the resolver still returns the
   module-relative path so downstream `FileNotFoundError` points at
   the canonical location.
5. **`test_real_module_relative_default_exists`** — sanity check:
   the canonical file ships with the repo and the default resolves
   to it.
6. **`test_subprocess_construction_succeeds_from_each_cwd`** —
   spawns a fresh Python process from each of `REPO_ROOT`,
   `molmetal/`, `molmetal/tests/`, and `/tmp` and asserts the
   resolver returns the canonical checkpoint.  This is the
   integration-level guard that catches the BUG-2 regression in
   the wild.

The test file uses **AST extraction** (not `import`) to load the
path-resolution surface, so it runs even on hosts where the torch
shared library is broken (e.g. this CI host's Python 3.14 + /opt/libtorch
ABI mismatch).  The original `test_deflex_wireup_phase2_pocket_macro.py`
tests still skip on this host for the same torch reason — they
require a working torch install and will pass on any host with
`pip install torch` working.

## Honest framing

* **The fix is structural, not measured.**  Tests pass on the
  resolver code; the end-to-end "model loads from /tmp CWD" was
  verified at the resolver level (test 6) but a true end-to-end
  check requires a working torch + numpy + rdkit environment
  (blocked on this host by the torch/Python 3.14 ABI mismatch
  unrelated to this fix).
* **Constraint respected: checkpoint not moved.** The
  `molmetal/models/pocket_macro_skeleton_v2.pt` file is untouched;
  the fix only changed how we locate it.
* **Additive: backwards-compatible.** The CWD-relative
  `models/pocket_macro_skeleton_v2.pt` fallback is preserved, so
  any older test or script that relied on it still works.
* **Path arithmetic was double-checked at runtime.**  An
  intermediate draft used `_MODULE_DIR.parent.parent.parent` (3
  levels up) which would have resolved to the repo root, not
  `molmetal/`.  The final code uses 2 `.parent` operations and the
  subprocess test (test 6) confirms the resolved path is
  `<REPO_ROOT>/molmetal/models/pocket_macro_skeleton_v2.pt`.
* **Unblocks callers that change CWD.**  Any script that previously
  invoked `PocketMacroInference()` from a non-project-root directory
  (e.g. pytest workers in `molmetal/tests/`, shell scripts in
  `/tmp/`, CI runners with absolute-path invocation) now
  transparently finds the canonical checkpoint.