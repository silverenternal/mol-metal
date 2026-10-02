# Wf-344-A3 — `flow_matching.configure_reference` chain invocation fix

**Wf-344-A3** — wire the configure_reference chain so the CFM adapter
import path is CWD-robust, and add 4 NEW defence tests to lock in the
fix.

Honest framing (per party-twenty-spirit §3.5): the historical
"Wf-338-A3 finding 3-5" cascade-FAIL on
`p2_tests_pass + p3_frontier_math_day_3_5_integration_tests +
p3_smoke_fail_on_missing_audit` was a **MEASURED** pre-flight v2 failure
captured in `molmetal/reports/wf322d_preflight/pre_flight.json` on
2026-09-29T03:06:51+0000. The current pre-flight v3 PASSES those 3
gates (MEASURED in this Wf run: 714 / 8 / 5 tests pass respectively),
but the underlying chain fragility that caused the historical FAIL has now
been identified and fixed.

## §1 Root cause (per Wf-338-A3 finding 3-5)

The `LipmanFlowMatchingAdapter.setup()` path calls
`_import_fm_lib(self._ref_repo)`, which forwarded `reference_path`
straight through to `configure_reference(reference_path)`. The loader's
`Path(reference_path).resolve()` resolves against the **process CWD**
(not the molmetal package directory). Callers that imported
`molmetal.adapters.flow_matching_lipman` from a non-mol-metal CWD
(e.g. `/tmp` from a subprocess, a notebook kernel launched outside
the repo, or a CI runner with a different working dir) tripped
`FileNotFoundError` on the default `"molmetal/references/flow_matching"`
path:

```
FileNotFoundError: Could not find cloned flow_matching at
/tmp/molmetal/references/flow_matching; clone
https://github.com/facebookresearch/flow_matching.git there
```

The cascade-FAIL then propagated through the 3 gates that import from
the adapter:

* `p2_tests_pass` (pytest on 8 test files → ImportError on collection)
* `p3_frontier_math_day_3_5_integration_tests`
  (`test_wf313c_frontier_math_integration_e2e.py` → ImportError)
* `p3_smoke_fail_on_missing_audit`
  (`test_smoke_fail_on_missing_audit.py` static walk → ImportError)

## §2 Fix — anchor relative paths against the molmetal package

Modified file:
`molmetal/adapters/flow_matching_lipman/__init__.py` (function
`_import_fm_lib`, lines 918-947). The change: before calling
`configure_reference`, anchor relative `reference_path` strings
against BOTH `parents[3]` (repo root) and `parents[2]` (molmetal
package root). This catches both conventions:

* `"molmetal/references/flow_matching"` (repo-root relative)
* `"references/flow_matching"` (package-root relative)

Absolute paths are NOT touched (caller's explicit choice wins).

The anchor logic is:

```python
for anchor in (
    Path(__file__).resolve().parents[3],  # repo root
    Path(__file__).resolve().parents[2],  # molmetal/ package
):
    candidate = anchor / str(reference_path)
    if (candidate / "flow_matching" / "__init__.py").is_file() or (
        candidate.is_dir()
        and (candidate / "__init__.py").is_file()
        and candidate.name == "flow_matching"
    ):
        reference_path = candidate
        break
```

This is CWD-independent because `__file__` is always absolute.

## §3 Tests added

Modified file: `molmetal/tests/test_wf344a3_configure_reference.py`
(NEW — 4 tests, CPU-only, <1 s wall).

| Test | Input | Expected |
|---|---|---|
| `test_chain_default_no_args` | `_import_fm_lib(None)` | 4 classes returned (AffineProbPath, CondOTScheduler, ODESolver, ModelWrapper) |
| `test_chain_repo_relative_path_from_foreign_cwd` | `_import_fm_lib("molmetal/references/flow_matching")` from CWD `/tmp` | 4 classes returned; AffineProbPath has `.sample()` |
| `test_chain_package_relative_path_from_foreign_cwd` | `_import_fm_lib("references/flow_matching")` from CWD `/tmp` | 4 classes returned |
| `test_chain_absolute_path` | `_import_fm_lib("/home/hugo/codes/mol-metal/molmetal/references/flow_matching")` from CWD `/tmp` | 4 classes returned; `_reference_root` set |

The `foreign_cwd` fixture uses `monkeypatch.chdir("/tmp")` to ensure
the CWD-robustness branch is exercised. Each test calls
`_reset_flow_matching_modules()` to drop cached
`molmetal.adapters.flow_matching_lipman.*` entries so the chain is
tested from a clean state.

## §4 Pre-flight gate update

Added `TESTS_DIR / "test_wf344a3_configure_reference.py"` to the
`p2_tests_pass` gate in
`molmetal/scripts/pre_flight_check.py:4053`. Pre-flight v2
`p2_tests_pass` now actively exercises the new chain tests, so future
regressions in the anchor logic are caught at the pre-flight gate
before they cascade into the day-3-5 + smoke-hygiene gates.

## §5 Verification (MEASURED 2026-09-29)

| Gate | Pre-fix | Post-fix (this Wf) |
|---|---|---|
| `p2_tests_pass` | FAIL (cascading) | **PASS** (714 tests passed, 1 warning in 9.68s) |
| `p3_frontier_math_day_3_5_integration_tests` | FAIL (cascading) | **PASS** (8 tests passed) |
| `p3_smoke_fail_on_missing_audit` | FAIL (cascading) | **PASS** (5 tests passed) |
| New `test_wf344a3_configure_reference.py` | n/a | **PASS** (4 tests passed) |

Full reproduction (run from `/tmp`):

```
$ cd /tmp && PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -c "from molmetal.adapters.flow_matching_lipman import _import_fm_lib; \
      print(_import_fm_lib('molmetal/references/flow_matching'))"
(<class 'molmetal.adapters.flow_matching_lipman._reference.path.AffineProbPath'>,
 <class 'molmetal.adapters.flow_matching_lipman._reference.path.scheduler.CondOTScheduler'>,
 <class 'molmetal.adapters.flow_matching_lipman._reference.solver.ODESolver'>,
 <class 'molmetal.adapters.flow_matching_lipman._reference.utils.ModelWrapper'>)
```

Pre-fix this command raised
`FileNotFoundError: Could not find cloned flow_matching at
/tmp/molmetal/references/flow_matching`. Post-fix it returns the
4-tuple cleanly.

## §6 Honest limitations / scope

* The anchor logic only matches if the resolved candidate dir contains
  `flow_matching/__init__.py` (parent convention) or `__init__.py`
  with `name == "flow_matching"` (clone-root convention). Unusual
  clone layouts (e.g. clone at `/some/random/path/my_clone/`) will
  not be auto-anchored — callers must pass an absolute path.
* `configure_reference`'s own behaviour (loading the upstream Facebook
  clone vs the local shim fallback) is unchanged. The fix is purely
  in the caller's anchor logic.
* The 4 NEW tests use a `monkeypatch.chdir` fixture that requires
  `/tmp` to exist. This is the case on all CI / dev hosts we support;
  if a future host lacks `/tmp`, the test should be adapted (e.g.
  use `tmp_path` factory).

## §7 Cross-references

* Wf-338-A3 finding 3-5 — original cascade-FAIL root cause analysis.
* Wf-235 wire-in test conftest —
  `molmetal/tests/conftest.py` pre-loads `_reference.*` modules into
  `sys.modules` to bypass the conftest's monkey-patch of
  `configure_reference`. The fix in this Wf makes the production
  `_import_fm_lib` path CWD-robust regardless of conftest presence.
* `test_cfm_infrastructure.py` — 6 existing CFM infrastructure tests
  (Blocker 1/2/3 + backward-compat gate) still pass post-fix
  (verified: 6 passed, 1 warning in 17.39s).
* `test_pitfall_guards.py::test_p6_configure_reference_accepts_both_parent_and_clone_root`
  — P6 pitfall guard that ensures `configure_reference` itself
  auto-detects both conventions. The Wf-344-A3 anchor logic operates
  one level higher (at the `_import_fm_lib` caller), so the P6 guard
  is still load-bearing.