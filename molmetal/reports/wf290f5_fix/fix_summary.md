# Wf-290F5 — Fix test_reproduce_R17

## error_before

```
$ python -m pytest molmetal/tests/test_reproduce_R17.py --co -q
==================================== ERRORS ====================================
____________ ERROR collecting molmetal/tests/test_reproduce_R17.py _____________
'reproduce_R17' not found in `markers` configuration option
=============================== short test summary info ============================
ERROR molmetal/tests/test_reproduce_R17.py - Failed: 'reproduce_R17' not found
!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!
no tests collected, 1 error in 0.29s
```

Root cause: `pyproject.toml` has `addopts = "--strict-markers"` and a hand-maintained
`[tool.pytest.ini_options].markers` list.  The list did not include `reproduce_R17`,
even though `molmetal/tests/test_reproduce_R17.py` decorates
`test_script_runs_and_emits_all_done` with `@pytest.mark.reproduce_R17` to make it
opt-out-able via `pytest -m "not reproduce_R17"`.  Strict mode rejected the
unknown marker during collection.

## fix_applied

Added the `reproduce_R17` marker to the marker registry in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "molmetal"]
norecursedirs = ["references", ".venv", ".git", "__pycache__"]
addopts = "--strict-markers"
markers = [
    "slow: marks tests as slow (deselect with '-m not slow')",
    "integration: marks tests requiring optional external services or data",
    "gpu: marks tests requiring a GPU (QuickVina-GPU, CFM retrain, etc.)",
    "gpu_blocked: marks tests whose real path is BLOCKED by the GPU outage (RX 7800 XT gfx1101 firmware-level SMU hang per wf_gpu_diag/diagnosis.md)",
    "reproduce_R17: marks tests that drive the wf_reproduce_R17.sh end-to-end driver (set MOLMETAL_RUN_REPRODUCE=1 to actually invoke; otherwise the marker exists so `-m 'not reproduce_R17'` opt-out works)",
]
```

Why registration (not commenting out the marker):

1. The marker has user-facing value — it powers the `-m "not reproduce_R17"`
   opt-out documented in the test module's docstring.  Removing it would silently
   break the inner-loop fast-CI lane.
2. `test_script_runs_and_emits_all_done` is gated by two `@pytest.mark.skipif`
   decorators (needs `uv` on PATH + `MOLMETAL_RUN_REPRODUCE=1`), so registering
   the marker does not slow anyone down by default.
3. Single-line, `pyproject.toml`-only change matches the repo's existing
   convention for marker registration (slow / integration / gpu / gpu_blocked).

## tests_after

Collection succeeds and the four lightweight static tests pass; the heavy
subprocess test is skipped (by design — it requires `MOLMETAL_RUN_REPRODUCE=1`):

```
$ python -m pytest molmetal/tests/test_reproduce_R17.py -v
collected 5 items

molmetal/tests/test_reproduce_R17.py::test_script_exists PASSED          [ 20%]
molmetal/tests/test_reproduce_R17.py::test_script_is_executable PASSED   [ 40%]
molmetal/tests/test_reproduce_R17.py::test_script_has_all_done_marker PASSED [ 60%]
molmetal/tests/test_reproduce_R17.py::test_script_has_eight_sections PASSED [ 80%]
molmetal/tests/test_reproduce_R17.py::test_script_runs_and_emits_all_done SKIPPED [100%]

=================== 4 passed, 1 skipped, 1 warning in 0.54s ==================
```

Opt-out lane also works as the module docstring promises:

```
$ python -m pytest molmetal/tests/test_reproduce_R17.py -m 'not reproduce_R17' -v
collected 5 items / 1 deselected / 4 selected

================== 4 passed, 1 deselected, 1 warning in 0.69s ==================
```

## report_path

`/home/hugo/codes/mol-metal/molmetal/reports/wf290f5_fix/fix_summary.md`

## files_touched

- `/home/hugo/codes/mol-metal/pyproject.toml` — appended `reproduce_R17` to
  `[tool.pytest.ini_options].markers`.