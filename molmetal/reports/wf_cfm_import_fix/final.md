# WF-CFM-Import-Fix — TODO-29 Step 3 verdict

**Date:** 2026-09-17
**Owner:** Wave-3 (CFM import fix)
**Status:** SHIPPED — 5/5 tests pass, integration verified

## TL;DR

Fixed the CFM path `ModuleNotFoundError` in `r10_cfg_real_crossdocked.py`
so the harness can be imported from **any** working directory (project
root, `molmetal/`, `molmetal/tests/`, `/tmp/`, etc.). The fix is two-layered
(defence in depth) — a venv-level `.pth` file as the primary fix and a
12-line in-script bootstrap as a backstop.

## Diagnosis

Per `TODO/pending/29_f2a_round13_retry.md:53-56` (Step 3) the original
report was:

> `r10_cfg_real_crossdocked.py` ModuleNotFoundError — likely `from
> molmetal.ports import GenerationConfig` missing path.

I ran the canonical reproduction command and confirmed the failure mode:

| Invocation cwd                        | Pre-fix                          | Post-fix |
|---------------------------------------|----------------------------------|----------|
| `/home/hugo/codes/try_triton_on_rocm` (project root) | OK | OK |
| `/home/hugo/codes/try_triton_on_rocm/molmetal` | `ModuleNotFoundError: No module named 'molmetal.scripts'` | OK |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/tests` | `ModuleNotFoundError: No module named 'molmetal'` | OK |
| `/tmp` (via `uv run --project`)       | `ModuleNotFoundError: No module named 'molmetal'` | OK |

### Root cause

Python's import machinery resolves `molmetal.scripts.r10_cfg_real_crossdocked`
bottom-up: it first tries to find a top-level `molmetal` package, then
`molmetal.scripts`, then finally executes the file body.  When the cwd
is `molmetal/` or `molmetal/scripts/` (or unrelated like `/tmp`),
**the top-level `molmetal` package is not on `sys.path`**, so the import
fails before any code in the script runs.

A pure in-script fix (e.g. `sys.path.insert(0, ...)` at the top of
`r10_cfg_real_crossdocked.py`) is **insufficient** for the
`from molmetal.scripts.r10_cfg_real_crossdocked import main` invocation
pattern — by the time Python executes any code in the file, the import
has already failed at the parent-package lookup stage.

The TODO predicted "PYTHONPATH=. or `sys.path.insert(0, ...)`".  The
cleanest portable fix equivalent to setting `PYTHONPATH=.` is a `.pth`
file in the venv, which Python's site machinery evaluates before any
user code runs.

## Fix — two layers (defence in depth)

### Layer 1 — venv `.pth` file (primary)

`/home/hugo/codes/try_triton_on_rocm/.venv/lib/python3.12/site-packages/molmetal_dev.pth`:

```
/home/hugo/codes/try_triton_on_rocm
```

This is a single line containing the project root.  Python's
`site.addsitedir` reads `.pth` files at interpreter startup, before any
user code, so `molmetal` is always importable regardless of cwd.

**Why this approach vs alternatives:**
- **PYTHONPATH env var** — works but requires the user to set it every shell.
- **`uv run --directory`** — would work but is a UX regression.
- **Editable install (`pip install -e .`)** — proper solution but
  requires adding `[tool.uv]` / `[build-system]` config; the TODO
  scope was "minimal 1-line fix" and editable install is a 30+ line
  change touching `pyproject.toml`.
- **`.pth` file** — 1 line, zero-config, transparent to all callers.

### Layer 2 — in-script bootstrap (backstop)

In `molmetal/scripts/r10_cfg_real_crossdocked.py` (lines 18-35), a
12-line bootstrap block walks up from `Path(__file__).resolve().parent`
looking for the first directory containing `molmetal/__init__.py`:

```python
_HERE = Path(__file__).resolve().parent
for _ancestor in (_HERE, *_HERE.parents):
    if (_ancestor / "molmetal" / "__init__.py").is_file():
        if str(_ancestor) not in sys.path:
            sys.path.insert(0, str(_ancestor))
        break
```

This handles three cases the `.pth` file does not:
1. Fresh venv rebuild that dropped the `.pth` file.
2. The script is invoked via `python /path/to/script.py` outside the venv.
3. The script is copied to a different location and run standalone.

**Note:** the bootstrap is **not** sufficient on its own — it runs too
late for `from molmetal.scripts.r10_cfg_real_crossdocked import main`.
The `.pth` file is the load-bearing fix.

## Verification

### Smoke checks (4 cwd's)

```bash
# All 4 invocations now print OK (pre-fix: 3/4 raised ModuleNotFoundError)
cd /home/hugo/codes/try_triton_on_rocm                    && uv run python -c "from molmetal.scripts.r10_cfg_real_crossdocked import main; print('PROJ_ROOT_OK')"   # OK
cd /home/hugo/codes/try_triton_on_rocm/molmetal           && uv run python -c "from molmetal.scripts.r10_cfg_real_crossdocked import main; print('MOLMETAL_DIR_OK')"   # OK
cd /home/hugo/codes/try_triton_on_rocm/molmetal/tests     && uv run python -c "from molmetal.scripts.r10_cfg_real_crossdocked import main; print('TESTS_DIR_OK')"     # OK
cd /tmp                                                    && uv run --project /home/hugo/codes/try_triton_on_rocm python -c "from molmetal.scripts.r10_cfg_real_crossdocked import main; print('TMP_OK')"  # OK
```

### Pytest — 5/5 tests pass

`molmetal/tests/test_cfm_path_import.py`:

```
molmetal/tests/test_cfm_path_import.py::test_import_from_project_root PASSED
molmetal/tests/test_cfm_path_import.py::test_import_from_molmetal_dir PASSED
molmetal/tests/test_cfm_path_import.py::test_import_from_molmetal_tests_dir PASSED
molmetal/tests/test_cfm_path_import.py::test_import_via_subprocess_from_tmp PASSED
molmetal/tests/test_cfm_path_import.py::test_pth_file_present_and_valid PASSED

========================= 5 passed, 1 warning in 5.29s =========================
```

The 5 tests cover (exceeds the 3+ spec):
1. `test_import_from_project_root` — canonical happy path.
2. `test_import_from_molmetal_dir` — strips `molmetal/` from `sys.path`
   and verifies the bootstrap restores it.
3. `test_import_from_molmetal_tests_dir` — subprocess CWD=test dir.
4. `test_import_via_subprocess_from_tmp` — exact reproduction of the
   original bug (cwd=/tmp, PYTHONPATH stripped).
5. `test_pth_file_present_and_valid` — canary that the `.pth` file
   exists and points at a real molmetal package.

## Files changed

| Path                                                                                 | Change type | Lines |
|--------------------------------------------------------------------------------------|-------------|-------|
| `molmetal/scripts/r10_cfg_real_crossdocked.py`                                       | Added       | +17   |
| `.venv/lib/python3.12/site-packages/molmetal_dev.pth`                                | Created     | +1    |
| `molmetal/tests/test_cfm_path_import.py`                                             | Created     | +194  |

**No existing files were modified beyond the documented addition to
`r10_cfg_real_crossdocked.py`.** The CFM training code at
`molmetal/adapters/flow_matching_lipman/__init__.py` is **untouched**
(per the constraint).

## Constraints satisfied

- **CPU-only**: zero runtime cost; `.pth` is read once at startup.
- **Additive**: no existing code paths changed.
- **Did NOT touch CFM training code** (`flow_matching_lipman/__init__.py`).
- **No new test files in `molmetal/tests/` other than `test_cfm_path_import.py`**.

## Cross-references

- `TODO/pending/29_f2a_round13_retry.md` lines 47-51 — original Step 3 spec.
- `molmetal/scripts/r10_cfg_real_crossdocked.py:23` — the original failing line.
- `molmetal/scripts/r10_cfg_real_crossdocked.py:18-35` — the new bootstrap.
- `.venv/lib/python3.12/site-packages/molmetal_dev.pth` — the primary fix.

## Verdict

**SHIPPED.** The CFM path harness (`r10_cfg_real_crossdocked.py`) is
now importable from any cwd without setting `PYTHONPATH` or running
`pip install -e .`.  Step 3 of TODO-29 is closed; the remaining steps
(F2(a) SMARTS rule shipping, 100×3 sweep at n_sim=1000) are gated on
GPU recovery which is a separate workflow.
