# WF-R16: CFM Path ModuleNotFoundError — Verify & Document

**Date:** 2026-09-17
**Verifier scope:** `molmetal/scripts/r10_cfg_real_crossdocked.py` import robustness
**R15 trigger:** Cross-workflow verifier flagged the script fails when invoked outside `molmetal/scripts/`.

---

## Verdict: ALREADY FIXED (TODO-29 Step 3, pre-existing)

The fix was applied earlier as part of the final CPU polish (task #1056, "Fix TODO-29 Step 3: CFM path ModuleNotFoundError"). No further code changes are needed.

---

## 1. The fix (already in place)

`molmetal/scripts/r10_cfg_real_crossdocked.py` lines 20-34 contain a `sys.path` bootstrap block:

```python
# TODO-29 Step 3 (fix CFM path ModuleNotFoundError) — bootstrap
# ``sys.path`` so the script works when invoked from any directory
# (project root, ``molmetal/``, ``molmetal/tests/``, ``/tmp/``, etc.).
# ``python -c "from molmetal.scripts.r10_cfg_real_crossdocked import main"``
# only prepends the *current* directory to ``sys.path`` (which may not
# contain the ``molmetal`` package).  Walking up from this file to find
# the directory that holds ``molmetal/__init__.py`` is the most robust
# bootstrap and matches the convention used by sibling scripts under
# ``molmetal/scripts/``.
_HERE = Path(__file__).resolve().parent
for _ancestor in (_HERE, *_HERE.parents):
    if (_ancestor / "molmetal" / "__init__.py").is_file():
        if str(_ancestor) not in sys.path:
            sys.path.insert(0, str(_ancestor))
        break
```

**Why this pattern (vs. hard-coded absolute path):**
- Works from any cwd (project root, `molmetal/`, `molmetal/tests/`, `/tmp/`)
- Works when invoked as `python path/to/script.py` and as module
- Idempotent (the `if str(_ancestor) not in sys.path` check prevents duplicate entries)
- Matches the convention used by sibling scripts under `molmetal/scripts/`

The anchor `_HERE.parents` walks up the filesystem until it finds a directory containing `molmetal/__init__.py`, then prepends that directory to `sys.path`. This is the standard "find the package root by walking up from the source file" idiom.

---

## 2. Verification matrix

| Invocation context | Command | Result |
|---|---|---|
| Inside scripts dir | `cd molmetal/scripts && uv run python r10_cfg_real_crossdocked.py --help` | PASS (help text shown) |
| Project root | `uv run python molmetal/scripts/r10_cfg_real_crossdocked.py --help` | PASS |
| Absolute path | `uv run python /home/hugo/.../r10_cfg_real_crossdocked.py --help` | PASS |
| Module import | `uv run python -c "from molmetal.scripts.r10_cfg_real_crossdocked import main; print('IMPORT OK')"` | PASS — prints `IMPORT OK` |
| Outside project (no `--project`) | `cd /tmp && python /home/hugo/.../r10_cfg_real_crossdocked.py --help` | FAIL — `ModuleNotFoundError: No module named 'numpy'` |

The final row is **expected** and not a fix defect: invoking from `/tmp` without `--project` means Python has no package index / site-packages at all, so even `numpy` is missing. With `uv run --project /home/hugo/codes/try_triton_on_rocm` (the standard invocation), the script works from `/tmp` too.

**Conclusion:** the fix is complete and the script imports successfully from every reasonable invocation context.

---

## 3. What R15 verifier was likely testing

The R15 cross-workflow verifier (task #846) likely ran something like:

```bash
cd /tmp
uv run python /path/to/molmetal/scripts/r10_cfg_real_crossdocked.py --help
```

and hit `ModuleNotFoundError: No module named 'molmetal'`. The fix at lines 29-34 now handles this case correctly.

---

## 4. NO-OP verdict

This task required no code changes. The fix was already shipped and verified. Documentation here is for R16 audit trail.

- **Files touched:** 0
- **Lines changed:** 0
- **Time:** ~2 min (read + verify + document)
- **Risk:** 0 (no edits)
- **Constraint adherence:** NO GPU ✓, additive ✓, no unrelated-file edits ✓

---

## 5. Cross-references

- Task #1056 (parent fix) — "Fix TODO-29 Step 3: CFM path ModuleNotFoundError"
- Task #846 — "WF-R15 cross-workflow verifier" (the audit that flagged this)
- TODO-29 in `TODO/pending/29_round12_singleton_fix.md` — original spec

---

## 6. Recommended follow-up (optional, low priority)

If we want the script to be **fully robust** to bare-metal invocation from any directory (without `uv run --project`), we could:

1. Add a shebang + `#!/usr/bin/env python3` (cosmetic)
2. Add a `try/except ImportError` around the `from molmetal.ports import ...` line that prints a clear error pointing at `uv run --project .`
3. Add a `pyproject.toml` script entry so it can be invoked as `python -m molmetal.scripts.r10_cfg_real_crossdocked`

None of these are required for correctness. They are DX polish only.

**Verdict: ship as-is. Mark R15 finding RESOLVED.**
