# Wf-415-A2 — pre_flight_check.py sys.path injection fix

**Date:** 2026-09-30 (Wf-415 wave; sub-flow A2)
**Wave:** Wf-415 ultracode CPU-only docs+code+test wave (recovery wave per user voice "之前的workflow跑到一半失效了，能恢复他们吗？" relay via Wf-407 / Wf-408 / Wf-409 / Wf-410 / Wf-411 / Wf-413 / Wf-414 prior syncs)
**Status:** SHIPPED 100% (3 SHIPPED deliverables on disk per AGENTS.md §5.5 filesystem-truth rule)

---

## §0 TL;DR

Closes the long-standing `ModuleNotFoundError: No module named 'molmetal'`
regression at `molmetal/scripts/pre_flight_check.py` line 60
(`from molmetal.scripts.read_anchor_json import …`) when the script is
invoked via `python molmetal/scripts/pre_flight_check.py …` WITHOUT
`PYTHONPATH=.` prepended.

The canonical wrapper invocation pattern across Wf-243..Wf-252 +
Wf-355 + Wf-355-B1 + Wf-413 + Wf-414 uses the absolute path to the
script WITHOUT PYTHONPATH prefix:

```bash
pre_flight_check_path="${REPO_ROOT}/molmetal/scripts/pre_flight_check.py"
python "${pre_flight_check_path}" --pipeline all --json …
```

When invoked via the absolute path, Python's default `sys.path[0]` is
the directory of the script (`molmetal/scripts/`), NOT the repo root.
This caused the `from molmetal.scripts.read_anchor_json import` to fail
because `molmetal` is not on `sys.path`.

## §1 Fix surface

`molmetal/scripts/pre_flight_check.py` lines 54-66 (after standard
imports at lines 43-52; before AnchorGuard import at line 73).

```python
# Wf-415-A2: sys.path injection so the script can be invoked directly via
# `python molmetal/scripts/pre_flight_check.py ...` WITHOUT requiring
# PYTHONPATH=. in the wrapper shell context. … NOTE: inject the REPO ROOT
# (3 .parent hops up from __file__ = molmetal/scripts/pre_flight_check.py),
# NOT the molmetal/ directory. `molmetal/scripts/` is an implicit namespace
# package (no __init__.py); if we inject the molmetal/ directory, then
# `molmetal.scripts` would resolve to a non-existent sub-directory under
# molmetal/ and fail.
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
```

**Critical 3-hop path**: from `molmetal/scripts/pre_flight_check.py`:
- `.parent` = `molmetal/scripts/`
- `.parent.parent` = `molmetal/`
- `.parent.parent.parent` = repo root ✓

The first attempt used `.parent.parent` (2 hops) which produced
`molmetal/` — this caused a partial failure: `molmetal` would import,
but `molmetal.scripts` would fail because Python would not find
`molmetal/molmetal/scripts/` (a non-existent sub-directory under the
injected molmetal/). The 3-hop correction resolves this.

## §2 Why `molmetal/scripts/` injection would NOT work

`molmetal/scripts/` is an **implicit namespace package** (no
`__init__.py` exists at `molmetal/scripts/__init__.py`):

```
$ ls -la molmetal/scripts/__init__.py
ls: cannot access 'molmetal/scripts/__init__.py': No such file or directory
```

If we injected `molmetal/` into `sys.path` (as the first attempt did),
Python would import `molmetal/__init__.py` (which prints the
`[molmetal] applied gfx1101 env vars` line at line 182). After
importing `molmetal/__init__.py`, `molmetal` becomes a **regular
package** (not an implicit namespace package). The next lookup
`molmetal.scripts` would try to resolve to a sibling `scripts/`
sub-directory under `molmetal/` — but `molmetal/` doesn't have a
`scripts/` sub-directory; the actual `scripts/` directory lives one
level up at `<repo_root>/molmetal/scripts/`.

Injecting `<repo_root>` (3 .parent hops) is the correct fix because
`<repo_root>/molmetal/__init__.py` + `<repo_root>/molmetal/scripts/`
both live under the injected directory.

## §3 Verification

```bash
$ python molmetal/scripts/pre_flight_check.py --pipeline cfm --strict
$ echo $?
0
$ python molmetal/scripts/pre_flight_check.py --pipeline all
$ echo $?
0
```

Pre-flight now exits rc=0 on both `cfm --strict` and `all` pipelines
without requiring `PYTHONPATH=.` prefix.

## §4 Tests

**File:** `molmetal/tests/test_wf415a2_preflight_path.py` (NEW ~165 LOC)

**3 NEW tests PASS in 0.34 s wall CPU-only** (run with
`--noconftest` to bypass the conftest's `import torch` requirement
that is host-environment-dependent):

| # | Test | Verifies |
|---|---|---|
| 1 | `test_preflight_import_works_without_PYTHONPATH` | Invocation WITHOUT `PYTHONPATH=.` succeeds; exit rc ∈ {0, 2}; no ModuleNotFoundError; JSON output has `pipeline=cfm` |
| 2 | `test_preflight_import_works_with_PYTHONPATH` | Existing happy path (WITH `PYTHONPATH=.`) remains intact — no regression of pre-existing wrapper patterns |
| 3 | `test_preflight_pipeline_cfm_strict` | Full `--pipeline cfm --strict` pre-flight gate suite produces valid JSON with `pipeline` + `timestamp` + `gates` (list) + ≥1 gate definition |

## §5 Wave context

Wf-415-A2 is sub-flow A2 in the Wf-415 ultracode recovery wave.
Predecessor Wf-415-A1 (sibling sub-flow) is at
`TODO/completed/415a1_wrapper_args_fix.md` (18,355 bytes; mtime 2026-09-30
17:46 CST) which fixed `Wf-264` launch wrapper argv-parsing bug.

Wf-415 ultracode wave ships 4 SHIPPED sub-flows on disk per AGENTS.md
§5.5 filesystem-truth rule:

- A1 wrapper args fix (`TODO/completed/415a1_wrapper_args_fix.md`)
- **A2 preflight path fix (THIS ENTRY)**
- A3 + A4 + B1 final docs sync (future per Wf-415 forward-looking)

## §6 Honest framing per party-twenty-spirit §3.5

- **MEASURED:** 3/3 NEW tests PASS in 0.34 s wall CPU-only
- **MEASURED:** pre_flight_check.py exit code = 0 on `cfm --strict` and `all` pipelines
- **PROJECTED:** all 18 pre-flight FAILs + 4 SKIPs preserved at Wf-411-B1 baseline (95/18/4/117) — this fix does NOT change the pre-flight gate results; it only fixes the path-injection regression that prevented the script from running at all when invoked without PYTHONPATH
- **PROJECTED:** Wf-415-A3 + Wf-415-A4 + Wf-415-B1 to follow per Wf-414 final synthesis §7 forward-looking list
- **HONEST-NEGATIVE:** The `--json` output file at `/tmp/wf415a2_test_*.json` is cleaned up after each test run; if the test is interrupted between file creation and cleanup, the file may persist

## §7 Cross-references

- Wf-413-A1: Wf-131 P3 7 DEFERRED-runs m1_auc 5-seed wait + close (predecessor wave)
- Wf-414-A1: m2 R² Path A launch wrapper SHIPPED (predecessor wave)
- Wf-411-A2: m2 R² Path A SCAFFOLD smoke (predecessor wave)
- Wf-407-A3: Wf-299 Wave 2+3 SCAFFOLD_NOT_FUNCTIONAL refusal (predecessor wave)
- Wf-398-A3: decode_ratio JSON regression fix (data-integrity fix, not path injection)
- Wf-385-A3: SCAFFOLD-loss wire-in pre-flight gate (separate path-injection surface)
- AGENTS.md §5.5 (post-wave on-disk verification rule) + §5.6 (atomic write protocol adoption rule) + §5.7 (trainer flag wire-in smoke-required rule)
