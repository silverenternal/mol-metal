#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Wf-A154 — standalone runner for the Wf-A131 repo-path-reference guard.

WHY THIS EXISTS
---------------
``molmetal/tests/test_wfa131_repo_path_reference_guard.py`` polices a real
defect class: the SSOT documents cite repo-relative paths that do not resolve
to anything on disk (AGENTS.md §5.5 filesystem-truth).  Its assertions live in
a pytest module, so the only way to consult it from the pre-flight was to
shell out to pytest — which drags in the whole conftest (and its CPU-lock
machinery) for what is a pure text scan.

This runner is a thin adapter: it imports the guard module and CALLS the
guard's own test functions, so there is exactly ONE implementation of the
assertions (the guard) and no second copy to drift out of sync.  When the
guard's assertions change, this runner inherits them for free.

Exit codes (the pre-flight gate convention, AGENTS.md §4):

    0 = PASS   every guard assertion held
    1 = FAIL   an assertion failed, or the guard could not be consulted
    2 = SKIP   the guard is not applicable on this tree

FAIL is fail-CLOSED on an import error: a guard that cannot be loaded is a
guard that is silently not protecting anything, which is the same failure the
guard exists to catch.

CPU-only by construction: no torch import, no CUDA call, no checkpoint read.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 2

GUARD_MODULE = "molmetal.tests.test_wfa131_repo_path_reference_guard"


def _collect_tests(module) -> list[tuple[str, object]]:
    """Every module-level ``test_*`` callable, in a stable order."""
    out = []
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        fn = getattr(module, name)
        if not callable(fn):
            continue
        out.append((name, fn))
    return out


def main() -> int:
    try:
        from molmetal.tests import (  # noqa: PLC0415 — deliberate lazy import
            test_wfa131_repo_path_reference_guard as guard,
        )
    except Exception as exc:  # noqa: BLE001 — fail-closed, never crash
        print(f"FAIL: the {GUARD_MODULE} guard could not be imported: "
              f"{type(exc).__name__}: {exc}")
        return EXIT_FAIL

    tests = _collect_tests(guard)
    if not tests:
        print(f"FAIL: {GUARD_MODULE} exposes no test_* function — the guard "
              "is empty and would pass vacuously.")
        return EXIT_FAIL

    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        params = inspect.signature(fn).parameters
        try:
            if "tmp_path" in params:
                with tempfile.TemporaryDirectory(prefix="wfa131_guard_") as td:
                    fn(Path(td))
            else:
                fn()
        except AssertionError as exc:
            failures.append((name, str(exc) or "(assertion failed)"))
        except Exception as exc:  # noqa: BLE001 — an unexpected raise is a FAIL
            failures.append(
                (name, f"UNEXPECTED {type(exc).__name__}: {exc}\n"
                       + traceback.format_exc(limit=6))
            )

    if failures:
        print(f"FAIL: the Wf-A131 repo-path-reference guard found "
              f"{len(failures)} failing assertion(s) out of {len(tests)} "
              f"(a repo-relative path cited by an SSOT document does not "
              f"resolve, or the audited baseline has drifted):")
        for name, detail in failures:
            print(f"  --- {name}")
            for line in detail.splitlines()[:14]:
                print(f"      {line}")
        return EXIT_FAIL

    print(f"PASS: the Wf-A131 repo-path-reference guard holds on all "
          f"{len(tests)} assertions (baseline "
          f"{getattr(guard, 'BASELINE_SIZE', '?')} adjudicated paths, "
          f"pattern baseline {getattr(guard, 'PATTERN_BASELINE_SIZE', '?')}).")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
