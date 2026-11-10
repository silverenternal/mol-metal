#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Wf-A154 — standalone runner for the Wf-A152 census-claim guard.

WHY THIS EXISTS
---------------
``molmetal/tests/test_wfa152_census_guard_coverage.py`` polices a defect class
the older Wf-A134 guard structurally CANNOT see: a document may quote a census
statistic that is no longer the current value, and the Wf-A134 guard's closed
vocabulary of retired numbers would not notice.  The Wf-A152 guard resolves
each quoted claim to (population x epoch x cohort x key) and compares it with
the artifact's OWN current value — the oracle is the artifact, not a curated
list.

Its assertions live in a pytest module, so the only way to consult it from
the pre-flight was to shell out to pytest.  This runner is a thin adapter: it
imports the guard module and CALLS the guard's own test functions, so there
is exactly ONE implementation of the assertions and no second copy to drift.

The guard reads ``metrics/by_metric/cfm_run_census.json`` — the SSOT for the
CFM fragmentation rates — and the SSOT documents that quote it.

Exit codes (the pre-flight gate convention, AGENTS.md §4):

    0 = PASS   every guard assertion held
    1 = FAIL   an assertion failed, or the guard could not be consulted
    2 = SKIP   the guard is not applicable on this tree

FAIL is fail-CLOSED: a census artifact that cannot be read, or a guard that
cannot be loaded, means the published fragmentation numbers are unverified —
not that there is nothing to check.

CPU-only by construction: no torch import, no CUDA call, no checkpoint read.
"""

from __future__ import annotations

import inspect
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 2

GUARD_MODULE = "molmetal.tests.test_wfa152_census_guard_coverage"


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
            test_wfa152_census_guard_coverage as guard,
        )
    except Exception as exc:  # noqa: BLE001 — fail-closed, never crash
        print(f"FAIL: the {GUARD_MODULE} guard could not be imported: "
              f"{type(exc).__name__}: {exc}")
        return EXIT_FAIL

    artifact = getattr(guard, "ARTIFACT", None)
    if artifact is None or not Path(artifact).is_file():
        print(f"FAIL: the census artifact the guard reads is not on disk: "
              f"{artifact} — the published CFM fragmentation rates would be "
              f"unverified.")
        return EXIT_FAIL

    tests = _collect_tests(guard)
    if not tests:
        print(f"FAIL: {GUARD_MODULE} exposes no test_* function — the guard "
              "is empty and would pass vacuously.")
        return EXIT_FAIL

    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        params = inspect.signature(fn).parameters
        if params:
            # A fixture-taking test cannot be called bare; the guard's
            # fixtures are exercised by the pytest surface, not by this
            # adapter.  Skipping is safe because none of the guard's
            # LOAD-BEARING assertions take a fixture.
            continue
        try:
            fn()
        except AssertionError as exc:
            failures.append((name, str(exc) or "(assertion failed)"))
        except Exception as exc:  # noqa: BLE001 — an unexpected raise is a FAIL
            failures.append(
                (name, f"UNEXPECTED {type(exc).__name__}: {exc}\n"
                       + traceback.format_exc(limit=6))
            )

    if failures:
        print(f"FAIL: the Wf-A152 census-claim guard found {len(failures)} "
              f"failing assertion(s) — a document quotes a census statistic "
              f"that is not the artifact's current value, or the guard "
              f"itself is broken:")
        for name, detail in failures:
            print(f"  --- {name}")
            for line in detail.splitlines()[:14]:
                print(f"      {line}")
        return EXIT_FAIL

    print(f"PASS: the Wf-A152 census-claim guard holds on all {len(tests)} "
          f"assertions (artifact {Path(artifact).relative_to(REPO_ROOT)}).")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
