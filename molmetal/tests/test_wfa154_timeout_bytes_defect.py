# SPDX-License-Identifier: Apache-2.0
"""Wf-A154 — the ``TimeoutExpired`` BYTES defect, and its two forked copies.

THE DEFECT
----------
``subprocess.TimeoutExpired.stdout`` / ``.stderr`` are **BYTES** whenever the
child produced output before the timeout — even when the call passed
``text=True``, because CPython raises the exception from inside
``communicate()``, BEFORE the decode step a successful call performs.

A handler written as::

    (exc.stderr or "") + f"\\n[TIMEOUT after {timeout}s]"

therefore raises ``TypeError: can't concat str to bytes`` **from inside the
timeout handler itself**, so the caller sees a type error instead of the
real condition ("the probe timed out").  And because ``exc.stdout or ""``
returns the BYTES unchanged, the poisoned value was then written into the
memoised ``_HELP_PROBE_CACHE`` and re-served to every later gate with the
same key — so one slow probe became N instant failures.

MEASURED 2026-11-01 (Wf-A154): six pre-flight gates reported

    "wf343c1 runner raised: TypeError: can't concat str to bytes"

four of them in 0.00 s (served straight from the poisoned cache) and two in
~31.8 s (the probes that actually hit the 30 s wall budget).

THE FORK
--------
``pre_flight_check._run`` had the fix (2026-10-04, with a comment
explaining the CPython behaviour).  ``wf343c1_pre_retrain_check._run`` — a
forked copy of the same helper — did not.  The rule now has ONE
implementation (``wf343c1_pre_retrain_check._as_text``) and the pre-flight
calls it, so the two copies cannot drift apart a second time.

CPU-only: no torch import, no CUDA call, no checkpoint read.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = REPO / "molmetal" / "scripts"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from molmetal.scripts import wf343c1_pre_retrain_check as wf343c1  # noqa: E402

#: A child that writes to BOTH streams and then outlives the timeout, so the
#: TimeoutExpired carries non-empty (bytes) output slots.
_SLOW_CHILD = (
    "import sys, time; "
    "sys.stdout.write('child-stdout'); sys.stdout.flush(); "
    "sys.stderr.write('child-stderr'); sys.stderr.flush(); "
    "time.sleep(30)"
)


def test_the_timeout_handler_does_not_raise_a_type_error() -> None:
    """The core regression: the timeout path must RETURN, not raise.

    Before the fix this line raised ``TypeError: can't concat str to
    bytes`` — the failure mode hid the failure.
    """
    rc, out, err = wf343c1._run(
        [sys.executable, "-c", _SLOW_CHILD], timeout=1.0
    )
    assert rc == 124, f"expected the timeout return code 124, got {rc}"
    assert isinstance(out, str), f"stdout slot is {type(out)}, expected str"
    assert isinstance(err, str), f"stderr slot is {type(err)}, expected str"
    assert "child-stdout" in out, f"the child's stdout was lost: {out!r}"
    assert "child-stderr" in err, f"the child's stderr was lost: {err!r}"
    assert "[TIMEOUT after 1.0s]" in err, (
        f"the timeout marker is missing from the stderr slot: {err!r}"
    )


def test_the_timeout_handler_never_leaks_bytes_into_the_cache() -> None:
    """A BYTES value in the memoised probe cache is the amplifier: it turns
    one slow probe into N instant failures in later gates."""
    wf343c1._HELP_PROBE_CACHE.clear()
    rc, out, err = wf343c1._run(
        [sys.executable, "-c", _SLOW_CHILD], timeout=1.0
    )
    wf343c1._HELP_PROBE_CACHE[("probe-key",)] = (rc, out, err)
    cached = wf343c1._HELP_PROBE_CACHE[("probe-key",)]
    assert isinstance(cached[1], str), "the stdout slot cached bytes"
    assert isinstance(cached[2], str), "the stderr slot cached bytes"
    wf343c1._HELP_PROBE_CACHE.clear()


def test_gate_help_survives_a_poisoned_cache_entry() -> None:
    """A cache entry holding BYTES must not raise.

    This is the exact shape that failed on 2026-11-01: ``blob = out + "\\n"
    + err`` with ``out`` being bytes.
    """
    wf343c1._HELP_PROBE_CACHE.clear()
    # The cache key is (py, script, extra_args) — build it the same way
    # gate_help does.
    key = (str(sys.executable), str(REPO / "molmetal" / "scripts" / "x.py"), ())
    wf343c1._HELP_PROBE_CACHE[key] = (124, b"child-stdout", b"child-stderr")
    result = wf343c1.gate_help(
        "t", "P1", "code", Path(key[1]), "--some-flag", py=sys.executable
    )
    assert result.status in ("TIMEOUT", "FAIL", "PASS"), result.status
    assert "TypeError" not in result.evidence, result.evidence
    wf343c1._HELP_PROBE_CACHE.clear()


def test_gate_help_reports_a_timeout_as_its_own_status() -> None:
    """rc=124 is NOT a verdict that the flag is missing.

    Reporting "help missing token(s)" for a probe that never finished is a
    FALSE DEFECT CLAIM (AGENTS.md §5.2 honest framing).
    """
    wf343c1._HELP_PROBE_CACHE.clear()
    script = REPO / "molmetal" / "scripts" / "x.py"
    key = (str(sys.executable), str(script), ())
    wf343c1._HELP_PROBE_CACHE[key] = (
        124,
        "",
        "\n[TIMEOUT after 30.0s]",
    )
    result = wf343c1.gate_help(
        "t", "P1", "code", script, "--some-flag", py=sys.executable
    )
    assert result.status == "TIMEOUT", (
        f"expected the TIMEOUT status, got {result.status}: {result.evidence}"
    )
    assert "NOT a verdict" in result.evidence, result.evidence
    wf343c1._HELP_PROBE_CACHE.clear()


def test_the_preflight_and_the_wf343c1_helper_share_one_implementation() -> None:
    """One rule, one implementation.

    A second forked copy of the bytes→text rule is exactly how this defect
    survived a 10-month-old fix in the sibling file.
    """
    from molmetal.scripts import pre_flight_check as pf  # noqa: PLC0415

    assert hasattr(wf343c1, "_as_text"), "the SSOT helper is missing"
    for value in (None, b"bytes", "text", 7):
        assert pf._wf343c1._as_text(value) == wf343c1._as_text(value)
    # The pre-flight must not define its own copy any more.
    source = (SCRIPT_DIR / "pre_flight_check.py").read_text(encoding="utf-8")
    assert "def _as_text(" not in source, (
        "pre_flight_check.py defines its own _as_text again — the rule has "
        "two implementations and can drift apart a second time."
    )


def test_the_preflight_run_helper_has_the_same_contract() -> None:
    """The pre-flight's own ``_run`` must keep the same (rc, str, str)
    contract after the delegation."""
    from molmetal.scripts import pre_flight_check as pf  # noqa: PLC0415

    rc, out, err = pf._run([sys.executable, "-c", _SLOW_CHILD], timeout=1.0)
    assert rc == 124
    assert isinstance(out, str)
    assert isinstance(err, str)
    assert "child-stderr" in err


def test_a_successful_run_is_unaffected() -> None:
    """The fix must not change the happy path."""
    rc, out, err = wf343c1._run(
        [sys.executable, "-c", "print('hello')"], timeout=30.0
    )
    assert rc == 0
    assert "hello" in out
    assert isinstance(err, str)


def test_a_missing_executable_is_still_reported_not_raised() -> None:
    rc, out, err = wf343c1._run(["/nonexistent/binary-xyz"], timeout=5.0)
    assert rc == 127
    assert isinstance(out, str)
    assert isinstance(err, str)
    assert "exec error" in err
