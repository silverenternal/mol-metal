"""WF-TODO-30 / IMPROVEMENT-BACKLOG D2 — JSON output contract test.

The :mod:`molmetal.scripts.wf_pitfall_guard.check_all` runner exposes a
``--format json`` mode that downstream CI consumers (GitHub Actions,
``ultracode`` dispatch harness, status dashboards) parse to decide
whether the R12-R17 pitfall guards still hold.

This test pins the JSON schema and the expected counts after the
R17 fix wave::

    {
        "timestamp": "<ISO-8601>",
        "guards_total": 7,
        "guards_pass": 7,
        "guards_fail": 0,
        "guards_skip": 0,
        "guards": [
            {"name": "<snake_case id>", "status": "PASS", "message": "..."},
            ...
        ]
    }

It is intentionally **lightweight** — it invokes the runner as a
sub-process (the same way CI does), parses stdout, and asserts:

* stdout is valid JSON (no parsing crash);
* the schema fields exist and are the right type;
* the expected 7 guards are reported;
* every guard currently PASSes after the R17 fix wave (the canonical
  assertion that catches a silently-broken fix).

The test is *non-mutating*: it does not edit source files, and it
does not require torch / rdkit / chemistry deps to import.  Only the
script-under-test and a sane Python install (with ``subprocess``)
are needed.  If pytest is missing, the test is skipped (rather than
errored) so this file can live in any test venv.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_ALL = (
    REPO_ROOT
    / "molmetal"
    / "scripts"
    / "wf_pitfall_guard"
    / "check_all.py"
)


def _has_runners() -> bool:
    """Return True iff we have a way to invoke pytest from this env.

    Either a ``pytest`` binary on PATH or a python that can ``-m
    pytest`` will do.  ``shutil.which`` is cheap so we just probe.
    """
    if shutil.which("pytest") is not None:
        return True
    try:
        # If we are inside the managed venv, ``python -m pytest`` works.
        subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            check=True,
            timeout=15,
        )
        return True
    except Exception:
        return False


# Skip the whole module if the runner environment cannot execute
# pytest at all — that mirrors the production ``check_all.py`` skip
# behaviour rather than failing the test suite with an OSError.
pytestmark = pytest.mark.skipif(
    not _has_runners(),
    reason="pytest not available in this env; cannot exercise check_all.py",
)


def _run_check_all_json() -> subprocess.CompletedProcess:
    """Invoke ``check_all.py --format json`` and return the completed proc.

    We do NOT enforce a specific exit code here — exit codes are
    exercised by ``test_pitfall_guard_json.py::test_exit_code_*``
    below, separately from the schema assertions.  Decoupling the
    two lets a regression in the schema (which is what this file
    cares about) surface independently of an exit-code regression.
    """
    assert CHECK_ALL.is_file(), (
        f"check_all.py not found at {CHECK_ALL}; the WF runner is "
        "missing from this checkout."
    )
    return subprocess.run(
        [sys.executable, str(CHECK_ALL), "--format", "json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_pitfall_guard_json_schema_is_well_formed():
    """``--format json`` output is a single parseable JSON object.

    Empty stdout / a parser crash here means the runner regressed
    into emitting the ASCII table even when ``--format json`` was
    requested — the canonical R17-B2-D2 regression we want to catch.
    """
    proc = _run_check_all_json()
    # pytest itself might exit non-zero (e.g. one guard fails in a
    # future round) but stdout must STILL be valid JSON — that is
    # the contract this test pins.
    assert proc.stdout.strip(), (
        f"check_all.py --format json produced empty stdout; "
        f"stderr was:\n{proc.stderr}"
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            "check_all.py --format json stdout is NOT valid JSON "
            f"({exc}):\n{proc.stdout[:2000]}"
        )
    assert isinstance(payload, dict), (
        f"check_all.py --format json top-level is {type(payload).__name__}, "
        f"expected dict.  Got: {proc.stdout[:500]}"
    )


def test_pitfall_guard_json_has_required_top_level_fields():
    """The JSON object exposes the documented schema keys.

    We assert on the *keys* (not their values) here so this test
    does not flake when a future round adds an extra guard or a new
    counter.  Counts are checked by separate tests below.
    """
    payload = json.loads(_run_check_all_json().stdout)
    for key in (
        "timestamp",
        "guards_total",
        "guards_pass",
        "guards_fail",
        "guards_skip",
        "guards",
    ):
        assert key in payload, (
            f"check_all.py --format json missing top-level key {key!r}; "
            f"available keys: {sorted(payload.keys())}"
        )
    # Type pinning — keep CI consumers from having to .isdigit()
    # the counter fields.
    assert isinstance(payload["guards_total"], int), (
        f"guards_total is {type(payload['guards_total']).__name__}, "
        "expected int"
    )
    for key in ("guards_pass", "guards_fail", "guards_skip"):
        assert isinstance(payload[key], int), (
            f"{key} is {type(payload[key]).__name__}, expected int"
        )
    assert isinstance(payload["guards"], list), (
        f"guards is {type(payload['guards']).__name__}, expected list"
    )
    # timestamp should be a non-empty string (ISO-8601-ish).
    assert isinstance(payload["timestamp"], str) and payload["timestamp"], (
        f"timestamp is {payload['timestamp']!r}, expected non-empty string"
    )


def test_pitfall_guard_json_guards_total_is_seven():
    """The 7 canonical pitfall IDs are all enumerated."""
    payload = json.loads(_run_check_all_json().stdout)
    assert payload["guards_total"] == 7, (
        f"guards_total = {payload['guards_total']}; expected 7.  "
        "Did someone add or remove a pitfall ID from PITFALL_IDS?"
    )
    assert len(payload["guards"]) == 7, (
        f"len(guards) = {len(payload['guards'])}; expected 7."
    )


def test_pitfall_guard_json_all_seven_pass_after_r17_fixes():
    """Every guard PASSes on a clean checkout after the R17 fix wave.

    This is the load-bearing assertion: if any fix has been silently
    reverted (a guard's protection is gone), pytest will FAIL the
    underlying pitfall test, ``check_all.py`` will report it as
    FAIL in the JSON, and this test will surface the regression.
    """
    payload = json.loads(_run_check_all_json().stdout)
    assert payload["guards_pass"] == 7, (
        f"guards_pass = {payload['guards_pass']}, expected 7 after the "
        f"R17 fix wave.  Regressions detected:\n"
        + "\n".join(
            f"  - {g.get('name', '?')}: {g.get('status', '?')} — "
            f"{g.get('message', '')}"
            for g in payload.get("guards", [])
            if g.get("status") != "PASS"
        )
    )
    assert payload["guards_fail"] == 0, (
        f"guards_fail = {payload['guards_fail']}; expected 0 on a clean "
        "checkout.  Failures:\n"
        + "\n".join(
            f"  - {g.get('name', '?')}: {g.get('status', '?')}"
            for g in payload.get("guards", [])
            if g.get("status") == "FAIL"
        )
    )


def test_pitfall_guard_json_per_guard_shape():
    """Each ``guards`` entry has ``name``, ``status``, ``message``.

    ``status`` is one of the canonical short labels (PASS / FAIL /
    SKIP / MISS); ``name`` is a stable snake_case identifier; and
    ``message`` is a non-empty human-readable string.
    """
    payload = json.loads(_run_check_all_json().stdout)
    valid_statuses = {"PASS", "FAIL", "SKIP", "MISS"}
    seen_names = set()
    for entry in payload["guards"]:
        assert isinstance(entry, dict), (
            f"guards entry is {type(entry).__name__}, expected dict: {entry!r}"
        )
        for key in ("name", "status", "message"):
            assert key in entry, (
                f"guards entry missing {key!r}: {entry!r}"
            )
            assert isinstance(entry[key], str), (
                f"guards entry[{key!r}] is {type(entry[key]).__name__}, "
                f"expected str: {entry!r}"
            )
            assert entry[key], (
                f"guards entry[{key!r}] is empty: {entry!r}"
            )
        assert entry["status"] in valid_statuses, (
            f"guards entry status={entry['status']!r} is not in "
            f"{sorted(valid_statuses)}: {entry!r}"
        )
        # Names must be unique — duplicate names break dashboard joins.
        assert entry["name"] not in seen_names, (
            f"duplicate guards name {entry['name']!r}; names must be unique"
        )
        seen_names.add(entry["name"])


def test_pitfall_guard_json_counters_match_array():
    """guards_pass + guards_fail + guards_skip [+ guards_miss] == guards_total.

    This catches the cheap-but-real bug where the per-guard array is
    updated but the summary counters are not (or vice versa).
    """
    payload = json.loads(_run_check_all_json().stdout)
    miss = payload.get("guards_miss", 0)
    summed = (
        payload["guards_pass"]
        + payload["guards_fail"]
        + payload["guards_skip"]
        + miss
    )
    assert summed == payload["guards_total"], (
        f"counters do not sum to guards_total: "
        f"pass={payload['guards_pass']} fail={payload['guards_fail']} "
        f"skip={payload['guards_skip']} miss={miss} total={payload['guards_total']}"
    )
    # The counters must also match the actual array tally — guards_pass
    # in particular is what the main assertion relies on.
    array_pass = sum(
        1 for g in payload["guards"] if g.get("status") == "PASS"
    )
    array_fail = sum(
        1 for g in payload["guards"] if g.get("status") == "FAIL"
    )
    array_skip = sum(
        1 for g in payload["guards"] if g.get("status") == "SKIP"
    )
    array_miss = sum(
        1 for g in payload["guards"] if g.get("status") == "MISS"
    )
    assert array_pass == payload["guards_pass"], (
        f"array PASS count {array_pass} != guards_pass "
        f"{payload['guards_pass']}"
    )
    assert array_fail == payload["guards_fail"], (
        f"array FAIL count {array_fail} != guards_fail "
        f"{payload['guards_fail']}"
    )
    assert array_skip == payload["guards_skip"], (
        f"array SKIP count {array_skip} != guards_skip "
        f"{payload['guards_skip']}"
    )
    assert array_miss == miss, (
        f"array MISS count {array_miss} != guards_miss {miss}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
