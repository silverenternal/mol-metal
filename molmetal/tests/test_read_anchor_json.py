"""Wf-302A — tests for ``molmetal.scripts.read_anchor_json.AnchorGuard``.

=====================================================================
Wf-291H ship surface: tests for the read_anchor_json helper.

The four behaviors we pin down here:

  1. **Fresh anchor passes** — a brand-new anchor with a matching
     ``version`` field returns the parsed dict and triggers no
     exception.
  2. **Stale anchor raises** — an anchor whose mtime is older than
     ``max_age_sec`` raises :class:`AnchorStaleError`.
  3. **Wrong version raises** — an anchor whose ``version_field``
     does not match ``required_version`` raises
     :class:`AnchorVersionMismatchError`.
  4. **Missing anchor raises** — a path that does not exist raises
     :class:`AnchorMissingError`.

These four behaviors directly map to the bug classes in
``molmetal/scripts/read_anchor_json.py`` (stale read, version drift,
missing read). The fifth "JSON parse" guard is implicitly covered
because every passing test parses real JSON; we add a small extra
parse-failure sanity check at the bottom to round out coverage.

Hardware: CPU-only. No GPU required.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest


# Ensure the ``molmetal.scripts`` package is importable when this test
# file is invoked from the repo root (which is how ``run_tests.sh``
# invokes it). ``molmetal`` itself is normally importable because the
# test runner sets ``PYTHONPATH``, but be defensive.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from molmetal.scripts.read_anchor_json import (  # noqa: E402
    AnchorGuard,
    AnchorJSONError,
    AnchorMissingError,
    AnchorStaleError,
    AnchorVersionMismatchError,
    read_anchor_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_anchor(path: Path, payload: dict) -> None:
    """Write ``payload`` as pretty JSON to ``path`` and refresh mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Touch mtime to NOW so the fresh-anchor tests don't accidentally
    # inherit a stale mtime from a prior test.
    os.utime(path, (time.time(), time.time()))


def _backdate_mtime(path: Path, age_sec: float) -> None:
    """Set ``path``'s mtime to ``age_sec`` seconds in the past."""
    past = time.time() - age_sec
    os.utime(path, (past, past))


# ---------------------------------------------------------------------------
# Test 1 — fresh anchor passes (success path)
# ---------------------------------------------------------------------------
def test_fresh_anchor_passes(tmp_path: Path) -> None:
    """A freshly-written anchor with matching version MUST read without error.

    Builds a tiny anchor dict resembling the Wf-287A ``m14_anchor.json``
    shape (``metric`` + ``value`` + ``wf_origin`` + ``status``) plus a
    ``version`` field that the helper pins against. Verifies that:

      * ``guard.read()`` returns the parsed dict.
      * ``guard.require()`` returns the same dict (alias).
      * The standalone ``read_anchor_json()`` convenience wrapper
        returns the same dict.
    """
    anchor_path = tmp_path / "fresh_anchor.json"
    payload = {
        "metric": "cfm_decode_real_m14_anchor",
        "wf_origin": "Wf-188",
        "value": 0.9958,
        "n_samples": 26801,
        "version": "m14_v1",
        "status": "MEASURED_PASS",
    }
    _write_anchor(anchor_path, payload)

    guard = AnchorGuard(
        path=anchor_path,
        max_age_sec=86400.0,
        required_version="m14_v1",
    )

    # read() — primary API
    parsed = guard.read()
    assert parsed == payload, f"expected parsed == payload; got {parsed!r}"
    assert parsed["value"] == 0.9958
    assert parsed["version"] == "m14_v1"

    # require() — alias API
    parsed2 = guard.require()
    assert parsed2 == payload, "require() must return the same dict as read()"

    # read_anchor_json() — convenience function
    parsed3 = read_anchor_json(
        path=anchor_path,
        max_age_sec=86400.0,
        required_version="m14_v1",
    )
    assert parsed3 == payload, "read_anchor_json() must match read()"


# ---------------------------------------------------------------------------
# Test 2 — stale anchor raises (max_age_sec enforcement)
# ---------------------------------------------------------------------------
def test_stale_anchor_raises(tmp_path: Path) -> None:
    """An anchor whose mtime is older than ``max_age_sec`` MUST raise :class:`AnchorStaleError`.

    Writes an anchor, backdates its mtime to 30 days ago, then asks for
    a 24 hr (86400 sec) max-age. The guard MUST trip with a named
    :class:`AnchorStaleError` carrying a useful message (age in sec).
    """
    anchor_path = tmp_path / "stale_anchor.json"
    payload = {
        "metric": "cfm_decode_real_m14_anchor",
        "value": 0.9958,
        "version": "m14_v1",
    }
    _write_anchor(anchor_path, payload)

    # Backdate to 30 days ago (2,592,000 sec). 24 hr = 86,400 sec,
    # so this is well outside any reasonable freshness window.
    _backdate_mtime(anchor_path, age_sec=30 * 86400.0)

    guard = AnchorGuard(
        path=anchor_path,
        max_age_sec=86400.0,  # 24 hr
    )
    with pytest.raises(AnchorStaleError) as excinfo:
        guard.read()

    # The error message MUST surface the measured age + the max_age_sec
    # so the caller can diagnose without re-stat'ing the file.
    msg = str(excinfo.value)
    assert "stale" in msg.lower(), f"stale message missing 'stale' keyword: {msg!r}"
    assert "max_age_sec" in msg, f"stale message missing max_age_sec context: {msg!r}"

    # require() must raise the same error class.
    with pytest.raises(AnchorStaleError):
        guard.require()

    # The convenience function must also raise AnchorStaleError.
    with pytest.raises(AnchorStaleError):
        read_anchor_json(path=anchor_path, max_age_sec=86400.0)


# ---------------------------------------------------------------------------
# Test 3 — wrong version raises (version pinning)
# ---------------------------------------------------------------------------
def test_wrong_version_raises(tmp_path: Path) -> None:
    """An anchor whose ``version_field`` value does NOT match MUST raise :class:`AnchorVersionMismatchError`.

    Writes an anchor with ``version="m14_v1"`` but constructs an
    :class:`AnchorGuard` with ``required_version="m14_v2"``. The
    guard MUST trip with a named
    :class:`AnchorVersionMismatchError` whose message includes both
    the expected and actual versions.

    Also verifies that:

      * When ``required_version=None`` the version check is skipped
        (the same file reads cleanly).
      * When the anchor lacks the ``version_field`` key entirely,
        a non-None ``required_version`` still trips the guard
        (``actual`` becomes ``None`` which != ``"m14_v2"``).
    """
    anchor_path = tmp_path / "wrong_version_anchor.json"
    payload = {
        "metric": "cfm_decode_real_m14_anchor",
        "value": 0.9958,
        "version": "m14_v1",
    }
    _write_anchor(anchor_path, payload)

    # Case A — strict version mismatch
    guard = AnchorGuard(
        path=anchor_path,
        max_age_sec=86400.0,
        required_version="m14_v2",
        version_field="version",
    )
    with pytest.raises(AnchorVersionMismatchError) as excinfo:
        guard.read()
    msg = str(excinfo.value)
    assert "m14_v2" in msg, f"expected version in error message: {msg!r}"
    assert "m14_v1" in msg, f"actual version in error message: {msg!r}"

    # Case B — required_version=None ⇒ check skipped
    guard_open = AnchorGuard(
        path=anchor_path,
        max_age_sec=86400.0,
        required_version=None,
    )
    parsed = guard_open.read()
    assert parsed == payload, "no required_version ⇒ skip version check"

    # Case C — anchor lacks the version_field entirely
    no_version_path = tmp_path / "no_version_anchor.json"
    _write_anchor(
        no_version_path,
        {"metric": "cfm_decode_real_m14_anchor", "value": 0.9958},
    )
    guard_no_field = AnchorGuard(
        path=no_version_path,
        max_age_sec=86400.0,
        required_version="m14_v2",
        version_field="version",
    )
    with pytest.raises(AnchorVersionMismatchError):
        guard_no_field.read()


# ---------------------------------------------------------------------------
# Test 4 — missing anchor raises (path-exists guard)
# ---------------------------------------------------------------------------
def test_missing_anchor_raises(tmp_path: Path) -> None:
    """A path that does not exist MUST raise :class:`AnchorMissingError`.

    The guard must trip before any stat / read attempt — the test
    verifies this implicitly by checking that the path does not
    exist on disk, and explicitly by checking the exception type.

    Also verifies that :class:`AnchorMissingError` is a subclass of
    :class:`FileNotFoundError` so legacy callers using bare
    ``except FileNotFoundError`` keep working.
    """
    missing_path = tmp_path / "does_not_exist.json"
    assert (
        not missing_path.exists()
    ), f"test setup error: {missing_path} should not exist"

    guard = AnchorGuard(path=missing_path, max_age_sec=86400.0)

    with pytest.raises(AnchorMissingError) as excinfo:
        guard.read()

    # Exception MUST surface the path so the caller can log it.
    assert str(missing_path) in str(
        excinfo.value
    ), f"missing-path message must contain path; got {excinfo.value!r}"

    # require() — same exception
    with pytest.raises(AnchorMissingError):
        guard.require()

    # Convenience function — same exception
    with pytest.raises(AnchorMissingError):
        read_anchor_json(path=missing_path)

    # Backward-compat: AnchorMissingError is a FileNotFoundError so
    # legacy ``except FileNotFoundError`` callers keep working.
    assert issubclass(
        AnchorMissingError, FileNotFoundError
    ), "AnchorMissingError must inherit FileNotFoundError for back-compat"


# ---------------------------------------------------------------------------
# Bonus — JSON parse guard (the Wf-291H §4 explicit-error contract)
# ---------------------------------------------------------------------------
def test_malformed_json_raises_anchor_json_error(tmp_path: Path) -> None:
    """Malformed JSON MUST raise :class:`AnchorJSONError` (the explicit-error contract).

    This is the Wf-291H §4 explicit-error contract: rather than letting
    a raw :class:`json.JSONDecodeError` bubble up with line numbers
    pointing at the helper internals, ``AnchorGuard`` wraps it as
    :class:`AnchorJSONError` with the anchor path prefixed.
    """
    bad_path = tmp_path / "malformed.json"
    bad_path.write_text("{not valid json at all", encoding="utf-8")
    # Refresh mtime so we hit the JSON path, not the stale path.
    os.utime(bad_path, (time.time(), time.time()))

    guard = AnchorGuard(path=bad_path, max_age_sec=86400.0)
    with pytest.raises(AnchorJSONError) as excinfo:
        guard.read()

    msg = str(excinfo.value)
    assert (
        str(bad_path) in msg
    ), f"AnchorJSONError message must include anchor path; got {msg!r}"


# ---------------------------------------------------------------------------
# Test 6 — Wf-305C anchor version pin (live-anchor integration test)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "anchor_path",
    [
        "molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json",
        "molmetal/reports/wf269c_cfm_real/wf269c_report.json",
        "molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.json",
        "molmetal/reports/wf219_krasnov_aligned/sweep_summary.json",
    ],
)
def test_anchor_version_pin_required_version_passes(anchor_path: str) -> None:
    """Wf-305C: All Wf-305C-pinned paper-grade anchors MUST carry 'version': '1.0.0'.

    After the Wf-305C version-pin pass, every anchor listed in the
    TODO-287A scope MUST carry ``"version": "1.0.0"`` at the top
    level. Construct an :class:`AnchorGuard` with
    ``required_version="1.0.0"`` and a very large
    ``max_age_sec`` (anchors written earlier today are still fresh
    under a 7-day window) — the guard MUST read cleanly and the
    returned dict MUST include ``"version": "1.0.0"``.

    On any pin failure the helper raises
    :class:`AnchorVersionMismatchError`, so a passing test is
    proof that the version-pin contract is being honored.
    """
    path = _REPO_ROOT / anchor_path
    if not path.exists():
        pytest.skip(f"anchor not on disk: {path}")

    # 2026-10-09 — calendar-independence fix. The window was 7 days and the
    # docstring above said "anchors written earlier today are still fresh",
    # which made this a TIME BOMB: it began failing 7 days after the Wf-188 /
    # Wf-253d / Wf-219 anchors were written, with no code change at all
    # (measured age at failure: 60929 s ≈ 16.9 days). A test whose subject is
    # the VERSION-PIN contract must not also assert that the repo's historical
    # artifacts are recent.
    #
    # Freshness is covered where it belongs:
    #   * test_fresh_anchor_passes / test_stale_anchor_raises in this file, both
    #     on synthetic tmp_path anchors (so they test the LOGIC, not the clock);
    #   * pre_flight_check.py's anchor-freshness gates, which judge real anchors
    #     at their intended 24 hr horizon and are the right place for a real
    #     "these artifacts have aged out" signal.
    guard = AnchorGuard(
        path=path,
        # Effectively unbounded: see above. The guard is still constructed with
        # required_version so the version-pin LOGIC is still what is exercised.
        max_age_sec=100 * 365 * 86400.0,
        required_version="1.0.0",
        version_field="version",
    )
    parsed = guard.read()

    assert parsed.get("version") == "1.0.0", (
        f"anchor at {path} must carry version='1.0.0'; "
        f"got version={parsed.get('version')!r}"
    )


def test_anchor_version_pin_mismatch_raises(tmp_path: Path) -> None:
    """Wf-305C: required_version='2.0.0' against a '1.0.0' anchor MUST raise.

    Pins down the NEGATIVE side of the version-pin contract: if a
    downstream consumer requests ``required_version='2.0.0'`` against
    an anchor stamped ``'1.0.0'`` (the current Wf-305C baseline), the
    helper MUST raise :class:`AnchorVersionMismatchError`. This is
    the loud-failure mode that protects consumers from silent
    schema drift — see ``read_anchor_json.py`` docstring §"Why a
    helper rather than ad-hoc json.loads".
    """
    anchor_path = tmp_path / "version_pin_mismatch.json"
    payload = {
        "metric": "cfm_decode_real_m14_anchor",
        "wf_origin": "Wf-188",
        "value": 0.9958,
        "version": "1.0.0",
        "status": "MEASURED_PASS",
    }
    _write_anchor(anchor_path, payload)

    guard = AnchorGuard(
        path=anchor_path,
        max_age_sec=86400.0,
        required_version="2.0.0",
        version_field="version",
    )
    with pytest.raises(AnchorVersionMismatchError) as excinfo:
        guard.read()
    msg = str(excinfo.value)
    assert (
        "2.0.0" in msg
    ), f"error message must contain expected version '2.0.0'; got {msg!r}"
    assert (
        "1.0.0" in msg
    ), f"error message must contain actual version '1.0.0'; got {msg!r}"


# ---------------------------------------------------------------------------
# Test 7 — Wf-306C AnchorGuard version-pin regression block
# ---------------------------------------------------------------------------
def test_anchor_version_pin_blocks_regression(tmp_path: Path) -> None:
    """Wf-306C: anchor version-pin must catch silent regressions to non-pinned versions.

    The Wf-305C version-pin contract says every opt-in paper-grade
    anchor MUST carry ``"version": "1.0.0"``. This test enforces the
    contract's regression-detection capability: if a future contributor
    silently bumps an anchor's version string (e.g. ``"1.0.1"`` or
    ``"2.0.0"``) WITHOUT updating the consumer's ``required_version``,
    the helper MUST trip with :class:`AnchorVersionMismatchError`
    BEFORE the consumer reads the anchor's stale values.

    The test exercises three regression variants:

      1. **Forward bump**: ``version="1.0.1"`` (a hypothetical patch
         release) against ``required_version="1.0.0"`` → FAIL.
      2. **Major bump**: ``version="2.0.0"`` (a hypothetical breaking
         release) against ``required_version="1.0.0"`` → FAIL.
      3. **Backward drift**: ``version="0.9.9"`` (a hypothetical
         pre-pinned legacy) against ``required_version="1.0.0"`` →
         FAIL.

    In all three cases the helper MUST raise ``AnchorVersionMismatchError``
    with both expected + actual versions in the message so the
    downstream consumer can diagnose the version drift without
    re-stat'ing the anchor file.

    This is the regression-block test the Wf-306C PART 3 deliverable
    adds; without it, a silent version bump in the anchor fleet
    could pass :func:`test_anchor_version_pin_required_version_passes`
    *only* if every consumer's ``required_version`` is bumped in
    lockstep — which is exactly the silent drift the Wf-291H §4
    explicit-error contract is designed to prevent.
    """
    pin = "1.0.0"
    cases = [
        ("forward_bump", "1.0.1"),
        ("major_bump", "2.0.0"),
        ("backward_drift", "0.9.9"),
    ]
    for label, version_value in cases:
        anchor_path = tmp_path / f"regression_{label}.json"
        payload = {
            "metric": "cfm_decode_real_m14_anchor",
            "wf_origin": "Wf-188",
            "value": 0.9958,
            "version": version_value,
            "status": "MEASURED_PASS",
        }
        _write_anchor(anchor_path, payload)

        guard = AnchorGuard(
            path=anchor_path,
            max_age_sec=86400.0,
            required_version=pin,
            version_field="version",
        )
        with pytest.raises(AnchorVersionMismatchError) as excinfo:
            guard.read()
        msg = str(excinfo.value)
        # The error MUST surface both the expected pin AND the actual
        # regression value so the consumer can diagnose without
        # re-stat'ing the file.
        assert pin in msg, (
            f"[{label}] error message must contain expected pin {pin!r}; "
            f"got {msg!r}"
        )
        assert version_value in msg, (
            f"[{label}] error message must contain actual version "
            f"{version_value!r}; got {msg!r}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
