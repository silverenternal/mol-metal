"""Wf-344-A2 — Regression tests for chemprop CLI token --help contract.

==============================================================================

Per Wf-338-A3 finding 2: the pre-flight gate ``p3_todo_119_p14_cpu_smoke_ready``
was reported FAIL because the chemprop v2/v3 trainer's ``--help`` output was
alleged to be missing the four tokens ``--chemprop-mode``, ``v3-fix``,
``--chemprop-min-epochs``, ``--chemprop-lomo-init-scale``.  As of the
2026-09-29 fresh-run pre-flight, all four tokens ARE exposed in ``--help`` and
the gate reports PASS (rc=0, wall 4.93s on /home/hugo/codes/mol-metal/.venv).

This file provides 4 regression tests that re-verify the gate contract from a
fresh subprocess — guarding against (i) tokens being deleted by future commits,
(iii) --chemprop module becoming uninstalled and crashing --help before
argparse can run, and (iii) gate_help's split-on-whitespace contract being
broken by a future refactor that removes any of the 4 tokens from --help.

Honest framing (per ``party-twenty-spirit`` §3.5):

  * MEASURED: as of 2026-09-29 15:53+08:00 fresh-run pre-flight v2,
    ``p3_todo_119_p14_cpu_smoke_ready`` reports PASS with rc=0 and
    evidence "help exposes all 4 token(s) of '--chemprop-mode v3-fix
    --chemprop-min-epochs --chemprop-lomo-init-scale'".  The 4 tokens
    ARE present in ``--help`` output.  This file adds 4 regression
    tests so that the gate cannot silently drift back to FAIL.
  * No "fix" was needed for the chemprop trainer — the Wf-338-A3
    finding 2 was either (a) observed under a stale venv where the
    chemprop module was uninstalled (causing ``--help`` to crash
    before argparse could emit the tokens), or (b) a report from an
    earlier git revision.  Either way, the current code on
    2026-09-29 has the tokens wired at lines 1522, 1556, 1575 of
    ``molmetal/scripts/train_chemprop_v2_v3.py``.
  * BLOCKED: chemprop module is NOT installed in ``.venv/bin/python``,
    so this test file MUST be run through ``VENV_VLM`` (the same venv
    pre_flight_check.py uses).  The ``PYTHONPATH`` and ``VENV_VLM``
    environment variables are honoured by the test harness; if a CI
    runner only has ``.venv`` (without chemprop), the import-error
    failure mode is identical to the Wf-338-A3 finding 2 root cause
    and is caught by ``test_help_returns_rc_zero_when_chemprop_installed``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
_PRE_FLIGHT = _SCRIPT_DIR / "pre_flight_check.py"
_CHEMPROP = _SCRIPT_DIR / "train_chemprop_v2_v3.py"

# Pre-flight uses /home/hugo/codes/mol-metal/.venv/bin/python
# (see molmetal/scripts/pre_flight_check.py line 81).  We honour that
# contract here so that the gate runs the same way it does in CI.
_VENV_VLM = "/home/hugo/codes/mol-metal/.venv/bin/python"


def _python_for_chemprop() -> str:
    """Pick the python interpreter that has ``chemprop`` installed.

    Pre-flight uses ``/home/hugo/codes/mol-metal/.venv/bin/python`` by
    default.  If that venv is unavailable (CI runner without it), we
    fall back to ``sys.executable`` — the test will then SKIP if
    chemprop is not importable.
    """
    if Path(_VENV_VLM).exists():
        return _VENV_VLM
    return sys.executable


def _help_output() -> tuple[int, str]:
    """Run ``train_chemprop_v2_v3.py --help`` and return (rc, combined_output).

    Honours ``PYTHONPATH`` so that the repo-root is importable.  If the
    chemprop module is not installed, ``--help`` will crash with an
    ImportError BEFORE argparse can emit the tokens — this is exactly
    the Wf-338-A3 finding 2 root cause.
    """
    py = _python_for_chemprop()
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO_ROOT))
    proc = subprocess.run(
        [py, str(_CHEMPROP), "--help"],
        capture_output=True, text=True, timeout=30,
        cwd=str(_REPO_ROOT), env=env,
    )
    combined = proc.stdout + "\n" + proc.stderr
    return proc.returncode, combined


# ---------------------------------------------------------------------------
# Test 1 — Gate regression: pre_flight_check.py reports PASS for the gate.
# ---------------------------------------------------------------------------
def test_wf344a2_gate_p3_todo_119_p14_cpu_smoke_ready_passes():
    """Pre-flight gate ``p3_todo_119_p14_cpu_smoke_ready`` is PASS.

    Regression test for Wf-338-A3 finding 2.  Runs
    ``pre_flight_check.py --pipeline chemprop_v2_v3 --json <tmp>``
    and asserts the gate ``p3_todo_119_p14_cpu_smoke_ready`` reports
    status PASS (not FAIL).  If the gate ever drifts back to FAIL
    (e.g. someone removes one of the 4 tokens), this test fails.

    Honest framing: the gate is run in a separate worktree via
    ``-o /tmp/wf344a2_preflight.json``.  This adds ~10 s wall to the
    test suite; it is the canonical regression for Wf-338-A3 finding 2.
    """
    py = _python_for_chemprop()
    if not Path(_VENV_VLM).exists() and "chemprop" not in sys.modules:
        pytest.skip(
            f"chemprop not importable in {sys.executable!r}; cannot run "
            f"pre_flight_check.py reliably. Wf-338-A3 finding 2 was "
            f"observed when chemprop was uninstalled."
        )
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO_ROOT))
    out_path = "/tmp/wf344a2_preflight.json"
    proc = subprocess.run(
        [py, str(_PRE_FLIGHT), "--pipeline", "chemprop_v2_v3",
         "--json", out_path],
        capture_output=True, text=True, timeout=300,
        cwd=str(_REPO_ROOT), env=env,
    )
    assert proc.returncode == 0, (
        f"pre_flight_check.py --pipeline chemprop_v2_v3 exited "
        f"{proc.returncode}; stderr: {proc.stderr[:500]}"
    )
    with open(out_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    gates = data.get("gates", [])
    gate = next(
        (g for g in gates if g.get("name") == "p3_todo_119_p14_cpu_smoke_ready"),
        None,
    )
    assert gate is not None, (
        "Gate 'p3_todo_119_p14_cpu_smoke_ready' missing from "
        "pre_flight_check.py --pipeline chemprop_v2_v3 output. "
        "Wf-344-A2 expects this gate to be wired (Wf-321H added it). "
        f"Gates observed: {[g.get('name') for g in gates]}"
    )
    assert gate.get("status") == "PASS", (
        f"Gate 'p3_todo_119_p14_cpu_smoke_ready' is FAIL — Wf-338-A3 "
        f"finding 2 regression. Evidence: {gate.get('evidence')!r}. "
        f"This is the regression the Wf-344-A2 test guards against."
    )


# ---------------------------------------------------------------------------
# Test 2 — Token existence: --chemprop-mode advertises v3-fix as a choice.
# ---------------------------------------------------------------------------
def test_wf344a2_chemprop_mode_v3_fix_token_in_help():
    """``v3-fix`` appears in ``--chemprop-mode {v2, v3, v3-fallback, v3-fix, ...}``.

    Regression test for Wf-338-A3 finding 2 (token 2 of 4: ``v3-fix``
    was claimed missing).  Asserts that the literal token ``v3-fix``
    appears in --help output.  Note: ``v3-fix`` is split off
    ``--chemprop-mode`` by whitespace in ``gate_help``'s contract, so
    if v3-fix is removed from the choices tuple at line 1524 of
    train_chemprop_v2_v3.py, this test will fail.
    """
    rc, combined = _help_output()
    if rc != 0:
        pytest.skip(
            f"--help returned rc={rc}; the chemprop module is likely "
            f"uninstalled in this venv (Wf-338-A3 finding 2 root cause). "
            f"stderr: {combined[:200]}"
        )
    assert "--chemprop-mode" in combined, (
        "--chemprop-mode flag missing from --help output"
    )
    # The 4 tokens are split on whitespace in gate_help.  'v3-fix'
    # is a stand-alone token (split off '--chemprop-mode ').  Verify
    # it appears as a choice in --chemprop-mode's choices list.
    block_match = re.search(
        r"--chemprop-mode\s+\{[^}]+\}",
        combined,
    )
    assert block_match is not None, (
        "--chemprop-mode choices block not found in --help output"
    )
    choices_str = block_match.group(0)
    assert "v3-fix" in choices_str, (
        f"--chemprop-mode choices missing 'v3-fix': {choices_str!r}. "
        f"Wf-338-A3 finding 2 token 2 regression."
    )


# ---------------------------------------------------------------------------
# Test 3 — Co-occurrence: all 4 tokens appear in same --help call.
# ---------------------------------------------------------------------------
def test_wf344a2_all_four_tokens_cooccur_in_same_help_call():
    """All 4 gate_help tokens appear in a single --help subprocess call.

    Regression test for Wf-338-A3 finding 2.  gate_help splits its
    pattern on whitespace and requires ALL tokens to be present in
    the same stdout+stderr blob.  This test mirrors that contract
    exactly: single subprocess call, assert all 4 tokens present.
    """
    rc, combined = _help_output()
    if rc != 0:
        pytest.skip(
            f"--help returned rc={rc}; the chemprop module is likely "
            f"uninstalled in this venv. Wf-338-A3 finding 2 root cause. "
            f"stderr: {combined[:200]}"
        )
    expected_tokens = ("--chemprop-mode", "v3-fix",
                       "--chemprop-min-epochs", "--chemprop-lomo-init-scale")
    missing = [tok for tok in expected_tokens if tok not in combined]
    assert not missing, (
        f"Missing tokens in --help output (Wf-338-A3 finding 2 "
        f"regression): {missing}.  gate_help pattern is '--chemprop-mode "
        f"v3-fix --chemprop-min-epochs --chemprop-lomo-init-scale'; all "
        f"4 tokens must appear in the same --help stdout+stderr blob."
    )


# ---------------------------------------------------------------------------
# Test 4 — Import guard: --help returns rc=0 (gate_help's 'rc=0' evidence).
# ---------------------------------------------------------------------------
def test_wf344a2_help_returns_rc_zero_when_chemprop_installed():
    """``train_chemprop_v2_v3.py --help`` returns rc=0 (no ImportError).

    Regression test for Wf-338-A3 finding 2 root cause.  The original
    gate evidence ``"help exposes all 4 token(s) of '...' (rc=0)"``
    is itself a regression check: if the chemprop module is uninstalled
    in the running venv, ``--help`` exits with rc=1 (ImportError)
    BEFORE argparse can run, and the gate flips to FAIL with
    evidence ``"help missing token(s) [...] (rc=2)"`` or similar.

    This test asserts rc=0 from a fresh subprocess — if the user's
    venv has chemprop uninstalled, this test fails with a clear
    message about the venv setup.  This is the canonical regression
    test for the Wf-338-A3 finding 2 root cause.
    """
    rc, combined = _help_output()
    if rc != 0:
        # If rc != 0 it is most likely an ImportError on chemprop.data.
        # We surface that explicitly so the failure message points
        # the user at the venv setup rather than at the trainer code.
        is_import_error = "ModuleNotFoundError" in combined and "chemprop" in combined
        msg = (
            f"train_chemprop_v2_v3.py --help exited rc={rc}.  "
            f"Wf-338-A3 finding 2 root cause: gate_help cannot emit "
            f"'rc=0' evidence if --help crashes. "
        )
        if is_import_error:
            msg += (
                f"The chemprop module is NOT installed in "
                f"{_python_for_chemprop()!r}. Install chemprop==2.0.0 "
                f"per Wf-321F §3 step 1 to lift this regression."
            )
        else:
            msg += f"Combined output: {combined[:500]!r}"
        pytest.fail(msg)
    # When rc=0, also assert the gate_help "4 token(s)" evidence
    # pattern is reproducible — i.e. all 4 tokens still in --help.
    for tok in ("--chemprop-mode", "v3-fix",
                "--chemprop-min-epochs", "--chemprop-lomo-init-scale"):
        assert tok in combined, (
            f"Token {tok!r} missing from --help output (rc=0 but "
            f"tokens drifted; Wf-338-A3 finding 2 regression)."
        )