# SPDX-License-Identifier: Apache-2.0
"""The sweep launcher must FAIL LOUDLY, not print a traceback and exit 0.

THE DEFECT THIS PINS
--------------------
``scripts/launch_cfm_l3_guidance_sweep.sh`` read its protocol values inside an
``echo "...$("$PY" -c ...)..."`` command substitution.  Bash discards a
substitution's exit status when the result is interpolated into a string, so
when the YAML lookup raised ``KeyError`` the traceback was printed to stdout,
the field came out blank, and the script still ended ``exit 0``.

AGENTS.md section 4 defines 0 = PASS.  A pre-flight that reports PASS while
printing a traceback is worse than one that fails loudly: it authorises a
multi-hour GPU sweep on a config nobody has successfully read.

RED CHECK: delete the ``if ! ... then exit 1`` guards from the launcher and
every ``test_*_fails_loudly`` below returns 0 instead of non-zero. Restore them
and the suite is green.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
LAUNCHER = REPO / "scripts" / "launch_cfm_l3_guidance_sweep.sh"
CFG = REPO / "configs" / "cfm_l3_guidance_sweep_2026_10_12.yaml"


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


def _assert_failed_loudly(r: subprocess.CompletedProcess, what: str) -> None:
    """Non-zero is NOT sufficient — a bash SYNTAX error also exits non-zero.

    Without this distinction a launcher that had been broken by a careless edit
    would pass every ``fails_loudly`` test below while being incapable of
    launching anything. That vacuous-red-check shape was hit for real while
    writing this file: the first mutation attempt left a dangling ``then`` and
    all six failure tests "passed" against a script bash could not parse.

    So a loud failure must (a) be a non-zero exit, (b) parse cleanly, and
    (c) print a FAIL line of its own -- not merely a traceback.
    """
    assert r.returncode != 0, f"{what}: exited 0, so the gate did not fire\n{r.stdout}"
    # rc=2 is bash's "syntax error / usage error"; rc=126/127 are "cannot
    # execute". A gate that fails for those reasons is not gating.
    assert r.returncode == 1, (
        f"{what}: exited {r.returncode}, which is bash's syntax/exec error "
        "rather than the script's own FAIL path. The launcher is broken, not "
        f"gating.\n{r.stdout}\n{r.stderr}"
    )
    assert "FAIL" in (r.stdout + r.stderr), (
        f"{what}: no FAIL line was printed, so the failure is still the "
        f"swallowed kind.\n{r.stdout}\n{r.stderr}"
    )


def test_launcher_parses():
    """A syntax error would make every other test in this file vacuous."""
    r = subprocess.run(
        ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True
    )
    assert r.returncode == 0, f"launcher does not parse:\n{r.stderr}"


@pytest.fixture
def cfg_copy(tmp_path):
    """A scratch copy of the real config; restored after every test."""
    backup = tmp_path / "cfg.bak.yaml"
    shutil.copy(CFG, backup)
    yield CFG
    shutil.copy(backup, CFG)


def _write_cfg(mutate) -> None:
    data = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    mutate(data)
    CFG.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_launcher_exists():
    assert LAUNCHER.is_file(), f"missing {LAUNCHER}"


def test_real_config_dry_run_passes():
    """The shipped config must still PASS — a gate that always fails is useless."""
    r = _run()
    assert r.returncode == 0, f"real config should pass, got rc={r.returncode}\n{r.stdout}"
    assert "Traceback" not in r.stdout, f"dry-run printed a traceback:\n{r.stdout}"
    assert "n_clouds/n_steps:" in r.stdout


def test_dry_run_prints_no_traceback():
    r = _run()
    assert "Traceback" not in (r.stdout + r.stderr)


def test_missing_protocol_section_fails_loudly(cfg_copy):
    """A KeyError inside the substitution must NOT become exit 0."""

    def _drop(data):
        data.pop("eval", None)

    _write_cfg(_drop)
    _assert_failed_loudly(_run(), "no eval: section")


def test_missing_sweep_section_fails_loudly(cfg_copy):
    def _drop(data):
        data.pop("sweep", None)

    _write_cfg(_drop)
    _assert_failed_loudly(_run(), "no sweep: section")


def test_control_arm_absent_fails_loudly(cfg_copy):
    """Without a 0.0 arm every delta is uninterpretable, so refuse to launch."""

    def _no_control(data):
        data["sweep"]["strengths"] = [0.01, 0.05, 0.15]

    _write_cfg(_no_control)
    _assert_failed_loudly(_run(), "grid without a 0.0 control arm")


def test_duplicate_arms_fail_loudly(cfg_copy):
    """A repeated arm spends GPU twice for the same number."""

    def _dup(data):
        data["sweep"]["strengths"] = [0.0, 0.05, 0.05]

    _write_cfg(_dup)
    _assert_failed_loudly(_run(), "duplicate arms")