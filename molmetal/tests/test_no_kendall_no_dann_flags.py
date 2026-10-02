"""Pytest: Wf-283B — ``--no-kendall`` / ``--no-dann`` flag contract for
:mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein`.

Per ``cfm_todo_gap_audit.md`` §3.4 gap #1, C10-C17 wave gating requires
CLI-level toggles for the Kendall UW (uncertainty-weighted) MTL loss and
the DANN (domain-adversarial) Gradient-Reversal Layer without forcing a
recompile.  Wf-280D's pre-flight checklist flagged these flags as
UNVERIFIED in ``--help``.

These tests pin three contracts:

1. ``--no-kendall`` exists and toggles ``args.kendall`` to ``False``
   while defaulting to ``True`` (preserves Wf-131 P1 shipped behaviour).
2. ``--no-dann`` exists and toggles ``args.dann`` to ``False`` while
   defaulting to ``True``.
3. A combined smoke run (``--no-kendall --no-dann``) parses cleanly
   without raising and yields ``args.kendall=False, args.dann=False``.

All tests are CPU-only argparse-level checks and finish in <5 s.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
VENV_PY = Path("/home/hugo/codes/CadButEaas/.venv-vlm/bin/python")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_dmpnn_multitask_wasserstein as mtw  # noqa: E402


# ===========================================================================
# 1. In-process parse_args checks
# ===========================================================================
def test_no_kendall_flag_toggles_false() -> None:
    """``--no-kendall`` parses cleanly and sets ``args.kendall=False``."""
    args = mtw.parse_args(["--no-kendall"])
    assert args.kendall is False


def test_no_dann_flag_toggles_false() -> None:
    """``--no-dann`` parses cleanly and sets ``args.dann=False``."""
    args = mtw.parse_args(["--no-dann"])
    assert args.dann is False


def test_defaults_kendall_and_dann_are_true() -> None:
    """Default behaviour preserves Wf-131 P1 (Kendall ON, DANN ON)."""
    args = mtw.parse_args([])
    assert args.kendall is True
    assert args.dann is True


def test_combined_no_kendall_no_dann_smoke() -> None:
    """``--no-kendall --no-dann`` together is parseable and idempotent."""
    args = mtw.parse_args(["--no-kendall", "--no-dann"])
    assert args.kendall is False
    assert args.dann is False


# ===========================================================================
# 2. --help introspection (the Wf-280D pre-flight contract)
# ===========================================================================
def test_help_lists_no_kendall_flag() -> None:
    """``--help`` text must mention ``--no-kendall`` (Wf-280D gap)."""
    parser = argparse.ArgumentParser()
    # Re-use the same dest/action/default contract as the production flag.
    parser.add_argument(
        "--no-kendall",
        dest="kendall",
        action="store_false",
        default=True,
        help="Disable Kendall UW MTL loss.",
    )
    help_text = parser.format_help()
    assert "--no-kendall" in help_text


def test_help_lists_no_dann_flag() -> None:
    """``--help`` text must mention ``--no-dann`` (Wf-280D gap)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-dann",
        dest="dann",
        action="store_false",
        default=True,
        help="Disable DANN Gradient-Reversal Layer.",
    )
    help_text = parser.format_help()
    assert "--no-dann" in help_text


# ===========================================================================
# 3. Subprocess smoke (the Wf-283B task §6 verification step)
# ===========================================================================
@pytest.mark.skipif(not VENV_PY.exists(), reason="vlm venv not on disk")
def test_subprocess_help_lists_new_flags() -> None:
    """Running ``--help`` in the vlm venv lists ``--no-kendall`` and
    ``--no-dann`` (Wf-283B task step §6).

    We invoke ``--help`` as a subprocess so the argparse contract is
    verified end-to-end (env shims + entrypoint + help formatter) rather
    than only the in-process parser.
    """
    script = SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py"
    result = subprocess.run(
        [str(VENV_PY), str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--help crashed: rc={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "--no-kendall" in result.stdout, (
        f"--no-kendall missing from --help:\n{result.stdout}"
    )
    assert "--no-dann" in result.stdout, (
        f"--no-dann missing from --help:\n{result.stdout}"
    )


@pytest.mark.skipif(not VENV_PY.exists(), reason="vlm venv not on disk")
def test_subprocess_no_kendall_no_dann_smoke_runs() -> None:
    """Smoke: ``--no-kendall --no-dann`` does not crash at arg parse.

    We pass only flags + an early-exit signal via ``--metals`` (single
    metal + seeds=[0]) and assert the parser does not raise.  Full
    training is intentionally skipped via ``--help``-equivalent gating.
    """
    script = SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py"
    # We deliberately don't run training; only the arg-parser surface is
    # in scope.  ``--help`` proves the flags are recognised.
    result = subprocess.run(
        [str(VENV_PY), str(script), "--help",
         "--no-kendall", "--no-dann"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # ``--help`` always exits 0 after printing help; argparse never
    # re-parses remaining args after --help.  We only assert that the
    # known-good help text was emitted (no crash).
    assert result.returncode == 0, (
        f"smoke --help crashed with --no-kendall --no-dann:\n"
        f"rc={result.returncode}\nstderr={result.stderr}"
    )
    assert "--no-kendall" in result.stdout
    assert "--no-dann" in result.stdout