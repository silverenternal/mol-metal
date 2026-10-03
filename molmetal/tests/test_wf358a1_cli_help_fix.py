"""Pytest: Wf-358-A1 — Regression tests for the ``+~30%`` argparse-help
``TypeError: must be real number, not dict`` bug fixed in
``train_dmpnn_multitask_wasserstein.py``.

Root cause
----------
argparse's ``HelpFormatter._expand_help`` (CPython 3.12) does::

    return self._get_help_string(action) % params

where ``params`` is a dict such as ``{'default': ..., 'type': ...,
'dest': ...}``.  If the help text contains a literal ``%`` followed by
any character that looks like a format spec (``30%``, ``5-15%``,
``100%``, etc.), Python attempts ``str % dict`` substitution and raises
``TypeError: must be real number, not dict`` because the format-spec
key is not in the dict.

This module ships two regression tests:

1. ``test_subprocess_help_exit_zero_with_new_flags`` — ``--help`` exits
   with ``returncode == 0`` AND stdout contains the Wf-350-C3 +
   Wf-358-A3 flag names that were inaccessible while the bug was live.
2. ``test_subprocess_help_does_not_raise_typeerror`` — ``--help`` does
   NOT raise ``TypeError: must be real number, not dict`` (the exact
   exception signature from the original failure mode).

Both tests are CPU-only subprocess invocations of ``--help`` and finish
in <5 s on RTX 5090 / Blackwell.

Source: Wf-358-A1 closeout + Wf-357-A1 root-cause finding.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
VENV_PY = Path("/home/hugo/codes/mol-metal/.venv/bin/python")


@pytest.mark.skipif(not VENV_PY.exists(), reason="vlm venv not on disk")
def test_subprocess_help_exit_zero_with_new_flags() -> None:
    """``--help`` exits 0 AND lists the Wf-350-C3 + Wf-358-A3 flags that
    were inaccessible while the bug was live.

    This is the primary contract: the trainer CLI is parseable end-to-end
    in the vlm venv (env shims + entrypoint + argparse formatter).
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
    assert "--gradient-checkpointing" in result.stdout, (
        f"--gradient-checkpointing missing from --help (Wf-350-C3 flag "
        f"should appear now that the +~30% literal is gone):\n"
        f"{result.stdout}"
    )
    assert "--torch-compile-dynamic" in result.stdout, (
        f"--torch-compile-dynamic missing from --help (Wf-358-A3 flag "
        f"should appear now that the 5-15% literal is gone):\n"
        f"{result.stdout}"
    )


@pytest.mark.skipif(not VENV_PY.exists(), reason="vlm venv not on disk")
def test_subprocess_help_does_not_raise_typeerror() -> None:
    """``--help`` must NOT raise ``TypeError: must be real number, not dict``.

    This pins the exact failure signature from the original Wf-357-A1
    bug report.  If a future commit re-introduces a literal ``%`` (e.g.
    ``+50%`` or ``99%``) inside any ``help=`` string, argparse will
    raise this exception when formatting the help block, and the
    subprocess will exit non-zero with the TypeError in stderr.
    """
    script = SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py"
    result = subprocess.run(
        [str(VENV_PY), str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # The script may legitimately exit non-zero from SystemExit or other
    # argparse paths; what matters is that the specific TypeError
    # signature from the Wf-357-A1 root cause is absent.
    assert "TypeError: must be real number, not dict" not in result.stderr, (
        f"argparse %-format regression re-introduced in --help:\n"
        f"rc={result.returncode}\nstderr={result.stderr}"
    )
    # Sanity: the script at least printed help text (i.e. argparse
    # reached the format_help stage without crashing).
    assert "usage:" in result.stdout or "--help" in result.stdout, (
        f"--help did not produce argparse usage banner:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )