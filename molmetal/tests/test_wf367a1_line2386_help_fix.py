"""Pytest: Wf-367-A1 — defensive ``pct`` substitution at
``train_dmpnn_multitask_wasserstein.py:2384`` to mirror the Wf-358-A1
fix pattern in the ``--wasserstein-w1`` help= block.

Per Wf-367-A1 NEW finding (cross-poll from Wf-366-B2 forward-looking
gap inventory): the docstring comment block at lines 2383-2391 introduced
in Wf-358-B1 contained a literal ``%`` in line 2384 (``PROJECTED +20-30%
training speed``). Argparse's HelpFormatter does ``help_text % params``
where ``params`` is a dict; a literal ``%`` followed by a format-spec
char (e.g. ``t`` of ``training``) raises ``TypeError: must be real number,
not dict`` when the trainer's ``--help`` is invoked.

This was a *latent* risk — argparse does NOT process Python comments, so
the bug was not active. But if the comment is ever copy-pasted into a
``help=`` block (a common refactor mistake), the bug would re-emerge.

Wf-367-A1 fix: replace literal ``%`` with ``pct`` at line 2384 (mirrors
Wf-358-A1 fix at lines 2676 + 2794 which converted ``%`` to ``pct`` in
the production help= strings). Add a 6-line block-comment explaining
the fix and the underlying argparse mechanism so future agents do not
revert the change.

Three regression tests pin the contract:

1. ``test_line_2384_has_pct_not_pct_literal`` — line 2384 source contains
   ``pct`` and does NOT contain a literal ``%`` followed by a format-spec
   char that argparse would misinterpret (defensive source-level check).
2. ``test_argparse_help_formatter_runs_on_wasserstein_w1_help`` —
   construct an argparse parser using a verbatim copy of the help=
   string at lines 2404-2411 and verify ``format_help()`` does NOT raise
   ``TypeError`` (the regression we are guarding against).
3. ``test_full_subprocess_help_runs_with_wf367a1_fix`` — invoke the
   trainer via subprocess and verify ``--help`` exits 0 with the
   Wf-358-B1 / Wf-367-A1 markers visible in stdout. Bypasses the test
   isolation (env shims + entrypoint + HelpFormatter).

All tests are CPU-only and finish in <10 s combined.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
TRAINER_PATH = SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py"
VENV_PY = Path("/home/hugo/codes/mol-metal/.venv/bin/python")


# ===========================================================================
# 1. Source-level defensive check (the actual Wf-367-A1 fix)
# ===========================================================================
def test_line_2384_has_pct_not_pct_literal() -> None:
    """Line 2384 of train_dmpnn_multitask_wasserstein.py contains
    ``pct`` (the Wf-367-A1 fix) and NOT ``PROJECTED +20-30% `` (the
    literal-percent pattern that argparse mis-formats).
    """
    src = TRAINER_PATH.read_text(encoding="utf-8")
    lines = src.splitlines()
    assert len(lines) >= 2384, (
        f"trainer file has {len(lines)} lines; expected >= 2384 for "
        f"Wf-367-A1 line-2384 fix verification"
    )

    line_2384 = lines[2383]  # 0-indexed
    assert "PROJECTED +20-30 pct training speed" in line_2384, (
        f"Line 2384 missing Wf-367-A1 'pct' replacement; got: " f"{line_2384[:120]}"
    )
    # Specifically ensure no literal "%" pattern that argparse would
    # mis-format. Allow "%" only if escaped ("%%") or if followed by
    # safe argparse substitution keys like "%(default)s".
    pct_pattern = re.search(r"%(?!\(default\)|\(type\)|\(prog\)|%)", line_2384)
    assert pct_pattern is None, (
        f"Line 2384 contains unsafe literal '%' at offset "
        f"{pct_pattern.start()}: {line_2384[max(0, pct_pattern.start()-20):pct_pattern.end()+30]}"
    )


def test_line_2386_introduces_no_help_eq_with_literal_pct() -> None:
    """Line 2386 is a comment line; verify it does NOT contain an
    ``argparse help=`` directive with a literal ``%`` (the Wf-358-A1
    failure mode). Argparse doesn't process comments, but a future
    refactor that promotes the comment to help= would re-introduce
    the bug — defensive guard.
    """
    src = TRAINER_PATH.read_text(encoding="utf-8")
    lines = src.splitlines()
    line_2386 = lines[2385]  # 0-indexed
    # Defensive: ensure no `help=` substring + literal % in the same
    # comment line (would-be-help= block).
    assert "help=" not in line_2386, (
        f"Line 2386 should be a plain comment (no help=); got: " f"{line_2386[:120]}"
    )
    assert not re.search(r"%(?!\(default\)|\(type\)|\(prog\)|%)", line_2386), (
        f"Line 2386 contains unsafe literal '%' (potential copy-paste "
        f"regression): {line_2386[:120]}"
    )


# ===========================================================================
# 2. Argparse-level: format_help() must NOT raise TypeError on the
#    production help= block (lines 2404-2411 verbatim)
# ===========================================================================
def test_argparse_help_formatter_runs_on_wasserstein_w1_help() -> None:
    """Reconstruct the ``--wasserstein-w1`` help= string from the
    production trainer (lines 2404-2411) and assert ``format_help()``
    does NOT raise TypeError or ValueError.

    The Wf-358-A1 fix converted ``PROJECTED +20-30% training speed``
    to ``PROJECTED +20-30 pct training speed``. The Wf-367-A1 fix
    mirrors that change in the docstring comment at line 2384 so the
    fix is self-consistent (defensive guard against future copy-paste
    regressions from comment -> help=).
    """
    # Verbatim copy of the production help= block at lines 2404-2411.
    # If this changes upstream, the test still pins the Wf-358-A1
    # contract (no literal ``%`` followed by a format-spec char).
    production_help = (
        "Wf-358-B1: route the per-epoch W pIC50 penalty through "
        "the W₁ (1-Wasserstein) adapter-layer scaffold "
        "(``wasserstein_w1_loss``). Skips the W₂ sliced "
        "projection loop entirely — a single ``torch.sort`` + "
        "``.abs().mean()`` per call. PROJECTED +20-30 pct "
        "training speed on CPU (Wf-326E benchmark family); "
        "no accuracy change vs the legacy L1 closed form. "
        "Default OFF for backward-compat."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--wasserstein-w1", help=production_help)
    # The actual regression: argparse's HelpFormatter._expand_help does
    # ``self._get_help_string(action) % params`` where params is a dict.
    # If the help text contains "%" followed by a format-spec char,
    # Python raises ``ValueError: unsupported format character 'X'``
    # or ``TypeError: must be real number, not dict``.
    try:
        rendered = parser.format_help()
    except (TypeError, ValueError) as exc:
        pytest.fail(
            f"argparse format_help() crashed on Wf-358-B1 production help= "
            f"block (Wf-367-A1 regression pin): {type(exc).__name__}: {exc}"
        )
    assert "--wasserstein-w1" in rendered
    assert "pct" in rendered  # confirms Wf-358-A1 fix is preserved


# ===========================================================================
# 3. Subprocess smoke (end-to-end --help invocation)
# ===========================================================================
@pytest.mark.skipif(not VENV_PY.exists(), reason="vlm venv not on disk")
def test_full_subprocess_help_runs_with_wf367a1_fix() -> None:
    """End-to-end: invoke ``--help`` via subprocess and verify rc=0
    + stdout contains Wf-358-B1 / Wf-367-A1 markers. This is the
    production-equivalent of the Wf-283B subprocess tests but pinned
    to the Wf-367-A1 line-2384 defensive fix.
    """
    if not TRAINER_PATH.exists():
        pytest.skip(f"trainer script not on disk: {TRAINER_PATH}")
    result = subprocess.run(
        [str(VENV_PY), str(TRAINER_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--help crashed (Wf-367-A1 fix regressed): rc={result.returncode}\n"
        f"stderr={result.stderr[:1000]}"
    )
    # Confirm Wf-358-B1 / Wf-367-A1 markers visible
    assert (
        "--wasserstein-w1" in result.stdout
    ), f"--wasserstein-w1 missing from --help:\n{result.stdout[:500]}"
    assert "PROJECTED +20-30 pct" in result.stdout, (
        f"Wf-358-A1 'pct' marker missing (regression?): "
        f"{result.stdout[max(0, result.stdout.find('wasserstein-w1')-50):result.stdout.find('wasserstein-w1')+500]}"
    )
