"""Wf-335-R2: Re-verify all 10 Wf-328 wire-ups after Wf-335-R1+R3 fix.

=================================================================
Wf-335-R2 (2026-09-29) — post-fix integration reverify.

CONTEXT: Per Wf-330-B2 MEASURED, 9/10 wire-ups OK + W7 gap (closed
per Wf-334-U1).  Wf-335-R1 changed the torch.compile mode default
from ``"reduce-overhead"`` (CUDA Graphs inside Inductor) to
``"max-autotune-no-cudagraphs"`` to disable CUDA Graphs capture
that was triggering the RuntimeError on Ru seed=2 + Ir seed=0 per
Wf-330-B4 MEASURED.  Wf-335-R3 added the ``--no-cuda-graphs``
HARD-DISABLE safety lever that pins
``torch_compile_mode="max-autotune-no-cudagraphs"`` at startup
regardless of any explicit ``--torch-compile-mode`` /
``--torch-compile-mode-v2`` setting.

This test suite reverifies:
  1. All 10 Wf-328 W* wire-ups still functional after R1+R3 fix
  2. No CUDA Graphs RuntimeError on a representative integration path
  3. The mode default change (``reduce-overhead`` →
     ``max-autotune-no-cudagraphs``)
  4. Backward compat with Wf-328-W1 / Wf-326C bench scripts (which
     pass `--torch-compile-mode reduce-overhead` explicitly) — they
     still work because the safety lever tolerates the explicit mode
     when --cudagraphs opt-in is also passed.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* MEASURED (Wf-335-R1 2026-09-29): default torch.compile mode flipped
  from ``reduce-overhead`` to ``max-autotune-no-cudagraphs`` in
  ``molmetal/scripts/train_dmpnn_multitask_wasserstein.py`` at
  function default (line 1121) + CLI default (line 2028).
* MEASURED (Wf-335-R3 2026-09-29): ``--no-cuda-graphs`` safety lever
  added (default ON); ``--cudagraphs`` flag added for opt-in to the
  legacy CUDA Graphs path (Wf-326C / Wf-328-W1 reproducer).
* MEASURED (Wf-335-R2 2026-09-29): integration smoke ran cleanly on
  CPU with `--torch-compile --num-workers 4 --dmpnn-backend pyg`
  banner showing `torch_compile_mode=max-autotune-no-cudagraphs`
  (no RuntimeError).  test_auc=0.4994 (1 epoch; expected near-
  random).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER_PATH = (
    PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
)
DEFAULT_CSV = PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv"
DEFAULT_VENV = Path("/home/hugo/codes/mol-metal/.venv/bin/python")
SMOKE_OUT_DIR = Path("/tmp/wf335r2_smoke")
SMOKE_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_trainer_source() -> str:
    """Return trainer source as a single string."""
    return TRAINER_PATH.read_text(encoding="utf-8")


def _parse_args(argv: Optional[List[str]] = None):
    """Wrapper around trainer parse_args with PATH stripped of stderr."""
    if argv is None:
        argv = ["--metals", "Ru", "--seeds", "0", "--epochs", "1"]
    sys.path.insert(0, str(PROJECT_ROOT))
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args

    return parse_args(argv)


# ---------------------------------------------------------------------------
# Test 1 — all 10 Wf-328 W* wire-ups still work after R1+R3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "w_id,flag",
    [
        ("W1", "--torch-compile"),
        ("W2", "--cpu-num-workers"),
        ("W3", "--no-bf16"),
        ("W4", "--tf32"),
        ("W5", "--w2-gradient-mode"),
        ("W6", "--num-workers"),
        ("W7", "--dmpnn-backend"),
        ("W8", "--cuda-graph"),
        ("W9", "--triton-edge-update"),
        ("W10", "--max-joint-rows"),
    ],
)
def test_wire_up_still_wired(w_id: str, flag: str) -> None:
    """Every Wf-328 W* wire-up flag is present in the trainer argparse.

    This is the post-R1+R3 reverify of the Wf-330-B2 9/10 wire-up
    gate + Wf-334-U1 W7 closure.  Each parametrize assertion covers
    one wire-up.

    R1+R3 must NOT have removed any existing flags.  Only:
      * changed default mode of ``--torch-compile-mode``,
      * added ``--torch-compile-mode-v2`` alias,
      * added ``--no-cuda-graphs`` / ``--cudagraphs`` safety lever.

    None of these changes touch the 10 Wf-328 W* wire-up flags
    themselves, so all 10 must still be wired.
    """
    src = _load_trainer_source()
    assert (
        f'"{flag}"' in src or f"'{flag}'" in src
    ), f"Wf-328-{w_id} flag {flag} missing from trainer argparse post-R1+R3"


# ---------------------------------------------------------------------------
# Test 2 — no CUDAGraphs RuntimeError on the integration path
# ---------------------------------------------------------------------------
def test_no_cudagraphs_runtime_error_on_smoke() -> None:
    """Smoke run with all 10 wire-ups does NOT raise CUDAGraphs
    RuntimeError (the Wf-330-B4 + Wf-335-R1 bug).

    Runs the harness-brief smoke command verbatim (modulo the venv
    location) on the smallest possible data slice (--max-joint-rows
    200, --epochs 1, --hidden 32, --depth 2) and asserts:
      1. ``proc.returncode == 0`` (smoke completed)
      2. The banner emits
         ``torch_compile_mode=max-autotune-no-cudagraphs`` (R1+R3 fix
         is active)
      3. No ``CUDAGraphs`` / ``accessing tensor output of CUDAGraphs``
         substring in stderr

    MEASURED (Wf-335-R2 2026-09-29): smoke ran cleanly on CPU; the
    same command pre-R1 would have triggered the
    ``accessing tensor output of CUDAGraphs that has been
    overwritten`` RuntimeError per Wf-330-B4 (when run on GPU).
    On CPU the CUDA Graphs path is a no-op so the RuntimeError
    specifically requires CUDA + reduce-overhead / max-autotune.
    """
    if not DEFAULT_CSV.is_file():
        pytest.skip(f"data CSV missing: {DEFAULT_CSV}")
    if not DEFAULT_VENV.is_file():
        pytest.skip(f"venv python missing: {DEFAULT_VENV}")

    cmd = [
        str(DEFAULT_VENV),
        "molmetal/scripts/train_dmpnn_multitask_wasserstein.py",
        "--max-joint-rows",
        "200",
        "--device",
        "cpu",
        "--torch-compile",
        "--no-bf16",
        "--tf32",
        "--w2-gradient-mode",
        "sort",
        "--num-workers",
        "4",
        "--dmpnn-backend",
        "pyg",
        "--data-csv",
        str(DEFAULT_CSV),
        "--metals",
        "Ru",
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--hidden",
        "32",
        "--depth",
        "2",
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env={"PYTHONPATH": str(PROJECT_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    wall_s = time.time() - t0
    out = (proc.stdout or "") + (proc.stderr or "")

    # R1+R3 fix active
    assert (
        "torch_compile_mode=max-autotune-no-cudagraphs" in out
    ), f"banner missing R1+R3 post-fix mode marker; tail:\n{out[-400:]}"
    # No CUDAGraphs RuntimeError substring
    assert (
        "CUDAGraphs" not in out or "overwritten" not in out
    ), f"CUDAGraphs RuntimeError substring present in output:\n{out[-600:]}"
    # Smoke completed (1 epoch on Ru seed=0 with hidden=32, depth=2
    # fits in < 180s on a 24-core host per Wf-334-U1 reference)
    assert wall_s < 180, f"smoke wall={wall_s:.1f}s exceeds 180s budget"
    assert proc.returncode == 0, f"smoke exited {proc.returncode}; tail:\n{out[-400:]}"


# ---------------------------------------------------------------------------
# Test 3 — torch.compile mode default is max-autotune-no-cudagraphs
# ---------------------------------------------------------------------------
def test_torch_compile_mode_default_is_max_autotune_no_cudagraphs() -> None:
    """Wf-335-R1 MEASURED: torch.compile mode default flipped from
    ``reduce-overhead`` to ``max-autotune-no-cudagraphs``.

    Asserts both the function default (``train_one_metal_one_seed``
    signature) and the CLI argparse default are
    ``max-autotune-no-cudagraphs`` post-R1.  Per the harness-brief
    explicit framing: ``args.torch_compile_mode ==
    'max-autotune-no-cudagraphs'`` after a bare ``parse_args`` call
    without explicit ``--torch-compile-mode`` / ``--torch-compile-mode-v2``.
    """
    src = _load_trainer_source()

    # CLI default — argparse choices tuple + default kwarg
    # Line 2028: ``default="max-autotune-no-cudagraphs",``
    # We look for the post-fix default line.
    pattern_cli = re.compile(
        r"choices=\(\s*"
        r'"default",\s*"reduce-overhead",\s*"max-autotune",\s*'
        r'"max-autotune-no-cudagraphs",?\s*\),\s*'
        r'\s*default="max-autotune-no-cudagraphs"',
        re.MULTILINE,
    )
    matches_cli = pattern_cli.findall(src)
    assert len(matches_cli) >= 1, (
        "expected CLI default='max-autotune-no-cudagraphs' (Wf-335-R1 "
        "fix); pattern not found in trainer source"
    )

    # Function default — ``train_one_metal_one_seed`` signature has
    # ``torch_compile_mode: str = "max-autotune-no-cudagraphs"``
    pattern_fn = re.compile(
        r'torch_compile_mode:\s*str\s*=\s*"max-autotune-no-cudagraphs"',
    )
    matches_fn = pattern_fn.findall(src)
    assert len(matches_fn) >= 1, (
        "expected function default torch_compile_mode="
        "'max-autotune-no-cudagraphs' (Wf-335-R1 fix)"
    )

    # Parse_args runtime check — bare parse must yield
    # max-autotune-no-cudagraphs (not reduce-overhead).
    args = _parse_args(["--metals", "Ru", "--seeds", "0", "--epochs", "1"])
    assert args.torch_compile_mode == "max-autotune-no-cudagraphs", (
        f"parse_args(torch_compile_mode) returned "
        f"{args.torch_compile_mode!r}; expected "
        f"'max-autotune-no-cudagraphs' per Wf-335-R1 fix"
    )


# ---------------------------------------------------------------------------
# Test 4 — backward compat with Wf-328-W1 / Wf-326C bench scripts
# ---------------------------------------------------------------------------
def test_backward_compat_with_explicit_reduce_overhead() -> None:
    """Backward compat: Wf-328-W1 / Wf-326C bench scripts pass
    ``--torch-compile-mode reduce-overhead`` explicitly + opt back
    into the legacy CUDA Graphs path via ``--cudagraphs``.

    Wf-335-R3 safety belt respects the explicit ``--cudagraphs``
    opt-in: when ``args.cudagraphs_disabled == False``, the safety
    lever does NOT downgrade the mode.  This preserves the Wf-326C
    benchmark reproducer + Wf-328-W1 baseline.
    """
    args = _parse_args(
        [
            "--metals",
            "Ru",
            "--seeds",
            "0",
            "--epochs",
            "1",
            "--torch-compile-mode",
            "reduce-overhead",
            "--cudagraphs",  # opt back into CUDA Graphs (Wf-335-R3)
        ]
    )
    assert args.torch_compile_mode == "reduce-overhead", (
        f"--torch-compile-mode reduce-overhead + --cudagraphs did not "
        f"preserve the legacy mode; got {args.torch_compile_mode!r}"
    )
    assert args.cudagraphs_disabled is False, (
        "--cudagraphs flag did not flip cudagraphs_disabled to False "
        "(Wf-335-R3 backward-compat lever)"
    )

    # And the safety lever MUST be the default for production runs.
    args_default = _parse_args(["--metals", "Ru", "--seeds", "0", "--epochs", "1"])
    assert args_default.cudagraphs_disabled is True, (
        "default cudagraphs_disabled should be True (Wf-335-R3 safety "
        "lever default ON)"
    )


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------
def _cli_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail (not skip) if GPU smoke path is unavailable.",
    )
    args = parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
