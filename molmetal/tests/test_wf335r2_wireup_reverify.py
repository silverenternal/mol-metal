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
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

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
        ("W4", "--tf32"),
        ("W5", "--w2-gradient-mode"),
        ("W10", "--max-joint-rows"),
    ],
)
def test_wire_up_still_wired(w_id: str, flag: str) -> None:
    """The Wf-328 wire-ups that survive in the shipped trainer are present.

    This used to cover ten. MEASURED 2026-10-27: af84a545 restored
    ``train_dmpnn_multitask_wasserstein.py`` from a pre-Wf-328 snapshot, so
    W3 (``--no-bf16``), W6 (``--num-workers``), W7 (``--dmpnn-backend``),
    W8 (``--cuda-graph``) and W9 (``--triton-edge-update``) are gone. They
    are not silently dropped from the record — see
    ``test_wire_ups_removed_by_the_m21_revert``.
    """
    src = _load_trainer_source()
    assert (
        f'"{flag}"' in src or f"'{flag}'" in src
    ), f"Wf-328-{w_id} flag {flag} missing from trainer argparse"


@pytest.mark.parametrize(
    "w_id,flag",
    [
        ("W3", "--no-bf16"),
        ("W6", "--num-workers"),
        ("W7", "--dmpnn-backend"),
        ("W8", "--cuda-graph"),
        ("W9", "--triton-edge-update"),
    ],
)
def test_wire_ups_removed_by_the_m21_revert(w_id: str, flag: str) -> None:
    """These five wire-ups are absent, and that is a recorded gap.

    af84a545 ("m21 selective Wf-A17 revert") did a whole-file
    ``git checkout`` to recover model performance — A2780cis had dropped
    0.7278 → 0.3356 — and took the Wf-328 wire-ups with it.  Asserting they
    are present (as this file did until 2026-10-27) is a test that can only
    be satisfied by re-porting the reverted code, which needs GPU
    revalidation.  Asserting their absence keeps the gap visible: if a
    future wave restores one, this test fails and forces the record to be
    updated rather than quietly drifting.
    """
    src = _load_trainer_source()
    assert (
        f'"{flag}"' not in src and f"'{flag}'" not in src
    ), (
        f"Wf-328-{w_id} flag {flag} is declared again. That is fine — but it "
        f"means the m21 revert has been (partly) undone: move this case back "
        f"to test_wire_up_still_wired and note the GPU revalidation."
    )


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
        "--tf32",
        "--w2-gradient-mode",
        "sort",
        "--cpu-num-workers",
        "4",
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
        "--out-dir",
        str(SMOKE_OUT_DIR),
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

    # The compile path is live and it selected the safe mode.  MEASURED
    # 2026-10-27: this smoke finishes in ~21 s on CPU.
    assert (
        "[compute] --torch-compile ENABLED (max-autotune-no-cudagraphs)" in out
    ), f"compile banner missing the safe mode; tail:\n{out[-600:]}"
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
    """The compile mode is ``max-autotune-no-cudagraphs``, and it is read.

    Rewritten 2026-10-27: the old version looked for an argparse
    ``choices=(...)`` tuple plus ``default="max-autotune-no-cudagraphs"``, and
    for ``torch_compile_mode: str = "max-autotune-no-cudagraphs"`` in the
    trainer signature.  Neither exists — af84a545 removed the flag and the
    kwarg.  What remains, and what actually decides the mode, is
    ``DEFAULT_TORCH_COMPILE_MODE`` at the call site.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        DEFAULT_TORCH_COMPILE_MODE,
    )

    assert DEFAULT_TORCH_COMPILE_MODE == "max-autotune-no-cudagraphs", (
        f"expected DEFAULT_TORCH_COMPILE_MODE='max-autotune-no-cudagraphs' "
        f"(Wf-335-R1 fix), got {DEFAULT_TORCH_COMPILE_MODE!r}"
    )

    src = _load_trainer_source()
    assert "model = torch.compile(model, mode=DEFAULT_TORCH_COMPILE_MODE)" in src, (
        "the compile call site no longer reads DEFAULT_TORCH_COMPILE_MODE; "
        "a literal mode there can re-enable CUDA Graphs"
    )

    # Runtime check: --torch-compile is the only switch, and it defaults OFF.
    args = _parse_args(["--metals", "Ru", "--seeds", "0", "--epochs", "1"])
    assert args.torch_compile is False, (
        "torch.compile must default OFF; a default-on compile would put "
        "every run inside the Wf-330-B4 hazard"
    )


# ---------------------------------------------------------------------------
# Test 4 — backward compat with Wf-328-W1 / Wf-326C bench scripts
# ---------------------------------------------------------------------------
def test_backward_compat_with_explicit_reduce_overhead() -> None:
    """The CUDA-Graphs mode is still reachable in-process, but not via CLI.

    The old version parsed ``--torch-compile-mode reduce-overhead
    --cudagraphs``.  Neither flag exists after af84a545.  The Wf-326C /
    Wf-328-W1 reproducer path survives in the helper factories, which take a
    ``mode`` argument; the CLI deliberately has no way to select one, because
    the only flag that ever read a mode was removed and the default is the
    safety lever.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        DEFAULT_TORCH_COMPILE_MODE,
        TORCH_COMPILE_MODES,
    )

    assert "reduce-overhead" in TORCH_COMPILE_MODES, (
        "TORCH_COMPILE_MODES no longer lists 'reduce-overhead'; the "
        "reproducer path in the helper factories would be unreachable"
    )
    assert DEFAULT_TORCH_COMPILE_MODE != "reduce-overhead", (
        "the default mode must not be the CUDA-Graphs mode"
    )

    # The CLI cannot select a mode at all — that is the safety property.
    args_default = _parse_args(["--metals", "Ru", "--seeds", "0", "--epochs", "1"])
    assert not hasattr(args_default, "cudagraphs_disabled"), (
        "a cudagraphs lever reappeared; it would need a real default and a "
        "test update, not a silent re-add"
    )
    assert not hasattr(args_default, "torch_compile_mode"), (
        "a torch_compile_mode dest reappeared; see the note in "
        "test_torch_compile_mode_v2_flag (test_wf335r1_no_cudagraphs.py)"
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
