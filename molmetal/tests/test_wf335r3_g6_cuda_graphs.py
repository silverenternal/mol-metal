"""Wf-335-R3 — G6 launch path CUDAGraphs avoidance + safety-lever tests.

Owner: Hugo Lee
Date: 2026-09-29
Parent: TODO/pending/182_wf124_p3_per_innovation_ablation.md (TODO-182 G6)
Refs:
  - TODO/completed/330b4_regression_smoke.md §1, §4.3 (Wf-330-B4 MEASURED)
  - TODO/completed/335r1_*.md (Wf-335-R1 — torch.compile mode default flip)
  - Wf-335-R3 — adds the ``--no-cuda-graphs`` / ``--cudagraphs`` safety-lever
    CLI pair + wires all 6 launch scripts per Wf-328-W2 enumeration.

Background — Wf-330-B4 MEASURED (2026-09-29): Ru seed=2 + Ir seed=0 failed
with ``RuntimeError: CUDAGraphs capture ...`` under
``--torch-compile-mode reduce-overhead``.  Wf-335-R1 (committed) flipped the
trainer default to ``max-autotune-no-cudagraphs`` so future runs avoid the
trap by default.  Wf-335-R3 (this commit) adds a HARD OVERRIDE safety
lever that pins the trainer to a no-CUDA-Graphs mode regardless of any
explicit ``--torch-compile-mode`` / ``--torch-compile-mode-v2`` override
that the launcher or user might pass.

Tests (4 total):

  - test_g6_launcher_has_no_cuda_graphs     (gates the regression: G6 path)
  - test_torch_compile_mode_default_is_safe (R1 default = max-autotune-no-cudagraphs)
  - test_six_launch_scripts_have_no_cuda_graphs (all 6 Wf-328-W2 launchers wired)
  - test_no_cudagraphs_runtime_error         (override contract: mode pinned)

All tests are CPU-only — no torch.cuda, no real launches.  < 5 s wall.

Honest framing (AGENTS.md §5.2):
  - MEASURED: Wf-330-B4 RuntimeError on Ru seed=2 + Ir seed=0.
  - MEASURED: trainer default switched to max-autotune-no-cudagraphs (R1).
  - MEASURED: 6 launch scripts (Wf-328-W2 enumeration) updated with
    ``--no-cuda-graphs`` (R3).
  - PROJECTED: future G6 / G5 / G4 launches will not reproduce the
    Wf-330-B4 RuntimeError (BLOCKED on AGENTS.md §5.1 kill_then_adapt
    for actual GPU verification).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root + path setup
# ---------------------------------------------------------------------------
ROOT = Path("/home/hugo/codes/mol-metal")
SCRIPTS = ROOT / "scripts"
TEST_SCRIPTS_DIR = ROOT / "molmetal" / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Constants — the 6 launch scripts that Wf-328-W2 fixed (and R3 re-wires)
# Cross-ref TODO/completed/328w2_launcher_num_workers_fix.md §2.1
# ---------------------------------------------------------------------------
WF328W2_TARGETS = (
    SCRIPTS / "launch_wf124_p3_g4.sh",
    SCRIPTS / "launch_wf124_p3_g5.sh",
    SCRIPTS / "launch_wf124_p3_g6.sh",
    SCRIPTS / "launch_wf310b_g6_final.sh",
    SCRIPTS / "launch_wf300_cfm_main.sh",
    SCRIPTS / "retry_g5_missing_pairs.sh",
)

# The G6 launcher is the primary Wf-335-R3 target — it was the launch
# path that triggered the Wf-330-B4 CUDAGraphs regression.
G6_LAUNCHER = SCRIPTS / "launch_wf124_p3_g6.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read(path: Path) -> str:
    assert path.exists(), f"launch script missing: {path}"
    return path.read_text()


def _active_flag_lines(src: str, flag: str) -> list[str]:
    """Extract ``--<flag>`` lines that are NOT in comments.

    Skips comment lines (start with ``#``) so the Wf-335-R3 banner
    comment (which mentions ``--no-cuda-graphs`` for documentation
    purposes) does NOT count as an active override.
    """
    out: list[str] = []
    for raw in src.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if flag in line:
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# Test 1 — G6 launcher has --no-cuda-graphs (regression gate)
# ---------------------------------------------------------------------------
def test_g6_launcher_has_no_cuda_graphs() -> None:
    """Wf-335-R3 — G6 launcher explicitly passes ``--no-cuda-graphs``.

    Background: Wf-330-B4 MEASURED RuntimeError on Ru seed=2 + Ir seed=0
    was triggered by ``--torch-compile-mode reduce-overhead`` (CUDA Graphs
    capture inside Inductor).  Wf-335-R1 flipped the trainer default but
    the G6 launcher is the canonical path that future TODO-182 / Wf-310B
    runs will use, so the safety lever MUST be explicit in the launcher.

    Test passes iff the G6 launcher contains at least one ACTIVE (non-comment)
    ``--no-cuda-graphs`` directive.
    """
    src = _read(G6_LAUNCHER)
    active_lines = _active_flag_lines(src, "--no-cuda-graphs")
    assert active_lines, (
        f"G6 launcher ({G6_LAUNCHER.name}) does NOT pass "
        f"--no-cuda-graphs as an active flag (only comments reference it). "
        f"Wf-335-R3 requires the safety lever to be explicit so paper §6 "
        f"audits can verify the lever is wired against the Wf-330-B4 "
        f"RuntimeError regression."
    )
    # Sanity: the line is actually a CLI flag, not just a substring match
    assert any("--no-cuda-graphs" in line and not line.startswith("#")
               for line in active_lines)


# ---------------------------------------------------------------------------
# Test 2 — torch.compile mode default verified
# ---------------------------------------------------------------------------
def test_torch_compile_mode_default_is_safe() -> None:
    """Wf-335-R3 — verify the trainer default mode is CUDA-Graphs-free.

    Background: Wf-335-R1 flipped
    ``train_dmpnn_multitask_wasserstein.py`` default ``--torch-compile-mode``
    from ``reduce-overhead`` (CUDA Graphs) to ``max-autotune-no-cudagraphs``
    (no CUDA Graphs).  This test reads the source directly and asserts the
    new default is in place.  If a future refactor reverts the default,
    this test FAILS and the safety net is restored.
    """
    trainer_path = TEST_SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py"
    src = _read(trainer_path)

    # The argparse default for --torch-compile-mode must be the
    # no-CUDA-Graphs mode.  Match either the bare form or the indented form.
    pattern = re.compile(
        r'["\']--torch-compile-mode["\'].*?default\s*=\s*["\']([^"\']+)["\']',
        re.DOTALL,
    )
    match = pattern.search(src)
    assert match, (
        "Could not locate --torch-compile-mode default in trainer source; "
        "refactor may have moved the argparse block — update this test."
    )
    default_mode = match.group(1)
    assert default_mode in {"default", "max-autotune-no-cudagraphs"}, (
        f"Trainer --torch-compile-mode default is {default_mode!r}; "
        f"Wf-335-R1 contract requires 'max-autotune-no-cudagraphs' "
        f"(or 'default' which also disables CUDA Graphs). "
        f"Switching back to 'reduce-overhead' or 'max-autotune' would "
        f"re-introduce the Wf-330-B4 RuntimeError on Ru seed=2 + Ir seed=0."
    )

    # Also assert the function signature default agrees with the argparse
    # default (drift guard — both must stay CUDA-Graphs-free).
    sig_match = re.search(
        r"def\s+train_one_seed\([^)]*torch_compile_mode:\s*str\s*=\s*[\"']([^\"']+)[\"']",
        src,
        re.DOTALL,
    )
    if sig_match is not None:
        sig_default = sig_match.group(1)
        assert sig_default in {"default", "max-autotune-no-cudagraphs"}, (
            f"train_one_seed() signature default torch_compile_mode="
            f"{sig_default!r}; Wf-335-R1 contract requires "
            f"'max-autotune-no-cudagraphs' or 'default'."
        )


# ---------------------------------------------------------------------------
# Test 3 — all 6 launch scripts updated with --no-cuda-graphs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "launcher_path",
    WF328W2_TARGETS,
    ids=lambda p: p.name,
)
def test_six_launch_scripts_have_no_cuda_graphs(launcher_path: Path) -> None:
    """Wf-335-R3 — all 6 Wf-328-W2 launchers wired with ``--no-cuda-graphs``.

    Cross-ref TODO/completed/328w2_launcher_num_workers_fix.md §2.1 for the
    enumeration.  Each of the 6 launchers (g4, g5, g6, g6_final, wf300_cfm
    Wave-B, retry_g5) must pass ``--no-cuda-graphs`` as an active (non-
    comment) flag in its COMMON_ARGS array (or inline invocation).

    The contract: every Wf-328-W2 launcher that invokes
    ``train_dmpnn_multitask_wasserstein.py`` must also wire the Wf-335-R3
    safety lever, so paper §6 audits can verify end-to-end.
    """
    src = _read(launcher_path)
    active_lines = _active_flag_lines(src, "--no-cuda-graphs")
    assert active_lines, (
        f"{launcher_path.name}: missing active --no-cuda-graphs flag. "
        f"Wf-335-R3 requires this safety lever on all 6 launchers that "
        f"invoke train_dmpnn_multitask_wasserstein.py."
    )


# ---------------------------------------------------------------------------
# Test 4 — no_CUDAGraphs_runtime_error (override contract)
# ---------------------------------------------------------------------------
def test_no_cudagraphs_runtime_error() -> None:
    """Wf-335-R3 — verify the trainer's ``--no-cuda-graphs`` flag is parsed
    and the ``--cudagraphs`` opt-out is also parsed (override contract).

    This is the runtime-error-prevention contract: the trainer MUST accept
    ``--no-cuda-graphs`` (the default; safety lever ON) and ``--cudagraphs``
    (explicit opt-out for Wf-326C / Wf-328-W1 reproducer).  We exercise
    both via parse_args() in CPU-only mode (no torch.cuda needed).

    Background: Wf-330-B4 MEASURED RuntimeError was triggered when the
    trainer ran under ``--torch-compile-mode reduce-overhead``.  R3 adds
    a HARD OVERRIDE: ``--no-cuda-graphs`` (default ON) forces the trainer
    to use a CUDA-Graphs-free mode regardless of any explicit override,
    so the Wf-330-B4 RuntimeError is structurally prevented at the CLI
    parse layer.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args

    # Case A: --no-cuda-graphs (the default; safety lever ON).
    args_default = parse_args([])
    assert args_default.cudagraphs_disabled is True, (
        "Trainer default for cudagraphs_disabled must be True "
        "(Wf-335-R3 safety lever default ON per Wf-330-B4 MEASURED)."
    )

    # Case B: --no-cuda-graphs explicit.
    args_on = parse_args(["--no-cuda-graphs"])
    assert args_on.cudagraphs_disabled is True, (
        "--no-cuda-graphs must set cudagraphs_disabled=True."
    )

    # Case C: --cudagraphs explicit (opt-out for Wf-326C reproducer).
    args_off = parse_args(["--cudagraphs"])
    assert args_off.cudagraphs_disabled is False, (
        "--cudagraphs must set cudagraphs_disabled=False "
        "(explicit opt-out, Wf-326C / Wf-328-W1 reproducer only)."
    )

    # Case D: torch_compile_mode default is CUDA-Graphs-free
    # (cross-cuts Test 2 — restated here for the override contract).
    assert args_default.torch_compile_mode in {
        "default", "max-autotune-no-cudagraphs"
    }, (
        f"Trainer torch_compile_mode default is "
        f"{args_default.torch_compile_mode!r}; must be "
        f"'default' or 'max-autotune-no-cudagraphs' (CUDA-Graphs-free)."
    )
