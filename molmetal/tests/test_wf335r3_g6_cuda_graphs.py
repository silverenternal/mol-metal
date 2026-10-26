"""Wf-335-R3 — G6 launch path CUDAGraphs avoidance + safety-lever tests.

Owner: Hugo Lee
Date: 2026-09-29 (rewritten 2026-10-27)
Parent: TODO/pending/182_wf124_p3_per_innovation_ablation.md (TODO-182 G6)
Refs:
  - TODO/completed/330b4_regression_smoke.md §1, §4.3 (Wf-330-B4 MEASURED)
  - TODO/completed/335r1_*.md (Wf-335-R1 — torch.compile mode default flip)

Background — Wf-330-B4 MEASURED (2026-09-29): Ru seed=2 + Ir seed=0 failed
with ``RuntimeError: CUDAGraphs capture ...`` under
``--torch-compile-mode reduce-overhead``.

What this file used to assert, and why that was wrong
-----------------------------------------------------
The 2026-09-29 version of this file asserted three things that never existed
in the trainer:

  * a ``--no-cuda-graphs`` flag passed by six launchers,
  * ``args.cudagraphs_disabled``, and
  * ``args.torch_compile_mode`` / ``--torch-compile-mode``.

MEASURED 2026-10-27: ``parse_args(["--no-cuda-graphs"])`` exits 2 with
"unrecognized arguments", and neither ``cudagraphs_disabled`` nor
``torch_compile_mode`` was an argparse dest. Two of the six enumerated
launchers (``launch_wf124_p3_g4.sh``, ``launch_wf300_cfm_main.sh``) had been
deleted, and the two surviving G5/G6 launchers had already removed
``--no-cuda-graphs`` in Wf-r28 / Wf-R32 for exactly that reason. The file was
therefore red for nine assertions about a CLI surface nobody had built.

What the hazard actually was
----------------------------
``TORCH_COMPILE_MODES`` and ``DEFAULT_TORCH_COMPILE_MODE`` were declared at
module scope in ``train_dmpnn_multitask_wasserstein.py`` and documented as the
choices of ``--torch-compile-mode``, but no flag read them: the only live
compile path was ``--torch-compile``, which hardcoded
``mode="reduce-overhead"`` — the CUDA-Graphs mode behind Wf-330-B4. The "safe
default" existed only in two helper factories with no production caller. The
CFM trainer had the same hardcoded mode under a comment claiming the
"Wf-335-R3 Blackwell cudagraphs safety contract".

The flag itself is gone because ``af84a545`` (Wf-A39-1, the m21 selective
revert) restored this trainer from a pre-Wf-335-R1 snapshot with a whole-file
``git checkout``. That commit was about model performance, not about compile
modes; the flag disappeared as collateral, and ``test_wf327f5_torch_compile.py``
then pinned its absence. Re-adding a flag would re-litigate that revert, so the
repair is smaller: the call site now reads ``DEFAULT_TORCH_COMPILE_MODE`` and
the CFM trainer uses the same constant.

Honest framing (AGENTS.md §5.2):
  - MEASURED: Wf-330-B4 RuntimeError on Ru seed=2 + Ir seed=0.
  - MEASURED 2026-10-27: the three assertions above failed against the trainer.
  - MEASURED 2026-10-27: after the repair, ``parse_args([])`` and
    ``parse_args(["--torch-compile"])`` both leave the compile mode at
    ``max-autotune-no-cudagraphs``; no CLI switch can select a graph mode.
  - PROJECTED: future G6 / G5 launches will not reproduce the Wf-330-B4
    RuntimeError (no GPU run performed under the current GPU0 embargo).
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

SAFE_MODES = {"default", "max-autotune-no-cudagraphs"}


# ---------------------------------------------------------------------------
# Launchers that invoke train_dmpnn_multitask_wasserstein.py.
#
# The 2026-09-29 list was Wf-328-W2's six-entry enumeration. Two of those
# files no longer exist (g4 was closed by TODO-180; wf300 was renamed to
# wf299). Enumerating deleted paths makes the test red for a reason that has
# nothing to do with CUDA Graphs, so the list is the set that is on disk now.
# ---------------------------------------------------------------------------
LAUNCHERS_INVOKING_DMPNN = (
    SCRIPTS / "launch_wf124_p3_g5.sh",
    SCRIPTS / "launch_wf124_p3_g6.sh",
    SCRIPTS / "launch_wf310b_g6_final.sh",
    SCRIPTS / "launch_wf299_cfm_main.sh",
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


def _active_lines(src: str) -> list[str]:
    """Lines that are not comments (full-line or trailing ``#``).

    The launchers document the removed flags in banner comments; those must
    not count as live arguments. A trailing-comment strip is included because
    these files put the flag and its rationale on the same line.
    """
    out: list[str] = []
    for raw in src.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        code = line.split("#", 1)[0].rstrip()
        if code:
            out.append(code)
    return out


def _enables_flag(src: str, flag: str) -> list[str]:
    """Active lines that pass ``flag`` as a standalone token.

    ``--torch-compile`` is a prefix of ``--torch-compile-mode``, so a plain
    substring search would report the mode flag as the enabling switch.
    """
    pattern = re.compile(rf"(?<![\w-]){re.escape(flag)}(?![\w-])")
    return [line for line in _active_lines(src) if pattern.search(line)]


# ---------------------------------------------------------------------------
# Test 1 — the G6 launch path cannot reach a CUDA-Graphs compile
# ---------------------------------------------------------------------------
def test_g6_launcher_does_not_enable_torch_compile() -> None:
    """G6 launcher never passes ``--torch-compile``.

    CUDA Graphs only exist behind ``torch.compile`` in this trainer, and
    ``--torch-compile`` defaults OFF. The G6 path that triggered Wf-330-B4 is
    therefore graph-free by construction, not by a lever that has to be
    remembered. If a future edit turns compile on for G6, this test fails and
    the mode default (test 2) decides whether that is safe.
    """
    src = _read(G6_LAUNCHER)
    enabling = _enables_flag(src, "--torch-compile")
    assert not enabling, (
        f"{G6_LAUNCHER.name} enables --torch-compile:\n  "
        + "\n  ".join(enabling)
        + "\nWf-330-B4 (Ru seed=2 + Ir seed=0) died in CUDAGraphs capture on "
        "this launch path. Leave compile OFF here; if it is genuinely needed, "
        "pair it with --torch-compile-mode max-autotune-no-cudagraphs and say "
        "so in the launcher."
    )


# ---------------------------------------------------------------------------
# Test 2 — the trainer's compile path is CUDA-Graphs-free
# ---------------------------------------------------------------------------
def test_torch_compile_mode_default_is_safe() -> None:
    """``--torch-compile`` compiles with the safe mode, and only that mode.

    Asserts the contract Wf-335-R1 declared but did not implement: the call
    site reads ``DEFAULT_TORCH_COMPILE_MODE`` instead of a literal. There is
    deliberately no ``--torch-compile-mode`` flag (af84a545 removed the one
    Wf-335-R1 added, as collateral of a whole-file revert), so the property is
    checked at the source, where the literal used to be.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        DEFAULT_TORCH_COMPILE_MODE,
        parse_args,
    )

    assert DEFAULT_TORCH_COMPILE_MODE in SAFE_MODES, (
        f"DEFAULT_TORCH_COMPILE_MODE={DEFAULT_TORCH_COMPILE_MODE!r}; "
        f"must be one of {sorted(SAFE_MODES)} (CUDA Graphs disabled)."
    )

    src = _read(TEST_SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py")
    # The live call site is an assignment; the helper factories take a ``mode``
    # parameter and the --help text quotes the call, so neither matches.
    call_re = re.compile(
        r"^\s*model\s*=\s*torch\.compile\(model,\s*mode\s*=\s*([^)\s]+)\s*\)"
    )
    modes = [
        m.group(1)
        for m in (call_re.match(line) for line in _active_lines(src))
        if m
    ]
    assert modes, (
        "No live torch.compile(model, mode=...) assignment found; the call "
        "site moved — update this test rather than deleting it."
    )
    assert modes == ["DEFAULT_TORCH_COMPILE_MODE"], (
        "The live compile call site uses "
        f"{modes!r} instead of DEFAULT_TORCH_COMPILE_MODE. A literal there "
        "means --torch-compile can select a CUDA-Graphs mode (Wf-330-B4)."
    )

    args = parse_args([])
    assert args.torch_compile is False, (
        "parse_args([]) must leave torch.compile OFF; a default-on compile "
        "would put every run inside the Wf-330-B4 hazard."
    )


# ---------------------------------------------------------------------------
# Test 3 — no D-MPNN launcher turns compile on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "launcher_path",
    LAUNCHERS_INVOKING_DMPNN,
    ids=lambda p: p.name,
)
def test_launch_scripts_do_not_enable_torch_compile(launcher_path: Path) -> None:
    """Every launcher that invokes the D-MPNN trainer leaves compile OFF.

    This is the property the 2026-09-29 version was trying to protect with a
    ``--no-cuda-graphs`` flag: no shipped launch path reaches CUDA Graphs.
    Stated against the switch that actually exists, it is checkable.
    """
    src = _read(launcher_path)
    enabling = _enables_flag(src, "--torch-compile")
    assert not enabling, (
        f"{launcher_path.name} enables --torch-compile:\n  "
        + "\n  ".join(enabling)
        + "\nNo shipped launcher has ever enabled it. If that changes, the "
        "mode must be pinned safe in the same argv."
    )


# ---------------------------------------------------------------------------
# Test 4 — parse_args override contract
# ---------------------------------------------------------------------------
def test_no_cudagraphs_runtime_error() -> None:
    """Enabling compile does not change the mode, because there is no mode knob.

    Two cases:

      A. no flags            → compile off
      B. ``--torch-compile`` → compile on, and the call site's mode is the
         module constant

    Case B is the Wf-330-B4 gate. Until 2026-10-27 the call site hardcoded
    ``mode="reduce-overhead"``, so B silently meant "CUDA Graphs ON" no matter
    what any default said. A flag cannot override it now because there is no
    flag; the only way to reach a graph mode is to edit
    ``DEFAULT_TORCH_COMPILE_MODE``, which this test would catch.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        DEFAULT_TORCH_COMPILE_MODE,
        parse_args,
    )

    # A
    args_default = parse_args([])
    assert args_default.torch_compile is False

    # B
    args_on = parse_args(["--torch-compile"])
    assert args_on.torch_compile is True
    assert DEFAULT_TORCH_COMPILE_MODE in SAFE_MODES, (
        f"--torch-compile would compile with {DEFAULT_TORCH_COMPILE_MODE!r}; "
        f"that is a CUDA-Graphs mode. Wf-330-B4 (Ru seed=2 + Ir seed=0) died "
        f"in CUDAGraphs capture on this exact path."
    )
    # The flag is the only compile switch; nothing can select a mode at the
    # CLI. Pin that, because re-adding a mode flag is how the hazard returns.
    assert not hasattr(args_on, "torch_compile_mode"), (
        "--torch-compile-mode reappeared. It was removed by af84a545 and its "
        "return must be paired with a default of "
        f"{DEFAULT_TORCH_COMPILE_MODE!r} and a test update, not silently."
    )


# ---------------------------------------------------------------------------
# Test 5 — the CFM trainer carries the same safe mode
# ---------------------------------------------------------------------------
def test_cfm_trainer_compile_path_is_graph_free() -> None:
    """The CFM trainer's compile path does not hardcode a CUDA-Graphs mode.

    Until 2026-10-27 it did — ``torch.compile(model, mode="reduce-overhead")``
    under a comment that claimed the Wf-335-R3 safety contract. Both trainers
    now read ``DEFAULT_TORCH_COMPILE_MODE``.
    """
    cfm_path = TEST_SCRIPTS_DIR / "train_cfm_yuelbond_wasserstein.py"
    src = _read(cfm_path)

    const = re.search(
        r'^DEFAULT_TORCH_COMPILE_MODE\s*=\s*["\']([^"\']+)["\']',
        src,
        re.MULTILINE,
    )
    assert const, (
        "train_cfm_yuelbond_wasserstein.py has no DEFAULT_TORCH_COMPILE_MODE "
        "constant; the compile call site is unconstrained again."
    )
    assert const.group(1) in SAFE_MODES, (
        f"CFM DEFAULT_TORCH_COMPILE_MODE={const.group(1)!r}; must be one of "
        f"{sorted(SAFE_MODES)}."
    )

    hardcoded = [
        line
        for line in _active_lines(src)
        if re.search(r"torch\.compile\(.*mode\s*=\s*[\"']", line)
    ]
    assert not hardcoded, (
        "CFM compile call passes a literal mode instead of "
        "DEFAULT_TORCH_COMPILE_MODE:\n  " + "\n  ".join(hardcoded)
    )
