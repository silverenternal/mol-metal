"""Wf-337-G2 — G6 launch wrapper v2 + G4/G5 parity + backward-compat tests.

Owner: Hugo Lee
Date: 2026-09-29
Parent: Wf-337 (post-Wf-335 + Wf-328 ultracode consolidation)
Refs:
  - TODO/completed/335r1_no_cudagraphs_fix.md (Wf-335-R1 — torch.compile mode flip)
  - TODO/completed/335r3_g6_no_cudagraphs.md (Wf-335-R3 — --no-cuda-graphs safety lever)
  - TODO/completed/328w2_launcher_num_workers_fix.md (Wf-328-W2 — --cpu-num-workers 8)
  - TODO/completed/328w5_w2_sort_wired.md (Wf-328-W5 — --w2-gradient-mode sort default)
  - TODO/completed/328w6_dataloader_factory_wired.md (Wf-328-W6 — --num-workers flag)

Background — Wf-337-G2 consolidates all Wf-328 + Wf-335 fixes into the
G4 / G5 / G6 launchers.  Each launcher must now carry:

  1. --cpu-num-workers 8       (Wf-328-W2)
  2. --num-workers 8            (Wf-328-W6 — factory override; prefetch=8)
  3. --w2-gradient-mode sort    (Wf-328-W5 — sort-based W2 default)
  4. --no-cuda-graphs           (Wf-335-R3 — CUDA Graphs hard-disable safety)
  5. --no-kendall / --no-dann   (Wf-296C — per-config ablation only; in G6
                                   CONFIRMED_FLAGS array, not COMMON_ARGS)

Tests (4 total):

  - test_g6_launcher_has_all_fixes  (primary v2 contract gate)
  - test_g5_launcher_parity         (G5 mirrors G6 v2 config)
  - test_g4_launcher_parity         (G4 mirrors G6 v2 config)
  - test_backward_compat            (sliced mode + cpu-num-workers fallback still work)

All tests are CPU-only — no torch.cuda, no real launches.  < 5 s wall.

Honest framing (AGENTS.md §5.2 + party-twenty-spirit §3.5):
  - MEASURED: trainer defaults are CUDA-Graphs-free (Wf-335-R1 commit)
  - MEASURED: trainer exposes --num-workers, --w2-gradient-mode, --no-kendall,
    --no-dann, --no-cuda-graphs CLI flags.
  - MEASURED: 3 launchers (g4/g5/g6) wired with all 4 fix flags.
  - PROJECTED: future G4/G5/G6 launches will benefit from the consolidated
    config (BLOCKED on AGENTS.md §5.1 kill_then_adapt for actual GPU bench).
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
# Constants — the 3 Wf-337-G2 target launchers
# Cross-ref TODO/completed/337g2_launcher_v2.md (this wave)
# ---------------------------------------------------------------------------
G4_LAUNCHER = SCRIPTS / "launch_wf124_p3_g4.sh"
G5_LAUNCHER = SCRIPTS / "launch_wf124_p3_g5.sh"
G6_LAUNCHER = SCRIPTS / "launch_wf124_p3_g6.sh"

# Parity triplet — all 3 must mirror the same v2 COMMON_ARGS contract.
WF337G2_TARGETS = (G4_LAUNCHER, G5_LAUNCHER, G6_LAUNCHER)

# Trainer source path for fallback / drift-guard checks.
TRAINER_PATH = TEST_SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py"


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
# Test 1 — G6 launcher has all Wf-328 + Wf-335 fixes
# ---------------------------------------------------------------------------
def test_g6_launcher_has_all_fixes() -> None:
    """Wf-337-G2 — G6 launcher (the v2 reference) has all consolidated fixes.

    The G6 launcher is the primary Wf-337-G2 target.  It must carry:

      - --cpu-num-workers 8        (Wf-328-W2)
      - --num-workers 8            (Wf-328-W6)
      - --w2-gradient-mode sort    (Wf-328-W5)
      - --no-cuda-graphs           (Wf-335-R3)
      - --no-kendall + --no-dann   (Wf-296C — in CONFIRMED_FLAGS array,
                                    not COMMON_ARGS, since these are
                                    per-config ablation levers)

    Test passes iff all 4 COMMON_ARGS fixes + the 2 CONFIRMED_FLAGS
    ablation levers are present as ACTIVE (non-comment) flags.
    """
    src = _read(G6_LAUNCHER)

    # (a) --cpu-num-workers 8
    cpu_lines = _active_flag_lines(src, "--cpu-num-workers")
    assert any("8" in line for line in cpu_lines), (
        f"G6 launcher ({G6_LAUNCHER.name}) missing --cpu-num-workers 8 "
        f"(Wf-328-W2 contract). Active lines: {cpu_lines}"
    )

    # (b) --num-workers 8 (Wf-328-W6)
    num_workers_lines = _active_flag_lines(src, "--num-workers")
    assert any("8" in line for line in num_workers_lines), (
        f"G6 launcher ({G6_LAUNCHER.name}) missing --num-workers 8 "
        f"(Wf-328-W6 contract — wires Wf-326B build_optimized_loader() "
        f"factory with prefetch_factor=8). Active lines: {num_workers_lines}"
    )

    # (c) --w2-gradient-mode sort (Wf-328-W5)
    w2_lines = _active_flag_lines(src, "--w2-gradient-mode")
    assert any("sort" in line for line in w2_lines), (
        f"G6 launcher ({G6_LAUNCHER.name}) --w2-gradient-mode is not "
        f"'sort' (Wf-328-W5 contract). Active lines: {w2_lines}"
    )
    # Anti-regression: ensure the legacy 'sliced' override is GONE from
    # COMMON_ARGS (it would prevent the 2-3× Wf-326A speedup).
    assert not any("sliced" in line for line in w2_lines), (
        f"G6 launcher ({G6_LAUNCHER.name}) still has --w2-gradient-mode "
        f"sliced active. Wf-328-W5 lifts this override for future "
        f"launchers (sliced retained only for paper §6 audit reproducibility "
        f"via explicit CLI). Active lines: {w2_lines}"
    )

    # (d) --no-cuda-graphs (Wf-335-R3)
    no_cuda_lines = _active_flag_lines(src, "--no-cuda-graphs")
    assert no_cuda_lines, (
        f"G6 launcher ({G6_LAUNCHER.name}) missing --no-cuda-graphs "
        f"safety lever (Wf-335-R3 contract — prevents Wf-330-B4 "
        f"RuntimeError on Ru seed=2 + Ir seed=0)."
    )

    # (e) --no-kendall in CONFIRMED_FLAGS (Wf-296C)
    no_kendall_lines = _active_flag_lines(src, "--no-kendall")
    assert no_kendall_lines, (
        f"G6 launcher ({G6_LAUNCHER.name}) missing --no-kendall "
        f"(Wf-296C — no_kendall per-config ablation lever)."
    )

    # (f) --no-dann in CONFIRMED_FLAGS (Wf-296C)
    no_dann_lines = _active_flag_lines(src, "--no-dann")
    assert no_dann_lines, (
        f"G6 launcher ({G6_LAUNCHER.name}) missing --no-dann "
        f"(Wf-296C — no_dann per-config ablation lever)."
    )


# ---------------------------------------------------------------------------
# Test 2 — G5 launcher parity
# ---------------------------------------------------------------------------
def test_g5_launcher_parity() -> None:
    """Wf-337-G2 — G5 launcher mirrors the G6 v2 COMMON_ARGS config.

    G5 (TODO-181, 50-epoch full sweep) shares the trainer + DataLoader
    stack with G6.  All Wf-328 + Wf-335 fixes MUST be wired identically
    so the G5 sweep runs under the same config baseline as the G6 sweep.

    Note: --no-kendall / --no-dann are G6-specific (per-config ablation
    only); G5 does NOT need them.  G5's CONFIRMED_FLAGS block is
    empty/absent (G5 runs the mtl_baseline only).
    """
    src = _read(G5_LAUNCHER)

    # (a) --cpu-num-workers 8
    cpu_lines = _active_flag_lines(src, "--cpu-num-workers")
    assert any("8" in line for line in cpu_lines), (
        f"G5 launcher ({G5_LAUNCHER.name}) missing --cpu-num-workers 8 "
        f"(Wf-328-W2 parity contract)."
    )

    # (b) --num-workers 8 (Wf-328-W6)
    num_workers_lines = _active_flag_lines(src, "--num-workers")
    assert any("8" in line for line in num_workers_lines), (
        f"G5 launcher ({G5_LAUNCHER.name}) missing --num-workers 8 "
        f"(Wf-328-W6 parity contract)."
    )

    # (c) --w2-gradient-mode sort (Wf-328-W5)
    w2_lines = _active_flag_lines(src, "--w2-gradient-mode")
    assert any("sort" in line for line in w2_lines), (
        f"G5 launcher ({G5_LAUNCHER.name}) --w2-gradient-mode is not "
        f"'sort' (Wf-328-W5 parity contract)."
    )
    assert not any("sliced" in line for line in w2_lines), (
        f"G5 launcher ({G5_LAUNCHER.name}) still has --w2-gradient-mode "
        f"sliced active. Wf-337-G2 lifts this for future launchers."
    )

    # (d) --no-cuda-graphs (Wf-335-R3)
    no_cuda_lines = _active_flag_lines(src, "--no-cuda-graphs")
    assert no_cuda_lines, (
        f"G5 launcher ({G5_LAUNCHER.name}) missing --no-cuda-graphs "
        f"safety lever (Wf-335-R3 parity contract)."
    )


# ---------------------------------------------------------------------------
# Test 3 — G4 launcher parity
# ---------------------------------------------------------------------------
def test_g4_launcher_parity() -> None:
    """Wf-337-G2 — G4 launcher mirrors the G6 v2 COMMON_ARGS config.

    G4 (TODO-180, 30-epoch remaining-metals-and-seeds) is the smallest of
    the 3 launchers (15 pairs, 1.8 hr wall).  Same parity contract as G5:
    all 4 Wf-328 + Wf-335 fixes wired identically so the trainer config
    baseline is consistent across G4 / G5 / G6.
    """
    src = _read(G4_LAUNCHER)

    # (a) --cpu-num-workers 8
    cpu_lines = _active_flag_lines(src, "--cpu-num-workers")
    assert any("8" in line for line in cpu_lines), (
        f"G4 launcher ({G4_LAUNCHER.name}) missing --cpu-num-workers 8 "
        f"(Wf-328-W2 parity contract)."
    )

    # (b) --num-workers 8 (Wf-328-W6)
    num_workers_lines = _active_flag_lines(src, "--num-workers")
    assert any("8" in line for line in num_workers_lines), (
        f"G4 launcher ({G4_LAUNCHER.name}) missing --num-workers 8 "
        f"(Wf-328-W6 parity contract)."
    )

    # (c) --w2-gradient-mode sort (Wf-328-W5)
    w2_lines = _active_flag_lines(src, "--w2-gradient-mode")
    assert any("sort" in line for line in w2_lines), (
        f"G4 launcher ({G4_LAUNCHER.name}) --w2-gradient-mode is not "
        f"'sort' (Wf-328-W5 parity contract)."
    )
    assert not any("sliced" in line for line in w2_lines), (
        f"G4 launcher ({G4_LAUNCHER.name}) still has --w2-gradient-mode "
        f"sliced active. Wf-337-G2 lifts this for future launchers."
    )

    # (d) --no-cuda-graphs (Wf-335-R3)
    no_cuda_lines = _active_flag_lines(src, "--no-cuda-graphs")
    assert no_cuda_lines, (
        f"G4 launcher ({G4_LAUNCHER.name}) missing --no-cuda-graphs "
        f"safety lever (Wf-335-R3 parity contract)."
    )


# ---------------------------------------------------------------------------
# Test 4 — backward compat (sliced mode + cpu-num-workers fallback still work)
# ---------------------------------------------------------------------------
def test_backward_compat() -> None:
    """Wf-337-G2 — backward-compat: trainer still accepts the legacy CLI surface.

    Wf-337-G2 lifts the --w2-gradient-mode sliced override from the G4/G5/G6
    launchers because future launches should inherit the Wf-328-W5 sort default.
    However the trainer MUST still accept --w2-gradient-mode sliced for:

      - Paper §6 reproducibility audits (Wf-131 P1 shipped behaviour bit-exact)
      - Existing Wf-260 dataloader test (passes sliced explicitly)
      - Wf-326C / Wf-328-W1 torch.compile reproducer (CUDA Graphs + sliced)

    Plus --cpu-num-workers remains the fallback when --num-workers is None.

    Test passes iff the trainer argparse still exposes all of these flags.
    """
    trainer_src = _read(TRAINER_PATH)

    # (a) --w2-gradient-mode choices still include 'sliced'
    w2_match = re.search(
        r'add_argument\(\s*["\']--w2-gradient-mode["\'].*?choices\s*=\s*\(([^)]+)\)',
        trainer_src,
        re.DOTALL,
    )
    assert w2_match is not None, (
        "Trainer argparse missing --w2-gradient-mode flag (refactor drift)."
    )
    w2_choices = w2_match.group(1)
    assert "sliced" in w2_choices and "sort" in w2_choices, (
        f"--w2-gradient-mode choices {w2_choices!r} must include both "
        f"'sliced' (paper §6 audit) and 'sort' (Wf-328-W5 production)."
    )

    # (b) --cpu-num-workers is still parsed (default 4 per Wf-260)
    assert "--cpu-num-workers" in trainer_src, (
        "Trainer argparse missing --cpu-num-workers flag (refactor drift; "
        "Wf-260 + Wf-328-W2 contract)."
    )

    # (c) --num-workers is still parsed (default None per Wf-328-W6)
    assert "--num-workers" in trainer_src, (
        "Trainer argparse missing --num-workers flag (Wf-328-W6 contract)."
    )

    # (d) --no-cuda-graphs is still parsed (Wf-335-R3)
    assert "--no-cuda-graphs" in trainer_src, (
        "Trainer argparse missing --no-cuda-graphs safety lever "
        "(Wf-335-R3 contract)."
    )

    # (e) --no-kendall + --no-dann are still parsed (Wf-296C)
    assert "--no-kendall" in trainer_src, (
        "Trainer argparse missing --no-kendall flag (Wf-296C contract)."
    )
    assert "--no-dann" in trainer_src, (
        "Trainer argparse missing --no-dann flag (Wf-296C contract)."
    )

    # (f) Trainer parse_args() round-trip — verify the CLI accepts the
    # full Wf-337-G2 v2 config without raising.
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args
    args = parse_args([
        "--cpu-num-workers", "8",
        "--num-workers", "8",
        "--w2-gradient-mode", "sort",
        "--no-cuda-graphs",
        "--no-kendall",
        "--no-dann",
    ])
    assert args.cpu_num_workers == 8
    assert args.num_workers == 8
    assert args.w2_gradient_mode == "sort"
    assert args.cudagraphs_disabled is True
    assert args.kendall is False
    assert getattr(args, "dann", True) is False
