"""Wf-337-G2 — G5/G6 launch wrapper contract + backward-compat tests.

Owner: Hugo Lee
Date: 2026-09-29 (contract rewritten 2026-10-20)
Parent: Wf-337 (post-Wf-335 + Wf-328 ultracode consolidation)
Refs:
  - TODO/completed/335r1_no_cudagraphs_fix.md (Wf-335-R1 — torch.compile mode flip)
  - TODO/completed/335r3_g6_no_cudagraphs.md (Wf-335-R3 — --no-cuda-graphs safety lever)
  - TODO/completed/328w2_launcher_num_workers_fix.md (Wf-328-W2 — --cpu-num-workers 8)
  - TODO/completed/328w5_w2_sort_wired.md (Wf-328-W5 — --w2-gradient-mode)
  - TODO/completed/328w6_dataloader_factory_wired.md (Wf-328-W6 — --num-workers flag)

Background — Wf-337-G2 consolidated the Wf-328 + Wf-335 fixes into the
G4 / G5 / G6 launchers.  Two of those "fixes" were later removed because
they did not exist in the trainer's argparse:

  * ``--num-workers 8``     (Wf-328-W6) — the DataLoader factory it fed was
                             removed with the m21 selective revert
                             (``af84a545``).  Wf-r28 (2026-10-07) removed the
                             flag from the launchers after all 15 G5 pairs
                             aborted with argparse rc=2 "unrecognized
                             arguments".
  * ``--no-cuda-graphs``    (Wf-335-R3) — same removal, same reason; CUDA
                             graphs are off by default because
                             ``--torch-compile`` is default OFF.

What the launchers carry today:

  1. --cpu-num-workers 8       (Wf-328-W2; accepted, inert — the trainer
                                builds batches itself)
  2. --w2-gradient-mode sliced (Wf-328-W5 §Integration points: G5/G6 keep
                                sliced bit-exact for the paper §6 audit)
  3. --no-kendall / --no-dann  (Wf-296C — per-config ablation; G6 only)

Tests (4 total):

  - test_g6_launcher_has_all_fixes  (primary contract gate)
  - test_g5_launcher_parity         (G5 mirrors the G6 config)
  - test_g4_launcher_parity         (G4 is ARCHIVED — assert it stays gone)
  - test_backward_compat            (trainer still accepts the surviving CLI)

All tests are CPU-only — no torch.cuda, no real launches.  < 5 s wall.

Honest framing (AGENTS.md §5.2 + party-twenty-spirit §3.5):
  - MEASURED: the trainer exposes --cpu-num-workers, --w2-gradient-mode,
    --no-kendall, --no-dann.  It does NOT expose --num-workers,
    --no-cuda-graphs or --bf16 (all removed by the m21 revert).
  - MEASURED: 2 active launchers (g5/g6) carry the surviving flags; g4 is
    archived under TODO/archive/wf437_simplify/.
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
# Constants — the target launchers
# ---------------------------------------------------------------------------
G4_LAUNCHER = SCRIPTS / "launch_wf124_p3_g4.sh"  # archived (Wf-437)
G5_LAUNCHER = SCRIPTS / "launch_wf124_p3_g5.sh"
G6_LAUNCHER = SCRIPTS / "launch_wf124_p3_g6.sh"

# Parity pair — both must mirror the same COMMON_ARGS contract.
WF337G2_TARGETS = (G5_LAUNCHER, G6_LAUNCHER)

# Trainer source path for fallback / drift-guard checks.
TRAINER_PATH = TEST_SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py"

# Flags that were removed from the trainer AND the launchers.  A reappearance
# in a launcher is an argparse rc=2 waiting to happen.
REMOVED_FLAGS = ("--num-workers", "--no-cuda-graphs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read(path: Path) -> str:
    assert path.exists(), f"launch script missing: {path}"
    return path.read_text()


def _active_flag_lines(src: str, flag: str) -> list[str]:
    """Extract ``--<flag>`` lines that are NOT in comments.

    Skips comment lines (start with ``#``) so the Wf-r28 / Wf-335-R3 banner
    comments (which mention the removed flags for documentation purposes) do
    NOT count as active overrides.
    """
    out: list[str] = []
    for raw in src.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if flag in line:
            out.append(line)
    return out


def _assert_launcher_contract(path: Path, *, ablation_levers: bool) -> None:
    src = _read(path)

    cpu_lines = _active_flag_lines(src, "--cpu-num-workers")
    assert any(re.search(r"--cpu-num-workers\s+8\b", line) for line in cpu_lines), (
        f"{path.name} missing --cpu-num-workers 8 (Wf-328-W2 contract). "
        f"Active lines: {cpu_lines}"
    )

    w2_lines = _active_flag_lines(src, "--w2-gradient-mode")
    assert any(re.search(r"--w2-gradient-mode\s+sliced\b", line) for line in w2_lines), (
        f"{path.name}: --w2-gradient-mode must be 'sliced' (Wf-328-W5 "
        f"§Integration points — G5/G6 keep the paper §6 audit path "
        f"bit-exact). Active lines: {w2_lines}"
    )

    # Removed flags must not be active.  ``--cpu-num-workers`` contains the
    # substring "num-workers", so match the leading "--num-workers" exactly.
    for flag in REMOVED_FLAGS:
        active = [
            line
            for line in _active_flag_lines(src, flag)
            if re.search(r"(?<!cpu-)" + re.escape(flag) + r"\b", line)
        ]
        assert not active, (
            f"{path.name} passes {flag}, which the trainer does not declare — "
            f"every pair would abort with argparse rc=2 (Wf-r28). "
            f"Active lines: {active}"
        )

    if ablation_levers:
        for lever in ("--no-kendall", "--no-dann"):
            assert _active_flag_lines(src, lever), (
                f"{path.name} missing {lever} (Wf-296C ablation lever)."
            )


# ---------------------------------------------------------------------------
# Test 1 — G6 launcher contract
# ---------------------------------------------------------------------------
def test_g6_launcher_has_all_fixes() -> None:
    """G6 launcher (the v2 reference) carries the surviving consolidated flags.

    Asserts ``--cpu-num-workers 8`` + ``--w2-gradient-mode sliced`` +
    ``--no-kendall`` / ``--no-dann``, and that the two flags removed by
    Wf-r28 (``--num-workers``, ``--no-cuda-graphs``) are not active.
    """
    _assert_launcher_contract(G6_LAUNCHER, ablation_levers=True)


# ---------------------------------------------------------------------------
# Test 2 — G5 launcher parity
# ---------------------------------------------------------------------------
def test_g5_launcher_parity() -> None:
    """G5 mirrors the G6 COMMON_ARGS config (minus the G6-only ablation levers)."""
    _assert_launcher_contract(G5_LAUNCHER, ablation_levers=False)


# ---------------------------------------------------------------------------
# Test 3 — G4 is archived
# ---------------------------------------------------------------------------
def test_g4_launcher_parity() -> None:
    """G4 was archived by Wf-437; assert it stays archived.

    The parity contract cannot be checked against a file that is not on disk,
    and silently dropping the check would hide a re-introduction of a launcher
    nobody maintains.  The archive location is named in the failure message so
    restoring it is a one-step decision rather than an archaeology exercise.
    """
    assert not G4_LAUNCHER.exists(), (
        f"{G4_LAUNCHER.name} is back on disk.  It was archived under "
        f"TODO/archive/wf437_simplify/ (Wf-437) — either restore it to the "
        f"parity contract in this file or remove it again."
    )


# ---------------------------------------------------------------------------
# Test 4 — backward compat (surviving CLI surface)
# ---------------------------------------------------------------------------
def test_backward_compat() -> None:
    """The trainer still accepts the surviving CLI surface, and only that.

    ``--w2-gradient-mode`` keeps BOTH choices so the paper §6 audit can pin
    ``sliced`` while a future launcher can opt into ``sort``.  The removed
    flags must stay removed — re-adding one silently would make the archived
    launcher recipe look valid again.
    """
    trainer_src = _read(TRAINER_PATH)

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
        f"'sliced' (shipped default / paper §6 audit) and 'sort' (Wf-328-W5 "
        f"opt-in path)."
    )

    assert "--cpu-num-workers" in trainer_src, (
        "Trainer argparse missing --cpu-num-workers flag (Wf-260 + Wf-328-W2)."
    )
    for lever in ("--no-kendall", "--no-dann"):
        assert lever in trainer_src, f"Trainer argparse missing {lever} (Wf-296C)."

    for removed in REMOVED_FLAGS:
        declared = re.search(
            r"add_argument\(\s*\n?\s*[\"']" + re.escape(removed) + r"[\"']",
            trainer_src,
        )
        assert declared is None, (
            f"Trainer declares {removed} again — it was removed with the m21 "
            f"revert and the launchers no longer pass it."
        )

    # Round-trip: the shipped config parses, the removed one does not.
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args

    args = parse_args(
        [
            "--cpu-num-workers",
            "8",
            "--w2-gradient-mode",
            "sliced",
            "--no-kendall",
            "--no-dann",
        ]
    )
    assert args.cpu_num_workers == 8
    assert args.w2_gradient_mode == "sliced"
    assert args.kendall is False
    assert getattr(args, "dann", True) is False

    with pytest.raises(SystemExit):
        parse_args(["--num-workers", "8"])
