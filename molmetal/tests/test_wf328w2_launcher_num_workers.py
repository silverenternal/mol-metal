"""Wf-328-W2 — Launch-script --cpu-num-workers override tests.

Owner: Hugo Lee
Date: 2026-09-29
Parent: TODO/pending/182_wf124_p3_per_innovation_ablation.md + Wf-326B dataloader_optimized
Per AGENTS.md §5.1 (kill_then_adapt) + §5.2 (MEASURED-vs-PROJECTED honest framing).

Background — Wf-326B MEASURED (2026-09-29 grep): 6 launch scripts hard-coded
``--cpu-num-workers 0`` which forced the trainer's manual-loop fallback path
(no DataLoader, no pin_memory, no workers).  Wf-260 had corrected the trainer
default to ``num_workers=4`` but the launchers reverted it to 0, so the GPU
starvation persisted.

Wf-328-W2 fix — replaces ``--cpu-num-workers 0`` with ``--cpu-num-workers 8``
(Wf-326B OPTIMAL_NUM_WORKERS on 24-core host) across 6 launch scripts:

  1. scripts/launch_wf124_p3_g4.sh         (TODO-180 G4 — 14 metal,seed pairs)
  2. scripts/launch_wf124_p3_g5.sh         (TODO-181 G5 — 5 metals x 3 seeds)
  3. scripts/launch_wf124_p3_g6.sh         (TODO-182 G6 — 4 cfgs x 5 metals x 3 seeds)
  4. scripts/launch_wf310b_g6_final.sh     (Wf-310B G6 final wrapper)
  5. scripts/launch_wf300_cfm_main.sh      (Wf-300C Wave-B — G6 C12 no_wass)
  6. scripts/retry_g5_missing_pairs.sh     (Wf-310A — Ru s=2 + Ir s=0 retry)

Tests (5 total):

  - test_no_launch_script_zero_override        (gates the regression)
  - test_six_launch_scripts_have_num_workers_8 (post-fix confirmation)
  - test_suggest_num_workers_returns_valid     (Wf-326B heuristic sanity)
  - test_suggest_num_workers_respects_cap      (cap-bounded honesty gate)
  - test_optimal_num_workers_constant_matches  (Wf-326B invariant)

All tests are CPU-only — no torch.cuda, no real launches.  < 5 s wall.

Honest framing (AGENTS.md §5.2):
  - MEASURED: 6 launch scripts (Wf-326B grep 2026-09-29) had --cpu-num-workers 0.
  - MEASURED: trainer default is 4 (Wf-260 fix in train_dmpnn_multitask_wasserstein.py:1315).
  - MEASURED: 24 physical cores (psutil.cpu_count(logical=False) = 24).
  - PROJECTED: OPTIMAL_NUM_WORKERS=8 (Wf-326B cap-bounded sweet spot).
  - PROJECTED: 3-5x throughput lift vs manual loop (BLOCKED on AGENTS.md §5.1).
  - 1 launch script (launch_wf306a_drugood_vina_gpu.sh) uses --cpu-num-workers 2
    (non-zero, not in Wf-328-W2 scope; documented as future-work).
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


# Import the Wf-326B heuristic — module name has digits, use importlib.
import importlib

_wf326b = importlib.import_module("molmetal.scripts.wf326b_dataloader_optimized")


# ---------------------------------------------------------------------------
# Constants — the 6 launch scripts that Wf-328-W2 fixes
# ---------------------------------------------------------------------------
WF328W2_TARGETS = (
    SCRIPTS / "launch_wf124_p3_g4.sh",
    SCRIPTS / "launch_wf124_p3_g5.sh",
    SCRIPTS / "launch_wf124_p3_g6.sh",
    SCRIPTS / "launch_wf310b_g6_final.sh",
    SCRIPTS / "launch_wf300_cfm_main.sh",
    SCRIPTS / "retry_g5_missing_pairs.sh",
)

# Two of the six were archived by Wf-437 (the simplify pass) and no longer
# exist under scripts/.  They cannot regress to `--cpu-num-workers 0`, so the
# tests below skip them rather than asserting against a file that is gone —
# but the names stay listed so a restore is re-checked automatically.
ARCHIVED_TARGETS = {
    "launch_wf124_p3_g4.sh": "archived under TODO/archive/wf437_simplify/",
    "launch_wf300_cfm_main.sh": "archived under TODO/archive/wf437_simplify/",
}


def _existing_targets():
    """The subset of WF328W2_TARGETS that is still on disk."""
    return [p for p in WF328W2_TARGETS if p.exists()]


# Allow the new --cpu-num-workers value to be either OPTIMAL_NUM_WORKERS (8)
# from Wf-326B, or the trainer default (4) from Wf-260 — both non-zero
# (manual-loop fallback).  Anything else is a regression.
NONZERO_VALUES = (4, 8, 12)  # trainer default + Wf-326B OPTIMAL + cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read(path: Path) -> str:
    assert path.exists(), f"launch script missing: {path}"
    return path.read_text()


def _active_num_workers_lines(src: str) -> list[str]:
    """Extract --cpu-num-workers <int> lines that are NOT in comments.

    Skips comment lines (start with ``#``) so the Wf-328-W2 header banner
    (which mentions the old ``--cpu-num-workers 0`` value for context)
    is not flagged as an active override.
    """
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "--cpu-num-workers" in line:
            out.append(line)
    return out


def _parse_num_workers_value(line: str) -> int:
    """Parse ``--cpu-num-workers <int>`` from a single shell line.

    Accepts ``--cpu-num-workers 8``, ``--cpu-num-workers 8 \\``, etc.
    """
    m = re.search(r"--cpu-num-workers\s+(-?\d+)", line)
    assert m, f"could not parse --cpu-num-workers value from line: {line!r}"
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Test 1 — no launch script hard-codes --cpu-num-workers 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", WF328W2_TARGETS, ids=lambda p: p.name)
def test_no_launch_script_zero_override(path: Path) -> None:
    """Gates the Wf-328-W2 regression — every active --cpu-num-workers
    override in the 6 fixed launch scripts must be non-zero (i.e. honors
    Wf-260 default 4 or Wf-326B OPTIMAL_NUM_WORKERS 8, not the manual-loop
    fallback 0 that caused GPU starvation)."""
    if not path.exists():
        pytest.skip(
            f"{path.name}: {ARCHIVED_TARGETS.get(path.name, 'missing from scripts/')}"
            f" — no file, nothing to regress"
        )
    src = _read(path)
    active = _active_num_workers_lines(src)
    assert active, f"{path.name} has no active --cpu-num-workers line"
    for line in active:
        val = _parse_num_workers_value(line)
        assert val != 0, (
            f"{path.name} still hard-codes --cpu-num-workers 0 "
            f"(manual-loop fallback, Wf-326B regression): {line!r}"
        )
        assert val in NONZERO_VALUES, (
            f"{path.name} uses --cpu-num-workers {val}; expected one of "
            f"{NONZERO_VALUES} (trainer default 4 / Wf-326B OPTIMAL 8 / "
            f"suggest_num_workers cap 12)"
        )


# ---------------------------------------------------------------------------
# Test 2 — post-fix confirmation: 6 scripts now use 8 (Wf-326B OPTIMAL)
# ---------------------------------------------------------------------------
def test_six_launch_scripts_have_num_workers_8() -> None:
    """Post-Wf-328-W2 confirmation — every surviving launch script uses
    ``--cpu-num-workers 8`` (Wf-326B OPTIMAL_NUM_WORKERS on the 24-core host).

    Two of the original six were archived by Wf-437; this checks the four that
    remain, and asserts that exactly four remain so a *further* silent removal
    still fails the test."""
    targets = _existing_targets()
    assert len(targets) == 4, (
        f"{len(targets)} of the six Wf-328-W2 launchers are on disk "
        f"({[p.name for p in targets]}); expected 4 — two were archived by "
        f"Wf-437.  If a third disappeared, say so here deliberately."
    )
    for path in targets:
        src = _read(path)
        active = _active_num_workers_lines(src)
        # Some scripts only override --cpu-num-workers in one wave (e.g. the
        # CFM Wave-A does not consume the flag).  Skip scripts with no active
        # override, but a script in this family that used to set it and no
        # longer does should be noticed — hence the explicit list check above.
        if not active:
            continue
        for line in active:
            val = _parse_num_workers_value(line)
            assert val == 8, (
                f"{path.name} uses --cpu-num-workers {val}; Wf-328-W2 expects 8 "
                f"(Wf-326B OPTIMAL_NUM_WORKERS on 24-core host): {line!r}"
            )


# ---------------------------------------------------------------------------
# Test 3 — suggest_num_workers() returns a valid value (Wf-326B sanity)
# ---------------------------------------------------------------------------
def test_suggest_num_workers_returns_valid() -> None:
    """Wf-326B suggest_num_workers() heuristic must return a non-negative
    int in [0, cap] for a range of (n_physical_cores, n_gpus, cap) inputs.

    Sanity bounds: 0 <= result <= cap (the cap is the honest-negative gate
    preventing runaway worker counts on big hosts)."""
    cases = [
        # (n_physical_cores, n_gpus, cap, expected_floor, expected_ceiling)
        (24, 1, 12, 0, 12),  # this host (24 physical), default cap
        (24, 1, 8, 0, 8),  # Wf-326B OPTIMAL cap
        (4, 1, 12, 0, 12),  # small box
        (64, 1, 12, 0, 12),  # large box, cap holds (no runaway)
        (1, 1, 12, 0, 12),  # 1-core box
        (24, 2, 12, 0, 12),  # 2 GPUs share cap
        (24, 1, 4, 0, 4),  # tight cap
    ]
    for n_phys, n_gpus, cap, lo, hi in cases:
        result = _wf326b.suggest_num_workers(
            n_physical_cores=n_phys,
            n_gpus=n_gpus,
            cap=cap,
        )
        assert isinstance(
            result, int
        ), f"suggest_num_workers returned non-int: {type(result).__name__}"
        assert lo <= result <= hi, (
            f"suggest_num_workers(n_physical_cores={n_phys}, n_gpus={n_gpus}, "
            f"cap={cap}) returned {result}; out of [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# Test 4 — suggest_num_workers() respects the cap (honesty gate)
# ---------------------------------------------------------------------------
def test_suggest_num_workers_respects_cap() -> None:
    """Cap is the Wf-326B honest-negative gate — must hold even on huge hosts
    (96-core / 128-core) so we never spawn more collate workers than the
    empirical sweet spot (12)."""
    huge_cores = [64, 96, 128, 256]
    cap = 12
    for n_phys in huge_cores:
        result = _wf326b.suggest_num_workers(n_physical_cores=n_phys, cap=cap)
        assert result <= cap, (
            f"suggest_num_workers broke the cap on {n_phys}-core host: "
            f"got {result}, cap={cap}"
        )


# ---------------------------------------------------------------------------
# Test 5 — OPTIMAL_NUM_WORKERS constant matches expected (Wf-326B invariant)
# ---------------------------------------------------------------------------
def test_optimal_num_workers_constant_matches() -> None:
    """Wf-326B invariant — OPTIMAL_NUM_WORKERS=8 is the documented sweet
    spot for the 24-core kanzi host.  This test pins the constant so a
    silent change here is caught at CI, not at the launch script.

    If this test fails after a deliberate constant update, re-read
    molmetal/scripts/wf326b_dataloader_optimized.py §Constants to confirm
    the new value is justified, then update the expected value here.
    """
    expected = 8
    assert _wf326b.OPTIMAL_NUM_WORKERS == expected, (
        f"OPTIMAL_NUM_WORKERS drifted from {expected} to "
        f"{_wf326b.OPTIMAL_NUM_WORKERS}; if intentional, update this test "
        f"AND the 6 launch scripts in scripts/launch_wf*.sh"
    )
    # And the launcher-fixed value should match this constant.
    # (Guard against silent drift between the Wf-326B constant and the
    # launch scripts that consume it.  The two archived by Wf-437 are skipped.)
    for path in _existing_targets():
        src = _read(path)
        active = _active_num_workers_lines(src)
        for line in active:
            val = _parse_num_workers_value(line)
            assert val == expected, (
                f"{path.name} uses --cpu-num-workers {val} but "
                f"OPTIMAL_NUM_WORKERS={expected}; drift detected"
            )
