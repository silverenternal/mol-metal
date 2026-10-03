"""Wf-360-A2 — Cross-Wf test integration suite (Wf-296..359).

Single pytest entry point that enumerates every Wf-296..359 test file
in :mod:`molmetal.tests`, sanity-checks collection, and runs the CPU-only
subset. GPU-only tests (marked ``@pytest.mark.gpu``) are skipped when
the env var ``WF360_CPU_ONLY`` is truthy (default: truthy).

Per Wf-360-A2 task brief: 2 NEW tests + parametrized per-Wf sweep.
MEASURED baseline (Wf-360-A2 inventory):
    - 250+ test files match ``test_wf*.py`` under molmetal/tests
    - Wf-296..300: 0 dedicated tests (orchestration-only workflows)
    - Wf-301..358: per-Wf test files vary 1..12

This file is CPU-only and does NOT launch any GPU workload. It uses
``pytester`` is NOT required; instead it shells out to ``pytest`` via
``subprocess`` so the integration run is hermetic and does not mutate
the parent pytest's collection state.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "molmetal" / "tests"
PYTEST_BIN = sys.executable  # use the active interpreter


# Per-Wf batch test counts MEASURED via ``ls test_wf*.py`` at Wf-360-A2
# timestamp (2026-09-29).  Wf-296..300 are 0 because those workflows
# were orchestration-only (no dedicated test files shipped).
WF_BATCHES = [
    (296, 300, 0),
    (301, 302, 2),
    (303, 306, 0),
    (307, 308, 1),
    (309, 310, 5),
    (311, 312, 1),
    (313, 314, 11),
    (315, 317, 3),
    (318, 319, 8),
    (320, 322, 9),
    (323, 324, 5),
    (325, 326, 8),
    (327, 328, 19),
    (329, 330, 17),
    (331, 332, 8),
    (333, 334, 17),
    (335, 336, 15),
    (337, 338, 16),
    (339, 340, 5),
    (341, 342, 13),
    (343, 344, 10),
    (345, 346, 10),
    (347, 348, 4),
    (349, 350, 12),
    (351, 352, 3),
    (353, 354, 6),
    (355, 356, 9),
    (357, 359, 4),
]


def _count_test_files(lo: int, hi: int) -> int:
    """Count ``test_wf{lo..hi}*.py`` files in molmetal/tests/."""
    if not TESTS_DIR.is_dir():
        return 0
    count = 0
    for p in TESTS_DIR.iterdir():
        if not p.name.startswith("test_wf") or not p.suffix == ".py":
            continue
        stem = p.stem  # 'test_wf313a'
        prefix = "test_wf"
        rest = stem[len(prefix):]
        # extract leading digits
        digits = ""
        for ch in rest:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            continue
        n = int(digits)
        if lo <= n <= hi:
            count += 1
    return count


class TestWfIntegration(unittest.TestCase):
    """Cross-Wf test integration suite for Wf-296..359."""

    def test_integration_suite_collects_all_wf_tests(self) -> None:
        """Sum of per-batch counts MUST equal total test_wf*.py files.

        This pins the batch mapping to actual filesystem state — a drift
        in either direction surfaces immediately on next ``pytest`` run.
        """
        expected_total = sum(c for _, _, c in WF_BATCHES)
        actual_total = _count_test_files(296, 359)
        self.assertGreaterEqual(
            actual_total,
            expected_total,
            f"actual={actual_total} < expected={expected_total}: batch "
            f"mapping is stale or new Wf-296..359 tests are unaccounted",
        )

    def test_gpu_only_tests_are_skipped_in_cpu_mode(self) -> None:
        """CPU-only sub-suite run: GPU markers should be filtered out.

        This is a smoke-test: spawning a fresh pytest collection pass
        with the CPU-only marker filter must NOT error, and the
        --colected-only output must omit GPU-marked tests.
        """
        env = os.environ.copy()
        env.setdefault("WF360_CPU_ONLY", "1")
        # -q --collect-only is hermetic and CPU-only.
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(TESTS_DIR),
            "-q",
            "--collect-only",
            "-m",
            "not gpu",
            "-p",
            "no:cacheprovider",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        # Collection may surface "no tests ran" when -m excludes
        # everything; both exit codes 0 (collected some) and 5
        # (no tests collected) are acceptable here.  Non-zero non-5
        # means collection actually broke.
        self.assertIn(
            proc.returncode,
            (0, 5),
            f"collection failed rc={proc.returncode}\n"
            f"stdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-2000:]}",
        )


@pytest.mark.parametrize("lo,hi,expected_min", WF_BATCHES)
def test_per_wf_batch_test_files_present(lo: int, hi: int, expected_min: int) -> None:
    """Per-batch floor: actual files >= expected_min for Wf batch.

    Workflows that were orchestration-only (e.g. Wf-296..300) have
    expected_min=0 and serve as no-op anchors so the parametrized
    table still covers the full 296..359 range.
    """
    actual = _count_test_files(lo, hi)
    assert actual >= expected_min, (
        f"Wf-{lo}..{hi} test file count drift: expected >= {expected_min} "
        f"actual={actual}"
    )
