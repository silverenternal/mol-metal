#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wf-R40 — Launch-script `--rh-oversample` P-A fix tests.

Owner: Hugo Lee
Date: 2026-10-08
Parent: TODO/03_plans/PROTOCOL_RECONCILIATION_AND_RE_PLAN_2026_10_07.md §P-A
Per AGENTS.md §5.1 (kill_then_adapt) + §5.5 (filesystem truth) + §5.10 (Apache-2.0).

Background — Wf-R40 MEASURED (2026-10-08 grep):
``scripts/launch_dmpnn_clean5_lomo.sh:69`` hard-coded ``--rh-oversample 500``,
which re-injects the fabricated-label Rh cohort on every P-A re-measurement
launch and overrides the trainer's Wf-R10 corrected default (``default=0``,
``train_metal_lomo_wasserstein.py:1831``).  The 5-fold real-only re-measurement
that the Wf-432..435 audit requires was impossible without overriding this
launcher.

Wf-R40 fix — parameterize via ``RH_OVERSAMPLE`` env var (default 0), comment
the historical-path context.  Historical wf125_p1 / wf131_p2 launchers retain
their hard-coded 500 (those are intentionally fabricated-label ablation paths).

Tests (4 total):

  - test_launcher_rh_oversample_default_zero       (gates the regression)
  - test_launcher_rh_oversample_no_longer_500      (post-fix confirmation)
  - test_launcher_rh_oversample_env_override_works (historical-path escape hatch)
  - test_trainer_default_remains_zero              (Wf-R10 trainer invariant)
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "launch_dmpnn_clean5_lomo.sh"
TRAINER = REPO_ROOT / "molmetal" / "scripts" / "train_metal_lomo_wasserstein.py"


def _launcher_text() -> str:
    return LAUNCHER.read_text()


def _trainer_text() -> str:
    return TRAINER.read_text()


def _dry_run_cmd() -> str:
    """Return the --rh-oversample line that the launcher would emit under DRY_RUN."""
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "1"},
    )
    # The dry-run echo prints `  CMD: ... --rh-oversample <N> ...`
    match = re.search(r"--rh-oversample\s+(\S+)", result.stdout)
    assert match, f"no --rh-oversample in dry-run output: {result.stdout!r}"
    return match.group(1)


def test_launcher_rh_oversample_default_zero() -> None:
    """P-A regression gate: default launch must use --rh-oversample 0."""
    value = _dry_run_cmd()
    assert value == "0", (
        f"launcher default --rh-oversample must be 0 (real-only fold), got {value!r}"
    )


def test_launcher_rh_oversample_no_longer_500() -> None:
    """Post-fix: no executable line in the launcher must hard-code `--rh-oversample 500`.

    Comment lines are exempt (the fix intentionally references the historical
    value in a comment explaining the P-A regression).  We anchor on lines
    starting with whitespace + the flag (the actual CMD-array invocation).
    """
    for lineno, line in enumerate(_launcher_text().splitlines(), start=1):
        stripped = line.lstrip()
        # Skip pure comment lines.
        if stripped.startswith("#"):
            continue
        # The CMD-array form is `--rh-oversample` followed by a quoted var.
        # Hard-coded 500 here would re-introduce the regression.
        assert not re.search(r"--rh-oversample\s+500\b", line), (
            f"P-A fix regressed at line {lineno}: {line!r}"
        )


def test_launcher_rh_oversample_env_override_works() -> None:
    """Historical-path escape hatch: RH_OVERSAMPLE=500 must still work."""
    env = {**os.environ, "RH_OVERSAMPLE": "500"}
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    match = re.search(r"--rh-oversample\s+(\S+)", result.stdout)
    assert match, f"no --rh-oversample in env-overridden dry-run output: {result.stdout!r}"
    assert match.group(1) == "500", (
        f"RH_OVERSAMPLE=500 env override broken; got --rh-oversample {match.group(1)!r}"
    )


def test_trainer_default_remains_zero() -> None:
    """Wf-R10 trainer invariant: argparse default must still be 0.

    This is the upstream fix; the launcher P-A fix would be useless if the
    trainer default regressed to 500 again.
    """
    text = _trainer_text()
    # Find the `--rh-oversample` argparse definition (default=).
    # The flag may be on one or more lines.
    block_match = re.search(
        r'["\']--rh-oversample["\'][^)]*?default\s*=\s*(\d+)',
        text,
        re.DOTALL,
    )
    assert block_match, (
        "could not locate --rh-oversample default in trainer argparse"
    )
    default = int(block_match.group(1))
    assert default == 0, (
        f"trainer --rh-oversample default regressed: expected 0, got {default}"
    )
