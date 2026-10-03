"""Wf-350-B2 — prereg.md pre-registration record template + launch script wiring tests (2 NEW tests).

Owner: Hugo Lee (autonomous Wf-350 ultracode unit)
Date: 2026-09-29
Parent: Wf-350 ultracode residual wave + Wf-343a-R1 finding #2
       (DeepMind §3.1 pre-registration record; cf. AGENTS.md §5.2 honest framing)

These 2 NEW tests verify the PREREG_TEMPLATE.md template schema + the
--prereg wiring into the 3 launch scripts (launch_wf124_p3_g5.sh +
launch_wf124_p3_g6.sh + recover_g6_failed_tuples.sh) per Wf-350-B2 task
contract:

  1. **test_prereg_template_schema_validity** — verifies the
     ``docs/PREREG_TEMPLATE.md`` template has all 7 required sections
     (§1 Research question + hypothesis, §2 Dataset(s) + cohort,
     §3 Hyperparameters + model architecture, §4 Evaluation metric(s)
     + acceptance gate, §5 Pre-registered ablation plan, §6 Pre-registered
     analysis plan, §7 Anti-patterns: NO post-hoc cherry-picking).

  2. **test_launch_scripts_accept_prereg_arg** — verifies that all 3
     launch scripts accept the --prereg <path> argument and emit either
     an OK schema-valid line or a WARNING when the prereg is unset.

These tests verify **structural integration** (the template file has
all 7 sections + the launch scripts parse --prereg) — they do NOT
actually invoke the launchers, which would require a USER_OK-granted
live GPU launch.

Per the Wf-350-B2 task contract:
  - 2 NEW tests
  - 0 modified existing tests
  - Tier-2 marker: AGENTS.md §5.4 — pre-registration is USER_OK-gated,
    NOT auto-enforced; these tests verify the surface exists, not that
    the user must use it.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Each test asserts ONE observable contract (template schema validity
  or launch-script argument parsing) so failures localise cleanly.
* Tests do NOT fabricate any metric values — every assertion traces to
  the on-disk PREREG_TEMPLATE.md file (test 1) or the launch script
  source (test 2).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]  # /home/hugo/codes/mol-metal


def test_prereg_template_schema_validity() -> None:
    """Wf-350-B2 task contract — ``docs/PREREG_TEMPLATE.md`` template
    must have all 7 required sections per DeepMind §3.1 prereg +
    Wf-343a-R1 finding #2 + AGENTS.md §5.2 honest framing.

    Verifies:
      - The template file exists at ``docs/PREREG_TEMPLATE.md``.
      - All 7 required sections are present (one per Wf-350-B2 §1-§7).
      - The template references the canonical schema fields
        (hypothesis, primary_metric, baseline, expected_effect_size,
        falsification_criterion).
    """
    template_path = REPO_ROOT / "docs" / "PREREG_TEMPLATE.md"
    assert template_path.exists(), f"FAIL: prereg template not found at {template_path}"

    text = template_path.read_text(encoding="utf-8")

    # Required sections per Wf-350-B2 §1-§7
    required_sections = [
        r"^## §1 Research question",
        r"^## §2 Dataset\(s\)",
        r"^## §3 Hyperparameters \+ model architecture",
        r"^## §4 Evaluation metric\(s\)",
        r"^## §5 Pre-registered ablation plan",
        r"^## §6 Pre-registered analysis plan",
        r"^## §7 Anti-patterns: NO post-hoc cherry-picking",
    ]
    for section_pattern in required_sections:
        assert re.search(section_pattern, text, re.MULTILINE), (
            f"FAIL: prereg template missing required section matching {section_pattern!r}"
        )

    # Canonical schema fields per DeepMind §3.1 (Wf-343a-R1 finding #2)
    canonical_fields = [
        "hypothesis",
        "primary_metric",
        "baseline",
        "expected_effect_size",
        "falsification_criterion",
    ]
    for field in canonical_fields:
        assert field in text, (
            f"FAIL: prereg template missing canonical schema field {field!r}"
        )

    # The template must reference both AGENTS.md §5.2 (honest framing)
    # and the Wf-350-B2 ship doc (self-reference).
    assert "AGENTS.md §5.2" in text, (
        "FAIL: prereg template must reference AGENTS.md §5.2 honest framing"
    )
    assert "Wf-350-B2" in text, (
        "FAIL: prereg template must reference Wf-350-B2 (this task)"
    )


@pytest.mark.parametrize(
    "launch_script",
    [
        "scripts/launch_wf124_p3_g5.sh",
        "scripts/launch_wf124_p3_g6.sh",
        "scripts/recover_g6_failed_tuples.sh",
    ],
)
def test_launch_scripts_accept_prereg_arg(launch_script: str) -> None:
    """Wf-350-B2 task contract — all 3 launch scripts must accept
    the --prereg <path> argument and emit either an OK schema-valid
    line or a WARNING when the prereg is unset.

    Verifies:
      - The launch script source references ``--prereg`` parsing logic.
      - The launch script (with --prereg docs/PREREG_TEMPLATE.md --dry-run)
        emits the OK line: ``OK: prereg.md schema valid``.
      - The launch script (with --dry-run only, no --prereg) emits
        the WARNING line: ``WARNING: --prereg <path> not supplied``.
    """
    script_path = REPO_ROOT / launch_script
    assert script_path.exists(), f"FAIL: launch script not found at {script_path}"

    text = script_path.read_text(encoding="utf-8")
    # Source must reference --prereg parsing
    assert "--prereg" in text, (
        f"FAIL: launch script {launch_script} does not parse --prereg argument"
    )
    # Source must mention PREREG verification
    assert "PREREG verification" in text or "PREREG_TEMPLATE" in text, (
        f"FAIL: launch script {launch_script} missing PREREG verification block"
    )

    # Subprocess check: --prereg <template> --dry-run should emit OK line.
    # Use timeout=60s since the dry-run path also probes CSV sanity + may
    # launch pre-flight; capture stderr too in case the OK line goes there.
    # For recover_g6_failed_tuples.sh, --dry-run must come FIRST so the
    # script's existing ${1:-} dry-run check (line 165) detects it (the
    # script's DRY_RUN override ignores the --prereg argument position).
    if launch_script == "scripts/recover_g6_failed_tuples.sh":
        argv_with = ["--dry-run", "--prereg", "docs/PREREG_TEMPLATE.md"]
        argv_without = ["--dry-run"]
    else:
        argv_with = ["--prereg", "docs/PREREG_TEMPLATE.md", "--dry-run"]
        argv_without = ["--dry-run"]
    result_with = subprocess.run(
        ["bash", str(script_path), *argv_with],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result_with.returncode == 0, (
        f"FAIL: {launch_script} {' '.join(argv_with)} exited rc={result_with.returncode}; "
        f"stdout={result_with.stdout!r}; stderr={result_with.stderr!r}"
    )
    combined_with = result_with.stdout + "\n" + result_with.stderr
    assert "OK: prereg.md schema valid" in combined_with, (
        f"FAIL: {launch_script} {' '.join(argv_with)} did not emit OK line; got:\n{combined_with[:1000]}"
    )

    # Subprocess check: --dry-run only should emit WARNING.
    result_without = subprocess.run(
        ["bash", str(script_path), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result_without.returncode == 0, (
        f"FAIL: {launch_script} --dry-run exited rc={result_without.returncode}"
    )
    combined_without = result_without.stdout + "\n" + result_without.stderr
    assert "WARNING: --prereg <path> not supplied" in combined_without, (
        f"FAIL: {launch_script} --dry-run did not emit WARNING line; got:\n{combined_without[:1000]}"
    )
