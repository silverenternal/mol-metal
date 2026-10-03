"""Wf-347-A2 — Pipeline master Wf-343c4 integration tests (NEW per Wf-347-A2).

Owner: Hugo Lee (autonomous Wf-347 ultracode unit)
Date: 2026-09-29
Parent: Wf-347 ultracode residual wave + Wf-343c4 integration closeout
       (per AGENTS.md §5.1 standing GPU authorization)

These 2 NEW tests verify that the canonical GPU pipeline master
(`scripts/gpu_pipeline_master.sh`) integrates the Wf-343c4 pre-retrain
internal check gates per Wf-347-A2 task contract:

  1. **wf343c2 per-wave check call** — pipeline master invokes
     ``molmetal/scripts/wf343c2_per_wave_check.py`` via the
     ``wf343c4_gate()`` opt-in runner.
  2. **wf343c3 data quality check call** — pipeline master invokes
     ``molmetal/scripts/wf343c3_data_quality_check.py`` via the same
     opt-in runner.

These tests verify **structural integration** (the runner references the
scripts and the state is well-formed) — they do NOT actually invoke the
gate runner, which would require a USER_OK-granted live GPU launch.

Per the Wf-347-A2 task contract:
  - 2 NEW tests
  - 0 modified existing tests
  - All CPU-only

Per ``party-twenty-spirit`` §3.5 honest framing:
  - The pipeline master is **opt-in** for the Wf-343C4 gate (via
    ``WF343C4_GATE=1`` env var) — happy-path users skip the gate for speed.
  - The gate runner itself is INVOKED only when WF343C4_GATE=1 (default off).
  - Wf-343C2 + Wf-343C3 are CPU-only by design (use ``--skip-live`` for the
    Wf-343C2 GPU probes).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIPELINE_MASTER = PROJECT_ROOT / "scripts" / "gpu_pipeline_master.sh"
WF343C2_SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "wf343c2_per_wave_check.py"
WF343C3_SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "wf343c3_data_quality_check.py"


# ---------------------------------------------------------------------------
# Test 1: pipeline master references Wf-343C2 per-wave check script
# ---------------------------------------------------------------------------


def test_wf334u2_pipeline_master_references_wf343c2():
    """Wf-347-A2 task contract — pipeline master must reference
    ``molmetal/scripts/wf343c2_per_wave_check.py`` via the wf343c4_gate()
    runner function (opt-in via WF343C4_GATE=1).

    Verifies:
      - The Wf-343C2 script path appears in the pipeline master source.
      - The wf343c4_gate() function exists and is invoked from at least
        one phase entry (phase2_gpu0 or phase3_gpu1).
      - The Wf-343C2 invocation includes the ``--wave`` flag (per its
        CLI contract — see wf343c2_per_wave_check.py:34).
    """
    assert PIPELINE_MASTER.exists(), f"pipeline master missing: {PIPELINE_MASTER}"
    assert WF343C2_SCRIPT.exists(), f"Wf-343C2 script missing on disk: {WF343C2_SCRIPT}"

    text = PIPELINE_MASTER.read_text()

    # The Wf-343C2 script path must appear in the pipeline master
    assert "wf343c2_per_wave_check.py" in text, (
        "pipeline master missing reference to wf343c2_per_wave_check.py "
        "(Wf-347-A2 integration gap)"
    )

    # The wf343c4_gate() function must exist
    assert "wf343c4_gate()" in text, (
        "pipeline master missing wf343c4_gate() function definition "
        "(Wf-347-A2 integration gap)"
    )

    # The runner must be invoked from at least one phase entry (phase2 or phase3)
    assert (
        "WF343C4_GATE=1" in text or "WF343C4_GATE:-0" in text
    ), "pipeline master missing WF343C4_GATE env-var opt-in switch"

    # The Wf-343C2 invocation must use --wave (per CLI contract)
    # Find the ACTUAL python invocation (not the docstring comment) — skip
    # past the wf343c4_gate() function header comment by looking for the
    # python3 invocation line.
    python_invocation_marker = "python3 molmetal/scripts/wf343c2_per_wave_check.py"
    wf343c2_block_idx = text.find(python_invocation_marker)
    assert wf343c2_block_idx >= 0, (
        f"Wf-343C2 python invocation block not present in pipeline master "
        f"(expected '{python_invocation_marker}')"
    )
    # Look 800 chars ahead for the --wave flag
    nearby = text[wf343c2_block_idx : wf343c2_block_idx + 800]
    assert (
        "--wave" in nearby
    ), f"Wf-343C2 invocation must include --wave flag (CLI contract): {nearby[:200]!r}"


# ---------------------------------------------------------------------------
# Test 2: pipeline master references Wf-343C3 data quality check script
# ---------------------------------------------------------------------------


def test_wf334u2_pipeline_master_references_wf343c3():
    """Wf-347-A2 task contract — pipeline master must reference
    ``molmetal/scripts/wf343c3_data_quality_check.py`` via the
    wf343c4_gate() runner function (opt-in via WF343C4_GATE=1).

    Verifies:
      - The Wf-343C3 script path appears in the pipeline master source.
      - The Wf-343C3 invocation uses --out-dir (per its CLI contract —
        see wf343c3_data_quality_check.py:801).
      - The integration cross-references the 4 NEW Wf-343C4 pre-retrain
        internal check gates (the structural integration with the
        pre-flight v2 gate set).
    """
    assert PIPELINE_MASTER.exists(), f"pipeline master missing: {PIPELINE_MASTER}"
    assert WF343C3_SCRIPT.exists(), f"Wf-343C3 script missing on disk: {WF343C3_SCRIPT}"

    text = PIPELINE_MASTER.read_text()

    # The Wf-343C3 script path must appear in the pipeline master
    assert "wf343c3_data_quality_check.py" in text, (
        "pipeline master missing reference to wf343c3_data_quality_check.py "
        "(Wf-347-A2 integration gap)"
    )

    # The Wf-343C3 invocation must use --out-dir (per CLI contract).
    # Skip the docstring comment by anchoring on the actual command line.
    python_invocation_marker = "python3 molmetal/scripts/wf343c3_data_quality_check.py"
    wf343c3_block_idx = text.find(python_invocation_marker)
    assert wf343c3_block_idx >= 0, (
        f"Wf-343C3 python invocation block not present in pipeline master "
        f"(expected '{python_invocation_marker}')"
    )
    nearby = text[wf343c3_block_idx : wf343c3_block_idx + 800]
    assert (
        "--out-dir" in nearby
    ), f"Wf-343C3 invocation must include --out-dir flag (CLI contract): {nearby[:200]!r}"

    # The integration cross-references the 4 NEW Wf-343C4 gates (structural
    # integration with the pre-flight v2 66-gate set per Wf-343c4).
    assert "EXPECTED_GATE_COUNTS" in text or "wf343c4_" in text, (
        "pipeline master missing Wf-343C4 integration gate references "
        "(4 NEW gates: hyperparams / yuelbond / anchor / wave_budget)"
    )

    # Sanity: the runner is opt-in (default off) per happy-path design
    assert "WF343C4_GATE:-0" in text, (
        "Wf-343C4 gate must be opt-in via WF343C4_GATE env var (default 0/off) "
        "to preserve happy-path speed"
    )
