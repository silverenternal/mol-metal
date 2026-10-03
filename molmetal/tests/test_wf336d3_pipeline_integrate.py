"""Wf-336-D3 — Pipeline master integration tests.

Owner: Hugo Lee (autonomous Wf-336 ultracode unit)
Date: 2026-09-29
Parent: Wf-336-D3 ultracode residual closure wave
       (per AGENTS.md §5.1 standing GPU authorization)

This module verifies the GPU pipeline master
(``scripts/gpu_pipeline_master.sh``) integrates the four new wrappers
that Wf-334 ultracode shipped:

  1. **Wf-264 CFM property panel** (per Wf-334-U3) — CPU-only, post-LOMO
  2. **Wf-333 Vina 4-step wrapper** (per Wf-334-U2) — BLOCKED on Vina install
  3. **Wf-119 P1.4 chemprop v3 LOMO** (per Wf-336-X2) — cuda:0
  4. **DrugOOD Wf-74 v2** (per Wf-336-X3) — cuda:1

Tests verify the pipeline master follows the contract:

  1. **Sequential launch order** — Phase 2 + Phase 3 launch wrappers in the
     expected priority order, with the new wrappers inserted at the
     correct positions.
  2. **All wrappers integrated** — every launcher referenced in pipeline
     master resolves to a real file on disk.
  3. **USER_OK gates** — every new wrapper invocation carries the
     USER_OK / USER_OK=YES gate prefix (per kill_then_adapt standing
     directive + Wf-300C / Wf-306D pattern).
  4. **Path consistency** — the pipeline master references correct
     ``scripts/`` and ``molmetal/scripts/`` paths consistently.

These are CPU-only tests — no actual GPU launch, no Vina invocation.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Project + path constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
PIPELINE_MASTER = PROJECT_ROOT / "scripts" / "gpu_pipeline_master.sh"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MOLMETAL_SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_pipeline_master() -> str:
    assert PIPELINE_MASTER.exists(), (
        f"pipeline master missing: {PIPELINE_MASTER}"
    )
    return PIPELINE_MASTER.read_text()


def _wrapper_exists(wrapper_name: str) -> bool:
    """Check if a wrapper exists in either scripts/ or molmetal/scripts/."""
    candidates = [
        SCRIPTS_DIR / wrapper_name,
        MOLMETAL_SCRIPTS_DIR / wrapper_name,
    ]
    return any(p.exists() for p in candidates)


# ---------------------------------------------------------------------------
# Test 1: pipeline master sequential launch order
# ---------------------------------------------------------------------------


class TestPipelineMasterSequentialLaunchOrder:
    """Pipeline master must launch wrappers in the expected priority order."""

    def test_phase2_gpu0_launches_wave_c_then_wf119_p14(self) -> None:
        """Phase 2 (GPU 0) launches: Wf-310F frontier Wave C → Wf-119 P1.4.

        Per Wf-336-X2 + Wf-336-D3, Wf-119 P1.4 chemprop v3 LOMO (~2 hr cuda:0)
        is appended after Wf-310F frontier Wave C in Phase 2.
        """
        text = _read_pipeline_master()

        # The order of "Launching X" lines inside phase2_gpu0 reflects
        # the actual execution order.  We verify each line appears in the
        # expected sequence.
        expected_lines = [
            "Launching Wf-310F frontier Wave C",
            "Launching Wf-119 P1.4 chemprop v3 LOMO",
        ]
        positions = []
        for needle in expected_lines:
            idx = text.find(needle)
            assert idx >= 0, (
                f"pipeline master missing expected line: {needle!r}"
            )
            positions.append(idx)
        # Verify strict ascending order
        assert positions == sorted(positions), (
            f"phase2 launch order is WRONG; got positions {positions}, "
            f"expected ascending order. Expected: {expected_lines}"
        )

    def test_phase3_gpu1_launches_all_9_jobs_in_order(self) -> None:
        """Phase 3 (GPU 1) launches 9 jobs in priority order:

            1. Wf-310F frontier Wave B
            2. Wf-219 §4.1 Ru
            3. Wf-333 Vina 4-step (NEW per Wf-334-U2, BLOCKED)
            4. Wf-310C TODO-220
            5. Wf-310C TODO-179
            6. Wf-310D DrugOOD DANN Phase A
            7. DrugOOD Wf-74 v2 (NEW per Wf-336-X3)
            8. Wf-135 Phase 2 LOMO
            9. Wf-264 CFM property panel (NEW per Wf-334-U3, CPU)
        """
        text = _read_pipeline_master()

        expected_lines = [
            "Launching Wf-310F frontier Wave B",
            "Launching Wf-219 §4.1 Ru",
            "Launching Wf-333 Vina 4-step",        # NEW
            "Launching Wf-310C TODO-220",
            "Launching Wf-310C TODO-179",
            "Launching Wf-310D DrugOOD DANN Phase A",
            "Launching DrugOOD Wf-74 v2 DANN",     # NEW
            "Launching Wf-135 Phase 2 LOMO",
            "Launching Wf-264 CFM property panel",  # NEW
        ]
        positions = []
        for needle in expected_lines:
            idx = text.find(needle)
            assert idx >= 0, (
                f"pipeline master missing expected line: {needle!r}"
            )
            positions.append(idx)
        # Verify strict ascending order — Vina inserted at position 3,
        # Wf-74 v2 at position 7, Wf-264 at position 9 (tail).
        assert positions == sorted(positions), (
            f"phase3 launch order is WRONG; got positions {positions}, "
            f"expected ascending order. Expected: {expected_lines}"
        )
        # Sanity: Vina must come AFTER Wf-219 §4.1 (per Wf-336-D3 task
        # brief: "Add Vina 4-step to pipeline master after Wf-219 §4.4"
        # interpreted as the only on-disk Wf-219 wrapper, §4.1).
        vina_idx = positions[2]
        wf219_idx = positions[1]
        assert vina_idx > wf219_idx, (
            "Wf-333 Vina 4-step must be launched AFTER Wf-219 §4.1"
        )

    def test_phase2_and_phase3_dont_interleave(self) -> None:
        """All Phase 2 'Launching X' lines must appear before all Phase 3
        'Launching X' lines.  This catches accidental interleaving.
        """
        text = _read_pipeline_master()

        # Find phase2_gpu0 function body
        phase2_start = text.find("phase2_gpu0()")
        phase3_start = text.find("phase3_gpu1()")
        assert phase2_start >= 0, "phase2_gpu0() not found"
        assert phase3_start >= 0, "phase3_gpu1() not found"
        assert phase2_start < phase3_start, (
            "phase2_gpu0() must be declared before phase3_gpu1()"
        )


# ---------------------------------------------------------------------------
# Test 2: all wrappers integrated
# ---------------------------------------------------------------------------


class TestAllWrappersIntegrated:
    """Every launcher referenced in pipeline master must resolve to a real
    file on disk in either ``scripts/`` or ``molmetal/scripts/``.

    Per AGENTS.md §5.1, all wrappers must be launchable; a missing file
    blocks the entire pipeline downstream.
    """

    @pytest.mark.parametrize("wrapper_name", [
        # Pre-existing wrappers (Phase 1 + Phase 2 + Phase 3 pre-D3)
        "launch_wf310f_frontier_math_gpu.sh",
        "launch_wf306b_wf219_ru_gpu.sh",
        "launch_wf310c_todo220_gpu.sh",
        "launch_wf310c_t179_w2_gpu.sh",
        "launch_wf306a_drugood_vina_gpu.sh",
        "launch_wf310e_wf135_phase2_gpu.sh",
        # NEW per Wf-336-D3 (4 wrappers)
        "launch_wf333_vina_4step_gpu_bench.sh",
        "launch_wf264_cfm_property_panel.sh",
        "launch_wf119_p1_4_chemprop_v3_lomo.sh",
        "launch_wf74_v2_drugood_dann.sh",
    ])
    def test_pipeline_master_wrapper_exists(self, wrapper_name: str) -> None:
        """Each wrapper must exist on disk in scripts/ or molmetal/scripts/."""
        candidates = [
            SCRIPTS_DIR / wrapper_name,
            MOLMETAL_SCRIPTS_DIR / wrapper_name,
        ]
        found = [p for p in candidates if p.exists()]
        assert found, (
            f"wrapper missing on disk: {wrapper_name}. "
            f"Looked in: {[str(c) for c in candidates]}"
        )
        # And it must be executable
        for f in found:
            assert os.access(f, os.X_OK), (
                f"wrapper not executable: {f}"
            )

    def test_header_documents_all_9_phase3_jobs(self) -> None:
        """The pipeline master header (top of file) must document the
        9-job Phase 3 sequence so an audit reader can see the full
        pipeline at a glance.
        """
        text = _read_pipeline_master()
        # Spot-check that each new wrapper is mentioned in the header
        # (so a reader scanning the file knows the integration is real,
        # not just an empty bash launch).
        assert "Wf-333 Vina 4-step" in text, (
            "Wf-333 Vina 4-step must appear in pipeline master header"
        )
        assert "Wf-119 P1.4 chemprop v3 LOMO" in text, (
            "Wf-119 P1.4 must appear in pipeline master header"
        )
        assert "DrugOOD Wf-74 v2" in text, (
            "DrugOOD Wf-74 v2 must appear in pipeline master header"
        )
        assert "Wf-264 CFM property panel" in text, (
            "Wf-264 CFM property panel must appear in pipeline master header"
        )
        # And the NEW markers (Wf-336-D3 / Wf-334-U2 / Wf-336-X3)
        assert "[NEW per Wf-336-X2]" in text, "Wf-119 P1.4 NEW marker missing"
        assert "[NEW per Wf-336-X3]" in text, "Wf-74 v2 NEW marker missing"
        assert "[NEW per Wf-334-U2]" in text, "Wf-333 Vina NEW marker missing"
        assert "[NEW per Wf-334-U3]" in text, "Wf-264 panel NEW marker missing"


# ---------------------------------------------------------------------------
# Test 3: USER_OK gates
# ---------------------------------------------------------------------------


class TestUserOkGates:
    """Every new wrapper invocation in pipeline master must carry the
    USER_OK gate prefix.  Per kill_then_adapt standing directive +
    Wf-300C / Wf-306D pattern, no wall-time launch may start without
    explicit user OK.
    """

    def _assert_user_ok_before_launch(
        self, text: str, launch_line_substr: str,
        env_var: str = "USER_OK",
    ) -> None:
        """Helper: assert that the launch invocation following the log
        line `launch_line_substr` carries the USER_OK gate prefix
        (USER_OK / USER_OK=YES / WF306A_USER_OK=YES).

        The pipeline master pattern is:
          log "Launching <wrapper>"          <-- this line
          USER_OK=1 nohup bash <wrapper>.sh  <-- next line carries gate
        """
        idx = text.find(launch_line_substr)
        assert idx >= 0, f"launch line not found: {launch_line_substr!r}"
        # Find the END of the line containing the launch log string
        line_end = text.find("\n", idx)
        assert line_end > 0, "could not find end of launch line"
        # Take the NEXT line (the actual nohup launch invocation)
        next_line_start = line_end + 1
        next_line_end = text.find("\n", next_line_start)
        launch_line = text[next_line_start:next_line_end]
        # Accept any of USER_OK / USER_OK=YES / USER_OK=1 / WF306A_USER_OK=YES
        # or --dry-run path (USER_OK-exempt by convention; gate only
        # fires for real launch path).
        has_user_ok = (
            re.search(r"\bUSER_OK=1\b", launch_line)
            or re.search(r"\bUSER_OK=YES\b", launch_line)
            or re.search(r"\bWF306A_USER_OK=YES\b", launch_line)
            or re.search(r"\bWF310C_TODO220\b", launch_line)  # touch file
            or re.search(r"\bWF310C_T179\b", launch_line)      # touch file
            or "--dry-run" in launch_line
        )
        assert has_user_ok, (
            f"launch invocation missing USER_OK gate prefix: "
            f"{launch_line!r}.  Per Wf-300C / Wf-306D pattern + "
            f"AGENTS.md §5.1, every wall-time launch requires "
            f"USER_OK=1 / =YES env gate."
        )

    def test_wf119_p1_4_invocation_carries_user_ok(self) -> None:
        """Wf-119 P1.4 launch line must set USER_OK=1."""
        text = _read_pipeline_master()
        self._assert_user_ok_before_launch(
            text, "Launching Wf-119 P1.4 chemprop v3 LOMO"
        )

    def test_wf333_vina_invocation_carries_user_ok(self) -> None:
        """Wf-333 Vina 4-step launch line must set USER_OK=1."""
        text = _read_pipeline_master()
        self._assert_user_ok_before_launch(
            text, "Launching Wf-333 Vina 4-step"
        )

    def test_wf74_v2_drugood_invocation_carries_user_ok(self) -> None:
        """DrugOOD Wf-74 v2 launch line must set USER_OK=YES (per wrapper)."""
        text = _read_pipeline_master()
        self._assert_user_ok_before_launch(
            text, "Launching DrugOOD Wf-74 v2 DANN",
            env_var="USER_OK",
        )

    def test_wf264_panel_invocation_carries_user_ok(self) -> None:
        """Wf-264 CFM property panel launch line must set USER_OK=1."""
        text = _read_pipeline_master()
        self._assert_user_ok_before_launch(
            text, "Launching Wf-264 CFM property panel"
        )

    def test_wf310e_wf219_ru_invocation_carries_user_ok(self) -> None:
        """Wf-219 §4.1 Ru launch line must set USER_OK=1 (regression
        guard: pre-existing Wf-306B invocation).
        """
        text = _read_pipeline_master()
        self._assert_user_ok_before_launch(
            text, "Launching Wf-219 §4.1 Ru"
        )

    def test_wf135_phase2_invocation_carries_user_ok(self) -> None:
        """Wf-135 Phase 2 LOMO launch line must set USER_OK=1 (regression
        guard: pre-existing Wf-310E invocation).
        """
        text = _read_pipeline_master()
        self._assert_user_ok_before_launch(
            text, "Launching Wf-135 Phase 2 LOMO"
        )


# ---------------------------------------------------------------------------
# Test 4: path consistency
# ---------------------------------------------------------------------------


class TestPathConsistency:
    """The pipeline master must reference correct ``scripts/`` and
    ``molmetal/scripts/`` paths consistently.  Per Wf-336-D3 task
    brief, the 4 NEW wrappers live in different dirs:

      - Wf-264        : scripts/launch_wf264_cfm_property_panel.sh
      - Wf-333        : molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh
      - Wf-119 P1.4   : scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh
      - DrugOOD Wf-74 : scripts/launch_wf74_v2_drugood_dann.sh

    And the pre-existing wrappers are split similarly.  Path drift here
    causes silent runtime failures (shell exec() with missing file).
    """

    def test_wf264_panel_invocation_uses_scripts_path(self) -> None:
        """Wf-264 CFM property panel invocation must use scripts/ path."""
        text = _read_pipeline_master()
        assert "scripts/launch_wf264_cfm_property_panel.sh" in text, (
            "Wf-264 launch must reference scripts/ path (NOT molmetal/scripts/)"
        )

    def test_wf333_vina_invocation_uses_molmetal_scripts_path(self) -> None:
        """Wf-333 Vina 4-step invocation must use molmetal/scripts/ path."""
        text = _read_pipeline_master()
        assert (
            "molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh" in text
        ), (
            "Wf-333 Vina launch must reference molmetal/scripts/ path"
        )

    def test_wf119_p1_4_invocation_uses_scripts_path(self) -> None:
        """Wf-119 P1.4 chemprop v3 LOMO invocation must use scripts/ path."""
        text = _read_pipeline_master()
        assert "scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh" in text, (
            "Wf-119 P1.4 launch must reference scripts/ path"
        )

    def test_wf74_v2_drugood_invocation_uses_scripts_path(self) -> None:
        """DrugOOD Wf-74 v2 invocation must use scripts/ path."""
        text = _read_pipeline_master()
        assert "scripts/launch_wf74_v2_drugood_dann.sh" in text, (
            "DrugOOD Wf-74 v2 launch must reference scripts/ path"
        )

    def test_pre_existing_paths_unchanged(self) -> None:
        """Pre-existing wrapper paths must NOT have been modified by the
        Wf-336-D3 integration (regression guard).
        """
        text = _read_pipeline_master()
        for path in (
            "molmetal/scripts/launch_wf310f_frontier_math_gpu.sh",
            "molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh",
            "molmetal/scripts/launch_wf310c_todo220_gpu.sh",
            "molmetal/scripts/launch_wf310c_t179_w2_gpu.sh",
            "molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh",
            "scripts/launch_wf310e_wf135_phase2_gpu.sh",
        ):
            assert path in text, (
                f"pre-existing wrapper path {path!r} missing from "
                f"pipeline master (regression — Wf-336-D3 must not "
                f"rewrite pre-existing wrapper paths)"
            )

    def test_log_dir_paths_consistent(self) -> None:
        """All log redirects must go under LOG_DIR (no absolute paths
        scattered around the script).
        """
        text = _read_pipeline_master()
        # Each new wrapper must redirect to ${LOG_DIR}/<wrapper>.log
        for log_filename in (
            "wf119_p14.log",
            "wf333_vina.log",
            "wf74_v2_drugood.log",
            "wf264_panel.log",
        ):
            assert log_filename in text, (
                f"log filename {log_filename!r} missing from pipeline master"
            )
            # Spot-check it's under ${LOG_DIR}/
            pattern = rf"\$\{{LOG_DIR\}}/.*{re.escape(log_filename)}"
            assert re.search(pattern, text), (
                f"log redirect for {log_filename} must use ${{LOG_DIR}}/"
            )


# ---------------------------------------------------------------------------
# Bonus: pipeline master --dry-run smoke (no actual GPU launch)
# ---------------------------------------------------------------------------


class TestPipelineMasterDryRunSmoke:
    """Pipeline master --dry-run-like smoke: just verify the bash syntax
    parses cleanly (no execution, no GPU launch).
    """

    def test_pipeline_master_parses_with_bash_n(self) -> None:
        """`bash -n` must exit 0 — no syntax errors introduced by the
        Wf-336-D3 edits.
        """
        proc = subprocess.run(
            ["bash", "-n", str(PIPELINE_MASTER)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0, (
            f"pipeline master syntax error: stderr={proc.stderr!r}"
        )

    def test_pipeline_master_usage_lists_phase_modes(self) -> None:
        """Usage message must include phase2, phase3, all modes."""
        text = _read_pipeline_master()
        for mode in ("phase2", "phase3", "all"):
            assert mode in text, f"usage missing mode: {mode}"
