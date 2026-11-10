"""Wf-341-P1 — Pipeline master integration tests (5 launchers + G2/G4/G5/G6 parity).

Owner: Hugo Lee (autonomous Wf-341 ultracode unit)
Date: 2026-09-29
Parent: Wf-341-P1 ultracode pipeline master integration test
       (per Wf-334-U2 + Wf-336-D3 + Wf-337-G2 + Wf-340-R2 ultracode chain)

This module verifies the GPU pipeline master
(``scripts/gpu_pipeline_master.sh``) integrates the 5 launcher scripts
that the Wf-124 / Wf-264 / Wf-333 ultracode wave shipped:

  1. **Wf-124 P3 G4 launcher** (per Wf-337-G2) — 14 (metal, seed) pairs
  2. **Wf-124 P3 G5 launcher** (per Wf-337-G2) — 15 (metal, seed) pairs
  3. **Wf-124 P3 G6 launcher** (per Wf-337-G2) — 60 (config, metal, seed)
  4. **Wf-264 CFM property panel** (per Wf-334-U3) — CPU-only, post-LOMO
  5. **Wf-333 Vina 4-step wrapper** (per Wf-334-U2) — BLOCKED on Vina install

Tests verify the pipeline master follows the contract:

  1. **All 5 launchers integrated** — every launcher either (a) directly
     invoked in the master, or (b) referenced in PHASE 1 in-flight comments,
     or (c) referenced via standalone wrapper (G4/G5/G6 are standalone
     per Wf-337-G2 design). All 5 wrappers must resolve to real files on disk.
  2. **Sequential launch order** — Phase 2 (GPU 0) + Phase 3 (GPU 1) launch
     wrappers in the expected priority order, with the new wrappers inserted
     at the correct positions.
  3. **USER_OK gates** — every wall-time wrapper invocation carries the
     USER_OK / USER_OK=YES / WF306A_USER_OK=YES / touch-file gate prefix
     (per kill_then_adapt standing directive + Wf-300C / Wf-306D pattern).
  4. **Backward compat** — pre-existing wrapper paths (Wf-310F / Wf-306B /
     Wf-310C / Wf-310E) unchanged; G4/G5/G6 wrappers + Wf-264 + Wf-333
     wrappers all parse cleanly under `bash -n`.

These are CPU-only tests — no actual GPU launch, no Vina invocation.
< 5 s wall.

Honest framing (AGENTS.md §5.2 + party-twenty-spirit §3.5):
  - MEASURED: pipeline master wires Wf-333 Vina + Wf-264 panel + G5 + G6
    per Wf-334-U2 + Wf-336-D3 + Wf-337-G2.
  - MEASURED: G4 launcher exists and parses cleanly; G4 is intentionally
    not invoked in the master (TODO-180 closed per Wf-124 ship wave).
  - PROJECTED: future launches will use the consolidated pipeline master.
  - HONEST_NEGATIVE: Vina 1.2.7 conda install BLOCKED per AGENTS.md §5.4.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project + path constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_MASTER = PROJECT_ROOT / "scripts" / "gpu_pipeline_master.sh"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MOLMETAL_SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
# Wf-A269 wave-3 (2026-11-02) — the Wf-437 simplification wave archived two
# of the five launchers this file pins under this directory.  Three other
# committed tests require the SAME paths NOT to exist under scripts/:
#   molmetal/tests/test_wf337g2_launcher_v2.py:176-185
#   molmetal/tests/test_wf327_ultracode_closeout.py:66,96
#   molmetal/tests/test_wf333t3_trainer_verify.py:278,304
# The archive is therefore canonical for those two, and this file's
# existence helper must look there too.
ARCHIVE_DIR = PROJECT_ROOT / "TODO" / "archive" / "wf437_simplify"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# The 5 launchers verified by this module
# ---------------------------------------------------------------------------

# Per Wf-341-P1 task brief: 5 launchers that must be either directly
# invoked in the pipeline master, or referenced via PHASE 1 in-flight
# comments, or exist as standalone wrappers on disk (G4/G5/G6 standalone
# per Wf-337-G2 design).
LAUNCHER_G4 = "launch_wf124_p3_g4.sh"  # TODO-180 (Wf-124 P3 G4) — 14 pairs
LAUNCHER_G5 = "launch_wf124_p3_g5.sh"  # TODO-181 (Wf-124 P3 G5) — 15 pairs
LAUNCHER_G6 = "launch_wf124_p3_g6.sh"  # TODO-182 (Wf-124 P3 G6) — 60 tuples
LAUNCHER_WF264 = "launch_wf264_cfm_property_panel.sh"  # Wf-264 CFM property panel (CPU)
LAUNCHER_WF333 = "launch_wf333_vina_4step_gpu_bench.sh"  # Wf-333 Vina 4-step (BLOCKED)
ALL_5_LAUNCHERS = [
    LAUNCHER_G4,
    LAUNCHER_G5,
    LAUNCHER_G6,
    LAUNCHER_WF264,
    LAUNCHER_WF333,
]

# Wf-A269 wave-3 (2026-11-02) — the ``--w2-gradient-mode`` value each
# launcher is EXPECTED to carry.  MEASURED from the launchers' own headers:
#   - G4 uses ``sort``  — "Per Wf-328-W5: --w2-gradient-mode sort (Wf-326A
#     MEASURED 2-3x speedup ...)"; G4 is archived and is NOT covered by the
#     authoritative ``test_wf334u4_launcher_verify.py``.
#   - G5/G6 use ``sliced`` — "Per Wf-328-W5 §Integration points: G5/G6 keep
#     --w2-gradient-mode sliced ... compared bit-exact against the Wf-131 P1
#     baseline in the paper §6 reproducibility audit (bc29aa90 flipped this
#     to `sort` in an unrelated venv-path commit; reverted 2026-10-20)".
#     ``test_wf334u4_launcher_verify.py`` asserts exactly this, so this map
#     and that guard agree instead of contradicting each other.
EXPECTED_W2_GRADIENT_MODE = {
    LAUNCHER_G4: "sort",
    LAUNCHER_G5: "sliced",
    LAUNCHER_G6: "sliced",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_pipeline_master() -> str:
    assert PIPELINE_MASTER.exists(), f"pipeline master missing: {PIPELINE_MASTER}"
    return PIPELINE_MASTER.read_text()


def _wrapper_exists(wrapper_name: str) -> list[Path]:
    '''Return all on-disk paths for a wrapper.

    Wf-A269 wave-3 (2026-11-02) — the archive home is now searched too.
    Before this, this file asserted that ``scripts/launch_wf124_p3_g4.sh``
    EXISTS while three other committed tests asserted that the SAME path
    does NOT exist; the contradiction is resolved in favour of the archive
    (the repo convention, and the one the other three tests enforce).

    The outer docstring delimiter is THREE SINGLE QUOTES and NOT three
    double quotes, on purpose: the SUPERSEDED RENDERING below carries the
    old function's own double-quoted docstring, and nesting three double
    quotes inside a three-double-quote docstring closes the outer one early
    (SyntaxError at the preserved line).  Three single quotes keep the
    preserved rendering's SOURCE BYTES byte-identical, which is what
    AGENTS.md §5.5 rule #6 asks for.

    SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
      """Return all on-disk paths for a wrapper (in scripts/ or molmetal/scripts/)."""
      candidates = [
          SCRIPTS_DIR / wrapper_name,
          MOLMETAL_SCRIPTS_DIR / wrapper_name,
      ]
    '''
    candidates = [
        SCRIPTS_DIR / wrapper_name,
        MOLMETAL_SCRIPTS_DIR / wrapper_name,
        ARCHIVE_DIR / wrapper_name,
    ]
    return [p for p in candidates if p.exists()]


def _parse_cleanly(path: Path) -> bool:
    """Return True if `bash -n <path>` exits 0 (no syntax errors)."""
    proc = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.returncode == 0


# ===========================================================================
# Test 1: pipeline master has all 5 launchers
# ===========================================================================


class TestPipelineMasterHasAll5Launchers:
    """Pipeline master must integrate (or reference) all 5 launchers."""

    @pytest.mark.parametrize("wrapper_name", ALL_5_LAUNCHERS)
    def test_each_launcher_exists_on_disk(self, wrapper_name: str) -> None:
        """Every one of the 5 launchers must resolve to a real file on disk.

        Per AGENTS.md §5.1, all wrappers must be launchable; a missing file
        blocks the entire pipeline downstream.
        """
        found = _wrapper_exists(wrapper_name)
        assert found, (
            f"launcher missing on disk: {wrapper_name}. "
            f"Looked in: {[str(SCRIPTS_DIR / wrapper_name), str(MOLMETAL_SCRIPTS_DIR / wrapper_name)]}"
        )
        # And it must be executable
        for f in found:
            assert os.access(f, os.X_OK), f"launcher not executable: {f}"

    @pytest.mark.parametrize("wrapper_name", ALL_5_LAUNCHERS)
    def test_each_launcher_parses_cleanly(self, wrapper_name: str) -> None:
        """Every one of the 5 launchers must parse cleanly under `bash -n`.

        Regression guard against syntax errors introduced by the Wf-337-G2
        consolidation wave.
        """
        found = _wrapper_exists(wrapper_name)
        assert found, f"launcher missing: {wrapper_name}"
        for f in found:
            assert _parse_cleanly(f), f"launcher has bash syntax errors: {f}"

    def test_pipeline_master_references_g5_and_g6_in_phase1(self) -> None:
        """Pipeline master PHASE 1 comment block must reference G5 + G6
        (per Wf-337-G2 design — G5/G6 are PHASE 1 in-flight launches,
        not directly invoked by the master).
        """
        text = _read_pipeline_master()
        # G5 retrain retry is PHASE 1 in-flight on GPU 0
        assert (
            "G5 retrain retry" in text or "TODO-181" in text
        ), "pipeline master PHASE 1 must reference G5 / TODO-181"
        # Wf-124 G6 60-tuple sweep is PHASE 1 in-flight on GPU 1
        assert (
            "Wf-124 G6" in text or "TODO-182" in text
        ), "pipeline master PHASE 1 must reference Wf-124 G6 / TODO-182"

    def test_pipeline_master_directly_invokes_wf333_and_wf264(self) -> None:
        """Pipeline master must DIRECTLY invoke Wf-333 + Wf-264 in Phase 3.

        Per Wf-334-U2 + Wf-334-U3:
          - Wf-333 Vina 4-step → phase3_gpu1 Job 3
          - Wf-264 CFM property panel → phase3_gpu1 Job 9
        """
        text = _read_pipeline_master()
        # Wf-333 direct invocation (in phase3_gpu1)
        assert (
            "launch_wf333_vina_4step_gpu_bench.sh" in text
        ), "pipeline master must directly invoke Wf-333 Vina 4-step"
        assert (
            "scripts/launch_wf333_vina_4step_gpu_bench.sh" in text
        ), "Wf-333 invocation must use a path that resolves on disk"
        # Wf-A269 wave-3 (2026-11-02) — the message above claims the path
        # RESOLVES, but the assertion was a bare substring test that never
        # touched the filesystem.  Make the claim true: resolve it.
        assert (
            SCRIPTS_DIR / "launch_wf333_vina_4step_gpu_bench.sh"
        ).is_file(), "Wf-333 invocation must use a path that resolves on disk"
        # Wf-264 direct invocation (in phase3_gpu1)
        assert (
            "launch_wf264_cfm_property_panel.sh" in text
        ), "pipeline master must directly invoke Wf-264 panel"
        assert (
            "scripts/launch_wf264_cfm_property_panel.sh" in text
        ), "Wf-264 invocation must use scripts/ path"

    def test_pipeline_master_g4_not_invoked_but_standalone(self) -> None:
        """Wf-124 P3 G4 launcher must NOT be directly invoked by the master
        (TODO-180 closed per Wf-124 ship wave; G4 is a standalone launcher
        for the 14-pair sweep that already completed per Wf-124 P3 §4).

        This is intentional per Wf-337-G2 design — G4 exists on disk as a
        standalone wrapper for re-runs, not as a queued GPU job.
        """
        # G4 wrapper must exist on disk
        g4_paths = _wrapper_exists(LAUNCHER_G4)
        assert g4_paths, "G4 launcher must exist on disk as standalone"

        # But it must NOT be invoked in the pipeline master (master
        # launches G5/G6/Wf-264/Wf-333 — not G4)
        text = _read_pipeline_master()
        assert "launch_wf124_p3_g4.sh" not in text, (
            "G4 launcher must NOT be invoked in pipeline master "
            "(standalone only; G5+G6 are the queued GPU jobs)"
        )

    def test_pipeline_master_total_direct_invocations(self) -> None:
        """Pipeline master must have ≥ 10 direct nohup bash invocations
        (2 in Phase 2 + 11 in Phase 3 per Wf-337-G2 + Wf-338-F1 final
        closure of 13 total launcher references).

        Regression guard: catches accidental wrapper removal.
        """
        text = _read_pipeline_master()
        nohup_count = len(re.findall(r"nohup bash\s+", text))
        assert nohup_count >= 10, (
            f"pipeline master has only {nohup_count} nohup bash invocations; "
            f"expected ≥ 10 (2 in Phase 2 + 11 in Phase 3). "
            f"This catches accidental wrapper removal."
        )


# ===========================================================================
# Test 2: sequential launch order
# ===========================================================================


class TestSequentialLaunchOrder:
    """Pipeline master must launch wrappers in the expected priority order."""

    def test_phase2_gpu0_launches_wave_c_then_wf119_p14(self) -> None:
        """Phase 2 (GPU 0) launches: Wf-310F frontier Wave C → Wf-119 P1.4."""
        text = _read_pipeline_master()
        expected_lines = [
            "Launching Wf-310F frontier Wave C",
            "Launching Wf-119 P1.4 chemprop v3 LOMO",
        ]
        positions = []
        for needle in expected_lines:
            idx = text.find(needle)
            assert idx >= 0, f"pipeline master missing line: {needle!r}"
            positions.append(idx)
        assert positions == sorted(
            positions
        ), f"phase2 launch order is WRONG; got positions {positions}"

    def test_phase3_gpu1_launches_11_jobs_in_priority_order(self) -> None:
        '''Phase 3 (GPU 1) launches 11 jobs in priority order:

         1. Wf-310F frontier Wave B
         2. Wf-219 §4.1 Ru
         3. Wf-333 Vina 4-step        (NEW per Wf-334-U2, BLOCKED)
         4. Wf-310C TODO-220
         5. Wf-310C TODO-179
         6. Wf-310D DrugOOD DANN Phase A
         7. DrugOOD Wf-74 v2 DANN     (NEW per Wf-336-X3)
         8. Wf-135 Phase 2 LOMO
         9. Wf-264 CFM property panel (NEW per Wf-334-U3, CPU)
        10. Wf-264 CFM panel RUN-ONCE (NEW per Wf-338-F1)
        11. W₂ sliced GPU smoke       (NEW per Wf-338-F1)

        Wf-A269 wave-3 (2026-11-02) — CORRECTION: the last expected line
        names ``cuda:0``, not ``cuda:1``.  Commit 87b026ee ("one resolver,
        one device — everything targets GPU 0") deliberately moved every
        job onto GPU 0; the pipeline master carries NO ``CUDA_VISIBLE_
        DEVICES`` pinning at all (measured: 0 hits), so the launchers pick
        the device.  The ``cuda:1`` expectation was written in bc29aa90 and
        was never updated after 87b026ee.  The function is still NAMED
        ``phase3_gpu1`` — that naming is stale, and it is left as-is here
        because renaming it would touch the pipeline master's dispatch
        surface (out of scope for a test-only repair).

        SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
            "Launching W₂ sliced GPU smoke on cuda:1",
        '''
        text = _read_pipeline_master()
        expected_lines = [
            "Launching Wf-310F frontier Wave B",
            "Launching Wf-219 §4.1 Ru",
            "Launching Wf-333 Vina 4-step",
            "Launching Wf-310C TODO-220",
            "Launching Wf-310C TODO-179",
            "Launching Wf-310D DrugOOD DANN Phase A",
            "Launching DrugOOD Wf-74 v2 DANN",
            "Launching Wf-135 Phase 2 LOMO",
            "Launching Wf-264 CFM property panel",
            "Launching Wf-264 CFM property panel RUN-ONCE",
            "Launching W₂ sliced GPU smoke on cuda:0",
        ]
        positions = []
        for needle in expected_lines:
            idx = text.find(needle)
            assert idx >= 0, f"pipeline master missing line: {needle!r}"
            positions.append(idx)
        assert positions == sorted(
            positions
        ), f"phase3 launch order is WRONG; got positions {positions}"

    def test_phase2_runs_before_phase3(self) -> None:
        """All Phase 2 'Launching X' lines must appear before all Phase 3
        'Launching X' lines (sequential dispatch)."""
        text = _read_pipeline_master()
        phase2_start = text.find("phase2_gpu0()")
        phase3_start = text.find("phase3_gpu1()")
        assert phase2_start >= 0, "phase2_gpu0() not found"
        assert phase3_start >= 0, "phase3_gpu1() not found"
        assert (
            phase2_start < phase3_start
        ), "phase2_gpu0() must be declared before phase3_gpu1()"

    def test_wf264_panel_runs_after_wf135_phase2(self) -> None:
        """Wf-264 panel must run AFTER Wf-135 Phase 2 LOMO (per Wf-334-U3
        design: panel is CPU-only at the tail of Phase 3 to avoid GPU
        contention with the 14-day LOMO job).
        """
        text = _read_pipeline_master()
        wf135_idx = text.find("Launching Wf-135 Phase 2 LOMO")
        wf264_idx = text.find("Launching Wf-264 CFM property panel")
        assert wf135_idx >= 0, "Wf-135 Phase 2 launch line missing"
        assert wf264_idx >= 0, "Wf-264 panel launch line missing"
        assert wf264_idx > wf135_idx, (
            "Wf-264 panel must run AFTER Wf-135 Phase 2 LOMO "
            "(per Wf-334-U3 design: post-LOMO at the tail)"
        )

    def test_wf333_vina_runs_after_wf219_ru(self) -> None:
        """Wf-333 Vina must run AFTER Wf-219 §4.1 Ru (per Wf-336-D3 task
        brief: Vina inserted after the only on-disk Wf-219 wrapper).
        """
        text = _read_pipeline_master()
        wf219_idx = text.find("Launching Wf-219 §4.1 Ru")
        wf333_idx = text.find("Launching Wf-333 Vina 4-step")
        assert wf219_idx >= 0
        assert wf333_idx >= 0
        assert wf333_idx > wf219_idx, "Wf-333 Vina must run AFTER Wf-219 §4.1 Ru"


# ===========================================================================
# Test 3: USER_OK gates
# ===========================================================================


class TestUserOkGates:
    """Every wall-time wrapper invocation must carry the USER_OK gate prefix.

    Per kill_then_adapt standing directive + Wf-300C / Wf-306D pattern,
    no wall-time launch may start without explicit user OK.
    """

    def _assert_user_ok_or_touch_file(self, text: str, launch_line_substr: str) -> None:
        """Assert the launch invocation after `launch_line_substr` carries
        a USER_OK gate (USER_OK=1 / =YES / WF306A_USER_OK=YES) or a touch-file
        gate (WF310C_TODO220 / WF310C_T179)."""
        idx = text.find(launch_line_substr)
        assert idx >= 0, f"launch line not found: {launch_line_substr!r}"
        line_end = text.find("\n", idx)
        assert line_end > 0
        next_line_start = line_end + 1
        next_line_end = text.find("\n", next_line_start)
        launch_line = text[next_line_start:next_line_end]
        has_gate = (
            re.search(r"\bUSER_OK=1\b", launch_line)
            or re.search(r"\bUSER_OK=YES\b", launch_line)
            or re.search(r"\bWF306A_USER_OK=YES\b", launch_line)
            or re.search(
                r"\bWF310F_NEXT_WAVE=", launch_line
            )  # wave flag is its own gate
            or re.search(r"\btouch /tmp/USER_OK_WF310C_TODO220\b", text)
            or re.search(r"\btouch /tmp/USER_OK_WF310C_T179\b", text)
        )
        assert (
            has_gate
        ), f"launch invocation missing USER_OK gate prefix: {launch_line!r}"

    @pytest.mark.parametrize(
        "launch_line_substr",
        [
            "Launching Wf-119 P1.4 chemprop v3 LOMO",
            "Launching Wf-219 §4.1 Ru",
            "Launching Wf-333 Vina 4-step",
            "Launching DrugOOD Wf-74 v2 DANN",
            "Launching Wf-135 Phase 2 LOMO",
            "Launching Wf-264 CFM property panel",
        ],
    )
    def test_user_ok_gate_present(self, launch_line_substr: str) -> None:
        """Every wall-time wrapper invocation in Phase 2 + Phase 3 must
        carry the USER_OK gate prefix."""
        text = _read_pipeline_master()
        self._assert_user_ok_or_touch_file(text, launch_line_substr)

    def test_touch_file_gates_for_wf310c(self) -> None:
        """Wf-310C TODO-220 + TODO-179 must use touch-file USER_OK gates
        (per Wf-300C pattern — touch /tmp/USER_OK_WF310C_T179 before
        invoking the wrapper).
        """
        text = _read_pipeline_master()
        assert (
            "touch /tmp/USER_OK_WF310C_TODO220" in text
        ), "TODO-220 must use touch-file USER_OK gate"
        assert (
            "touch /tmp/USER_OK_WF310C_T179" in text
        ), "TODO-179 must use touch-file USER_OK gate"

    def test_wf333_user_ok_required_per_wrapper(self) -> None:
        """Wf-333 Vina 4-step wrapper must REFUSE without USER_OK=1
        (per Wf-334-U2 audit — gate enforced by wrapper itself, exit 64).
        """
        # Wf-A269 wave-3 (2026-11-02) — CORRECTION: this composed
        # reference named ``molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh``
        # — a path that does NOT exist (the launcher is at ``scripts/``), so
        # the test asserted on a MISSING file and failed for the wrong
        # reason.  The wave-3 launcher move put it at ``scripts/``.
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
        #   wrapper = MOLMETAL_SCRIPTS_DIR / LAUNCHER_WF333
        wrapper = SCRIPTS_DIR / LAUNCHER_WF333
        assert wrapper.exists(), f"Wf-333 wrapper missing: {wrapper}"
        env = {**os.environ, "USER_OK": ""}
        proc = subprocess.run(
            [str(wrapper), "--dry-run"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 64, (
            f"Wf-333 wrapper should refuse without USER_OK=1, "
            f"got exit {proc.returncode}"
        )

    def test_wf264_user_ok_required_per_wrapper(self) -> None:
        """Wf-264 panel wrapper must REFUSE without USER_OK=1 on the
        REAL launch path (dry-run path is exempt by convention).

        Per Wf-334-U3 + Wf-306D, the wrapper checks USER_OK gate BEFORE
        any pre-flight / real-run work.
        """
        wrapper = SCRIPTS_DIR / LAUNCHER_WF264
        assert wrapper.exists(), f"Wf-264 wrapper missing: {wrapper}"
        # Invoke WITHOUT --dry-run and WITHOUT USER_OK=1 → must refuse (exit 1)
        env = {**os.environ, "USER_OK": ""}
        proc = subprocess.run(
            [str(wrapper)],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode != 0, (
            f"Wf-264 wrapper should refuse without USER_OK=1, "
            f"got exit {proc.returncode}"
        )
        # Stderr/stdout must contain USER_OK gate message
        combined = proc.stdout + proc.stderr
        assert (
            "USER_OK" in combined
        ), f"Wf-264 wrapper missing USER_OK gate message: {combined[:500]}"

    def test_g6_user_ok_required_per_wrapper(self) -> None:
        """Wf-124 P3 G6 wrapper must REFUSE without USER_OK=1 on the
        REAL launch path (dry-run path is exempt by convention).

        Per Wf-306D, this 7.5 hr GPU launch is USER_OK-gated.
        """
        wrapper = SCRIPTS_DIR / LAUNCHER_G6
        assert wrapper.exists(), f"G6 wrapper missing: {wrapper}"
        env = {**os.environ, "USER_OK": ""}
        proc = subprocess.run(
            [str(wrapper)],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode != 0, (
            f"G6 wrapper should refuse without USER_OK=1, "
            f"got exit {proc.returncode}"
        )
        combined = proc.stdout + proc.stderr
        assert (
            "USER_OK" in combined
        ), f"G6 wrapper missing USER_OK gate message: {combined[:500]}"


# ===========================================================================
# Test 4: backward compat
# ===========================================================================


class TestBackwardCompat:
    """Pre-existing wrappers unchanged + G4/G5/G6 wrappers + Wf-264 + Wf-333
    wrappers all parse cleanly under `bash -n` (no syntax regressions from
    the Wf-334 + Wf-336 + Wf-337 + Wf-338 + Wf-341 ultracode wave)."""

    def test_pre_existing_wrapper_paths_unchanged(self) -> None:
        """Pre-existing wrapper paths in pipeline master must NOT have been
        modified by the Wf-341 ultracode integration (regression guard).
        """
        text = _read_pipeline_master()
        for path in (
            "scripts/launch_wf310f_frontier_math_gpu.sh",
            "scripts/launch_wf306b_wf219_ru_gpu.sh",
            "scripts/launch_wf310c_todo220_gpu.sh",
            "scripts/launch_wf310c_t179_w2_gpu.sh",
            "scripts/launch_wf306a_drugood_vina_gpu.sh",
            "scripts/launch_wf310e_wf135_phase2_gpu.sh",
        ):
            assert path in text, (
                f"pre-existing wrapper path {path!r} missing from pipeline "
                f"master (regression — Wf-341 must not rewrite pre-existing "
                f"wrapper paths)"
            )

    @pytest.mark.parametrize(
        "wrapper_name",
        [
            "launch_wf310f_frontier_math_gpu.sh",
            "launch_wf306b_wf219_ru_gpu.sh",
            "launch_wf310c_todo220_gpu.sh",
            "launch_wf310c_t179_w2_gpu.sh",
            "launch_wf306a_drugood_vina_gpu.sh",
            "launch_wf310e_wf135_phase2_gpu.sh",
        ],
    )
    def test_pre_existing_wrappers_parse_cleanly(self, wrapper_name: str) -> None:
        """Every pre-existing wrapper must still parse cleanly under `bash -n`
        (regression guard against accidental breakage).
        """
        found = _wrapper_exists(wrapper_name)
        assert found, f"pre-existing wrapper missing: {wrapper_name}"
        for f in found:
            assert _parse_cleanly(f), f"pre-existing wrapper has syntax errors: {f}"

    def test_pipeline_master_parses_cleanly(self) -> None:
        """Pipeline master itself must parse cleanly under `bash -n`."""
        assert _parse_cleanly(
            PIPELINE_MASTER
        ), f"pipeline master has syntax errors: {PIPELINE_MASTER}"

    def test_pipeline_master_cli_supports_phase_modes(self) -> None:
        """Pipeline master CLI must support phase2|phase3|all modes."""
        text = _read_pipeline_master()
        for mode in ("phase2", "phase3", "all"):
            assert mode in text, f"usage missing mode: {mode}"
        # And must use strict mode
        assert (
            "set -uo pipefail" in text or "set -euo pipefail" in text
        ), "pipeline master must use strict mode"

    def test_wf337_g2_fixes_wired_into_g4_g5_g6(self) -> None:
        '''Per Wf-337-G2, G4 + G5 + G6 launchers must all carry the
        consolidated Wf-328 + Wf-335 fixes:
          - --cpu-num-workers 8 (Wf-328-W2)
          - --num-workers 8 (Wf-328-W6)
          - --w2-gradient-mode <value> (Wf-328-W5)
          - --no-cuda-graphs (Wf-335-R3)

        Backward-compat guard: if Wf-341 ultracode rewires any of these,
        all 3 launchers must reflect the change.

        Wf-A269 wave-3 (2026-11-02) — CORRECTION: the ``--w2-gradient-mode``
        value is now asserted PER LAUNCHER, not as a blanket ``sort``.

        MEASURED root cause: this test was written in bc29aa90, the same
        commit that flipped G5/G6 to ``sort``; that launcher flip was
        REVERTED on 2026-10-20 (the launchers' own headers record it:
        "G5/G6 keep --w2-gradient-mode sliced ... compared bit-exact
        against the Wf-131 P1 baseline").  The test kept the pre-revert
        expectation and contradicted the AUTHORITATIVE guard
        ``molmetal/tests/test_wf334u4_launcher_verify.py`` (which asserts
        ``--w2-gradient-mode sliced`` for G5/G6 and
        ``DEFAULT_W2_GRADIENT_MODE == "sliced"``).

        The per-launcher map is STRICTLY STRONGER than the blanket
        assertion: it pins the exact value, so a silent flip of G5/G6 back
        to ``sort`` — which is what bc29aa90 did — still fails here.

        SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
          - --w2-gradient-mode sort (Wf-328-W5)
        ...
                assert (
                    "--w2-gradient-mode sort" in text
                ), f"{wrapper_name} missing --w2-gradient-mode sort (Wf-328-W5)"
        '''
        for wrapper_name in (LAUNCHER_G4, LAUNCHER_G5, LAUNCHER_G6):
            paths = _wrapper_exists(wrapper_name)
            assert paths, f"launcher missing: {wrapper_name}"
            expected_w2 = EXPECTED_W2_GRADIENT_MODE[wrapper_name]
            for p in paths:
                text = p.read_text()
                assert (
                    "--cpu-num-workers 8" in text
                ), f"{wrapper_name} missing --cpu-num-workers 8 (Wf-328-W2)"
                assert (
                    "--num-workers 8" in text
                ), f"{wrapper_name} missing --num-workers 8 (Wf-328-W6)"
                assert (
                    f"--w2-gradient-mode {expected_w2}" in text
                ), (
                    f"{wrapper_name} missing --w2-gradient-mode {expected_w2} "
                    f"(Wf-328-W5)"
                )
                assert (
                    "--no-cuda-graphs" in text
                ), f"{wrapper_name} missing --no-cuda-graphs safety lever (Wf-335-R3)"

    def test_run_wf264_panel_wrapper_exists(self) -> None:
        """Per Wf-338-F1, the run-once wrapper for Wf-264 panel
        (scripts/run_wf264_cfm_property_panel.sh) must exist on disk and
        be referenced in the pipeline master.
        """
        run_once = SCRIPTS_DIR / "run_wf264_cfm_property_panel.sh"
        assert run_once.exists(), f"run-once wrapper missing: {run_once}"
        assert os.access(
            run_once, os.X_OK
        ), f"run-once wrapper not executable: {run_once}"

        text = _read_pipeline_master()
        assert (
            "run_wf264_cfm_property_panel.sh" in text
        ), "pipeline master must reference run-once Wf-264 wrapper"

    def test_gpu_smoke_w2_sliced_wrapper_exists(self) -> None:
        """Per Wf-338-F1, the W₂ sliced GPU smoke wrapper
        (scripts/gpu_smoke_w2_sliced.sh) must exist on disk and be
        referenced in the pipeline master.
        """
        w2_smoke = SCRIPTS_DIR / "gpu_smoke_w2_sliced.sh"
        assert w2_smoke.exists(), f"W₂ sliced smoke wrapper missing: {w2_smoke}"
        assert os.access(
            w2_smoke, os.X_OK
        ), f"W₂ sliced smoke wrapper not executable: {w2_smoke}"

        text = _read_pipeline_master()
        assert (
            "gpu_smoke_w2_sliced.sh" in text
        ), "pipeline master must reference W₂ sliced smoke wrapper"


# ===========================================================================
# Bonus (not in the original 4 but useful for Wf-341 honest framing)
# ===========================================================================


class TestIntegrationVerdict:
    """Final integration verdict for Wf-341-P1."""

    def test_all_5_launchers_integration_verdict(self) -> None:
        """Print a per-launcher integration verdict (CPU-only, no GPU)."""
        text = _read_pipeline_master()

        # Per-launcher status
        verdicts = []
        for wrapper_name in ALL_5_LAUNCHERS:
            paths = _wrapper_exists(wrapper_name)
            if not paths:
                verdicts.append(f"{wrapper_name}: MISSING")
                continue
            in_master = wrapper_name in text
            # Check USER_OK gate or touch-file gate
            has_gate = (
                re.search(rf"USER_OK[= ]+1?\s+.*?{re.escape(wrapper_name)}", text)
                is not None
                or "WF310F_NEXT_WAVE=" in text
                and wrapper_name in {"launch_wf310f_frontier_math_gpu.sh"}
                or wrapper_name == LAUNCHER_WF264  # directly invoked
                or wrapper_name == LAUNCHER_WF333  # directly invoked
            )
            status = "INVOKED" if in_master else "STANDALONE"
            verdicts.append(
                f"{wrapper_name}: {status} | paths={[p.name for p in paths]} | "
                f"parses_cleanly={all(_parse_cleanly(p) for p in paths)}"
            )

        # Sanity: at least the Wf-264 + Wf-333 + G5 + G6 must be referenced
        assert LAUNCHER_WF264 in text, "Wf-264 panel must be invoked in master"
        assert LAUNCHER_WF333 in text, "Wf-333 Vina must be invoked in master"
        # G5 / G6 are referenced via PHASE 1 in-flight comments (NOT
        # directly invoked — they are pre-existing in-flight launches)
        # PHASE 1 comment says "G5 retrain retry" + "Wf-124 G6 60-tuple sweep"
        assert (
            "G5 retrain" in text
        ), "G5 retrain retry must be referenced in master (PHASE 1 comment)"
        assert (
            "Wf-124 G6" in text
        ), "Wf-124 G6 60-tuple sweep must be referenced in master (PHASE 1 comment)"
        # G4 is standalone (intentional)
        assert LAUNCHER_G4 not in text, "G4 is standalone (must NOT be in master)"

        # Print the verdict summary (visible in pytest -v output)
        print("\n=== Wf-341-P1 Integration Verdict ===")
        for v in verdicts:
            print(f"  {v}")
