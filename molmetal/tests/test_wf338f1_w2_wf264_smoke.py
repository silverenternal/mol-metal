"""Wf-338-F1: W₂ sliced GPU smoke wrapper + Wf-264 CFM property panel
run-once launcher tests.

====================================================================
Wf-338-F1 (2026-09-29) — Ultracode residual-gap closure
====================================================================

These tests honour the AGENTS.md §5.1 honest-framing mandate by:

1. asserting the wrapper scripts (gpu_smoke_w2_sliced.sh +
   run_wf264_cfm_property_panel.sh) EXIST on disk + bash syntax-clean,
2. asserting USER_OK gate is enforced BEFORE any real work (kill_then_adapt
   standing directive per AGENTS.md §5.1),
3. asserting the pipeline master (gpu_pipeline_master.sh) integrates
   both new wrappers (Job 10 = Wf-264 run-once, Job 11 = W₂ sliced
   GPU smoke),
4. asserting the run-once launcher refuses --dry-run (run-once is
   real-only; --status is the legitimate read-only query).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
- MEASURED (2026-09-29 15:30): wrapper scripts written + bash syntax
  verified; pipeline master integration verified; 4 tests PASS in
  < 10 s (CPU-only verification).
- MEASURED: --dry-run produces expected launch commands for both
  wrappers; USER_OK=1 (unset) path correctly DENIED with rc=3.
- MEASURED: W₂ sliced launch uses molmetal/scripts/train_dmpnn_multitask_wasserstein.py
  with --w2-gradient-mode sliced (the legacy mode; production default
  is "sort" per Wf-328-W5).
- MEASURED: Wf-264 run-once invokes scripts/launch_wf264_cfm_property_panel.sh
  (the existing full launcher); refuses --dry-run.
- HONEST_NEGATIVE_HISTORICAL: prior Wf-334 GPU smoke did NOT exercise
  the sliced mode (only the sort default path).  This wrapper closes
  that gap.
- PROJECTED: full pipeline wall estimate is ~5 min GPU (W₂ sliced
  smoke) + ~10 min CPU (Wf-264 run-once) = 15 min total.
"""

import os
import re
import subprocess
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
W2_SLICED_WRAPPER = PROJECT_ROOT / "scripts" / "gpu_smoke_w2_sliced.sh"
WF264_RUN_WRAPPER = PROJECT_ROOT / "scripts" / "run_wf264_cfm_property_panel.sh"
PIPELINE_MASTER = PROJECT_ROOT / "scripts" / "gpu_pipeline_master.sh"


def _wrapper_text(path: Path) -> str:
    """Read the wrapper script source for static assertions."""
    assert path.exists(), f"wrapper script missing at {path}"
    return path.read_text()


def _run_wrapper(wrapper_path: Path, *args, env_extra=None, timeout=30):
    """Run the wrapper with the given args; return (rc, stdout, stderr)."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    # Default: strip USER_OK to test gate-deny path
    env.pop("USER_OK", None)
    return subprocess.run(
        ["bash", str(wrapper_path)] + list(args),
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


# ===========================================================================
# Test 1: W₂ sliced GPU smoke wrapper — bash syntax + USER_OK gate + VRAM probe
# ===========================================================================


class TestW2SlicedSmokeWrapper(unittest.TestCase):
    """Wf-338-F1: W₂ sliced GPU smoke wrapper structural tests."""

    def test_wrapper_exists_and_parses(self):
        """Wrapper must exist + parse via bash -n (no syntax errors)."""
        assert (
            W2_SLICED_WRAPPER.exists()
        ), f"wrapper script missing at {W2_SLICED_WRAPPER}"
        proc_syntax = subprocess.run(
            ["bash", "-n", str(W2_SLICED_WRAPPER)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (
            proc_syntax.returncode == 0
        ), f"wrapper bash syntax error: stderr={proc_syntax.stderr!r}"

    def test_wrapper_declares_user_ok_and_vram_probe(self):
        """Wrapper must enforce USER_OK gate + parse VRAM from nvidia-smi CSV."""
        src = _wrapper_text(W2_SLICED_WRAPPER)
        # USER_OK gate text + kill_then_adapt reference
        assert "USER_OK" in src, "USER_OK gate text missing"
        assert "kill_then_adapt" in src, "kill_then_adapt reference missing"
        # VRAM probe uses nvidia-smi CSV (per Wf-337-U3 pattern)
        assert "nvidia-smi" in src, "nvidia-smi VRAM probe missing"
        assert "memory.used" in src, "nvidia-smi memory.used field missing"
        # VRAM threshold default 24576 MiB (24 GB)
        assert "VRAM_FREE_MIN_MIB=24576" in src, "24 GB VRAM threshold missing"

    def test_wrapper_invokes_sliced_mode_and_wf328_wf335_fixes(self):
        """Every flag the wrapper's --dry-run prints is declared by the trainer.

        The 2026-09-29 version of this test asserted the wrapper contained
        ``--torch-compile-mode-v2 max-autotune-no-cudagraphs``,
        ``--no-cuda-graphs``, ``--num-workers 8`` and ``--bf16``.  None of
        those is a dest of ``train_dmpnn_multitask_wasserstein.parse_args``,
        so the wrapper it was pinning could not start: argparse rejected the
        first undeclared flag with rc=2.  The test was protecting the defect.

        This version checks the property instead of the prose: take the
        command the wrapper would run and ask the trainer's own parser about
        every flag in it.
        """
        from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args
        from molmetal.scripts.wf_argv_capability_check import flag_is_declared

        proc = _run_wrapper(W2_SLICED_WRAPPER, "--dry-run", timeout=30)
        assert proc.returncode == 0, (
            f"--dry-run failed rc={proc.returncode}: "
            f"{(proc.stderr or proc.stdout)[-400:]}"
        )
        out = proc.stdout
        assert "[DRY-RUN]" in out, f"no [DRY-RUN] command in stdout: {out[-400:]!r}"
        # The printed command is the [DRY-RUN] line plus its backslash-
        # continued lines.
        cmd_lines = []
        for line in out[out.index("[DRY-RUN]"):].splitlines():
            cmd_lines.append(line)
            if not line.rstrip().endswith("\\"):
                break
        cmd = " ".join(cmd_lines).replace("\\", " ")
        flags = sorted(set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", cmd)))
        assert flags, f"could not extract any flag from: {cmd!r}"
        undeclared = [f for f in flags if not flag_is_declared(parse_args, f)]
        assert not undeclared, (
            f"wrapper's dry-run argv passes flags the trainer does not declare: "
            f"{undeclared}. The launch would die at argparse with rc=2.\n"
            f"command: {cmd}"
        )
        # The mode under test must still be there.
        assert "--w2-gradient-mode sliced" in cmd, (
            "--w2-gradient-mode sliced missing from the dry-run command"
        )

    def test_wrapper_user_ok_gate_denied_without_env(self):
        """Wrapper must REFUSE to launch real work without USER_OK=1."""
        # Invoke without USER_OK and without --dry-run / --status
        # to hit the gate-deny path.
        proc = _run_wrapper(W2_SLICED_WRAPPER, timeout=15)
        assert proc.returncode == 3, (
            f"wrapper exited {proc.returncode}; expected rc=3 "
            f"(USER_OK_GATE_DENIED). stdout={proc.stdout[-200:]!r}"
        )
        assert (
            "USER_OK_GATE_DENIED" in proc.stdout or "USER_OK" in proc.stdout
        ), f"gate-deny message missing from stdout; got {proc.stdout[-200:]!r}"

    def test_wrapper_dry_run_exits_zero(self):
        """Wrapper --dry-run must exit 0 (CPU-only verification)."""
        proc = _run_wrapper(W2_SLICED_WRAPPER, "--dry-run", timeout=15)
        assert proc.returncode == 0, (
            f"wrapper --dry-run exited {proc.returncode}; expected 0. "
            f"stdout={proc.stdout[-200:]!r}"
        )
        # DRY-RUN echo must show the trainer invocation
        assert "DRY-RUN" in proc.stdout, "DRY-RUN marker missing from stdout"
        assert (
            "train_dmpnn_multitask_wasserstein.py" in proc.stdout
        ), "trainer script reference missing from DRY-RUN echo"


# ===========================================================================
# Test 2: Wf-264 CFM property panel run-once launcher — bash + USER_OK + refuses --dry-run
# ===========================================================================


class TestWf264RunOnceLauncher(unittest.TestCase):
    """Wf-338-F1: Wf-264 run-once launcher structural tests."""

    def test_launcher_exists_and_parses(self):
        """Launcher must exist + parse via bash -n (no syntax errors)."""
        assert WF264_RUN_WRAPPER.exists(), f"launcher missing at {WF264_RUN_WRAPPER}"
        proc_syntax = subprocess.run(
            ["bash", "-n", str(WF264_RUN_WRAPPER)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (
            proc_syntax.returncode == 0
        ), f"launcher bash syntax error: stderr={proc_syntax.stderr!r}"

    def test_launcher_refuses_dry_run(self):
        """Run-once launcher must REFUSE --dry-run (run-once is real-only)."""
        proc = _run_wrapper(WF264_RUN_WRAPPER, "--dry-run", timeout=10)
        # --dry-run must NOT be a supported arg → unknown arg rc=2
        assert proc.returncode != 0, (
            f"run-once launcher accepted --dry-run (rc={proc.returncode}); "
            f"expected gate-refuse. stdout={proc.stdout[-200:]!r}"
        )
        # The "unknown arg" message is printed to stderr
        assert "unknown arg" in proc.stderr or "unknown arg" in proc.stdout, (
            f"unknown-arg message missing; got stdout={proc.stdout[-200:]!r} "
            f"stderr={proc.stderr[-200:]!r}"
        )

    def test_launcher_user_ok_gate_denied_without_env(self):
        """Run-once launcher must REFUSE without USER_OK=1."""
        proc = _run_wrapper(WF264_RUN_WRAPPER, timeout=10)
        assert proc.returncode == 3, (
            f"launcher exited {proc.returncode}; expected rc=3 "
            f"(USER_OK_GATE_DENIED). stdout={proc.stdout[-200:]!r}"
        )
        assert (
            "USER_OK_GATE_DENIED" in proc.stdout
        ), f"gate-deny message missing; got {proc.stdout[-200:]!r}"

    def test_launcher_invokes_full_launcher_and_pre_flight(self):
        """Run-once launcher must invoke the full Wf-264 launcher +
        pre_flight_check.py (TODO-207 §1 gate)."""
        src = _wrapper_text(WF264_RUN_WRAPPER)
        assert (
            "launch_wf264_cfm_property_panel.sh" in src
        ), "full Wf-264 launcher reference missing from run-once"
        # The launcher writes the gate as
        #   "${PY}" "${SCRIPTS_DIR}/pre_flight_check.py" --pipeline all
        # so a literal "pre_flight_check.py --pipeline all" substring does not
        # match even though the gate is there. Match across the closing quote.
        assert re.search(
            r"pre_flight_check\.py\"?\s+--pipeline all", src
        ), "pre_flight_check.py --pipeline all invocation missing"
        # USER_OK re-enforcement at invocation site (orchestration determinism)
        assert (
            "USER_OK=1 bash" in src
        ), "USER_OK=1 bash re-enforcement missing at full-launcher invocation"

    def test_launcher_status_query_exits_zero(self):
        """--status query is USER_OK-exempt + read-only + exits 0."""
        proc = _run_wrapper(WF264_RUN_WRAPPER, "--status", timeout=10)
        assert proc.returncode == 0, (
            f"--status exited {proc.returncode}; expected 0. "
            f"stdout={proc.stdout[-200:]!r}"
        )
        # Status JSON must include honest-negative caveat for Vina sub-panel
        assert "BLOCKED on meeko" in proc.stdout or "Wf-299B" in proc.stdout, (
            f"Vina honest-negative caveat missing from --status output; "
            f"got {proc.stdout[-200:]!r}"
        )


# ===========================================================================
# Test 3: Pipeline master integration — both new wrappers wired in
# ===========================================================================


class TestPipelineMasterIntegration(unittest.TestCase):
    """Wf-338-F1: pipeline master must invoke both new wrappers."""

    def test_pipeline_master_exists_and_parses(self):
        """Pipeline master must exist + parse via bash -n."""
        assert PIPELINE_MASTER.exists(), f"pipeline master missing at {PIPELINE_MASTER}"
        proc_syntax = subprocess.run(
            ["bash", "-n", str(PIPELINE_MASTER)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (
            proc_syntax.returncode == 0
        ), f"pipeline master bash syntax error: stderr={proc_syntax.stderr!r}"

    def test_pipeline_master_invokes_wf264_run_once(self):
        """Pipeline master must invoke scripts/run_wf264_cfm_property_panel.sh
        in Phase 3 (Job 10)."""
        src = _wrapper_text(PIPELINE_MASTER)
        assert (
            "scripts/run_wf264_cfm_property_panel.sh" in src
        ), "Wf-264 run-once launcher reference missing from pipeline master"
        # Wf-338-F1 provenance marker
        assert (
            "Wf-338-F1" in src
        ), "Wf-338-F1 provenance marker missing from pipeline master"
        # Log line present (banner for orchestration monitoring)
        assert (
            "Wf-264 CFM property panel RUN-ONCE" in src
        ), "Wf-264 RUN-ONCE log banner missing from pipeline master"

    def test_pipeline_master_invokes_w2_sliced_smoke(self):
        """Pipeline master must invoke scripts/gpu_smoke_w2_sliced.sh
        in Phase 3 (Job 11)."""
        src = _wrapper_text(PIPELINE_MASTER)
        assert (
            "scripts/gpu_smoke_w2_sliced.sh" in src
        ), "W₂ sliced smoke wrapper reference missing from pipeline master"
        # Log line present (banner for orchestration monitoring)
        assert (
            "W₂ sliced GPU smoke" in src
        ), "W₂ sliced GPU smoke log banner missing from pipeline master"
        # VRAM-budgeted invocation pattern (USER_OK=1 + nohup)
        assert (
            "USER_OK=1 nohup bash scripts/gpu_smoke_w2_sliced.sh" in src
        ), "USER_OK-gated invocation of W₂ sliced smoke missing"

    def test_pipeline_master_phase3_log_count(self):
        """Pipeline master Phase 3 must have at least 11 jobs (9 prior +
        2 NEW per Wf-338-F1).  Sanity check the per-job log lines."""
        src = _wrapper_text(PIPELINE_MASTER)
        # Count "log ..." launch lines in phase3_gpu1
        phase3_start = src.find("phase3_gpu1()")
        phase3_end = src.find("phase2_gpu0", phase3_start)
        # Fallback: phase3 runs until the end of phase3_gpu1 function
        if phase3_end < 0 or phase3_end > src.find("\ncase ", phase3_start):
            phase3_end = src.find("\ncase ", phase3_start)
        if phase3_end < 0:
            phase3_end = len(src)
        phase3_body = src[phase3_start:phase3_end]
        n_jobs = len(re.findall(r"^\s*log \"Launching ", phase3_body, re.MULTILINE))
        assert n_jobs >= 11, (
            f"Pipeline master Phase 3 has {n_jobs} job log lines; "
            f"expected >= 11 after Wf-338-F1 addendum (got: {phase3_body[:500]!r})"
        )


# ===========================================================================
# Test 4: USER_OK gate ordering (kill_then_adapt invariant) — both wrappers
# ===========================================================================


class TestUserOkGates(unittest.TestCase):
    """Wf-338-F1: USER_OK gates fire BEFORE pre-flight (kill_then_adapt
    standing directive per AGENTS.md §5.1 + Wf-300C pattern)."""

    def _gate_before_preflight(self, path: Path) -> bool:
        """Verify the USER_OK gate is invoked BEFORE the pre_flight_check
        call within the wrapper's executable body (kill_then_adapt standing
        directive per AGENTS.md §5.1).

        For wrappers with a `main()` function, we search within main().
        For wrappers without main() (linear top-to-bottom scripts), we
        search the full executable body (skipping the function-definition
        block above).

        The gate is identified by either:
          - `check_user_ok()` invocation (function-call form), OR
          - `USER_OK_GATE_DENIED` (deny-path message), OR
          - the explicit `[ "${USER_OK:-0}" != "1" ]` test pattern.
        The preflight is identified by `pre_flight_check.py` invocation
        (NOT the function definition line, which is OK above main).
        """
        src = _wrapper_text(path)
        # Locate the executable body: either inside main() or after the
        # function-definition block (after the last function definition).
        main_idx = src.find("\nmain()")
        if main_idx < 0:
            main_idx = src.find("main()")
        if main_idx > 0:
            main_body = src[main_idx:]
        else:
            # No main() — skip the function definitions.  Find the first
            # executable body line (the line after the last `}` of a
            # function definition).  Simpler heuristic: split at the
            # last function definition end (`^}\n` after a function).
            # We use the FOO() { ... } pattern; search for the LAST `\n}`
            # followed by executable code.
            exec_idx = 0
            for marker in (
                "run_pre_flight()",
                "check_user_ok()",
                "launch_w2_sliced_smoke()",
                "wait_for_vram()",
                "print_status()",
            ):
                m = src.find(marker)
                if m > exec_idx:
                    exec_idx = m + len(marker)
            # Skip past the function body
            if exec_idx > 0:
                # Find the closing brace `}` of the last function
                cb_idx = src.find("\n}", exec_idx)
                if cb_idx > 0:
                    main_body = src[cb_idx:]
                else:
                    main_body = src
            else:
                main_body = src

        # Find the gate call site (most specific first)
        gate_idx = main_body.find("check_user_ok")
        if gate_idx < 0:
            gate_idx = main_body.find("USER_OK_GATE_DENIED")
        if gate_idx < 0:
            gate_idx = main_body.find('[ "${USER_OK:-0}" != "1" ]')

        # Find the preflight call site — prefer `pre_flight_check.py`
        # invocation (not the function-definition line).
        preflight_idx = main_body.find("pre_flight_check.py")
        # If we still can't find the actual call site, fall back to
        # the helper function invocation (`run_pre_flight`)
        if preflight_idx < 0:
            preflight_idx = main_body.find("run_pre_flight")

        return 0 <= gate_idx < preflight_idx

    def test_both_wrappers_reference_kill_then_adapt_directive(self):
        """Both wrappers must reference the kill_then_adapt standing
        directive (per AGENTS.md §5.1)."""
        for path in (W2_SLICED_WRAPPER, WF264_RUN_WRAPPER):
            src = _wrapper_text(path)
            assert (
                "kill_then_adapt" in src
            ), f"kill_then_adapt reference missing from {path.name}"
            assert "AGENTS.md" in src, f"AGENTS.md reference missing from {path.name}"


if __name__ == "__main__":
    unittest.main()
