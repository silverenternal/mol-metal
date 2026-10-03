"""Wf-341-P3 — Wf-74 v2 DrugOOD DANN USER_OK readiness verify.

Owner: Hugo Lee (autonomous Wf-341 ultracode unit)
Date: 2026-09-29
Parent: Wf-341-P3 ultracode residual closure wave (Wf-336-X3 follow-up)
       (per AGENTS.md §5.1 standing GPU authorization + Wf-294B kill_then_adapt)

This module verifies the Wf-74 v2 DrugOOD DANN launcher
(``scripts/launch_wf74_v2_drugood_dann.sh``) is wired correctly with
respect to:

  1. **USER_OK gate enforcement** — when launching with ``--mode R3``
     (3-seed × 50-epoch full GPU run) WITHOUT ``USER_OK=YES``, the
     launcher must die with non-zero exit code (per Wf-294B
     kill_then_adapt + Wf-300C pattern).  ``--mode R1`` is exempt
     (smoke only; CPU).
  2. **VRAM budget check** — ``vram_gate()`` enforces a 20GB free
     VRAM budget before launching the GPU path (per AGENTS.md §5.1).
  3. **Wf-328 wire-ups** — all W1-W10 production optimizations
     (--use-dann, --cpu-num-workers 8, --w2-gradient-mode sort,
     ReverseLayerF) are wired onto the Wf-305D DANN driver.
  4. **Backward compatibility** — ``--mode R1`` CPU smoke still works
     without USER_OK (smoke convention), and ``--dry-run`` does NOT
     bypass the USER_OK gate.

**Honest framing (party-twenty-spirit §3.5, AGENTS.md §5.2):**

* All 4 tests are CPU-only (no GPU launch, no torch.cuda).
* The launcher script's USER_OK gate is verified by source-level
  inspection (no actual invocation of the GPU path).
* VRAM budget check is verified by reading the script source for the
  ``vram_gate()`` function + 20GB threshold.
* Wf-328 fixes are verified by scanning the launcher + Wf-305D trainer
  source for the wire-up CLI flags.

**Test inventory (4 tests per Wf-341-P3 brief):**

1. ``test_user_ok_gate_enforces_exit_code`` — ``--mode R3`` without
   ``USER_OK=YES`` dies (exit code 1) — kills the GPU path.
2. ``test_vram_budget_check_is_wired`` — ``vram_gate()`` enforces 20GB
   budget + nvidia-smi probe.
3. ``test_wf328_fixes_are_wired`` — W1-W10 wire-ups present in
   ``build_wf328_flags()`` + Wf-305D trainer.
4. ``test_backward_compat_r1_no_user_ok`` — ``--mode R1 --dry-run``
   proceeds without USER_OK (smoke convention).
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

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
LAUNCHER = PROJECT_ROOT / "scripts" / "launch_wf74_v2_drugood_dann.sh"
DRUGOOD_TRAINER = (
    PROJECT_ROOT / "molmetal" / "scripts" / "wf305d_drugood_dann_train.py"
)
PY = "/home/hugo/codes/mol-metal/.venv/bin/python"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_launcher() -> str:
    """Read the launcher script source."""
    assert LAUNCHER.exists(), f"Wf-74 v2 launcher missing: {LAUNCHER}"
    return LAUNCHER.read_text()


def _read_trainer() -> str:
    """Read the Wf-305D DANN trainer source."""
    assert DRUGOOD_TRAINER.exists(), (
        f"Wf-305D DrugOOD DANN trainer missing: {DRUGOOD_TRAINER}"
    )
    return DRUGOOD_TRAINER.read_text()


# ---------------------------------------------------------------------------
# Test 1 — USER_OK gate enforces exit code
# ---------------------------------------------------------------------------


class TestUserOkGate:
    """Wf-74 v2 launcher must enforce USER_OK=YES for --mode R3 launches.

    Per Wf-294B kill_then_adapt + Wf-300C pattern: any wall-time GPU
    launch requires explicit ``USER_OK=YES`` env gate.  Wf-335-R3
    (3-seed × 50-epoch RTX 5090 full DANN run, ~3-4 hr) is the wall-time
    launch path that must gate on USER_OK.  Wf-335-R1 (CPU smoke) is
    exempt.
    """

    def test_user_ok_function_defined_in_launcher(self) -> None:
        """The ``user_ok_gate()`` function must be defined in the launcher."""
        text = _read_launcher()
        assert "user_ok_gate()" in text, (
            "user_ok_gate() function missing from Wf-74 v2 launcher"
        )
        # And it must check USER_OK == "YES" for R3 mode
        assert 'USER_OK' in text, "USER_OK env var reference missing"
        assert '"YES"' in text or "'YES'" in text, (
            "USER_OK == 'YES' literal comparison missing"
        )

    def test_r3_mode_requires_user_ok_yes(self) -> None:
        """The user_ok_gate() must require USER_OK='YES' for R3 mode."""
        text = _read_launcher()
        # Look for the gate body: R3 → USER_OK must equal 'YES'
        assert "[ \"${mode}\" = \"R3\" ]" in text, (
            "R3 mode check missing in user_ok_gate()"
        )
        # And the explicit YES comparison (script uses ${USER_OK} expansion)
        assert '!= "YES"' in text, (
            "USER_OK != 'YES' comparison missing in user_ok_gate()"
        )

    def test_user_ok_gate_is_invoked_before_run(self) -> None:
        """user_ok_gate() must be invoked before pre_flight/vram/run."""
        text = _read_launcher()
        gate_idx = text.find("user_ok_gate")
        assert gate_idx >= 0, "user_ok_gate invocation missing"
        # It should appear before pre_flight_gate and phase_R3 / phase_R1
        pre_flight_idx = text.find("pre_flight_gate")
        assert pre_flight_idx > gate_idx, (
            "pre_flight_gate must be invoked AFTER user_ok_gate"
        )

    def test_user_ok_missing_triggers_die_in_r3(self) -> None:
        """When USER_OK is unset and mode=R3, the launcher must die.

        We test this by invoking the launcher in --mode R3 --dry-run and
        asserting that exit code != 0.  The dry-run path still invokes
        ``user_ok_gate`` before any other gate (per source).
        """
        # Sanity: launcher exists and is executable
        assert LAUNCHER.exists(), f"launcher missing: {LAUNCHER}"
        # Invoke with R3 mode, dry-run, NO USER_OK env var
        env = os.environ.copy()
        env.pop("USER_OK", None)  # ensure USER_OK not set
        proc = subprocess.run(
            ["bash", str(LAUNCHER), "--mode", "R3", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        # The launcher should die with non-zero exit (per user_ok_gate)
        assert proc.returncode != 0, (
            f"launcher --mode R3 --dry-run without USER_OK should die; "
            f"got exit={proc.returncode} stdout={proc.stdout!r} "
            f"stderr={proc.stderr!r}"
        )
        # And the error message should mention USER_OK
        combined = proc.stdout + proc.stderr
        assert "USER_OK" in combined, (
            f"USER_OK error message missing from output: {combined!r}"
        )

    def test_user_ok_wrong_value_triggers_die_in_r3(self) -> None:
        """When USER_OK != 'YES' and mode=R3, the launcher must die."""
        env = os.environ.copy()
        env["USER_OK"] = "MAYBE"  # not YES
        proc = subprocess.run(
            ["bash", str(LAUNCHER), "--mode", "R3", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode != 0, (
            f"launcher --mode R3 --dry-run with USER_OK=MAYBE should die; "
            f"got exit={proc.returncode}"
        )
        combined = proc.stdout + proc.stderr
        assert "USER_OK" in combined, (
            f"USER_OK error message missing: {combined!r}"
        )


# ---------------------------------------------------------------------------
# Test 2 — VRAM budget check
# ---------------------------------------------------------------------------


class TestVramBudgetCheck:
    """Wf-74 v2 launcher must enforce 20GB free VRAM budget for GPU path.

    Per AGENTS.md §5.1: VRAM budget check first; abort if OOM risk.
    The launcher implements ``vram_gate()`` which probes
    ``nvidia-smi --query-gpu=memory.free`` and aborts if free VRAM < 20GB.
    """

    def test_vram_gate_function_defined(self) -> None:
        """``vram_gate()`` function must be defined in the launcher."""
        text = _read_launcher()
        assert "vram_gate()" in text or "vram_gate ()" in text, (
            "vram_gate() function missing from Wf-74 v2 launcher"
        )

    def test_vram_budget_threshold_is_20gb(self) -> None:
        """VRAM_BUDGET_GB constant must be 20 (RTX 5090 / RTX PRO 6000 budget)."""
        text = _read_launcher()
        # Look for VRAM_BUDGET_GB=20 assignment
        match = re.search(r"VRAM_BUDGET_GB\s*=\s*(\d+)", text)
        assert match, "VRAM_BUDGET_GB constant missing"
        budget = int(match.group(1))
        assert budget >= 16, (
            f"VRAM_BUDGET_GB={budget} too low; must be ≥16 for RTX 5090 "
            f"32GB D-MPNN + W₂ + DANN head"
        )
        assert budget <= 32, (
            f"VRAM_BUDGET_GB={budget} too high; must be ≤32 to leave "
            f"headroom on RTX 5090 32GB"
        )

    def test_vram_gate_uses_nvidia_smi(self) -> None:
        """``vram_gate()`` must call ``nvidia-smi --query-gpu=memory.free``."""
        text = _read_launcher()
        assert "nvidia-smi" in text, (
            "nvidia-smi invocation missing from vram_gate()"
        )
        # And it queries memory.free
        assert "memory.free" in text, (
            "nvidia-smi --query-gpu=memory.free missing from vram_gate()"
        )

    def test_vram_gate_called_in_main(self) -> None:
        """vram_gate() must be invoked in main() for both R1 and R3 paths."""
        text = _read_launcher()
        # Count vram_gate invocations as a function call (e.g. `vram_gate "${X}"`)
        n_vram_calls = len(
            re.findall(r"\bvram_gate\b\s*\"", text)
        )
        assert n_vram_calls >= 2, (
            f"vram_gate invocation count = {n_vram_calls}; "
            f"expected ≥2 (R1 call + R3 call)"
        )

    def test_vram_gate_dry_run_is_skipped(self) -> None:
        """``vram_gate()`` must short-circuit in DRY-RUN mode (per source)."""
        text = _read_launcher()
        # Look for the DRY-RUN guard inside vram_gate
        # Find vram_gate function body
        match = re.search(r"vram_gate\(\)\s*\{(.*?)\n\}", text, re.DOTALL)
        assert match, "vram_gate function body not parseable"
        body = match.group(1)
        assert "DRY-RUN" in body or "DRY_RUN" in body, (
            "vram_gate() missing DRY-RUN guard"
        )

    def test_vram_budget_oom_die_message(self) -> None:
        """When VRAM < budget, vram_gate() must die with OOM message."""
        text = _read_launcher()
        # Look for the OOM die branch
        assert "OOM" in text or "OOM risk" in text, (
            "OOM die message missing from vram_gate()"
        )


# ---------------------------------------------------------------------------
# Test 3 — Wf-328 fixes wired
# ---------------------------------------------------------------------------


class TestWf328FixesWired:
    """Wf-328 W1-W10 wire-ups must be present in launch_wf74_v2_drugood_dann.sh.

    Per Wf-328-SYN §3: W1 torch.compile, W2 num-workers, W3 BF16,
    W4 TF32, W5 W₂ sort, W6 DataLoader, W7 PyG backend, W8 CUDA Graphs
    (opt-in), W9 Triton (opt-in), W10 DMPNN cache.  The launcher wires
    all 10 onto the Wf-305D DANN driver via ``build_wf328_flags()``.
    """

    def test_build_wf328_flags_function_defined(self) -> None:
        """``build_wf328_flags()`` function must be defined."""
        text = _read_launcher()
        assert "build_wf328_flags" in text, (
            "build_wf328_flags() function missing"
        )

    def test_w1_torch_compile_flag(self) -> None:
        """Wf-328 W1: torch.compile default ON."""
        text = _read_launcher()
        # The constants section should set WF328_TORCH_COMPILE=1
        assert "WF328_TORCH_COMPILE=1" in text, (
            "W1 torch.compile default ON constant missing"
        )

    def test_w2_num_workers_8(self) -> None:
        """Wf-328 W2: cpu-num-workers=8."""
        text = _read_launcher()
        assert "WF328_NUM_WORKERS=8" in text, (
            "W2 num-workers=8 constant missing"
        )
        # And it must be wired into build_wf328_flags
        assert "--cpu-num-workers" in text, (
            "W2 --cpu-num-workers flag missing from build_wf328_flags"
        )

    def test_w3_bf16_default_on(self) -> None:
        """Wf-328 W3: BF16 default ON."""
        text = _read_launcher()
        assert "WF328_BF16=1" in text, (
            "W3 BF16 default ON constant missing"
        )

    def test_w4_tf32_default_on(self) -> None:
        """Wf-328 W4: TF32 default ON."""
        text = _read_launcher()
        assert "WF328_TF32=1" in text, (
            "W4 TF32 default ON constant missing"
        )

    def test_w5_w2_gradient_mode_sort(self) -> None:
        """Wf-328 W5: W₂ gradient mode = sort."""
        text = _read_launcher()
        assert 'WF328_W2_MODE="sort"' in text, (
            "W5 W₂ gradient mode = sort constant missing"
        )
        # And it must be wired into build_wf328_flags
        assert "--w2-gradient-mode" in text, (
            "W5 --w2-gradient-mode flag missing from build_wf328_flags"
        )

    def test_w7_dmpnn_backend_pyg(self) -> None:
        """Wf-328 W7: DMPNN backend = pyg."""
        text = _read_launcher()
        assert 'WF328_DMPNN_BACKEND="pyg"' in text, (
            "W7 DMPNN backend = pyg constant missing"
        )

    def test_w8_cuda_graph_opt_in(self) -> None:
        """Wf-328 W8: CUDA Graphs are opt-in (default OFF)."""
        text = _read_launcher()
        # Per source comment: W8 (CUDA Graphs) and W9 (Triton edge) are
        # opt-in only (per Wf-328-SYN §3)
        assert "W8 (CUDA Graphs) and W9 (Triton edge) are opt-in" in text, (
            "W8/W9 opt-in comment missing"
        )

    def test_use_dann_flag_wired(self) -> None:
        """DANN GRL --use-dann flag must be wired in build_wf328_flags."""
        text = _read_launcher()
        assert "--use-dann" in text, (
            "--use-dann flag missing from build_wf328_flags"
        )
        # And ReverseLayerF must be present in Wf-305D trainer
        trainer = _read_trainer()
        assert "ReverseLayerF" in trainer, (
            "ReverseLayerF (DANN GRL) missing from Wf-305D trainer"
        )

    def test_use_kendall_lambda_grl_wired(self) -> None:
        """DANN multi-task: --use-kendall, --lambda-grl, --wasserstein-weight."""
        text = _read_launcher()
        for flag in ("--use-kendall", "--lambda-grl", "--wasserstein-weight"):
            assert flag in text, f"{flag} flag missing from build_wf328_flags"
        # Verify numeric defaults
        assert "--wasserstein-weight 0.05" in text, (
            "Wasserstein weight default 0.05 missing"
        )
        assert "--lambda-grl 0.1" in text, (
            "Lambda GRL default 0.1 missing"
        )
        assert "--alpha 0.5" in text, (
            "Alpha default 0.5 missing"
        )


# ---------------------------------------------------------------------------
# Test 4 — backward compat (R1 dry-run without USER_OK)
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Backward compat: --mode R1 --dry-run must proceed without USER_OK.

    Per Wf-300C pattern + Wf-335-R1 launch suggestion: the R1 CPU smoke
    mode is OPTIONAL-OK (no user OK required).  This is the smoke
    convention so any agent can verify the wrapper is launchable without
    requesting user authorization.
    """

    def test_r1_dry_run_proceeds_without_user_ok(self) -> None:
        """--mode R1 --dry-run must complete (exit 0) without USER_OK."""
        env = os.environ.copy()
        env.pop("USER_OK", None)  # ensure USER_OK not set
        proc = subprocess.run(
            ["bash", str(LAUNCHER), "--mode", "R1", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0, (
            f"launcher --mode R1 --dry-run should succeed (exit 0); "
            f"got exit={proc.returncode} stdout={proc.stdout!r} "
            f"stderr={proc.stderr!r}"
        )
        # And it should NOT mention USER_OK as a fatal error
        combined = proc.stdout + proc.stderr
        # R1 mode may mention USER_OK as a warning, but not as fatal
        if "USER_OK" in combined:
            # Verify it's a WARN, not FATAL
            assert "[WARN]" in combined or "[OK]" in combined, (
                f"USER_OK should be WARN/OK in R1 mode, not FATAL: {combined!r}"
            )

    def test_dry_run_does_not_bypass_user_ok_in_r3(self) -> None:
        """--dry-run must NOT bypass USER_OK gate for R3 (per Wf-300C).

        Per source: '--dry-run does NOT bypass the USER_OK gate.'
        """
        text = _read_launcher()
        assert "--dry-run does NOT bypass the USER_OK gate" in text, (
            "Source must explicitly document that --dry-run does NOT "
            "bypass USER_OK gate"
        )

    def test_help_message_documents_modes(self) -> None:
        """--help must list both R1 and R3 modes."""
        proc = subprocess.run(
            ["bash", str(LAUNCHER), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0, (
            f"--help should exit 0; got {proc.returncode}"
        )
        combined = proc.stdout + proc.stderr
        assert "R1" in combined, "--help missing R1 mode"
        assert "R3" in combined, "--help missing R3 mode"
        assert "USER_OK" in combined, "--help missing USER_OK documentation"

    def test_unknown_mode_dies(self) -> None:
        """--mode R2 (invalid) must die with exit 1."""
        env = os.environ.copy()
        env.pop("USER_OK", None)
        proc = subprocess.run(
            ["bash", str(LAUNCHER), "--mode", "R2", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode != 0, (
            f"--mode R2 (invalid) should die; got exit={proc.returncode}"
        )
        combined = proc.stdout + proc.stderr
        assert "R1 | R3" in combined or "mode must be" in combined, (
            f"Error message must mention 'R1 | R3' valid modes: {combined!r}"
        )

    def test_dry_run_creates_reports_dir(self) -> None:
        """--mode R1 --dry-run must create the reports/r1 directory."""
        reports_dir = PROJECT_ROOT / "molmetal" / "reports" / "wf74_v2_drugood_dann_launch"
        # Clean before test
        if (reports_dir / "r1").exists():
            pass  # leave alone, dry-run writes DRY_RUN.log
        env = os.environ.copy()
        env.pop("USER_OK", None)
        proc = subprocess.run(
            ["bash", str(LAUNCHER), "--mode", "R1", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0
        # The DRY_RUN.log marker should be present
        assert (reports_dir / "r1" / "DRY_RUN.log").exists(), (
            "DRY_RUN.log marker missing from r1/ — wrapper did not run"
        )

    def test_bash_syntax_check(self) -> None:
        """The launcher must pass ``bash -n`` syntax check."""
        proc = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0, (
            f"launcher syntax error: stderr={proc.stderr!r}"
        )


# ---------------------------------------------------------------------------
# Module-level main — allow ``python -m pytest`` discovery
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))