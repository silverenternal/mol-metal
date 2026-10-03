"""Wf-337-G3 — G6 launcher regression test for scripts/launch_wf124_p3_g6.sh.

Owner: Hugo Lee
Date: 2026-09-29
Parent: TODO/pending/182_wf124_p3_per_innovation_ablation.md

Per Wf-330-B4: G6 launcher broke when stale args (--cuda-graphs default ON +
--cpu-num-workers 0 override) leaked into the wrapper. Wf-328-W2 + Wf-335-R3
fixed both. Wf-337-G3 locks down the post-fix state so future edits can't
re-introduce stale args.

4 NEW tests (CPU-only — bash + grep + subprocess.run on --dry-run path only):
  1. test_g6_launcher_parses — bash syntax + dry-run executes cleanly + reports 60 tuples
  2. test_g6_launcher_has_all_required_flags — Wf-328 num-workers + Wf-335 no-cuda-graphs + Wf-296C no-kendall/no-dann
  3. test_g6_launcher_pre_flight_integration — pre-flight + per-wave + budget tracker ordering
  4. test_g6_launcher_user_ok_gate — USER_OK blocks real launch, --dry-run bypasses it

Honest framing (party-twenty-spirit §3.5):
  - Regression test = prevent re-introduction of stale args
  - CPU-only test (no torch.cuda, no actual training)
  - Per AGENTS.md §5.1: standing GPU auth + USER_OK gate
  - MEASURED: tests run against scripts/launch_wf124_p3_g6.sh on 2026-09-29
  - PROJECTED: prevents re-introduction of Wf-330-B4 regression class
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = ROOT / "scripts" / "launch_wf124_p3_g6.sh"
DATA_CSV = ROOT / ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
TRAINER = ROOT / "molmetal/scripts/train_dmpnn_multitask_wasserstein.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_wrapper() -> str:
    assert WRAPPER.exists(), f"G6 wrapper missing: {WRAPPER}"
    return WRAPPER.read_text()


def _body_start(src: str) -> int:
    """Index of first executable line after shebang + header comment block."""
    marker = "\nset -e\n"
    idx = src.find(marker)
    if idx < 0:
        return 0
    return idx + len(marker)


# ---------------------------------------------------------------------------
# Test 1: G6 launcher parses + dry-run executes cleanly
# ---------------------------------------------------------------------------


def test_g6_launcher_parses():
    """G6 wrapper must have valid bash syntax + --dry-run executes cleanly
    + reports 60 (config × metal × seed) tuples PLANNED."""
    # bash -n syntax check
    proc = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (
        proc.returncode == 0
    ), f"bash syntax check failed: rc={proc.returncode} stderr={proc.stderr}"

    # --dry-run execution
    src = _read_wrapper()
    assert "--dry-run" in src, "G6 wrapper must support --dry-run mode"
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "USER_OK": "1"},
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"dry-run must exit 0; got rc={proc.returncode}. "
        f"stderr={proc.stderr[-300:]!r}"
    )

    # 60 tuples enumerated
    out = proc.stdout
    assert (
        "60 (config, metal, seed) tuples PLANNED" in out
    ), f"dry-run must report 60 tuples PLANNED; tail: {out[-400:]!r}"

    # 4 configs x 5 metals x 3 seeds = 60 — verify the cartesian product
    # is enumerated by counting per-cfg lines in the dry-run stdout.
    n_cfg_lines = sum(
        1
        for line in out.splitlines()
        if re.search(r"cfg=(no_multi_task|no_wass|no_kendall|no_dann)", line)
    )
    assert n_cfg_lines >= 60, (
        f"expected ≥ 60 cfg= lines in dry-run stdout (4×5×3 cartesian); "
        f"got {n_cfg_lines}"
    )


# ---------------------------------------------------------------------------
# Test 2: G6 launcher has all required flags (Wf-328 + Wf-335 + Wf-296C fixes)
# ---------------------------------------------------------------------------


def test_g6_launcher_has_all_required_flags():
    """Lock down ALL the Wf-328 / Wf-335 / Wf-296C fixes are present in the
    wrapper. This is the regression-prevention core: any future PR that
    drops one of these flags breaks this test."""
    src = _read_wrapper()

    # Wf-328-W2 — --cpu-num-workers 8 (NOT 0 — that was the Wf-326B-reverted
    # override that broke G5). Must be exactly 8 per Wf-326B OPTIMAL_NUM_WORKERS
    # on 24-core host.
    assert "--cpu-num-workers 8" in src, (
        "Wf-328-W2 fix missing: --cpu-num-workers 8 must be wired. "
        "If you see --cpu-num-workers 0 in COMMON_ARGS, the Wf-326B-reverted "
        "override is back — DO NOT ship."
    )
    # Negative: --cpu-num-workers 0 must NOT appear as an ACTUAL ARGUMENT
    # (not in a comment that explains the reverted value). Scan only the
    # COMMON_ARGS array contents.
    common_args_match = re.search(r"COMMON_ARGS=\((.*?)\)", src, re.DOTALL)
    assert common_args_match is not None, "COMMON_ARGS array not found in G6 wrapper"
    common_args_body = common_args_match.group(1)
    assert (
        "--cpu-num-workers 8" in common_args_body
    ), "Wf-328-W2 fix missing: --cpu-num-workers 8 must be in COMMON_ARGS array"
    assert not re.search(r"--cpu-num-workers\s+0\b", common_args_body), (
        "Wf-326B-reverted --cpu-num-workers 0 leaked back into COMMON_ARGS. "
        "Per Wf-328-W2, the only valid value is 8 (24-core host OPTIMAL)."
    )

    # Wf-335-R3 — --no-cuda-graphs SAFETY LEVER. Per Wf-330-B4 the default
    # CUDA Graphs caused RuntimeError on Ru seed=2 + Ir seed=0 under
    # --torch-compile-mode reduce-overhead. The lever MUST be present
    # and wired (default ON in trainer).
    assert "--no-cuda-graphs" in src, (
        "Wf-335-R3 fix missing: --no-cuda-graphs SAFETY LEVER must be wired. "
        "Without this, G6 will re-trigger Wf-330-B4 RuntimeError on Ru/Ir seeds."
    )

    # Wf-296C — --no-kendall + --no-dann must be in CONFIRMED_FLAGS dict so
    # the per-innovation ablation actually ablates (per-config, not blanket).
    # Verified via grep on train_dmpnn_multitask_wasserstein.py lines 1323/1331.
    assert (
        '"--no-kendall"' in src or "'--no-kendall'" in src
    ), "Wf-296C fix missing: --no-kendall must be in CONFIRMED_FLAGS dict."
    assert (
        '"--no-dann"' in src or "'--no-dann'" in src
    ), "Wf-296C fix missing: --no-dann must be in CONFIRMED_FLAGS dict."

    # Wf-296C — VERIFIED_CONFIGS must list all 4 (no skip-with-WARN path)
    # i.e. UNVERIFIED_CONFIGS must be EMPTY.
    assert (
        "VERIFIED_CONFIGS=" in src
    ), "VERIFIED_CONFIGS dispatch table missing from G6 wrapper."
    assert (
        'UNVERIFIED_CONFIGS=""' in src
    ), "Wf-296C fix: UNVERIFIED_CONFIGS must be empty (all 4 flags verified)."

    # Trainer must also support these flags (cross-reference the CLI surface
    # that the wrapper depends on — guards against trainer refactor that
    # silently drops the flag alias).
    assert TRAINER.exists(), f"trainer missing: {TRAINER}"
    trainer_src = TRAINER.read_text()
    assert (
        "--no-cuda-graphs" in trainer_src
    ), "trainer missing --no-cuda-graphs CLI flag — wrapper calls will fail."
    assert (
        '"--no-kendall"' in trainer_src or "'--no-kendall'" in trainer_src
    ), "trainer missing --no-kendall CLI flag — Wf-296C ablation broken."
    assert (
        '"--no-dann"' in trainer_src or "'--no-dann'" in trainer_src
    ), "trainer missing --no-dann CLI flag — Wf-296C ablation broken."


# ---------------------------------------------------------------------------
# Test 3: G6 launcher pre-flight integration (kill_then_adapt ordering)
# ---------------------------------------------------------------------------


def test_g6_launcher_pre_flight_integration():
    """Per kill_then_adapt + TODO-207 + AGENTS.md §1.4: pre_flight_check
    must be invoked BEFORE per_wave_checklist + wave_budget_tracker, and
    BEFORE any per-tuple runner. Per AGENTS.md §1.4, STALE pre-flight
    blocks PASS — wrapper must invoke pre_flight_check.py (not skip it)."""
    src = _read_wrapper()
    body = src[_body_start(src) :]

    # All 3 wave-tool guards must be invoked
    assert (
        "pre_flight_check.py --pipeline all" in body
    ), "pre_flight_check.py --pipeline all not invoked in G6 wrapper"
    assert (
        "per_wave_checklist.py --wave" in body
    ), "per_wave_checklist.py not invoked in G6 wrapper"
    assert (
        "wave_budget_tracker.py" in body
    ), "wave_budget_tracker.py not invoked in G6 wrapper"

    # Ordering invariant: pre_flight must precede BOTH per_wave_checklist
    # AND wave_budget_tracker (kill_then_adapt — STALE pre-flight is a
    # hard block before any GPU launch). The actual ordering in the wrapper
    # is pre_flight → per_wave → budget_tracker (per Wf-296C convention).
    pre_flight_idx = body.find("pre_flight_check.py --pipeline all")
    budget_idx = body.find("wave_budget_tracker.py")
    per_wave_idx = body.find("per_wave_checklist.py --wave")
    assert pre_flight_idx > 0, "pre_flight_check --pipeline all not invoked"
    assert budget_idx > 0, "wave_budget_tracker not invoked"
    assert per_wave_idx > 0, "per_wave_checklist not invoked"
    assert pre_flight_idx < budget_idx, (
        f"pre_flight_check (idx {pre_flight_idx}) must precede "
        f"wave_budget_tracker (idx {budget_idx})"
    )
    assert pre_flight_idx < per_wave_idx, (
        f"pre_flight_check (idx {pre_flight_idx}) must precede "
        f"per_wave_checklist (idx {per_wave_idx})"
    )

    # Each invocation is guarded by `|| { echo FAIL; exit 1; }` so a STALE
    # pre-flight actually halts the wrapper (not silently continues).
    for tool in (
        "pre_flight_check.py --pipeline all",
        "per_wave_checklist.py --wave",
        "wave_budget_tracker.py",
    ):
        idx = body.find(tool)
        # Look at the next 80 chars after the tool name for the `||` guard
        tail = body[idx : idx + 200]
        assert "||" in tail and "exit 1" in tail, (
            f"tool {tool!r} invocation lacks `|| {{ ... exit 1 }}` guard. "
            f"A STALE pre-flight would silently continue without this. "
            f"tail: {tail[:120]!r}"
        )

    # Pre-flight must come BEFORE the per-tuple runner (the for-loop that
    # actually launches GPU jobs). The for-loop is in the `run_tuple`
    # function call site (lines 181-188), distinguished by the
    # `run_tuple "${cfg}" "${metal}" "${seed}"` call which is unique to
    # the real-launch path (dry-run block also has a for-loop but never
    # calls run_tuple).
    run_loop_idx = body.find('run_tuple "${cfg}" "${metal}" "${seed}"')
    assert (
        run_loop_idx > 0
    ), "per-tuple runner loop (run_tuple invocation) not found in G6 wrapper body"
    assert pre_flight_idx < run_loop_idx, (
        f"pre_flight_check (idx {pre_flight_idx}) must precede per-tuple "
        f"runner loop (idx {run_loop_idx})"
    )


# ---------------------------------------------------------------------------
# Test 4: G6 launcher USER_OK gate
# ---------------------------------------------------------------------------


def test_g6_launcher_user_ok_gate():
    """Per Wf-306D + AGENTS.md §5.1 standing GPU auth: 7.5h wall launch
    requires explicit USER_OK=1. The gate MUST:
      - be checked BEFORE pre_flight (so we don't even probe GPU without OK)
      - block invocation when USER_OK is unset (exit non-zero)
      - emit the kill_then_adapt standing directive message
      - be bypassed by --dry-run (which exits 0 without touching GPU)
    """
    src = _read_wrapper()
    body = src[_body_start(src) :]

    # Gate structure present
    assert "USER_OK" in src, "USER_OK gate text missing"
    assert "kill_then_adapt" in src, "kill_then_adapt reference missing"

    # Gate ordering: gate MUST come BEFORE pre_flight (kill_then_adapt
    # standing directive — no GPU probe without user OK).
    gate_idx = body.find('if [[ -z "${USER_OK:-}" ]]')
    pre_flight_idx = body.find("pre_flight_check.py --pipeline all")
    assert gate_idx > 0, "USER_OK gate if-block not found"
    assert pre_flight_idx > 0, "pre_flight_check invocation not found"
    assert gate_idx < pre_flight_idx, (
        f"USER_OK gate (idx {gate_idx}) must appear BEFORE pre_flight "
        f"(idx {pre_flight_idx}). Without this, the wrapper probes GPU "
        f"even when user has not OK'd the launch."
    )

    # Actual end-to-end: invocation without USER_OK exits non-zero
    proc = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "USER_OK"},
        timeout=15,
    )
    assert proc.returncode != 0, (
        f"wrapper exited 0 without USER_OK; expected gate block. "
        f"stdout={proc.stdout[-200:]!r}"
    )
    assert (
        "USER OK REQUIRED" in proc.stdout
    ), f"gate block message missing from stdout; got {proc.stdout[-200:]!r}"

    # --dry-run bypasses the gate (exits 0, never touches GPU, USER_OK
    # is irrelevant in dry-run path)
    proc_dry = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "USER_OK"},
        timeout=30,
    )
    assert proc_dry.returncode == 0, (
        f"--dry-run must exit 0 even without USER_OK; got rc={proc_dry.returncode}. "
        f"stdout={proc_dry.stdout[-200:]!r} stderr={proc_dry.stderr[-200:]!r}"
    )
    # Dry-run path must exit BEFORE the gate (per script flow at line 92-127)
    # — verified by absence of "USER OK REQUIRED" in dry-run stdout
    assert "USER OK REQUIRED" not in proc_dry.stdout, (
        f"dry-run path leaked into USER_OK gate; dry-run should exit at "
        f"the if-block before reaching the gate. stdout={proc_dry.stdout[-200:]!r}"
    )
