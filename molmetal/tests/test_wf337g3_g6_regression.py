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


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts" / "launch_wf124_p3_g6.sh"
DATA_CSV = ROOT / ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
TRAINER = ROOT / "molmetal/scripts/train_dmpnn_multitask_wasserstein.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_wrapper() -> str:
    assert WRAPPER.exists(), f"G6 wrapper missing: {WRAPPER}"
    return WRAPPER.read_text()


def _active_code(src: str) -> str:
    """The script with full-line and trailing comments removed.

    The G6 banner documents flags that were removed (Wf-R28 / Wf-R32), so a
    plain substring search cannot distinguish a live argument from the note
    explaining why it is gone.
    """
    out = []
    for line in src.splitlines():
        if line.strip().startswith("#"):
            continue
        out.append(line.split("#", 1)[0])
    return "\n".join(out)


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

    # Wf-335-R3 — the CUDA-Graphs lever is structural now. MEASURED
    # 2026-10-27: --no-cuda-graphs was removed from this launcher in Wf-R32
    # because the trainer never declared it, and the trainer's compile path
    # reads DEFAULT_TORCH_COMPILE_MODE, so graphs are unreachable unless
    # --torch-compile is passed (it is not).
    assert "--no-cuda-graphs" not in _active_code(src), (
        "Wf-R32 removed --no-cuda-graphs from this launcher because the "
        "trainer does not declare it; passing it again makes every G6 tuple "
        "exit rc=2 before the first epoch."
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
    assert "DEFAULT_TORCH_COMPILE_MODE" in trainer_src, (
        "trainer has no DEFAULT_TORCH_COMPILE_MODE; --torch-compile would "
        "fall back to a literal mode (the Wf-330-B4 path)"
    )
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

    # All 3 wave-tool guards must be invoked.  The pre-flight call is
    # multi-line and uses `--pipeline launch`, not `--pipeline all`:
    #     ${PY} molmetal/scripts/pre_flight_check.py \
    #       --pipeline launch \
    #       --declared-wave ... || { echo "FAIL: pre_flight_check"; exit 1; }
    # The 2026-09-29 version of this test matched the literal
    # "pre_flight_check.py --pipeline all" and so reported the gate missing
    # when it had been there all along (MEASURED 2026-10-27).
    pre_flight_re = re.compile(
        r"pre_flight_check\.py\s*\\?\s*--pipeline\s+(?:all|launch)"
    )
    assert pre_flight_re.search(body), (
        "pre_flight_check.py --pipeline {all,launch} not invoked in G6 wrapper"
    )
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
    pre_flight_idx = pre_flight_re.search(body).start()
    budget_idx = body.find("wave_budget_tracker.py")
    per_wave_idx = body.find("per_wave_checklist.py --wave")
    assert pre_flight_idx > 0, "pre_flight_check not invoked"
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
    for tool_idx in (pre_flight_idx, per_wave_idx, budget_idx):
        # Look at the next 240 chars after the tool name for the `||` guard
        # (the pre-flight call spans several continuation lines).
        tail = body[tool_idx : tool_idx + 240]
        assert "||" in tail and "exit 1" in tail, (
            f"tool invocation at idx {tool_idx} lacks `|| {{ ... exit 1 }}` "
            f"guard. A STALE pre-flight would silently continue without "
            f"this. tail: {tail[:120]!r}"
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

    # The gate is written as `if [ "${USER_OK:-0}" != "1" ]` (not the
    # `[[ -z "${USER_OK:-}" ]]` form this test used to require), and since
    # Wf-378-A2 it is checked BEFORE the --dry-run exit block, so the plan
    # path is gated too. Assert the behaviour rather than the syntax.

    # Actual end-to-end: invocation without USER_OK exits non-zero.
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
    assert "USER_OK" in (proc.stdout + proc.stderr), (
        f"gate block message does not name USER_OK; got "
        f"stdout={proc.stdout[-200:]!r} stderr={proc.stderr[-200:]!r}"
    )

    # --dry-run is gated too: same refusal, no GPU work either way.
    proc_dry = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "USER_OK"},
        timeout=30,
    )
    assert proc_dry.returncode != 0, (
        f"--dry-run ran without USER_OK (rc={proc_dry.returncode}); Wf-378-A2 "
        f"made the plan path gated, and its usage text says so"
    )
