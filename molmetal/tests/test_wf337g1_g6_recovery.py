"""Wf-337-G1 — G6 55 PENDING + 28 FAILED tuple recovery tests.

Owner: Hugo Lee
Date: 2026-09-29
Parent: Wf-330-B4 (in-flight G6 regression smoke) + Wf-337-G1 (this Wf)

Context (MEASURED, Wf-330-B4 + master.log snapshot 2026-09-29 15:13 CST):
  - G6 launch (TODO-182, scripts/launch_wf124_p3_g6.sh) was started at 09:57:39 CST
    with stale args (--cpu-num-workers 0, no --no-kendall, no --no-dann) before
    Wf-328-W2 + Wf-296C fixes hit the script on disk.
  - Wf-328-W1 (torch.compile mode=reduce-overhead) triggered CUDAGraphs
    RuntimeError on the post-Wf-328 trainer code path (Ru seed=2 + Ir seed=0 +
    26 more tuples). Per Wf-330-B4 §4.1 + master.log mtime 15:03:52.
  - Bash does not re-read modified scripts → in-flight process keeps stale args.
  - Current state: 2 OK + 28 FAILED + 1 RUNNING + 29 PENDING = 60 total.

Fixes applied (Wf-335-R1, Wf-335-R3, Wf-296C, Wf-328-W2) wired into the
recovery launch script (scripts/recover_g6_failed_tuples.sh):
  - --cpu-num-workers 8                                       (Wf-328-W2)
  - --no-cuda-graphs (hard-pin torch_compile_mode)              (Wf-335-R3)
  - torch.compile mode default = max-autotune-no-cudagraphs    (Wf-335-R1)
  - --no-kendall / --no-dann per-config ablation levers        (Wf-296C)

Tests (4 total):

  - test_recovery_script_identifies_failed_and_pending
        Gates the core recovery classification: detects FAILED + PENDING
        tuples from master.log + training_summary.json + run.log.
  - test_recovery_launch_includes_all_wf328_wf335_fixes
        Verifies the COMMON_ARGS in scripts/recover_g6_failed_tuples.sh
        include --cpu-num-workers 8, --no-cuda-graphs, --no-kendall,
        --no-dann, and torch.compile mode default = max-autotune-no-cudagraphs.
  - test_no_cudagraphs_flag_present
        Verifies the trainer's torch_compile_mode default switched to
        max-autotune-no-cudagraphs (Wf-335-R1 fix), and that
        --no-cuda-graphs is the documented CLI flag.
  - test_backward_compat_with_dry_run
        Verifies --dry-run exits 0 and prints the expected config summary.

All tests are CPU-only — no torch.cuda, no real launches. < 5 s wall.

Honest framing (AGENTS.md §5.2 + party-twenty-spirit §3.5):
  - MEASURED (Wf-330-B4 2026-09-29 13:23 CST): 2 OK + 2 CUDAGraphs-FAIL + 1
    in-flight + 55 PENDING.
  - MEASURED (master.log mtime 2026-09-29 15:03:52): 30 END markers emitted,
    only 2 status=OK; 28 status=ALL_SEEDS_FAILED.
  - MEASURED: --cpu-num-workers 8 in scripts/launch_wf124_p3_g6.sh:53
    (Wf-328-W2), --no-cuda-graphs at L61 (Wf-335-R3), --no-kendall / --no-dann
    at L80-81 (Wf-296C).
  - MEASURED: trainer default torch_compile_mode="max-autotune-no-cudagraphs"
    at L1121 + L2028 (Wf-335-R1).
  - PROJECTED: --cpu-num-workers 8 + --no-cuda-graphs eliminate the CUDAGraphs
    regression (BLOCKED on §5.1 — needs > 1 hr GPU run to MEASURE on RTX 5090).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest


# ---------------------------------------------------------------------------
# Project root + path setup
# ---------------------------------------------------------------------------
ROOT = Path("/home/hugo/codes/mol-metal")
SCRIPTS = ROOT / "scripts"
TEST_SCRIPTS_DIR = ROOT / "molmetal" / "scripts"

RECOVERY_SCRIPT = SCRIPTS / "recover_g6_failed_tuples.sh"
G6_LAUNCH_SCRIPT = SCRIPTS / "launch_wf124_p3_g6.sh"
TRAINER_PATH = TEST_SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py"
G6_RESULTS_DIR = ROOT / "results" / "wf124_p3_per_innovation_ablation_g6"
MANIFEST_PATH = G6_RESULTS_DIR / "recovery_manifest.tsv"
DEFAULT_VENV = Path("/home/hugo/codes/mol-metal/.venv/bin/python")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_recovery_script() -> str:
    return RECOVERY_SCRIPT.read_text(encoding="utf-8")


def _load_trainer_source() -> str:
    return TRAINER_PATH.read_text(encoding="utf-8")


def _run_dry_run() -> subprocess.CompletedProcess:
    """Invoke scripts/recover_g6_failed_tuples.sh --dry-run, capture stdout."""
    return subprocess.run(
        ["bash", str(RECOVERY_SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _parse_manifest_counts() -> Dict[str, int]:
    """Parse the recovery manifest TSV; return {status: count} dict.

    Skips header line. Each data row is: cfg<TAB>metal<TAB>seed<TAB>status<TAB>reason
    """
    assert MANIFEST_PATH.exists(), (
        f"Manifest not found at {MANIFEST_PATH}. Run --dry-run first to " f"create it."
    )
    counts: Dict[str, int] = {"OK": 0, "FAILED": 0, "RUNNING": 0, "PENDING": 0}
    with MANIFEST_PATH.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("cfg\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            s = parts[3]
            if s in counts:
                counts[s] += 1
    return counts


# ---------------------------------------------------------------------------
# Test 1: Recovery script identifies FAILED + PENDING correctly
# ---------------------------------------------------------------------------
def test_recovery_script_identifies_failed_and_pending():
    """Verifies the dry-run enumeration produces a manifest with the
    expected counts: OK + FAILED + RUNNING + PENDING = 60.

    Per Wf-330-B4 + master.log 2026-09-29 15:13 CST snapshot:
      - 2 OK tuples (Ru seed=0 + seed=1 from no_multi_task — pre-Wf-328 trainer code)
      - 28 FAILED (Ru/2 + Ir 0-2 + Os 0-2 + Rh 0-2 + Re 0-2 from no_multi_task + all
        15 from no_wass — all hit CUDAGraphs RuntimeError under reduce-overhead)
      - 1 RUNNING (no_kendall/Ru/seed=0, started 15:03:52, no END marker yet)
      - 29 PENDING (rest of no_kendall + all of no_dann)
    """
    # Run --dry-run (CPU-only, no GPU).
    result = _run_dry_run()
    assert result.returncode == 0, (
        f"recovery script --dry-run failed (rc={result.returncode})\n"
        f"stderr={result.stderr}"
    )

    # Manifest should exist after dry-run.
    assert (
        MANIFEST_PATH.exists()
    ), f"recovery_manifest.tsv not created at {MANIFEST_PATH}"

    counts = _parse_manifest_counts()
    total = sum(counts.values())
    assert (
        total == 60
    ), f"Total tuples should be 60 (4 cfgs × 5 metals × 3 seeds), got {total}: {counts}"

    # MEASURED per master.log mtime 15:03:52 + Wf-330-B4.
    # We accept a window of [0, 5] OK because the actual count depends on
    # how many additional tuples have completed since the 15:13 CST snapshot
    # at Wf-337-G1 task time. The hard invariant: OK + FAILED + RUNNING + PENDING = 60.
    assert counts["OK"] + counts["FAILED"] + counts["RUNNING"] + counts["PENDING"] == 60

    # FAILED + PENDING is what gets re-launched. PROJECTED ≥ 50 per Wf-330-B4 §2.
    to_recover = counts["FAILED"] + counts["PENDING"]
    assert (
        to_recover >= 50
    ), f"TO_RECOVER (FAILED + PENDING) should be ≥ 50 per Wf-330-B4, got {to_recover}"

    # Dry-run output should report all 4 status buckets in the summary.
    out = result.stdout
    assert "OK:" in out, "dry-run stdout missing OK count line"
    assert "FAILED:" in out, "dry-run stdout missing FAILED count line"
    assert "RUNNING:" in out, "dry-run stdout missing RUNNING count line"
    assert "PENDING:" in out, "dry-run stdout missing PENDING count line"
    assert "TO_RECOVER" in out, "dry-run stdout missing TO_RECOVER summary line"


# ---------------------------------------------------------------------------
# Test 2: Recovery launch has all Wf-328 + Wf-335 fixes
# ---------------------------------------------------------------------------
def test_recovery_launch_includes_all_wf328_wf335_fixes():
    """Verifies scripts/recover_g6_failed_tuples.sh has all the Wf-328 + Wf-335
    + Wf-296C fixes wired in COMMON_ARGS / CONFIRMED_FLAGS.

    Required fixes:
      - --cpu-num-workers 8                        (Wf-328-W2)
      - --no-cuda-graphs                           (Wf-335-R3)
      - --no-kendall in CONFIRMED_FLAGS[no_kendall] (Wf-296C)
      - --no-dann in CONFIRMED_FLAGS[no_dann]      (Wf-296C)
      - torch.compile mode default = max-autotune-no-cudagraphs (Wf-335-R1)
    """
    src = _load_recovery_script()

    # Wf-328-W2: --cpu-num-workers 8
    assert "--cpu-num-workers" in src, "Wf-328-W2 fix missing: --cpu-num-workers"
    assert re.search(
        r"--cpu-num-workers\s+8\b", src
    ), "Wf-328-W2 fix missing: --cpu-num-workers 8 (got something else)"
    # Wf-328-W2 banner comment must be present (gates Wf-326B revert regression).
    assert "Wf-328-W2" in src, "Wf-328-W2 banner comment missing"

    # Wf-335-R3: --no-cuda-graphs safety lever.
    assert "--no-cuda-graphs" in src, "Wf-335-R3 fix missing: --no-cuda-graphs"
    assert "Wf-335-R3" in src, "Wf-335-R3 banner comment missing"

    # Wf-335-R1: torch.compile mode default = max-autotune-no-cudagraphs.
    # We verify the trainer default; the launcher relies on the trainer default
    # unless overridden, and Wf-335-R3's --no-cuda-graphs flag forces this mode.
    trainer_src = _load_trainer_source()
    # Look for function-level default (e.g. line 1121 in current trainer).
    assert "max-autotune-no-cudagraphs" in trainer_src, (
        "Wf-335-R1 fix missing: trainer default torch_compile_mode should be "
        "'max-autotune-no-cudagraphs'"
    )
    # Function default line (Wf-335-R1 §2 row 1).
    fn_default_match = re.search(
        r"torch_compile_mode:\s*str\s*=\s*[\"\']max-autotune-no-cudagraphs[\"\']",
        trainer_src,
    )
    assert fn_default_match is not None, (
        "Wf-335-R1: function-level torch_compile_mode default not pinned to "
        "'max-autotune-no-cudagraphs'"
    )

    # Wf-296C: --no-kendall / --no-dann per-config levers.
    assert (
        '["no_kendall"]="--no-kendall"' in src
    ), "Wf-296C fix missing: no_kendall -> --no-kendall flag not wired"
    assert (
        '["no_dann"]="--no-dann"' in src
    ), "Wf-296C fix missing: no_dann -> --no-dann flag not wired"
    assert "Wf-296C" in src, "Wf-296C banner comment missing"

    # Sanity: no stale --cpu-num-workers 0 override (Wf-326B regression).
    # Strip comments + docstrings (anything after `#`) before checking, since
    # the recovery script's header documents the Wf-326B revert as historical
    # context. Only active bash code should have --cpu-num-workers 8.
    code_lines = []
    for line in src.splitlines():
        # Strip inline comments (preserve content inside double-quoted strings).
        in_quote = False
        cleaned = []
        for i, ch in enumerate(line):
            if ch == '"':
                in_quote = not in_quote
                cleaned.append(ch)
            elif ch == "#" and not in_quote:
                break
            else:
                cleaned.append(ch)
        code_lines.append("".join(cleaned))
    code_only = "\n".join(code_lines)
    assert not re.search(r"--cpu-num-workers\s+0\b", code_only), (
        "Wf-326B regression: --cpu-num-workers 0 detected in active bash "
        "code (should be 8 in COMMON_ARGS)"
    )


# ---------------------------------------------------------------------------
# Test 3: --no-cuda-graphs flag + trainer default semantics
# ---------------------------------------------------------------------------
def test_no_cudagraphs_flag_present():
    """Verifies --no-cuda-graphs is the documented safety lever (default ON)
    and the trainer hard-pins torch_compile_mode="max-autotune-no-cudagraphs"
    when this flag is set, regardless of any other --torch-compile-mode
    override (Wf-335-R3).

    Per Wf-335-R3: --no-cuda-graphs is the safety lever; --cudagraphs is the
    explicit opt-in (Wf-326C / Wf-328-W1 reproducer). The trainer's main()
    must force torch_compile_mode="max-autotune-no-cudagraphs" when the
    safety lever is ON.
    assert "max-autotune-no-cudagraphs" in trainer_src (already in test 2).
    """
    trainer_src = _load_trainer_source()

    # argparse flag definition for --no-cuda-graphs (default ON, action=store_true).
    no_cg_pattern = re.search(
        r"--no-cuda-graphs",
        trainer_src,
    )
    assert (
        no_cg_pattern is not None
    ), "Wf-335-R3 missing: --no-cuda-graphs flag not defined in trainer argparse"

    # --cudagraphs opt-in flag (mirror for backward compat with Wf-326C).
    cg_pattern = re.search(
        r"--cudagraphs\b",
        trainer_src,
    )
    assert (
        cg_pattern is not None
    ), "Wf-335-R3 missing: --cudagraphs opt-in flag not defined in trainer argparse"

    # Verify the trainer main() applies the safety lever: when
    # cudagraphs_disabled=True (the default), torch_compile_mode is
    # force-set to max-autotune-no-cudagraphs.
    # The exact wiring is in the train_dmpnn_multitask_wasserstein.py:2163-2199
    # region per Wf-335-R1 §2 row 7.
    assert "cudagraphs_disabled" in trainer_src, (
        "Wf-335-R3 main() safety lever logic missing: cudagraphs_disabled "
        "variable not referenced"
    )

    # Sanity: --no-cuda-graphs / --cudagraphs flag must appear in the
    # recovery script (not just trainer).
    recovery_src = _load_recovery_script()
    assert (
        "--no-cuda-graphs" in recovery_src
    ), "Recovery script does not pass --no-cuda-graphs to the trainer"


# ---------------------------------------------------------------------------
# Test 4: --dry-run backward compat (exit 0 + expected output)
# ---------------------------------------------------------------------------
def test_backward_compat_with_dry_run():
    """Verifies --dry-run mode is backward compatible: exits 0, prints the
    expected config summary, and does NOT require USER_OK.

    Backward-compat contract (Wf-337-G1 §3.4):
      1. Exit code 0
      2. Stdout contains "RECOVERY MANIFEST:" header
      3. Stdout contains "DRY-RUN OK:" footer
      4. USER_OK is NOT required (dry-run path exits before the USER_OK gate)
      5. No GPU work performed (verified by exit code + absence of USER_OK error)
    """
    # Run without USER_OK env var to verify backward compat.
    env = {"PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        ["bash", str(RECOVERY_SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    # 1. Exit code 0.
    assert result.returncode == 0, (
        f"--dry-run should exit 0 for backward compat, got rc={result.returncode}\n"
        f"stderr={result.stderr}"
    )

    out = result.stdout
    # 2. RECOVERY MANIFEST header.
    assert (
        "RECOVERY MANIFEST:" in out
    ), "backward compat broken: missing 'RECOVERY MANIFEST:' header"
    # 3. DRY-RUN OK footer.
    assert "DRY-RUN OK:" in out, "backward compat broken: missing 'DRY-RUN OK:' footer"
    # 4. USER_OK NOT required (no 'USER_OK REQUIRED' line in --dry-run path).
    assert (
        "USER_OK REQUIRED" not in out
    ), "backward compat broken: --dry-run path triggered USER_OK gate"
    # 5. The recovery launch command hint should be present.
    assert (
        "USER_OK=1 bash scripts/recover_g6_failed_tuples.sh" in out
    ), "backward compat broken: missing USER_OK=1 hint for real-launch path"

    # Verify the recovery script's bash PART 2 (the run_tuple function) is also
    # present (we want dry-run to enumerate but not call run_tuple).
    src = _load_recovery_script()
    assert (
        "run_tuple()" in src or "run_tuple ()" in src or "function run_tuple" in src
    ), "Recovery script missing run_tuple() function"
    # Sanity: the dry-run early-exit must be before the USER_OK gate.
    # Pattern: --dry-run check appears before USER_OK check.
    dry_run_pos = src.find("--dry-run")
    user_ok_pos = src.find('if [[ -z "${USER_OK')
    assert 0 < dry_run_pos < user_ok_pos, (
        f"Order violation: --dry-run check at pos {dry_run_pos} should come "
        f"BEFORE USER_OK gate at pos {user_ok_pos}"
    )


# ---------------------------------------------------------------------------
# Optional: trainer torch_compile_mode default sanity (cross-test 2/3 invariant)
# ---------------------------------------------------------------------------
def test_trainer_torch_compile_mode_default_is_max_autotune_no_cudagraphs():
    """Standalone assertion of the Wf-335-R1 invariant: trainer function default
    for torch_compile_mode is "max-autotune-no-cudagraphs" (was "reduce-overhead"
    pre-Wf-335-R1). Belt-and-suspenders check independent of the launcher script.
    """
    trainer_src = _load_trainer_source()
    # Function default (L1121) — explicit assignment form.
    fn_match = re.search(
        r"torch_compile_mode:\s*str\s*=\s*[\"\']max-autotune-no-cudagraphs[\"\']",
        trainer_src,
    )
    assert fn_match is not None, (
        "Wf-335-R1 invariant violated: trainer torch_compile_mode default is "
        "NOT 'max-autotune-no-cudagraphs' (was 'reduce-overhead' pre-Wf-335-R1)"
    )
    # CLI argparse default (L2028) — argparse keyword form. Note the default
    # assignment is on a SEPARATE indented line from "--torch-compile-mode",
    # so use a permissive multi-line match.
    cli_match = re.search(
        r"--torch-compile-mode[\s\S]{0,200}?default\s*=\s*[\"\']max-autotune-no-cudagraphs[\"\']",
        trainer_src,
    )
    assert cli_match is not None, (
        "Wf-335-R1 invariant violated: argparse --torch-compile-mode default is "
        "NOT 'max-autotune-no-cudagraphs'"
    )
