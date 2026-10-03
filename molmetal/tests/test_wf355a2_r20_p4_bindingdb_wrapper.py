"""Wf-355-A2 — R20 §4 BindingDB 1.07M launch wrapper tests.

Owner: Hugo Lee
Date: 2026-09-29
Parent: TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §3-§4
Sister: TODO/completed/354b2_r20_p4_binding_db_wrapper.md (HONEST-NEGATIVE predecessor)
Per Wf-355-A2 §4: 4 tests covering USER_OK gate, wrapper bash syntax,
download script bash syntax, and pre-flight + per-wave + data quality
wiring.

These tests are CPU-only — no torch.cuda calls, no actual training,
no BindingDB network download. They inspect the wrapper script's
source + execute it in --dry-run mode (which requires USER_OK=YES to
pass the gate) to validate structural correctness.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = ROOT / "scripts" / "launch_wf310e_r20_p4_bindingdb_1.07M.sh"
DOWNLOAD = ROOT / "scripts" / "download_bindingdb_1.07M.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    assert path.exists(), f"file missing: {path}"
    return path.read_text()


def _body_start(src: str) -> int:
    """Index of the first executable line in the wrapper (skip header
    comment block + shebang). Tries multiple markers since wrapper uses
    `set -euo pipefail`."""
    for marker in ("\nset -euo pipefail\n", "\nset -e\n"):
        idx = src.find(marker)
        if idx > 0:
            return idx + len(marker)
    return 0


# ---------------------------------------------------------------------------
# Test 1: USER_OK gate blocks real launch (without USER_OK → exit non-zero)
# ---------------------------------------------------------------------------


def test_user_ok_gate_enforced():
    """Without USER_OK env var, the wrapper must exit non-zero BEFORE
    touching GPU. --dry-run does NOT bypass the gate (per Wf-294B
    kill_then_adapt + AGENTS.md §5.1 standing GPU grant)."""
    src = _read(WRAPPER)
    body = src[_body_start(src) :]

    # USER_OK gate text present in wrapper source
    assert "USER_OK" in src, "USER_OK gate text missing from wrapper"
    assert "kill_then_adapt" in src, "kill_then_adapt reference missing"

    # Gate is enforced with explicit YES comparison (not just "set")
    assert (
        'USER_OK != "YES"' in body or "USER_OK:-}" in body
    ), "USER_OK gate enforcement (== 'YES' check) missing"

    # End-to-end: invocation without USER_OK exits non-zero
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "USER_OK"},
        timeout=15,
    )
    assert proc.returncode != 0, (
        f"wrapper exited 0 without USER_OK; expected gate block. "
        f"stdout={proc.stdout[-200:]!r} stderr={proc.stderr[-200:]!r}"
    )
    # Gate refusal message
    assert (
        "USER_OK" in proc.stderr or "USER_OK" in proc.stdout
    ), "USER_OK gate refusal message missing from output"


# ---------------------------------------------------------------------------
# Test 2: launch wrapper bash syntax is valid
# ---------------------------------------------------------------------------


def test_launch_wrapper_bash_syntax():
    """The launch wrapper must parse cleanly under bash -n (no execution)."""
    assert WRAPPER.exists(), f"launch wrapper missing: {WRAPPER}"
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"bash -n syntax check FAILED on {WRAPPER}: " f"stderr={result.stderr!r}"
    )

    # Wrapper declares set -euo pipefail (per project convention)
    src = _read(WRAPPER)
    assert "set -euo pipefail" in src, "wrapper missing strict mode (set -euo pipefail)"

    # Wrapper is executable
    import stat

    mode = WRAPPER.stat().st_mode
    assert mode & stat.S_IXUSR, f"wrapper not executable: mode={oct(mode)}"


# ---------------------------------------------------------------------------
# Test 3: download script bash syntax is valid
# ---------------------------------------------------------------------------


def test_download_script_bash_syntax():
    """The BindingDB download script must parse cleanly under bash -n."""
    assert DOWNLOAD.exists(), f"download script missing: {DOWNLOAD}"
    result = subprocess.run(
        ["bash", "-n", str(DOWNLOAD)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"bash -n syntax check FAILED on {DOWNLOAD}: " f"stderr={result.stderr!r}"
    )

    src = _read(DOWNLOAD)
    # Script declares strict mode
    assert (
        "set -euo pipefail" in src
    ), "download script missing strict mode (set -euo pipefail)"

    # Script references the wf175 metal filter
    assert (
        "wf175_bindingdb_metal_filter.py" in src
    ), "download script does not reference wf175 metal filter"

    # Script handles --verify-only / --force-download / --yes flags
    for flag in ("--verify-only", "--force-download", "--yes"):
        assert flag in src, f"download script missing CLI flag: {flag}"

    # Script is executable
    import stat

    mode = DOWNLOAD.stat().st_mode
    assert mode & stat.S_IXUSR, f"download script not executable: mode={oct(mode)}"


# ---------------------------------------------------------------------------
# Test 4: pre-flight + per-wave + data quality wired
# ---------------------------------------------------------------------------


def test_preflight_perwave_data_quality_wired():
    """The launch wrapper must invoke pre_flight_check, per_wave_checklist,
    and the data quality gate before any GPU work (TODO-207 v3 +
    Wf-294B kill_then_adapt ordering invariant)."""
    src = _read(WRAPPER)
    body = src[_body_start(src) :]

    # Look for actual function CALL sites (not header-comment mentions).
    # pre_flight_gate() is the actual call to pre_flight_check.py --pipeline all.
    pre_flight_call_idx = body.find("pre_flight_gate\n")
    if pre_flight_call_idx < 0:
        pre_flight_call_idx = body.find("pre_flight_gate ")
    assert (
        pre_flight_call_idx > 0
    ), "pre_flight_gate() function call missing from wrapper body"

    # budget_gate() function call site
    budget_call_idx = body.find("budget_gate\n")
    if budget_call_idx < 0:
        budget_call_idx = body.find("budget_gate ")
    assert budget_call_idx > 0, "budget_gate() function call missing"
    assert (
        pre_flight_call_idx < budget_call_idx
    ), "pre_flight_gate() must precede budget_gate()"

    # per_wave_gate() function call site
    per_wave_call_idx = body.find("per_wave_gate\n")
    if per_wave_call_idx < 0:
        per_wave_call_idx = body.find("per_wave_gate ")
    assert per_wave_call_idx > 0, "per_wave_gate() function call missing"
    assert (
        budget_call_idx < per_wave_call_idx
    ), "budget_gate() must precede per_wave_gate()"

    # USER_OK gate call site precedes pre_flight_gate (kill_then_adapt)
    user_ok_call_idx = body.find('user_ok_gate "${MODE}"')
    assert user_ok_call_idx > 0, "user_ok_gate call site missing"
    assert user_ok_call_idx < pre_flight_call_idx, (
        f"USER_OK gate call (idx {user_ok_call_idx}) must precede "
        f"pre_flight_gate call (idx {pre_flight_call_idx})"
    )

    # Data quality gate checks BindingDB raw TSV + filtered CSV
    assert (
        "BindingDB_All.tsv" in body
    ), "data quality gate does not check for raw BindingDB TSV"
    assert (
        "metal_binding_db_full.csv" in body
    ), "data quality gate does not check for metal-filtered CSV"
    # Gate G5: no NaN pIC50 sweep
    assert (
        "NaN" in body or "nan" in body.lower()
    ), "gate G5 (no NaN pIC50) not present in data quality check"

    # Wf-328 wire-up flags present (mirror the production trainer)
    for flag in ("--cpu-num-workers", "--w2-gradient-mode"):
        assert flag in body, f"Wf-328 wire-up flag missing: {flag}"

    # VRAM budget gate
    assert "VRAM" in body or "vram" in body.lower(), "VRAM budget check missing"

    # Acceptance gate (Affinity Pearson r)
    assert (
        "Pearson" in body or "0.65" in body
    ), "Affinity Pearson r acceptance gate not declared"
