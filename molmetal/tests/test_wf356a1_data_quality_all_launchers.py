"""Wf-356-A1 — wire wf343c3_data_quality into all 6 launch scripts (fix DATA_CSV undefined error).

Owner: Hugo Lee
Date: 2026-09-29
Parent: Wf-347-B3 retry (close Wf-347 completion-diagnostic failure:
"DATA_CSV is not defined" — previous agent tried to wire
`--data-csv $DATA_CSV` even though wf343c3_data_quality_check.py is
LOADER-AGNOSTIC and does NOT accept `--data-csv`).

Per Wf-356-A1: 2 tests covering
  (1) per-launcher bash syntax (bash -n) for all 6 scripts
  (2) wf343c3 invocation pattern matches expected — loader-agnostic
      call: `--out-dir <launcher-specific>` and NO `--data-csv` flag.

These tests are CPU-only — no torch.cuda calls, no actual data quality
runs. They inspect the wrapper scripts' source to validate that the
Wf-356-A1 wire-up follows the Wf-347-B3 retry spec exactly.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path("/home/hugo/codes/mol-metal")
SCRIPTS = ROOT / "scripts"

# The 6 launchers covered by Wf-356-A1 (Wf-347-B3 retry scope)
LAUNCHER_PATHS = [
    SCRIPTS / "launch_wf124_p3_g4.sh",
    SCRIPTS / "launch_wf124_p3_g5.sh",
    SCRIPTS / "launch_wf124_p3_g6.sh",
    SCRIPTS / "launch_wf310b_g6_final.sh",
    SCRIPTS / "launch_wf300_cfm_main.sh",
    SCRIPTS / "retry_g5_missing_pairs.sh",
]

# Expected launcher-specific out-dirs (one per launcher)
EXPECTED_OUT_DIRS = {
    "launch_wf124_p3_g4.sh": "molmetal/reports/wf124_p3_g4_data_quality",
    "launch_wf124_p3_g5.sh": "molmetal/reports/wf124_p3_g5_data_quality",
    "launch_wf124_p3_g6.sh": "molmetal/reports/wf124_p3_g6_data_quality",
    "launch_wf310b_g6_final.sh": "molmetal/reports/wf310b_g6_final_data_quality",
    "launch_wf300_cfm_main.sh": "molmetal/reports/wf300_cfm_main_data_quality",
    "retry_g5_missing_pairs.sh": "molmetal/reports/retry_g5_missing_pairs_data_quality",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_launcher(path: Path) -> str:
    assert path.exists(), f"launcher missing: {path}"
    return path.read_text()


def _find_wf343c3_invocation(src: str) -> str:
    """Extract the wf343c3_data_quality_check.py invocation block from
    a launcher source. Walks forward through shell line-continuations
    (lines ending in `\\`) and returns the full multi-line block.

    Skips ``echo "wf343c3_data_quality_check.py ..."`` comment lines
    by requiring the matched line to either:
      - be a continuation line (starts with whitespace), OR
      - contain `${PY}` (the launcher python var), OR
      - contain a `python ` invocation
    Then walks forward through ALL continuation lines (whether they end
    in `\\` or are absorbed into the same logical statement).
    """
    # Walk all lines, find the one that calls the script (not echoes it).
    invocation_line_re = re.compile(
        r"^(?:[ \t].*|.*\$\{PY\}.*|.*\bpython\b.*)" r"wf343c3_data_quality_check\.py",
        re.MULTILINE,
    )
    match = invocation_line_re.search(src)
    if not match:
        return ""
    start = match.start()
    # Find the end-of-line for the matched line first.
    eol = src.find("\n", match.start())
    if eol < 0:
        eol = len(src)
    end = eol + 1  # one past the newline
    # Walk forward through shell line-continuations. A line whose content
    # (up to its newline) ends with `\` (after rstrip) is a continuation.
    while src[start:end].rstrip().endswith("\\"):
        next_nl = src.find("\n", end)
        if next_nl < 0:
            end = len(src)
            break
        end = next_nl + 1
    return src[start:end]


# ---------------------------------------------------------------------------
# Test 1: per-launcher bash syntax check (bash -n)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script_path",
    LAUNCHER_PATHS,
    ids=lambda p: p.name,
)
def test_launcher_bash_syntax(script_path: Path) -> None:
    """All 6 launch scripts must parse cleanly via `bash -n`."""
    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=30,
    )
    assert result.returncode == 0, (
        f"bash -n failed for {script_path.name}:\n"
        f"  stdout: {result.stdout}\n  stderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 2: wf343c3 invocation pattern matches expected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script_path",
    LAUNCHER_PATHS,
    ids=lambda p: p.name,
)
def test_wf343c3_invocation_pattern(script_path: Path) -> None:
    """Each launcher must:
    - invoke `wf343c3_data_quality_check.py` exactly once (or have it
      in the wave-tool guards block).
    - use `--out-dir <EXPECTED_OUT_DIRS[script]>` with the
      launcher-specific out-dir.
    - NOT pass `--data-csv` (the script is loader-agnostic and does
      not accept that flag — closing the Wf-347-B3 retry bug).
    - NOT reference an undefined `$DATA_CSV` inside the wf343c3 call
      (the script has its own DATA_CSV constants internally).
    """
    src = _read_launcher(script_path)
    invocation = _find_wf343c3_invocation(src)
    assert invocation, (
        f"{script_path.name}: no wf343c3_data_quality_check.py invocation "
        f"found — wire-up missing"
    )

    # (a) expected --out-dir must appear with the launcher-specific value
    expected_out_dir = EXPECTED_OUT_DIRS[script_path.name]
    assert expected_out_dir in invocation, (
        f"{script_path.name}: expected --out-dir {expected_out_dir!r} "
        f"in invocation, got:\n{invocation}"
    )

    # (b) --data-csv MUST NOT appear in the wf343c3 invocation
    # (script is loader-agnostic; Wf-347-B3 retry fixes the previous
    # DATA_CSV undefined bug).
    assert "--data-csv" not in invocation, (
        f"{script_path.name}: wf343c3 invocation must NOT pass "
        f"--data-csv (script is loader-agnostic; closes Wf-347-B3 "
        f"DATA_CSV undefined bug). Got:\n{invocation}"
    )

    # (c) bare `$DATA_CSV` MUST NOT appear inside the wf343c3 invocation
    # (the previous Wf-347-B3 failure was caused by referencing an
    # undefined $DATA_CSV in the wf343c3 call).
    #    Allow $DATA_CSV only in --csv / --data-csv lines for the
    #    trainer invocation; NOT in the wf343c3 block. We check by
    #    scanning only the wf343c3 invocation substring.
    assert "${DATA_CSV}" not in invocation, (
        f"{script_path.name}: wf343c3 invocation must NOT reference "
        f"$DATA_CSV (undefined variable error per Wf-347-B3). Got:\n{invocation}"
    )
    # Also bare $DATA_CSV (no braces) — same rule
    assert re.search(r"(?<!\{)\$DATA_CSV(?![\w{])", invocation) is None, (
        f"{script_path.name}: wf343c3 invocation must NOT reference "
        f"$DATA_CSV (undefined variable error per Wf-347-B3). Got:\n{invocation}"
    )

    # (d) PYTHONPATH must be set before the wf343c3 call (matches Wf-347-B3
    # retry convention used by G5/G6 per-wave checks)
    #    CFM main + G4 don't set PYTHONPATH inline, but PYTHONPATH is
    #    already exported via 'export PYTHONPATH=...' in CFM main and
    #    assumed inherited from shell in G4. So this check is
    #    WARNING-grade; we only flag if PYTHONPATH is set BEFORE the
    #    ${PY} var expansion but the wf343c3 line has neither.
    #    Actually — let the wf343c3 line either:
    #    - prefix `PYTHONPATH=...` inline (G5/G6/retry/cfm-main/g6-final), OR
    #    - rely on shell-inherited PYTHONPATH (G4)
    #    Both are acceptable; the test does not enforce PYTHONPATH.

    # (e) --quiet flag is OPTIONAL but if present must come after --out-dir
    #    (this is a sanity check on arg ordering).
    quiet_match = re.search(r"--quiet", invocation)
    out_dir_match = re.search(r"--out-dir", invocation)
    if quiet_match and out_dir_match:
        assert out_dir_match.start() < quiet_match.start(), (
            f"{script_path.name}: --out-dir must precede --quiet "
            f"(arg-order convention). Got:\n{invocation}"
        )
