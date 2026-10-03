"""Pytest: Wf-367-A2 — close the pre-flight FAIL cascade on
``p3_wf343c2_per_wave_runtime_check``.

Per Wf-358-D1 + Wf-362-B1 + Wf-364-A1 + Wf-365-A2: after Wf-364-A1 closed
the upstream ``p2_tests_pass`` FAIL, the remaining FAIL on
``p3_wf343c2_per_wave_runtime_check`` was caused by the pre-flight gate
invoking ``wf343c2_per_wave_check.py`` with both ``--skip-live`` and
``--strict``.  The ``--skip-live`` flag legitimately produces 3 SKIPs
(hyperparams_config_complete + gpu_vram_available + disk_free_space),
and ``--strict`` converts any SKIP into FAIL (exit 2).

Wf-367-A2 fix: remove ``--strict`` from the pre-flight gate's
``gate_wf343c2_per_wave_runtime_check`` invocation in
``pre_flight_check.py``.  The wave launcher / CLI path keeps ``--strict``
for explicit user opt-in (e.g. ``wf343c2_per_wave_check.py --strict``
from a wave bash script).

This module ships ONE regression test that pins the Wf-367-A2 contract:

1. ``test_gate_wf343c2_per_wave_runtime_check_no_strict_invocation`` —
   subprocess-invokes the pre-flight gate function directly with the
   production default args (no --strict, --skip-live only) and asserts
   that it returns PASS with rc=0.  The gate's invocation does NOT
   pass --strict (per Wf-367-A2 fix).

CPU-only.  Wall budget: < 30 s on RTX 5090 / Blackwell.

Source: Wf-367-A2 closeout + Wf-364-A1 §4.2 root-cause recommendation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
PREFLIGHT_CHECK_SCRIPT = SCRIPT_DIR = SCRIPTS_DIR / "pre_flight_check.py"
VENV_PY = Path("/home/hugo/codes/mol-metal/.venv/bin/python")

# Pre-flight invocation tokens that we want to ensure are NOT used by the
# gate.  Per Wf-367-A2 fix: --strict was removed from
# gate_wf343c2_per_wave_runtime_check's args list because --skip-live
# legitimately produces 3 SKIPs (which --strict converts to FAIL exit 2).
DISALLOWED_GATE_INVOCATION_TOKENS = ("--strict",)


@pytest.mark.skipif(not VENV_PY.exists(), reason="vlm venv not on disk")
def test_gate_wf343c2_per_wave_runtime_check_no_strict_invocation() -> None:
    """The pre-flight gate ``p3_wf343c2_per_wave_runtime_check`` invokes
    ``wf343c2_per_wave_check.py`` WITHOUT ``--strict`` so that the
    legitimate ``--skip-live`` SKIPs do not cascade to FAIL.

    Wf-367-A2 contract: drop ``--strict`` from the pre-flight gate's
    invocation.  Previously, ``--skip-live --strict`` produced rc=2
    (SKIP-as-FAIL) and the gate reported FAIL even though all 4 PASS-able
    sub-checks (preflight_freshness_24h + preflight_no_fail_gates +
    yuelbond_decoder_registered + wave_budget_remaining_nonneg) PASSed.
    """
    assert PREFLIGHT_CHECK_SCRIPT.is_file(), (
        f"pre_flight_check.py missing: {PREFLIGHT_CHECK_SCRIPT}"
    )
    # Use the pre-flight script's GateResult machinery by invoking the
    # gate function directly via a tiny driver.  Easier: subprocess-run
    # the pre-flight script with --pipeline all and grep the gate's
    # evidence string for the disallowed tokens.  But that requires a
    # full fresh-run (~150s wall).  Instead, we exercise the gate
    # function directly via a one-shot inline driver.
    driver_script = SCRIPTS_DIR / "_wf367a2_inline_gate_runner.py"
    driver_script.write_text(
        '"""Inline driver for Wf-367-A2 regression test (deleted after run)."""\n'
        'import sys\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n'
        'from molmetal.scripts.pre_flight_check import gate_wf343c2_per_wave_runtime_check\n'
        'result = gate_wf343c2_per_wave_runtime_check(\n'
        '    "p3_wf343c2_per_wave_runtime_check",\n'
        '    "P3", "wf347a1",\n'
        '    Path(__file__).resolve().parent / "wf343c2_per_wave_check.py",\n'
        ')\n'
        'print(f"STATUS={result.status}")\n'
        'print(f"EVIDENCE={result.evidence}")\n'
    )
    try:
        proc = subprocess.run(
            [str(VENV_PY), str(driver_script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
    finally:
        if driver_script.exists():
            driver_script.unlink()
    assert proc.returncode == 0, (
        f"inline gate driver crashed (rc={proc.returncode}); "
        f"stdout={proc.stdout[-500:]} stderr={proc.stderr[-500:]}"
    )
    # Parse STATUS / EVIDENCE from stdout
    out_lines = proc.stdout.splitlines()
    status_line = next((l for l in out_lines if l.startswith("STATUS=")), "")
    evidence_line = next((l for l in out_lines if l.startswith("EVIDENCE=")), "")
    assert status_line, f"STATUS= line missing from driver stdout: {proc.stdout!r}"
    assert evidence_line, (
        f"EVIDENCE= line missing from driver stdout: {proc.stdout!r}"
    )
    status = status_line.split("=", 1)[1]
    evidence = evidence_line.split("=", 1)[1]
    # Primary contract: the gate must PASS (per Wf-367-A2 fix).
    assert status == "PASS", (
        f"Wf-367-A2 contract VIOLATED: gate status={status!r} (expected PASS). "
        f"evidence={evidence!r}. This is the remaining FAIL from "
        f"Wf-364-A1 §4.2 — the pre-flight gate was invoking "
        f"wf343c2_per_wave_check.py with --skip-live --strict which produces "
        f"3 SKIPs → exit 2 → FAIL. Wf-367-A2 dropped --strict so the gate "
        f"now reports PASS_WITH_SKIP (exit 0)."
    )
    # Secondary contract: the gate function source MUST NOT include a
    # literal '--strict' token in the subprocess args list (other than in
    # comments).  We assert this by reading the function source and
    # ensuring the args=[ ... ] literal that follows the 'args = [' line
    # within ``gate_wf343c2_per_wave_runtime_check`` does NOT contain
    # ``"--strict"`` (with quotes — i.e. as a subprocess arg, not as a
    # code comment or docstring token).
    pre_flight_src = PREFLIGHT_CHECK_SCRIPT.read_text()
    # Find the gate function body (from 'def gate_wf343c2_per_wave_runtime_check'
    # to the next 'def ' at column 0).
    import re
    m = re.search(
        r"^def gate_wf343c2_per_wave_runtime_check\(.*?(?=\n^def |\Z)",
        pre_flight_src,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert m, "could not locate gate_wf343c2_per_wave_runtime_check in source"
    gate_src = m.group(0)
    # Strip Python comments and docstrings (between triple-quotes) so we
    # only inspect actual code tokens.  Crude but effective for our
    # purpose — we just want to confirm '--strict' is not in the args
    # list literal.
    code_only = re.sub(r'"""[\s\S]*?"""', "", gate_src)
    code_only = re.sub(r"#.*$", "", code_only, flags=re.MULTILINE)
    assert '"--strict"' not in code_only, (
        f"Wf-367-A2 contract VIOLATED: gate function source contains "
        f"'\"--strict\"' as a subprocess arg token. The pre-flight gate "
        f"must NOT pass --strict (per Wf-367-A2 fix). Drop it from the "
        f"args=[ ... ] list in gate_wf343c2_per_wave_runtime_check."
    )