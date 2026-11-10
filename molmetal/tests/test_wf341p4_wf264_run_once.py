"""Wf-341-P4 — Wf-264 CFM property panel run-once wrapper tests.

====================================================================
Wf-341-P4 (2026-09-29) — Run-once wrapper audit + tests
====================================================================

Owner: Hugo Lee + ultracode agent
Date: 2026-09-29
Parent: Wf-334-U3 (Wf-264 launcher WRITTEN) + Wf-336-U3 (USER_OK-gated)
        + Wf-338-F1 (run-once wrapper WRITTEN)
Source: Wf-341 ultracode residual-gap closure wave — "address ALL
        discovered issues; workflow can be long".

These tests are CPU-only — they inspect the run-once wrapper's source
to validate:
  1. parse correctness (bash -n),
  2. pre-flight integration (TODO-207 §1 gate),
  3. Wf-328 honest-negative framing contract (RDKit post-hoc, NOT the
     D-MPNN Wasserstein trainer — so Wf-328 flags are NA here),
  4. USER_OK gate enforcement (kill_then_adapt standing directive).

Per AGENTS.md §5.2 (honest framing):
  - MEASURED (2026-09-29): the Wf-264 panel ran successfully on the
    Wf-188 MetalCytoToxDB ligand population (n=26,801) per
    molmetal/reports/wf264_panel/panel_report.md.
  - MEASURED: Wf-338-F1 run-once wrapper WRITTEN at
    scripts/run_wf264_cfm_property_panel.sh.
  - HONEST_NEGATIVE_RDKIT: Wf-328 optimization flags (--torch-compile /
    --bf16 / --tf32 / --w2-gradient-mode / --num-workers) do NOT apply
    to the property panel (RDKit SA + QED compute on pre-existing
    SMILES).  The wrapper documents this; tests assert the contract.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FULL_LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "launch_wf264_cfm_property_panel.sh"
)
RUN_ONCE_WRAPPER = ROOT / "scripts" / "run_wf264_cfm_property_panel.sh"
PRE_FLIGHT = ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
PANEL_SCRIPT = ROOT / "molmetal" / "scripts" / "cfm_property_panel.py"
SMILES_CSV = ROOT / "molmetal" / "data" / "metalcytotox_rdkit_features.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    assert path.exists(), f"file missing: {path}"
    return path.read_text()


def _executable_body(src: str) -> str:
    """Index of the first executable line after the header comment block.

    Skip the shebang + header docstring (lines starting with #) and the
    function-definition blocks above `main()`.  Fall back to the full
    source if the helper conventions do not match.
    """
    main_idx = src.find("\nmain()")
    if main_idx < 0:
        main_idx = src.find("main()")
    if main_idx > 0:
        return src[main_idx:]
    return src


# ---------------------------------------------------------------------------
# Test 1: run-once launcher parses (bash -n + structural assertions)
# ---------------------------------------------------------------------------


def test_run_once_launcher_parses_and_structure():
    """Run-once wrapper must:
    - exist at scripts/run_wf264_cfm_property_panel.sh,
    - parse via bash -n (no syntax errors),
    - declare set -uo pipefail (or stricter) for safe scripting,
    - reference PROJECT_ROOT + the full Wf-264 launcher it wraps,
    - exit-code contract: 0=PASS / 1=FAIL / 2=SKIP / 3=USER_OK_DENIED.
    """
    assert RUN_ONCE_WRAPPER.exists(), f"run-once wrapper missing at {RUN_ONCE_WRAPPER}"
    assert (
        FULL_LAUNCHER.exists()
    ), f"full launcher missing at {FULL_LAUNCHER} (run-once needs it)"

    src = _read(RUN_ONCE_WRAPPER)

    # bash -n syntax check
    proc_syntax = subprocess.run(
        ["bash", "-n", str(RUN_ONCE_WRAPPER)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (
        proc_syntax.returncode == 0
    ), f"wrapper bash syntax error: stderr={proc_syntax.stderr!r}"

    # strict mode (set -uo pipefail or stricter)
    assert (
        re.search(r"set\s+-[a-z]*[uo]", src) or "set -euo pipefail" in src
    ), "wrapper must declare strict mode (set -uo pipefail or stricter)"

    # structural references
    assert "PROJECT_ROOT" in src, "PROJECT_ROOT constant missing"
    assert (
        "launch_wf264_cfm_property_panel.sh" in src
    ), "full Wf-264 launcher reference missing"
    assert (
        "pre_flight_check.py" in src
    ), "pre_flight_check.py invocation missing (TODO-207 §1 gate)"

    # exit-code contract documented in header
    for code, label in (
        ("0", "PASS"),
        ("1", "FAIL"),
        ("2", "SKIP"),
        ("3", "USER_OK"),
    ):
        assert re.search(
            rf"\b{code}\b\s*=\s*{label}", src, re.IGNORECASE
        ), f"exit-code {code}={label} contract missing from header"


# ---------------------------------------------------------------------------
# Test 2: pre-flight integration (TODO-207 §1 gate wired correctly)
# ---------------------------------------------------------------------------


def test_preflight_integration_wired():
    """Run-once wrapper must invoke pre_flight_check.py --pipeline all
    with the correct pipeline flag (TODO-207 §1 gate).  The gate must
    be wired BEFORE the actual panel run (kill_then_adapt ordering
    invariant per Wf-300C + Wf-306D — but here USER_OK gate is the
    outer ordering; pre-flight fires after USER_OK and before launch).
    """
    src = _read(RUN_ONCE_WRAPPER)
    body = _executable_body(src)

    # pre-flight check wired with correct flag
    assert (
        "pre_flight_check.py --pipeline all" in src
    ), "pre_flight_check.py --pipeline all invocation missing from wrapper"
    assert "--pipeline all" in src, "--pipeline all flag missing (TODO-207 §1 gate)"

    # Pre-flight is invoked via the project's venv python (per AGENTS.md
    # §5.1: .venv/bin/python, NOT the unrelated /home/hugo/cades/CadButEaas
    # VLM venv).
    assert "/home/hugo/codes/mol-metal/.venv/bin/python" in src or (
        "${PY:-" in src and ".venv/bin/python" in src
    ), (
        "wrapper must use /home/hugo/codes/mol-metal/.venv/bin/python "
        "(AGENTS.md §5.1 venv convention)"
    )

    # Preflight result is checked (FAIL → abort)
    preflight_idx = body.find("pre_flight_check.py --pipeline all")
    assert preflight_idx > 0, "preflight invocation not found in executable body"
    # Fail-fast branch: a `if > ! ...` OR an explicit FAIL log must follow
    fail_after = body[preflight_idx:].find('"PRE_FLIGHT FAIL"')
    fail_after = fail_after if fail_after > 0 else body[preflight_idx:].find("exit 1")
    assert fail_after > 0, (
        "no fail-fast branch after preflight invocation — wrapper must "
        "abort if pre-flight fails (TODO-207 §1.4 standing rule)"
    )

    # The full launcher also re-enforces pre-flight (orchestration
    # determinism: wrapper + launcher each gate independently).
    full_src = _read(FULL_LAUNCHER)
    assert (
        "pre_flight_check.py --pipeline all" in full_src
    ), "full launcher must also re-invoke pre-flight (defence in depth)"


# ---------------------------------------------------------------------------
# Test 3: Wf-328 fixes / HONEST_NEGATIVE for RDKit post-hoc panel
# ---------------------------------------------------------------------------


def test_wf328_honest_negative_for_rdkit_post_hoc():
    """Wf-328 optimization flags (--torch-compile / --bf16 / --tf32 /
    --w2-gradient-mode / --num-workers) DO NOT apply to cfm_property_panel.py
    (RDKit SA + QED post-hoc scorer).  Per AGENTS.md §5.2 honest-framing
    mandate, this HONEST_NEGATIVE must be documented in the wrapper header
    + dry-run echo.  This test asserts the contract.

    For the run-once wrapper: it invokes the FULL launcher (which is
    where the honest-negative contract lives); the run-once itself does
    not duplicate the documentation.  We assert:
      (a) the run-once wrapper does NOT silently wire Wf-328 flags
          (i.e. those flags should NOT appear in the wrapper), AND
      (b) the run-once wrapper delegates to the full launcher (where
          the Wf-328 honest-negative contract IS present).
    """
    run_once_src = _read(RUN_ONCE_WRAPPER)
    full_src = _read(FULL_LAUNCHER)

    # (a) Run-once must NOT silently wire Wf-328 flags.
    # The Wf-328 flags are GPU / torch.compile / DataLoader concepts.
    # cfm_property_panel.py is RDKit-only — it has no torch.compile /
    # BF16 / TF32 / W₂ / DataLoader code path.  If any of these flags
    # appear in the run-once wrapper, that's a bug (the wrapper would
    # be silently optimising a CPU-only RDKit pipeline).
    for flag in (
        "--torch-compile",
        "--bf16",
        "--tf32",
        "--w2-gradient-mode",
    ):
        assert flag not in run_once_src, (
            f"Wf-328 flag {flag} unexpectedly appears in run-once wrapper — "
            f"this is a CPU-only RDKit panel; Wf-328 flags are NA here "
            f"(HONEST_NEGATIVE per AGENTS.md §5.2)"
        )
    # --num-workers is a special case: it's a DataLoader concept that
    # does NOT apply because the panel processes SMILES sequentially.
    # If --num-workers appears in the run-once wrapper, it should be
    # in a documentation context (a comment explaining why it's NOT
    # wired), NOT as a CLI flag passed to the script.
    if "--num-workers" in run_once_src:
        # Acceptable only if it's in a comment line
        nw_lines = [
            line for line in run_once_src.split("\n") if "--num-workers" in line
        ]
        for line in nw_lines:
            stripped = line.lstrip()
            assert stripped.startswith("#"), (
                f"--num-workers appears as active code (not a comment) in "
                f"run-once wrapper: {line!r}"
            )

    # (b) Full launcher (which the run-once delegates to) MUST document
    # the Wf-328 NA contract.
    for flag in (
        "--torch-compile",
        "--bf16",
        "--tf32",
        "--w2-gradient-mode",
        "--num-workers",
    ):
        assert flag in full_src, (
            f"Wf-328 flag {flag} missing from full launcher header "
            f"(honest-negative contract per AGENTS.md §5.2)"
        )
    # Honest-negative framing language in full launcher
    assert ("NA" in full_src) or (
        "do NOT" in full_src
    ), "Wf-328 NA / do-not-apply framing language missing from full launcher"
    # The Wf-328 NA echo must surface in dry-run output
    assert (
        "Wf-328 optimization flags: NA for this panel" in full_src
    ), "Wf-328 NA dry-run echo missing from full launcher source"


# ---------------------------------------------------------------------------
# Test 4: USER_OK gate (kill_then_adapt standing directive)
# ---------------------------------------------------------------------------


def test_user_ok_gate_enforced_before_real_launch():
    """Run-once wrapper must enforce USER_OK=1 BEFORE any real work
    (kill_then_adapt standing directive per AGENTS.md §5.1 + Wf-300C
    pattern + Wf-306D).

    Invariants:
      - USER_OK=1 unset (or != "1") → exit 3 (USER_OK_GATE_DENIED),
      - The deny message must be logged to the wrapper's log file,
      - --status query is USER_OK-exempt (read-only state),
      - USER_OK gate fires BEFORE pre-flight check (gate-first ordering
        to fail fast without expensive pre-flight work).
    """
    src = _read(RUN_ONCE_WRAPPER)
    body = _executable_body(src)

    # (1) USER_OK gate is declared (with text or `[ "${USER_OK:-0}" != "1" ]` pattern)
    assert "USER_OK" in src, "USER_OK gate text missing from wrapper"
    assert (
        '[ "${USER_OK:-0}" != "1" ]' in body
        or "check_user_ok" in body
        or "USER_OK_GATE_DENIED" in body
    ), (
        "USER_OK gate enforcement not found (expected `[ ${USER_OK:-0} != 1 ]` "
        "or `check_user_ok` call or USER_OK_GATE_DENIED marker)"
    )

    # (2) kill_then_adapt reference (AGENTS.md §5.1 standing directive)
    assert (
        "kill_then_adapt" in src
    ), "kill_then_adapt standing directive reference missing"

    # (3) Gate-first ordering: USER_OK gate fires BEFORE pre-flight check
    # (so the user can deny without burning time on pre-flight).
    # Prefer the executable-body pattern (`[ "${USER_OK:-0}" != "1" ]`)
    # which only appears in the actual gate code — not the
    # `USER_OK_GATE_DENIED` log message which appears in BOTH the
    # header (line 28-29 narrative) AND the gate block (line 115 log
    # line).  We need the actual gate site, not the first occurrence.
    if '[ "${USER_OK:-0}" != "1" ]' in body:
        gate_idx = body.find('[ "${USER_OK:-0}" != "1" ]')
    elif "check_user_ok" in body:
        gate_idx = body.find("check_user_ok")
    else:
        gate_idx = body.find("USER_OK_GATE_DENIED")
    # Skip past the header narrative by starting the preflight search
    # after the first executable code line.  We use the gate_idx (which
    # we just located in the executable body) as a lower bound — the
    # preflight invocation must appear AFTER the gate.
    preflight_idx = body.find("pre_flight_check.py --pipeline all", gate_idx)
    assert gate_idx > 0, "USER_OK gate site not found in executable body"
    assert preflight_idx > 0, "preflight invocation not found in executable body"
    assert gate_idx < preflight_idx, (
        f"USER_OK gate (idx {gate_idx}) must fire BEFORE pre-flight "
        f"(idx {preflight_idx}) per kill_then_adapt ordering invariant"
    )

    # (4) End-to-end: invocation without USER_OK exits with rc=3.
    # We strip USER_OK from env explicitly (in case the test harness
    # has USER_OK set globally).
    env_no_user_ok = {k: v for k, v in os.environ.items() if k != "USER_OK"}
    proc = subprocess.run(
        ["bash", str(RUN_ONCE_WRAPPER)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env_no_user_ok,
        timeout=15,
    )
    assert proc.returncode == 3, (
        f"wrapper exited rc={proc.returncode} without USER_OK; "
        f"expected rc=3 (USER_OK_GATE_DENIED). "
        f"stdout={proc.stdout[-200:]!r}"
    )
    assert (
        "USER_OK_GATE_DENIED" in proc.stdout
    ), f"gate-deny message missing from stdout; got {proc.stdout[-200:]!r}"

    # (5) --status query is USER_OK-exempt and exits 0.
    proc_status = subprocess.run(
        ["bash", str(RUN_ONCE_WRAPPER), "--status"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env_no_user_ok,
        timeout=15,
    )
    assert proc_status.returncode == 0, (
        f"--status exited rc={proc_status.returncode}; expected 0. "
        f"stdout={proc_status.stdout[-200:]!r}"
    )
    # Status JSON must include honest-negative caveat for Vina sub-panel
    assert (
        "BLOCKED on meeko" in proc_status.stdout or "Wf-299B" in proc_status.stdout
    ), (
        f"Vina honest-negative caveat missing from --status output; "
        f"got {proc_status.stdout[-300:]!r}"
    )
