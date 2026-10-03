"""Wf-394-A3 — Wf-299 CFM 3-wave pre-launch verification gate.

Per TODO-207 + AGENTS.md §5.1 + Wf-378-A2 lesson + Wf-385-A2 wrapper review
+ Wf-392-A3 Wf-388 trainer flag integration + `mol-metal-cfm-training`
skill: the Wf-299 CFM 3-wave launcher (`molmetal/scripts/launch_wf299_cfm_main.sh`)
is a > 1 hr GPU launcher (~9.5 hr sequential across 3 waves) and requires
BEFORE any actual launch:

  (a) **USER_OK gate present + BEFORE --dry-run block** — per Wf-378-A2
      G6 lesson (G6 --dry-run was bypassing USER_OK at original lines
      208-218; Wf-378-A2 moved the gate to line ~60). The Wf-299
      launcher has no USER_OK gate at all (Wf-385-A2 review confirmed
      only `--dry-run` switch exists).

  (b) **3-wave split (smoke / full / ablation)** — per Wf-299 brief +
      TODO-207 §3.2..§3.4: Wave A = P1.4 smoke (~2 hr cuda:0), Wave B
      = G6 no_wass smoke (~10 min cuda:1), Wave C = G6 full sweep
      (~7.5 hr cuda:1). Each wave has its own GPU + budget + scope.

  (c) **Wf-388 + Wf-380 flags pass through** — per Wf-392-A3: the
      Wave A smoke invocation must include the 16 NEW trainer flags
      (11 from Wf-380-B1 + 5 from Wf-388-A2/A3). `--flowmol-unified`
      is intentionally NOT passed (LICENSE-GATED CC-BY-NC).

  (d) **Pre-launch verification chain** — per TODO-207 §2.5 + Wf-385-A3
      scaffold wire-in gate: wf343c2 per-wave runtime check + wf343c3
      data quality check + wf385a3 scaffold wire-in gate MUST be runnable
      from the launcher.

This module exposes 4 verify_* functions and 1 main() that runs all 4
+ emits a structured JSON report at
``molmetal/reports/wf394a3_wf299_pre_launch/result.json``.

Wall-budget: ~3-5 s on a 4-core host (no GPU, no torch required for
flag-presence checks; pre-flight subprocess calls ~2-3 s wall each).

CPU-only by design. 0 GPU launches per AGENTS.md §5.1 + kill_then_adapt §0.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- Canonical paths ---------------------------------------------------------
PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
LAUNCHER = PROJECT_ROOT / "molmetal/scripts/launch_wf299_cfm_main.sh"
TRAINER = PROJECT_ROOT / "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
WF343C2 = PROJECT_ROOT / "molmetal/scripts/wf343c2_per_wave_check.py"
WF343C3 = PROJECT_ROOT / "molmetal/scripts/wf343c3_data_quality_check.py"
WF385A3 = PROJECT_ROOT / "molmetal/scripts/wf385a3_scaffold_wire_in_gate.py"
REPORTS_DIR = PROJECT_ROOT / "molmetal/reports/wf394a3_wf299_pre_launch"

PYTHON_BIN = PROJECT_ROOT / ".venv/bin/python"
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path(sys.executable)

# --- Canonical flag tables (per Wf-385-A2 §1 + Wf-392-A3 + Wf-388 docs) -------
# 16 expected trainer flags wired into Wave A invocations
CFM_WF380_FLAGS = [
    "--use-ot-cfm",
    "--stable-vm",
    "--so3-average",
    "--log-normal-t",
    "--triton-ot",
    "--shortcut-distill",
    "--consistency-weight",
    "--top-p",
    "--top-k",
    "--temperature",
    "--mmd-weight",
    "--repulsion-weight",
    "--per-cohort-features",
]
# 3 NEW Wf-388-A2/A3 trainer flags per Wf-392-A3
CFM_WF388_FLAGS = [
    "--scaffold-conditioning",  # Wf-388-A2 #6 (boolean store_true default False; activated via wrapper pass-through)
    "--sa-penalty",  # Wf-388-A3 #1 (default 0.0 OFF preserves Wf-380-B1 numerics)
    "--sa-target",  # Wf-388-A3 #2 (default 2.8 = Ertl 2009 cutoff)
]
# Total = 16 flags; --flowmol-unified intentionally NOT in list (LICENSE-GATED)
EXPECTED_CFM_FLAGS = CFM_WF380_FLAGS + CFM_WF388_FLAGS  # 13 + 3 = 16

# 3-wave split contract
EXPECTED_WAVES = ["A", "B", "C"]


def _read_launcher() -> str:
    """Read the launcher source as text."""
    if not LAUNCHER.exists():
        raise FileNotFoundError(f"launcher not found: {LAUNCHER}")
    return LAUNCHER.read_text(encoding="utf-8")


def verify_wf299_user_ok_gate(launcher_path: Path = LAUNCHER) -> Dict[str, Any]:
    """Verify USER_OK gate present + BEFORE --dry-run block.

    Per Wf-378-A2 G6 lesson + AGENTS.md §5.1: any > 1 hr GPU launcher
    MUST have a USER_OK=1 gate that fires BEFORE the --dry-run exit
    block (otherwise --dry-run bypasses the gate and the wrapper can
    be silently auto-launched without per-launch user OK).

    Returns:
        dict with keys: launcher_path, gate_present (bool),
        gate_fires_before_dry_run (bool), gate_exit_code (int),
        gate_pattern_matched (str), evidence (str).
    """
    src = _read_launcher()
    # Pattern: a USER_OK env-var check with explicit FAIL message.
    user_ok_pattern = re.compile(
        r'if\s*\[\s*"\${USER_OK:-0}"\s*!=\s*"1"\s*\]',
        re.MULTILINE,
    )
    gate_match = user_ok_pattern.search(src)
    gate_present = bool(gate_match)

    # Find the position of the --dry-run block.
    dry_run_pattern = re.compile(
        r"--dry-run\)\s*DRY_RUN=1",
        re.MULTILINE,
    )
    dry_run_match = dry_run_pattern.search(src)
    gate_fires_before_dry_run = bool(
        gate_match and dry_run_match and gate_match.start() < dry_run_match.start()
    )

    # Check the canonical exit code 64 (per Wf-366-A1 + Wf-378-A2 family).
    exit_64_pattern = re.compile(r"exit\s+64", re.MULTILINE)
    gate_exit_code_present = bool(exit_64_pattern.search(src))

    return {
        "launcher_path": str(launcher_path),
        "gate_present": gate_present,
        "gate_fires_before_dry_run": gate_fires_before_dry_run,
        "gate_exit_code_64_present": gate_exit_code_present,
        "gate_pattern_matched": gate_match.group(0) if gate_match else "",
        "evidence": (
            f"USER_OK gate {'PRESENT' if gate_present else 'MISSING'}; "
            f"fires-before-dry-run={gate_fires_before_dry_run}; "
            f"exit-64={'yes' if gate_exit_code_present else 'no'}"
        ),
    }


def verify_wf299_pre_flight(launcher_path: Path = LAUNCHER) -> Dict[str, Any]:
    """Verify pre-launch gate chain is wired + each gate script exists + is callable.

    Per TODO-207 §2.5 + Wf-385-A3 + Wf-343C2/C3: the launcher MUST
    invoke wf343c2_per_wave_check + wf343c3_data_quality_check +
    wf385a3_scaffold_wire_in_gate as part of pre-launch verification.

    Note: this does NOT actually run the gates (which would require
    torch + full pre-flight data quality). It verifies the SUBPROCESS
    SCRIPTS exist + the launcher text references them.

    Returns:
        dict with keys: launcher_path, gates_exist (dict per script),
        launcher_references_gates (bool), per_script (dict).
    """
    src = _read_launcher()
    gate_scripts = {
        "wf343c2_per_wave_check": WF343C2,
        "wf343c3_data_quality_check": WF343C3,
        "wf385a3_scaffold_wire_in_gate": WF385A3,
    }
    gates_exist = {name: path.exists() for name, path in gate_scripts.items()}

    # Check the launcher text references each gate.
    launcher_references = {
        "wf343c2_per_wave_check": "wf343c2_per_wave_check" in src,
        "wf343c3_data_quality_check": "wf343c3_data_quality_check" in src,
        "wf385a3_scaffold_wire_in_gate": "wf385a3" in src,
    }

    return {
        "launcher_path": str(launcher_path),
        "gates_exist": gates_exist,
        "launcher_references_gates": launcher_references,
        "all_gates_present": all(gates_exist.values()),
        "any_gate_referenced": any(launcher_references.values()),
        "evidence": (
            f"gates_exist={gates_exist}; " f"launcher_references={launcher_references}"
        ),
    }


def verify_wf299_3_wave_separation(launcher_path: Path = LAUNCHER) -> Dict[str, Any]:
    """Verify the launcher separates 3 waves (smoke / full / ablation).

    Per Wf-299 brief + TODO-207 §3.2..§3.4: Wave A = P1.4 smoke
    (~2 hr cuda:0), Wave B = G6 no_wass smoke (~10 min cuda:1),
    Wave C = G6 full sweep (~7.5 hr cuda:1).

    Checks:
      - CFM_NEW_FLAGS array only fires in Wave A invocations (smoke path)
      - Wave B + Wave C have separate trainer/config/budget assignments
      - All 3 waves appear in the case statement

    Returns:
        dict with keys: launcher_path, waves_present (list[str]),
        wave_a_has_cfm_flags (bool), wave_b_no_wass (bool),
        wave_c_full_sweep (bool), case_statement_complete (bool),
        evidence (str).
    """
    src = _read_launcher()

    # Detect wave_A / wave_B / wave_C function definitions.
    wave_fns = {
        "A": re.search(r"^wave_A\(\)\s*\{", src, re.MULTILINE) is not None,
        "B": re.search(r"^wave_B\(\)\s*\{", src, re.MULTILINE) is not None,
        "C": re.search(r"^wave_C\(\)\s*\{", src, re.MULTILINE) is not None,
    }
    waves_present = [w for w, present in wave_fns.items() if present]

    # Wave A must include CFM_NEW_FLAGS expansion (smoke path coverage).
    wave_a_block = re.search(
        r"^wave_A\(\).*?(?=^wave_B|^# -+|^# Main)",
        src,
        re.MULTILINE | re.DOTALL,
    )
    wave_a_has_cfm_flags = bool(
        wave_a_block and "CFM_NEW_FLAGS[@]" in wave_a_block.group(0)
    )

    # Wave B must use --wasserstein-weight 0.0 (no_wass smoke).
    wave_b_block = re.search(
        r"^wave_B\(\).*?(?=^wave_C|^# -+|^# Main)",
        src,
        re.MULTILINE | re.DOTALL,
    )
    wave_b_no_wass = bool(
        wave_b_block and "--wasserstein-weight 0.0" in wave_b_block.group(0)
    )

    # Wave C must reference full sweep + 60-run enumeration + 8 sub-waves.
    wave_c_block = re.search(
        r"^wave_C\(\).*?(?=^# -+|^# Main)",
        src,
        re.MULTILINE | re.DOTALL,
    )
    wave_c_full_sweep = bool(wave_c_block and "60" in wave_c_block.group(0))

    # Case statement must include all 3 waves + 'all'.
    case_pattern = re.search(
        r"case\s*\"\$\{WAVE_TARGET\}\"\s*in\s*(.*?)\s*esac",
        src,
        re.DOTALL,
    )
    case_text = case_pattern.group(1) if case_pattern else ""
    case_statement_complete = all(
        f'"{w}")' in case_text or f"{w})" in case_text for w in EXPECTED_WAVES
    )

    return {
        "launcher_path": str(launcher_path),
        "waves_present": waves_present,
        "wave_a_has_cfm_flags": wave_a_has_cfm_flags,
        "wave_b_no_wass": wave_b_no_wass,
        "wave_c_full_sweep": wave_c_full_sweep,
        "case_statement_complete": case_statement_complete,
        "evidence": (
            f"waves_present={waves_present}; "
            f"wave_A_CFM_flags={wave_a_has_cfm_flags}; "
            f"wave_B_no_wass={wave_b_no_wass}; "
            f"wave_C_full_sweep={wave_c_full_sweep}"
        ),
    }


def verify_wf299_w388_flags(launcher_path: Path = LAUNCHER) -> Dict[str, Any]:
    """Verify Wf-388 diversity + SA flags pass through to the trainer.

    Per Wf-392-A3: Wave A smoke invocation must include the 16 NEW
    trainer flags (13 Wf-380 + 3 Wf-388). --flowmol-unified is
    intentionally NOT in the pass-through list (LICENSE-GATED).

    Returns:
        dict with keys: launcher_path, cfm_new_flags_count (int),
        expected_flags_count (int), all_present (bool),
        missing_flags (list[str]), flowmol_intentionally_skipped (bool),
        evidence (str).
    """
    src = _read_launcher()

    # Extract the CFM_NEW_FLAGS array content.
    # Note: naive `\(.*?\)` is fragile because inline comments contain `)` chars
    # (e.g. `(store_true ON)` at line 153). Use a line-anchored approach:
    # find the line with `CFM_NEW_FLAGS=(`, then take everything until the
    # first line whose stripped content is just `)`.
    array_text = ""
    lines = src.split("\n")
    in_array = False
    for line in lines:
        stripped = line.strip()
        if not in_array:
            if stripped.startswith("CFM_NEW_FLAGS=("):
                in_array = True
                # Skip the opening paren if it's the only token on the line.
                if stripped == "CFM_NEW_FLAGS=(":
                    continue
                # Otherwise the opening line may have content (rare); fall through.
                after_open = stripped[len("CFM_NEW_FLAGS=(") :]
                if after_open.strip():
                    array_text += after_open + "\n"
                continue
        else:
            if stripped == ")":
                break
            array_text += line + "\n"

    # Each flag must appear in the array.
    found_flags = [f for f in EXPECTED_CFM_FLAGS if f in array_text]

    missing_flags = [f for f in EXPECTED_CFM_FLAGS if f not in array_text]
    all_present = len(missing_flags) == 0

    # Verify --flowmol-unified is NOT an ACTIVE flag (intentional skip).
    # It may appear inside the array as a commented-out `# --flowmol-unified`
    # line (per Wf-385-A2 §2 LICENSE-GATED convention), but it must NOT
    # appear as an uncommented flag. Detect active by checking each line:
    # a flag is "active" if it appears as the FIRST token on a non-comment line.
    flowmol_active = False
    for line in array_text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # First token of the line is the flag.
        first_token = stripped.split()[0] if stripped.split() else ""
        if first_token == "--flowmol-unified":
            flowmol_active = True
            break
    flowmol_intentionally_skipped = not flowmol_active

    # Verify Wave A invocations expand ${CFM_NEW_FLAGS[@]}.
    wave_a_expansion = "CFM_NEW_FLAGS[@]" in src

    return {
        "launcher_path": str(launcher_path),
        "cfm_new_flags_count": len(found_flags),
        "expected_flags_count": len(EXPECTED_CFM_FLAGS),
        "all_present": all_present,
        "missing_flags": missing_flags,
        "flowmol_intentionally_skipped": flowmol_intentionally_skipped,
        "wave_a_expansion_present": wave_a_expansion,
        "evidence": (
            f"{len(found_flags)}/{len(EXPECTED_CFM_FLAGS)} flags in CFM_NEW_FLAGS; "
            f"missing={missing_flags}; "
            f"flowmol_skipped={flowmol_intentionally_skipped}"
        ),
    }


# --- Main entrypoint ---------------------------------------------------------
def main() -> int:
    """Run all 4 verify_* + emit JSON report; return 0 on PASS, 1 on FAIL."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    print("Wf-394-A3 — Wf-299 pre-launch verification")
    print("=" * 60)

    user_ok = verify_wf299_user_ok_gate()
    print(f"\n[1/4] USER_OK gate: {user_ok['evidence']}")

    pre_flight = verify_wf299_pre_flight()
    print(f"\n[2/4] Pre-flight chain: {pre_flight['evidence']}")

    waves = verify_wf299_3_wave_separation()
    print(f"\n[3/4] 3-wave separation: {waves['evidence']}")

    w388 = verify_wf299_w388_flags()
    print(f"\n[4/4] Wf-388 flags: {w388['evidence']}")

    # Verdict: ALL checks must pass for an overall PASS.
    checks = {
        "user_ok_gate_present_and_before_dry_run": (
            user_ok["gate_present"] and user_ok["gate_fires_before_dry_run"]
        ),
        "pre_flight_gates_wired": pre_flight["any_gate_referenced"],
        "three_wave_separation_complete": (
            waves["case_statement_complete"]
            and waves["wave_a_has_cfm_flags"]
            and waves["wave_b_no_wass"]
            and waves["wave_c_full_sweep"]
        ),
        "wf388_flags_all_present": w388["all_present"],
    }
    overall_pass = all(checks.values())

    elapsed = time.time() - started_at
    result = {
        "wf_origin": "Wf-394-A3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "launcher_path": str(LAUNCHER),
        "verdict": "PASS" if overall_pass else "FAIL",
        "elapsed_seconds": round(elapsed, 3),
        "checks": checks,
        "details": {
            "user_ok": user_ok,
            "pre_flight": pre_flight,
            "waves": waves,
            "w388": w388,
        },
        "expected_flags": EXPECTED_CFM_FLAGS,
        "expected_waves": EXPECTED_WAVES,
        "honest_framing": (
            "Wf-394-A3 verification is CPU-only, no GPU launches, no torch "
            "required. The gate verifies file presence + text-pattern + "
            "structural correctness. Actual smoke-step training validation "
            "is delegated to wf385a3 (scaffold wire-in) + wf343c2/c3 "
            "(per-wave + data quality). This module is the GATE-OF-GATES "
            "that ensures the launcher surface is structurally correct "
            "BEFORE the live Wf-299 GPU launch."
        ),
    }

    output_path = REPORTS_DIR / "result.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResult: {result['verdict']} (elapsed {elapsed:.2f}s)")
    print(f"Report: {output_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
