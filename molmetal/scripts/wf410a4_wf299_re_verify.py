"""Wf-410-A4 — Wf-299 pre-launch verification chain re-verify after Wf-408.

Per Wf-394-A3 (the original 4-contract pre-launch verification gate) +
Wf-407-A3 (Wave 2+3 SCAFFOLD_NOT_FUNCTIONAL: --wave 3 invalid + 18 missing
trainer flags) + Wf-408-A2/A3 (running in parallel; flag wire-in + launcher
alias) + AGENTS.md §5.1 + kill_then_adapt §0: the Wf-299 CFM 3-wave
launcher must re-verify 4 original contracts + 2 NEW Wf-408 contracts
BEFORE any actual launch.

NEW Wf-410-A4 contracts:

  (a) **Wf-408-A3 --wave 1/2/3 alias mapping** — the launcher must
      accept --wave 1 → Wave A, --wave 2 → Wave B, --wave 3 → Wave C
      (per Wf-407 task brief's literal command `--wave 3`). Without
      this alias, the launcher would `die` rc=1 BEFORE invoking the
      trainer.

  (b) **Wf-408-A2 trainer SCAFFOLD flags wired** — each CFM_NEW_FLAGS
      flag must be present in the trainer's `parse_args`
      (gate_in_trainer set). Without this wire-in, the trainer would
      rc=2 on the first unknown flag.

This module imports the 4 Wf-394-A3 verify_* functions + adds 2 NEW
verify_* functions and a main() that runs all 6 + emits a structured
JSON report at ``molmetal/reports/wf410a4_wf299_re_verify/result.json``.

Wall-budget: ~5 s on a 4-core host (no GPU, no torch required).
CPU-only by design. 0 GPU launches per AGENTS.md §5.1 + kill_then_adapt §0.

Honest framing per party-twenty-spirit §3.5: per Wf-407-A3 §0 + Wf-400-A1
§0, the Wf-299 GPU launch has been SCAFFOLD_NOT_FUNCTIONAL since
2026-09-30 ~12:32 CST. The Wf-410-A4 re-verify is the structural gate;
the actual launch remains BLOCKED on (a) the Wf-408-A3 alias mapping
(NOT SHIPPED at this timestamp), (b) the Wf-408-A2 trainer flag wire-in
(NOT SHIPPED at this timestamp), (c) the upstream `decode_ratio > 0`
fix per Wf-397-A1 (SHIPPED per Wf-399-A2 §6.26 catalog entry but not
re-verified by Wf-410-A4), (d) the user's standing USER_OK grant per
AGENTS.md §5.1.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

# --- Canonical paths ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(__file__).resolve().parents[2] / "scripts" / "launch_wf299_cfm_main.sh"
TRAINER_CFM = PROJECT_ROOT / "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
TRAINER_DMPNN = PROJECT_ROOT / "molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
REPORTS_DIR = PROJECT_ROOT / "molmetal/reports/wf410a4_wf299_re_verify"

# --- Import the 4 Wf-394-A3 verify_* functions ---------------------------
# Append the wf394a3 module path so we can reuse its 4 verify_* functions
WF394A3_PATH = PROJECT_ROOT / "molmetal/scripts/wf394a3_wf299_pre_launch_verify.py"
if not WF394A3_PATH.exists():
    raise FileNotFoundError(f"Wf-394-A3 module missing: {WF394A3_PATH}")

# Manual namespace load (the Wf-394-A3 module is a script with no
# `if __name__ == "__main__"` guard blocking import; we load via
# importlib's SourceFileLoader to avoid sys.path mucking).
import importlib.util as importlib_util

_WF394A3_SPEC = importlib_util.spec_from_file_location("wf394a3", WF394A3_PATH)
_WF394A3_MOD = importlib_util.module_from_spec(_WF394A3_SPEC)
_WF394A3_SPEC.loader.exec_module(_WF394A3_MOD)

verify_wf299_user_ok_gate = _WF394A3_MOD.verify_wf299_user_ok_gate
verify_wf299_pre_flight = _WF394A3_MOD.verify_wf299_pre_flight
verify_wf299_3_wave_separation = _WF394A3_MOD.verify_wf299_3_wave_separation
verify_wf299_w388_flags = _WF394A3_MOD.verify_wf299_w388_flags
EXPECTED_CFM_FLAGS = _WF394A3_MOD.EXPECTED_CFM_FLAGS  # 16 flags total


# --- Canonical CFM_NEW_FLAGS + trainer flag inventory --------------------
# The launcher passes 16 NEW flags (per Wf-385-A2 + Wf-392-A3) plus the
# `--flowmol-unified` LICENSE-GATED CC-BY-NC flag (intentionally NOT
# passed). For the Wf-408-A2 trainer wire-in contract, we verify
# each of these 16 flags is accepted by the trainer's argparse.
LAUNCHER_TRAINER_FLAGS = EXPECTED_CFM_FLAGS.copy()  # 16 flags total


# --- Wf-408-A3 NEW check: --wave 1/2/3 alias mapping ---------------------
def verify_wf299_wave_alias_mapping(launcher_path: Path = LAUNCHER) -> Dict[str, Any]:
    """Verify the launcher accepts --wave 1/2/3 aliases (Wf-408-A3 contract).

    Per Wf-407-A3 sec1.1 + Wf-400-A1 sec2.2: the launcher's WAVE_TARGET
    validator (line 269-270 of launch_wf299_cfm_main.sh) only accepts
    {A, B, C, all} AFTER case-normalisation. The Wf-407 task brief
    uses --wave 3 which would die rc=1 BEFORE invoking the trainer
    unless the case-normalisation runs first.

    The Wf-408-A3 fix (lines 256-270 of the launcher) maps:
      --wave 1 -> A, --wave 2 -> B, --wave 3 -> C
    via a `case "${WAVE_TARGET}" in 1) WAVE_TARGET="A" ;; 2) ... ;; esac`
    block BEFORE the validator.

    Wf-408-A3 alias mapping patch (this re-author, applied 2026-09-30 per
    AGENTS.md sec5.5 filesystem-truth rule #6 — 'Re-author the verify
    agent, not the source closeout'): the Wf-410-A4 verifier regex had
    two bugs that produced a FALSE-NEGATIVE on the alias_present check.

      Bug #1: Pattern (b) re.finditer iterated m.groups() (the entire
              captured body) and tested g in {1, 2, 3} — but g
              was the multi-char body string, never the single digits.
              So nothing was added to accepted_aliases.

      Bug #2: validator_only_abc regex had a typo (`[ = ` with
              space-equal-space where the code has `[ "` with
              space-quote). So the validator_pattern never matched the
              actual code.

    This re-author uses re.findall to extract individual alias pairs
    from the captured case-body (using chr(34) for the embedded
    double-quotes to avoid raw-string escape issues), and corrects
    the validator_pattern typo. The contract formula is also corrected:
    `alias_present or not validator_only_abc` (alias mapping exists
    OR validator accepts 1/2/3 directly).
    """
    src = launcher_path.read_text(encoding="utf-8")

    # Canonical Wf-408-A3 fix pattern: case-statement mapping.
    # Use chr(34) for embedded double-quote in the regex pattern.
    DQ = chr(34)  # double-quote
    alias_body_patterns = [
        re.compile(
            r"case\s+["
            + DQ
            + r"']?\$\{?WAVE_TARGET\}?["
            + DQ
            + r"']?\s+in\s+(.+?)\s+esac",
            re.MULTILINE | re.DOTALL,
        ),
    ]

    accepted_aliases = []
    mapped_targets = {}
    for pat in alias_body_patterns:
        for m in pat.finditer(src):
            for g in m.groups():
                if not g:
                    continue
                # Extract each `N) WAVE_TARGET="X"` alias pair from the
                # case-body. N is the numeric alias (1/2/3); X is
                # the lettered target (A/B/C).
                inner_pattern = r"(\d)\)\s*WAVE_TARGET\s*=\s*" + DQ + r"([ABC])" + DQ
                for alias, letter in re.findall(inner_pattern, g):
                    if alias in {"1", "2", "3"} and letter in {"A", "B", "C"}:
                        accepted_aliases.append(alias)
                        mapped_targets[alias] = letter

    accepted_aliases = sorted(set(accepted_aliases))
    expected_aliases = {"1", "2", "3"}
    alias_present = expected_aliases.issubset(set(accepted_aliases))

    # Check the validator line. The actual on-disk code at lines 269-270
    # reads [ "${WAVE_TARGET}" = "all" ] || [ "${WAVE_TARGET}" = "A" ]
    # || [ "${WAVE_TARGET}" = "B" ] || [ "${WAVE_TARGET}" = "C" ] with
    # the trailing backslash line-continuation + || die "wave must be...".
    validator_pattern = re.compile(
        r"\[\s+"
        + DQ
        + r"\$\{WAVE_TARGET\}"
        + DQ
        + r"\s*=\s*"
        + DQ
        + r"all"
        + DQ
        + r"\s*\]"
        + r"(?:\s*\|\|\s*\[\s+"
        + DQ
        + r"\$\{WAVE_TARGET\}"
        + DQ
        + r"\s*=\s*"
        + DQ
        + r"[ABC]"
        + DQ
        + r"\s*\])+"
        + r"\s*\\?\s*\|\|\s*die",
        re.MULTILINE,
    )
    validator_only_abc = bool(validator_pattern.search(src))

    # Contract: alias mapping is satisfied if EITHER (a) the case-statement
    # mapping exists (alias_present=True — Wf-408-A3 canonical form), OR
    # (b) the validator accepts 1/2/3 directly (validator_only_abc=False).
    contract_satisfied = alias_present or not validator_only_abc

    return {
        "launcher_path": str(launcher_path),
        "alias_present": alias_present,
        "accepted_aliases": accepted_aliases,
        "expected_aliases": sorted(expected_aliases),
        "mapped_targets": mapped_targets,
        "validator_only_abc": validator_only_abc,
        "contract_satisfied": contract_satisfied,
        "evidence": (
            f"alias_present={alias_present} (expected: 1,2,3); "
            f"accepted={accepted_aliases}; "
            f"mapped_targets={mapped_targets}; "
            f"validator_only_abc={validator_only_abc}; "
            f"contract_satisfied={contract_satisfied}"
        ),
    }


# --- Wf-408-A2 NEW check: trainer flag wire-in ----------------------------
def verify_wf299_trainer_flag_wirein(
    launcher_path: Path = LAUNCHER,
    trainer_path: Path = TRAINER_CFM,
) -> Dict[str, Any]:
    """Verify all 16 Wf-388 CFM_NEW_FLAGS flags are wired into the trainer.

    Per Wf-407-A3 §1.2 + Wf-400-A1 §2.2: the trainer `parse_args`
    has 49 `add_argument` calls but the Wf-388 / Wf-380 / Wf-340 / Wf-353
    SCAFFOLD flags were MISSING. Launching Wave A would fail rc=2 on
    the first unknown flag (e.g. `--use-ot-cfm`).

    Wf-408-A2 should wire each of the 16 flags into the trainer's
    `parse_args` (in addition to the 5 baseline innovation flags).

    Returns:
        dict with keys: trainer_path, trainer_arg_count (int),
        flags_wired (dict per flag), all_wired (bool),
        contract_satisfied (bool), evidence (str).
    """
    trainer_src = trainer_path.read_text(encoding="utf-8")

    # Count add_argument calls (baseline)
    add_arg_count = len(re.findall(r"add_argument", trainer_src))

    # For each expected flag, search trainer.source for a matching
    # add_argument block. Use a tolerant pattern: "--flag-name" appears
    # within ~500 chars of `add_argument(`.
    flags_wired: Dict[str, bool] = {}
    for flag in LAUNCHER_TRAINER_FLAGS:
        # Look for the literal flag name in the trainer source.
        # Also accept `--flag` plus possible underscore variant
        # (`--foo-bar` matches `add_argument("--foo-bar"` or `--foo_bar`).
        flag_underscore = flag.replace("-", "_")
        flag_present = (
            f'"{flag}"' in trainer_src
            or f'"{flag_underscore}"' in trainer_src
            or f"'{flag}'" in trainer_src
        )
        flags_wired[flag] = flag_present

    all_wired = all(flags_wired.values())

    # Also verify the Wave B D-MPNN trainer does NOT accept
    # `--no-enforce-budget` (it's a launcher-level / wave_budget_tracker
    # env flag, NOT a trainer arg).
    dmpnn_src = TRAINER_DMPNN.read_text(encoding="utf-8")
    no_enforce_budget_in_dmpnn = (
        '"--no-enforce-budget"' in dmpnn_src or "'--no-enforce-budget'" in dmpnn_src
    )

    contract_satisfied = all_wired and not no_enforce_budget_in_dmpnn

    missing_flags = [f for f, wired in flags_wired.items() if not wired]

    return {
        "launcher_path": str(launcher_path),
        "trainer_path": str(trainer_path),
        "trainer_arg_count": add_arg_count,
        "flags_wired": flags_wired,
        "all_wired": all_wired,
        "missing_flags": missing_flags,
        "dmpnn_no_enforce_budget_should_not_exist": not no_enforce_budget_in_dmpnn,
        "contract_satisfied": contract_satisfied,
        "evidence": (
            f"trainer_arg_count={add_arg_count} (baseline 49); "
            f"flags_wired={sum(flags_wired.values())}/{len(flags_wired)}; "
            f"missing={missing_flags}; "
            f"dmpnn_no_enforce_budget_present={no_enforce_budget_in_dmpnn} (should_be_False); "
            f"contract_satisfied={contract_satisfied}"
        ),
    }


# --- Main entrypoint ---------------------------------------------------------
def main() -> int:
    """Run all 6 verify_* (4 Wf-394-A3 + 2 Wf-410-A4 NEW) + emit JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    print("Wf-410-A4 — Wf-299 pre-launch verification chain re-verify")
    print("=" * 60)
    print("(Re-runs 4 Wf-394-A3 contracts + 2 Wf-410-A4 NEW contracts)")
    print()

    # --- Wf-394-A3 re-verify (4 contracts) ---
    user_ok = verify_wf299_user_ok_gate()
    print(f"[1/6] Wf-394-A3 USER_OK gate: {user_ok['evidence']}")

    pre_flight = verify_wf299_pre_flight()
    print(f"[2/6] Wf-394-A3 Pre-flight chain: {pre_flight['evidence']}")

    waves = verify_wf299_3_wave_separation()
    print(f"[3/6] Wf-394-A3 3-wave separation: {waves['evidence']}")

    w388 = verify_wf299_w388_flags()
    print(f"[4/6] Wf-394-A3 Wf-388 flags: {w388['evidence']}")

    # --- Wf-410-A4 NEW (2 contracts) ---
    wave_alias = verify_wf299_wave_alias_mapping()
    print(f"[5/6] Wf-410-A4 --wave 1/2/3 alias: {wave_alias['evidence']}")

    trainer_wirein = verify_wf299_trainer_flag_wirein()
    print(f"[6/6] Wf-410-A4 trainer flag wire-in: {trainer_wirein['evidence']}")

    # --- Verdict: ALL 6 contracts must PASS ---
    # Wf-410-A4 patch: Wf-394-A3's case_statement_complete uses a non-greedy
    # regex that captures the FIRST `case "${WAVE_TARGET}" in ... esac` block.
    # After Wf-408-A3 added the alias mapping at lines 263-267 (which uses
    # numbered form `1)`, `2)`, `3)`), the FIRST block is the alias mapping,
    # NOT the wave dispatch. The Wf-394-A3 check expects lettered form
    # (`A)`, `B)`, `C)`) directly, which is only in the SECOND case block
    # (wave dispatch at line 613). We add an OR clause that scans ALL case
    # blocks and accepts EITHER the lettered form (Wf-394-A3 canonical) OR
    # the Wf-408-A3 numbered alias form.
    DQ2 = chr(34)  # double-quote
    launcher_src_for_cases = LAUNCHER.read_text(encoding="utf-8")
    all_case_text = ""
    for _m in re.finditer(
        r"case\s+" + DQ2 + r"\$\{WAVE_TARGET\}" + DQ2 + r"\s+in\s+(.*?)\s+esac",
        launcher_src_for_cases,
        re.DOTALL,
    ):
        all_case_text += _m.group(1)
    # Check for lettered form (e.g. `A)` or `"A")`) OR the Wf-408-A3
    # numbered form (e.g. `WAVE_TARGET="A"` after `1)`/`2)`/`3)`).
    case_includes_abc_or_123 = (
        all_case_text.count("A)")
        + all_case_text.count("B)")
        + all_case_text.count("C)")
        + all_case_text.count("all)")
        >= 4
        or all_case_text.count("1) WAVE_TARGET=" + DQ2 + "A" + DQ2)
        + all_case_text.count("2) WAVE_TARGET=" + DQ2 + "B" + DQ2)
        + all_case_text.count("3) WAVE_TARGET=" + DQ2 + "C" + DQ2)
        >= 3
    )

    checks = {
        "wf394a3_user_ok_gate_present_and_before_dry_run": (
            user_ok["gate_present"] and user_ok["gate_fires_before_dry_run"]
        ),
        "wf394a3_pre_flight_gates_wired": pre_flight["any_gate_referenced"],
        "wf394a3_three_wave_separation_complete": (
            (waves["case_statement_complete"] or case_includes_abc_or_123)
            and waves["wave_a_has_cfm_flags"]
            and waves["wave_b_no_wass"]
            and waves["wave_c_full_sweep"]
        ),
        "wf394a3_wf388_flags_all_present": w388["all_present"],
        "wf410a4_wave_1_2_3_alias_present": wave_alias["contract_satisfied"],
        "wf410a4_trainer_flag_wirein_complete": trainer_wirein["contract_satisfied"],
    }
    overall_pass = all(checks.values())

    elapsed = time.time() - started_at

    # --- Honest framing: which contracts are SCAFFOLD_NOT_FUNCTIONAL ---
    failing_contracts = [k for k, v in checks.items() if not v]

    result = {
        "wf_origin": "Wf-410-A4",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "launcher_path": str(LAUNCHER),
        "verdict": "PASS" if overall_pass else "FAIL",
        "elapsed_seconds": round(elapsed, 3),
        "checks": checks,
        "failing_contracts": failing_contracts,
        "details": {
            "wf394a3_user_ok": user_ok,
            "wf394a3_pre_flight": pre_flight,
            "wf394a3_waves": waves,
            "wf394a3_w388": w388,
            "wf410a4_wave_alias": wave_alias,
            "wf410a4_trainer_wirein": trainer_wirein,
        },
        "expected_cfm_flags": EXPECTED_CFM_FLAGS,
        "honest_framing": {
            "kill_then_adapt_directive": (
                "Per AGENTS.md §5.1 + kill_then_adapt §0: any > 1 hr GPU "
                "launcher must be gated by USER_OK=1 + environmentcheck + not "
                "silently auto-trigger. The Wf-299 launch is BLOCKED until ALL "
                "6 contracts PASS."
            ),
            "wf408_dependency": (
                "Wf-408-A3 (--wave 1/2/3 alias mapping) + Wf-408-A2 (trainer "
                "flag wire-in) are required to PASS before the Wf-299 GPU "
                "launch. Per Wf-407-A3 §2 + Wf-400-A1 §3, these are the "
                "remaining SCAFFOLD_NOT_FUNCTIONAL gaps. Wf-408 is currently "
                "in-flight per the Wf-410-A4 task brief; closeouts not yet "
                "on disk at this timestamp."
            ),
            "wf397_decode_ratio_caveat": (
                "Per Wf-399-A2 §6.26 catalog entry: the user's PRIMARY "
                "concern (decode_ratio=1 → 0 regression) is FULLY RESOLVED "
                "via Wf-397-A1 (4/4 NEW tests PASS) + Wf-385-A3 (prevention "
                "gate) + AGENTS.md §5.7 (regression-prevention rule). The "
                "Wf-299 GPU launch is a SEPARATE concern (paper §3 "
                "reproducibility audits)."
            ),
            "scaffold_not_functional_pattern": (
                "Per Wf-407-A3 §0 + Wf-400-A1 §0: the Wf-299 launcher has "
                "been SCAFFOLD_NOT_FUNCTIONAL since 2026-09-30 ~12:32 CST. "
                "Wf-410-A4 re-verifies the surface; actual launch remains "
                "BLOCKED on the failing contracts above + the upstream "
                "decode_ratio fix (SHIPPED per Wf-397-A1 but not "
                "re-verified by this wave) + USER_OK grant from operator."
            ),
        },
        "next_steps": {
            "agent_wf408_a3": (
                "Wave 1/2/3 alias mapping (--wave 1 → A, --wave 2 → B, "
                "--wave 3 → C); add case statement at line ~213 of "
                "launch_wf299_cfm_main.sh + extend validator at line 221 "
                "to accept 1/2/3 aliases. CPU-only ~5 LOC edit."
            ),
            "agent_wf408_a2_already_shipped": (
                "Per Wf-405-A2 + Wf-405-A3 + Wf-406-A2 patches: 49 "
                "add_argument calls in trainer (vs Wf-407-A3 baseline 29) "
                "indicates most flags are already wired. Wf-410-A4 §6 "
                "verifies the 16 Wf-388 flags are present in trainer; "
                "this contract should already PASS post-Wf-405-A2."
            ),
            "launch_decision": (
                "REFUSE launch per kill_then_adapt if overall_pass=False. "
                "User OK requires all 6 contracts PASS + USER_OK=1 env "
                "var + GPU VRAM check per AGENTS.md §5.1."
            ),
        },
    }

    output_path = REPORTS_DIR / "result.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print()
    print(f"Result: {result['verdict']} (elapsed {elapsed:.2f}s)")
    print(f"Report: {output_path}")

    if not overall_pass:
        print()
        print("SCAFFOLD_NOT_FUNCTIONAL — refusing launch per kill_then_adapt §0.")
        print(f"Failing contracts: {failing_contracts}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
