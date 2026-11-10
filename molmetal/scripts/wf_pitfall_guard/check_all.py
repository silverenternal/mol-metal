#!/usr/bin/env python
"""WF-TODO-30 Pitfall Reinforce — single-shot runner for all 7 guards.

Runs every pitfall guard in
:mod:`molmetal.tests.test_pitfall_guards` and prints a concise
PASS / FAIL matrix.  Designed to be called from the ultracode
dispatch harness as a CPU-only verification gate.

Exit code
=========

* ``0`` — every guard passed (all 7 PASS).
* ``1`` — at least one guard failed (a regression slipped in).
* ``2`` — all guards were SKIPPED (no guard actually ran; usually
  means the runtime environment cannot import the modules being
  guarded — e.g. a bare-bones CI box without the project's chemistry
  deps).  Also returned when the runner itself crashes (missing
  fixture, etc).

Usage
=====

    cd /home/hugo/codes/mol-metal
    uv run python molmetal/scripts/wf_pitfall_guard/check_all.py

Or via the ``--format`` flag for a machine-readable summary (CI
consumers, dashboards, JSON-aware tooling)::

    uv run python molmetal/scripts/wf_pitfall_guard/check_all.py --format json

The ``--format human`` (default) prints a human-readable ASCII
table; ``--format json`` prints a single JSON object on stdout for
downstream tooling.

Honest framing
==============

This script does NOT mutate any source file.  It only *observes* the
current state of the repo and asserts that every shipped fix is still
in place.  To verify the guards actually catch regressions, delete
the corresponding fix in a scratch branch, re-run, and confirm the
guard fires.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_FILE = REPO_ROOT / "molmetal" / "tests" / "test_pitfall_guards.py"

# Canonical pitfall IDs (per TODO-30).  Order matches the test file.
PITFALL_IDS: Tuple[str, ...] = (
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
)

# Friendly labels for the report.
PITFALL_LABEL: Dict[str, str] = {
    "P1": "OOV atoms → vocab_mask + 20-element default vocab",
    "P2": "SynFlowNet shim TypeError → per-rule try/except",
    "P3": "MetalLigandExchange 84-rule matrix + cisplatin seed default",
    "P4": "BF16 atom_logits clamp ±50 + FP32 AMP fallback",
    "P5": "r4_lambda_only_run SAFETY_MAX ≥ 10000 floor",
    "P6": "Reference clone location auto-detect (parent vs clone root)",
    "P7": "ModelWrapper keyword-only t fallback (functools.partial fix)",
}

# Stable, machine-readable short identifiers used in the JSON output.
# These mirror the label keywords in snake_case so dashboards can
# join on them without re-parsing free text.
PITFALL_SHORT_NAME: Dict[str, str] = {
    "P1": "oov_atom_vocab_mask_20elem",
    "P2": "synflownet_shim_typeerror_fallback",
    "P3": "metal_ligand_exchange_84rule_cisplatin_seed",
    "P4": "bf16_atom_logits_clamp_pm50_fp32_amp_fallback",
    "P5": "r4_lambda_safety_max_10k_floor",
    "P6": "reference_clone_location_autodetect",
    "P7": "model_wrapper_kwonly_t_fallback",
}


def _run_pytest(verbose: bool = False) -> Tuple[int, str, str]:
    """Invoke pytest on the guard test file; return (rc, stdout, stderr).

    **AGENTS.md §5.9 — the interpreter that runs the tests must be the
    interpreter that runs this runner**, i.e. ``sys.executable -m pytest``.

    The ORIGINAL implementation preferred the BARE ``pytest`` on PATH "so we
    land in the project's managed virtualenv where the test deps are
    installed".  **That reasoning is backwards.**  A bare ``pytest`` lands in
    WHATEVER environment provides it: MEASURED 2026-11-04 on this host,
    ``which pytest`` resolves to
    ``/home/linuxbrew/.linuxbrew/opt/python@3.14/bin/python3.14`` — a
    Homebrew python 3.14 with **no torch** — so the repo's own
    ``molmetal/tests/conftest.py`` died on ``import torch`` and
    ``smoke_pitfalls.sh`` reported ``FAIL runner_crashed rc=2`` for a reason
    that had NOTHING to do with the seven pitfall guards.  A measurement
    taken under an unpinned interpreter is a measurement nobody may quote
    (AGENTS.md §5.2).

    The bare form is retained as a LAST resort (an environment where
    pytest is installed but not importable by ``sys.executable``), and it
    is only reached when the pinned attempt produced **no output at all**:
    the loop below returns as soon as a run produced output, and a CRASHED
    pytest always produces output — that output is a VERDICT, not an
    invitation to retry under a different interpreter.
    """
    rel = str(TEST_FILE.relative_to(REPO_ROOT))
    cmds: List[List[str]] = [
        [sys.executable, "-m", "pytest", rel, "-v", "--tb=short", "--no-header"],
        ["pytest", rel, "-v", "--tb=short", "--no-header"],
    ]
    last_err = ""
    for cmd in cmds:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode == 0 or proc.stdout or proc.stderr:
            return proc.returncode, proc.stdout, proc.stderr
        last_err = proc.stderr or proc.stdout
    return 1, "", last_err or "pytest could not be invoked in this env"


def _parse_pytest_output(out: str) -> Dict[str, str]:
    """Parse ``pytest -v`` output; return ``{test_name: PASS|FAIL|SKIP}``."""
    results: Dict[str, str] = {}
    for line in out.splitlines():
        line = line.rstrip()
        # pytest -v prints ``test_pitfall_guards.py::test_xxx PASSED [..]``
        # or ``::test_xxx FAILED`` or ``::test_xxx SKIPPED``.
        if "::" not in line:
            continue
        for token in ("PASSED", "FAILED", "SKIPPED"):
            if f" {token}" in line:
                name = line.split("::", 1)[1].split(" ")[0]
                results[name] = token
                break
    return results


def _resolve_guard_status(
    pid: str,
    results: Dict[str, str],
) -> Tuple[str, str]:
    """Return ``(status_short, message)`` for a given pitfall ID.

    ``status_short`` is one of ``PASS`` / ``FAIL`` / ``SKIP`` / ``MISS``
    (the canonical short labels used in both the human table and the
    JSON output).  ``message`` is the human-readable explanation,
    typically the friendly pitfall label plus a short outcome phrase.
    """
    matched: Optional[str] = None
    for test_name, status in results.items():
        if pid in test_name and "pitfall_guard_matrix" in test_name:
            matched = status
            break
    if matched is None:
        # Fall back to the test_fn by ID name (test_p1_...,
        # test_p2_..., etc.)  These are the canonical sub-tests
        # that pytest runs when ``test_pitfall_guard_matrix`` is
        # expanded via parametrize.
        for test_name, status in results.items():
            if f"_{pid.lower()}_" in test_name:
                matched = status
                break
    label = PITFALL_LABEL[pid]
    if matched is None:
        return "MISS", f"{label} — guard test not found in pytest output"
    if matched == "PASSED":
        return "PASS", f"{label} — guard passed"
    if matched == "FAILED":
        return "FAIL", f"{label} — guard FAILED (regression slipped in)"
    if matched == "SKIPPED":
        return "SKIP", f"{label} — guard skipped (runtime dep missing)"
    return matched, f"{label} — guard returned status {matched}"


def _format_table(results: Dict[str, str]) -> str:
    """Build a human-readable ASCII table summarising the results."""
    name_w = max(
        len("Pitfall"),
        max(len(pid) for pid in PITFALL_IDS),
    )
    label_w = max(len("Description"), max(len(l) for l in PITFALL_LABEL.values()))
    status_w = max(len("Status"), 8)

    lines: List[str] = []
    header = (
        f"{'Pitfall':<{name_w}}  {'Description':<{label_w}}  " f"{'Status':<{status_w}}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    n_pass = n_fail = n_skip = n_miss = 0
    for pid in PITFALL_IDS:
        label = PITFALL_LABEL[pid]
        status_short, _ = _resolve_guard_status(pid, results)
        if status_short == "PASS":
            n_pass += 1
        elif status_short == "FAIL":
            n_fail += 1
        elif status_short == "SKIP":
            n_skip += 1
        else:
            n_miss += 1
        lines.append(f"{pid:<{name_w}}  {label:<{label_w}}  {status_short:<{status_w}}")
    lines.append("-" * len(header))
    summary = f"PASS={n_pass} FAIL={n_fail} SKIP={n_skip} MISS={n_miss}"
    lines.append(summary)
    return "\n".join(lines)


def _format_json(results: Dict[str, str]) -> str:
    """Build a machine-readable JSON summary.

    The schema is::

        {
            "timestamp": "<ISO-8601 UTC>",
            "guards_total": <int>,
            "guards_pass": <int>,
            "guards_fail": <int>,
            "guards_skip": <int>,
            "guards": [
                {"name": "<stable snake_case id>",
                 "status": "PASS" | "FAIL" | "SKIP" | "MISS",
                 "message": "<human-readable one-liner>"},
                ...
            ]
        }

    The summary counters (``guards_*``) and the per-guard status are
    kept in lockstep so consumers can rely on either representation.
    """
    guards: List[Dict[str, str]] = []
    n_pass = n_fail = n_skip = n_miss = 0
    for pid in PITFALL_IDS:
        status_short, message = _resolve_guard_status(pid, results)
        if status_short == "PASS":
            n_pass += 1
        elif status_short == "FAIL":
            n_fail += 1
        elif status_short == "SKIP":
            n_skip += 1
        else:
            n_miss += 1
        guards.append(
            {
                "name": PITFALL_SHORT_NAME[pid],
                "status": status_short,
                "message": message,
            }
        )
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "guards_total": len(PITFALL_IDS),
        "guards_pass": n_pass,
        "guards_fail": n_fail,
        "guards_skip": n_skip,
        "guards": guards,
    }
    # Always-on notes for CI consumers — make missing / never-run
    # states visible without parsing stderr.
    if n_miss:
        payload["guards_miss"] = n_miss
    return json.dumps(payload, indent=2)


def _compute_exit_code(n_pass: int, n_fail: int, n_skip: int, n_miss: int) -> int:
    """Return 0 if all PASS, 1 if any FAIL, 2 if all SKIP, else 1.

    The "all SKIP" branch signals "nothing actually ran" — a CI box
    without the chemistry deps would otherwise silently pass with a
    misleading green dot.  Reporting exit 2 forces a human to look at
    why the guards were all skipped.  Mixed states (some PASS, some
    SKIP) still report exit 0 as long as nothing FAILED, because at
    least *some* guards did observe the codebase.
    """
    if n_fail > 0:
        return 1
    total = n_pass + n_fail + n_skip + n_miss
    if n_skip > 0 and n_pass == 0 and n_fail == 0 and n_miss == 0:
        # Every single guard was skipped → nothing ran.
        return 2
    if total == 0:
        # Nothing happened at all (no guards observed).
        return 2
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all 7 R12-R17 pitfall guards and print a summary."
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("human", "json"),
        default="human",
        help=(
            "Output format: 'human' (default, ASCII table on stdout) "
            "or 'json' (single JSON object on stdout, CI-friendly)."
        ),
    )
    parser.add_argument(
        "--pytest-args",
        type=str,
        default="",
        help="Extra args passed through to pytest (quoted).",
    )
    # Backwards compatibility: ``--json`` is kept as an alias for
    # ``--format json`` so existing callers / hooks don't break.
    parser.add_argument(
        "--json",
        action="store_true",
        help="Alias for ``--format json`` (deprecated; prefer --format).",
    )
    args = parser.parse_args(argv)

    if args.json:
        args.output_format = "json"

    if not TEST_FILE.is_file():
        print(f"FATAL: test file not found at {TEST_FILE}", file=sys.stderr)
        return 2

    t0 = time.time()
    rc, stdout, stderr = _run_pytest()
    dt = time.time() - t0
    if rc != 0 and "error" in stderr.lower():
        # Runner-level crash (pytest missing, etc.) — surface on
        # stderr for both human and JSON consumers, then exit 2.
        msg = f"FATAL: pytest crashed (rc={rc}) in {dt:.1f}s:\n{stderr}"
        if args.output_format == "json":
            payload = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                "guards_total": 0,
                "guards_pass": 0,
                "guards_fail": 0,
                "guards_skip": 0,
                "guards": [],
                "fatal": msg,
            }
            print(json.dumps(payload, indent=2))
        else:
            print(msg, file=sys.stderr)
        return 2

    results = _parse_pytest_output(stdout)

    # Tally once for both formatters and the exit-code decision.
    n_pass = n_fail = n_skip = n_miss = 0
    for pid in PITFALL_IDS:
        status_short, _ = _resolve_guard_status(pid, results)
        if status_short == "PASS":
            n_pass += 1
        elif status_short == "FAIL":
            n_fail += 1
        elif status_short == "SKIP":
            n_skip += 1
        else:
            n_miss += 1

    if args.output_format == "json":
        # Re-emit JSON so the counters and the per-guard array agree
        # with the exit-code decision below.
        out_json = _format_json(results)
        # Patch the JSON counters in-place so we don't recompute twice.
        payload = json.loads(out_json)
        payload["guards_pass"] = n_pass
        payload["guards_fail"] = n_fail
        payload["guards_skip"] = n_skip
        if n_miss:
            payload["guards_miss"] = n_miss
        print(json.dumps(payload, indent=2))
    else:
        print(_format_table(results))
        print(f"\npytest rc={rc}  wall={dt:.1f}s")

    return _compute_exit_code(n_pass, n_fail, n_skip, n_miss)


if __name__ == "__main__":
    raise SystemExit(main())
