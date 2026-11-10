#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""MEASURE the exit code of every launch-gate tool under a FAILING input.

WHY (Wf-A269 wave-11, 2026-11-02).  The DEAD-GATE class (catalog
§6.269.17) is "a check that looks like a check but CANNOT FAIL".  One of
its four sub-modes is the exit-code form: a wrapper keys its gate on a
tool's exit code, and that tool returns 0 *by construction* on exactly
the failure the gate is supposed to catch.  Two live instances:

  * ``wave_budget_tracker.py`` returns 0 on every path (its ``main()``
    ends with ``return 0``), even when ``remaining_hr`` is negative and
    waves are DEFERRED.
  * ``per_wave_checklist.py`` returns 0 even when its own JSON records
    ``preflight_fail_count = 1`` and a P0 blocker with ``status = FAIL``.

The prose that says so rots.  This probe replaces the prose with a
MEASURED, re-measurable registry: for every launch-gate tool it runs
the tool against a synthetic FAILING input and records the exit code
the tool ACTUALLY returns.  The registry is the single source of truth
for the guard (``wf_launch_gate_liveness.py``, arm B) — the guard must
NOT hardcode "per_wave_checklist.py always exits 0" from a docstring,
because that is exactly how this repo's status lines rot.

THE REGISTRY CONTRACT (the probe writes it; the guard reads it)

    {
      "schema_version": 1,
      "measured_at": "<ISO-8601 with timezone>",
      "measured_by": "molmetal/scripts/wf_launch_gate_exitcode_probe.py",
      "host": "<one-line host summary>",
      "tools": {
        "<tool file basename>": {
          "exit_code_under_failing_input": <int>,
          "command": "<the exact copy-pasteable command that produced it>",
          "failing_input": "<what makes this input FAIL>",
          "can_carry_fail": <true|false>,
          "verdict_location": "<where the FAIL verdict actually lives, or null>"
        }
      }
    }

``can_carry_fail`` is true iff the exit code under a failing input is
NON-ZERO.  ``verdict_location`` names where the verdict really lives --
for a tool whose verdict lives in a report rather than in its exit
code, it names the report field AND the shared consumer
(``molmetal/scripts/wf_launch_gate_verdict.py``) sub-command that reads
it.  The guard is told which consumer to use so it never has to
re-implement report parsing.

SAFETY / SCOPE

  * CPU-only.  Every tool is launched with ``CUDA_VISIBLE_DEVICES=""``
    and the repo's own venv interpreter.  No trainer, sampler, sweep or
    GPU job is launched.  ``--help`` imports nothing beyond the stdlib
    and the repo's atomic writer (no torch, no GPU).
  * Every crafted failing input is written under a work directory in
    ``/tmp`` (default ``/tmp/wf_launch_gate_exitcode_probe``).  Nothing
    is written into the repo except the registry file itself, and
    nothing is written into ``metrics/`` or ``paper/``.
  * The work directory is NOT a random temp dir: it is stable, so the
    ``command`` recorded in the registry stays copy-pasteable and
    re-runnable by a reader (the fixtures are still on disk).

THIS PROBE IS NOT A DEAD GATE ITSELF (catalog §6.269.17 / research §B6)

  A measuring tool that always reports the same thing measures nothing.
  Two properties make this one load-bearing:

    1. It EXECUTES the tools.  Nothing in the registry is a constant —
       every ``exit_code_under_failing_input`` is the return code of a
       real subprocess run whose argv is recorded next to it.
    2. ``--compare <registry>`` re-reads a baseline registry and prints
       a field-by-field DIFF against the freshly measured values, so a
       stale or tampered registry is visible.  With ``--fail-on-diff``
       the differences become a non-zero exit.

Usage

    python molmetal/scripts/wf_launch_gate_exitcode_probe.py
    python molmetal/scripts/wf_launch_gate_exitcode_probe.py --only per_wave_checklist.py
    python molmetal/scripts/wf_launch_gate_exitcode_probe.py --compare /tmp/baseline.json
    python molmetal/scripts/wf_launch_gate_exitcode_probe.py --list

Stdout invariant (AGENTS.md §4): the LAST line's first token is
``PASS`` / ``FAIL`` / ``SKIP``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"

DEFAULT_REGISTRY = SCRIPTS_DIR / "wf_launch_gate_exitcode_registry.json"
DEFAULT_WORK_DIR = Path("/tmp/wf_launch_gate_exitcode_probe")

SCHEMA_VERSION = 1
MEASURED_BY = "molmetal/scripts/wf_launch_gate_exitcode_probe.py"

# Every subprocess inherits this: CPU-only, and no PYTHONPATH surprises.
CHILD_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONWARNINGS": "ignore",
}

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from wf386a3_atomic_write import atomic_write_file  # noqa: E402


# ---------------------------------------------------------------------------
# Probe specifications
# ---------------------------------------------------------------------------


class ToolProbe:
    """One tool + the synthetic FAILING input it is measured against.

    ``builder(work)`` writes the fixtures into ``work`` (a per-tool
    sub-directory of the work root) and returns the argv to execute.
    """

    def __init__(
        self,
        tool: str,
        failing_input: str,
        verdict_location: Optional[str],
        builder: Callable[[Path], List[str]],
        timeout: int = 600,
    ) -> None:
        self.tool = tool
        self.failing_input = failing_input
        self.verdict_location = verdict_location
        self.builder = builder
        self.timeout = timeout

    @property
    def tool_path(self) -> Path:
        return SCRIPTS_DIR / self.tool


# --- fixture builders ------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _crafted_preflight(path: Path) -> Path:
    """A pre-flight report whose summary says FAIL and whose P0 gate FAILs.

    This is the shared fixture for ``per_wave_checklist.py`` and
    ``wf343c2_per_wave_check.py``: both read a pre-flight report and
    derive their verdict from it.
    """
    return _write_json(
        path,
        {
            "summary": {
                "pass": 10,
                "fail": 1,
                "skip": 0,
                "timeout": 0,
                "total": 11,
            },
            "gates": [
                {
                    "name": "p0_blocker_probe",
                    "priority": "P0",
                    "status": "FAIL",
                    "evidence": "crafted failing input for the exit-code probe",
                },
                {
                    "name": "p1_probe_ok",
                    "priority": "P1",
                    "status": "PASS",
                    "evidence": "crafted",
                },
            ],
        },
    )


def _build_pre_flight(work: Path) -> List[str]:
    out = work / "pre_flight.json"
    return [
        str(VENV_PY),
        str(SCRIPTS_DIR / "pre_flight_check.py"),
        "--pipeline",
        "all",
        "--json",
        str(out),
    ]


def _build_wave_budget(work: Path) -> List[str]:
    out = work / "budget.json"
    return [
        str(VENV_PY),
        str(SCRIPTS_DIR / "wave_budget_tracker.py"),
        "--budget-hr",
        "0.1",
        "--report-json",
        str(out),
    ]


def _build_per_wave_checklist(work: Path) -> List[str]:
    pf = _crafted_preflight(work / "pre_flight_fail.json")
    return [
        str(VENV_PY),
        str(SCRIPTS_DIR / "per_wave_checklist.py"),
        "--wave",
        "Wf-A269-Probe-Wave",
        "--preflight",
        str(pf),
        "--out",
        str(work / "per_wave.json"),
        "--out-md",
        str(work / "per_wave.md"),
    ]


def _build_wf343c2(work: Path) -> List[str]:
    pf = _crafted_preflight(work / "pre_flight_fail.json")
    return [
        str(VENV_PY),
        str(SCRIPTS_DIR / "wf343c2_per_wave_check.py"),
        "--wave",
        "Wf-A269-Probe-Wave",
        "--preflight",
        str(pf),
        "--preflight-live",
        "--out",
        str(work / "wf343c2.json"),
        "--skip-live",
    ]


_WF343C3_DRIVER = '''"""Crafted-loader-failure driver for wf343c3_data_quality_check.py.

The tool's CLI has no flag that injects a loader failure, and the probe
may not mutate the repo's data files, so the crafted frame is supplied by
patching ONE loader inside the tool's own module namespace and then
calling the tool's own main().  The tool's own exit-code logic decides
the verdict; the driver never decides anything itself.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tool", required=True)
ap.add_argument("--crafted-csv", required=True)
ap.add_argument("--tool-arg", action="append", default=[])
args = ap.parse_args()

tool = Path(args.tool).resolve()
spec = importlib.util.spec_from_file_location("wf343c3_dq_probe", tool)
mod = importlib.util.module_from_spec(spec)
sys.modules["wf343c3_dq_probe"] = mod
spec.loader.exec_module(mod)

crafted_csv = Path(args.crafted_csv).resolve()
mod._load_binding_db = lambda: pd.read_csv(crafted_csv)

sys.exit(mod.main(list(args.tool_arg)))
'''

# A crafted loader frame that FAILs the tool's own checks: one row has a
# NaN pIC50 (check 5 -> FAIL) and the class balance is 50/50 (check 4 PASS).
_CRAFTED_BINDING_DB_CSV = """smiles,metal,cell_line,protein,pIC50,active
CCO,Ru,A2780,carbonic anhydrase,4.5,1
c1ccccc1,Ir,A2780,carbonic anhydrase,,1
CC(=O)O,Os,MCF7,carbonic anhydrase,5.1,0
CCN,Re,MCF7,carbonic anhydrase,3.2,0
"""


def _build_wf343c3(work: Path) -> List[str]:
    driver = _write(work / "drive_wf343c3_data_quality_check.py", _WF343C3_DRIVER)
    csv_path = _write(work / "crafted_binding_db_fail.csv", _CRAFTED_BINDING_DB_CSV)
    return [
        str(VENV_PY),
        str(driver),
        "--tool",
        str(SCRIPTS_DIR / "wf343c3_data_quality_check.py"),
        "--crafted-csv",
        str(csv_path),
        "--tool-arg=--strict",
        "--tool-arg=--out-dir",
        f"--tool-arg={work / 'out'}",
    ]


def _build_abort_recovery(work: Path) -> List[str]:
    # The crafted failing log is the TRIGGER artefact a wrapper greps for;
    # abort_recovery.py itself consumes the checkpoint, not the log.
    _write(
        work / "train_ru_seed0.log",
        "2026-11-02 10:00:00,123 ERROR agent aborted: CUDA out of memory "
        "(OOM) while training metal=Ru seed=0\n"
        "Traceback (most recent call last):\n"
        "  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n",
    )
    # A checkpoint candidate that torch.load cannot read -> the resume
    # verification fails -> recovered is False.
    _write(work / "ckpt_ru_seed0.pt", "this is not a torch checkpoint\n")
    return [
        str(VENV_PY),
        str(SCRIPTS_DIR / "abort_recovery.py"),
        "--pattern",
        "OOM",
        "--metal",
        "Ru",
        "--ckpt-glob",
        str(work / "ckpt_*.pt"),
        "--no-default-glob",
        "--report-json",
        str(work / "recovery.json"),
    ]


def _build_verdict_consumer(work: Path) -> List[str]:
    pf = _crafted_preflight(work / "pre_flight_fail.json")
    return [
        str(VENV_PY),
        str(SCRIPTS_DIR / "wf_launch_gate_verdict.py"),
        "preflight",
        "--report",
        str(pf),
    ]


PROBES: List[ToolProbe] = [
    ToolProbe(
        tool="pre_flight_check.py",
        failing_input=(
            "the repo's own LIVE pre-flight (--pipeline all) on this tree: at "
            "least one gate FAILs, so the report's summary.fail > 0"
        ),
        verdict_location=(
            "the process exit code (1 when summary.fail > 0) AND the "
            "`summary` block of the --json report; consume with "
            "`wf_launch_gate_verdict.py preflight --report <pre_flight.json>`"
        ),
        builder=_build_pre_flight,
        timeout=1800,
    ),
    ToolProbe(
        tool="wave_budget_tracker.py",
        failing_input=(
            "--budget-hr 0.1 against the repo's auto-discovered wave list: the "
            "tracker must DEFER waves and report remaining_hr < 0"
        ),
        verdict_location=(
            "NOT the exit code (0 on every path): the `remaining_hr` and "
            "`deferred_waves` fields of the --report-json report; consume with "
            "`wf_launch_gate_verdict.py budget --report <budget.json>`"
        ),
        builder=_build_wave_budget,
    ),
    ToolProbe(
        tool="per_wave_checklist.py",
        failing_input=(
            "a crafted pre-flight report with summary.fail=1 and a P0 blocker "
            "gate with status=FAIL, passed via --preflight"
        ),
        verdict_location=(
            "NOT the exit code (0 on every path): the `preflight_fail_count` "
            "and `p0_blocker_gates[].status` fields of the --out JSON; consume "
            "with `wf_launch_gate_verdict.py checklist --report <per_wave.json>`"
        ),
        builder=_build_per_wave_checklist,
    ),
    ToolProbe(
        tool="wf343c2_per_wave_check.py",
        failing_input=(
            "the same crafted pre-flight report (summary.fail=1, a P0 gate "
            "FAIL) via --preflight --preflight-live; --skip-live keeps the "
            "GPU/disk probes out of the measurement"
        ),
        verdict_location=(
            "the process exit code (1 when any check FAILs) AND the "
            "`overall_status` / `checks[].status` of the --out JSON"
        ),
        builder=_build_wf343c2,
    ),
    ToolProbe(
        tool="wf343c3_data_quality_check.py",
        failing_input=(
            "a crafted loader CSV (one NaN pIC50 row) injected into the "
            "tool's own _load_binding_db by a generated driver, so ONE loader "
            "FAILs; the tool itself is run with --strict"
        ),
        verdict_location=(
            "the process exit code (1 when a loader FAILs, or when --strict "
            "escalates a loader-level SKIP) AND the `loader_reports[].status` "
            "of the --out-dir report.json"
        ),
        builder=_build_wf343c3,
    ),
    ToolProbe(
        tool="abort_recovery.py",
        failing_input=(
            "a crafted failing train log carrying the OOM signature plus a "
            "checkpoint candidate that torch.load cannot read; --pattern OOM "
            "--metal Ru are both supplied (the log is the trigger artefact a "
            "wrapper greps for; the tool itself consumes the checkpoint)"
        ),
        verdict_location=(
            "the process exit code (1 when `recovered` is false) AND the "
            "`recovered` / `verify` fields of the --report-json report"
        ),
        builder=_build_abort_recovery,
    ),
    ToolProbe(
        tool="wf_launch_gate_verdict.py",
        failing_input=(
            "a crafted pre-flight report with summary.fail=1 (the shared "
            "verdict CONSUMER; a FAIL here is the expected behaviour)"
        ),
        verdict_location=(
            "the process exit code (1 on FAIL) and the first stdout token; "
            "with --json the verdict is the `verdict` / `detail` fields"
        ),
        builder=_build_verdict_consumer,
    ),
]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.update(CHILD_ENV)
    # Make sure the repo's venv interpreter is the only one in play.
    env["PATH"] = str(VENV_PY.parent) + os.pathsep + env.get("PATH", "")
    return env


def _command_string(argv: Sequence[str]) -> str:
    """The exact copy-pasteable command, GPU-disabled and absolute."""
    return "CUDA_VISIBLE_DEVICES=\"\" " + " ".join(shlex.quote(a) for a in argv)


def measure(probe: ToolProbe, work_root: Path) -> Dict[str, Any]:
    """Run one tool against its failing input and record what happened."""
    work = work_root / probe.tool.replace(".py", "")
    if work.exists():
        for child in sorted(work.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    work.mkdir(parents=True, exist_ok=True)

    argv = probe.builder(work)
    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=probe.timeout,
            env=_child_env(),
            cwd=str(REPO_ROOT),
        )
        exit_code = proc.returncode
        stdout_bytes = len(proc.stdout.encode("utf-8", "replace"))
        stderr_bytes = len(proc.stderr.encode("utf-8", "replace"))
        stdout_head = (proc.stdout or "").strip().splitlines()[:3]
    except subprocess.TimeoutExpired:
        timed_out = True
        exit_code = -1
        stdout_bytes = 0
        stderr_bytes = 0
        stdout_head = []
    wall = round(time.perf_counter() - started, 3)

    can_carry_fail = bool(not timed_out and exit_code != 0)
    entry: Dict[str, Any] = {
        "exit_code_under_failing_input": exit_code,
        "command": _command_string(argv),
        "failing_input": probe.failing_input,
        "can_carry_fail": can_carry_fail,
        "verdict_location": probe.verdict_location,
        "diagnostics": {
            "timed_out": timed_out,
            "timeout_sec": probe.timeout,
            "wall_sec": wall,
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
            "stdout_head": stdout_head,
            "work_dir": str(work),
        },
    }
    return entry


# ---------------------------------------------------------------------------
# Registry assembly
# ---------------------------------------------------------------------------


def host_summary() -> str:
    return (
        f"{platform.system()}-{platform.release()}; "
        f"node={platform.node()}; "
        f"python={platform.python_version()}; "
        f"CUDA_VISIBLE_DEVICES='' (CPU-only)"
    )


def build_registry(entries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "measured_by": MEASURED_BY,
        "host": host_summary(),
        "tools": entries,
    }


def diff_registry(
    measured: Dict[str, Any], baseline: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    """Differences between a measurement and a baseline registry.

    Returns ``(field_diffs, coverage_diffs)``.  A FIELD difference is a
    value that was measured differently from the baseline -- it is the
    load-bearing signal and it is what ``--fail-on-diff`` trips on.  A
    COVERAGE difference is a tool that is in one registry and not the
    other; it is reported (a registry that silently lost a tool is
    visible) but it is not a field difference.
    """
    field_diffs: List[str] = []
    coverage_diffs: List[str] = []
    m_tools = measured.get("tools", {}) or {}
    b_tools = baseline.get("tools", {}) or {}
    for name in sorted(set(m_tools) | set(b_tools)):
        if name not in b_tools:
            coverage_diffs.append(f"COVERAGE {name}: measured, absent from the baseline")
            continue
        if name not in m_tools:
            coverage_diffs.append(f"COVERAGE {name}: baseline only, not measured in this run")
            continue
        for key in (
            "exit_code_under_failing_input",
            "can_carry_fail",
            "verdict_location",
        ):
            mv = m_tools[name].get(key)
            bv = b_tools[name].get(key)
            if mv != bv:
                field_diffs.append(
                    f"DIFF {name}.{key}: measured={mv!r} baseline={bv!r}"
                )
    return field_diffs, coverage_diffs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_argv(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wf_launch_gate_exitcode_probe.py",
        description=(
            "Measure the exit code every launch-gate tool returns under a "
            "synthetic FAILING input, and write the measured registry the "
            "dead-gate guard reads. CPU-only: no trainer, sampler, sweep or "
            "GPU job is launched."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=(
            "where the measured registry JSON is written "
            f"(default: {DEFAULT_REGISTRY})"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "optional path for the detailed measurement report (per-tool "
            "argv, return code, byte counts, wall time); the registry file "
            "itself is written to --registry"
        ),
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help=(
            "root directory for the crafted failing inputs "
            f"(default: {DEFAULT_WORK_DIR}); never inside the repo"
        ),
    )
    p.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="TOOL.py",
        help="probe only this tool (repeatable)",
    )
    p.add_argument(
        "--compare",
        type=Path,
        default=None,
        help=(
            "baseline registry to compare the fresh measurement against; "
            "differences are printed as DIFF lines"
        ),
    )
    p.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="exit 1 when --compare finds any difference",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list the tools this probe measures and exit",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_argv(argv)

    if args.list:
        print(f"{len(PROBES)} launch-gate tools measured by {MEASURED_BY}")
        for probe in PROBES:
            print(f"  {probe.tool}")
        return 0

    selected = PROBES
    if args.only:
        wanted = {name.strip() for name in args.only}
        selected = [p for p in PROBES if p.tool in wanted]
        missing = wanted - {p.tool for p in selected}
        if missing:
            print(f"FAIL unknown tool(s): {', '.join(sorted(missing))}")
            return 1
        if not selected:
            print("SKIP no tool selected")
            return 2

    work_root = args.work_dir
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"MEASURED {len(selected)} launch-gate tools (CPU-only, CUDA_VISIBLE_DEVICES=\"\")")
    print(f"work_dir: {work_root}")

    entries: Dict[str, Dict[str, Any]] = {}
    report: Dict[str, Any] = {}
    n_ok = 0
    for idx, probe in enumerate(selected, start=1):
        entry = measure(probe, work_root)
        entries[probe.tool] = entry
        report[probe.tool] = {
            "argv": entry["command"],
            "returncode": entry["exit_code_under_failing_input"],
            "timed_out": entry["diagnostics"]["timed_out"],
            "wall_sec": entry["diagnostics"]["wall_sec"],
            "stdout_bytes": entry["diagnostics"]["stdout_bytes"],
            "stderr_bytes": entry["diagnostics"]["stderr_bytes"],
            "stdout_head": entry["diagnostics"]["stdout_head"],
        }
        verdict = "CAN_CARRY_FAIL" if entry["can_carry_fail"] else "CANNOT_CARRY_FAIL"
        if not entry["diagnostics"]["timed_out"]:
            n_ok += 1
        print("")
        print(f"[{idx}/{len(selected)}] {probe.tool}")
        print(f"      command: {entry['command']}")
        print(f"      exit_code_under_failing_input: {entry['exit_code_under_failing_input']}")
        print(f"      verdict: {verdict}")
        print(f"      verdict_location: {entry['verdict_location']}")
        print(
            "      diagnostics: wall={wall}s stdout={so}B stderr={se}B "
            "timed_out={to}".format(
                wall=entry["diagnostics"]["wall_sec"],
                so=entry["diagnostics"]["stdout_bytes"],
                se=entry["diagnostics"]["stderr_bytes"],
                to=entry["diagnostics"]["timed_out"],
            )
        )
        for line in entry["diagnostics"]["stdout_head"]:
            print(f"      stdout| {line}")

    registry = build_registry(entries)

    n_diff = 0
    if args.compare:
        baseline_path = args.compare
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"FAIL could not read the baseline registry {baseline_path}: {exc}")
            return 1
        field_diffs, coverage_diffs = diff_registry(registry, baseline)
        n_diff = len(field_diffs)
        print("")
        print(f"COMPARE against {baseline_path}")
        for line in coverage_diffs:
            print(f"      {line}")
        if coverage_diffs:
            print(f"COVERAGE DIFFERENCES: {len(coverage_diffs)}")
        if field_diffs:
            for line in field_diffs:
                print(f"      {line}")
            print(f"DIFFERENCES: {len(field_diffs)}")
        else:
            print("      no field differences")
            print("DIFFERENCES: 0")

    registry_path = args.registry
    atomic_write_file(
        registry_path,
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
    )
    print("")
    print(f"registry written: {registry_path}")

    if args.out:
        atomic_write_file(
            args.out,
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "measured_at": registry["measured_at"],
                    "measured_by": MEASURED_BY,
                    "host": registry["host"],
                    "tools": report,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )
        print(f"measurement report written: {args.out}")

    can = [t for t, e in entries.items() if e["can_carry_fail"]]
    cannot = [t for t, e in entries.items() if not e["can_carry_fail"]]
    print("")
    print("SUMMARY")
    print(f"  CAN_CARRY_FAIL    : {', '.join(can) if can else '(none)'}")
    print(f"  CANNOT_CARRY_FAIL : {', '.join(cannot) if cannot else '(none)'}")

    if args.fail_on_diff and n_diff:
        print(f"FAIL {n_diff} field(s) differ from the baseline registry")
        return 1
    if n_ok < len(selected):
        print(f"FAIL {len(selected) - n_ok}/{len(selected)} tool(s) could not be measured")
        return 1
    print(f"PASS {len(selected)}/{len(selected)} tool(s) measured; registry is MEASURED, not asserted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
