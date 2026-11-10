# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

r"""Wf-A269 wave-10 C1-7 — the DEAD SECOND BLOCKER of the Wf-299 pre-flight.

WHAT THIS FILE GUARDS (Wf-A269 wave-10, 2026-11-02)
----------------------------------------------------
``scripts/launch_wf299_cfm_main.sh`` gated its pre-flight TWICE:

    if ! "${PY}" "${PRE_FLIGHT}" --pipeline all \
            > "${REPORTS_DIR}/pre_flight.json" 2>&1; then
        die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
    fi
    ...
    [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v2 §1)"

The SECOND of those was a DEAD BRANCH.  MEASURED: ``pre_flight_check.py``
returns 1 **iff** ``n_fail > 0`` (molmetal/scripts/pre_flight_check.py:
:17741-17742 — ``if n_fail > 0: return 1``), so the rc guard above had
already terminated the script before the count-based blocker could ever
run.  A gate that cannot fail is the wave-10 defect class: it looks like a
check, it prints nothing when it is skipped, and a reader believes a
measurement happened.

THE REPAIR (C1-7)
-----------------
ONE pre-flight failure decision, and it is a NUMERIC COMPARISON on the
CAPTURED exit code of the pre-flight tool:

    local pf_rc=0
    "${PY}" "${PRE_FLIGHT}" --pipeline all --json "${REPORTS_DIR}/pre_flight.json" \
        > "${REPORTS_DIR}/pre_flight.stdout.log" 2>&1 || pf_rc=$?
    ...
    if [ "${pf_rc}" -ne 0 ] || [ "${pf_verdict}" != "PASS" ]; then
        warn "pre_flight_check ${pf_verdict} (rc=${pf_rc}; ${pf_detail})"
        die "pre_flight_check ${pf_verdict} — refusing to launch (rc=${pf_rc}; ...)"
    fi

The report-based verdict (the shared consumer
``molmetal/scripts/wf_launch_gate_verdict.py``) is OR-ed into the SAME
decision — one ``die``, two fail-closed signals, never a second blocker.
The count-based blocker survives only inside the AGENTS.md §5.5 rule #6
verbatim-preserved comment block, which the assertions strip before
looking at the CODE.

THE TESTING METHOD
-------------------
Two layers, both hermetic and CPU-only:

1. STATIC — the wrapper's ACTIVE code (full-line comments stripped, so the
   verbatim-preserved superseded rendering cannot satisfy an assertion)
   must carry exactly ONE pre-flight failure decision, that decision must be
   a numeric comparison on a captured rc, and the count-based second
   blocker must be gone from the CODE.
2. BEHAVIOURAL — the wrapper's ``pre_flight_gate`` function is extracted and
   executed in a sandbox harness with a stubbed ``${PY}`` and a stubbed
   pre-flight report.  The gate must be LIVE: a failing pre-flight must
   kill the launch exactly once, a clean one must let it through, and a
   pre-flight that measured NOTHING must be refused (UNVERIFIED, never a
   PASS).

No GPU, no torch, no trainer launch, no repo file is written: every path
the harness touches lives under pytest's ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from molmetal.tests.wfa269_report_fixture import stamp_live_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "launch_wf299_cfm_main.sh"
CHECKER = REPO_ROOT / "molmetal" / "scripts" / "wf_launch_gate_verdict.py"
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _code_only(text: str) -> str:
    """Strip full-line shell comments.

    The wrapper keeps the superseded rendering of a repaired gate in a
    comment block (AGENTS.md §5.5 rule #6), so the dead strings are still
    on disk.  The assertion has to be about the CODE that runs.
    """
    out = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def _function_source(text: str, name: str) -> str:
    """Return the bash source of the named top-level function."""
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.startswith(f"{name}() {{"):
            start = idx
            break
    if start is None:
        raise AssertionError(f"function {name}() not found in {WRAPPER.name}")
    out = [lines[start]]
    for line in lines[start + 1:]:
        out.append(line)
        if line == "}":
            return "\n".join(out)
    raise AssertionError(f"function {name}() is never closed in {WRAPPER.name}")


def _pre_flight_gate_code() -> str:
    """The ACTIVE (comment-free) source of the wrapper's pre-flight gate."""
    raw = WRAPPER.read_text(encoding="utf-8")
    return _code_only(_function_source(raw, "pre_flight_gate"))


def _report(n_fail: int) -> str:
    """A pre-flight report in the shape the shared checker reads.

    Wf-A269 wave-16/17 (2026-11-03) — ADDITIVE.  The shared checker became
    FAIL-CLOSED: it refuses a report with no ``last_refresh`` stamp, no
    ``provenance`` block, a foreign commit SHA, a foreign gate-source hash,
    or a payload hash that does not match the report's own content.  Every
    one of those refusals is correct, but this fixture would then be
    refused for a reason unrelated to what these tests are about, so it now
    carries what the producer would really have written (see
    ``molmetal/tests/wfa269_report_fixture.py``).

    SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    the payload below used to carry the ``summary`` and ``gates`` keys alone.

    2026-11-04 (Wf-A269 dead-gates sweep) — ADDITIVE.  The gate rows now
    carry a ``priority``.  The producer writes
    ``GateResult.to_dict()`` = name / priority / pipeline / status /
    evidence / wall_sec for every gate, and the shared checker's preflight
    arm now enforces the P0 blocker rule its sibling arm already enforced,
    so it reads that field.  The rows here are the report's BLOCKERS, so
    they are P0.
    """
    payload = {
        "summary": {
            "pass": 3 - n_fail,
            "fail": n_fail,
            "skip": 0,
            "timeout": 0,
            "total": 3,
        },
        "gates": [
            {
                "name": f"gate_{i}",
                "priority": "P0",
                "pipeline": "all",
                "status": "PASS",
            }
            for i in range(3 - n_fail)
        ]
        + [
            {
                "name": f"gate_{i}",
                "priority": "P0",
                "pipeline": "all",
                "status": "FAIL",
            }
            for i in range(n_fail)
        ],
    }
    return json.dumps(stamp_live_provenance(payload))


_STUB_PY = r"""#!/usr/bin/env bash
# Stub ${PY} for the Wf-A269 wave-10 C1-7 harness.  It dispatches on $1:
#   * the pre-flight tool  -> writes a fake report, exits FAKE_PF_RC
#   * the shared checker   -> the REAL checker (it only reads a report)
#   * the four chain gates -> exit 0 (the chain verdict is not under test)
set -u
REAL_PY="__REAL_PY__"
case "$1" in
    *pre_flight_check.py)
        out=""
        prev=""
        for a in "$@"; do
            if [ "${prev}" = "--json" ]; then out="${a}"; fi
            prev="${a}"
        done
        if [ "${FAKE_PF_WRITE_REPORT:-1}" = "1" ]; then
            if [ -n "${out}" ]; then
                printf '%s\n' "${FAKE_PF_REPORT}" > "${out}"
            else
                printf '%s\n' "${FAKE_PF_REPORT}"
            fi
        fi
        exit "${FAKE_PF_RC:-0}"
        ;;
    *wf343c2_per_wave_check.py|*wf343c3_data_quality_check.py|\
*wf385a3_scaffold_wire_in_gate.py|*wf394a3_wf299_pre_launch_verify.py)
        exit 0
        ;;
    *)
        exec "${REAL_PY}" "$@"
        ;;
esac
"""

_HARNESS = r"""#!/usr/bin/env bash
set -euo pipefail

REPORTS_DIR="__REPORTS__"
PY="__STUB_PY__"
REAL_PY="__REAL_PY__"
PRE_FLIGHT="__PRE_FLIGHT__"
GATE_VERDICT_CHECKER="__CHECKER__"
DRY_RUN=0
WF343C2_CHECK="__WF343C2__"
WF343C3_CHECK="__WF343C3__"
WF385A3_GATE="__WF385A3__"
WF394A3_VERIFY="__WF394A3__"

mkdir -p "${REPORTS_DIR}"

log()  { printf '[LOG] %s\n' "$*"; }
ok()   { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
die()  { printf '[FATAL] %s\n' "$*"; exit 1; }

__FUNCTIONS__

pre_flight_gate
printf '[HARNESS-DONE] rc=0\n'
"""


def _function_or_fallback(raw: str, name: str, fallback: str) -> str:
    """The wrapper's own function, or a stand-in when it has none.

    The fallbacks keep the harness usable against the PRE-repair tree (which
    had no shared-verdict helpers), so a behavioural run on the old code is
    a real measurement rather than a collection error.
    """
    try:
        return _function_source(raw, name)
    except AssertionError:
        return fallback


def _run_harness(tmp_path: Path, *, pf_rc: int, report: str | None) -> subprocess.CompletedProcess:
    """Execute the wrapper's pre_flight_gate in a sandbox.

    ``pf_rc`` is the exit code the stubbed pre-flight tool returns;
    ``report`` is the JSON report it writes (``None`` = it writes nothing).
    """
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(exist_ok=True)

    stub = tmp_path / "stub_py.sh"
    stub.write_text(
        _STUB_PY.replace("__REAL_PY__", str(VENV_PY)), encoding="utf-8"
    )
    stub.chmod(0o755)

    raw = WRAPPER.read_text(encoding="utf-8")
    functions = "\n\n".join(
        _function_or_fallback(raw, name, fallback)
        for name, fallback in (
            (
                "run_gate_check",
                'run_gate_check() { printf "PASS"; }',
            ),
            (
                "gate_detail",
                'gate_detail() { printf ""; }',
            ),
            (
                "gate_verdict_from_rc",
                'gate_verdict_from_rc() { case "$1" in 0) printf "PASS";; '
                '1) printf "FAIL";; *) printf "UNVERIFIED";; esac; }',
            ),
            (
                "pre_flight_gate",
                "",
            ),
        )
    )

    harness = tmp_path / "harness.sh"
    harness.write_text(
        _HARNESS.replace("__REPORTS__", str(reports_dir))
        .replace("__STUB_PY__", str(stub))
        .replace("__REAL_PY__", str(VENV_PY))
        .replace(
            "__PRE_FLIGHT__",
            str(REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"),
        )
        .replace("__CHECKER__", str(CHECKER))
        .replace(
            "__WF343C2__",
            str(REPO_ROOT / "molmetal" / "scripts" / "wf343c2_per_wave_check.py"),
        )
        .replace(
            "__WF343C3__",
            str(REPO_ROOT / "molmetal" / "scripts" / "wf343c3_data_quality_check.py"),
        )
        .replace(
            "__WF385A3__",
            str(REPO_ROOT / "molmetal" / "scripts" / "wf385a3_scaffold_wire_in_gate.py"),
        )
        .replace(
            "__WF394A3__",
            str(REPO_ROOT / "molmetal" / "scripts" / "wf394a3_wf299_pre_launch_verify.py"),
        )
        .replace("__FUNCTIONS__", functions),
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "FAKE_PF_RC": str(pf_rc),
        "FAKE_PF_WRITE_REPORT": "0" if report is None else "1",
        "FAKE_PF_REPORT": report or "",
    }
    return subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def _fatal_lines(output: str) -> list[str]:
    return [ln for ln in output.splitlines() if ln.startswith("[FATAL]")]


# ---------------------------------------------------------------------------
# Test 1 (STATIC) — exactly ONE pre-flight failure decision
# ---------------------------------------------------------------------------


def test_preflight_has_exactly_one_failure_decision() -> None:
    """The pre-flight must not decide twice.

    MEASURED before the repair: the wrapper had BOTH an rc guard
    (`if ! "${PY}" "${PRE_FLIGHT}" ...; then die; fi`) AND a count-based
    blocker (`[ "${n_fail}" -eq 0 ] || die ...`).  The second was
    unreachable — pre_flight_check.py returns 1 iff n_fail > 0
    (pre_flight_check.py:17741-17742) — so it was a DEAD BRANCH.
    """
    assert WRAPPER.is_file(), f"wrapper missing: {WRAPPER}"
    body = _pre_flight_gate_code()

    decisions = [
        ln
        for ln in body.splitlines()
        if re.search(r"\bdie\b", ln) and "pre_flight" in ln.lower()
    ]
    assert len(decisions) == 1, (
        f"{WRAPPER.name}: the pre-flight must decide EXACTLY ONCE, but "
        f"{len(decisions)} pre-flight failure decisions are in the active "
        f"code: {decisions!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 (STATIC) — the surviving decision is a numeric rc comparison
# ---------------------------------------------------------------------------


def test_preflight_decision_is_a_numeric_comparison_on_the_captured_rc() -> None:
    """The one surviving decision must compare the tool's exit code.

    The exit code of the gate tool is the verdict channel AGENTS.md §4
    fixes for every gate in this repo (0=PASS, 1=FAIL, 2=SKIP), and
    pre_flight_check.py returns 1 iff n_fail > 0
    (pre_flight_check.py:17741-17742).  The rc must therefore be
    CAPTURED and compared NUMERICALLY — not thrown away.
    """
    body = _pre_flight_gate_code()

    captures = re.findall(r"\|\|\s*([A-Za-z_][A-Za-z0-9_]*)=\$\?", body)
    assert captures, (
        f"{WRAPPER.name}: the pre-flight tool's exit code is never CAPTURED "
        f"— the decision cannot be a comparison on it"
    )
    rc_vars = {v for v in captures if v.lower().endswith("rc") or v.lower().endswith("_rc")}
    assert rc_vars, (
        f"{WRAPPER.name}: the captured exit code(s) {sorted(captures)} do not "
        f"name an rc variable"
    )

    numeric = re.findall(
        r"\[\s*\"\$\{(" + "|".join(sorted(rc_vars)) + r")\}\"\s+-(?:eq|ne)\s+\d+\s*\]",
        body,
    )
    assert numeric, (
        f"{WRAPPER.name}: the captured rc {sorted(rc_vars)} is never compared "
        f"numerically in the pre-flight decision — the surviving check must "
        f"be a numeric comparison on the captured rc"
    )


def test_the_dead_count_blocker_is_not_in_the_active_code() -> None:
    """The unreachable count-based blocker must be gone from the CODE.

    It may (and does) survive in the AGENTS.md §5.5 rule #6
    verbatim-preserved comment block; the assertions strip full-line
    comments, so only the CODE is judged.
    """
    body = _pre_flight_gate_code()
    assert '[ "${n_fail}" -eq 0 ]' not in body, (
        f"{WRAPPER.name}: the DEAD count-based second blocker is still in the "
        f"active code — a check that cannot fail (pre_flight_check.py "
        f"returns 1 iff n_fail > 0, so the rc guard always fires first)"
    )


# ---------------------------------------------------------------------------
# Test 4 (BEHAVIOURAL) — the gate is LIVE: it can fail
# ---------------------------------------------------------------------------


def test_failing_preflight_kills_the_launch_exactly_once(tmp_path: Path) -> None:
    """A failing pre-flight must stop the launch, and stop it ONCE.

    MEASURED before the repair: the count-based blocker could never run,
    so the wrapper had a second decision that no input could ever reach.
    """
    proc = _run_harness(tmp_path, pf_rc=1, report=_report(n_fail=1))

    fatals = _fatal_lines(proc.stdout + proc.stderr)
    assert len(fatals) == 1, (
        f"a FAILING pre-flight must produce exactly ONE [FATAL] line, got "
        f"{len(fatals)}: {fatals!r}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    assert "pre_flight" in fatals[0].lower(), (
        f"the [FATAL] line must name the pre-flight gate: {fatals[0]!r}"
    )
    assert proc.returncode != 0, (
        "a failing pre-flight must leave the wrapper with a non-zero exit "
        f"code (got {proc.returncode})"
    )


# ---------------------------------------------------------------------------
# Test 5 (BEHAVIOURAL) — a clean pre-flight must let the launch through
# ---------------------------------------------------------------------------


def test_clean_preflight_passes_the_gate(tmp_path: Path) -> None:
    """The gate must not be stuck closed either."""
    proc = _run_harness(tmp_path, pf_rc=0, report=_report(n_fail=0))

    assert not _fatal_lines(proc.stdout + proc.stderr), (
        "a CLEAN pre-flight must not kill the launch:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert re.search(r"^\[OK\] pre_flight", proc.stdout, re.MULTILINE), (
        "a clean pre-flight must emit the counts-bearing [OK] pre_flight "
        f"line:\n--- stdout ---\n{proc.stdout}"
    )
    assert proc.returncode == 0, (
        f"a clean pre-flight must leave the wrapper with exit 0, got "
        f"{proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 6 (BEHAVIOURAL) — a pre-flight that measured NOTHING is not a PASS
# ---------------------------------------------------------------------------


def test_unmeasurable_preflight_is_refused_not_passed(tmp_path: Path) -> None:
    """A pre-flight that wrote no report must be UNVERIFIED, never a PASS.

    This is the fail-closed direction of the decision: the gate refuses to
    launch rather than reporting a clean bill of health it never measured.
    """
    proc = _run_harness(tmp_path, pf_rc=2, report=None)

    fatals = _fatal_lines(proc.stdout + proc.stderr)
    assert len(fatals) == 1, (
        f"a pre-flight that produced NO report must produce exactly ONE "
        f"[FATAL] line, got {len(fatals)}: {fatals!r}\n--- stdout ---\n"
        f"{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert proc.returncode != 0, (
        "an UNMEASURED pre-flight must not leave the wrapper with exit 0"
    )
