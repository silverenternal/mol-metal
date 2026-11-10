r"""Wf-A269 wave-10 — the DEAD-GATE class: a check that cannot fail.

WHAT THIS FILE GUARDS (Wf-A269 wave-10, 2026-11-02)
----------------------------------------------------
A "gate" that cannot fail is worse than no gate: it prints PASS and the
reader believes a measurement happened.  Four live instances of that class
were found and repaired:

  1. ``scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`` called
     ``pre_flight_check.py --skip-anchor-guard`` — a flag the tool does NOT
     define, so argparse exited 2 and no gate ever ran.
  2. The same wrapper grepped the pre-flight's stdout for the literal
     ``62 PASS`` (a stale Wf-328-SYN baseline) — and did it through
     ``tail -5``, while the report's ``summary`` block sits at the TOP of the
     JSON, so the target could never be seen even if the tool had run.
  3. The same wrapper grepped the per-wave checklist's stdout for
     ``'"status": "PASS"\|PASS'``, but the tool itself prints NOTHING on
     stdout: its verdict is in the JSON it writes via ``--out``.  (MEASURED
     2026-11-02: the only thing on that stdout is the molmetal package's
     import banner — "[molmetal] applied env vars: ..." — 88 bytes.)
  4. The same wrapper called ``wave_budget_tracker.py
     --cfm-yuelbond-not-required`` — a flag the tool does NOT define (only
     ``--cfm-yuelbond-required`` exists) — and grepped the tool's stdout for
     ``PASS|OK|STATUS``, which that tool also never prints.

  5. ``scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh`` gated a phase on
     ``grep -q "phase complete\|TEST.*Pearson\|VAL.*Pearson"`` — and NONE of
     those three markers has a producer anywhere in the repo
     (``train_chemprop_v2_v3.py`` contains ZERO case-insensitive "pearson"
     occurrences), while the trainer's exit code — the authoritative signal —
     was thrown away by ``wait ... || true``.

  6. The same bindingdb wrapper gated on ``per_wave_checklist.py``'s EXIT
     CODE — which is always 0 for a checklist FAILURE (MEASURED: the tool
     exits 0 while its own JSON records ``preflight_fail_count = 1`` and a P0
     blocker with ``status = FAIL``), wrote the checklist to the tool's
     SHARED default path, and read the tool's STALE default pre-flight
     report instead of its own.

  7. The same bindingdb wrapper captured the pre-flight report with
     ``> pre_flight.json 2>&1``, so the "JSON" file held the tool's stderr
     log lines interleaved with the payload — ``json.load`` on it raises
     "Expecting value: line 1 column 2".

THE TESTING METHOD (and its one deliberate blind spot)
------------------------------------------------------
The wrappers preserve the superseded rendering VERBATIM in a comment block
(AGENTS.md §5.5 rule #6), so the dead flag STRINGS are still on disk.  A
substring test would therefore be satisfied by the comment.  These tests
strip the full-line comments first and assert against the CODE, which is
what actually runs.

The blind spot this method does NOT cover: an inline trailing comment
(``cmd --flag # note``) is not stripped, so a flag mentioned only in such a
comment would still be seen as "active".  Every occurrence in the two
repaired wrappers is a full-line comment, so the assertion is exact for them;
a future inline-comment occurrence would need the pattern re-checked by
hand.

CPU-only: no GPU, no torch, no trainer launch.  The only subprocesses are
``bash <wrapper> --dry-run`` (which the wrappers themselves make
GPU-free) and the pure-python checker.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from molmetal.tests.wfa269_report_fixture import stamp_live_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
CHECKER = REPO_ROOT / "molmetal" / "scripts" / "wf_launch_gate_verdict.py"

P5_WRAPPER = SCRIPTS / "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh"
BINDINGDB_WRAPPER = SCRIPTS / "launch_wf310e_r20_p4_bindingdb_1.07M.sh"
CHEMPROP_TRAINER = REPO_ROOT / "molmetal" / "scripts" / "train_chemprop_v2_v3.py"

P5_REPORTS = REPO_ROOT / "molmetal" / "reports" / "wf355a3_r20_p5_tmqm_plat"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _code_only(text: str) -> str:
    """Strip full-line shell comments.

    The wrappers keep the superseded rendering of a repaired gate in a
    comment block (AGENTS.md §5.5 rule #6), so the dead flag strings are
    still on disk.  The assertion has to be about the CODE that runs, not
    about the record of what used to run.
    """
    out = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def _run_checker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def _write_report(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _real_report(summary: dict, **extra) -> dict:
    """A synthetic pre-flight report shaped like a REAL producer report.

    Wf-A269 wave-16/17 (2026-11-03) — ADDITIVE.  The shared checker became
    FAIL-CLOSED: it refuses a report with no ``last_refresh`` stamp, no
    ``provenance`` block, a foreign commit SHA, a foreign gate-source hash,
    or a payload hash that does not match the report's own content.  Every
    one of those refusals is correct, but a fixture that omitted the fields
    would be refused for a reason unrelated to what these tests are about,
    so the fixture now carries what the producer would really have written
    (see ``molmetal/tests/wfa269_report_fixture.py``).

    SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    the payloads below used to be bare ``{"summary": {...}}`` dicts, and the
    bare-dict form is still what the assertions below are ABOUT — the
    summary block alone is what the verdict turns on.

    2026-11-04 (Wf-A269 dead-gates sweep) — ADDITIVE.  The preflight arm of
    the shared checker now also enforces the P0 BLOCKER rule that its sibling
    arm (``checklist``) has always enforced: a P0 gate that is not PASS must
    not authorise a launch.  To read that, the consumer needs the report's
    ``gates`` rows, and a real producer report always carries them — the
    producer writes ``GateResult.to_dict()`` = name / priority / pipeline /
    status / evidence / wall_sec for EVERY gate.  The bare-summary form the
    fixture used above is not a report the producer could have written, so
    the fixture now carries the six P0 blockers the ``all`` and ``launch``
    pipelines really register (MEASURED 2026-11-04 via
    ``build_gates("all")`` / ``build_gates("launch")``), all PASS.  The
    consumer checks the P0 rows a report CARRIES, not the set it owes, so
    this fixture does not need to track the pipeline.
    """
    payload: dict = {
        "summary": summary,
        "gates": [
            {"name": name, "priority": "P0", "pipeline": "all", "status": "PASS"}
            for name in (
                "p0_config_schema_verify",
                "p0_csv_hash_verify",
                "p0_scaffold_zero_leak",
                "p0_metalcytotox_v1_known_leaky",
                "p0_nci60_split_no_leak",
                "p0_gate_inventory_audit",
            )
        ],
    }
    payload.update(extra)
    return stamp_live_provenance(payload)


# ---------------------------------------------------------------------------
# Test 1 — the repaired wrappers carry no dead flag in their CODE
# ---------------------------------------------------------------------------


def test_p5_wrapper_has_no_undefined_tool_flag_in_its_code() -> None:
    """No flag the tool does not define may reach a tool from the wrapper.

    MEASURED before the repair: `pre_flight_check.py --skip-anchor-guard`
    exits 2 with "unrecognized arguments", and
    `wave_budget_tracker.py --cfm-yuelbond-not-required` exits 1 with the
    same message.  Both calls therefore measured nothing.
    """
    assert P5_WRAPPER.is_file(), f"wrapper missing: {P5_WRAPPER}"
    code = _code_only(P5_WRAPPER.read_text())

    for dead_flag in (
        "--skip-anchor-guard",
        "--cfm-yuelbond-not-required",
    ):
        assert dead_flag not in code, (
            f"{P5_WRAPPER.name}: the ACTIVE code still passes {dead_flag} to a "
            f"tool that does not define it — the gate would exit on argparse "
            f"and measure nothing"
        )

    # The tools' real flags must be present instead.
    assert "--pipeline all" in code, "the pre-flight must still be invoked"
    assert "--json" in code, "the pre-flight must write a JSON report to read"
    assert "--report-json" in code, (
        "the wave budget tracker must write a JSON report to read"
    )
    assert "--out" in code, (
        "the per-wave checklist must write its JSON report to read "
        "(it prints nothing on stdout)"
    )


def test_p5_wrapper_reads_the_gate_verdicts_from_json_not_stdout() -> None:
    """The gate verdict must come from a report, never from a stdout grep.

    A stdout grep against a tool that prints nothing is the signature of the
    dead gate: the success branch is unreachable and the failure branch is
    the only one that can ever run.
    """
    code = _code_only(P5_WRAPPER.read_text())
    assert "wf_launch_gate_verdict.py" in code, (
        "the wrapper must read its gate verdicts through the shared checker "
        "(molmetal/scripts/wf_launch_gate_verdict.py)"
    )
    for dead_grep in (
        'grep -q \'"status": "PASS"\\|PASS\'',
        'grep -q "62 PASS"',
        'grep -q "PASS\\|OK\\|STATUS"',
    ):
        assert dead_grep not in code, (
            f"{P5_WRAPPER.name}: the ACTIVE code still greps stdout for "
            f"{dead_grep!r} — no producer prints that"
        )


def test_bindingdb_wrapper_gates_on_a_marker_the_trainer_really_prints() -> None:
    """Every marker the abort gate greps for must have a producer.

    MEASURED before the repair: "phase complete" has NO producer repo-wide
    and `train_chemprop_v2_v3.py` has ZERO case-insensitive "pearson"
    occurrences, so the success branch was unreachable and every phase died
    with "aborted".
    """
    code = _code_only(BINDINGDB_WRAPPER.read_text())
    marker = "=== Wf-200 FINAL (JSON) ==="
    assert marker in code, (
        f"{BINDINGDB_WRAPPER.name}: the abort gate must check a marker the "
        f"trainer actually prints"
    )
    assert marker in CHEMPROP_TRAINER.read_text(), (
        f"the marker {marker!r} has no producer in {CHEMPROP_TRAINER.name} — "
        f"the gate would be dead again"
    )
    # The assertion is about GREPS, not about the word: the wrapper
    # legitimately DECLARES a Pearson threshold ("expected_pearson_r_gate")
    # and LOGS it; what it must not do is grep a log for a Pearson marker
    # that no producer emits.
    for line in code.splitlines():
        if "grep" not in line:
            continue
        for dead_marker in ("phase complete", "Pearson"):
            assert dead_marker not in line, (
                f"{BINDINGDB_WRAPPER.name}: the ACTIVE code greps for "
                f"{dead_marker!r}, which no producer emits: {line.strip()!r}"
            )

    # The trainer's exit code is the authoritative signal and must not be
    # discarded any more.
    assert "|| train_rc=$?" in code, (
        "the wrapper must capture the trainer's exit code instead of "
        "discarding it with `wait ... || true`"
    )


def test_bindingdb_checklist_gate_reads_a_verdict_not_an_exit_code() -> None:
    """The per-wave checklist's EXIT CODE cannot carry its verdict.

    MEASURED (2026-11-02): run against a synthetic pre-flight report with
    one FAIL gate, `per_wave_checklist.py` exits 0 while its own JSON
    records `preflight_fail_count = 1` and a P0 blocker with
    `status = FAIL`.  The wrapper therefore has to read the report.
    """
    code = _code_only(BINDINGDB_WRAPPER.read_text())
    assert 'run_gate_check checklist' in code, (
        "the checklist gate must read its verdict from the report through the "
        "shared checker"
    )
    # The checklist must read THIS launch's own pre-flight report and write
    # to THIS launch's own report path — not the tool's shared defaults.
    assert '--preflight "${REPORTS_DIR}/pre_flight.json"' in code, (
        "the checklist must read THIS launch's pre-flight report, not the "
        "tool's stale DEFAULT one"
    )
    assert '--out "${REPORTS_DIR}/per_wave.json"' in code, (
        "the checklist must write to THIS launch's report path, not the "
        "tool's SHARED default path"
    )


def test_bindingdb_preflight_gate_writes_real_json() -> None:
    """The pre-flight report must be a machine-readable JSON file.

    MEASURED before the repair: the report was captured with
    `> pre_flight.json 2>&1`, so the file held the tool's STDERR log lines
    interleaved with the JSON payload — `json.load` on such a file raises
    "Expecting value: line 1 column 2".
    """
    code = _code_only(BINDINGDB_WRAPPER.read_text())
    assert '--json "${REPORTS_DIR}/pre_flight.json"' in code, (
        "the pre-flight must write its report to a real JSON file "
        "(`--json <path>`); a bare `--json` writes NO file"
    )
    assert '> "${REPORTS_DIR}/pre_flight.json"' not in code, (
        "the pre-flight report must not be captured by redirecting stdout+"
        "stderr into the .json path — that produces a non-JSON file"
    )
    assert 'run_gate_check preflight' in code, (
        "the pre-flight gate must read its verdict from the report through "
        "the shared checker"
    )


def test_bindingdb_budget_gate_reads_the_tracker_report() -> None:
    """The wave budget tracker's exit code is always 0 (MEASURED).

    Its verdict lives in the JSON it writes via --report-json, so a gate
    that only looks at the exit code cannot fail.
    """
    code = _code_only(BINDINGDB_WRAPPER.read_text())
    assert 'run_gate_check budget' in code, (
        "the budget gate must read the tracker's verdict from its report"
    )


# ---------------------------------------------------------------------------
# Test 2 — the shared checker's behaviour (unit level)
# ---------------------------------------------------------------------------


def test_checker_file_exists_and_is_importable() -> None:
    assert CHECKER.is_file(), f"the shared gate checker is missing: {CHECKER}"
    proc = _run_checker("--help")
    assert proc.returncode == 0, f"--help failed: {proc.stderr[-300:]!r}"
    for kind in ("preflight", "checklist", "budget"):
        assert kind in proc.stdout, f"the checker has no '{kind}' sub-command"


def test_checker_preflight_verdicts(tmp_path: Path) -> None:
    """FAIL on a FAIL gate, FAIL on a TIMEOUT, PASS otherwise."""
    clean = _write_report(
        tmp_path,
        "pf_clean.json",
        _real_report({"pass": 114, "fail": 0, "skip": 12, "timeout": 0, "total": 126}),
    )
    proc = _run_checker("preflight", "--report", str(clean))
    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.split()[0] == "PASS"

    failed = _write_report(
        tmp_path,
        "pf_fail.json",
        _real_report({"pass": 114, "fail": 2, "skip": 12, "timeout": 0, "total": 128}),
    )
    proc = _run_checker("preflight", "--report", str(failed))
    assert proc.returncode == 1, proc.stdout
    assert proc.stdout.split()[0] == "FAIL"

    timed_out = _write_report(
        tmp_path,
        "pf_timeout.json",
        _real_report({"pass": 114, "fail": 0, "skip": 12, "timeout": 1, "total": 127}),
    )
    proc = _run_checker("preflight", "--report", str(timed_out))
    assert proc.returncode == 1, (
        "a TIMEOUT gate did not obtain its measurement, so it must not "
        f"count as a PASS: {proc.stdout!r}"
    )


def test_checker_checklist_verdicts(tmp_path: Path) -> None:
    """A P0 blocker that did not PASS is not a PASS."""
    ok_report = _write_report(
        tmp_path,
        "cl_ok.json",
        {
            "error": "",
            "preflight_pass_count": 96,
            "preflight_fail_count": 0,
            "p0_blocker_gates": [
                {"gate_name": "g1", "status": "PASS", "evidence": "x"},
                {"gate_name": "g2", "status": "PASS", "evidence": "x"},
            ],
        },
    )
    proc = _run_checker("checklist", "--report", str(ok_report))
    assert proc.returncode == 0, proc.stdout

    bad_blocker = _write_report(
        tmp_path,
        "cl_bad.json",
        {
            "error": "",
            "preflight_pass_count": 96,
            "preflight_fail_count": 0,
            "p0_blocker_gates": [
                {"gate_name": "g1", "status": "PASS", "evidence": "x"},
                {"gate_name": "g2", "status": "SKIP", "evidence": "x"},
            ],
        },
    )
    proc = _run_checker("checklist", "--report", str(bad_blocker))
    assert proc.returncode == 1, (
        "a P0 blocker gate that was SKIPPED was never verified, so it must "
        f"not count as a PASS: {proc.stdout!r}"
    )

    dirty_preflight = _write_report(
        tmp_path,
        "cl_dirty.json",
        {
            "error": "",
            "preflight_pass_count": 94,
            "preflight_fail_count": 2,
            "p0_blocker_gates": [{"gate_name": "g1", "status": "PASS", "evidence": "x"}],
        },
    )
    proc = _run_checker("checklist", "--report", str(dirty_preflight))
    assert proc.returncode == 1, proc.stdout

    errored = _write_report(
        tmp_path, "cl_err.json", {"error": "preflight file not found: x.json"}
    )
    proc = _run_checker("checklist", "--report", str(errored))
    assert proc.returncode == 1, proc.stdout


def test_checker_budget_verdicts(tmp_path: Path) -> None:
    """A wave that does not fit the session cap must FAIL."""
    fits = _write_report(
        tmp_path,
        "bg_fit.json",
        {
            "remaining_hr": 7.0,
            "deferred_waves": [],
            "honest_framing": "10.0h budget sufficient; 7.0h buffer",
        },
    )
    proc = _run_checker("budget", "--report", str(fits), "--wave", "tmqm_plat")
    assert proc.returncode == 0, proc.stdout

    over = _write_report(
        tmp_path,
        "bg_over.json",
        {
            "remaining_hr": -2.0,
            "deferred_waves": [],
            "honest_framing": "10.0h budget exceeded by 2.0h",
        },
    )
    proc = _run_checker("budget", "--report", str(over), "--wave", "tmqm_plat")
    assert proc.returncode == 1, proc.stdout

    deferred = _write_report(
        tmp_path,
        "bg_def.json",
        {
            "remaining_hr": 5.0,
            "deferred_waves": ["tmqm_plat"],
            "honest_framing": "wave deferred",
        },
    )
    proc = _run_checker("budget", "--report", str(deferred), "--wave", "tmqm_plat")
    assert proc.returncode == 1, proc.stdout


def test_checker_reports_a_missing_report_as_skip(tmp_path: Path) -> None:
    proc = _run_checker("preflight", "--report", str(tmp_path / "nope.json"))
    assert proc.returncode == 2, proc.stdout
    assert proc.stdout.split()[0] == "SKIP"


# ---------------------------------------------------------------------------
# Test 3 — the repaired p5 wrapper actually distinguishes PASS from FAIL
# ---------------------------------------------------------------------------


def _p5_constant(text: str, name: str) -> int:
    m = re.search(rf"^{name}=(\d+)", text, re.M)
    assert m, f"the wrapper no longer declares {name}"
    return int(m.group(1))


def _p5_dry_run(extra_args: list[str]) -> None:
    env = {**os.environ, "USER_OK": "1", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        ["bash", str(P5_WRAPPER), "--dry-run", *extra_args],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"the p5 dry-run {extra_args} failed (rc={proc.returncode}): "
        f"{proc.stderr[-500:]!r}"
    )


def test_p5_dry_run_records_a_real_budget_verdict() -> None:
    """The budget gate must FAIL when the wave does not fit the session cap.

    This is the end-to-end proof that the repaired gate can fail: the p5
    full sweep projects WALL_BUDGET_HR against the SESSION_CAP_HR session
    cap, and the verdict has to follow that arithmetic — FAIL when the
    projection is above the cap, PASS when it is inside it.  Before the
    repair the gate always took its warn-and-continue branch and the plan
    said nothing at all about the budget.

    The record read is the MODE-SPECIFIC plan (plan.full_5seed.json), not the
    canonical plan.json: the canonical path is overwritten by whichever mode
    ran last, so reading it would make this test depend on test order.
    """
    wrapper_text = P5_WRAPPER.read_text()
    full_hr = _p5_constant(wrapper_text, "WALL_BUDGET_HR")
    cap_hr = _p5_constant(wrapper_text, "SESSION_CAP_HR")

    _p5_dry_run([])

    plan_path = P5_REPORTS / "plan.full_5seed.json"
    assert plan_path.is_file(), (
        f"the mode-specific plan record is missing: {plan_path} — a mode's "
        f"record must not be destroyable by another mode (AGENTS.md §5.5 "
        f"rule #6)"
    )
    plan = json.loads(plan_path.read_text())
    assert "preflight_verdict" in plan, (
        "the plan record must carry the MEASURED verdict of each pre-launch "
        "gate"
    )
    assert plan["preflight_verdict"] == "UNVERIFIED", (
        "a dry-run skips the live pre-flight, so its verdict must be "
        f"UNVERIFIED — never PASS (got {plan['preflight_verdict']!r})"
    )
    assert plan["checklist_verdict"] == "UNVERIFIED", (
        "a dry-run skips the per-wave checklist, so its verdict must be "
        f"UNVERIFIED — never PASS (got {plan['checklist_verdict']!r})"
    )
    expected = "FAIL" if full_hr > cap_hr else "PASS"
    assert plan["budget_verdict"] == expected, (
        f"the budget gate did not follow the arithmetic: the full sweep "
        f"projects {full_hr} hr against a {cap_hr} hr session cap, so the "
        f"verdict must be {expected} (got {plan['budget_verdict']!r})"
    )
    assert plan["session_cap_hr"] == cap_hr, (
        "the plan record must carry the session cap the verdict was measured "
        f"against (got {plan.get('session_cap_hr')!r}, expected {cap_hr})"
    )

    # The DRY_RUN.log must carry the same verdicts on its face.
    dry_log_path = P5_REPORTS / "DRY_RUN.full_5seed.log"
    assert dry_log_path.is_file(), f"the mode-specific DRY_RUN.log is missing: {dry_log_path}"
    dry_log = dry_log_path.read_text()
    assert "gate_verdict:" in dry_log, (
        "the DRY_RUN.log must carry the gate verdicts so the plan says what "
        "was and was not measured"
    )
    assert f"budget={expected}" in dry_log, (
        f"the DRY_RUN.log must record budget={expected}"
    )


def test_p5_smoke_dry_run_budget_verdict_follows_the_arithmetic() -> None:
    """The smoke sweep (3 hr) fits the 10 hr cap, so its verdict is PASS."""
    wrapper_text = P5_WRAPPER.read_text()
    smoke_hr = _p5_constant(wrapper_text, "WALL_BUDGET_SMOKE_HR")
    cap_hr = _p5_constant(wrapper_text, "SESSION_CAP_HR")

    _p5_dry_run(["--smoke-only"])

    plan_path = P5_REPORTS / "plan.smoke_2seed.json"
    assert plan_path.is_file(), (
        f"the mode-specific plan record is missing: {plan_path}"
    )
    plan = json.loads(plan_path.read_text())
    expected = "FAIL" if smoke_hr > cap_hr else "PASS"
    assert plan["budget_verdict"] == expected, (
        f"the smoke sweep projects {smoke_hr} hr against a {cap_hr} hr cap, "
        f"so the verdict must be {expected} (got {plan['budget_verdict']!r})"
    )


def test_p5_verdict_vocabulary_has_no_fourth_token() -> None:
    """A skipped gate says UNVERIFIED — never a fourth token.

    AGENTS.md §5.2: a gate that was SKIPPED says UNVERIFIED, never PASS.
    The repaired wrapper used to emit "SKIPPED" from its "tool is missing"
    branch while its dry-run branch emitted "UNVERIFIED" for the same
    condition, so the plan record's verdict field had four possible values
    and no consumer knew three of them.
    """
    code = _code_only(P5_WRAPPER.read_text())
    for token in ('PRE_FLIGHT_VERDICT="SKIPPED"', 'CHECKLIST_VERDICT="SKIPPED"'):
        assert token not in code, (
            f"the ACTIVE code still emits the fourth verdict token {token!r} "
            f"— a gate that was not verified says UNVERIFIED"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
