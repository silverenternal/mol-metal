"""Wf-363-A1 — Per-pair DRY_RUN.log schema validation for 5 launch wrappers.

Owner: Hugo Lee
Date: 2026-09-29
Parent: Wf-355 + Wf-356-A1 (5 NEW launch wrappers shipped with DRY_RUN.log
emission); Wf-363-A1 (per-pair DRY_RUN.log schema polish + validation).
Per AGENTS.md §5.2 (MEASURED-vs-PROJECTED honest framing).  The 10 hr session
cap is NOT from AGENTS.md §5.1 — that section is the standing GPU
authorization and states no wall cap.  The cap's source of truth is
molmetal/scripts/gpu_launch_aggregator.py::SESSION_CAP_HR (line 62; rationale
at lines 573-575 — "each launcher's remaining_hr is computed against the 10h
SESSION_CAP_HR (so a single launcher that exceeds 10h individually is flagged
OVER)").

SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
Per AGENTS.md §5.1 (kill_then_adapt) + §5.2 (MEASURED-vs-PROJECTED honest framing).

Background — Per Wf-355 + Wf-356-A1: 5 NEW launch wrappers were shipped with
DRY_RUN.log emission (each emits to ``${REPORTS_DIR}/<subset|pair>/DRY_RUN.log``
on ``--dry-run``).  The pre-Wf-363-A1 schema had only ``trainer`` / ``subset``
/ ``flags`` / ``gpu`` / ``budget`` / ``ts`` — and inconsistent across wrappers.

Wf-363-A1 polish — single shared helper ``scripts/_dry_run_log_helper.sh``
provides a uniform ``emit_dry_run_log`` function.  All 5 wrappers source the
helper and emit DRY_RUN.log with the canonical schema:

    DRY-RUN launcher for <title>
      wrapper: <wrapper basename>
      mode:    <mode label>
      trainer: <trainer path>
      pair:    <pair label>          # repeated per (subset, pair, or cell)
      ...
      flags:   <flags>
      gpu:     <device>
      budget:  <hours> hr
      budget_label: <MEASURED|PROJECTED>        # Wf-A269 wave-9
      wall_policy: <single_launch|whole_sweep_sequential>   # Wf-A269 wave-9
      per_pair_budget_hr: <hours> hr           # Wf-A269 wave-9 (whole sweep)
      user_ok: <REQUIRED|OPTIONAL_smoke|REQUIRED_STATUS_HOLD|OPEN|n/a>
      capability: <OK|BLOCKED|UNVERIFIED>      # Wf-A269 wave-9 (optional)
      ts:      <ISO-8601 timestamp>
    per_pair:
      <pair label>
      ...

The 5 wrappers are:

  1. scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh
       (Wf-355-A1; subset=pairs; R1+R3 modes)
  2. scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh
       (Wf-355-A2; per-seed; R1+R3 modes)
  3. scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
       (Wf-355-A3; (cell, seed) pairs; smoke_2seed+full_5seed modes)
  4. scripts/launch_wf131_p3_deferred_runs.sh
       (Wf-355-B1; (metal, seed) pairs; DEFERRED_7_runs mode)
  5. scripts/launch_wf146_deferred_runs.sh
       (Wf-355-B2; (metal, seed) pairs; R1+R3 modes)

Tests (5 total, all CPU-only):

  - test_per_wrapper_dry_run_log_schema         (per-wrapper schema check)
  - test_per_pair_enumeration_consistent        (pair lines == per_pair block)
  - test_budget_within_session_cap              (budget ≤ 10 hr, unless the
    producer DECLARES a whole-sweep policy and a per-pair budget that the
    test then verifies for internal consistency)
  - test_budget_carries_an_honest_label         (budget_label MEASURED|PROJECTED)
  - test_capability_verdict_is_declared_and_valid  (the capability field is
    a checked field, not a comment)

All tests use the shared helper ``emit_dry_run_log`` for golden output where
useful.  Wall < 5 s; no torch.cuda, no real launches.

Honest framing (AGENTS.md §5.2):
  - MEASURED: 5 wrappers SHIPPED per Wf-355 + Wf-356-A1 (per-pair emission).
  - MEASURED: existing 12 DRY_RUN.log files at Wf-363-A1 baseline (per find
    ``molmetal/reports -name DRY_RUN.log``).
  - PROJECTED: 5 wrappers → 5/5 NEW schema-conformant logs (target).
  - PROJECTED: per-pair enumeration preserved (pair: lines == per_pair: block).
  - PROJECTED: per-wrapper budget within 10 hr session cap, or declared as a
    whole sweep whose per-pair budget is inside the cap.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root + path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
TEST_SCRIPTS_DIR = ROOT / "molmetal" / "scripts"
REPORTS = ROOT / "molmetal" / "reports"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# The 5 launch wrappers (Wf-355 + Wf-356-A1 SHIPPED per Wf-363-A1)
# ---------------------------------------------------------------------------
WF363A1_WRAPPERS = (
    SCRIPTS / "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh",
    SCRIPTS / "launch_wf310e_r20_p4_bindingdb_1.07M.sh",
    SCRIPTS / "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh",
    SCRIPTS / "launch_wf131_p3_deferred_runs.sh",
    SCRIPTS / "launch_wf146_deferred_runs.sh",
)

# Required schema fields (Wf-363-A1 polish)
WF363A1_REQUIRED_FIELDS = (
    "wrapper:",
    "mode:",
    "trainer:",
    "flags:",
    "gpu:",
    "budget:",
    "user_ok:",
    "ts:",
)

# 10 hr session cap.  SOURCE OF TRUTH:
# molmetal/scripts/gpu_launch_aggregator.py::SESSION_CAP_HR (line 62; rationale
# at lines 573-575 — "each launcher's remaining_hr is computed against the 10h
# SESSION_CAP_HR (so a single launcher that exceeds 10h individually is flagged
# OVER)").  NOT AGENTS.md §5.1: that section is the standing GPU
# authorization and states no wall cap.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
# 10 hr session cap (per AGENTS.md §5.1 + Wf-294B kill_then_adapt)
SESSION_CAP_HR = 10.0

# User-OK gate statuses
VALID_USER_OK = {"REQUIRED", "OPTIONAL_smoke", "REQUIRED_STATUS_HOLD", "OPEN", "n/a"}

# Wf-A269 wave-9 (2026-11-02) - the MEASURED-vs-PROJECTED label that every
# budget must carry (AGENTS.md §5.2).  A DRY_RUN.log is emitted BEFORE the
# launch, so PROJECTED is the honest default.
VALID_BUDGET_LABELS = {"MEASURED", "PROJECTED"}

# Wf-A269 wave-9 (2026-11-02) - the argv-capability verdict.  A field that no
# test checks is a comment, so the wrapper that RUNS the gate must declare it.
VALID_CAPABILITY = {"OK", "BLOCKED", "UNVERIFIED"}
CAPABILITY_DECLARING_WRAPPERS = (
    "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh",
)


# ---------------------------------------------------------------------------
# Helper: invoke the wrapper with --dry-run and capture the DRY_RUN.log path
# ---------------------------------------------------------------------------
def _invoke_wrapper(wrapper: Path, *args: str, env_extra: dict | None = None) -> Path:
    """Invoke the wrapper with --dry-run. Returns the subprocess result.

    Note: USER_OK gate is enforced before any --dry-run work.
      - launch_wf131_p3_deferred_runs.sh gates on USER_OK=1
      - launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh gates on USER_OK=1
      - launch_wf146_deferred_runs.sh R3 gates on USER_OK=YES
      - launch_wf310e_r20_p4_bindingdb_1.07M.sh R3 gates on USER_OK=YES
      - launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh R3 gates on USER_OK=YES
    Per-wrapper gating is handled in the test dispatch below.
    """
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    cmd = ["bash", str(wrapper), "--dry-run", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    return proc


# Per-wrapper USER_OK gating (Wf-363-A1)
WF363A1_USER_OK = {
    "launch_wf131_p3_deferred_runs.sh": "1",
    "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh": "1",
    "launch_wf146_deferred_runs.sh": "YES",
    "launch_wf310e_r20_p4_bindingdb_1.07M.sh": "YES",
    "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh": "YES",
}

# Wf-A269 wave-9 (2026-11-02) — one invocation PER LISTED RECORD.
#
# The schema test used to invoke each wrapper in ONE representative mode and
# then assert the schema of EVERY record that wrapper lists in
# WF363A1_LOG_PATHS — including records that invocation never regenerated.
# A leftover from an earlier run then decided the outcome: the bindingdb R3
# record (2026-09-29) carried no `budget_label`, so the schema test failed
# on a file that the CURRENT wrapper had never been asked to write, and the
# result depended on which other test had happened to run first (the drugood
# R3 record was fresh only because an unrelated test had just regenerated
# it).  That is the stale-record class this wave repairs: a record that
# asserts something nobody measured NOW.
#
# Each tuple below is one `--dry-run` invocation; the set of invocations for
# a wrapper regenerates every record that wrapper lists.
WF363A1_INVOCATIONS = {
    # canonical log only — the default (7 DEFERRED runs) mode writes it
    "launch_wf131_p3_deferred_runs.sh": ((),),
    # r1/DRY_RUN.log + r3/DRY_RUN.log
    "launch_wf146_deferred_runs.sh": (("--mode", "R1"), ("--mode", "R3")),
    # DRY_RUN.log (canonical) + DRY_RUN.full_5seed.log + DRY_RUN.smoke_2seed.log
    "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh": ((), ("--smoke-only",)),
    # r1/DRY_RUN.log + r3/DRY_RUN.log
    "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh": (
        ("--mode", "R1"),
        ("--mode", "R3"),
    ),
    # r1/DRY_RUN.log + r3/DRY_RUN.log
    "launch_wf310e_r20_p4_bindingdb_1.07M.sh": (
        ("--mode", "R1"),
        ("--mode", "R3"),
    ),
}


def _parse_dry_run_log(path: Path) -> dict:
    """Parse a DRY_RUN.log into a dict of field → list-of-values."""
    out: dict[str, list[str]] = {}
    if not path.exists():
        raise FileNotFoundError(f"DRY_RUN.log missing: {path}")
    for line in path.read_text().splitlines():
        line = line.rstrip()
        if (
            not line
            or line.startswith("DRY-RUN launcher")
            or line.startswith("per_pair")
        ):
            continue
        # Each indented line: "  <key>: <value>"
        m = re.match(r"^\s{2}([a-z_]+):\s*(.*)$", line)
        if m:
            key, value = m.group(1), m.group(2).strip()
            out.setdefault(key, []).append(value)
    return out


# Per-wrapper log candidates (Wf-363-A1)
WF363A1_LOG_PATHS = {
    "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh": (
        REPORTS / "wf310e_r20_p3_drugood_full_launch/r1/DRY_RUN.log",
        REPORTS / "wf310e_r20_p3_drugood_full_launch/r3/DRY_RUN.log",
    ),
    "launch_wf310e_r20_p4_bindingdb_1.07M.sh": (
        REPORTS / "wf355a2_r20_p4_bindingdb/r1/DRY_RUN.log",
        REPORTS / "wf355a2_r20_p4_bindingdb/r3/DRY_RUN.log",
    ),
    "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh": (
        REPORTS / "wf355a3_r20_p5_tmqm_plat/DRY_RUN.log",
        # Wf-A269 wave-9 (2026-11-02): the mode-specific durable record.  The
        # canonical path is single-path, so a smoke dry-run used to destroy
        # the full-sweep record; the mode-specific file survives.
        REPORTS / "wf355a3_r20_p5_tmqm_plat/DRY_RUN.full_5seed.log",
        REPORTS / "wf355a3_r20_p5_tmqm_plat/DRY_RUN.smoke_2seed.log",
    ),
    "launch_wf131_p3_deferred_runs.sh": (
        REPORTS / "wf131_p3_deferred_runs/DRY_RUN.log",
    ),
    "launch_wf146_deferred_runs.sh": (
        REPORTS / "wf146_chemprop_lomo/r1/DRY_RUN.log",
        REPORTS / "wf146_chemprop_lomo/r3/DRY_RUN.log",
    ),
}


def _gather_logs_for_wrapper(wrapper_name: str) -> list[Path]:
    """Return DRY_RUN.log paths expected for a specific wrapper."""
    return [p for p in WF363A1_LOG_PATHS.get(wrapper_name, ()) if p.exists()]


@pytest.fixture(scope="module")
def fresh_dry_run_records() -> float:
    """Regenerate every DRY_RUN record the assertions below read.

    Wf-A269 wave-9 (2026-11-02): these tests used to read whatever happened
    to be on disk.  A leftover from an earlier run therefore decided the
    outcome, and the result depended on which other test had run first — the
    bindingdb R3 record (2026-09-29) carried no `budget_label`, so the
    honesty assertion failed on a file the CURRENT wrapper had never been
    asked to write, while the drugood R3 record happened to be fresh only
    because an unrelated test had just regenerated it.

    This fixture regenerates every listed record and returns the timestamp
    the regeneration started at.  Each test then asserts that the records it
    reads are newer than that timestamp, so no stale artefact can decide a
    live assertion.

    Returns:
        The `time.time()` at which the regeneration started.
    """
    t0 = time.time()
    for wrapper in WF363A1_WRAPPERS:
        user_ok_value = WF363A1_USER_OK.get(wrapper.name, "YES")
        env_extra = {"USER_OK": user_ok_value}
        invocations = WF363A1_INVOCATIONS.get(wrapper.name, (("--mode", "R1"),))
        for args in invocations:
            proc = _invoke_wrapper(wrapper, *args, env_extra=env_extra)
            assert proc.returncode == 0, (
                f"{wrapper.name} --dry-run {' '.join(args)} failed "
                f"(rc={proc.returncode}); stderr tail: {proc.stderr[-400:]}"
            )
    return t0


def _assert_records_are_fresh(
    wrapper_name: str, logs: list[Path], t0: float
) -> None:
    """Every asserted record must have been written by this test session."""
    for log in logs:
        assert log.stat().st_mtime >= t0 - 2, (
            f"{wrapper_name} → {log.name}: not regenerated by this run "
            f"(mtime={log.stat().st_mtime} < t0={t0}); a stale record must "
            f"not be able to decide a live assertion"
        )


# ---------------------------------------------------------------------------
# Test #1 — Per-wrapper DRY_RUN.log schema validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wrapper", WF363A1_WRAPPERS, ids=lambda p: p.name)
def test_per_wrapper_dry_run_log_schema(
    wrapper: Path, fresh_dry_run_records: float
) -> None:
    """Each wrapper MUST emit DRY_RUN.log with the canonical schema.

    Required fields (Wf-363-A1):
      wrapper: / mode: / trainer: / flags: / gpu: / budget: / user_ok: / ts:
    """
    assert wrapper.exists(), f"wrapper missing: {wrapper}"

    # Wf-A269 wave-9 (2026-11-02): the `fresh_dry_run_records` fixture has
    # already regenerated EVERY record this wrapper lists, so the schema
    # assertion below can never be satisfied by a leftover from an earlier
    # run.
    t0 = fresh_dry_run_records

    # Only check logs specific to THIS wrapper (avoid cross-wrapper noise)
    logs = _gather_logs_for_wrapper(wrapper.name)
    assert logs, f"{wrapper.name} produced no DRY_RUN.log files"

    # Every asserted record must have been written by THIS test run — a
    # stale record must not be able to decide the schema assertion.
    _assert_records_are_fresh(wrapper.name, logs, t0)

    # Each emitted log MUST contain the required schema fields
    for log in logs:
        parsed = _parse_dry_run_log(log)
        for field in WF363A1_REQUIRED_FIELDS:
            key = field.rstrip(":")
            assert key in parsed, (
                f"{wrapper.name} → {log.name}: missing field '{key}'; "
                f"got keys: {list(parsed.keys())}"
            )
        # user_ok MUST be one of the valid statuses
        user_ok = parsed["user_ok"][-1]
        assert user_ok in VALID_USER_OK, (
            f"{wrapper.name} → {log.name}: invalid user_ok '{user_ok}'; "
            f"valid: {sorted(VALID_USER_OK)}"
        )


# ---------------------------------------------------------------------------
# Test #2 — Per-pair enumeration consistency
# ---------------------------------------------------------------------------
def test_per_pair_enumeration_consistent(fresh_dry_run_records: float) -> None:
    """``pair:`` lines MUST equal ``per_pair:`` block entries (consistent enumeration).

    A wrapper that emits N ``pair:`` lines must also emit N entries under the
    ``per_pair:`` block (downstream tooling counts both).
    """
    # Iterate per-wrapper so each wrapper's logs are validated independently
    for wrapper_name in WF363A1_LOG_PATHS:
        logs = _gather_logs_for_wrapper(wrapper_name)
        if not logs:
            continue  # skip if not produced yet (other tests handle schema)
        # Wf-A269 wave-9 (2026-11-02): only assert on records this session
        # regenerated; a stale artefact must not decide a live assertion.
        _assert_records_are_fresh(wrapper_name, logs, fresh_dry_run_records)
        for log in logs:
            text = log.read_text()
            pair_lines = [
                line.split("pair:", 1)[1].strip()
                for line in text.splitlines()
                if line.strip().startswith("pair:")
            ]
            # Find the per_pair block
            per_pair_block_lines: list[str] = []
            in_block = False
            for line in text.splitlines():
                if line.strip().startswith("per_pair:"):
                    in_block = True
                    continue
                if in_block:
                    if line.startswith("  ") and line.strip():
                        per_pair_block_lines.append(line.strip())
                    elif line.strip() == "":
                        continue
                    else:
                        in_block = False
            # If pair lines exist, they must equal per_pair block entries.
            # If no pair lines (subset-only log), skip consistency check.
            if pair_lines:
                assert len(pair_lines) == len(per_pair_block_lines), (
                    f"{wrapper_name} → {log.name}: pair: lines ({len(pair_lines)}) "
                    f"!= per_pair: block entries ({len(per_pair_block_lines)}); "
                    f"pair_lines={pair_lines[:5]}... per_pair={per_pair_block_lines[:5]}..."
                )


# ---------------------------------------------------------------------------
# Test #3 — Budget estimate within 10 hr session cap
# ---------------------------------------------------------------------------
def test_budget_within_session_cap(fresh_dry_run_records: float) -> None:
    """Per-wrapper ``budget:`` MUST be ≤ 10 hr session cap, unless the
    PRODUCER declares a whole-sweep policy and a per-pair budget.

    Per molmetal/scripts/gpu_launch_aggregator.py::SESSION_CAP_HR (line 62;
    rationale at lines 573-575): no per-launch wall > 10 hr without explicit
    user override.  The cap bounds a single LAUNCH.  A wrapper that runs a
    whole sweep of N SEQUENTIAL launches legitimately has a sweep total above
    the cap while each launch stays inside it.

    (AGENTS.md §5.1 is the standing GPU authorization and states no wall cap;
    AGENTS.md §5.2 is the honest-framing rule, applied in the "Honest
    framing" block of this module's docstring.)

    SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    Per AGENTS.md §5.1 + Wf-294B kill_then_adapt: no per-launch wall > 10 hr
    without explicit user override.  The cap bounds a single LAUNCH.  A
    wrapper that runs a whole sweep of N SEQUENTIAL launches legitimately has
    a sweep total above the cap while each launch stays inside it.

    Wf-A269 wave-9 (2026-11-02) - REPAIR (2nd revision).  The first
    implementation of the whole-sweep exception was REFUTED by adversarial
    review for two independent reasons, both of which are structural:

      (a) it divided the budget ITSELF (``budget_hr / n_pairs``), so the
          consumer re-derived a quantity that only the producer knows;
      (b) it gated the exception on the log's OWN ``user_ok:`` field - a
          free-text string the producer wrote about itself, i.e. a
          SELF-SIGNED exemption.  "user_ok: REQUIRED" says nothing about wall
          time.
    It also contradicted this file's own docstring, which promised "≤ 10 hr
    **or marked DEFERRED**" while NO defer-check existed anywhere.

    The exception is now implemented as a PRODUCER DECLARATION that the
    consumer VERIFIES:

      1. the producer declares ``wall_policy: whole_sweep_sequential``;
      2. the producer declares ``per_pair_budget_hr`` - the budget of ONE
         launch, which the producer alone can compute;
      3. the declared per-pair budget must be within the session cap;
      4. the declaration must be INTERNALLY CONSISTENT: the declared
         per-pair budget, multiplied by the declared pair count, must
         reconstruct the declared sweep total (1 % relative tolerance).

    Rule (4) is what makes this a check rather than a rubber stamp: a
    producer that misstates the total, the pair count, or the per-pair
    budget fails.  No division happens here, and no self-signed label is
    trusted.  A producer that wants an above-cap sweep must state its
    arithmetic and stand behind it.

    MEASURED for the one wrapper that qualifies:
    launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh declares
    ``budget: 12 hr`` / ``wall_policy: whole_sweep_sequential`` /
    ``per_pair_budget_hr: 0.6000 hr`` over 20 (cell, seed) pairs, and
    launches those 20 tuples SEQUENTIALLY, one trainer process per tuple.

    SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
                assert budget_hr <= SESSION_CAP_HR, (
                    f"{wrapper_name} → {log.name}: budget {budget_hr} hr exceeds "
                    f"session cap {SESSION_CAP_HR} hr; either split or mark DEFERRED"
                )
    """
    for wrapper_name in WF363A1_LOG_PATHS:
        logs = _gather_logs_for_wrapper(wrapper_name)
        if not logs:
            continue
        # Wf-A269 wave-9 (2026-11-02): only assert on records this session
        # regenerated; a stale artefact must not decide a live assertion.
        _assert_records_are_fresh(wrapper_name, logs, fresh_dry_run_records)
        for log in logs:
            parsed = _parse_dry_run_log(log)
            n_pairs = len(parsed.get("pair", []))
            policy = (parsed.get("wall_policy") or [None])[-1]
            # Each wrapper may emit multiple budget: lines (one per phase)
            for budget_str in parsed["budget"]:
                m = re.match(r"([\d.]+)\s*hr", budget_str)
                assert m, f"{log.name}: unparseable budget '{budget_str}'"
                budget_hr = float(m.group(1))
                if budget_hr <= SESSION_CAP_HR:
                    continue
                # Above the cap: the ONLY accepted exemption is the
                # producer's whole-sweep declaration.
                assert policy == "whole_sweep_sequential", (
                    f"{wrapper_name} → {log.name}: budget {budget_hr} hr exceeds "
                    f"session cap {SESSION_CAP_HR} hr and the log declares no "
                    f"wall_policy: whole_sweep_sequential (declared: {policy!r}); "
                    f"either split the sweep, or declare the whole-sweep policy "
                    f"and its per-pair budget"
                )
                per_pair_raw = (parsed.get("per_pair_budget_hr") or [None])[-1]
                assert per_pair_raw is not None, (
                    f"{wrapper_name} → {log.name}: wall_policy is "
                    f"whole_sweep_sequential but no per_pair_budget_hr was declared"
                )
                m2 = re.match(r"([\d.]+)\s*hr", per_pair_raw)
                assert m2, (
                    f"{wrapper_name} → {log.name}: unparseable "
                    f"per_pair_budget_hr '{per_pair_raw}'"
                )
                per_pair_hr = float(m2.group(1))
                assert per_pair_hr <= SESSION_CAP_HR, (
                    f"{wrapper_name} → {log.name}: declared per-pair budget "
                    f"{per_pair_hr} hr exceeds the session cap "
                    f"{SESSION_CAP_HR} hr; the cap bounds ONE launch"
                )
                assert n_pairs >= 2, (
                    f"{wrapper_name} → {log.name}: wall_policy is "
                    f"whole_sweep_sequential but only {n_pairs} pair(s) were "
                    f"enumerated; that is a single launch, not a sweep"
                )
                # Internal consistency: the declared per-pair budget must
                # reconstruct the declared sweep total.
                reconstructed = per_pair_hr * n_pairs
                tol = 0.01 * max(budget_hr, 1.0)
                assert abs(reconstructed - budget_hr) <= tol, (
                    f"{wrapper_name} → {log.name}: the declaration is not "
                    f"internally consistent: per_pair_budget_hr {per_pair_hr} "
                    f"x {n_pairs} pairs = {reconstructed:.4f} hr, but the log "
                    f"declares budget {budget_hr} hr"
                )


# ---------------------------------------------------------------------------
# Test #4 — The budget carries a MEASURED/PROJECTED label
# ---------------------------------------------------------------------------
def test_budget_carries_an_honest_label(fresh_dry_run_records: float) -> None:
    """Every ``budget:`` MUST sit next to a ``budget_label:`` MEASURED|PROJECTED.

    A DRY_RUN.log is emitted BEFORE the launch, so its budget is a PROJECTION.
    Wf-A269 wave-9 (2026-11-02): the budget line carried no label at all, so
    a reader could not tell a projection from a measurement - which is the
    AGENTS.md §5.2 MEASURED-vs-PROJECTED rule applied to the schema itself.
    """
    seen = 0
    for wrapper_name in WF363A1_LOG_PATHS:
        logs = _gather_logs_for_wrapper(wrapper_name)
        # Wf-A269 wave-9 (2026-11-02): only assert on records this session
        # regenerated; a stale artefact must not decide a live assertion.
        _assert_records_are_fresh(wrapper_name, logs, fresh_dry_run_records)
        for log in logs:
            parsed = _parse_dry_run_log(log)
            labels = parsed.get("budget_label")
            assert labels, (
                f"{wrapper_name} → {log.name}: no budget_label; a budget "
                f"without a MEASURED/PROJECTED label violates AGENTS.md §5.2"
            )
            label = labels[-1]
            assert label in VALID_BUDGET_LABELS, (
                f"{wrapper_name} → {log.name}: invalid budget_label '{label}'; "
                f"valid: {sorted(VALID_BUDGET_LABELS)}"
            )
            seen += 1
    if not seen:
        pytest.skip("no DRY_RUN.log on disk to check (run the wrappers with --dry-run)")


# ---------------------------------------------------------------------------
# Test #5 — The capability verdict is declared AND valid where it is claimed
# ---------------------------------------------------------------------------
def test_capability_verdict_is_declared_and_valid(
    fresh_dry_run_records: float,
) -> None:
    """A ``capability:`` field MUST carry a valid verdict, and the wrappers
    that RUN a capability gate MUST declare it.

    Wf-A269 wave-9 (2026-11-02): the capability verdict (OK / BLOCKED /
    UNVERIFIED) was added to the log and to plan.json but NO test asserted it,
    so R2's honesty rested on an unverified field.  A field that nothing
    checks is a comment.
    """
    for wrapper_name in CAPABILITY_DECLARING_WRAPPERS:
        logs = _gather_logs_for_wrapper(wrapper_name)
        # Wf-A269 wave-9 (2026-11-02): only assert on records this session
        # regenerated; a stale artefact must not decide a live assertion.
        _assert_records_are_fresh(wrapper_name, logs, fresh_dry_run_records)
        for log in logs:
            parsed = _parse_dry_run_log(log)
            caps = parsed.get("capability")
            assert caps, (
                f"{wrapper_name} → {log.name}: this wrapper runs a capability "
                f"gate, so the log MUST declare capability: "
                f"{sorted(VALID_CAPABILITY)}"
            )
            cap = caps[-1]
            assert cap in VALID_CAPABILITY, (
                f"{wrapper_name} → {log.name}: invalid capability '{cap}'; "
                f"valid: {sorted(VALID_CAPABILITY)}"
            )
