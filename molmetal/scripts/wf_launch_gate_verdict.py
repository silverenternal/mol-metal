#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Turn a launcher's pre-launch gate REPORT into an honest PASS / FAIL verdict.

WHY (Wf-A269 wave-10, 2026-11-02).  The pre-launch gates of the R20 launch
wrappers were DEAD GATES: each one looked like a check but could not fail.

Three distinct failure modes were found, all in
``scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh``:

  1. **A flag the tool does not define.**  The pre-flight was invoked with
     ``--skip-anchor-guard`` and the wave budget tracker with
     ``--cfm-yuelbond-not-required``.  Neither flag exists, so argparse exited
     2 / 1 immediately, no measurement was taken, and the wrapper's
     warn-and-continue branch was taken every single time.
  2. **A stdout grep for a marker no producer prints.**  The per-wave
     checklist was grepped for ``'"status": "PASS"\\|PASS'`` on stdout, but the
     tool prints NOTHING on stdout -- its verdict lives in the JSON it writes
     via ``--out``.  The success branch was unreachable; the failure branch was
     the only reachable one, and it only warned.
  3. **A stale target.**  The pre-flight's success branch grepped the last 5
     stdout lines for the literal ``62 PASS`` -- a Wf-328-SYN baseline that no
     longer matches the gate count -- while the report's ``summary`` block
     sits at the TOP of the JSON, not in the tail.

Because every one of those tools exited INSTANTLY, the dry-run looked clean
and the wrapper looked ready to launch, although it had never run a
pre-flight at all.

This module is the ONE place that knows how to read the three report
formats, so every launcher can share it instead of re-implementing a grep.

It is deliberately a CONSUMER of a report, never a producer: it reads a file
that a tool already wrote and decides PASS / FAIL from that file's own
contents.  It never re-runs the tool and it never infers a verdict from the
absence of output.

Sub-commands
------------
``preflight --report <pre_flight.json>``
    PASS iff the report is FRESH, carries a usable ``summary`` block, and
    that block has ``fail == 0`` AND ``timeout == 0``.
    A TIMEOUT is a FAIL here on purpose (Wf-A135): a gate that timed out did
    not obtain its measurement, so "the gate passed" was never established.
    SKIP gates are reported in the detail line but do not fail the verdict:
    the repo runs 11-12 SKIPs by design and the pre-flight's own
    ``--strict`` (which turns SKIP into a non-zero exit) is opt-in.

    2026-11-04 (Wf-A269 dead-gates sweep) — the rendering above is
    SUPERSEDED; it is PRESERVED VERBATIM per AGENTS.md §5.5 rule #6.  It
    says NOTHING about the P0 blockers, and this module's OTHER arm
    (``checklist``) has always required every P0 gate to be PASS.  So a
    report whose P0 gate SKIPped was a FAIL in the checklist arm and a PASS
    in this one — and this is the arm the 15 launch wrappers read.  The
    preflight arm now carries the SAME P0 rule (see ``check_preflight``).

    The consumer is FAIL-CLOSED (Wf-A269 wave-16, 2026-11-03).  Two ways a
    report used to authorise a launch it had not earned:

    * **no ``summary``** — the old implementation read
      ``report.get("summary") or {}`` and counted FAIL/TIMEOUT out of it, so
      a truncated or wrong-schema report yielded ``fail=0, timeout=0`` and
      was reported PASS.  A gate that cannot fail is the defect (§6.269.17).
    * **no freshness** — the consumer never looked at the report's age, so
      a report from last week authorised today's launch.  The pre-flight now
      stamps every report with ``last_refresh`` (ISO-8601 UTC, written by
      the tool that measured the gates), and a report older than 24 h is
      STALE ⇒ FAIL — the AGENTS.md §4 rule that had been documented for
      many waves but never implemented.

``checklist --report <per_wave.json>``
    PASS iff the checklist recorded no ``error``, the pre-flight it read has
    ``preflight_fail_count == 0``, and every P0 BLOCKER gate has
    ``status == "PASS"``.  A P0 blocker that was SKIPPED is not a PASS: the
    blocker was never verified.

``budget --report <wave_budget.json> [--wave NAME]``
    PASS iff ``remaining_hr >= 0`` and (when ``--wave`` is given) that wave is
    not in ``deferred_waves``.  ``remaining_hr < 0`` means the wave does not
    fit the session cap (AGENTS.md §5.1 / Wf-294B kill_then_adapt), which is
    a real, actionable verdict -- the sweep has to be split.

Usage
-----
    python molmetal/scripts/wf_launch_gate_verdict.py preflight --report r/pre_flight.json
    python molmetal/scripts/wf_launch_gate_verdict.py checklist --report r/per_wave.json
    python molmetal/scripts/wf_launch_gate_verdict.py budget --report r/budget.json --wave tmqm_plat

Stdout invariant (AGENTS.md §4): the first token of stdout is ``PASS`` /
``FAIL`` / ``SKIP``; the rest of the line is the reason.  Exit 0=PASS,
1=FAIL, 2=SKIP (the report is missing or unreadable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_SKIP = "SKIP"

_EXIT = {VERDICT_PASS: 0, VERDICT_FAIL: 1, VERDICT_SKIP: 2}

#: The repo this consumer lives in — the anchor for every provenance check.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The freshness window.  AGENTS.md §4 "Stale flag rule": ``now -
#: last_refresh > 24 hr`` ⇒ STALE ⇒ auto-verify refuses PASS.
STALE_AFTER_SEC = 24 * 3600

#: A stamp further in the future than this is a clock problem or a forgery,
#: not a fresh report.  Small negative ages (a few seconds of skew) are
#: tolerated rather than failed.
FUTURE_TOLERANCE_SEC = 300


def canonical_payload_sha256(report: Mapping[str, Any]) -> str:
    """SHA-256 of a report's own CONTENT, excluding its ``provenance`` block.

    Wf-A269 wave-17 (2026-11-03).  This is the ONE canonical definition of
    the form, and it lives in the CONSUMER because the consumer is the light
    module: the producer imports it from here (see
    ``molmetal/scripts/pre_flight_check.py``), so the two sides cannot drift
    into hashing different bytes.  A drift between the two definitions would
    make every verdict FAIL, which is fail-closed but useless.

    Why the ``provenance`` block is excluded: the provenance block is
    metadata ABOUT the payload, and it cannot contain a hash of itself.  The
    fields it carries are each bound separately (the commit SHA against the
    live HEAD, the gate-source hash against the gate source on disk).

    WHAT THIS IS AND IS NOT — stated honestly (AGENTS.md §5.2).  It is a
    CONTENT-INTEGRITY binding, not a signature: it catches an edited field, a
    truncated write, a report copied over from another run, and a report
    whose bytes no longer describe the measurement it claims.  It does NOT
    prove authorship, because anyone who can edit the file can recompute the
    hash.  Authenticity is the commit-SHA + gate-source-hash pair's job, and
    even that is an integrity binding rather than a signature.

    The canonical form is ``sort_keys=True`` + ``separators=(",", ":")`` +
    ``ensure_ascii=False``, i.e. key order and whitespace are normalised
    away so the hash depends on the CONTENT and not on how the JSON happens
    to be laid out.  ``json`` round-trips Python floats via ``repr``, which
    is exact and stable, so a value written as ``1.0`` reads back as ``1.0``
    and not as ``1``.
    """
    payload = {k: v for k, v in report.items() if k != "provenance"}
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_report(path: Path) -> Optional[dict]:
    """Read a JSON report, or None when it cannot be read."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _live_head_sha() -> Optional[str]:
    """The current commit SHA of the repo this consumer lives in.

    None when git cannot be reached — reported as NOT MEASURED, never as a
    defect (AGENTS.md §5.2) and never as a silent default.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _sha256_of_file(path: Path) -> Optional[str]:
    """SHA-256 of ``path``, or None when it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _p0_blockers(report: Dict[str, Any]) -> Optional[List[Tuple[str, str]]]:
    """``(name, status)`` of every P0 blocker gate the report carries.

    Returns ``None`` when the report has no usable ``gates`` list — the
    caller then refuses the report, because a report without the gate rows
    cannot show that the blockers were verified.

    2026-11-04 (Wf-A269 dead-gates sweep).  This is the helper the
    preflight arm's new P0 rule uses; it reads the SAME field the producer
    writes (``GateResult.to_dict()``'s ``name`` / ``priority`` / ``status``)
    and the same shape the checklist arm reads (``gate_name`` / ``status``),
    so the two arms cannot drift apart on the question "which gates are the
    blockers?".

    A gate entry that is not a mapping is IGNORED rather than treated as a
    P0 gate: a malformed row is not evidence that a blocker passed, but it
    is also not a P0 gate the producer claimed to have measured, and the
    report's own ``summary`` is what counts the gates.  (The "no P0 gate at
    all" case is refused by the caller, fail-closed.)
    """
    gates = report.get("gates")
    if not isinstance(gates, list):
        return None
    out: List[Tuple[str, str]] = []
    for entry in gates:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("priority", "")).strip().upper() != "P0":
            continue
        name = str(entry.get("name", entry.get("gate_name", "?")))
        status = str(entry.get("status", "?")).strip().upper()
        out.append((name, status))
    return out


def _provenance_verdict(report: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """FAIL when the verdict was not earned for THIS tree.

    Wf-A269 wave-16 (2026-11-03).  The frontier rule this encodes: "a
    verdict is earned at a specific commit SHA and does not carry forward
    across a HEAD move", with "stale treated as missing" — a mismatch is a
    FAIL, never a warning and never a PASS.

    Two bindings are checked, and both are content-addressed rather than
    wall-clock-addressed:

    * **gate-source binding** — the report records the SHA-256 of the gate
      source that produced it, so a report cannot be read as the output of a
      DIFFERENT gate implementation (the Bazel "action cache over declared
      inputs" idea: hard-fail on a changed input, not on a clock).
    * **revision binding** — the report records the commit SHA it was
      earned at; a moved HEAD means the verdict does not carry forward.
    * **payload binding** — the report records the SHA-256 of its OWN
      content (Wf-A269 wave-17, 2026-11-03).  The two bindings above say
      WHICH code produced the report; they say nothing about whether the
      report still says what it said.  MEASURED before this check existed:
      editing ``summary.fail`` from 1 to 0 in a real report still yielded
      PASS, because neither the commit SHA nor the gate-source hash covers
      the report's body.  This is a content-integrity binding, not a
      signature (see ``canonical_payload_sha256``).

    Returns None when the report is bound to this tree, or a
    ``(verdict, reason)`` tuple when it is not.
    """
    prov = report.get("provenance")
    if not isinstance(prov, dict):
        return (
            VERDICT_FAIL,
            "the pre-flight report carries no 'provenance' block -- a verdict "
            "with no provenance cannot be shown to belong to this tree",
        )

    recorded_payload = prov.get("payload_sha256")
    if not recorded_payload:
        return (
            VERDICT_FAIL,
            "the pre-flight report carries no payload hash -- it predates "
            "the content binding, so its body cannot be checked against "
            "itself",
        )
    actual_payload = canonical_payload_sha256(report)
    if actual_payload != recorded_payload:
        return (
            VERDICT_FAIL,
            "the report's own content does not match the payload hash it "
            f"carries (recorded {str(recorded_payload)[:12]}, on disk "
            f"{actual_payload[:12]}) -- the file was altered after the tool "
            "wrote it",
        )

    rel = prov.get("gate_source")
    recorded = prov.get("gate_source_sha256")
    if rel and recorded:
        current = _sha256_of_file(REPO_ROOT / str(rel))
        if current is None:
            return (
                VERDICT_FAIL,
                f"the gate source {rel} named by the report cannot be read "
                f"from {REPO_ROOT} -- the report cannot be checked against the "
                f"gate that is on disk now",
            )
        if current != recorded:
            return (
                VERDICT_FAIL,
                f"the report was produced by a DIFFERENT version of the gate "
                f"source ({rel}: recorded sha256 {str(recorded)[:12]}, on disk "
                f"{current[:12]}) -- the verdict does not describe the gate "
                f"that would run now",
            )

    recorded_sha = prov.get("git_sha")
    live = _live_head_sha()
    if recorded_sha and live and recorded_sha != live:
        return (
            VERDICT_FAIL,
            f"the report was earned at commit {str(recorded_sha)[:12]} but HEAD "
            f"is {live[:12]} -- a verdict does not carry forward across a HEAD "
            f"move (stale is treated as missing); re-run pre_flight_check.py",
        )
    return None


def _freshness_verdict(report: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """FAIL when the report cannot prove it was measured recently.

    Returns ``None`` when the report is fresh, or a ``(verdict, reason)``
    tuple when it is not.

    The stamp is the report's OWN ``last_refresh`` field — an ISO-8601 UTC
    instant written by ``pre_flight_check.py`` at the moment it measured the
    gates.  It is deliberately NOT the file mtime: ``touch`` forges an mtime
    trivially, and the repo's own ``wf343c2_per_wave_check.py`` calls that
    "fabricated freshness".  A report with no stamp is not a fresh report;
    it is a report whose age cannot be established.
    """
    raw = report.get("last_refresh")
    if not raw:
        return (
            VERDICT_FAIL,
            "the pre-flight report carries no 'last_refresh' stamp, so its "
            "freshness cannot be verified -- a report without a stamp is "
            "not a fresh report",
        )
    try:
        stamp = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return (
            VERDICT_FAIL,
            f"the pre-flight report's 'last_refresh' stamp is not an "
            f"ISO-8601 timestamp: {raw!r}",
        )
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if age > STALE_AFTER_SEC:
        return (
            VERDICT_FAIL,
            f"the pre-flight report is STALE: age {age / 3600:.1f} h > "
            f"{STALE_AFTER_SEC / 3600:.0f} h (AGENTS.md §4 stale-flag rule) "
            f"-- re-run pre_flight_check.py to refresh it",
        )
    if age < -FUTURE_TOLERANCE_SEC:
        return (
            VERDICT_FAIL,
            f"the pre-flight report's 'last_refresh' stamp is "
            f"{-age / 3600:.1f} h in the FUTURE ({raw!r}) -- a clock problem "
            f"or a forged stamp, not a fresh report",
        )
    return None


def check_preflight(report: Dict[str, Any]) -> Tuple[str, str]:
    """PASS iff the report is FRESH, has a usable summary, no gate FAILED or
    TIMED OUT, and every P0 blocker gate PASSed.

    Wf-A269 wave-16 (2026-11-03): the previous implementation read
    ``summary = report.get("summary") or {}`` and counted FAIL / TIMEOUT out
    of it.  A report with NO summary therefore yielded ``fail=0,
    timeout=0`` and was reported PASS — i.e. a truncated or wrong-schema
    report authorised a launch.  A gate that cannot fail is the defect
    (AGENTS.md §6.269.17).  The consumer is now FAIL-CLOSED on both counts:
    an absent / non-dict ``summary`` is a FAIL, and so is a report whose
    ``last_refresh`` stamp is missing, unparseable, or older than 24 h.

    2026-11-04 (Wf-A269 dead-gates sweep) — THE P0 ARM.  The rendering
    superseded here is PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:

        "PASS iff the report is FRESH, has a usable summary, and no gate
         FAILED or TIMED OUT."

    i.e. the arm's PASS condition said NOTHING about the P0 blockers.  This
    module's OTHER arm (``check_checklist``) has always required every P0
    gate to be PASS ("P0 blocker gate(s) are not PASS"), so ONE report was
    a FAIL in the checklist arm and a PASS in this one.  The preflight arm
    is the one the 15 launch wrappers read
    (``molmetal/scripts/wf_launch_gate_verdict.py preflight --report ...``),
    so the hole was live on the launch path.

    THE RULE IS NOT INVENTED HERE — it is copied from ``check_checklist``
    below, which the repo already ships.  MEASURED on this tree
    (2026-11-04): every live consumer wrapper runs the pre-flight with
    ``--pipeline launch`` or ``--pipeline all``, and BOTH pipelines
    register the SAME 6 P0 gates (``p0_config_schema_verify``,
    ``p0_csv_hash_verify``, ``p0_scaffold_zero_leak``,
    ``p0_metalcytotox_v1_known_leaky``, ``p0_nci60_split_no_leak``,
    ``p0_gate_inventory_audit``), all 6 of which MEASURED PASS on this host
    (6 PASS / 0 FAIL / 0 SKIP / 0 WARN / 0 TIMEOUT, 47.19 s wall).  The arm
    is therefore a NO-OP on the current report and a BLOCK on a report whose
    blocker did not pass.

    HONEST LIMIT (stated, not implied): the arm constrains the P0 gates the
    report CARRIES.  A report whose ``gates`` list is absent, or which
    carries no P0 gate at all, is refused (fail-closed) rather than
    unconstrained — but the consumer does not know WHICH P0 gates a given
    pipeline owes, so it cannot detect a report that measured a DIFFERENT
    set.  The producer owns that list.
    """
    summary = report.get("summary")
    if not isinstance(summary, dict):
        return (
            VERDICT_FAIL,
            f"the pre-flight report carries no usable 'summary' block "
            f"(got {type(summary).__name__}) -- a report without a summary "
            f"cannot be shown to have passed any gate",
        )

    n_pass = int(summary.get("pass", 0) or 0)
    n_fail = int(summary.get("fail", 0) or 0)
    n_skip = int(summary.get("skip", 0) or 0)
    n_timeout = int(summary.get("timeout", 0) or 0)
    n_total = int(summary.get("total", 0) or 0)
    detail = (
        f"pass={n_pass} fail={n_fail} skip={n_skip} "
        f"timeout={n_timeout} total={n_total}"
    )
    if n_total <= 0:
        return (
            VERDICT_FAIL,
            f"the pre-flight report measures {n_total} gate(s) -- no gate was "
            f"run, so nothing can be shown to have passed [{detail}]",
        )
    if n_fail:
        return VERDICT_FAIL, f"the pre-flight reports {n_fail} FAIL gate(s) [{detail}]"
    if n_timeout:
        return (
            VERDICT_FAIL,
            f"the pre-flight reports {n_timeout} TIMEOUT gate(s) -- the "
            f"measurement was NOT obtained [{detail}]",
        )

    # ---- the P0 blockers -------------------------------------------------
    # The rule the checklist arm already enforces (see ``check_checklist``):
    # a P0 blocker that is not PASS did not verify what it claims to have
    # verified, so it must not authorise a launch.
    p0 = _p0_blockers(report)
    if p0 is None:
        return (
            VERDICT_FAIL,
            "the pre-flight report carries no usable 'gates' list -- the P0 "
            "blockers cannot be shown to have been verified",
        )
    if not p0:
        return (
            VERDICT_FAIL,
            "the pre-flight report measures no P0 blocker gate -- the P0 "
            "blockers cannot be shown to have been verified",
        )
    bad = [
        f"{name}={status}" for name, status in p0 if status != VERDICT_PASS
    ]
    if bad:
        return (
            VERDICT_FAIL,
            f"P0 blocker gate(s) are not PASS: {', '.join(bad)}",
        )

    stale = _freshness_verdict(report)
    if stale is not None:
        return stale

    unbound = _provenance_verdict(report)
    if unbound is not None:
        return unbound

    return VERDICT_PASS, f"the pre-flight reports no FAIL and no TIMEOUT [{detail}]"


def check_checklist(report: Dict[str, Any]) -> Tuple[str, str]:
    """PASS iff the checklist read a clean pre-flight and every P0 passed.

    Wf-A269 wave-16 (2026-11-03): fail-closed on a WRONG-SCHEMA checklist.
    The old implementation defaulted a missing ``preflight_fail_count`` to
    0 and a missing ``p0_blocker_gates`` to ``[]``, so a checklist that
    carried neither key was reported PASS — the same fail-open shape as the
    preflight arm above.  ``per_wave_checklist.py`` always writes BOTH keys
    (its own defaults at :756-757, the measured values at :789/:795), so a
    missing key means a malformed report, not an empty one.
    """
    err = str(report.get("error") or "")
    if err:
        return VERDICT_FAIL, f"the per-wave checklist recorded an error: {err}"

    if "preflight_fail_count" not in report:
        return (
            VERDICT_FAIL,
            "the per-wave checklist carries no 'preflight_fail_count' field "
            "-- the pre-flight it read cannot be shown to be clean",
        )

    n_fail = int(report.get("preflight_fail_count", 0) or 0)
    n_pass = int(report.get("preflight_pass_count", 0) or 0)
    if n_fail:
        return (
            VERDICT_FAIL,
            f"the pre-flight the checklist read has {n_fail} FAIL gate(s) "
            f"(pass={n_pass})",
        )

    if "p0_blocker_gates" not in report:
        return (
            VERDICT_FAIL,
            "the per-wave checklist carries no 'p0_blocker_gates' list -- "
            "the P0 blockers cannot be shown to have been verified",
        )

    p0: List[Dict[str, Any]] = list(report.get("p0_blocker_gates") or [])
    bad = [
        f"{g.get('gate_name', '?')}={str(g.get('status', '?')).upper()}"
        for g in p0
        if str(g.get("status", "")).upper() != VERDICT_PASS
    ]
    if bad:
        return (
            VERDICT_FAIL,
            f"P0 blocker gate(s) are not PASS: {', '.join(bad)}",
        )
    return (
        VERDICT_PASS,
        f"{len(p0)} P0 blocker gate(s) PASS; the pre-flight it read has "
        f"{n_fail} FAIL / {n_pass} PASS",
    )


def check_budget(
    report: Dict[str, Any], wave: Optional[str] = None
) -> Tuple[str, str]:
    """PASS iff the wave fits the session cap and is not deferred."""
    framing = str(report.get("honest_framing") or "")

    deferred = list(report.get("deferred_waves") or [])
    if wave and wave in deferred:
        return (
            VERDICT_FAIL,
            f"wave '{wave}' is DEFERRED by the wave budget tracker [{framing}]",
        )

    if "remaining_hr" not in report:
        return VERDICT_FAIL, "the budget report carries no remaining_hr field"
    remaining = float(report.get("remaining_hr") or 0.0)
    if remaining < 0:
        return (
            VERDICT_FAIL,
            f"the wave does not fit the session cap: remaining_hr={remaining} "
            f"[{framing}]",
        )
    return VERDICT_PASS, f"remaining_hr={remaining} [{framing}]"


def _verdict_for(kind: str, report_path: Path, wave: Optional[str]) -> Tuple[str, str]:
    report = _load_report(report_path)
    if report is None:
        return VERDICT_SKIP, f"the {kind} report is missing or unreadable: {report_path}"
    if kind == "preflight":
        return check_preflight(report)
    if kind == "checklist":
        return check_checklist(report)
    if kind == "budget":
        return check_budget(report, wave=wave)
    return VERDICT_SKIP, f"unknown gate kind: {kind}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Decide PASS / FAIL for a launcher's pre-launch gate from the "
            "report the gate tool already wrote."
        )
    )
    ap.add_argument(
        "--json", action="store_true", help="emit the verdict as JSON"
    )
    sub = ap.add_subparsers(dest="kind", required=True)

    p_pf = sub.add_parser("preflight", help="a pre_flight_check.py --json report")
    p_pf.add_argument("--report", required=True, type=Path)

    p_cl = sub.add_parser("checklist", help="a per_wave_checklist.py --out report")
    p_cl.add_argument("--report", required=True, type=Path)

    p_bg = sub.add_parser("budget", help="a wave_budget_tracker.py --report-json report")
    p_bg.add_argument("--report", required=True, type=Path)
    p_bg.add_argument(
        "--wave",
        default=None,
        help="the wave name to look for in the report's deferred_waves list",
    )

    args = ap.parse_args(argv)

    wave = getattr(args, "wave", None)
    verdict, detail = _verdict_for(args.kind, args.report, wave)

    if args.json:
        print(
            json.dumps(
                {
                    "verdict": verdict,
                    "kind": args.kind,
                    "report": str(args.report),
                    "wave": wave,
                    "detail": detail,
                },
                indent=2,
            )
        )
    else:
        print(f"{verdict} {detail}")

    return _EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
