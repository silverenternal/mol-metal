# Licensed under the Apache License, Version 2.0.
r"""Wf-A269 wave-18 (2026-11-03) — the FAIL-CLOSED launch-gate guard.

WHAT THIS FILE GUARDS
---------------------
The repo's launch surface has two gates that used to be DEAD GATES — checks
that looked like checks but could not fail (AGENTS.md §6.269.17):

  1. **the verdict CONSUMER** (``molmetal/scripts/wf_launch_gate_verdict.py``)
     read a report with ``report.get("summary") or {}`` and counted FAIL and
     TIMEOUT out of it, so a truncated or wrong-schema report yielded
     ``fail=0, timeout=0`` and was reported PASS;
  2. **the verdict PRODUCER** (``molmetal/scripts/pre_flight_check.py``)
     exited 0 for "1 TIMEOUT, 0 FAILs", while the consumer said FAIL for the
     SAME report — two channels disagreeing about one artefact, with 39 of
     the 55 wrappers reading only the exit code.

This file pins BOTH, plus the third leg the wave repaired: the 16 launch
wrappers that gate on the pre-flight must read the verdict through the
consumer, from a report the tool actually wrote as JSON, with the previous
report removed first.

THE MUTATION REQUIREMENT (this is the point of the file)
--------------------------------------------------------
Every assertion here that claims "a verdict must FLIP" is proven by PLANTING
ONE DELIBERATE DEFECT and requiring the verdict to flip.  A guard that cannot
be made to fail is not a guard; it is a comment with a pytest import.  So:

  * Part A plants a defect in the REPORT and requires FAIL;
  * Part C plants a defect in the WRAPPER (five distinct ones, one per
    contract rule) and requires the matching finding to appear;
  * Part D plants defects in the exit-code TABLE and requires the exit code
    to change.

WHAT THIS FILE DOES NOT COVER (stated, not implied)
---------------------------------------------------
* It does not run the pre-flight.  The 129-gate ``all`` pipeline takes
  ~248 s and the ``launch`` pipeline ~102 s; a unit test that runs either is
  a test nobody runs.  The producer's report SHAPE is pinned statically and
  by a direct call to ``_build_provenance`` instead.
* It does not run a launcher.  Every launcher here is a GPU launcher; running
  one is out of scope for a CPU-only guard (and forbidden by the standing
  0-GPU directive of this wave).
* The launcher checks are TEXT checks.  They prove the SHAPE of the gate, not
  that the gate passes on this machine.

CPU-ONLY BY CONSTRUCTION: this file imports no torch, spawns no trainer, and
allocates no device memory.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.wf_launch_gate_verdict import (  # noqa: E402
    VERDICT_FAIL,
    VERDICT_PASS,
    canonical_payload_sha256,
    check_budget,
    check_checklist,
    check_preflight,
)
from molmetal.tests.wfa269_launcher_contract import (  # noqa: E402
    broken_variants,
    contract_findings,
    report_paths,
    resolved_report_path,
)

CONSUMER = REPO_ROOT / "molmetal" / "scripts" / "wf_launch_gate_verdict.py"
PRODUCER = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: The gate source a report is bound to (the producer names itself).
GATE_SOURCE_REL = "molmetal/scripts/pre_flight_check.py"

#: The contract rules this file proves reachable.
CONTRACT_RULES = ("C2", "C3", "C4", "C5", "C6")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _live_head_sha() -> Optional[str]:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _bound_report(
    summary: Optional[Dict[str, Any]] = None,
    *,
    last_refresh: Optional[str] = None,
    git_sha: Optional[str] = None,
    gate_source: Optional[str] = GATE_SOURCE_REL,
    gate_source_sha256: Optional[str] = None,
    payload_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """A report bound to THIS tree, exactly as the producer would write it.

    The binding is the same three-part shape the producer uses: a
    ``last_refresh`` freshness stamp, a provenance block naming the commit
    and the gate source, and a SHA-256 of the report's own content.
    """
    report: Dict[str, Any] = {
        "pipeline": "launch",
        # 2026-11-04 (Wf-A269 dead-gates sweep): the fixture's gate row now
        # carries a REALISTIC priority.  The producer writes
        # ``GateResult.to_dict()`` = name / priority / pipeline / status /
        # evidence / wall_sec, so a row WITHOUT a priority is not a report
        # the producer could have written — and the preflight arm's new P0
        # rule reads that field.  The row is a P0 one, which is what the
        # ``all`` and ``launch`` pipelines actually carry (MEASURED 2026-11-04:
        # both register the SAME 6 P0 gates).
        "gates": [
            {
                "name": "g_sample",
                "priority": "P0",
                "pipeline": "launch",
                "status": "PASS",
                "evidence": "sample",
                "wall_sec": 0.0,
            }
        ],
        "summary": (
            summary
            if summary is not None
            else {"pass": 10, "fail": 0, "skip": 2, "timeout": 0, "total": 12}
        ),
        # NOTE: `is None` and not `or` — an EMPTY stamp is a defect the
        # freshness arm must see, and `or` would silently replace it with
        # "now" and make the mutation unreachable.
        "last_refresh": (
            datetime.now(timezone.utc).isoformat()
            if last_refresh is None
            else last_refresh
        ),
    }
    provenance: Dict[str, Any] = {
        "level": "L1",
        "git_sha": _live_head_sha() if git_sha is None else git_sha,
        "git_dirty": None,
    }
    if gate_source:
        provenance["gate_source"] = gate_source
        if gate_source_sha256 is not None:
            provenance["gate_source_sha256"] = gate_source_sha256
        else:
            gate_path = REPO_ROOT / gate_source
            provenance["gate_source_sha256"] = (
                _sha256_file(gate_path)
                if gate_path.exists()
                # A gate source that is NOT on disk is the case the
                # consumer's "cannot be read" arm exists for, so the report
                # must still carry a (necessarily wrong) hash.
                else "0" * 64
            )
    report["provenance"] = provenance
    report["provenance"]["payload_sha256"] = (
        canonical_payload_sha256(report)
        if payload_sha256 is None
        else payload_sha256
    )
    return report


def _consumer_launchers() -> List[Path]:
    """The wrappers that gate on the pre-flight THROUGH the shared consumer."""
    out = []
    for path in sorted(SCRIPTS_DIR.glob("*.sh")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "wf_launch_gate_verdict" in text:
            out.append(path)
    return out


# ---------------------------------------------------------------------------
# Part A — the consumer's fail-closed arms
# ---------------------------------------------------------------------------


def test_a_good_report_passes() -> None:
    """The baseline: a report bound to this tree with no FAIL/TIMEOUT PASSes."""
    verdict, reason = check_preflight(_bound_report())
    assert verdict == VERDICT_PASS, reason


@pytest.mark.parametrize(
    "mutate, expect_fragment",
    [
        pytest.param(lambda r: r.pop("summary"), "summary", id="no-summary"),
        pytest.param(lambda r: r.update(summary=None), "summary", id="null-summary"),
        pytest.param(lambda r: r.update(summary=[]), "summary", id="list-summary"),
        pytest.param(lambda r: r.update(summary="PASS"), "summary", id="str-summary"),
    ],
)
def test_b_a_report_without_a_usable_summary_fails(
    mutate, expect_fragment: str
) -> None:
    """MUTATION: break the summary -> the verdict MUST flip to FAIL.

    The pre-repair consumer read ``report.get("summary") or {}`` and counted
    FAIL/TIMEOUT out of an empty dict, so every one of these four shapes was
    reported PASS.  A report that cannot be shown to have passed any gate
    must not authorise a launch.
    """
    report = _bound_report()
    mutate(report)
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert expect_fragment in reason


def test_c_a_report_measuring_zero_gates_fails() -> None:
    """MUTATION: zero gates measured -> FAIL (nothing can have passed)."""
    report = _bound_report(summary={"pass": 0, "fail": 0, "skip": 0, "timeout": 0, "total": 0})
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert "gate(s)" in reason


def test_d_a_failing_gate_fails_the_verdict() -> None:
    """MUTATION: one FAIL gate -> FAIL."""
    report = _bound_report(
        summary={"pass": 10, "fail": 1, "skip": 2, "timeout": 0, "total": 13}
    )
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert "FAIL" in reason


def test_e_a_timeout_gate_fails_the_verdict() -> None:
    """MUTATION: one TIMEOUT gate -> FAIL ("the measurement was NOT obtained").

    This is the TIMEOUT-BLINDNESS class, and it is the arm the producer's
    exit code used to disagree with.  A gate that timed out did not verify
    what it claims to have verified, so it must not authorise a launch.
    """
    report = _bound_report(
        summary={"pass": 11, "fail": 0, "skip": 2, "timeout": 1, "total": 14}
    )
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert "TIMEOUT" in reason


def test_f_a_skip_is_not_a_failure() -> None:
    """A SKIP alone does NOT fail the verdict.

    Stated explicitly because the fix must not over-reach: the repo runs
    10-12 SKIPs BY DESIGN, so an arm that failed on SKIP would block every
    launch for no reason.
    """
    report = _bound_report(
        summary={"pass": 118, "fail": 0, "skip": 11, "timeout": 0, "total": 129}
    )
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_PASS, reason


@pytest.mark.parametrize(
    "stamp, expect_fragment",
    [
        pytest.param("", "last_refresh", id="missing"),
        pytest.param("not-a-timestamp", "ISO-8601", id="unparseable"),
    ],
)
def test_g_a_report_without_a_usable_freshness_stamp_fails(
    stamp: str, expect_fragment: str
) -> None:
    """MUTATION: break the freshness stamp -> FAIL."""
    report = _bound_report(last_refresh=stamp)
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert expect_fragment in reason


def test_h_a_stale_report_fails() -> None:
    """MUTATION: age the stamp past 24 h -> FAIL (AGENTS.md §4 stale-flag rule)."""
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    report = _bound_report(last_refresh=old)
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert "STALE" in reason


def test_i_a_future_stamp_fails() -> None:
    """MUTATION: stamp 1 h in the FUTURE -> FAIL (forged / wrong clock)."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    report = _bound_report(last_refresh=future)
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert "FUTURE" in reason


def test_j_a_report_without_provenance_fails() -> None:
    """MUTATION: drop the provenance block -> FAIL.

    "A verdict with no provenance cannot be shown to belong to this tree."
    """
    report = _bound_report()
    report.pop("provenance")
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert "provenance" in reason


def test_k_a_report_without_a_payload_hash_fails() -> None:
    """MUTATION: drop the payload hash -> FAIL (the body cannot be checked)."""
    report = _bound_report(payload_sha256="")
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert "payload" in reason


def test_l_an_edited_report_fails() -> None:
    """MUTATION: edit the report's own body -> FAIL.

    This is the wave-17 headline.  MEASURED before the payload binding
    existed: editing ``summary.fail`` from 1 to 0 in a real report still
    yielded PASS, because neither the commit SHA nor the gate-source hash
    covers the report's BODY.  The verdict must flip.
    """
    report = _bound_report(
        summary={"pass": 10, "fail": 1, "skip": 2, "timeout": 0, "total": 13}
    )
    assert check_preflight(report)[0] == VERDICT_FAIL  # the honest FAIL

    report["summary"]["fail"] = 0  # the forgery
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert "payload hash" in reason


def test_m_a_report_from_another_commit_fails() -> None:
    """MUTATION: bind the report to a foreign commit -> FAIL.

    Frontier rule (SLSA-L1 / in-toto): "a verdict is earned at a specific
    commit SHA and does not carry forward across a HEAD move", with "stale
    is treated as missing".
    """
    report = _bound_report(git_sha="0" * 40)
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert "HEAD" in reason


def test_n_a_report_from_another_gate_implementation_fails() -> None:
    """MUTATION: bind the report to a different gate-source hash -> FAIL."""
    report = _bound_report(gate_source_sha256="0" * 64)
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert "DIFFERENT version of the gate" in reason


def test_o_a_report_bound_to_a_missing_gate_source_fails() -> None:
    """MUTATION: name a gate source that is not on disk -> FAIL."""
    report = _bound_report(gate_source="molmetal/scripts/no_such_gate.py")
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL
    assert "cannot be read" in reason


# --- the checklist and budget arms ----------------------------------------


def test_p_a_checklist_without_a_preflight_fail_count_fails() -> None:
    """MUTATION: drop the key -> FAIL (the pre-flight it read is unknown)."""
    verdict, reason = check_checklist({"p0_blocker_gates": []})
    assert verdict == VERDICT_FAIL
    assert "preflight_fail_count" in reason


def test_q_a_checklist_without_p0_blocker_gates_fails() -> None:
    """MUTATION: drop the key -> FAIL (the P0 blockers are unverified)."""
    verdict, reason = check_checklist({"preflight_fail_count": 0})
    assert verdict == VERDICT_FAIL
    assert "p0_blocker_gates" in reason


def test_r_a_checklist_with_a_non_passing_p0_blocker_fails() -> None:
    """MUTATION: flip one P0 blocker from PASS to SKIP -> FAIL."""
    checklist = {
        "preflight_fail_count": 0,
        "p0_blocker_gates": [{"gate_name": "g", "status": "PASS"}],
    }
    assert check_checklist(checklist)[0] == VERDICT_PASS
    checklist["p0_blocker_gates"][0]["status"] = "SKIP"
    verdict, reason = check_checklist(checklist)
    assert verdict == VERDICT_FAIL
    assert "P0" in reason


def test_s_a_budget_report_without_remaining_hr_fails() -> None:
    """MUTATION: drop the key -> FAIL (the wave's fit is unknown)."""
    verdict, reason = check_budget({"honest_framing": "x"})
    assert verdict == VERDICT_FAIL
    assert "remaining_hr" in reason


def test_t_a_negative_remaining_budget_fails() -> None:
    """MUTATION: negative remaining_hr -> FAIL."""
    verdict, reason = check_budget({"remaining_hr": -0.5, "honest_framing": "x"})
    assert verdict == VERDICT_FAIL
    assert "does not fit" in reason


# ---------------------------------------------------------------------------
# Part B — the producer's stamps
# ---------------------------------------------------------------------------


def test_u_the_producer_stamps_freshness_and_provenance() -> None:
    """The producer must WRITE the three things the consumer reads.

    AGENTS.md §4 stated a stale-flag rule for many waves while
    ``grep -c last_refresh`` on the producer returned 0 — the documented
    contract was false, so every consumer that honoured it had nothing to
    honour.  This pins the contract back into the producer.
    """
    src = PRODUCER.read_text(encoding="utf-8")
    assert '"last_refresh"' in src, "the producer does not stamp last_refresh"
    assert '"provenance"' in src, "the producer does not attach a provenance block"
    assert "payload_sha256" in src, "the producer does not bind its own payload"


def test_v_the_producer_builds_a_bound_provenance_block() -> None:
    """BEHAVIOURAL: the producer's own provenance builder names the tree.

    Direct call rather than a source scan, so a rename that drops the
    binding fails here.
    """
    from molmetal.scripts.pre_flight_check import _build_provenance

    prov = _build_provenance(datetime.now(timezone.utc).isoformat())
    for key in ("level", "git_sha", "gate_source", "gate_source_sha256"):
        assert key in prov, f"the provenance block has no '{key}'"
    assert prov["gate_source"] == GATE_SOURCE_REL
    assert prov["gate_source_sha256"] == _sha256_file(REPO_ROOT / GATE_SOURCE_REL)


def test_w_the_producer_and_consumer_share_one_hash_definition() -> None:
    """The two sides must not drift into hashing different bytes.

    A drift would make every verdict FAIL — fail-closed, but useless.  So
    the producer IMPORTS the consumer's definition rather than repeating it.
    """
    src = PRODUCER.read_text(encoding="utf-8")
    assert "canonical_payload_sha256" in src
    assert (
        "from molmetal.scripts.wf_launch_gate_verdict import" in src
        or "import wf_launch_gate_verdict" in src
    ), "the producer does not import the shared hash definition"


def test_x_the_producer_writes_its_json_atomically() -> None:
    """The report is written with the repo's atomic primitive (§5.6).

    A concurrent reader must never see a half-written report: the consumer
    reads it while the producer may still be writing it.
    """
    src = PRODUCER.read_text(encoding="utf-8")
    assert "atomic_write_file" in src
    assert "atomic_write_file(args.json" in src


def test_y_the_producer_and_consumer_agree_on_the_payload_hash() -> None:
    """BEHAVIOURAL: a report written the producer's way verifies.

    This closes the loop: the fixture the consumer arms above are built
    with the consumer's hash function, and this asserts the producer's own
    stamp of the same content matches — so a report that the producer wrote
    is one the consumer accepts.
    """
    report = _bound_report()
    recorded = report["provenance"]["payload_sha256"]
    assert recorded == canonical_payload_sha256(report)
    assert check_preflight(report)[0] == VERDICT_PASS


# ---------------------------------------------------------------------------
# Part C — the launcher contract
# ---------------------------------------------------------------------------


def test_z_there_are_enough_consumer_launchers_to_be_worth_guarding() -> None:
    """MEASURED count, so a mass deletion cannot silently empty the surface."""
    launchers = _consumer_launchers()
    assert len(launchers) >= 16, (
        f"expected at least 16 wrappers reaching the shared consumer, found "
        f"{len(launchers)}: {[p.name for p in launchers]}"
    )


@pytest.mark.parametrize("path", _consumer_launchers(), ids=lambda p: p.name)
def test_aa_every_consumer_launcher_satisfies_the_contract(path: Path) -> None:
    """Every wrapper that gates on the pre-flight must satisfy C1-C6.

    C1  it really invokes the pre-flight;
    C2  the invocation writes a report to a PATH (so a consumer can read it);
    C3  the report path is not a merged stdout+stderr capture (which is not
        JSON at all);
    C4  the previous report is removed first ("stale is treated as missing");
    C5  no TIMEOUT-blind ``grep '"fail": 0'`` idiom in live code;
    C6  the verdict comes from the shared FAIL-CLOSED consumer, for the
        pre-flight kind.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    findings = contract_findings(text)
    # The lowercase twin is deliberately INERT (Wf-A269 wave-18 retired it):
    # it launches nothing, so it is exempt from the launch contract and is
    # pinned separately by test_ac below.
    if path.name == "_wrapper_template.sh":
        assert findings == ["C1: the wrapper never invokes pre_flight_check.py"]
        return
    assert findings == [], f"{path.name}: {findings}"


@pytest.mark.parametrize("path", _consumer_launchers(), ids=lambda p: p.name)
def test_ab_the_contract_detector_can_fail(path: Path) -> None:
    """THE MUTATION PROOF for Part C.

    For each contract rule, plant ONE deliberate defect in the wrapper and
    require the matching finding to APPEAR.  A rule whose mutation does not
    flip the detector is a rule that cannot fail — §6.269.17's dead-gate
    class, wearing the costume of a test.
    """
    if path.name == "_wrapper_template.sh":
        pytest.skip("the retired twin is pinned by test_ac, not by the contract")
    text = path.read_text(encoding="utf-8", errors="replace")
    variants = broken_variants(text)
    built = {rule for rule, _ in variants}
    missing = set(CONTRACT_RULES) - built
    assert not missing, (
        f"{path.name}: no mutation could be built for {sorted(missing)} — the "
        f"rule is untested, not satisfied"
    )
    for rule, broken in variants:
        findings = contract_findings(broken)
        assert any(f.startswith(rule + ":") for f in findings), (
            f"{path.name}: the {rule} mutation did not flip the detector "
            f"(findings: {findings})"
        )


def test_ac_the_retired_lowercase_template_twin_is_inert() -> None:
    """The second, wrong wrapper template must not be able to launch anything.

    It used to be a live DEAD GATE: it tested that the pre-flight report
    FILE EXISTED (existence is not a verdict) and then wrote the literal
    string ``pre_flight_status: PASS`` into its dry-run log regardless of
    any measurement.
    """
    path = SCRIPTS_DIR / "_wrapper_template.sh"
    text = path.read_text(encoding="utf-8")
    # Its live code must not exec a trainer, probe a GPU, or write a verdict.
    from molmetal.tests.wfa269_launcher_contract import logical_lines

    live = "\n".join(line for _ln, line in logical_lines(text))
    assert "exec " not in live, "the retired twin still execs something"
    assert "nvidia-smi" not in live, "the retired twin still probes a GPU"
    assert "pre_flight_status: PASS" not in live, (
        "the retired twin still writes a hard-coded PASS verdict"
    )
    # It must still carry the superseded rendering (AGENTS.md §5.5 rule #6).
    assert "SUPERSEDED RENDERING PRESERVED VERBATIM" in text


def test_ad_the_canonical_template_satisfies_the_contract() -> None:
    """The template new wrappers are copied from must satisfy the contract.

    A template that carries a broken gate propagates it to every wrapper
    written from it, which is how a dead gate becomes a family of dead
    gates.
    """
    path = SCRIPTS_DIR / "_WRAPPER_TEMPLATE.sh"
    text = path.read_text(encoding="utf-8")
    assert contract_findings(text) == []


def test_ae_the_report_paths_of_the_launchers_are_distinct() -> None:
    """Two wrappers writing the SAME report path would clobber each other.

    A wrapper that reads a report another wrapper just wrote reads a verdict
    earned by a different measurement.
    """
    seen: Dict[str, str] = {}
    for path in _consumer_launchers():
        if path.name == "_wrapper_template.sh":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for _ln, expr in report_paths(text):
            if not expr:
                continue
            resolved = resolved_report_path(text, expr)
            key = resolved.replace("${", "").replace("}", "")
            owner = seen.get(key)
            assert owner is None, (
                f"{path.name} and {owner} both write the report path "
                f"{resolved!r}"
            )
            seen[key] = path.name
    assert len(seen) >= 16, f"only {len(seen)} distinct report paths were found"


# ---------------------------------------------------------------------------
# Part D — the producer's exit code
# ---------------------------------------------------------------------------


#: The producer's exit-code tail, matched EXACTLY.
#:
#: The end marker is anchored with ``^    return 0$`` — four spaces and the
#: whole line.  A plain ``"    return 0\n"`` substring search would also
#: match the EIGHT-space-indented ``        return 0`` of a nested branch,
#: truncating the block early.  That truncation was not harmless: it made
#: the arm fail for the wrong reason (a ``None`` exit code rather than a
#: wrong one), which is the difference between a guard that reports the
#: defect and a guard that reports its own accident.
_EXIT_TAIL_RE = re.compile(
    r"^    if n_fail > 0:\n        return 1\n.*?^    return 0$",
    re.M | re.S,
)


def _exit_code_fn():
    """The producer's exit-code decision, as an executable function.

    Extracted from the file rather than re-implemented, so the test measures
    the SHIPPED behaviour and cannot drift from it.  A missing marker is a
    hard failure, not a skip: a skip here would make the whole arm vacuous.
    """
    src = PRODUCER.read_text(encoding="utf-8")
    match = _EXIT_TAIL_RE.search(src)
    if match is None:
        pytest.fail(
            "the producer's exit-code tail no longer matches "
            "'if n_fail > 0: return 1 ... return 0' — this guard must be "
            "updated to match the shipped code"
        )
    block = textwrap.dedent(match.group(0))
    fn = (
        "def _exit_code(args, n_fail, n_skip, n_timeout):\n"
        + textwrap.indent(block, "    ")
    )
    namespace: Dict[str, Any] = {}
    exec(compile(fn, str(PRODUCER), "exec"), namespace)
    return namespace["_exit_code"]


class _Args:
    def __init__(self, strict: bool) -> None:
        self.strict = strict


def _exit_codes() -> Dict[Tuple[bool, int, int, int], int]:
    """The producer's exit code for a table of (strict, fail, skip, timeout)."""
    fn_ = _exit_code_fn()
    table: Dict[Tuple[bool, int, int, int], int] = {}
    for strict in (False, True):
        for n_fail in (0, 1):
            for n_skip in (0, 11):
                for n_timeout in (0, 1):
                    table[(strict, n_fail, n_skip, n_timeout)] = fn_(
                        _Args(strict), n_fail, n_skip, n_timeout
                    )
    return table


def test_af_the_exit_code_table_is_the_contracted_one() -> None:
    """The exit code is a LAUNCH-AUTHORISATION channel, so it must be exact.

    MEASURED 2026-11-03: 39 of the 55 wrappers that call the pre-flight
    gate on its EXIT CODE and never read the report; the other 16 read the
    report through the shared consumer.  Before this wave the two channels
    disagreed about the SAME artefact — the exit code said 0 for "1 TIMEOUT,
    0 FAILs" while the consumer said FAIL.

    The table (strict, fail, skip, timeout) -> exit code:

        (False, 0, 11, 0) -> 0    a SKIP alone is not a failure
        (False, 0,  0, 1) -> 1    a TIMEOUT is not a failure, but it is not
                                 a PASS either: the measurement was not obtained
        (False, 1,  0, 0) -> 1    a FAIL always wins
        (True,  0, 11, 0) -> 2    --strict turns a SKIP into a non-zero exit
        (True,  0,  0, 1) -> 2    --strict still turns a TIMEOUT into 2
        (True,  1,  0, 0) -> 1    a FAIL outranks --strict
    """
    table = _exit_codes()
    expected = {
        (False, 0, 11, 0): 0,
        (False, 0, 0, 1): 1,
        (False, 1, 0, 0): 1,
        (True, 0, 11, 0): 2,
        (True, 0, 0, 1): 2,
        (True, 1, 0, 0): 1,
    }
    for key, want in expected.items():
        got = table[key]
        assert got == want, (
            f"exit code for (strict={key[0]}, fail={key[1]}, skip={key[2]}, "
            f"timeout={key[3]}) is {got}, expected {want}"
        )


def test_ag_the_exit_code_flips_when_a_timeout_appears() -> None:
    """MUTATION: add a TIMEOUT to a clean run -> the exit code MUST change.

    Without this assertion the "TIMEOUT is not a defect verdict" reading
    could come back, and 39 wrappers would again authorise a launch on a
    measurement that was never obtained.
    """
    table = _exit_codes()
    assert table[(False, 0, 11, 0)] == 0
    assert table[(False, 0, 11, 1)] == 1, (
        "a TIMEOUT no longer changes the exit code — the TIMEOUT-blind "
        "class is back"
    )


def test_ah_the_exit_code_still_rejects_a_failure_under_strict() -> None:
    """MUTATION: --strict must not change a FAIL's exit code from 1."""
    table = _exit_codes()
    assert table[(False, 1, 0, 0)] == 1
    assert table[(True, 1, 0, 0)] == 1
    assert table[(True, 1, 11, 1)] == 1


def test_ai_the_exit_code_does_not_change_on_a_skip_alone() -> None:
    """The repo runs 10-12 SKIPs BY DESIGN; they must not block a launch."""
    table = _exit_codes()
    assert table[(False, 0, 0, 0)] == 0
    assert table[(False, 0, 11, 0)] == 0


# ---------------------------------------------------------------------------
# Part E — the consumer's own CLI contract
# ---------------------------------------------------------------------------


def test_aj_the_consumer_cli_fails_closed_on_a_broken_report(tmp_path: Path) -> None:
    """BEHAVIOURAL, end to end through the CLI: a broken report => exit 1.

    The unit arms above call ``check_preflight`` directly.  This one runs the
    tool the wrappers actually run, so a regression in the argument parsing
    or in the exit-code mapping is caught too.
    """
    report_path = tmp_path / "preflight.json"
    report_path.write_text(
        json.dumps({"summary": {"pass": 1, "fail": 0, "skip": 0, "timeout": 0, "total": 1}}),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(CONSUMER),
            "preflight",
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 1, (
        f"the consumer reported exit {proc.returncode} for a report with no "
        f"provenance and no freshness stamp:\n{proc.stdout}\n{proc.stderr}"
    )


def test_ak_the_consumer_cli_fails_closed_on_a_missing_report(tmp_path: Path) -> None:
    """A MISSING report is a FAIL, not a SKIP and not a PASS."""
    proc = subprocess.run(
        [
            sys.executable,
            str(CONSUMER),
            "preflight",
            "--report",
            str(tmp_path / "does_not_exist.json"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0, (
        f"the consumer reported exit {proc.returncode} for a MISSING report"
    )


# ---------------------------------------------------------------------------
# Part F — the P0 arm of the PREFLIGHT consumer (2026-11-04, Wf-A269
# dead-gates sweep)
#
# DEFECT D: ``check_preflight`` read ONLY ``summary.fail`` / ``summary.timeout``
# and said NOTHING about the P0 blockers, while ``check_checklist`` — the SAME
# module's other arm — has always required every P0 gate to be PASS.  ONE
# report was therefore a FAIL in one arm and a PASS in the other, and the
# preflight arm is the one the 15 launch wrappers read.
#
# MEASURED before the repair: a report whose summary said
# ``fail=0, timeout=0`` and whose P0 blocker said ``SKIP`` was reported
# PASS by the preflight arm and FAIL by the checklist arm.
# ---------------------------------------------------------------------------


def _restamp(report: Dict[str, Any]) -> Dict[str, Any]:
    """Re-bind the report's payload hash after a deliberate mutation.

    The consumer is bound to the report's OWN content (Wf-A269 wave-17), so a
    mutation that is not re-stamped would be refused for the WRONG reason
    (the payload hash) and the rule under test would never be exercised.
    """
    report["provenance"]["payload_sha256"] = canonical_payload_sha256(report)
    return report


def _p0_report(
    status: str, *, summary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """A bound report whose single P0 blocker has `status`."""
    report = _bound_report(
        summary=summary
        or {"pass": 118, "fail": 0, "skip": 11, "timeout": 0, "total": 129}
    )
    report["gates"][0]["status"] = status
    return _restamp(report)


def test_fa_a_p0_blocker_that_skipped_fails_the_preflight_arm() -> None:
    """MUTATION: the P0 blocker's status PASS -> SKIP -> the verdict MUST flip.

    This is the defect: the summary still said ``fail=0, timeout=0``, so the
    pre-repair arm reported PASS — while the checklist arm reported FAIL for
    the very same report.
    """
    assert check_preflight(_p0_report("PASS"))[0] == VERDICT_PASS
    verdict, reason = check_preflight(_p0_report("SKIP"))
    assert verdict == VERDICT_FAIL, reason
    assert "P0" in reason, reason


def test_fb_the_two_arms_agree_on_a_skipped_p0_blocker() -> None:
    """THE POINT of the repair: ONE report, ONE verdict, in BOTH arms."""
    report = _p0_report("SKIP")
    preflight = check_preflight(report)[0]
    checklist = check_checklist(
        {
            "preflight_fail_count": 0,
            "preflight_pass_count": 118,
            "p0_blocker_gates": [
                {"gate_name": g["name"], "status": g["status"]}
                for g in report["gates"]
                if str(g.get("priority", "")).upper() == "P0"
            ],
        }
    )[0]
    assert preflight == checklist == VERDICT_FAIL, (
        f"the two arms disagree: preflight={preflight} checklist={checklist}"
    )


@pytest.mark.parametrize("status", ["SKIP", "WARN", "FAIL", "TIMEOUT"])
def test_fc_every_non_pass_p0_status_is_refused(status: str) -> None:
    """MUTATION: each non-PASS status of a P0 blocker -> FAIL."""
    verdict, reason = check_preflight(_p0_report(status))
    assert verdict == VERDICT_FAIL, reason
    assert "P0" in reason, reason


def test_fd_a_report_without_a_gates_list_fails() -> None:
    """MUTATION: drop the ``gates`` list -> FAIL (the blockers are unknown)."""
    report = _bound_report()
    report.pop("gates")
    _restamp(report)
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL, reason
    assert "gates" in reason, reason


def test_fe_a_report_with_no_p0_gate_fails() -> None:
    """MUTATION: measure no P0 gate at all -> FAIL (nothing was verified)."""
    report = _bound_report()
    report["gates"][0]["priority"] = "P1"
    _restamp(report)
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_FAIL, reason
    assert "P0" in reason, reason


def test_ff_a_non_p0_gate_that_skipped_is_still_tolerated() -> None:
    """The arm must NOT over-reach: a P1/P2 SKIP does not block a launch.

    The repo runs 10-12 SKIPs BY DESIGN (Wf-A135); only the P0 blockers are
    launch blockers.  Without this control the new arm would be a blanket
    "every gate must PASS", which the repo has never wanted — and a
    repair that over-reaches is a defect of its own.
    """
    report = _bound_report()
    report["gates"][0]["priority"] = "P1"
    report["gates"][0]["status"] = "SKIP"
    report["gates"].append({"name": "p0_ok", "priority": "P0", "status": "PASS"})
    _restamp(report)
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_PASS, reason


def test_fg_the_p0_arm_is_not_a_no_op(monkeypatch) -> None:
    """MUTATION PROOF: the arm reads the P0 STATUS — hide it and the verdict flips.

    The pre-repair preflight arm had no P0 rule at all: it counted FAIL and
    TIMEOUT out of ``summary`` and said nothing else.  This assertion is what
    makes the new arm a GUARD rather than a comment — when the arm can no
    longer see the blocker's status, a SKIP-ed blocker is reported PASS
    again.
    """
    from molmetal.scripts import wf_launch_gate_verdict as consumer

    report = _p0_report("SKIP")
    assert check_preflight(report)[0] == VERDICT_FAIL

    # The pre-repair semantics: the arm cannot see the blocker's status.
    monkeypatch.setattr(
        consumer, "_p0_blockers", lambda _report: [("g_sample", "PASS")]
    )
    verdict, reason = check_preflight(report)
    assert verdict == VERDICT_PASS, (
        "MUTATION NOT CAUGHT: with the blocker's status hidden the verdict "
        f"did not flip back to PASS ({reason})"
    )
