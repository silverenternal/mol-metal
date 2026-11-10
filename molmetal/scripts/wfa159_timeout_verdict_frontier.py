#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A159 Branch F -- the published-practice frontier for the rule
"TIMEOUT is its own verdict, not a defect claim".

WHAT THIS TOOL IS
-----------------
Wf-A158 (2026-11-01) applied ``pre_flight_check._timeout_gate_result`` to
three pre-flight gates: a wall-clock timeout now reports status ``TIMEOUT``
instead of ``FAIL``, because a timeout means the MEASUREMENT WAS NOT
OBTAINED and is therefore not evidence that the thing under test is broken
(AGENTS.md §5.2: a genuine defect is reported as FAIL, and only a genuine
defect may be reported as FAIL).

That is a DESIGN DECISION about what the repo reports about its own health.
This branch asks an external question: what does published practice actually
say?  The deliverable is a COMPARISON MATRIX -- one row per retrieved
source, one column per question -- NOT a vindication.

Four questions, one column each:

  Q0  TIMEOUT_NOT_DEFECT_CLAIM
      Does the source treat a timeout as NOT a correctness failure (a
      distinct outcome class)?
  Q1  BROKEN_VS_UNMEASURED
      Does the source distinguish "the subject is broken" from "we did not
      get a measurement"?
  Q2  STANDARD_NAME
      What is the STANDARD NAME the source uses for that distinction?
  Q3  BUDGET_VS_OBSERVED_COST
      Does the source state a rule for setting the wall-clock budget
      relative to the observed cost (and for not letting a marginal budget
      turn a load spike into a false verdict)?

Verdicts: AGREES / DISAGREES / DISTINGUISHES / NOT-STATED / NOT-RETRIEVED.
A source that DISAGREES is a FIRST-CLASS RESULT.  If mainstream CI practice
collapses "timed out" into "failed", then this repo's split is a deliberate,
defensible DEVIATION and the honest framing is that -- not "the literature
agrees with us".

OFFLINE BY CONSTRUCTION
-----------------------
Every fact below was retrieved LIVE in this session (2026-11-01) with
WebSearch + WebFetch and is embedded here as a literal constant, so this
script is deterministic and re-runnable with NO network access.  The
per-source quotes are the EXACT strings returned by the fetch; the
retrieval log records url -> retrieved-at -> what was extracted.

Every cell carries {verdict, quote, quote_context, locator, url,
retrieved_at, note}.  The self-check enforces:
  * every cell has a locator + a url + a retrieval date;
  * a cell that claims a verdict (AGREES / DISAGREES / DISTINGUISHES) MUST
    carry a non-empty quote, and that quote MUST be a substring of its own
    quote_context -- so no cell can claim a quote it does not carry;
  * a cell that claims NOT-STATED / NOT-RETRIEVED MUST carry an EMPTY
    quote -- so a cell cannot both claim an absence and smuggle a quote;
  * no torch import, no CUDA, no GPU probe, no checkpoint load.

CPU-ONLY: stdlib only (plus the repo's own atomic-write helper, also
stdlib-only).  No RDKit, no torch, no network.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

#: Snapshot of the torch-related modules already present when THIS module
#: was imported.  Under a pytest session the repo's own conftest imports
#: torch before any test runs (molmetal/_device.py), so "torch is absent
#: from sys.modules" is NOT a property this script can own.  What it CAN
#: own is: importing this module must not ADD a single torch module.
_TORCH_MODULES_AT_IMPORT = frozenset(
    name for name in sys.modules if name == "torch" or name.startswith("torch.")
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The session in which every fact below was retrieved live.  The WebFetch
#: tool does not expose a per-call clock, so the retrieval date is recorded
#: at DATE granularity.  It is a constant, so re-runs are byte-identical.
SESSION_DATE = "2026-11-01"

DEFAULT_OUT_DIR = REPO / "molmetal" / "reports" / "wfa159_timeout_verdict_frontier"
JSON_NAME = "timeout_verdict_matrix.json"
MD_NAME = "timeout_verdict_matrix.md"

VERDICTS: Tuple[str, ...] = (
    "AGREES",
    "DISAGREES",
    "DISTINGUISHES",
    "NOT-STATED",
    "NOT-RETRIEVED",
)

CLAIMING_VERDICTS = ("AGREES", "DISAGREES", "DISTINGUISHES")
ABSENCE_VERDICTS = ("NOT-STATED", "NOT-RETRIEVED")

QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "Q0_TIMEOUT_NOT_DEFECT_CLAIM",
        "column": "Q0 timeout != defect claim",
        "question": (
            "Does the source treat a timeout as NOT a correctness failure "
            "(i.e. as a distinct outcome class rather than a failure of the "
            "subject under test)?"
        ),
        "verdict_rule": (
            "AGREES = timeout is a distinct outcome class; DISAGREES = the "
            "timeout is reported as a failure of the subject; DISTINGUISHES = "
            "the timeout is handled distinctly but stays inside a failure "
            "vocabulary (e.g. a named reason)."
        ),
    },
    {
        "id": "Q1_BROKEN_VS_UNMEASURED",
        "column": "Q1 broken vs. unmeasured",
        "question": (
            "Does the source distinguish 'the subject under test is broken' "
            "from 'we did not obtain a measurement'?"
        ),
        "verdict_rule": (
            "AGREES = the distinction is made explicitly; DISAGREES = the "
            "source collapses the two; DISTINGUISHES = the distinction is "
            "made but with different vocabulary or only partially."
        ),
    },
    {
        "id": "Q2_STANDARD_NAME",
        "column": "Q2 standard name",
        "question": (
            "What STANDARD NAME does the source use for the 'no measurement "
            "obtained' outcome class?"
        ),
        "verdict_rule": (
            "AGREES = the source names a distinct outcome class; DISAGREES = "
            "the source has no name for it (it collapses into failure); "
            "DISTINGUISHES = a named sub-state inside a failure (e.g. a "
            "reason string)."
        ),
    },
    {
        "id": "Q3_BUDGET_VS_OBSERVED_COST",
        "column": "Q3 budget vs. observed cost",
        "question": (
            "Does the source state a rule for setting the wall-clock budget "
            "relative to the OBSERVED cost (and for preventing a marginal "
            "budget from turning a load spike into a false verdict)?"
        ),
        "verdict_rule": (
            "AGREES = an observed-cost-relative rule is stated (percentile, "
            "false-timeout rate, padding); DISTINGUISHES = a flat or "
            "size-tiered default with no observed-cost rule; NOT-RETRIEVED / "
            "NOT-STATED = nothing on the budget."
        ),
    },
]

QUESTION_IDS: Tuple[str, ...] = tuple(q["id"] for q in QUESTIONS)


def _cell(
    verdict: str,
    *,
    quote: str = "",
    quote_context: str = "",
    locator: str,
    note: str = "",
) -> Dict[str, Any]:
    """Build one matrix cell.  ``url`` / ``retrieved_at`` are filled in by
    :func:`_source` from the source-level record, so every cell of a source
    necessarily carries the same url + retrieval date."""
    assert verdict in VERDICTS, f"bad verdict {verdict!r}"
    return {
        "verdict": verdict,
        "quote": quote,
        "quote_context": quote_context,
        "locator": locator,
        "url": "",
        "retrieved_at": "",
        "note": note,
    }


def _source(
    sid: str,
    name: str,
    kind: str,
    url: str,
    tool: str,
    cells: Dict[str, Dict[str, Any]],
    retrieved_ok: bool = True,
) -> Dict[str, Any]:
    """Fill the source-level url + retrieval date into every cell."""
    assert set(cells) == set(QUESTION_IDS), (
        f"{sid}: cells {sorted(cells)} != questions {sorted(QUESTION_IDS)}"
    )
    for cell in cells.values():
        cell["url"] = url
        cell["retrieved_at"] = SESSION_DATE
    return {
        "id": sid,
        "name": name,
        "kind": kind,
        "url": url,
        "retrieved_at": SESSION_DATE,
        "tool": tool,
        "retrieved_ok": retrieved_ok,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# THE CORPUS -- every quote below was retrieved live on 2026-11-01
# ---------------------------------------------------------------------------

SOURCES: List[Dict[str, Any]] = [
    _source(
        "SRC-01",
        "pytest-timeout (signal / thread timeout methods)",
        "test-harness",
        "https://pytest-timeout.readthedocs.io/en/latest/",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "DISAGREES",
                quote=(
                    "the signal handler will dump the stack of any other "
                    "threads running to stderr and use ``pytest.fail()`` to "
                    "interrupt the test."
                ),
                quote_context=(
                    "Under the `signal` method: \"the signal handler will dump "
                    "the stack of any other threads running to stderr and use "
                    "``pytest.fail()`` to interrupt the test.\" and \"the "
                    "pytest process is not terminated and the test run can "
                    "complete normally\"."
                ),
                locator="section 'Timeout Methods' > 'signal'",
                note=(
                    "pytest-timeout has NO timeout-specific outcome class: a "
                    "timeout is an ordinary failing test. Under the `thread` "
                    "method the process is killed outright -- \"other pytest "
                    "features, like e.g. JUnit XML output or fixture teardown, "
                    "will not function normally\"."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "NOT-STATED",
                locator=(
                    "searched 'Timeout Methods' (thread, signal), 'Session "
                    "Timeout', 'Timeouts in Fixture Teardown' -- no statement "
                    "separating 'the subject is broken' from 'no measurement "
                    "obtained'"
                ),
                note=(
                    "The retrieved text does not make this distinction. The "
                    "plugin's vocabulary is pytest's own: a timed-out test is "
                    "a failing test."
                ),
            ),
            "Q2_STANDARD_NAME": _cell(
                "NOT-STATED",
                locator=(
                    "same sections as Q1 -- no timeout outcome-class name is "
                    "introduced anywhere in the retrieved text"
                ),
                note=(
                    "No distinct name exists on this surface; the timeout "
                    "surfaces as a normal test failure."
                ),
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "NOT-RETRIEVED",
                locator=(
                    "no timeout-sizing guidance present in the retrieved "
                    "rendering of the page (a second fetch of the same URL in "
                    "this session returned HTTP 404)"
                ),
                note=(
                    "No budget-sizing rule is quoted from this source. This "
                    "is a retrieval gap, not evidence that the project has no "
                    "such guidance."
                ),
            ),
        },
    ),
    _source(
        "SRC-02",
        "Bazel -- Test Encyclopedia ('Role of the test runner')",
        "test-harness",
        "https://bazel.build/reference/test-encyclopedia",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "DISAGREES",
                quote="it may choose to kill the test and treat the run as a failure.",
                quote_context=(
                    "\"If a test takes too long to execute, exceeds some "
                    "resource limit, or the test runner otherwise detects "
                    "prohibited behavior, it may choose to kill the test and "
                    "treat the run as a failure.\" and \"Any other result is "
                    "considered a test failure.\""
                ),
                locator="section 'Role of the test runner'",
                note=(
                    "Also: \"The runner must not report the test as passing "
                    "after sending a signal to the test process or any "
                    "children thereof.\" At the BUILD level Bazel collapses a "
                    "timeout into a failure -- contrast SRC-03, where Bazel's "
                    "event protocol keeps TIMEOUT as its own status."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "NOT-STATED",
                locator=(
                    "section 'Role of the test runner' -- the encyclopedia "
                    "states the failure rule but never separates 'the subject "
                    "is broken' from 'no measurement obtained'"
                ),
                note=(
                    "The nearest thing in the same document is that the "
                    "runner 'may choose to kill the test and treat the run as "
                    "a failure' -- a single failure class, no second class."
                ),
            ),
            "Q2_STANDARD_NAME": _cell(
                "NOT-STATED",
                locator=(
                    "section 'Role of the test runner' -- no timeout-specific "
                    "outcome name in the encyclopedia's result vocabulary"
                ),
                note=(
                    "The word TIMEOUT does exist in Bazel, but in SRC-03's "
                    "event protocol, not in this document."
                ),
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "DISTINGUISHES",
                quote=(
                    "Tests which do not explicitly specify a timeout have one "
                    "implied based on the test's size as follows:"
                ),
                quote_context=(
                    "\"Tests which do not explicitly specify a timeout have one "
                    "implied based on the test's size as follows: small -> "
                    "short (60 seconds), medium -> moderate (300 seconds), "
                    "large -> long (900 seconds), enormous -> eternal (3600 "
                    "seconds)\" and \"A \\\"large\\\" test with no explicit "
                    "timeout setting will be allotted 900 seconds to run.\""
                ),
                locator="section 'Role of the test runner' (default timeout by size)",
                note=(
                    "A size-tier policy keyed to a DECLARED cost class -- "
                    "published practice, but not a rule derived from an "
                    "observed latency distribution."
                ),
            ),
        },
    ),
    _source(
        "SRC-03",
        "Bazel -- test_status.proto, enum BlazeTestStatus (Build Event Protocol)",
        "build-protocol",
        "https://bazel.googlesource.com/bazel/+/HEAD/src/main/protobuf/test_status.proto",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "AGREES",
                quote="TIMEOUT = 3;",
                quote_context=(
                    "enum BlazeTestStatus { NO_STATUS = 0; PASSED = 1; "
                    "FLAKY = 2; TIMEOUT = 3; FAILED = 4; INCOMPLETE = 5; "
                    "REMOTE_FAILURE = 6; FAILED_TO_BUILD = 7; "
                    "BLAZE_HALTED_BEFORE_TESTING = 8; }  // \"Detailed status "
                    "data for a TestRunnerAction execution.\""
                ),
                locator="src/main/protobuf/test_status.proto, enum BlazeTestStatus",
                note=(
                    "A timeout is a first-class status, numerically distinct "
                    "from FAILED (4). This is the strongest mainstream "
                    "build-system support for this repo's split."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "DISTINGUISHES",
                quote="INCOMPLETE = 5;",
                quote_context=(
                    "enum BlazeTestStatus { NO_STATUS = 0; PASSED = 1; "
                    "FLAKY = 2; TIMEOUT = 3; FAILED = 4; INCOMPLETE = 5; "
                    "REMOTE_FAILURE = 6; FAILED_TO_BUILD = 7; "
                    "BLAZE_HALTED_BEFORE_TESTING = 8; }"
                ),
                locator="src/main/protobuf/test_status.proto, enum BlazeTestStatus",
                note=(
                    "The same enum separates INCOMPLETE, REMOTE_FAILURE and "
                    "FAILED_TO_BUILD from FAILED -- the closest published "
                    "analogue to 'the subject broke' vs 'the measurement was "
                    "not obtained'. The per-value comments are absent from the "
                    "current proto, so their intended semantics are NOT "
                    "retrieved here."
                ),
            ),
            "Q2_STANDARD_NAME": _cell(
                "AGREES",
                quote="TIMEOUT = 3;",
                quote_context=(
                    "enum BlazeTestStatus { ... TIMEOUT = 3; FAILED = 4; ... }"
                ),
                locator="src/main/protobuf/test_status.proto, enum BlazeTestStatus",
                note=(
                    "The published status name is exactly the word this repo "
                    "uses: TIMEOUT."
                ),
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "NOT-STATED",
                locator=(
                    "src/main/protobuf/test_status.proto -- the enum carries no "
                    "budget guidance"
                ),
                note="See SRC-02 for Bazel's size-tiered default timeouts.",
            ),
        },
    ),
    _source(
        "SRC-04",
        "CMake / CTest -- TIMEOUT test property",
        "test-harness",
        "https://cmake.org/cmake/help/latest/prop_test/TIMEOUT.html",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "DISTINGUISHES",
                quote=(
                    "If it exceeds that the test process will be killed and "
                    "ctest will move to the next test."
                ),
                quote_context=(
                    "\"How many seconds to allow for this test.\" \"This "
                    "property if set will limit a test to not take more than "
                    "the specified number of seconds to run.\" \"If it exceeds "
                    "that the test process will be killed and ctest will move "
                    "to the next test.\" \"This setting takes precedence over "
                    "CTEST_TEST_TIMEOUT.\""
                ),
                locator="prop_test/TIMEOUT.html, property description",
                note=(
                    "The timeout gets its OWN handling path (kill the process, "
                    "then continue with the next test) rather than being "
                    "treated as an ordinary assertion failure. Whether CTest's "
                    "summary counts it as failed, and whether it labels the "
                    "outcome 'Timeout', was NOT-RETRIEVED from a primary page "
                    "in this session -- the ctest(1) man page does not state "
                    "it and the only description found was in a "
                    "search-engine summary, which is therefore not quoted."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "NOT-STATED",
                locator=(
                    "prop_test/TIMEOUT.html -- describes the kill-and-continue "
                    "behaviour only; no broken-vs-unmeasured vocabulary"
                ),
                note="No such distinction appears on this page.",
            ),
            "Q2_STANDARD_NAME": _cell(
                "NOT-STATED",
                locator=(
                    "prop_test/TIMEOUT.html -- the KNOB is named TIMEOUT, but "
                    "no per-test outcome label is named"
                ),
                note=(
                    "TIMEOUT here names the budget property, not an outcome "
                    "class."
                ),
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "NOT-RETRIEVED",
                locator=(
                    "prop_test/TIMEOUT.html -- states a PRECEDENCE rule "
                    "(TIMEOUT property > CTEST_TEST_TIMEOUT) but no sizing "
                    "rule; the 1500 s default appeared only in a "
                    "search-engine summary and is therefore not quoted"
                ),
                note="Retrieval gap, not evidence of absence.",
            ),
        },
    ),
    _source(
        "SRC-05",
        "Go toolchain -- `go test -timeout` flag documentation",
        "test-harness",
        "https://pkg.go.dev/cmd/go",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "DISAGREES",
                quote="If a test binary runs longer than duration d, panic.",
                quote_context=(
                    "\"-timeout d / If a test binary runs longer than duration "
                    "d, panic.\" and \"If d is 0, the timeout is disabled.\" and "
                    "\"The default is 10 minutes (10m).\""
                ),
                locator="section 'Testing flags' > '-timeout d'",
                note=(
                    "The timeout aborts the whole test binary; there is no "
                    "per-test timeout outcome class."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "NOT-STATED",
                locator="section 'Testing flags' -- no such distinction stated",
                note="No such distinction appears in the retrieved text.",
            ),
            "Q2_STANDARD_NAME": _cell(
                "NOT-STATED",
                locator=(
                    "section 'Testing flags' -- the flag is named '-timeout' "
                    "but no outcome-class name is given"
                ),
                note="See SRC-06 for the runtime wording.",
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "DISTINGUISHES",
                quote="The default is 10 minutes (10m).",
                quote_context=(
                    "\"-timeout d / If a test binary runs longer than duration "
                    "d, panic.\" and \"The default is 10 minutes (10m).\""
                ),
                locator="section 'Testing flags' > '-timeout d'",
                note=(
                    "A flat default for every package; no rule relating the "
                    "budget to an observed cost."
                ),
            ),
        },
    ),
    _source(
        "SRC-06",
        "Go standard library -- testing package source (the timeout panic)",
        "test-harness",
        "https://go.dev/src/testing/testing.go",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "DISTINGUISHES",
                quote='panic("test timed out after " + timeout.String())',
                quote_context=(
                    "In func (m *M) Run() (code int): \"m.deadline = "
                    "time.Now().Add(*timeout)\" ... \"<-time.After(*timeout)\" "
                    "... " + 'panic("test timed out after " + timeout.String())'
                ),
                locator="src/testing/testing.go, func (m *M) Run() (code int) -- the watchdog goroutine",
                note=(
                    "The verdict text is NOT a bare 'fail': the panic names the "
                    "timeout. But the process aborts and the package is "
                    "reported as failed, so the timeout never becomes its own "
                    "outcome class."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "NOT-STATED",
                locator=(
                    "src/testing/testing.go -- the panic text names the "
                    "timeout but nothing states that the subject was not "
                    "judged"
                ),
                note="No such distinction appears in the retrieved text.",
            ),
            "Q2_STANDARD_NAME": _cell(
                "AGREES",
                quote='panic("test timed out after " + timeout.String())',
                quote_context=(
                    "panic(\"test timed out after \" + timeout.String())"
                ),
                locator="src/testing/testing.go, func (m *M) Run()",
                note=(
                    "The only published name on this surface is the panic text "
                    "'test timed out after <d>' -- a message, not an outcome "
                    "label."
                ),
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "NOT-STATED",
                locator="src/testing/testing.go -- no budget-sizing rule",
                note="See SRC-05 for the flat 10-minute default.",
            ),
        },
    ),
    _source(
        "SRC-07",
        "GitHub Actions -- jobs.<job_id>.timeout-minutes",
        "ci-platform",
        "https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "AGREES",
                quote=(
                    "The maximum number of minutes to let a job run before "
                    "GitHub automatically cancels it. Default: 360"
                ),
                quote_context=(
                    "\"The maximum number of minutes to let a job run before "
                    "GitHub automatically cancels it. Default: 360\" and \"the "
                    "job will be canceled when the execution time limit is met "
                    "instead.\""
                ),
                locator=(
                    "section 'jobs.<job_id>.timeout-minutes' (anchor "
                    "#jobsjob_idtimeout-minutes)"
                ),
                note=(
                    "'automatically cancels it' -- cancellation, not failure. "
                    "The machine-readable form of that distinction is in "
                    "SRC-08."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "NOT-STATED",
                locator=(
                    "section 'jobs.<job_id>.timeout-minutes' -- no statement "
                    "that a cancelled job says nothing about the code"
                ),
                note=(
                    "The docs do not draw the broken-vs-unmeasured line; they "
                    "only report cancellation. (Caveat: 'cancelled' is also "
                    "GitHub's status for a MANUAL cancellation, so the "
                    "conclusion alone is not conclusive evidence about a "
                    "timeout -- see SRC-08 for the timeout-specific value.)"
                ),
            ),
            "Q2_STANDARD_NAME": _cell(
                "NOT-STATED",
                locator=(
                    "section 'jobs.<job_id>.timeout-minutes' -- the wording is "
                    "'cancels'/'canceled'; the machine-readable status name is "
                    "in SRC-08"
                ),
                note="See SRC-08 for `timed_out`.",
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "DISTINGUISHES",
                quote="Default: 360",
                quote_context=(
                    "\"The maximum number of minutes to let a job run before "
                    "GitHub automatically cancels it. Default: 360\""
                ),
                locator="section 'jobs.<job_id>.timeout-minutes'",
                note=(
                    "A flat platform default (360 minutes); no "
                    "observed-cost-relative rule."
                ),
            ),
        },
    ),
    _source(
        "SRC-08",
        "GitHub REST API -- Get a job for a workflow run (conclusion enum)",
        "ci-platform",
        "https://docs.github.com/en/rest/actions/workflow-jobs",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "AGREES",
                quote=(
                    "`success`, `failure`, `neutral`, `cancelled`, `skipped`, "
                    "`timed_out`, `action_required`, `null`"
                ),
                quote_context=(
                    "Conclusion values: \"`success`, `failure`, `neutral`, "
                    "`cancelled`, `skipped`, `timed_out`, `action_required`, "
                    "`null`\" Status values: \"`queued`, `in_progress`, "
                    "`completed`, `waiting`, `requested`, `pending`\""
                ),
                locator="REST API 'Get a job for a workflow run' -- job.conclusion schema",
                note=(
                    "A timeout is reported under its OWN conclusion "
                    "`timed_out`, not under `failure`. This is the single "
                    "clearest piece of mainstream CI support for the repo's "
                    "split."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "DISTINGUISHES",
                quote="`timed_out`",
                quote_context=(
                    "Conclusion values: \"`success`, `failure`, `neutral`, "
                    "`cancelled`, `skipped`, `timed_out`, `action_required`, "
                    "`null`\""
                ),
                locator="REST API 'Get a job for a workflow run' -- job.conclusion schema",
                note=(
                    "The vocabulary can express 'the run did not produce a "
                    "result' without asserting a defect, but the page does "
                    "not spell out the semantics of each value."
                ),
            ),
            "Q2_STANDARD_NAME": _cell(
                "AGREES",
                quote="`timed_out`",
                quote_context=(
                    "Conclusion values include \"`timed_out`\" alongside "
                    "\"`failure`\" and \"`cancelled`\"."
                ),
                locator="REST API 'Get a job for a workflow run' -- job.conclusion schema",
                note=(
                    "The published status name is `timed_out` -- the same word "
                    "this repo uses, in snake_case."
                ),
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "NOT-STATED",
                locator=(
                    "REST API workflow-jobs page -- the schema states no "
                    "budget-sizing rule (the platform default is in SRC-07)"
                ),
                note="See SRC-07 for the flat 360-minute default.",
            ),
        },
    ),
    _source(
        "SRC-09",
        "GitLab CI -- the job `timeout` keyword",
        "ci-platform",
        "https://docs.gitlab.com/ci/yaml/",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "DISTINGUISHES",
                quote=(
                    "If a job exceeds its timeout, it fails with the reason "
                    "`execution expired`."
                ),
                quote_context=(
                    "\"Define a custom job-level timeout that takes precedence "
                    "over the project-wide setting.\" and \"If a job exceeds "
                    "its timeout, it fails with the reason `execution "
                    "expired`.\""
                ),
                locator="section 'timeout' (job keyword)",
                note=(
                    "GitLab collapses the timeout into a FAILURE but keeps a "
                    "distinct, named reason string -- a compromise position "
                    "between 'distinct outcome class' and 'plain failure'."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "NOT-STATED",
                locator=(
                    "section 'timeout' -- a reason string is given but no "
                    "statement that the job's subject was never judged"
                ),
                note="No such distinction appears in the retrieved text.",
            ),
            "Q2_STANDARD_NAME": _cell(
                "AGREES",
                quote="`execution expired`",
                quote_context=(
                    "\"If a job exceeds its timeout, it fails with the reason "
                    "`execution expired`.\""
                ),
                locator="section 'timeout' (job keyword)",
                note=(
                    "A distinct, named reason -- but nested UNDER a failed "
                    "job, so the name alone does not split the outcome classes."
                ),
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "NOT-STATED",
                locator=(
                    "section 'timeout' -- states precedence over the "
                    "project-wide setting only"
                ),
                note="No sizing rule retrieved.",
            ),
        },
    ),
    _source(
        "SRC-10",
        "JUnit XML wire format -- outcome elements, as quoted by the pytest docs",
        "report-format",
        "https://docs.pytest.org/en/stable/how-to/output.html",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "DISTINGUISHES",
                quote='<xs:element ref="error" minOccurs="0" maxOccurs="unbounded"/>',
                quote_context=(
                    "The JUnit schema quoted by pytest lists three outcome "
                    "children of <testcase>: "
                    + '<xs:element ref="failure" minOccurs="0" maxOccurs="unbounded"/>'
                    + ", "
                    + '<xs:element ref="error" minOccurs="0" maxOccurs="unbounded"/>'
                    + ", "
                    + '<xs:element ref="skipped" minOccurs="0" maxOccurs="1"/>'
                ),
                locator="section 'Creating JUnitXML format files' (the embedded Jenkins JUnit schema)",
                note=(
                    "The wire format has three distinct outcome elements. The "
                    "retrieved text does NOT state which element a TIMEOUT "
                    "maps to -- that sub-claim is NOT-RETRIEVED here and is "
                    "not asserted."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "AGREES",
                quote='<xs:element ref="skipped" minOccurs="0" maxOccurs="1"/>',
                quote_context=(
                    "Distinct outcome elements of <testcase>: \"failure\", "
                    "\"error\" and \"skipped\" -- i.e. an assertion failure, an "
                    "error, and 'not run' are three different things. As "
                    "quoted by pytest: "
                    + '<xs:element ref="skipped" minOccurs="0" maxOccurs="1"/>'
                ),
                locator="section 'Creating JUnitXML format files' (the embedded Jenkins JUnit schema)",
                note=(
                    "`error` (the run broke) vs `failure` (an assertion "
                    "failed) vs `skipped` (no result) is the published "
                    "vocabulary for separating 'the subject is broken' from "
                    "'no measurement was obtained'."
                ),
            ),
            "Q2_STANDARD_NAME": _cell(
                "AGREES",
                quote='<xs:element ref="error" minOccurs="0" maxOccurs="unbounded"/>',
                quote_context=(
                    "The retrieved schema's outcome elements are \"failure\", "
                    "\"error\" and \"skipped\". As quoted by pytest: "
                    + '<xs:element ref="error" minOccurs="0" maxOccurs="unbounded"/>'
                ),
                locator="section 'Creating JUnitXML format files' (the embedded Jenkins JUnit schema)",
                note=(
                    "The standard names are `error` and `skipped` (plus "
                    "`failure`); there is NO `timeout` element in the "
                    "retrieved schema."
                ),
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "NOT-STATED",
                locator=(
                    "section 'Creating JUnitXML format files' -- a wire format "
                    "says nothing about budgets"
                ),
                note="No sizing rule retrieved.",
            ),
        },
    ),
    _source(
        "SRC-11",
        "Kubernetes -- Pod lifecycle (phase `Failed`, reason `DeadlineExceeded`)",
        "infra-platform",
        "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "DISTINGUISHES",
                quote=(
                    "The `reason` could be `DeadlineExceeded` if the Pod's "
                    "`activeDeadlineSeconds` was exceeded."
                ),
                quote_context=(
                    "Pod phase `Failed`: \"All containers in the Pod have "
                    "terminated, and at least one container has terminated in "
                    "failure. That is, the container either exited with "
                    "non-zero status or was terminated by the system, and is "
                    "not set for automatic restarting.\" and \"The `reason` "
                    "could be `DeadlineExceeded` if the Pod's "
                    "`activeDeadlineSeconds` was exceeded.\""
                ),
                locator="section 'Pod phase' > 'Failed'",
                note=(
                    "Phase-level `Failed` with a reason-level "
                    "`DeadlineExceeded`: the deadline is named, but it lives "
                    "inside the failure vocabulary."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "NOT-STATED",
                locator=(
                    "section 'Pod phase' > 'Failed' -- the reason is named but "
                    "no statement says the workload was never judged"
                ),
                note="No such distinction appears in the retrieved text.",
            ),
            "Q2_STANDARD_NAME": _cell(
                "AGREES",
                quote="`DeadlineExceeded`",
                quote_context=(
                    "\"The `reason` could be `DeadlineExceeded` if the Pod's "
                    "`activeDeadlineSeconds` was exceeded.\""
                ),
                locator="section 'Pod phase' > 'Failed'",
                note="A named reason inside the failure vocabulary.",
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "NOT-STATED",
                locator=(
                    "section 'Pod phase' -- `activeDeadlineSeconds` is "
                    "described, but no rule relates it to an observed cost"
                ),
                note="No sizing rule retrieved.",
            ),
        },
    ),
    _source(
        "SRC-12",
        "AWS Builders' Library -- 'Timeouts, retries, and backoff with jitter'",
        "practice",
        "https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "DISTINGUISHES",
                quote=(
                    "Many kinds of failures become apparent as requests taking "
                    "longer than usual, and potentially never completing."
                ),
                quote_context=(
                    "\"Timeouts are the maximum amount of time that a client "
                    "waits for a request to complete.\" and \"Many kinds of "
                    "failures become apparent as requests taking longer than "
                    "usual, and potentially never completing.\""
                ),
                locator="section 'Timeouts'",
                note=(
                    "A timeout is treated as one KIND OF FAILURE -- but the "
                    "same section introduces FALSE timeouts, i.e. timeouts "
                    "that are not failures at all (see Q1)."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "AGREES",
                quote="choose an acceptable rate of false timeouts (such as 0.1%).",
                quote_context=(
                    "\"start with the latency metrics of the downstream "
                    "service.\" \"choose an acceptable rate of false timeouts "
                    "(such as 0.1%).\" \"the corresponding latency percentile "
                    "on the downstream service (p99.9 in this example).\""
                ),
                locator="section 'Timeouts' (choosing the timeout)",
                note=(
                    "'false timeout' is the published name for exactly the "
                    "split this repo encodes: the client gave up before the "
                    "operation could be judged."
                ),
            ),
            "Q2_STANDARD_NAME": _cell(
                "AGREES",
                quote="choose an acceptable rate of false timeouts (such as 0.1%).",
                quote_context=(
                    "\"choose an acceptable rate of false timeouts (such as "
                    "0.1%).\""
                ),
                locator="section 'Timeouts' (choosing the timeout)",
                note="'false timeout'.",
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "AGREES",
                quote=(
                    "the corresponding latency percentile on the downstream "
                    "service (p99.9 in this example)."
                ),
                quote_context=(
                    "\"start with the latency metrics of the downstream "
                    "service.\" \"choose an acceptable rate of false timeouts "
                    "(such as 0.1%).\" \"the corresponding latency percentile "
                    "on the downstream service (p99.9 in this example).\" "
                    "\"adding some padding helps us avoid small latency "
                    "increases that cause high numbers of timeouts.\""
                ),
                locator="section 'Timeouts' (choosing the timeout)",
                note=(
                    "The only source in this corpus that states an "
                    "OBSERVED-COST-RELATIVE rule: pick a latency PERCENTILE, "
                    "declare the tolerated FALSE-TIMEOUT rate, and add "
                    "PADDING so a load spike does not manufacture timeouts."
                ),
            ),
        },
    ),
    _source(
        "SRC-13",
        "arXiv:2412.17859 -- 'The Unreasonable Effectiveness of Open Science in AI: A Replication Study'",
        "meta-science",
        "https://arxiv.org/html/2412.17859v1",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "AGREES",
                quote=(
                    "If none of the reproducibility experiments was completed "
                    "successfully, then the result of the reproducibility "
                    "study is classified as No Result."
                ),
                quote_context=(
                    "\"If none of the reproducibility experiments was completed "
                    "successfully, then the result of the reproducibility "
                    "study is classified as No Result.\" Contrast: \"If all "
                    "reproducibility experiments of a study failed, then the "
                    "reproducibility study is classified as Failure.\""
                ),
                locator=(
                    "§6 'Classifying Results' > 'Study Result Classification' "
                    "(https://arxiv.org/html/2412.17859v1#Sx5.SSx2)"
                ),
                note=(
                    "A budget-exhausted reproduction attempt is No Result, not "
                    "Failure. §7: \"we did not reproduce six and obtained no "
                    "result for five\"."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "AGREES",
                quote=(
                    "Some reproducibility studies were aborted before all "
                    "experiments were attempted, or the time limit was reached."
                ),
                quote_context=(
                    "\"Some reproducibility studies were aborted before all "
                    "experiments were attempted, or the time limit was "
                    "reached.\" \"We also aborted reproducibility studies where "
                    "the hardware demand exceeded what was available to us.\" "
                    "and \"the articles did not provide enough information for "
                    "us to reproduce them.\""
                ),
                locator=(
                    "§5 'The Replication Study' "
                    "(https://arxiv.org/html/2412.17859v1#Sx4)"
                ),
                note=(
                    "Three distinct 'no measurement' causes -- time budget, "
                    "hardware, missing information -- none of which is a "
                    "failure of the artifact under reproduction."
                ),
            ),
            "Q2_STANDARD_NAME": _cell(
                "AGREES",
                quote="No Result (NR), and Not Started (NS).",
                quote_context=(
                    "Table 1 caption: \"Results of the reproducibility studies "
                    "can be Success (S), Partial Success (PS), Failed (F)\" ... "
                    "\"No Result (NR), and Not Started (NS).\" and §7: \"we did "
                    "not reproduce six and obtained no result for five\""
                ),
                locator="Table 1 caption (§1) and §7 'Results'",
                note=(
                    "'No Result' is the published term for 'we did not get a "
                    "measurement' in the reproduction literature."
                ),
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "AGREES",
                quote=(
                    "The maximum time we spent on a reproducibility study was "
                    "40 hours of focused work."
                ),
                quote_context=(
                    "\"The maximum time we spent on a reproducibility study was "
                    "40 hours of focused work.\" \"we spent all available time "
                    "(40 hours) for 12 studies without completing all "
                    "experiments.\" \"Some reproducibility studies were aborted "
                    "before all experiments were attempted, or the time limit "
                    "was reached.\""
                ),
                locator="§5 'The Replication Study' and §7 'Results'",
                note=(
                    "A per-study wall budget, published in advance; and the "
                    "consequence policy when the budget IS exhausted is "
                    "No Result, not Failure."
                ),
            ),
        },
    ),
    _source(
        "SRC-14",
        "SWE-bench harness -- swebench.harness.infra_failure",
        "ml-harness",
        "https://www.swebench.com/SWE-bench/api/harness",
        "WebFetch",
        {
            "Q0_TIMEOUT_NOT_DEFECT_CLAIM": _cell(
                "AGREES",
                quote=(
                    'Separates "the environment broke" from "the model\'s patch '
                    'was wrong"'
                ),
                quote_context=(
                    "Module docstring: \"Post-hoc classification of "
                    "infrastructure failures (#586).\" and "
                    + 'Separates "the environment broke" from "the model\'s patch was wrong"'
                ),
                locator="module swebench.harness.infra_failure (Harness API docs)",
                note=(
                    "A published ML evaluation harness that separates "
                    "environment failure from subject failure."
                ),
            ),
            "Q1_BROKEN_VS_UNMEASURED": _cell(
                "AGREES",
                quote="It is advisory.",
                quote_context=(
                    "\"Classification is post-hoc and read-only.\" \"It never "
                    "runs commands in a container and never decides whether an "
                    "instance gets evaluated\" \"so it cannot drop an instance "
                    "from a run.\" \"It is advisory.\" \"A flagged instance "
                    "stays in `unresolved_ids` / `error_ids` exactly as before, "
                    "so the scoring denominator is unchanged.\""
                ),
                locator="module swebench.harness.infra_failure -- advisory behaviour",
                note=(
                    "Note the deliberate asymmetry with this repo: SWE-bench "
                    "keeps the flagged instance INSIDE the failure "
                    "denominator and only annotates it, whereas this repo "
                    "removes the timeout from the defect verdict entirely."
                ),
            ),
            "Q2_STANDARD_NAME": _cell(
                "AGREES",
                quote="TIER_ENVIRONMENT = 'environment'",
                quote_context=(
                    "\"TIER_ENVIRONMENT = 'environment'\" and "
                    "\"TIER_AMBIGUOUS = 'ambiguous'\" and the signature tuple "
                    "\"('tests_timed_out', TIER_AMBIGUOUS, 'Timeout error: "
                    "\\d+ seconds exceeded')\""
                ),
                locator=(
                    "module swebench.harness.infra_failure -- TIER_ENVIRONMENT "
                    "/ TIER_AMBIGUOUS / INFRA_FAILURE_SIGNATURES"
                ),
                note=(
                    "The published term is 'infrastructure failure', with an "
                    "explicit 'ambiguous' tier for a timeout."
                ),
            ),
            "Q3_BUDGET_VS_OBSERVED_COST": _cell(
                "NOT-STATED",
                locator=(
                    "module swebench.harness.infra_failure -- classifies "
                    "outcomes post hoc; no budget-sizing rule"
                ),
                note="No sizing rule retrieved.",
            ),
        },
    ),
]

#: What was extracted from each URL in this session.  One entry per URL,
#: in retrieval order.
RETRIEVAL_LOG: List[Dict[str, str]] = [
    {
        "url": "https://bazel.build/reference/test-encyclopedia",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "Section 'Role of the test runner': the kill-and-treat-as-failure "
            "rule, 'Any other result is considered a test failure.', and the "
            "size-derived default timeouts (60/300/900/3600 s)."
        ),
    },
    {
        "url": "https://pkg.go.dev/testing",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "Only T.Deadline was in the retrieved window; the -timeout flag "
            "text was NOT found here and was taken from pkg.go.dev/cmd/go."
        ),
    },
    {
        "url": "https://pkg.go.dev/cmd/go",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "Section 'Testing flags': '-timeout d / If a test binary runs "
            "longer than duration d, panic.', 'If d is 0, the timeout is "
            "disabled.', 'The default is 10 minutes (10m).'"
        ),
    },
    {
        "url": "https://bazel.build/reference/be/common-definitions",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "No TestResult.Status enum on this page -- the enum was instead "
            "retrieved from the proto source (below)."
        ),
    },
    {
        "url": "https://go.dev/src/testing/testing.go",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "func (m *M) Run(): the deadline assignment, the time.After wait, "
            "and the timeout panic string."
        ),
    },
    {
        "url": "https://cmake.org/cmake/help/latest/prop_test/TIMEOUT.html",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "The TIMEOUT property description, including 'If it exceeds that "
            "the test process will be killed and ctest will move to the next "
            "test.' and the CTEST_TEST_TIMEOUT precedence sentence."
        ),
    },
    {
        "url": "https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "Section 'jobs.<job_id>.timeout-minutes' (anchor "
            "#jobsjob_idtimeout-minutes): the 360-minute default and the "
            "'automatically cancels it' wording."
        ),
    },
    {
        "url": "https://pytest-timeout.readthedocs.io/en/latest/",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "Sections 'Timeout Methods' (thread, signal), 'Specifying the "
            "Timeout Method', 'Session Timeout', 'Timeouts in Fixture "
            "Teardown'. A second fetch of the same URL in this session "
            "returned HTTP 404, so the FAQ sizing guidance is NOT quoted."
        ),
    },
    {
        "url": "https://docs.github.com/en/rest/actions/workflow-jobs",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "The job.conclusion schema: success / failure / neutral / "
            "cancelled / skipped / timed_out / action_required / null, and the "
            "job.status values."
        ),
    },
    {
        "url": "https://docs.gitlab.com/ci/yaml/",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "The job `timeout` keyword: 'If a job exceeds its timeout, it "
            "fails with the reason `execution expired`.'"
        ),
    },
    {
        "url": "https://docs.pytest.org/en/stable/how-to/output.html",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "Section 'Creating JUnitXML format files': the embedded Jenkins "
            "JUnit schema with the failure / error / skipped testcase "
            "children."
        ),
    },
    {
        "url": "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "Section 'Pod phase' > 'Failed': the phase definition and the "
            "`DeadlineExceeded` reason sentence."
        ),
    },
    {
        "url": "https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "Section 'Timeouts': the false-timeout rate (0.1%), the latency "
            "percentile (p99.9) and the padding sentence."
        ),
    },
    {
        "url": "https://arxiv.org/html/2412.17859v1",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "Table 1 caption (Success / Partial Success / Failed / No Result / "
            "Not Started), §5 (the 40-hour budget and the abort causes), §6 "
            "'Classifying Results' (the Failure vs No Result definitions), §7 "
            "(the aggregate counts)."
        ),
    },
    {
        "url": "https://www.swebench.com/SWE-bench/api/harness",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "The swebench.harness.infra_failure module docstring, the "
            "TIER_ENVIRONMENT / TIER_AMBIGUOUS constants, the "
            "INFRA_FAILURE_SIGNATURES tuples (including tests_timed_out), and "
            "the advisory / denominator-preserving behaviour."
        ),
    },
    {
        "url": "https://bazel.googlesource.com/bazel/+/HEAD/src/main/protobuf/test_status.proto",
        "retrieved_at": SESSION_DATE,
        "tool": "WebFetch",
        "extracted": (
            "enum BlazeTestStatus with all nine values and the enum-level "
            "comment."
        ),
    },
]

# ---------------------------------------------------------------------------
# FINDINGS -- the three answers the branch brief asked for
# ---------------------------------------------------------------------------

FINDINGS: List[Dict[str, Any]] = [
    {
        "id": "F1",
        "question": (
            "Does any published practice distinguish 'the subject is broken' "
            "from 'we did not get a measurement'?"
        ),
        "answer": "YES, in part of the corpus -- but NOT in the mainstream test-runner layer.",
        "supporting": [
            "SRC-14",
            "SRC-13",
            "SRC-12",
            "SRC-10",
            "SRC-03",
        ],
        "contradicting": ["SRC-01", "SRC-02", "SRC-05", "SRC-09"],
        "note": (
            "The distinction IS published -- 'infrastructure failure' vs the "
            "subject's failure (SRC-14), 'No Result' vs 'Failure' (SRC-13), "
            "'false timeout' (SRC-12), the JUnit error/failure/skipped triple "
            "(SRC-10), and Bazel's INCOMPLETE / REMOTE_FAILURE / "
            "FAILED_TO_BUILD (SRC-03). But the tools most people actually run "
            "-- pytest-timeout (SRC-01), Bazel's build-level failure rule "
            "(SRC-02), Go's panic-the-binary rule (SRC-05) -- collapse a "
            "timeout into a failure."
        ),
    },
    {
        "id": "F2",
        "question": "What is the STANDARD NAME for the distinction?",
        "answer": (
            "There is NO single de-facto standard across this corpus; the "
            "closest general-purpose terms are 'No Result' (reproduction "
            "literature) and 'infrastructure failure' (benchmark harnesses)."
        ),
        "supporting": ["SRC-13", "SRC-14", "SRC-12", "SRC-08", "SRC-03"],
        "contradicting": [],
        "note": (
            "Retrieved names, each from its own source: `timed_out` "
            "(GitHub Actions conclusion, SRC-08), `TIMEOUT` (Bazel Build Event "
            "Protocol status, SRC-03), 'execution expired' (GitLab reason, "
            "SRC-09), `DeadlineExceeded` (Kubernetes reason, SRC-11), 'error' "
            "and 'skipped' (JUnit elements, SRC-10), 'No Result' (SRC-13), "
            "'infrastructure failure' / 'ambiguous' (SWE-bench, SRC-14), "
            "'false timeout' (AWS, SRC-12). The repo's status word TIMEOUT "
            "coincides exactly with Bazel's published status name and with "
            "GitHub's `timed_out` in snake_case, so adopting it in the "
            "evidence text is defensible."
        ),
    },
    {
        "id": "F3",
        "question": (
            "Is there published practice on the BUDGET question -- how the "
            "budget should be set relative to the observed cost, and how to "
            "avoid a marginal budget turning a load spike into a false "
            "verdict?"
        ),
        "answer": (
            "YES, but only ONE source in this corpus states an "
            "observed-cost-relative rule; the rest state flat or size-tiered "
            "defaults."
        ),
        "supporting": ["SRC-12", "SRC-13", "SRC-02"],
        "contradicting": [],
        "note": (
            "SRC-12 (AWS) is the rule: start from the downstream latency "
            "metrics, pick a percentile (p99.9 in its example), declare the "
            "tolerated false-timeout rate (0.1% in its example), and add "
            "padding so a small latency increase does not manufacture "
            "timeouts. SRC-02 (Bazel) keys the default to a declared size "
            "tier; SRC-05 (Go, flat 10 m) and SRC-07 (GitHub, flat 360 m) "
            "state flat defaults; none of those three references an observed "
            "cost. SRC-13 shows the consequence policy when a budget IS "
            "exhausted: record No Result, not Failure."
        ),
    },
]

#: The honest headline.  NOT 'the literature validates our design'.
CONCLUSION = (
    "The repo's TIMEOUT-is-not-a-FAIL split is a DELIBERATE DEVIATION from the "
    "majority convention of mainstream test runners, and it is ALIGNED with "
    "the vocabulary-aware tail of the same literature. Of the 14 retrieved "
    "sources, the ones that speak to the question directly split: pytest-"
    "timeout (SRC-01), Bazel's build-level rule (SRC-02) and Go's panic "
    "(SRC-05) DISAGREE -- a timeout is a failure; GitHub Actions' `timed_out` "
    "conclusion (SRC-08), Bazel's Build Event Protocol TIMEOUT status "
    "(SRC-03), the replication literature's 'No Result' class (SRC-13) and "
    "SWE-bench's 'infrastructure failure' classifier (SRC-14) AGREE. The "
    "honest framing is 'a defensible deviation, consistent with the newer "
    "and harness-aware part of the literature', NOT 'the literature agrees "
    "with us'."
)

CAVEATS: List[str] = [
    "Retrieval dates are recorded at DATE granularity (the session date, "
    "2026-11-01): the WebFetch tool exposes no per-call clock.",
    "Quotations are the exact strings returned by the fetch tool's "
    "extraction, not re-reads of the page by eye; a second fetch of a page "
    "can legitimately return a different rendering.",
    "Three sub-claims are retrieval gaps, NOT evidence of absence: (a) "
    "pytest-timeout's FAQ sizing guidance (a second fetch of the same URL "
    "returned HTTP 404); (b) whether CTest labels a timed-out test "
    "'Timeout' and counts it as failed (the ctest(1) man page does not say, "
    "and the only description found was in a search-engine summary, which "
    "is therefore not quoted); (c) CTest's 1500 s default, for the same "
    "reason.",
    "Tool documentation is used as evidence of a TOOL's semantics, which is "
    "the tool's own normative statement of its behaviour; that is not the "
    "same as an independent study of how users read the resulting numbers.",
    "Corpus composition is a convenience sample chosen by the branch brief "
    "(CI harnesses, one ML harness, one reproduction study, one "
    "infrastructure platform, one practice article). It is NOT a systematic "
    "survey, so the AGREES/DISAGREES counts below carry NO statistical "
    "weight and must not be quoted as 'X% of published practice'.",
    "Verdicts are INTERPRETATION over the quoted text. The quotes are "
    "MEASURED (retrieved live in this session); the verdicts are the "
    "analyst's reading of them.",
    "This branch did NOT test whether the repo's own TIMEOUT gates would be "
    "re-classified by any of these sources -- it only asks what the sources "
    "say. It did NOT re-run the pre-flight, did NOT touch any gate, and did "
    "NOT touch metrics/, registry/, docs/, TODO/ or paper/.",
]


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------

CELL_KEYS = (
    "verdict",
    "quote",
    "quote_context",
    "locator",
    "url",
    "retrieved_at",
    "note",
)


def _check(name: str, ok: bool, detail: str) -> Tuple[str, bool, str]:
    return (name, bool(ok), detail)


def self_check(matrix: Dict[str, Any]) -> List[Tuple[str, bool, str]]:
    """Run every invariant.  Returns a list of (name, ok, detail)."""
    checks: List[Tuple[str, bool, str]] = []
    sources = matrix["sources"]

    # C1 -- schema: every source and every cell has the required keys.
    missing: List[str] = []
    for src in sources:
        for key in ("id", "name", "kind", "url", "retrieved_at", "tool", "cells"):
            if key not in src:
                missing.append(f"{src.get('id', '?')}.{key}")
        for qid, cell in src["cells"].items():
            for key in CELL_KEYS:
                if key not in cell:
                    missing.append(f"{src.get('id', '?')}.{qid}.{key}")
    checks.append(
        _check(
            "C1 schema: every source/cell carries all required keys",
            not missing,
            f"{len(missing)} missing key(s)" + (f": {missing[:5]}" if missing else ""),
        )
    )

    # C2 -- verdict vocabulary.
    bad = [
        f"{s['id']}.{q}"
        for s in sources
        for q, c in s["cells"].items()
        if c["verdict"] not in VERDICTS
    ]
    checks.append(
        _check(
            "C2 every verdict is in the fixed vocabulary",
            not bad,
            f"{len(bad)} bad verdict(s)" + (f": {bad[:5]}" if bad else ""),
        )
    )

    # C3 -- every cell has a locator + a url + a retrieval date.
    no_locator = [
        f"{s['id']}.{q}"
        for s in sources
        for q, c in s["cells"].items()
        if not str(c["locator"]).strip()
    ]
    no_url = [
        f"{s['id']}.{q}"
        for s in sources
        for q, c in s["cells"].items()
        if not str(c["url"]).strip().startswith("http")
    ]
    no_date = [
        f"{s['id']}.{q}"
        for s in sources
        for q, c in s["cells"].items()
        if not str(c["retrieved_at"]).strip()
    ]
    checks.append(
        _check(
            "C3 every cell has a locator + url + retrieved_at",
            not (no_locator or no_url or no_date),
            f"locator-missing={len(no_locator)} url-missing={len(no_url)} "
            f"date-missing={len(no_date)}",
        )
    )

    # C4 -- a cell that claims a verdict MUST carry a quote, and the quote
    #      must be a substring of its own quote_context (no smuggled quote).
    bad_claim: List[str] = []
    for s in sources:
        for q, c in s["cells"].items():
            if c["verdict"] in CLAIMING_VERDICTS:
                if not c["quote"].strip():
                    bad_claim.append(f"{s['id']}.{q} (no quote)")
                elif c["quote"] not in c["quote_context"]:
                    bad_claim.append(f"{s['id']}.{q} (quote not in context)")
    checks.append(
        _check(
            "C4 every claiming cell carries a quote that it actually holds",
            not bad_claim,
            f"{len(bad_claim)} violation(s)" + (f": {bad_claim[:5]}" if bad_claim else ""),
        )
    )

    # C5 -- a cell that claims an ABSENCE must not carry a quote.
    bad_absence = [
        f"{s['id']}.{q}"
        for s in sources
        for q, c in s["cells"].items()
        if c["verdict"] in ABSENCE_VERDICTS and c["quote"].strip()
    ]
    checks.append(
        _check(
            "C5 a NOT-STATED / NOT-RETRIEVED cell carries no quote",
            not bad_absence,
            f"{len(bad_absence)} violation(s)"
            + (f": {bad_absence[:5]}" if bad_absence else ""),
        )
    )

    # C6 -- every source answers every question.
    incomplete = [
        s["id"] for s in sources if set(s["cells"]) != set(QUESTION_IDS)
    ]
    checks.append(
        _check(
            "C6 every source answers every question",
            not incomplete,
            f"{len(incomplete)} incomplete source(s)"
            + (f": {incomplete[:5]}" if incomplete else ""),
        )
    )

    # C7 -- the summary is DERIVED, not typed: recompute and compare.
    recomputed: Dict[str, Dict[str, int]] = {}
    for qid in QUESTION_IDS:
        counts = {v: 0 for v in VERDICTS}
        for s in sources:
            counts[s["cells"][qid]["verdict"]] += 1
        recomputed[qid] = counts
    checks.append(
        _check(
            "C7 the summary counts are derived from the cells",
            recomputed == matrix["summary"]["verdict_counts"],
            "recomputed == stored"
            if recomputed == matrix["summary"]["verdict_counts"]
            else "recomputed != stored",
        )
    )

    # C8 -- no torch / no CUDA brought in by THIS module, and no torch
    #      import in this file's source text.  The invariant is "no NEW
    #      torch module", not "torch absent": the repo's own pytest
    #      conftest imports torch before any test runs.
    src_text = Path(__file__).read_text(encoding="utf-8")
    torch_import = re.search(r"^\s*(import\s+torch|from\s+torch\s+import)", src_text, re.M)
    torch_now = frozenset(
        name for name in sys.modules if name == "torch" or name.startswith("torch.")
    )
    new_torch = sorted(torch_now - _TORCH_MODULES_AT_IMPORT)
    checks.append(
        _check(
            "C8 no torch import (source scan) and no NEW torch module loaded",
            torch_import is None and not new_torch,
            "clean"
            if torch_import is None and not new_torch
            else f"torch_import={bool(torch_import)} new_torch={new_torch[:3]}",
        )
    )

    # C9 -- every url used by a cell appears in the retrieval log.
    logged = {entry["url"] for entry in matrix["retrieval_log"]}
    unlogged = sorted(
        {c["url"] for s in sources for c in s["cells"].values()} - logged
    )
    checks.append(
        _check(
            "C9 every cell url appears in the retrieval log",
            not unlogged,
            f"{len(unlogged)} unlogged url(s)" + (f": {unlogged}" if unlogged else ""),
        )
    )

    # C10 -- the findings cite only sources that exist.
    known = {s["id"] for s in sources}
    dangling = sorted(
        {sid for f in matrix["findings"] for sid in f["supporting"] + f["contradicting"]}
        - known
    )
    checks.append(
        _check(
            "C10 every finding cites a source that exists in the matrix",
            not dangling,
            f"{len(dangling)} dangling citation(s)"
            + (f": {dangling}" if dangling else ""),
        )
    )

    return checks


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SHORT_QUOTE_LIMIT = 220


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _cell_brief(cell: Dict[str, Any]) -> str:
    if cell["verdict"] in ABSENCE_VERDICTS:
        return f"**{cell['verdict']}**"
    quote = _one_line(cell["quote"])
    if len(quote) > SHORT_QUOTE_LIMIT:
        quote = quote[: SHORT_QUOTE_LIMIT - 3] + "..."
    return f"**{cell['verdict']}** — “{quote}”"


def build_matrix() -> Dict[str, Any]:
    """Assemble the matrix.  Pure function of the module constants."""
    verdict_counts: Dict[str, Dict[str, int]] = {}
    for qid in QUESTION_IDS:
        counts = {v: 0 for v in VERDICTS}
        for src in SOURCES:
            counts[src["cells"][qid]["verdict"]] += 1
        verdict_counts[qid] = counts
    return {
        "wave": "Wf-A159",
        "branch": "F",
        "artifact": "timeout-verdict-frontier comparison matrix",
        "generated_by": "molmetal/scripts/wfa159_timeout_verdict_frontier.py",
        "session_date": SESSION_DATE,
        "retrieval_mode": (
            "LIVE WebSearch + WebFetch on 2026-11-01; every fact below is "
            "embedded as a literal constant so this script is offline and "
            "deterministic."
        ),
        "repo_rule_under_test": (
            "pre_flight_check._timeout_gate_result: a wall-clock timeout "
            "reports status TIMEOUT (not FAIL) because the measurement was "
            "NOT obtained, so it is not a verdict on the subject under test."
        ),
        "questions": QUESTIONS,
        "verdict_vocabulary": list(VERDICTS),
        "sources": SOURCES,
        "retrieval_log": RETRIEVAL_LOG,
        "findings": FINDINGS,
        "conclusion": CONCLUSION,
        "caveats": CAVEATS,
        "summary": {
            "n_sources": len(SOURCES),
            "n_cells": len(SOURCES) * len(QUESTION_IDS),
            "verdict_counts": verdict_counts,
        },
    }


def render_markdown(matrix: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append

    add("# Wf-A159 Branch F — timeout-verdict frontier: a COMPARISON MATRIX")
    add("")
    add(
        f"**Artifact** `{JSON_NAME}` + `{MD_NAME}` · **session** "
        f"{matrix['session_date']} · **sources** {matrix['summary']['n_sources']} "
        f"· **cells** {matrix['summary']['n_cells']}"
    )
    add("")
    add(
        "> **What this is.** A COMPARISON MATRIX of what published practice "
        "says about timeout-vs-failure semantics, retrieved live in the "
        f"{matrix['session_date']} session and embedded as a literal "
        "constant by `molmetal/scripts/wfa159_timeout_verdict_frontier.py` "
        "(offline, deterministic, CPU-only)."
    )
    add("")
    add(
        f"> **The rule under test.** {matrix['repo_rule_under_test']}"
    )
    add("")
    add(
        "> **Read this first.** A source that DISAGREES is a first-class "
        "result. The matrix does NOT conclude that the literature validates "
        "the repo's design."
    )
    add("")

    # ---- the matrix table
    add("## 1. The matrix")
    add("")
    header = "| Source | Kind | " + " | ".join(q["column"] for q in QUESTIONS) + " |"
    add(header)
    add("|" + "---|" * (2 + len(QUESTIONS)))
    for src in matrix["sources"]:
        cells = " | ".join(
            _cell_brief(src["cells"][q["id"]]) for q in QUESTIONS
        )
        add(f"| **{src['id']}** {src['name']} | {src['kind']} | {cells} |")
    add("")

    # ---- verdict counts
    add("### Verdict counts (per question)")
    add("")
    add("| Question | " + " | ".join(VERDICTS) + " |")
    add("|" + "---|" * (1 + len(VERDICTS)))
    for q in QUESTIONS:
        counts = matrix["summary"]["verdict_counts"][q["id"]]
        add(
            f"| {q['id']} | " + " | ".join(str(counts[v]) for v in VERDICTS) + " |"
        )
    add("")
    add(
        "> These counts are a census of the 14 sources in THIS corpus, which "
        "is a convenience sample chosen by the branch brief. They carry no "
        "statistical weight."
    )
    add("")

    # ---- per-question meaning
    add("### What each column's verdict means")
    add("")
    for q in QUESTIONS:
        add(f"- **`{q['id']}`** — {q['question']}")
        add(f"  - verdict rule: {q['verdict_rule']}")
    add("")

    # ---- per-source detail
    add("## 2. Per-source detail")
    add("")
    for src in matrix["sources"]:
        add(f"### {src['id']} — {src['name']}")
        add("")
        add(f"- **kind**: {src['kind']}")
        add(f"- **url**: {src['url']}")
        add(f"- **retrieved**: {src['retrieved_at']} via {src['tool']}")
        add("")
        for q in QUESTIONS:
            cell = src["cells"][q["id"]]
            add(f"- **{q['id']}** → **{cell['verdict']}**")
            add(f"  - locator: {cell['locator']}")
            if cell["verdict"] in ABSENCE_VERDICTS:
                add("  - quote: _(none — this cell asserts an absence)_")
            else:
                add(f"  - quote: “{_one_line(cell['quote'])}”")
            if cell["note"]:
                add(f"  - note: {cell['note']}")
        add("")

    # ---- findings
    add("## 3. Findings — the three questions the brief asked")
    add("")
    for f in matrix["findings"]:
        add(f"### {f['id']}. {f['question']}")
        add("")
        add(f"**{f['answer']}**")
        add("")
        add(f"- supporting sources: {', '.join(f['supporting'])}")
        if f["contradicting"]:
            add(f"- sources that go the other way: {', '.join(f['contradicting'])}")
        add(f"- {f['note']}")
        add("")

    add("## 4. Honest conclusion")
    add("")
    add(matrix["conclusion"])
    add("")

    # ---- retrieval log
    add("## 5. Retrieval log")
    add("")
    add("| # | url | retrieved_at | tool | what was extracted |")
    add("|---|---|---|---|---|")
    for i, entry in enumerate(matrix["retrieval_log"], start=1):
        add(
            f"| {i} | {entry['url']} | {entry['retrieved_at']} | "
            f"{entry['tool']} | {_one_line(entry['extracted'])} |"
        )
    add("")

    add("## 6. Caveats — what this branch did NOT establish")
    add("")
    for cav in matrix["caveats"]:
        add(f"- {cav}")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="directory for the JSON + markdown artifacts",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    matrix = build_matrix()

    checks = self_check(matrix)
    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} — {detail}")

    if matrix["summary"]["n_sources"] == 0:
        print("SKIP: empty corpus — nothing to check")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(matrix, indent=2, ensure_ascii=False) + "\n"
    md_text = render_markdown(matrix)
    atomic_write_file(out_dir / JSON_NAME, json_text)
    atomic_write_file(out_dir / MD_NAME, md_text)

    print(f"wrote {out_dir / JSON_NAME}")
    print(f"wrote {out_dir / MD_NAME}")

    if failed:
        print(f"FAIL: {len(failed)} of {len(checks)} self-checks failed")
        return 1
    print(
        f"PASS: {len(checks)}/{len(checks)} self-checks; "
        f"{matrix['summary']['n_sources']} sources, "
        f"{matrix['summary']['n_cells']} cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
