# Wf-A159 Branch F — timeout-verdict frontier: a COMPARISON MATRIX

**Artifact** `timeout_verdict_matrix.json` + `timeout_verdict_matrix.md` · **session** 2026-11-01 · **sources** 14 · **cells** 56

> **What this is.** A COMPARISON MATRIX of what published practice says about timeout-vs-failure semantics, retrieved live in the 2026-11-01 session and embedded as a literal constant by `molmetal/scripts/wfa159_timeout_verdict_frontier.py` (offline, deterministic, CPU-only).

> **The rule under test.** pre_flight_check._timeout_gate_result: a wall-clock timeout reports status TIMEOUT (not FAIL) because the measurement was NOT obtained, so it is not a verdict on the subject under test.

> **Read this first.** A source that DISAGREES is a first-class result. The matrix does NOT conclude that the literature validates the repo's design.

## 1. The matrix

| Source | Kind | Q0 timeout != defect claim | Q1 broken vs. unmeasured | Q2 standard name | Q3 budget vs. observed cost |
|---|---|---|---|---|---|
| **SRC-01** pytest-timeout (signal / thread timeout methods) | test-harness | **DISAGREES** — “the signal handler will dump the stack of any other threads running to stderr and use ``pytest.fail()`` to interrupt the test.” | **NOT-STATED** | **NOT-STATED** | **NOT-RETRIEVED** |
| **SRC-02** Bazel -- Test Encyclopedia ('Role of the test runner') | test-harness | **DISAGREES** — “it may choose to kill the test and treat the run as a failure.” | **NOT-STATED** | **NOT-STATED** | **DISTINGUISHES** — “Tests which do not explicitly specify a timeout have one implied based on the test's size as follows:” |
| **SRC-03** Bazel -- test_status.proto, enum BlazeTestStatus (Build Event Protocol) | build-protocol | **AGREES** — “TIMEOUT = 3;” | **DISTINGUISHES** — “INCOMPLETE = 5;” | **AGREES** — “TIMEOUT = 3;” | **NOT-STATED** |
| **SRC-04** CMake / CTest -- TIMEOUT test property | test-harness | **DISTINGUISHES** — “If it exceeds that the test process will be killed and ctest will move to the next test.” | **NOT-STATED** | **NOT-STATED** | **NOT-RETRIEVED** |
| **SRC-05** Go toolchain -- `go test -timeout` flag documentation | test-harness | **DISAGREES** — “If a test binary runs longer than duration d, panic.” | **NOT-STATED** | **NOT-STATED** | **DISTINGUISHES** — “The default is 10 minutes (10m).” |
| **SRC-06** Go standard library -- testing package source (the timeout panic) | test-harness | **DISTINGUISHES** — “panic("test timed out after " + timeout.String())” | **NOT-STATED** | **AGREES** — “panic("test timed out after " + timeout.String())” | **NOT-STATED** |
| **SRC-07** GitHub Actions -- jobs.<job_id>.timeout-minutes | ci-platform | **AGREES** — “The maximum number of minutes to let a job run before GitHub automatically cancels it. Default: 360” | **NOT-STATED** | **NOT-STATED** | **DISTINGUISHES** — “Default: 360” |
| **SRC-08** GitHub REST API -- Get a job for a workflow run (conclusion enum) | ci-platform | **AGREES** — “`success`, `failure`, `neutral`, `cancelled`, `skipped`, `timed_out`, `action_required`, `null`” | **DISTINGUISHES** — “`timed_out`” | **AGREES** — “`timed_out`” | **NOT-STATED** |
| **SRC-09** GitLab CI -- the job `timeout` keyword | ci-platform | **DISTINGUISHES** — “If a job exceeds its timeout, it fails with the reason `execution expired`.” | **NOT-STATED** | **AGREES** — “`execution expired`” | **NOT-STATED** |
| **SRC-10** JUnit XML wire format -- outcome elements, as quoted by the pytest docs | report-format | **DISTINGUISHES** — “<xs:element ref="error" minOccurs="0" maxOccurs="unbounded"/>” | **AGREES** — “<xs:element ref="skipped" minOccurs="0" maxOccurs="1"/>” | **AGREES** — “<xs:element ref="error" minOccurs="0" maxOccurs="unbounded"/>” | **NOT-STATED** |
| **SRC-11** Kubernetes -- Pod lifecycle (phase `Failed`, reason `DeadlineExceeded`) | infra-platform | **DISTINGUISHES** — “The `reason` could be `DeadlineExceeded` if the Pod's `activeDeadlineSeconds` was exceeded.” | **NOT-STATED** | **AGREES** — “`DeadlineExceeded`” | **NOT-STATED** |
| **SRC-12** AWS Builders' Library -- 'Timeouts, retries, and backoff with jitter' | practice | **DISTINGUISHES** — “Many kinds of failures become apparent as requests taking longer than usual, and potentially never completing.” | **AGREES** — “choose an acceptable rate of false timeouts (such as 0.1%).” | **AGREES** — “choose an acceptable rate of false timeouts (such as 0.1%).” | **AGREES** — “the corresponding latency percentile on the downstream service (p99.9 in this example).” |
| **SRC-13** arXiv:2412.17859 -- 'The Unreasonable Effectiveness of Open Science in AI: A Replication Study' | meta-science | **AGREES** — “If none of the reproducibility experiments was completed successfully, then the result of the reproducibility study is classified as No Result.” | **AGREES** — “Some reproducibility studies were aborted before all experiments were attempted, or the time limit was reached.” | **AGREES** — “No Result (NR), and Not Started (NS).” | **AGREES** — “The maximum time we spent on a reproducibility study was 40 hours of focused work.” |
| **SRC-14** SWE-bench harness -- swebench.harness.infra_failure | ml-harness | **AGREES** — “Separates "the environment broke" from "the model's patch was wrong"” | **AGREES** — “It is advisory.” | **AGREES** — “TIER_ENVIRONMENT = 'environment'” | **NOT-STATED** |

### Verdict counts (per question)

| Question | AGREES | DISAGREES | DISTINGUISHES | NOT-STATED | NOT-RETRIEVED |
|---|---|---|---|---|---|
| Q0_TIMEOUT_NOT_DEFECT_CLAIM | 5 | 3 | 6 | 0 | 0 |
| Q1_BROKEN_VS_UNMEASURED | 4 | 0 | 2 | 8 | 0 |
| Q2_STANDARD_NAME | 9 | 0 | 0 | 5 | 0 |
| Q3_BUDGET_VS_OBSERVED_COST | 2 | 0 | 3 | 7 | 2 |

> These counts are a census of the 14 sources in THIS corpus, which is a convenience sample chosen by the branch brief. They carry no statistical weight.

### What each column's verdict means

- **`Q0_TIMEOUT_NOT_DEFECT_CLAIM`** — Does the source treat a timeout as NOT a correctness failure (i.e. as a distinct outcome class rather than a failure of the subject under test)?
  - verdict rule: AGREES = timeout is a distinct outcome class; DISAGREES = the timeout is reported as a failure of the subject; DISTINGUISHES = the timeout is handled distinctly but stays inside a failure vocabulary (e.g. a named reason).
- **`Q1_BROKEN_VS_UNMEASURED`** — Does the source distinguish 'the subject under test is broken' from 'we did not obtain a measurement'?
  - verdict rule: AGREES = the distinction is made explicitly; DISAGREES = the source collapses the two; DISTINGUISHES = the distinction is made but with different vocabulary or only partially.
- **`Q2_STANDARD_NAME`** — What STANDARD NAME does the source use for the 'no measurement obtained' outcome class?
  - verdict rule: AGREES = the source names a distinct outcome class; DISAGREES = the source has no name for it (it collapses into failure); DISTINGUISHES = a named sub-state inside a failure (e.g. a reason string).
- **`Q3_BUDGET_VS_OBSERVED_COST`** — Does the source state a rule for setting the wall-clock budget relative to the OBSERVED cost (and for preventing a marginal budget from turning a load spike into a false verdict)?
  - verdict rule: AGREES = an observed-cost-relative rule is stated (percentile, false-timeout rate, padding); DISTINGUISHES = a flat or size-tiered default with no observed-cost rule; NOT-RETRIEVED / NOT-STATED = nothing on the budget.

## 2. Per-source detail

### SRC-01 — pytest-timeout (signal / thread timeout methods)

- **kind**: test-harness
- **url**: https://pytest-timeout.readthedocs.io/en/latest/
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **DISAGREES**
  - locator: section 'Timeout Methods' > 'signal'
  - quote: “the signal handler will dump the stack of any other threads running to stderr and use ``pytest.fail()`` to interrupt the test.”
  - note: pytest-timeout has NO timeout-specific outcome class: a timeout is an ordinary failing test. Under the `thread` method the process is killed outright -- "other pytest features, like e.g. JUnit XML output or fixture teardown, will not function normally".
- **Q1_BROKEN_VS_UNMEASURED** → **NOT-STATED**
  - locator: searched 'Timeout Methods' (thread, signal), 'Session Timeout', 'Timeouts in Fixture Teardown' -- no statement separating 'the subject is broken' from 'no measurement obtained'
  - quote: _(none — this cell asserts an absence)_
  - note: The retrieved text does not make this distinction. The plugin's vocabulary is pytest's own: a timed-out test is a failing test.
- **Q2_STANDARD_NAME** → **NOT-STATED**
  - locator: same sections as Q1 -- no timeout outcome-class name is introduced anywhere in the retrieved text
  - quote: _(none — this cell asserts an absence)_
  - note: No distinct name exists on this surface; the timeout surfaces as a normal test failure.
- **Q3_BUDGET_VS_OBSERVED_COST** → **NOT-RETRIEVED**
  - locator: no timeout-sizing guidance present in the retrieved rendering of the page (a second fetch of the same URL in this session returned HTTP 404)
  - quote: _(none — this cell asserts an absence)_
  - note: No budget-sizing rule is quoted from this source. This is a retrieval gap, not evidence that the project has no such guidance.

### SRC-02 — Bazel -- Test Encyclopedia ('Role of the test runner')

- **kind**: test-harness
- **url**: https://bazel.build/reference/test-encyclopedia
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **DISAGREES**
  - locator: section 'Role of the test runner'
  - quote: “it may choose to kill the test and treat the run as a failure.”
  - note: Also: "The runner must not report the test as passing after sending a signal to the test process or any children thereof." At the BUILD level Bazel collapses a timeout into a failure -- contrast SRC-03, where Bazel's event protocol keeps TIMEOUT as its own status.
- **Q1_BROKEN_VS_UNMEASURED** → **NOT-STATED**
  - locator: section 'Role of the test runner' -- the encyclopedia states the failure rule but never separates 'the subject is broken' from 'no measurement obtained'
  - quote: _(none — this cell asserts an absence)_
  - note: The nearest thing in the same document is that the runner 'may choose to kill the test and treat the run as a failure' -- a single failure class, no second class.
- **Q2_STANDARD_NAME** → **NOT-STATED**
  - locator: section 'Role of the test runner' -- no timeout-specific outcome name in the encyclopedia's result vocabulary
  - quote: _(none — this cell asserts an absence)_
  - note: The word TIMEOUT does exist in Bazel, but in SRC-03's event protocol, not in this document.
- **Q3_BUDGET_VS_OBSERVED_COST** → **DISTINGUISHES**
  - locator: section 'Role of the test runner' (default timeout by size)
  - quote: “Tests which do not explicitly specify a timeout have one implied based on the test's size as follows:”
  - note: A size-tier policy keyed to a DECLARED cost class -- published practice, but not a rule derived from an observed latency distribution.

### SRC-03 — Bazel -- test_status.proto, enum BlazeTestStatus (Build Event Protocol)

- **kind**: build-protocol
- **url**: https://bazel.googlesource.com/bazel/+/HEAD/src/main/protobuf/test_status.proto
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **AGREES**
  - locator: src/main/protobuf/test_status.proto, enum BlazeTestStatus
  - quote: “TIMEOUT = 3;”
  - note: A timeout is a first-class status, numerically distinct from FAILED (4). This is the strongest mainstream build-system support for this repo's split.
- **Q1_BROKEN_VS_UNMEASURED** → **DISTINGUISHES**
  - locator: src/main/protobuf/test_status.proto, enum BlazeTestStatus
  - quote: “INCOMPLETE = 5;”
  - note: The same enum separates INCOMPLETE, REMOTE_FAILURE and FAILED_TO_BUILD from FAILED -- the closest published analogue to 'the subject broke' vs 'the measurement was not obtained'. The per-value comments are absent from the current proto, so their intended semantics are NOT retrieved here.
- **Q2_STANDARD_NAME** → **AGREES**
  - locator: src/main/protobuf/test_status.proto, enum BlazeTestStatus
  - quote: “TIMEOUT = 3;”
  - note: The published status name is exactly the word this repo uses: TIMEOUT.
- **Q3_BUDGET_VS_OBSERVED_COST** → **NOT-STATED**
  - locator: src/main/protobuf/test_status.proto -- the enum carries no budget guidance
  - quote: _(none — this cell asserts an absence)_
  - note: See SRC-02 for Bazel's size-tiered default timeouts.

### SRC-04 — CMake / CTest -- TIMEOUT test property

- **kind**: test-harness
- **url**: https://cmake.org/cmake/help/latest/prop_test/TIMEOUT.html
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **DISTINGUISHES**
  - locator: prop_test/TIMEOUT.html, property description
  - quote: “If it exceeds that the test process will be killed and ctest will move to the next test.”
  - note: The timeout gets its OWN handling path (kill the process, then continue with the next test) rather than being treated as an ordinary assertion failure. Whether CTest's summary counts it as failed, and whether it labels the outcome 'Timeout', was NOT-RETRIEVED from a primary page in this session -- the ctest(1) man page does not state it and the only description found was in a search-engine summary, which is therefore not quoted.
- **Q1_BROKEN_VS_UNMEASURED** → **NOT-STATED**
  - locator: prop_test/TIMEOUT.html -- describes the kill-and-continue behaviour only; no broken-vs-unmeasured vocabulary
  - quote: _(none — this cell asserts an absence)_
  - note: No such distinction appears on this page.
- **Q2_STANDARD_NAME** → **NOT-STATED**
  - locator: prop_test/TIMEOUT.html -- the KNOB is named TIMEOUT, but no per-test outcome label is named
  - quote: _(none — this cell asserts an absence)_
  - note: TIMEOUT here names the budget property, not an outcome class.
- **Q3_BUDGET_VS_OBSERVED_COST** → **NOT-RETRIEVED**
  - locator: prop_test/TIMEOUT.html -- states a PRECEDENCE rule (TIMEOUT property > CTEST_TEST_TIMEOUT) but no sizing rule; the 1500 s default appeared only in a search-engine summary and is therefore not quoted
  - quote: _(none — this cell asserts an absence)_
  - note: Retrieval gap, not evidence of absence.

### SRC-05 — Go toolchain -- `go test -timeout` flag documentation

- **kind**: test-harness
- **url**: https://pkg.go.dev/cmd/go
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **DISAGREES**
  - locator: section 'Testing flags' > '-timeout d'
  - quote: “If a test binary runs longer than duration d, panic.”
  - note: The timeout aborts the whole test binary; there is no per-test timeout outcome class.
- **Q1_BROKEN_VS_UNMEASURED** → **NOT-STATED**
  - locator: section 'Testing flags' -- no such distinction stated
  - quote: _(none — this cell asserts an absence)_
  - note: No such distinction appears in the retrieved text.
- **Q2_STANDARD_NAME** → **NOT-STATED**
  - locator: section 'Testing flags' -- the flag is named '-timeout' but no outcome-class name is given
  - quote: _(none — this cell asserts an absence)_
  - note: See SRC-06 for the runtime wording.
- **Q3_BUDGET_VS_OBSERVED_COST** → **DISTINGUISHES**
  - locator: section 'Testing flags' > '-timeout d'
  - quote: “The default is 10 minutes (10m).”
  - note: A flat default for every package; no rule relating the budget to an observed cost.

### SRC-06 — Go standard library -- testing package source (the timeout panic)

- **kind**: test-harness
- **url**: https://go.dev/src/testing/testing.go
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **DISTINGUISHES**
  - locator: src/testing/testing.go, func (m *M) Run() (code int) -- the watchdog goroutine
  - quote: “panic("test timed out after " + timeout.String())”
  - note: The verdict text is NOT a bare 'fail': the panic names the timeout. But the process aborts and the package is reported as failed, so the timeout never becomes its own outcome class.
- **Q1_BROKEN_VS_UNMEASURED** → **NOT-STATED**
  - locator: src/testing/testing.go -- the panic text names the timeout but nothing states that the subject was not judged
  - quote: _(none — this cell asserts an absence)_
  - note: No such distinction appears in the retrieved text.
- **Q2_STANDARD_NAME** → **AGREES**
  - locator: src/testing/testing.go, func (m *M) Run()
  - quote: “panic("test timed out after " + timeout.String())”
  - note: The only published name on this surface is the panic text 'test timed out after <d>' -- a message, not an outcome label.
- **Q3_BUDGET_VS_OBSERVED_COST** → **NOT-STATED**
  - locator: src/testing/testing.go -- no budget-sizing rule
  - quote: _(none — this cell asserts an absence)_
  - note: See SRC-05 for the flat 10-minute default.

### SRC-07 — GitHub Actions -- jobs.<job_id>.timeout-minutes

- **kind**: ci-platform
- **url**: https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **AGREES**
  - locator: section 'jobs.<job_id>.timeout-minutes' (anchor #jobsjob_idtimeout-minutes)
  - quote: “The maximum number of minutes to let a job run before GitHub automatically cancels it. Default: 360”
  - note: 'automatically cancels it' -- cancellation, not failure. The machine-readable form of that distinction is in SRC-08.
- **Q1_BROKEN_VS_UNMEASURED** → **NOT-STATED**
  - locator: section 'jobs.<job_id>.timeout-minutes' -- no statement that a cancelled job says nothing about the code
  - quote: _(none — this cell asserts an absence)_
  - note: The docs do not draw the broken-vs-unmeasured line; they only report cancellation. (Caveat: 'cancelled' is also GitHub's status for a MANUAL cancellation, so the conclusion alone is not conclusive evidence about a timeout -- see SRC-08 for the timeout-specific value.)
- **Q2_STANDARD_NAME** → **NOT-STATED**
  - locator: section 'jobs.<job_id>.timeout-minutes' -- the wording is 'cancels'/'canceled'; the machine-readable status name is in SRC-08
  - quote: _(none — this cell asserts an absence)_
  - note: See SRC-08 for `timed_out`.
- **Q3_BUDGET_VS_OBSERVED_COST** → **DISTINGUISHES**
  - locator: section 'jobs.<job_id>.timeout-minutes'
  - quote: “Default: 360”
  - note: A flat platform default (360 minutes); no observed-cost-relative rule.

### SRC-08 — GitHub REST API -- Get a job for a workflow run (conclusion enum)

- **kind**: ci-platform
- **url**: https://docs.github.com/en/rest/actions/workflow-jobs
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **AGREES**
  - locator: REST API 'Get a job for a workflow run' -- job.conclusion schema
  - quote: “`success`, `failure`, `neutral`, `cancelled`, `skipped`, `timed_out`, `action_required`, `null`”
  - note: A timeout is reported under its OWN conclusion `timed_out`, not under `failure`. This is the single clearest piece of mainstream CI support for the repo's split.
- **Q1_BROKEN_VS_UNMEASURED** → **DISTINGUISHES**
  - locator: REST API 'Get a job for a workflow run' -- job.conclusion schema
  - quote: “`timed_out`”
  - note: The vocabulary can express 'the run did not produce a result' without asserting a defect, but the page does not spell out the semantics of each value.
- **Q2_STANDARD_NAME** → **AGREES**
  - locator: REST API 'Get a job for a workflow run' -- job.conclusion schema
  - quote: “`timed_out`”
  - note: The published status name is `timed_out` -- the same word this repo uses, in snake_case.
- **Q3_BUDGET_VS_OBSERVED_COST** → **NOT-STATED**
  - locator: REST API workflow-jobs page -- the schema states no budget-sizing rule (the platform default is in SRC-07)
  - quote: _(none — this cell asserts an absence)_
  - note: See SRC-07 for the flat 360-minute default.

### SRC-09 — GitLab CI -- the job `timeout` keyword

- **kind**: ci-platform
- **url**: https://docs.gitlab.com/ci/yaml/
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **DISTINGUISHES**
  - locator: section 'timeout' (job keyword)
  - quote: “If a job exceeds its timeout, it fails with the reason `execution expired`.”
  - note: GitLab collapses the timeout into a FAILURE but keeps a distinct, named reason string -- a compromise position between 'distinct outcome class' and 'plain failure'.
- **Q1_BROKEN_VS_UNMEASURED** → **NOT-STATED**
  - locator: section 'timeout' -- a reason string is given but no statement that the job's subject was never judged
  - quote: _(none — this cell asserts an absence)_
  - note: No such distinction appears in the retrieved text.
- **Q2_STANDARD_NAME** → **AGREES**
  - locator: section 'timeout' (job keyword)
  - quote: “`execution expired`”
  - note: A distinct, named reason -- but nested UNDER a failed job, so the name alone does not split the outcome classes.
- **Q3_BUDGET_VS_OBSERVED_COST** → **NOT-STATED**
  - locator: section 'timeout' -- states precedence over the project-wide setting only
  - quote: _(none — this cell asserts an absence)_
  - note: No sizing rule retrieved.

### SRC-10 — JUnit XML wire format -- outcome elements, as quoted by the pytest docs

- **kind**: report-format
- **url**: https://docs.pytest.org/en/stable/how-to/output.html
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **DISTINGUISHES**
  - locator: section 'Creating JUnitXML format files' (the embedded Jenkins JUnit schema)
  - quote: “<xs:element ref="error" minOccurs="0" maxOccurs="unbounded"/>”
  - note: The wire format has three distinct outcome elements. The retrieved text does NOT state which element a TIMEOUT maps to -- that sub-claim is NOT-RETRIEVED here and is not asserted.
- **Q1_BROKEN_VS_UNMEASURED** → **AGREES**
  - locator: section 'Creating JUnitXML format files' (the embedded Jenkins JUnit schema)
  - quote: “<xs:element ref="skipped" minOccurs="0" maxOccurs="1"/>”
  - note: `error` (the run broke) vs `failure` (an assertion failed) vs `skipped` (no result) is the published vocabulary for separating 'the subject is broken' from 'no measurement was obtained'.
- **Q2_STANDARD_NAME** → **AGREES**
  - locator: section 'Creating JUnitXML format files' (the embedded Jenkins JUnit schema)
  - quote: “<xs:element ref="error" minOccurs="0" maxOccurs="unbounded"/>”
  - note: The standard names are `error` and `skipped` (plus `failure`); there is NO `timeout` element in the retrieved schema.
- **Q3_BUDGET_VS_OBSERVED_COST** → **NOT-STATED**
  - locator: section 'Creating JUnitXML format files' -- a wire format says nothing about budgets
  - quote: _(none — this cell asserts an absence)_
  - note: No sizing rule retrieved.

### SRC-11 — Kubernetes -- Pod lifecycle (phase `Failed`, reason `DeadlineExceeded`)

- **kind**: infra-platform
- **url**: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **DISTINGUISHES**
  - locator: section 'Pod phase' > 'Failed'
  - quote: “The `reason` could be `DeadlineExceeded` if the Pod's `activeDeadlineSeconds` was exceeded.”
  - note: Phase-level `Failed` with a reason-level `DeadlineExceeded`: the deadline is named, but it lives inside the failure vocabulary.
- **Q1_BROKEN_VS_UNMEASURED** → **NOT-STATED**
  - locator: section 'Pod phase' > 'Failed' -- the reason is named but no statement says the workload was never judged
  - quote: _(none — this cell asserts an absence)_
  - note: No such distinction appears in the retrieved text.
- **Q2_STANDARD_NAME** → **AGREES**
  - locator: section 'Pod phase' > 'Failed'
  - quote: “`DeadlineExceeded`”
  - note: A named reason inside the failure vocabulary.
- **Q3_BUDGET_VS_OBSERVED_COST** → **NOT-STATED**
  - locator: section 'Pod phase' -- `activeDeadlineSeconds` is described, but no rule relates it to an observed cost
  - quote: _(none — this cell asserts an absence)_
  - note: No sizing rule retrieved.

### SRC-12 — AWS Builders' Library -- 'Timeouts, retries, and backoff with jitter'

- **kind**: practice
- **url**: https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **DISTINGUISHES**
  - locator: section 'Timeouts'
  - quote: “Many kinds of failures become apparent as requests taking longer than usual, and potentially never completing.”
  - note: A timeout is treated as one KIND OF FAILURE -- but the same section introduces FALSE timeouts, i.e. timeouts that are not failures at all (see Q1).
- **Q1_BROKEN_VS_UNMEASURED** → **AGREES**
  - locator: section 'Timeouts' (choosing the timeout)
  - quote: “choose an acceptable rate of false timeouts (such as 0.1%).”
  - note: 'false timeout' is the published name for exactly the split this repo encodes: the client gave up before the operation could be judged.
- **Q2_STANDARD_NAME** → **AGREES**
  - locator: section 'Timeouts' (choosing the timeout)
  - quote: “choose an acceptable rate of false timeouts (such as 0.1%).”
  - note: 'false timeout'.
- **Q3_BUDGET_VS_OBSERVED_COST** → **AGREES**
  - locator: section 'Timeouts' (choosing the timeout)
  - quote: “the corresponding latency percentile on the downstream service (p99.9 in this example).”
  - note: The only source in this corpus that states an OBSERVED-COST-RELATIVE rule: pick a latency PERCENTILE, declare the tolerated FALSE-TIMEOUT rate, and add PADDING so a load spike does not manufacture timeouts.

### SRC-13 — arXiv:2412.17859 -- 'The Unreasonable Effectiveness of Open Science in AI: A Replication Study'

- **kind**: meta-science
- **url**: https://arxiv.org/html/2412.17859v1
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **AGREES**
  - locator: §6 'Classifying Results' > 'Study Result Classification' (https://arxiv.org/html/2412.17859v1#Sx5.SSx2)
  - quote: “If none of the reproducibility experiments was completed successfully, then the result of the reproducibility study is classified as No Result.”
  - note: A budget-exhausted reproduction attempt is No Result, not Failure. §7: "we did not reproduce six and obtained no result for five".
- **Q1_BROKEN_VS_UNMEASURED** → **AGREES**
  - locator: §5 'The Replication Study' (https://arxiv.org/html/2412.17859v1#Sx4)
  - quote: “Some reproducibility studies were aborted before all experiments were attempted, or the time limit was reached.”
  - note: Three distinct 'no measurement' causes -- time budget, hardware, missing information -- none of which is a failure of the artifact under reproduction.
- **Q2_STANDARD_NAME** → **AGREES**
  - locator: Table 1 caption (§1) and §7 'Results'
  - quote: “No Result (NR), and Not Started (NS).”
  - note: 'No Result' is the published term for 'we did not get a measurement' in the reproduction literature.
- **Q3_BUDGET_VS_OBSERVED_COST** → **AGREES**
  - locator: §5 'The Replication Study' and §7 'Results'
  - quote: “The maximum time we spent on a reproducibility study was 40 hours of focused work.”
  - note: A per-study wall budget, published in advance; and the consequence policy when the budget IS exhausted is No Result, not Failure.

### SRC-14 — SWE-bench harness -- swebench.harness.infra_failure

- **kind**: ml-harness
- **url**: https://www.swebench.com/SWE-bench/api/harness
- **retrieved**: 2026-11-01 via WebFetch

- **Q0_TIMEOUT_NOT_DEFECT_CLAIM** → **AGREES**
  - locator: module swebench.harness.infra_failure (Harness API docs)
  - quote: “Separates "the environment broke" from "the model's patch was wrong"”
  - note: A published ML evaluation harness that separates environment failure from subject failure.
- **Q1_BROKEN_VS_UNMEASURED** → **AGREES**
  - locator: module swebench.harness.infra_failure -- advisory behaviour
  - quote: “It is advisory.”
  - note: Note the deliberate asymmetry with this repo: SWE-bench keeps the flagged instance INSIDE the failure denominator and only annotates it, whereas this repo removes the timeout from the defect verdict entirely.
- **Q2_STANDARD_NAME** → **AGREES**
  - locator: module swebench.harness.infra_failure -- TIER_ENVIRONMENT / TIER_AMBIGUOUS / INFRA_FAILURE_SIGNATURES
  - quote: “TIER_ENVIRONMENT = 'environment'”
  - note: The published term is 'infrastructure failure', with an explicit 'ambiguous' tier for a timeout.
- **Q3_BUDGET_VS_OBSERVED_COST** → **NOT-STATED**
  - locator: module swebench.harness.infra_failure -- classifies outcomes post hoc; no budget-sizing rule
  - quote: _(none — this cell asserts an absence)_
  - note: No sizing rule retrieved.

## 3. Findings — the three questions the brief asked

### F1. Does any published practice distinguish 'the subject is broken' from 'we did not get a measurement'?

**YES, in part of the corpus -- but NOT in the mainstream test-runner layer.**

- supporting sources: SRC-14, SRC-13, SRC-12, SRC-10, SRC-03
- sources that go the other way: SRC-01, SRC-02, SRC-05, SRC-09
- The distinction IS published -- 'infrastructure failure' vs the subject's failure (SRC-14), 'No Result' vs 'Failure' (SRC-13), 'false timeout' (SRC-12), the JUnit error/failure/skipped triple (SRC-10), and Bazel's INCOMPLETE / REMOTE_FAILURE / FAILED_TO_BUILD (SRC-03). But the tools most people actually run -- pytest-timeout (SRC-01), Bazel's build-level failure rule (SRC-02), Go's panic-the-binary rule (SRC-05) -- collapse a timeout into a failure.

### F2. What is the STANDARD NAME for the distinction?

**There is NO single de-facto standard across this corpus; the closest general-purpose terms are 'No Result' (reproduction literature) and 'infrastructure failure' (benchmark harnesses).**

- supporting sources: SRC-13, SRC-14, SRC-12, SRC-08, SRC-03
- Retrieved names, each from its own source: `timed_out` (GitHub Actions conclusion, SRC-08), `TIMEOUT` (Bazel Build Event Protocol status, SRC-03), 'execution expired' (GitLab reason, SRC-09), `DeadlineExceeded` (Kubernetes reason, SRC-11), 'error' and 'skipped' (JUnit elements, SRC-10), 'No Result' (SRC-13), 'infrastructure failure' / 'ambiguous' (SWE-bench, SRC-14), 'false timeout' (AWS, SRC-12). The repo's status word TIMEOUT coincides exactly with Bazel's published status name and with GitHub's `timed_out` in snake_case, so adopting it in the evidence text is defensible.

### F3. Is there published practice on the BUDGET question -- how the budget should be set relative to the observed cost, and how to avoid a marginal budget turning a load spike into a false verdict?

**YES, but only ONE source in this corpus states an observed-cost-relative rule; the rest state flat or size-tiered defaults.**

- supporting sources: SRC-12, SRC-13, SRC-02
- SRC-12 (AWS) is the rule: start from the downstream latency metrics, pick a percentile (p99.9 in its example), declare the tolerated false-timeout rate (0.1% in its example), and add padding so a small latency increase does not manufacture timeouts. SRC-02 (Bazel) keys the default to a declared size tier; SRC-05 (Go, flat 10 m) and SRC-07 (GitHub, flat 360 m) state flat defaults; none of those three references an observed cost. SRC-13 shows the consequence policy when a budget IS exhausted: record No Result, not Failure.

## 4. Honest conclusion

The repo's TIMEOUT-is-not-a-FAIL split is a DELIBERATE DEVIATION from the majority convention of mainstream test runners, and it is ALIGNED with the vocabulary-aware tail of the same literature. Of the 14 retrieved sources, the ones that speak to the question directly split: pytest-timeout (SRC-01), Bazel's build-level rule (SRC-02) and Go's panic (SRC-05) DISAGREE -- a timeout is a failure; GitHub Actions' `timed_out` conclusion (SRC-08), Bazel's Build Event Protocol TIMEOUT status (SRC-03), the replication literature's 'No Result' class (SRC-13) and SWE-bench's 'infrastructure failure' classifier (SRC-14) AGREE. The honest framing is 'a defensible deviation, consistent with the newer and harness-aware part of the literature', NOT 'the literature agrees with us'.

## 5. Retrieval log

| # | url | retrieved_at | tool | what was extracted |
|---|---|---|---|---|
| 1 | https://bazel.build/reference/test-encyclopedia | 2026-11-01 | WebFetch | Section 'Role of the test runner': the kill-and-treat-as-failure rule, 'Any other result is considered a test failure.', and the size-derived default timeouts (60/300/900/3600 s). |
| 2 | https://pkg.go.dev/testing | 2026-11-01 | WebFetch | Only T.Deadline was in the retrieved window; the -timeout flag text was NOT found here and was taken from pkg.go.dev/cmd/go. |
| 3 | https://pkg.go.dev/cmd/go | 2026-11-01 | WebFetch | Section 'Testing flags': '-timeout d / If a test binary runs longer than duration d, panic.', 'If d is 0, the timeout is disabled.', 'The default is 10 minutes (10m).' |
| 4 | https://bazel.build/reference/be/common-definitions | 2026-11-01 | WebFetch | No TestResult.Status enum on this page -- the enum was instead retrieved from the proto source (below). |
| 5 | https://go.dev/src/testing/testing.go | 2026-11-01 | WebFetch | func (m *M) Run(): the deadline assignment, the time.After wait, and the timeout panic string. |
| 6 | https://cmake.org/cmake/help/latest/prop_test/TIMEOUT.html | 2026-11-01 | WebFetch | The TIMEOUT property description, including 'If it exceeds that the test process will be killed and ctest will move to the next test.' and the CTEST_TEST_TIMEOUT precedence sentence. |
| 7 | https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions | 2026-11-01 | WebFetch | Section 'jobs.<job_id>.timeout-minutes' (anchor #jobsjob_idtimeout-minutes): the 360-minute default and the 'automatically cancels it' wording. |
| 8 | https://pytest-timeout.readthedocs.io/en/latest/ | 2026-11-01 | WebFetch | Sections 'Timeout Methods' (thread, signal), 'Specifying the Timeout Method', 'Session Timeout', 'Timeouts in Fixture Teardown'. A second fetch of the same URL in this session returned HTTP 404, so the FAQ sizing guidance is NOT quoted. |
| 9 | https://docs.github.com/en/rest/actions/workflow-jobs | 2026-11-01 | WebFetch | The job.conclusion schema: success / failure / neutral / cancelled / skipped / timed_out / action_required / null, and the job.status values. |
| 10 | https://docs.gitlab.com/ci/yaml/ | 2026-11-01 | WebFetch | The job `timeout` keyword: 'If a job exceeds its timeout, it fails with the reason `execution expired`.' |
| 11 | https://docs.pytest.org/en/stable/how-to/output.html | 2026-11-01 | WebFetch | Section 'Creating JUnitXML format files': the embedded Jenkins JUnit schema with the failure / error / skipped testcase children. |
| 12 | https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/ | 2026-11-01 | WebFetch | Section 'Pod phase' > 'Failed': the phase definition and the `DeadlineExceeded` reason sentence. |
| 13 | https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter | 2026-11-01 | WebFetch | Section 'Timeouts': the false-timeout rate (0.1%), the latency percentile (p99.9) and the padding sentence. |
| 14 | https://arxiv.org/html/2412.17859v1 | 2026-11-01 | WebFetch | Table 1 caption (Success / Partial Success / Failed / No Result / Not Started), §5 (the 40-hour budget and the abort causes), §6 'Classifying Results' (the Failure vs No Result definitions), §7 (the aggregate counts). |
| 15 | https://www.swebench.com/SWE-bench/api/harness | 2026-11-01 | WebFetch | The swebench.harness.infra_failure module docstring, the TIER_ENVIRONMENT / TIER_AMBIGUOUS constants, the INFRA_FAILURE_SIGNATURES tuples (including tests_timed_out), and the advisory / denominator-preserving behaviour. |
| 16 | https://bazel.googlesource.com/bazel/+/HEAD/src/main/protobuf/test_status.proto | 2026-11-01 | WebFetch | enum BlazeTestStatus with all nine values and the enum-level comment. |

## 6. Caveats — what this branch did NOT establish

- Retrieval dates are recorded at DATE granularity (the session date, 2026-11-01): the WebFetch tool exposes no per-call clock.
- Quotations are the exact strings returned by the fetch tool's extraction, not re-reads of the page by eye; a second fetch of a page can legitimately return a different rendering.
- Three sub-claims are retrieval gaps, NOT evidence of absence: (a) pytest-timeout's FAQ sizing guidance (a second fetch of the same URL returned HTTP 404); (b) whether CTest labels a timed-out test 'Timeout' and counts it as failed (the ctest(1) man page does not say, and the only description found was in a search-engine summary, which is therefore not quoted); (c) CTest's 1500 s default, for the same reason.
- Tool documentation is used as evidence of a TOOL's semantics, which is the tool's own normative statement of its behaviour; that is not the same as an independent study of how users read the resulting numbers.
- Corpus composition is a convenience sample chosen by the branch brief (CI harnesses, one ML harness, one reproduction study, one infrastructure platform, one practice article). It is NOT a systematic survey, so the AGREES/DISAGREES counts below carry NO statistical weight and must not be quoted as 'X% of published practice'.
- Verdicts are INTERPRETATION over the quoted text. The quotes are MEASURED (retrieved live in this session); the verdicts are the analyst's reading of them.
- This branch did NOT test whether the repo's own TIMEOUT gates would be re-classified by any of these sources -- it only asks what the sources say. It did NOT re-run the pre-flight, did NOT touch any gate, and did NOT touch metrics/, registry/, docs/, TODO/ or paper/.
