# Wf-A170 — Refutation of Branch 1 ("the GPU-required test repair")

**VERDICT: NOT REFUTED.** Tag: MEASURED. 0 GPU used.
Single venv (`/home/hugo/codes/mol-metal/.venv/bin/python`); ruff 0.6.9 from the pre-commit
isolated env. Full evidence: `refutation.json` (same directory).

Every MEASURED claim in the branch report reproduces on disk under a CPU-locked session.

## Checks

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Branch's own claims re-derived | PASS | pre-repair RED reproduced by direct measurement (gate SKIP under lock; HEAD T1 `status=='PASS'` and T2 `status in ('PASS','FAIL')` both evaluate False; HEAD T3 pin `94` vs live `129` evaluates False); live counts `all=129 launch=131 frontier=25 chemprop=18 cfm=8`; AST census T1 9/8, T2 5/2, T3 9/2 (exactly as claimed); `3 passed` + `4 passed`; `safe_overwrite_check` 0.5482; ruff `check` clean; no torch import |
| 2 | Negative controls actually bite | PASS | pre-repair file -> hygiene test 2 RED; mutants: module `pytestmark` -> test 3 RED; per-test `skipif` -> test 3 RED; gate stubbed PASS -> control RED; gate stubbed silent SKIP -> control RED; unmutated -> GREEN |
| 3 | Not vacuous | PASS | no `pytestmark`, no decorators, no `pytest.skip`; the one literal `pytest.mark.skipif` string is docstring prose (line 89); 23 asserts run in the locked session; HEAD assertions 15/6/10 -> 17/7/11 with the pin the only non-verbatim change (94 preserved verbatim, dated) |
| 4 | No scope creep | PASS | exactly 2 files written (both mtime 12:24:06); `conftest.py` untouched; every forbidden path's mtime < 12:24 |
| 5 | `metrics/` untouched | PASS | md5 `a022e54fd2938b9f333fd8438f6d3d27`, 38 lines, before == after, diff empty |
| 6 | Artefact honestly regenerated | PASS | fresh `check_blackwell_env.py` run differs only in volatile GPU telemetry (power/temp/util/timestamp); all structural fields byte-identical |

## Surviving concerns (none falsifies a stated claim)

- **B (medium)** — the re-synced pin `== 129` is coupled to the UNCOMMITTED worktree
  (`pre_flight_check.py` HEAD carries `all: 126`), so a clean HEAD checkout is still RED.
  A stale literal was re-synced to another stale literal rather than dropped.
- **A (medium-low)** — the shipped docstring says "NONE of the 3 failures had anything to do
  with what they guard"; false for T3, whose declared guard IS the gate-count invariant. The
  branch report item 2 concedes the separate stale pin but the sentence survives verbatim.
- **C (low)** — the masked branch tolerates only `SKIP`; a probe TIMEOUT (`GATE_STATUS_TIMEOUT`)
  would fail it, re-introducing for the timeout path the defect Wf-A128 fixed.
- **E (low)** — ruff-format regression: HEAD's file was format-clean, the branch's is not
  (3 hunks). `ruff format` was not run; `ruff check` was.
- **D (low) / F, G (informational)** — 12 GPU-visible asserts unrun (disclosed); the "negative
  control" is a positive control on the GATE, not on the pre-repair file; the `gpu_count=`
  cross-check is absent from T3(f) by construction.
