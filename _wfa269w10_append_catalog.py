#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Append the Wf-A269 wave-10 entry to docs/HONEST_NEGATIVE_CATALOG.md.

Append-only + dated (AGENTS.md section 5.5 rule #6).  The existing body is
copied byte-for-byte; the superseded rendering is never edited or deleted.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT / "molmetal" / "scripts"))

from wf386a3_atomic_write import atomic_write_file  # noqa: E402

CATALOG = REPO_ROOT / "docs" / "HONEST_NEGATIVE_CATALOG.md"

ENTRY = r"""

### §6.269.17 — **THE DEAD-GATE CLASS: A CHECK THAT CANNOT FAIL** (Wf-A269 wave-10, the main thread, 2026-11-02)

**Net +0. Cumulative 324 → 324.** This is a **wave closure**, not a new honest-negative row, exactly as §6.269.16 was. Seven dead gates were found and **repaired**; a repaired defect is a **closed** defect, not an added one. The one live finding this wave produced (the p5 full sweep's 12 hr projection against a 10 hr session cap) is recorded **inside** this entry and is now carried by the plan artefact itself (`plan.full_5seed.json::budget_verdict = "FAIL"`), so it does not need a separate ACTIVE row either.

**⇒ A NOTE ON THE SECTION NUMBERING, so the gap is not read as a deletion.** §6.269.16 exists as a **staged file** (`/tmp/wfa269k_sec_6_269_16.md`, 25 537 B, the wave-5 closure) that has **not yet been appended to this catalogue**. This entry is therefore numbered **§6.269.17** so it cannot collide with it. The gap between §6.269.15 (in this file) and §6.269.17 (this entry) is **deliberate and traceable to that staged file** — it is **not** a removal (AGENTS.md §5.5 rule #6 forbids silent deletion).

**⇒ AND A NOTE ON THE ACTIVE COUNTER, because it is DRIFTED.** The catalogue's own last internal ACTIVE statement is **275** (§6.224 catalog-count math, `ACTIVE 274 → 275`, at line 21 082), while AGENTS.md's gloss says **279 active**. That is a **4-count drift MEASURED 2026-11-02**. This entry does **not** attempt to re-derive it: §6.269.6 ruled that **a census is a coverage census, not a defect count**, and a re-derivation of an ACTIVE count is a census. The drift is **recorded here**, not silently reconciled.

---

#### §6.269.17.1 — WHAT THE CLASS IS, AND WHY IT IS WORSE THAN NO GATE AT ALL

A **dead gate** is a check that *looks* like a check but **cannot fail**. It is not a missing gate — a missing gate is visible. It is a gate that reports a verdict, and the verdict is a **constant**.

The class has **three sub-modes**, all found in the two launch wrappers this wave repaired:

| Sub-mode | Shape | Consequence |
|---|---|---|
| **(a) an undefined flag** | a wrapper invokes a tool with a flag the tool does **not** define | argparse exits 1/2 **instantly**; the tool measures nothing; the wrapper takes its "carry on" branch **every time** |
| **(b) a stdout grep for a marker no producer emits** | the wrapper greps the tool's **stdout** for a string the tool never prints | the success branch is **unreachable**; only the failure branch can run — and it only warns |
| **(c) a stale literal target** | the wrapper greps for a literal (e.g. `62 PASS`) that no longer matches the gate count | same as (b), plus the target rots silently as the gate count moves |

**⇒ THE INSTANT-EXIT MASKING EFFECT — why this class survives review.** Sub-mode (a) is the dangerous one, because the tool exits **immediately**. A launch wrapper that had **never run a pre-flight** therefore produces a dry-run that **looks clean**: no error, no timeout, no missing file, just a fast successful-looking plan. The absence of a measurement is indistinguishable from a passing measurement. **A gate that cannot fail is worse than no gate, because it buys the reader the confidence that no gate would.**

---

#### §6.269.17.2 — THE SEVEN INSTANCES (all MEASURED, all REPAIRED)

| # | File | The dead construct | How it was dead |
|---|---|---|---|
| 1 | `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | `pre_flight_check.py --skip-anchor-guard` | the flag does **not** exist → argparse exit 2, "unrecognized arguments"; the pre-flight **never ran** |
| 2 | same | `wave_budget_tracker.py --cfm-yuelbond-not-required` | the flag does **not** exist (only `--cfm-yuelbond-required` does) → exit 1; the tracker **never ran** |
| 3 | same | pre-flight stdout grepped via `tail -5` for the literal `62 PASS` | sub-mode (c): the report's `summary` block sits at the **TOP** of the JSON, so the last 5 lines can never contain it; `62` is also a stale Wf-328-SYN baseline |
| 4 | same | per-wave checklist stdout grepped for `'"status": "PASS"\|PASS'` | sub-mode (b): **MEASURED 2026-11-02 — the checklist's own stdout is 88 bytes, and every byte is the molmetal package's import banner** ("[molmetal] applied env vars: …"). The tool's verdict is in the JSON it writes via `--out`. |
| 5 | same | wave budget tracker stdout grepped for `PASS\|OK\|STATUS` | sub-mode (b): the tracker prints **nothing** on stdout and **always exits 0**; its verdict is in the JSON it writes via `--report-json` |
| 6 | `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh` | the phase gate `grep -q "phase complete\|TEST.*Pearson\|VAL.*Pearson"` | sub-mode (b): **none** of the three markers has a producer repo-wide — `molmetal/scripts/train_chemprop_v2_v3.py` contains **ZERO** case-insensitive `pearson` occurrences, and "phase complete" has **no** producer at all. The wrapper's `wait "${pid}" \|\| true` also **threw away the trainer's exit code**, which is the authoritative signal. |
| 7 | same | the per-wave gate `if ! per_wave_checklist.py …; then die; fi` | **MEASURED 2026-11-02 — the tool exits 0 even when its own checklist records a FAILED P0 blocker.** Reproduced with a synthetic pre-flight report carrying one FAIL gate: `exit code = 0`, while the checklist JSON says `preflight_fail_count = 1` and P0 blocker `g_bad = FAIL`. Two further defects rode along: `--out` was not passed, so the checklist landed at the tool's **SHARED** default path (`molmetal/reports/wf262_checklist/per_wave.json`) while the file the error message pointed at held only stderr text; and `--preflight` was not passed, so the checklist read the tool's **stale DEFAULT** pre-flight report (`molmetal/reports/wf259_run/pre_flight.json`) instead of this launch's own. |

**⇒ A BONUS DEFECT FOUND WHILE REPAIRING #7 — the "JSON" report was not JSON.** The bindingdb wrapper captured the pre-flight with `> pre_flight.json 2>&1`, so the file held the tool's **stderr log lines interleaved with** the JSON payload on stdout. `json.load` on such a file raises `Expecting value: line 1 column 2` — **the same defect as the stale `molmetal/reports/wf_preflight/preflight.json`**, whose first line is `[pre_flight_check] INFO …`. A gate whose evidence file is not machine-readable cannot be consumed by a machine. **REPAIRED**: `--json <path>` writes a real JSON file (the flag is `type=Path, nargs="?"`, so a **bare `--json` writes NO file at all**), and stdout goes to a separate `.log`.

**⇒ AND A FOURTH VERDICT-VOCABULARY DEFECT.** The p5 wrapper's "tool is missing" branches emitted the verdict token **`SKIPPED`** while its dry-run branches emitted **`UNVERIFIED`** for the same condition. The plan record's verdict field therefore had **four** possible values (`PASS` / `FAIL` / `SKIPPED` / `UNVERIFIED`) and no consumer knew three of them. AGENTS.md §5.2: **a gate that was SKIPPED says `UNVERIFIED`, never PASS.** **REPAIRED** to `UNVERIFIED` on both branches.

---

#### §6.269.17.3 — THE REPAIR: ONE SHARED CONSUMER, NOT SEVEN MORE GREPS

`molmetal/scripts/wf_launch_gate_verdict.py` (NEW; Apache-2.0 header) is the **single place** that knows how to read the three report formats:

- `preflight --report r.json` → PASS iff `summary.fail == 0` **and** `summary.timeout == 0`. A TIMEOUT is a FAIL here **on purpose** (Wf-A135): a gate that timed out did not obtain its measurement, so "the gate passed" was never established.
- `checklist --report r.json` → PASS iff the checklist recorded **no** `error`, the pre-flight it read has `preflight_fail_count == 0`, **and** every P0 **blocker** gate has `status == "PASS"`. A P0 blocker that was SKIPPED is **not** a PASS — the blocker was never verified.
- `budget --report r.json [--wave NAME]` → PASS iff `remaining_hr >= 0` and the wave is not in `deferred_waves`.

Stdout invariant (AGENTS.md §4): first token is `PASS` / `FAIL` / `SKIP`; exit 0 / 1 / 2. A missing or unreadable report is **`SKIP`** — never PASS.

**⇒ THE SHAPE OF THE FIX: THE CONSUMER READS THE PRODUCER'S REPORT, AND NEVER RE-RUNS THE TOOL.** This is the **verdict-as-consumer** pattern. The alternative — re-running the tool to see whether it fails — would double the cost of every gate and re-introduce the ambiguity of "which run's report was this?".

**⇒ THE DRY-RUN HONESTY.** In a dry-run the live pre-flight is **skipped** (it takes ~400 s, MEASURED), and the checklist is **skipped** (its input is the skipped pre-flight). Both therefore carry the verdict **`UNVERIFIED`** — **never PASS**. The budget tracker **runs** (0.22 s) and yields a **real** verdict. The wrapper records all three on the face of the plan record and of the `DRY_RUN.log` (new opt-in `--gate-verdict` field in `scripts/_dry_run_log_helper.sh`), so the plan says **what was and was not measured** instead of leaving it in the wrapper's stderr.

**⇒ THE REPAIRED BINDINGDB GATES** (file `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh`): the pre-flight gate now writes a real JSON report and reads its verdict through the shared checker (`:277`); the budget gate reads the tracker's report instead of its always-zero exit code (`:312`); the per-wave checklist gate writes to **this launch's** report path, reads **this launch's** pre-flight report, and dies on a non-PASS verdict (`:369`); the phase abort gate captures the trainer's exit code (`wait "${pid}" || train_rc=$?`, `:608`) and greps for `=== Wf-200 FINAL (JSON) ===` — a marker `molmetal/scripts/train_chemprop_v2_v3.py:2375` **actually prints** (`:2377` returns `0 if ran_cleanly else 2`).

---

#### §6.269.17.4 — THE REGRESSION FILE, AND WHY IT STRIPS COMMENTS FIRST

`molmetal/tests/test_wf_a269_wave10_dead_gates.py` — **14 tests, 14 PASS (MEASURED 2026-11-02, CPU-only, `CUDA_VISIBLE_DEVICES=""`)**.

**⇒ THE TESTING PROBLEM, AND THE METHOD THAT SOLVES IT.** AGENTS.md §5.5 rule #6 requires the superseded rendering to be **preserved verbatim** — so the dead flag **strings are still on disk**, inside comment blocks. A naive substring assertion would be satisfied by the comment and would pass on a wrapper that was never repaired. The tests therefore **strip the full-line comments first** and assert against the **CODE**, which is what actually runs.

**⇒ THE BLIND SPOT THIS METHOD DOES NOT COVER, STATED RATHER THAN HIDDEN.** An **inline trailing comment** (`cmd --flag # note`) is not stripped, so a flag mentioned only in such a comment would still read as "active". Every occurrence in the two repaired wrappers is a **full-line** comment, so the assertion is exact **for them**; a future inline-comment occurrence needs the pattern re-checked by hand. §6.269.17's own discipline: **a guard's coverage bound is part of the finding, not an afterthought.**

**⇒ THE TESTS ARE NOT ALL STRING MATCHES.** Two of them are **end-to-end**: the p5 dry-run is actually executed, and the test asserts that the budget verdict **follows the arithmetic** — FAIL iff `WALL_BUDGET_HR > SESSION_CAP_HR`, PASS otherwise — read from the **mode-specific** plan record (`plan.full_5seed.json` / `plan.smoke_2seed.json`, the wave-9 durable copies). The canonical `plan.json` is overwritten by whichever mode ran last, so a test reading it would depend on **test order**; reading the mode-specific record does not.

---

#### §6.269.17.5 — THE ONE LIVE FINDING THIS WAVE PRODUCED

**The p5 full sweep does not fit the session cap.** `WALL_BUDGET_HR=12` against `SESSION_CAP_HR=10` (AGENTS.md §5.1 + Wf-294B kill_then_adapt) ⇒ the wave budget tracker's own report says `remaining_hr = -2.0` and `10.0h budget exceeded by 2.0h` ⇒ the **budget gate verdict is `FAIL`** (MEASURED, on the plan record: `"budget_verdict": "FAIL"`, `"session_cap_hr": 10`). The **smoke** sweep (`WALL_BUDGET_SMOKE_HR=3`) fits and yields `PASS`.

**⇒ THE SESSION CAP AND THE WAVE BUDGET ARE NOT THE SAME QUANTITY, AND THE REPAIR ADDS THE FORMER.** The wave budget is what the wave is **expected** to cost; the session cap is the number it is **measured against**. The wrapper had only the former, so its "budget gate" could not answer the question that matters. **⇒ This finding was INVISIBLE before the repair** — the dead gate always took its warn-and-continue branch, and the plan record said **nothing at all** about the budget. It is now a `FAIL` on the face of `plan.full_5seed.json`. **The wave must be split, or the cap raised by explicit user decision.**

---

#### §6.269.17.6 — THE VERIFICATION (MEASURED, CPU-only, 0 GPU)

| Check | Result |
|---|---|
| `bash -n` on all three edited scripts | **PARSE OK** (p5, bindingdb, `_dry_run_log_helper.sh`) |
| `molmetal/tests/test_wf_a269_wave10_dead_gates.py` | **14 passed** |
| the 7 affected suites + the new file (8 files) | **8 failed, 117 passed, 1 skipped** |
| the 8 failures | **byte-identical to the recorded baseline** `/tmp/wfa269i/after_11files_v2.log` (same 8 test IDs) ⇒ **ZERO new failures** |
| `git status --porcelain metrics/` before vs after | **0 diff** (40 entries both sides) — no published metric was touched |
| GPU watchdog `/tmp/wfa269k_watchdog_attributable.log` | **0 lines** — 0 GPU allocations |

**⇒ THE 8 PRE-EXISTING FAILURES ARE NOT MINE, AND THEY ARE A DIFFERENT DEFECT CLASS.** All 8 are the **USER_OK gate vocabulary**: the wrappers disagree with each other and with the test tables about the *accepted value* of `USER_OK` (`launch_wf124_p3_g5.sh` accepts `1`; `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` documents `YES`; `test_wf365b2_7_wrappers_final.py` parametrizes `YES` for Wf-355-A1 and `1` for Wf-355-B2 and `None` for Wf-124-G5, whose wrapper **refuses even a dry-run without `USER_OK=1`**). This is the **mirror image** of the dead-gate class: here the gate is loud and does fire, but the **producer and the consumer disagree about the value that opens it**. **LEFT OPEN and handed to the next wave** — it is a **cross-wrapper vocabulary** defect, and repairing it means deciding a canonical value, which touches launchers that are not this wave's subject.

---

#### §6.269.17.7 — WHAT IS STILL OPEN (and the census that is NOT a defect count)

**⇒ THE CLASS IS BROADER THAN THE SEVEN REPAIRED.** A repo-wide census of `grep -c … || true` over `scripts/*.sh` found **SIX further wrappers** still carrying the pre-flight count idiom: `launch_wf146_deferred_runs.sh:228-229`, `launch_wf299_cfm_main.sh:346-347`, `launch_wf306a_drugood_vina_gpu.sh:238-239`, `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh:227-228`, `launch_wf310f_frontier_math_gpu.sh:196-197`, `launch_wf74_v2_drugood_dann.sh:274`. **This count is a COVERAGE census, NOT a defect count** (§6.269.6): whether each of those six gates is *live* depends on **how that wrapper captures the pre-flight report** — and the bindingdb wrapper proved that a capture method can make the same idiom dead in one wrapper and live in another. **Each needs its own MEASUREMENT; none is condemned by this entry.**

**⇒ THE REMAINING OPEN ITEMS FROM THIS WAVE.**

1. **The census above** — six wrappers to measure, one at a time, with the shared checker as the arbiter.
2. **The USER_OK vocabulary class** (§6.269.17.6) — a canonical value must be decided; it touches launchers outside this wave's subject.
3. **`scripts/_dry_run_log_helper.sh` is shared** by the 5 wrappers of the Wf-363-A1 schema; the new `--gate-verdict` field is **OPT-IN** (like `--capability`), so no existing caller changed behaviour — but the other wrappers do **not** yet declare a gate verdict, so their plans still say nothing about what was measured.
4. **The `molmetal/reports/wf_preflight/preflight.json` file is still not JSON** (it starts with `[pre_flight_check] INFO …`). It is a **stale artefact** and was **not** repaired here — its producer is a different wrapper's capture method, and rewriting a published artefact's content is a §5.5 event, not a silent fix.
5. **USER-gated, untouched**: the U1 commit (the two `wfa155_*` metric JSONs) + `run_ledger.py repoint`; the 24 dead pre-training gates + the 2 orphan gates + `EXPECTED_GATE_COUNTS`; `paper/refs.bib`; `paper/figures/fig4_homotype_vs_tanimoto.py`; `docs/METRIC_CITABILITY_RULE.md:288-292`; `docs/PREDICTIVE_LINE_INDEX.md`'s m21 entry; A201's 0.4074 arm (needs a GPU re-run — **forbidden** by the standing 0-GPU directive); Fix B (`_smoke_device()` fail-closed); Fix C (`pre_flight_check.py:21`'s docstring); the §5.9 amendment proper; installing `ctxlint` / `lychee`.

**⇒ THE TRANSFERABLE RULE.** **A gate's verdict must be read from the artefact the gate's tool actually wrote, by a consumer that knows how to read it — and a gate whose only signal is an exit code must be checked against what that tool actually returns on failure.** Both halves of this wave's class were found by **running the tool and reading its own report**, not by reading the wrapper: the checklist's exit code was measured against its JSON, and the pre-flight's `--json` flag was read from its argparse definition (`type=Path, nargs="?"` — a bare `--json` writes **no file**). **A wrapper is a consumer; a consumer that infers a verdict from the absence of output is not a gate.**
"""


def main() -> int:
    original = CATALOG.read_text(encoding="utf-8")
    assert "§6.269.17" not in original, "the entry is already present — refusing to double-append"
    assert original.endswith("\n"), "the catalogue must end with a newline before appending"
    merged = original + ENTRY
    atomic_write_file(CATALOG, merged, encoding="utf-8")
    # §5.5 filesystem truth: prove the file on disk, not just the write.
    after = CATALOG.read_text(encoding="utf-8")
    assert after.startswith(original), "the original body was modified — this must be append-only"
    assert after.endswith(ENTRY), "the entry did not land at the tail"
    print(f"APPEND OK: {CATALOG} {len(original)} -> {len(after)} bytes (+{len(after) - len(original)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
