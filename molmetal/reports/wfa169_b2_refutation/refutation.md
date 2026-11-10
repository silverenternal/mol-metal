# Wf-A169 — refutation of branch 2 (execution verdict of the 24 gates)

**VERDICT: NOT REFUTED** on the execution verdict (19 PASS / 4 WARN / 1 FAIL / 0 RAISED over 24)
and on the history correction. **One sub-claim is REFUTED AS WRITTEN**: the pre-flight goes red
by **TWO** gates, not one, unless the count pins are bumped in the same edit.

Tag: MEASURED 2026-11-01. 0 GPU (`CUDA_VISIBLE_DEVICES=""`), single venv
(`.venv/bin/python`). Full evidence: `refutation.json`.

## Checks

| # | Check | Result |
|---|---|---|
| C1 | Branch's own numbers re-derived (3 independent arg sets) | PASS — 19/4/1/0 x3, FAIL identity + 4 WARN identities confirmed, spec-literal 16/6/2 reproduced exactly |
| C2 | Regression test bites | N/A (branch 2 shipped no test); surrogate: arg-count cross-check 0/24 mismatches, witness-arg tally 19/4/1 |
| C3 | Branch applied no patch to the working tree | PASS — 0 of the 24 registrations in the worktree; the uncommitted diff is another branch's D1/D2 emitter repair |
| C4 | Are the 24 unregistered? | Live tree: 0 registrations (no double-registration). History: **CONFIRMED** — the deleted `.bak.wf437` blob (359,637 B, `all`=127) holds all 24 tuples; AST diff vs bc29aa90 = exactly 29 removed (15 p3_protocol + 5 mock_to_real + 9 hooks), 0 added |
| C5 | Does wiring redden the pre-flight by one gate? | **REFUTED AS WRITTEN** — measured: `all` 129->153, `launch` 131->155; `p3_gate_count_consistency_audit` FAILs `+DRIFT (silent gate bloat) actual=153 vs expected=129` unless the pin is bumped |
| C6 | Blast-radius complete? | INCOMPLETE — `EXPECTED_GATE_COUNTS['all'/'launch']`, `test_wf327f11_gate_count_consistency.py:98-104`, branch 1's resync patch (pins 129/131 = pre-wiring), `AGENTS.md:36` prose pin; `p3_ssot_census_claim_check` scope NOT MEASURED |
| C7 | Are the 9 L5 PASSes verification? | TAUTOLOGICAL — the gate writes the hook JSON it certifies (`pre_flight_check.py:12407-12453`); branch disclosed this |

## Surviving concerns

1. Bump `EXPECTED_GATE_COUNTS['all']` 129->153 AND `['launch']` 131->155 in the same edit as the wiring,
   else the count-drift audit is a SECOND failing gate. Branch 1's two patches on disk do not do this.
2. Branch 1's `patch_resync_stale_pins.diff` pins the PRE-wiring values (129/131) -> applying both patches
   leaves `test_wf327f11_gate_count_consistency.py` RED on the `launch` pipeline.
3. `AGENTS.md:36`'s "129 gates (131 on launch)" goes false after wiring; a third FAIL via the census-claim
   gate is possible — NOT MEASURED.
4. The history correction is right about the ARTEFACT but must not be read as "the gates ran": the
   registrations lived only in a purged backup blob, never in the live path.
5. The branch's "the 10,643-byte gap is exactly them" is imprecise (300-line diff, also carries the
   count-comment rewrite + Wf-439 venv fix); the 29-tuple removal itself is exact.
6. The branch's caveat (a) is CONFIRMED for `eval_per_seed_per_split`, `per_cell_breakdown`, `eval-5fold-cv`
   (spec + old TODO docs only, no trainer); the wider "10 of 24" estimate was NOT re-derived.
