# Wf-A175 — adversarial refutation of the Wf-A172 stale-pin repair (branch PINS)

Date: 2026-11-01. Scope: `molmetal/scripts/wfa169_gate_count_pins.py` (census),
`molmetal/tests/test_wfa169_gate_count_pins.py` (census test), the 4 repaired test
files, and `molmetal/scripts/pre_flight_check.py`.
CPU-only throughout (`CUDA_VISIBLE_DEVICES=""`, `/home/hugo/codes/mol-metal/.venv/bin/python`).
READ-ONLY: nothing outside `molmetal/reports/wfa175_pin_repair_refutation/` was written
(§5.3/§5.5 respected); all writes use `wf386a3_atomic_write.atomic_write_file` (§5.6).

**VERDICT: the repair's MEASURED greenness is real, but the repair is INCOMPLETE as a
class repair and its own guard suite freezes the remainder.** The narrow claims in the
branch brief all hold (§1). Five findings below are genuine defects (D2-D5, D9), one is a
coverage regression that is inherent to the repair form but unstated (D6), and one is a
coverage regression that contradicts the branch brief's own warning (D7).

---

## §0 — what I confirmed (could not refute)

**S1 — 0 stale pins and the exact Counter.** MEASURED.

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -c "import sys;sys.path.insert(0,'molmetal/scripts');
import wfa169_gate_count_pins as m;c=m.pin_census();print(c['stale_pin_count'],c['stale_pin_files'])"
-> 0 []
Counter({'dynamic': 20, 'ok': 8, 'floor-satisfied': 6, 'derived-unresolved': 1})
```

Identical to the branch brief. `counts_match` is True; live and SSOT both read
all 129 / launch 131 / frontier 25 / chemprop_v2_v3 18 / cfm 8.

**S2 — the repo's own census test passes.** MEASURED.

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_wfa169_gate_count_pins.py -q
-> 18 passed, 15 warnings in 64.28s
```

**S3 — every formerly-covered file is still scanned.** MEASURED. The union of
`KNOWN_PIN_FILES` (6), `WFA172_REPAIRED_FILES` (4) and `WF_A169_RED_FILES` (6) is **9**
files; all 9 still produce census rows (`coverage_complete_scanned: true`). No file was
silently dropped from the census.

**S4 — HEAD's SSOT matches HEAD's runtime.** MEASURED, without a checkout
(`git show HEAD:molmetal/scripts/pre_flight_check.py` piped to `exec(compile(...))`
with `__file__` set to the repo path):

```
HEAD live    : all 126 / launch 128 / frontier 25 / chemprop_v2_v3 18 / cfm 7
HEAD expected: all 126 / launch 128 / frontier 25 / chemprop_v2_v3 18 / cfm 7
HEAD match   : True
```

The branch brief's HEAD numbers are correct.

---

## D1 — the retired-pin count in the repair's own docstring contradicts the census artifact

MEASURED. The Wf-A169 census artifact recorded **15** stale pins in 4 files:

```
molmetal/reports/wfa169_dead_gate_repair/pin_census.md:22   | — of which STALE | 15 | 4 |
molmetal/reports/wfa169_dead_gate_repair/pin_census.json     stale_pin_count 15, stale_pin_file_count 4
   test_wf323f_m4_sa_qed_subcriterion.py 262 (all, 57)   263 (frontier, 24)          -> 2
   test_wf433a1_gate_v2_wire.py           108 120 139 140 365                        -> 5
   test_wfa127_residue_closure.py        152 153 154 155                            -> 4
   test_wfa128_preflight_classification.py 482 483 484 485                          -> 4
```

The repair's census test says otherwise, at
`molmetal/tests/test_wfa169_gate_count_pins.py:289-291`:

> "moved **ten** stale count pins across **three** files onto the SSOT constant, and moved
> the **four** stale baseline assertions in `test_wf433a1_gate_v2_wire.py`"

10 + 4 = **14**, but the census it cites measured **15** — `test_wf433a1_gate_v2_wire.py`
had **five** stale rows, not four. The branch brief repeats the same miscount ("10 stale
executable gate-count pins were repaired across 4 files" — the 10 pins were in 3 files).
Under AGENTS.md §1 rule 1 / §5.2 a retired-pin count is a number that must trace to the
artifact; here it does not.

## D2 — 5 of the 8 remaining "ok" pins are the SAME hard-coded-literal class, and the census's own tests FREEZE them

MEASURED. The 8 `ok` rows break down as 3 cross-source guards (no literal of their own)
and **5 hard-coded literals** — exactly the class Wf-A172 retired:

| file:line | kind | pinned | class |
|---|---|---|---|
| test_preflight_ssot_guard_wiring.py:312 | assert-doc-derived | 129 | doc→code sync guard (legitimate) |
| test_wfa127_residue_closure.py:108 | assert-doc-derived | 129 | doc→code sync guard (legitimate) |
| test_wfa127_residue_closure.py:142 | assert-doc-derived | 129 | file→code sync guard (legitimate) |
| **test_wf350a2_compute_manifest.py:361** | expected-gate-counts-const | 25 | **literal — same class** |
| **test_wf350c1_gpu_env_probe.py:544** | expected-gate-counts-const | 129 | **literal — same class** |
| **test_wf415a3_preflight_fail_audit.py:150** | summary-total | 129 | **literal — same class** |
| **test_wfa159_timeout_verdict_census.py:430** | len-build-gates | 129 | **literal — same class** |
| **test_wfa159_timeout_verdict_census.py:431** | len-build-gates | 131 | **literal — same class** |

`test_wf350c1_gpu_env_probe.py:538-543` is itself a comment recording that this pin was
stale (94) and was re-synced — the same recurring failure, re-synced rather than retired.
These 5 are green only because the working tree's counts happen to match; they go red on
the next legitimate gate addition (they were red at HEAD: 94 / 117 / 129 / 131 against
HEAD's 126 / 128).

**The freeze is the sharper finding.** The census test cannot be updated to allow their
repair without editing the census test itself:

- `test_wfa169_gate_count_pins.py:355-356` — `assert len(exact) == 2` and
  `assert all(p.status == "ok" ...)` for `test_wfa159_timeout_verdict_census.py`.
- `test_wfa169_gate_count_pins.py:311-314` — pins the site
  `("molmetal/tests/test_wf415a3_preflight_fail_audit.py", 150)` and asserts
  `by_site[site].pinned == pins.live_gate_counts()["all"]`.
- `test_wfa169_gate_count_pins.py:279` — `assert len(exact) >= len(KNOWN_PIN_FILES)`
  (= 6).

MEASURED derivation from `ok_pin_count` = 8: retiring the 5 literals leaves 3, and
`3 >= 6` is false, so the suite REDs if the remaining 5 are repaired. The census therefore
**locks in** the defect class it exists to eliminate.

## D3 — NO test guards the repo-wide "0 stale pins" invariant

MEASURED by grep over `molmetal/tests/` and `molmetal/scripts/`:

```
grep -rn "stale_pin_count" molmetal/tests/ molmetal/scripts/ | grep -v wfa169_gate_count_pins.py
-> (no hits)
```

Inside the census test, `stale_pin_count` appears exactly once, at line 330, inside
`test_the_census_is_not_vacuous`, as a *self-consistency* check
(`len(stale) == census["stale_pin_count"]`). Nothing anywhere asserts
`stale_pin_count == 0`. The "0 stale pins repo-wide" state is therefore a one-time
measurement with **no ongoing guard** — precisely the gap that produced the 15 stale pins
in the first place (Wf-A154 re-synced 1 of 6 files and missed the other 5).

## D4 — the census scans every `.py` under `molmetal/reports/` except ONE directory, so report artifacts flip it red

MEASURED. `_iter_python_files` delegates to `wfa156_text_scan.iter_repo_files(...,
untracked=True)`, and the only exclusion is `CENSUS_OUTPUT_REL` =
`molmetal/reports/wfa169_dead_gate_repair` (`wfa169_gate_count_pins.py:99-106`). I placed a
5-line fixture inside this branch's own report directory:

```
molmetal/reports/wfa175_pin_repair_refutation/neg_control_literal.py
    def test_x(): assert len(build_gates("all")) == 57
```

and the census went from **0 stale** to **6 stale** (5 more from a copy of HEAD's
`test_wf433a1_gate_v2_wire.py`):

```
stale_pin_count: 6 ['molmetal/reports/wfa175_pin_repair_refutation/head_test_wf433a1.py',
                     'molmetal/reports/wfa175_pin_repair_refutation/neg_control_literal.py']
```

(Those two `.py` files were deleted afterwards — they are not part of this deliverable, and
the census is back to 0 stale.) The repo holds ~70 report directories; a future wave that
ships a `.py` fixture or a patched-file copy in a report directory silently turns the
"0 stale pins repo-wide" claim red, and the census's own test suite would NOT catch it
(it asserts no stale pin only inside the 4 repaired files).

## D5 — the repaired `test_wf433a1_gate_v2_wire.py` is still RED: a FALSE GREEN at file level

MEASURED:

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
  molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py molmetal/tests/test_wfa127_residue_closure.py \
  molmetal/tests/test_wfa128_preflight_classification.py molmetal/tests/test_wf433a1_gate_v2_wire.py -q --tb=no
-> 25 failed, 60 passed
   24 x test_wf432a1_each_new_gate_appears_in_all_pipeline[...]
    1 x test_wf432a1_all_32_new_gates_present_in_build_gates
```

The other three repaired files are green. In the fourth, the 4 repaired count assertions
pass, but the file's `EXPECTED_NEW_GATES` tuple still names **32** gates of which only the
8 mock/honest survivors are registered in `build_gates("all")` — the other 24 were removed
by the Wf-437 A2 audit. Measured both trees: the live mock/honest set is **8** at HEAD and
**8** in the working tree; the 3 working-tree-only gates are
`p3_run_ledger_provenance`, `p3_ssot_census_claim_check`, `p3_ssot_path_reference_check`.

These are gate-**presence** assertions, not count pins, so the census's vocabulary cannot
see them. The census's green light on that file is therefore not a statement about the
file's health.

## D6 — the repair shrank the census's stale-DETECTABLE surface from 9 files / 24 pins to 6 / 8

MEASURED from the pre-repair artifact vs the live census:

| | before (pin_census.json, 2026-11-01 12:31) | after (live) |
|---|---|---|
| Counter | stale 15, dynamic 14, ok 9, floor 6, derived-unresolved 1 | dynamic 20, ok 8, floor 6, derived-unresolved 1 |
| exact-status files | 9 | 6 |
| executable pins | 24 | 8 |

The three files that left the exact-status set are `test_wf323f_m4_sa_qed_subcriterion.py`,
`test_wf433a1_gate_v2_wire.py` and `test_wfa128_preflight_classification.py`: they now
contribute only rows that **cannot** be stale (`dynamic` = runtime-vs-SSOT self-comparison,
`floor` = monotonic). `KNOWN_PIN_FILES`
(`test_wfa169_gate_count_pins.py:74-81`) was edited down to the new, smaller 6-file set
rather than holding the pre-repair 9. This is inherent to the repair form — a
runtime-vs-SSOT assertion cannot be stale — but it must be stated, because the census's
"0 stale" now covers a third fewer stale-detectable sites than it did the day before.

## D7 — the repair's own loop form is invisible to the census, against the branch brief's own warning

The branch brief warns: "the census resolves a pipeline ONLY from a STRING CONSTANT, so the
replacements must name pipelines literally ... a loop over a tuple would make the file drop
out of the census entirely." The Wf-A172 repair in
`molmetal/tests/test_wf433a1_gate_v2_wire.py:180-186` does exactly that:

```
for pipeline in ("frontier", "chemprop_v2_v3", "cfm"):
    runtime = len(_pfc.build_gates(pipeline))
    expected = _pfc.EXPECTED_GATE_COUNTS[pipeline]
    assert runtime == expected
```

MEASURED with two synthetic controls (written to a temp dir, then deleted):

```
literal form  assert len(build_gates("all")) == 57
              -> [('stale','len-build-gates','all',57)]
loop form     for pipeline in ("frontier","chemprop_v2_v3","cfm"):
                  assert len(build_gates(pipeline)) == 57
              -> []            # ZERO pins — invisible
```

Consequence: the census's only row for `test_wf433a1_gate_v2_wire.py` is now the pre-existing
floor at line 475 (`assert len(runners) >= 127`), so that file's "no stale pin" verdict
rests on a row unrelated to the repaired assertions. And the SSOT-vs-runtime guard for
`frontier` / `chemprop_v2_v3` / `cfm` is guarded nowhere else: the pre-flight's single
runtime consistency gate, `p3_gate_count_consistency_audit`
(`molmetal/scripts/pre_flight_check.py:10566`, registered at `:15669-15674` with
`expected_total=EXPECTED_GATE_COUNTS["all"]`), covers the **`all` pipeline only**.

## D8 — HEAD-robustness: the new assertions hold, the repaired FILE does not

MEASURED. The 4 repaired files' new assertions are runtime-vs-SSOT, so they hold on any tree
whose SSOT tracks its own runtime — HEAD's does (S4), and the 8 survivor gate names in the
new `test_wf432a1_no_existing_gate_renamed_or_removed` are registered at HEAD as well
(measured: 0 missing at HEAD, 0 missing in the working tree).

But:
- **`test_wf433a1_gate_v2_wire.py:475` keeps a leftover floor** `assert len(runners) >= 127`.
  Rescanning the working tree with HEAD's live counts reports it:
  `floor-violated molmetal/tests/test_wf433a1_gate_v2_wire.py 475 len-of-build-gates-floor all pinned=127 live=126`.
  The repaired file is therefore RED on a clean checkout of HEAD.
- **`test_wfa127_residue_closure.py` and `test_wfa128_preflight_classification.py` do not
  exist at HEAD** (`git status` → `??`). Their HEAD-robustness is vacuous.
- The same rescan with HEAD's live counts reports **7 stale** pins (they are green only
  against the working tree's 129/131): `test_preflight_ssot_guard_wiring.py:312`,
  `test_wf350c1_gpu_env_probe.py:544`, `test_wf415a3_preflight_fail_audit.py:150`,
  `test_wfa127_residue_closure.py:108` and `:142`,
  `test_wfa159_timeout_verdict_census.py:430` and `:431`. The "0 stale pins repo-wide"
  claim is a working-tree statement, not a tree-independent one.

---

## §summary of judgements

| # | Claim under test | Verdict |
|---|---|---|
| S1 | 0 stale pins; Counter as stated | CONFIRMED |
| S2 | repo census test passes | CONFIRMED (18 passed) |
| S3 | all formerly-covered files still scanned | CONFIRMED (9/9) |
| S4 | HEAD SSOT == HEAD runtime | CONFIRMED (126/128/25/18/7) |
| D1 | "10 + 4" retired pins | REFUTED — the cited artifact measured 15 |
| D2 | the 8 "ok" pins are a different class | REFUTED for 5 of 8; they are the same class, and the census tests freeze them |
| D3 | the zero-stale state is guarded | REFUTED — no test asserts it |
| D4 | the census is robust to report artifacts | REFUTED — a 5-line report fixture → 6 stale |
| D5 | the repaired files are clean | REFUTED for 1 of 4 — 25 failures remain (gate presence, outside the census's vocabulary) |
| D6 | coverage held | REFUTED — exact-status coverage 9 files/24 pins → 6/8 |
| D7 | the repair used census-visible forms | REFUTED — the loop form is invisible (measured) |
| D8 | HEAD-robust | PARTIAL — new assertions hold; the repaired file is RED at HEAD (floor ≥ 127 vs 126) and 2 of 4 files do not exist at HEAD |

**Recommendation (read-only, no writes performed):** the minimal follow-up is to (i) retire
the 5 remaining literal pins the same way the 15 were retired and relax the three
census-test assertions that freeze them, (ii) add a repo-wide
`assert census["stale_pin_count"] == 0` guard, (iii) exclude `molmetal/reports/**` from the
census scan the way `CENSUS_OUTPUT_REL` excludes one directory today, and (iv) replace the
tuple loop in `test_wf433a1_gate_v2_wire.py` with literal-named assertions so the census can
see them. None of these were applied here — this branch is read-only.
