# Wf-A175 — adversarial refutation of the PINS-branch audit

Date: 2026-11-01. Window: 19:12–19:24 local. CPU-only throughout
(`CUDA_VISIBLE_DEVICES=""` + `/home/hugo/codes/mol-metal/.venv/bin/python`).
READ-ONLY on `paper/**`, `docs/HONEST_NEGATIVE_CATALOG.md`, `AGENTS.md`,
`TODO/**`, `README.md`, `molmetal/scripts/pre_flight_check.py`, `metrics/**`.
Writes confined to `molmetal/reports/wfa175_pin_repair_refutation/refutation/`
and written via `wf386a3_atomic_write.atomic_write_file` (AGENTS.md §5.6).
No git mutation.

**VERDICT: 0 of 9 claims refuted. 5 SURVIVED, 4 IMPRECISE.** Every load-bearing
number in the PINS audit reproduced exactly when I re-measured it — including
the 0→6 stale flip, the 25-failed pytest, and the 8-of-32 gate registration.
The four IMPRECISE verdicts are wording/citation defects in the audit, not
errors of substance; the precise wording is given per claim below.

**The tree moved mid-audit.** At 19:17:33 a concurrent branch rewrote
`molmetal/tests/test_wf433a1_gate_v2_wire.py` (mtime), replacing the
gate-PRESENCE assertion with a deferral tripwire. Claim 6's "25 failed / 60
passed" reproduced EXACTLY at 19:14 and is now 62 passed / 0 failed at 19:20.
Two cited line numbers shifted with it (:475 → :635, :199 → :341).
`molmetal/scripts/pre_flight_check.py` has not been touched since 11:54:50,
so the live/SSOT counts are unaffected.

---

## CLAIM 1 — 0 stale pins + the exact Counter → **SURVIVED**

Live `pin_census()` at 19:12 and again at 19:19:45:

    stale_pin_count = 0   stale_pin_files = []
    Counter({'dynamic': 20, 'ok': 8, 'floor-satisfied': 6, 'derived-unresolved': 1})
    counts_match = True
    live == expected == {all 129, launch 131, frontier 25, chemprop_v2_v3 18, cfm 8}

`pytest molmetal/tests/test_wfa169_gate_count_pins.py` → **18 passed**
(65.78 s), run twice.

## CLAIM 2 — retired-pin count 15 vs "ten + four" → **SURVIVED**

`molmetal/tests/test_wfa169_gate_count_pins.py:289-291`: "moved ten stale
count pins across three files onto the SSOT constant, and moved the four stale
baseline assertions in `test_wf433a1_gate_v2_wire.py`". The artifact
`molmetal/reports/wfa169_dead_gate_repair/pin_census.md:22` says
"— of which STALE | 15 | 4", and rows 3–7 of its table are **5** stale pins in
`test_wf433a1_gate_v2_wire.py` (lines 108, 120, 139, 140, 365).
`pin_census.json`: stale 15 / 4 files, executable 24 / 9 files.
Documented ≤14 vs MEASURED 15 → at least one retired pin is unaccounted for.
(The "ten … across three files" half matches the artifact exactly.)

## CLAIM 3 — 5 of the 8 "ok" pins are the same class, and are frozen → **SURVIVED**

Live "ok" rows: `wf350a2:361` (25), `wf350c1:544` (129), `wf415a3:150` (129),
`wfa159:430` (129), `wfa159:431` (131) are hard-coded literals in the retired
forms (`EXPECTED_GATE_COUNTS["frontier"] == 25`,
`preflight_summary["total"] == 129`, `len(build_gates("all")) == 129`) — the
same two shapes the repair retired (`patch_resync_stale_pins.diff`:
`assert len(build_gates("all")) == 127`, `assert EXPECTED_GATE_COUNTS["all"] == 127`).
The other three (`preflight_ssot_guard_wiring:312`, `wfa127:108`, `wfa127:142`)
are doc-derived cross-checks, not literals.

Freezing confirmed: `:279` is `assert len(exact) >= len(KNOWN_PIN_FILES)` with
`len(KNOWN_PIN_FILES)==6` → retiring 5 literals leaves exact=3 < 6 → FAIL;
`:355` `assert len(exact) == 2` for the wfa159 file → FAIL if its 2 literals go;
`:311-314` site-pins `("…test_wf415a3_preflight_fail_audit.py", 150)` and
requires the row to EXIST with status "ok" → FAIL if the literal is replaced.
Shorthand only: ":279 len(exact)>=6" is literally
`len(exact) >= len(KNOWN_PIN_FILES)`.

## CLAIM 4 — zero-stale is unguarded → **IMPRECISE**

Load-bearing half CONFIRMED: `stale_pin_count` occurs in the test file ONLY at
`:330` (`assert len(stale) == census["stale_pin_count"]` — self-consistency);
`stale_pin_file_count` only at `:331`. Repo-wide `git grep -ln stale_pin -- .`
over TRACKED files returns **ZERO hits** — the census script AND its test are
both untracked, so no pre-commit hook and no CI job can see them; `.pre-commit-config.yaml`
and `.github/workflows/ci.yml` contain no pin-census gate. No repo-wide
zero-stale assertion exists.

BUT "no test anywhere asserts it is 0" is **false as written**: a FILE-SCOPED
zero-stale guard exists and holds — `test_the_wfa172_repaired_files_hold_no_stale_pin`
(`:286-305`) asserts no stale pin in each of the 4 `WFA172_REPAIRED_FILES`.
PRECISE: *the repo-wide zero-stale invariant is unguarded; the only zero-stale
assertion is scoped to the 4 repaired files*. Consequence MEASURED: my
report-dir fixture carried 6 repo-wide stale pins and the census suite still
reported **18 passed**.

## CLAIM 5 — census counts report-dir `.py` files → **SURVIVED**

Mechanism read from source and reproduced live. `wfa169_gate_count_pins.py:98`
`CENSUS_OUTPUT_REL`; `:724-737` `_iter_python_files` has **no** exclusion;
`:995` `pin_census` calls `scan_gate_count_pins(...)` without the `exclude`
parameter the function declares at `:853`; only the PROSE scan (`:765`) excludes
`CENSUS_OUTPUT_REL`.

EXPERIMENT (fixture inside this report dir, since deleted):

    before          : stale 0, files_scanned_python 3054
    with fixture    : stale 6, stale_pin_files = [.../refutation/_probe_fixture.py]
                      files_scanned_python 3055
                      rows 57/127/24/14/6/94 vs live 129/131/25/18/8
    census suite    : 18 passed (WITH the fixture present)
    after delete    : stale 0, Counter {dynamic 20, ok 8, floor-satisfied 6, derived-unresolved 1}

Two corrections to the supporting clauses: (a) for the EXECUTABLE-pin scan the
exclusion set is **empty**, not "only CENSUS_OUTPUT_REL" — the census's OWN
output dir is scanned too (its 3 `.py` files `regen_block.py`, `regen_patch.py`,
`wfa169b2_exec_verdict.py` are in the scan and yield 0 rows today), so the
docstring's "output directory is EXCLUDED from the scan" is prose-only-true;
(b) "5-line" is fixture-content-dependent (mine: 11 lines, 6 stale pins).

## CLAIM 6 — one repaired file still RED; 8 of 32 gates registered → **SURVIVED (time-indexed)**

Reproduced EXACTLY at 19:14: pytest of the 4 repaired files → **25 failed,
60 passed** (16.15 s), every failure in `test_wf433a1_gate_v2_wire.py` =
1× `all_32_new_gates_present_in_build_gates` + 24× `each_new_gate_appears_in_all_pipeline[…]`.
Registration MEASURED both ways: `EXPECTED_NEW_GATES` = 32; in-tree
`build_gates("all")` registers **8** (24 missing); at HEAD (`exec` of
`git show HEAD:molmetal/scripts/pre_flight_check.py`, no checkout; HEAD
`build_gates("all")`=126) also **8** (24 missing). `metrics/` verified unchanged.

SUPERSEDED at 19:17:33 by the concurrent rewrite — the same 4-file pytest now
returns **62 passed / 0 failed** at 19:20. The 8-of-32 fact stands
(`pre_flight_check.py` untouched since 11:54:50).

## CLAIM 7 — coverage regressed; KNOWN_PIN_FILES edited down → **IMPRECISE**

First clause CONFIRMED: pre-repair `pin_census.json` = executable **24 pins /
9 files** (stale 15/4, ok 9, floor 6, dynamic 14); live now = **8 pins / 6
files**. The three dropped files (wf323f, wf433a1, wfa128_preflight) now carry
only dynamic/floor rows.

Second clause WRONG: `KNOWN_PIN_FILES` was **not edited down**. Its length was
6 BEFORE the repair and is 6 AFTER — the test's own SUPERSEDED tuple
(`:255-262`, preserved verbatim) lists the pre-repair 6 =
{wfa127, wfa128, wf433a1, wf350c1, wf323f, wf415a3}; today's 6 =
{preflight_ssot_guard_wiring, wf350a2, wf350c1, wf415a3, wfa127, wfa159} — a
**3-out / 3-in swap of equal length**. The guard's form did not weaken either:
the bound is `len(exact) >= len(KNOWN_PIN_FILES)`, and the 3 dropped files stay
covered (dynamic/floor rows) by the `WFA172_REPAIRED_FILES` zero-stale guard.
PRECISE: *exact-status coverage regressed 9 files/24 pins → 6 files/8 pins;
file-level coverage of all 9 was preserved and the tuple was swapped 3-for-3.*

## CLAIM 8 — the loop form is invisible to the census → **IMPRECISE (wrong line citation)**

Synthetic probes via `scan_file_for_pins` on tmp files:

    'assert len(build_gates("all")) == 57'        -> 1 row, status stale, pinned 57, live 129
    for pipeline in ("frontier","chemprop_v2_v3","cfm"):  -> 0 rows of ANY kind
    'assert len(build_gates("all")) == EXPECTED_GATE_COUNTS["all"]' -> 1 row, status 'dynamic'

So the exact-pin vocabulary (ok/stale/unknown-pipeline) cannot see the loop
form at all, while it sees the direct-SSOT form as "dynamic".
`pre_flight_check.py:15669` is indeed the ONLY explicit runtime consistency
registration (`grep expected_total=` → one hit, `EXPECTED_GATE_COUNTS["all"]`)
→ "all" only.

CITATION ERROR: the Wf-A172 loop is **not** at `test_wf433a1_gate_v2_wire.py:180-186`
(lines 180-186 are the repair's explanatory comment). It was at **:199**
(`for pipeline in (...)`, assert at :202) at measurement time and is at **:341**
after the concurrent rewrite.

## CLAIM 9 — HEAD-robust is PARTIAL → **IMPRECISE**

HEAD SSOT == HEAD runtime CONFIRMED by `exec` of
`git show HEAD:molmetal/scripts/pre_flight_check.py` (no checkout): SSOT and
runtime both {all 126, launch 128, frontier 25, chemprop_v2_v3 18, cfm 7}.
`git cat-file -e HEAD:…` for `test_wfa127_residue_closure.py` and
`test_wfa128_preflight_classification.py` → "exists on disk, but not in HEAD" →
CONFIRMED absent at HEAD. The WT floor `assert len(runners) >= 127` was at
:475 (grep 19:16) and the census classified it as `len-of-build-gates-floor`
pinned 127 / floor-satisfied / live 129; after the rewrite it is at :635.

COUNTER to the wording: "floor-VIOLATED at HEAD's 126" mixes two trees.
HEAD's own copy of the file is 385 lines and contains **no** `>= 127` floor —
it carries `assert _pfc.EXPECTED_GATE_COUNTS["all"] == 127` (:108) and
`assert len(set(names)) == 127` (:120). The floor exists only in the working
tree. PRECISE: *the WT-only floor is satisfied in the WT tree (129 ≥ 127) but
is incompatible with HEAD's 126-gate runtime, so the WT test file cannot be run
against HEAD's pre-flight; HEAD's own copy has no such floor.* "The 4 new
assertions are runtime-vs-SSOT" could not be independently enumerated (the
census test is untracked → no diff); it is consistent with what I read
(`:275-314` compare against `pins.live_gate_counts()` and against counts
re-derived from the same census run).

---

## Supplementary findings (not among the 9 claims)

1. **The census is untracked.** `git ls-files` returns neither
   `molmetal/scripts/wfa169_gate_count_pins.py` nor
   `molmetal/tests/test_wfa169_gate_count_pins.py`. A clean checkout of HEAD
   contains **no pin census at all** — the 0-stale property exists only in the
   working tree, and no CI/pre-commit hook can enforce it.
2. **The census test depends on untracked files.** `test_the_census_covers_every_wfa169_red_file`
   requires `test_wfa127_residue_closure.py` and `test_wfa128_preflight_classification.py`
   to exist on disk; committing the census test without them breaks a clean
   checkout.
3. **The census's own output dir is not excluded from the executable scan**
   (prose-only exclusion) — 3 `.py` files there are scanned, 0 rows today.

## Repo state after this branch

Only `molmetal/reports/wfa175_pin_repair_refutation/refutation/` was written.
The probe fixture `_probe_fixture.py` was deleted and the census re-verified
(19:19:45: stale 0, Counter {dynamic 20, ok 8, floor-satisfied 6, derived-unresolved 1}).
`metrics/` unchanged. No git mutation.
