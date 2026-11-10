# Wf-A170 — Synthesis (decision-ready brief)

**Tag: MEASURED** (2026-11-01). CPU-only, **0 GPU** (`CUDA_VISIBLE_DEVICES=""` on every
invocation), single venv `/home/hugo/codes/mol-metal/.venv/bin/python`.
Every number below traces to a file on disk. `metrics/` was **UNCHANGED** before/after every
pytest run in this synthesis (`git status --porcelain metrics/` md5 `a022e54fd2938b9f333fd8438f6d3d27`, 38 lines, diff empty).

---

## §0 — HEADLINE

Three branches, three adversarial refutations, **nothing committed**: every Wf-A170 output is an
uncommitted working-tree file. The measured RED surface moved **35 → 32** failures over the
six-file set; the Wf-A159 census artefact is a **RE-PIN** (site list unchanged, 35/35 TIMEOUT /
0 UNHANDLED before and after); the census-freshness census measures **39 sha-pinning artefacts,
1823 pins, 160 STALE**; the Wf-A169 write-up carries **three material errors** plus a fourth
mis-stated bucket; and the one remaining user-facing decision is unchanged — **the 24 gates stay
unregistered until the user rules.**

---

## §1 — WHAT EACH BRANCH CHANGED (files, one line each; SHIPPED status)

### Branch 1 — GPU-required test hygiene (REPAIR)

| file | change | SHIPPED? |
|---|---|---|
| `molmetal/tests/test_wf350c1_gpu_env_probe.py` (M, tracked) | count pin `== 94` → `== 129` (superseded `94` kept VERBATIM in a dated comment, AGENTS.md §5.5 rule 6); the two SKIP-vs-PASS assertions made two-sided behind a new `_session_gpu_masked()` branch (env-only, no torch, no subprocess); the GPU-visible `else:` branch keeps the ORIGINAL assertions verbatim | **NOT COMMITTED** (working tree only) |
| `molmetal/tests/test_wfa170_gpu_required_test_hygiene.py` (??, new) | 4 tests: negative control (blank `CUDA_VISIBLE_DEVICES` ⇒ gate SKIPs, never PASSes, evidence non-empty) + AST guards (≥1 unconditional AND ≥1 conditional assert per repaired test; no blanket `skipif`) + end-to-end subprocess re-run | **NOT COMMITTED** |

MEASURED: `3 passed` + `4 passed` = **7 passed / 0 failed** (53.33 s). `metrics/` unchanged.
AST census of the repaired file: T1 9 unconditional / 8 conditional asserts; T2 5 / 2; T3 9 / 2.
Not verified (0-GPU rule): the GPU-visible `else:` branches were never executed.

### Branch 2 — census freshness

| file | change | SHIPPED? |
|---|---|---|
| `molmetal/reports/wfa159_timeout_verdict_census/timeout_verdict_census.json` | **REGENERATED** by the producer: `target_sha256` `56ac01199c82…` → `ae878bcd5249…` (= live `pre_flight_check.py` sha), `target_lines` 17436 → 17608 | **NOT COMMITTED** (and git-ignored by `.gitignore:158` ⇒ `molmetal/reports/**/*.json`) |
| `molmetal/scripts/wfa170_census_freshness_guard.py` (??, new) | stdlib-only guard (no torch, no subprocess; git HEAD read from `.git` as files) | **NOT COMMITTED** |
| `molmetal/tests/test_wfa170_census_freshness_guard.py` (??, new) | 15 tests: negative + positive control, vacuity guard, AST+runtime CPU-only guard, read-only guard, report-truthfulness | **NOT COMMITTED** |
| `molmetal/reports/wfa170_census_freshness/{report.json,report.md,patch_preflight_gate.diff}` (??, new) | the census report + a **SPEC** patch; the patch was **NOT applied** to `pre_flight_check.py` (`git apply --check` rc=0 on a scratch copy only) | **NOT COMMITTED** |

### Branch 3 — adversarial re-derivation of the Wf-A169 finding set

| file | change | SHIPPED? |
|---|---|---|
| `molmetal/reports/wfa170_refutation/{refutation.json,refutation.md}` (??, new) | read-only re-derivation; **no repo file touched** | **NOT COMMITTED** |

### Refutation artefacts (all `??`, untracked)

`molmetal/reports/wfa170_gpu_required_test_repair_refutation/` ·
`molmetal/reports/wfa170_census_freshness_refutation/` ·
`molmetal/reports/wfa170_refutation/`

### Pre-existing uncommitted state (NOT this wave's)

`molmetal/scripts/pre_flight_check.py` (M) carries the Wf-A169 **D1/D2** hook-emitter repair
(sha256 `ae878bcd…`, 734,797 B, mtime 11:54:50) — **uncommitted**.
`molmetal/scripts/wfa159_timeout_verdict_census.py` is **untracked**.

---

## §2 — THE CURRENT RED TEST SURFACE (MEASURED)

Command (six files, CPU-locked):

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q --no-header -p no:cacheprovider \
  molmetal/tests/test_wf433a1_gate_v2_wire.py molmetal/tests/test_wf350c1_gpu_env_probe.py \
  molmetal/tests/test_wfa127_residue_closure.py molmetal/tests/test_wfa128_preflight_classification.py \
  molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py molmetal/tests/test_wfa159_timeout_verdict_census.py
```

### Before → after

| state | failed | passed | source on disk |
|---|---|---|---|
| 2026-11-01 12:03 | 36 | 72 | `docs/HONEST_NEGATIVE_CATALOG.md:21182` (§6.266.2) |
| 2026-11-01 12:19 | 35 | 73 | `molmetal/reports/wfa169_dead_gate_repair/stale_pin_diagnosis.md:26` |
| 2026-11-01 12:23 (5-file set) | 35 | 53 | `wfa170_refutation/refutation.json` |
| 2026-11-01 12:31 (5-file set) | 32 | 56 | same |
| **2026-11-01, MEASURED BY THIS SYNTHESIS (six-file set)** | **32** | **76** | 62.53 s |
| claimed "29 failed / 79 passed" (post-patch) | 29 | 79 | **NOT REPRODUCIBLE** — the re-sync was applied to **1 of 5** files |

Δ since 12:19 = **−3 failures / +3 passed**, all three from `test_wf350c1_gpu_env_probe.py`
(branch 1's repair). The 5-file set is today's six-file set minus
`test_wfa159_timeout_verdict_census.py` (20 collected tests, 0 failures) ⇒ 32/56 ✓, so the
refutation's 12:31 number is today's state.

### Exact breakdown of what remains RED (32) and WHY

| # | class | test ids (MEASURED) | why it is RED | who can close it |
|---|---|---|---|---|
| **24** | unregistered gates | `test_wf433a1_gate_v2_wire.py::test_wf432a1_each_new_gate_appears_in_all_pipeline[…]` (24 params) | the 24 Wf-432-A1 gates are DEFINED (`p3_protocol_alignment_dataset_match` occurs exactly **once** in `pre_flight_check.py` — the `def`) and registered in **NO** pipeline | **USER** (§7) |
| **1** | same class | `test_wf433a1_gate_v2_wire.py::test_wf432a1_all_32_new_gates_present_in_build_gates` | asserts all 32 Wf-432-A1 gates are present in `all` | **USER** |
| **7** | stale count pins | `…::test_wf432a1_expected_gate_count_const_127`, `…::test_wf432a1_build_gates_all_unique_count_is_127`, `…::test_wf432a1_frontier_pipeline_subset_unchanged`, `…::test_wf432a1_no_existing_gate_renamed_or_removed`, `test_wfa127_residue_closure.py::test_the_gate_counts_are_what_this_wave_measured`, `test_wfa128_preflight_classification.py::test_the_registered_gate_count_did_not_change`, `test_wf323f_m4_sa_qed_subcriterion.py::test_wf323f_expected_gate_counts_bumped` | pins `127` / `14` / `6` / `"all": 57` / `"frontier": 24` vs the live `129` | **mechanical** — `patch_resync_stale_pins.diff` staged, `git apply --check` PASS, **unapplied** |
| **0** | session artefacts | — | the 2 SKIP-vs-PASS failures lived in `test_wf350c1_gpu_env_probe.py` (HEAD `:170` `status == "PASS"`, `:274` `status in ("PASS","FAIL")` — the gate returns **SKIP** under the repo's import-time CPU lock) and are now GREEN under branch 1's two-sided contract | closed |
| **0** | artefact-staleness | — | the 1 census-artefact failure was cleared by branch 2's regeneration; `test_wfa159_timeout_verdict_census.py` = **20 passed** | closed |
| **32** | | | | |

Also MEASURED: live gate counts by import — `all=129 launch=131 frontier=25 chemprop_v2_v3=18 cfm=8`
== `EXPECTED_GATE_COUNTS` (worktree). **HEAD's** constant is `{all: 126, launch: 128, frontier: 25,
chemprop_v2_v3: 18, cfm: 7}` (AST-parsed from `git show HEAD:…`).

Applying the staged re-sync takes the surface **32 → 25** (the 24 + the all-32). Wiring the 24
takes it to **0**.

---

## §3 — THE CENSUS ARTEFACT: **RE-PIN ONLY** (site list unchanged)

Site-level diff, MEASURED by this synthesis (old = the producer re-run on the 17436-line
predecessor, `/tmp/wfa159/census_patched.json`; new = the shipped artefact):

* site sets **EQUAL** — 35 sites / 31 distinct `(enclosing_function, call)` keys;
* `old["summary"] == cur["summary"]` → **True**: **35 sites / 35 TIMEOUT / 0 UNHANDLED**
  (denominator = **35**), `by_role` producer 1 / gate 31 / helper 3, `by_effect`
  `PRODUCES_RC_124` 1 / `TIMEOUT_VERDICT` 34 / rest 0;
* the **only** substantive field that moved is `line`, on **5 sites**: **+62, +143, +158, +158, +172**
  — the file grew 17436 → 17608 lines;
* `file` differs only because the predecessor run named the scratch path
  `/tmp/wfa159/pre_flight_check.py`.

⇒ **The site list did NOT change substantively. Only the sha moved**
(`56ac01199c82…` → `ae878bcd5249…`). The old pin matched **NEITHER** HEAD (`4c648e48237b…`,
15487 lines) **NOR** the worktree — it was a snapshot of a file that no longer exists.

---

## §4 — THE CENSUS-FRESHNESS CENSUS

`molmetal/reports/wfa170_census_freshness/report.md` (status **FAIL**):

* JSON artefacts scanned: **3993** (30 unparseable);
* artefacts carrying ≥1 SHA-256 pin: **39** (**denominator**);
* pins checked: **1823** ⇒ FRESH **1656**, STALE **160**, MISSING_TARGET **7**,
  SKIPPED_TOO_LARGE 0, UNPAIRED **138**;
* **Primary scope** (`target_sha256` + `target_file`): **1** artefact, **0 stale** (the wfa159
  census itself is now FRESH);
* **Generalized scope** (any `<stem>_sha256` + named file): **38** artefacts, **1822** pins,
  **160 stale** in 4 artefacts — `wfa166_run_ledger_census/census.json`,
  `wfa166_run_ledger_provenance/audit_real_ledger.json`,
  `wf30_pocket_prep/830c/prep_meta.json`, `wf30_pocket_prep/pocket_status.json`.

### Would a FAILing gate block launches today? **NO** — as shipped it cannot

The patch is an **advisory hook inside `main()`**, is **NOT** registered in `build_gates()`, never
changes the exit code, and leaves `EXPECTED_GATE_COUNTS` byte-identical at `{all 129, launch 131}`
— **and it is not applied**. Hypothetically, if made enforcing: **primary scope NO** (0 stale of 1);
**generalized scope YES — 160 stale pins would block every launch today.** So the generalized
scope is a real backlog and a real decision, not a gate currently in the way.

---

## §5 — REFUTATION VERDICTS

| refutation | of | verdict | headline |
|---|---|---|---|
| `wfa170_gpu_required_test_repair_refutation/refutation.md` | branch 1 | **NOT REFUTED** | 8/8 checks PASS; 4 mutants each turn a test RED (negative controls bite); not vacuous (no `pytestmark` / `skipif` / decorator / `pytest.skip`; HEAD asserts 15/6/10 → now 17/7/11); exactly 2 files written; `metrics/` byte-identical |
| `wfa170_census_freshness_refutation/refutation.md` | branch 2 | **NOT REFUTED** | 8/8 PASS; producer re-run JSON-equal to shipped minus `generated_at`; 1823/160/1656/7/138/39 reproduced exactly; stale SET byte-identical to a fresh run; 160/160 stale re-hashed independently, 0 false positives; `git apply --check` rc=0; `EXPECTED_GATE_COUNTS` byte-identical |
| `wfa170_refutation/refutation.md` | the Wf-A169 finding set | **PARTLY REFUTED** | every mechanical measurement reproduces; the **CAUSE** is wrong and three of its numbers do not reproduce; the "NEVER run" framing is false |

### Surviving concerns that need action, not just a note

* **R1-A** the repaired file's docstring (lines 56-64) still says *"NONE of the 3 failures had
  anything to do with what they guard"* — **FALSE for T3**, whose declared guard IS the gate-count
  invariant. (Text-only fix.)
* **R1-B (medium)** the re-synced `== 129` is coupled to the **UNCOMMITTED** worktree — HEAD
  carries `all: 126` (AST-parsed, MEASURED by this synthesis) ⇒ a clean HEAD checkout is still
  RED; a stale literal was re-synced to another stale literal.
* **R1-C (low)** the masked branch tolerates only `SKIP`; a probe `GATE_STATUS_TIMEOUT` would
  fail it.
* **R1-E (low)** `ruff format` was never run on the repaired file (HEAD's was format-clean, 3 hunks differ).
* **R2-C-A (substantive)** mutants M4 (generalized pairing off) / M5 (primary scope off) / M6
  (targets never resolve) each leave the suite at 15/15 PASSED — the 160-stale finding is TRUE
  but deletable by a later refactor.
* **R2-C-B (substantive)** the patch's claim *"wrapped in try/except, so a missing or broken guard
  can never break a pre-flight run"* is **false for the import**: `from molmetal.scripts import
  wfa170_census_freshness_guard` is **module-level**, outside the try/except — reproduced uncaught
  `ImportError`; the guard file is also untracked.
* **R2-C-C (minor)** the guard's `_write` hand-rolls `{path}.tmp` + `os.replace` instead of §5.6's
  `wf386a3_atomic_write.atomic_write_file`.
* **R2-C-D (minor)** 4 of the 15 guard tests read the git-ignored `report.json` ⇒ they FAIL on a
  clean checkout.
* **R2-C-E (minor)** `n_json_artefacts_scanned` is a moving target (3993 → 3994 → 3998) under
  parallel branches; guard runtime MEASURED **18.2 s** (the report says ~15 s).

---

## §6 — MATERIAL CORRECTIONS TO THE Wf-A169 WRITE-UP

Target: `docs/HONEST_NEGATIVE_CATALOG.md:§6.266` (+ `molmetal/reports/wfa169_dead_gate_repair/README.md`
and `registration_table.md`).

1. **"NEVER RAN" is FALSE.** §6.266 heading + §6.266.0 ("for the 30 days since 2026-10-02, they
   have NEVER RUN") → they **RAN on 2026-10-02** (`TODO/completed/432a1_pre_training_gate_v2.md`
   §3: 149 total / 139 pass / 3 fail / 3 skip; the 32 new gates 26 PASS / 4 WARN / 2 FAIL), then
   were **deliberately removed** by the Wf-437 A2 audit.
2. **REVERSE "never written".** `registration_table.md:19` ("so no registration call was ever
   written (not 'written, then removed')") and `README.md:40` → `pre_flight_check.py:14441-14450`
   (HEAD `:12480`) is the repo's **own committed comment**: Wf-437 A2 removed **29**
   "SCAFFOLD-only doc-trackers" (15 `p3_protocol_*` + 5 `p3_mock_to_real_*` + 9
   `krasnov_*`/`rusanov_*`/`drugood_*`), keeping the 3 `p3_honest_neg_closure_*`. `git log -S`
   shows only that the **REGISTERED state was never COMMITTED**.
3. **LEG 3 counts: 115 / 119 → 133 / 137** (union over 5 pipelines; or live `all` 126 → 129).
   The 4 added names are correct; `p3_cfm_fragmentation_preflight` is `cfm`-pipeline only.
4. **LEG 2's 10,643 B gap covers 29 tuples, not 24** (367 B/tuple).
5. **§6.266.2's "36 failed / 72 passed"** is the 12:03 reading only; 12:19 over the same six
   files was **35 failed / 73 passed**; the claimed post-patch "29 failed / 79 passed" is **not
   reproducible** (re-sync applied to 1 of 5 files). MEASURED now: **32 failed / 76 passed**.
6. **§6.266.2b's "3 GPU-required-session-artefact" bucket is mis-stated — and the refutation's
   rebuttal is ALSO wrong.** HEAD's 3 `test_wf350c1_gpu_env_probe.py` failures were **1 stale pin
   (`== 94`) + 2 genuine SKIP-vs-PASS session artefacts** (`git show HEAD:…:170`, `:274`, `:334`).
   The catalog's own table + §6.266.2b are **CORRECT**; the refutation's "the bucket is EMPTY,
   they were stale pins of 94" is **wrong on 2 of the 3**.
7. **The 6th file IS identified**: `molmetal/tests/test_wfa159_timeout_verdict_census.py` —
   **20 collected tests** (MEASURED), 1 failed / 19 passed at 12:03 (the stale census artefact).
   The refutation failed to find it.
8. **Label the D1/D2 repair UNCOMMITTED** (`pre_flight_check.py`, sha256 `ae878bcd…`, 734,797 B).
9. **Correct framing**: *"deferred by an audit whose deferral was never revisited"* — 5 of the 29
   were **restored** by Wf-R42 once proven REAL (`TODO/completed/321d_preflight_gate_audit.md:148-152`);
   the remaining 24 were never re-examined. D3 is direct evidence the deferral's premise no
   longer holds.
10. **HONEST NEGATIVE SURVIVES**: wiring the 24 makes `all` **FAIL** on
    `p3_protocol_alignment_baseline_paper_refs_bib` and **WARN** on 4 optional artefacts —
    **19 PASS / 4 WARN / 1 FAIL** of 24
    (`molmetal/reports/wfa169_dead_gate_repair/apply_probe.json`).

---

## §7 — WHAT THE MAIN THREAD MUST DO NEXT

### A. CATALOG / FIX — no user decision needed

1. **Append a new §6.268** to `docs/HONEST_NEGATIVE_CATALOG.md` carrying corrections 1-10 above
   plus this wave's measured numbers (32/76 RED surface, the 35-site re-pin, the
   39/1823/160/1656/7/138 census, the advisory posture). Do **not** edit §6.266 in place
   (§5.5 rule 6: additive + dated, superseded renderings preserved VERBATIM).
2. **Additively correct** `registration_table.md:19` and `README.md:40` (keep the superseded
   wording visible + dated).
3. **Fix the three items nobody needs to approve**: the branch-1 docstring claim (R1-A),
   `ruff format` on `test_wf350c1_gpu_env_probe.py` (R1-E), and the advisory patch's
   module-level import (R2-C-B) + the guard's `_write` (R2-C-C) — before either can ship.
4. **Record the new honest-negative class**: 160 STALE SHA pins of 1822 across 38 artefacts, with
   the Wf-A166 ledger-provenance pins as the largest contributor — plus the guard's own limits
   (138 UNPAIRED pins, a 3993→3998 moving denominator, 4 of 15 tests unreadable on a clean
   checkout, 18.2 s runtime).
5. **Decide on `patch_resync_stale_pins.diff`** (7 stale pins, mechanical, `git apply --check`
   PASS): applying it takes the RED surface **32 → 25**. NOTE it re-syncs to **129**, the
   *uncommitted* worktree's count; if the 24 are ever wired a **second** re-sync to 153 is
   required.
6. **Nothing in this wave is committed.** `molmetal/reports/**/*.json` is git-ignored
   (`.gitignore:158`), so the census artefact and the guard's `report.json` need `git add -f`
   (or they die with the branch). `pre_flight_check.py` and `test_wf350c1_gpu_env_probe.py` are
   modified-but-uncommitted.

### B. ASK THE USER — the only true blockers

1. **The 24 gates.** Two sub-parts, both required:
   (a) **§5.3 OK to edit `paper/refs.bib`** — the 1 FAIL is
   `p3_protocol_alignment_baseline_paper_refs_bib`, missing `krasnov2026metalcytotoxdb`,
   `rusanov2025platinai`, `ji2022drugood`;
   (b) **a ruling on the spec's own §1.3 routing contradiction** — L5's 9 hooks do **not** route
   into `all`, so the reachable total is **118**, not the spec's 127, and wiring all 24 gives
   **153**, contradicting the spec's own routing rule.
   Until both land, `patch_wire_24_gates.diff` stays staged and the 24 stay RED.
2. **Re-audit vs wire.** The honest framing is "deferred by an audit whose deferral was never
   revisited", and D3 shows the premise no longer holds (sources hold **25 / 9 / 3** records
   against the spec's **125+ / 45+ / 12**; the per-(seed × fold) data exists elsewhere and is
   not read). The ruling: **re-audit the 29 SCAFFOLD-only classifications, or wire the 24 as-is?**
3. *(Smaller, optional)* **Should the census-freshness guard ever become an ENFORCING gate?**
   Primary scope ships with zero blast radius today (0 stale of 1); generalized scope blocks
   every launch (160 stale of 1822 across 38 artefacts). Generalized first, or primary only?
