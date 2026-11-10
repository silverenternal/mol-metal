# Wf-A169 branch 2 — EXECUTION VERDICT for the 24 unregistered gates

**Date:** 2026-11-01 · **Branch:** Wf-A169 / branch 2 (execution verdict)
**Denominator:** **24** (8 L1 + 7 L2 + 9 L5) — every one of the 24 gates was
executed against the CURRENT tree.
**Machine-readable:** `molmetal/reports/wfa169_dead_gate_repair/verdict_table.json`
**Runner:** `molmetal/reports/wfa169_dead_gate_repair/wfa169b2_exec_verdict.py`
(`CUDA_VISIBLE_DEVICES="" .venv/bin/python <runner>` — read-only, CPU-only, 0 GPU)

---

## 0. HISTORY CORRECTION — the "never registered" premise is FALSE

The main thread's proof was `git log --all -S'p3_protocol_alignment_dataset_match'
-- molmetal/scripts/pre_flight_check.py` → one commit. That search is
**path-restricted**, and it misses the registration.

**MEASURED:** the 24 gates **WERE registered**, with `EXPECTED_GATE_COUNTS["all"] == 127`.
The registration block survives in a **deleted backup**:

| Fact | Value | Source |
|---|---|---|
| Backup blob | `TODO/archive/wf437_simplify/pre_flight_check.py.bak.wf437` | `git show aee5f8ab^:TODO/archive/wf437_simplify/pre_flight_check.py.bak.wf437` |
| Backup size | **359,637 bytes** | `wc -c` on the blob |
| Backup `"all"` constant | **127** | `grep -oE '"all": *[0-9]+'` → `"all": 127` |
| Live file at `bc29aa90` | 348,994 bytes, `"all": 98` | `git show bc29aa90:molmetal/scripts/pre_flight_check.py` |
| Live file **today** | 732,059 bytes, `"all": 129` | `stat` + live import |
| Registration tuples for the 24 in the backup | **present** (2–3 occurrences each) | `grep -cE "^ *\"<gate>\","` over the backup blob |
| Registration tuples for the 24 in the live file at every one of the 41 commits that touched it | **0** | 41-commit sweep, `grep -cE '^                "<gate>",\$'` |
| Backup deleted by | `aee5f8ab` "chore(repo): purge 83 tracked junk paths" (2026-10-03 17:51) | `git log --all --diff-filter=D -- '*pre_flight_check*'` |

So the true sequence is: **Wf-432-A1 shipped the 32 gates registered (127)**;
a later wave (**Wf-437 A2 audit / Wf-439**, commit `bc29aa90`, 2026-10-02 14:37)
**physically removed 29 SCAFFOLD-only registration tuples (127 → 98)** and kept
the helper functions as dead code. The worktree comment at
`molmetal/scripts/pre_flight_check.py:14441-14450` ("Wf-437 simplify: 29
SCAFFOLD-only gate tuples REMOVED per brief / Wf-437 A2 audit") is **TRUE**;
the main thread's "never run / never registered" reading is **FALSE**.
The 10,643-byte gap between 359,637 and 348,994 is exactly the removed tuples.

**Consequence for this branch:** the *exact original argument tuples* are
recoverable from the backup, so this report runs the gates with **the arguments
they were originally registered with** — not with a guessed arg set.

---

## 1. Status tally (denominator = 24)

| Arg set | PASS | WARN | FAIL | RAISED |
|---|---|---|---|---|
| **`original_registration`** (PRIMARY — the recovered original tuples) | **19** | **4** | **1** | **0** |
| `witness_test` (`molmetal/tests/test_wf433a1_gate_v2_wire.py` arg sets) | 19 | 4 | 1 | 0 |
| `spec_literal` (spec §2/§3 "Source" lines read literally) | 16 | 6 | **2** | 0 |

**The primary and the witness-test arg sets agree exactly.** The spec-literal
variant differs only because the spec names `train_platinai.py`, a file that has
never existed (the real trainer is `train_platinai_4cell.py`) — that spelling
adds 1 FAIL and 3 WARN.

Live gate counts on this tree (from the same import):
`len(build_gates("all")) = 129`, `len(build_gates("launch")) = 131`,
`EXPECTED_GATE_COUNTS = {"all": 129, "launch": 131, "frontier": 25,
"chemprop_v2_v3": 18, "cfm": 8}`.

---

## 2. Per-gate verdict table (24 rows)

`args missing on disk` counts the *original-registration* arguments that do not
exist. Evidence is truncated to 300 chars; the **full** evidence string is in
`verdict_table.json`.

| # | Layer | Gate | Status | args missing | Evidence (truncated) | alt arg sets |
|---|---|---|---|---|---|---|
| 1 | L1 | `p3_protocol_alignment_dataset_match` | PASS | 0 | all 7 Tier-1 LEAD metric datasets on disk: metalcytotox=metalcytotox_5metal_multitask.csv, platinai=platinai_4cell.csv, … | wit=PASS, spec=PASS |
| 2 | L1 | `p3_protocol_alignment_split_match` | PASS | 0 | scaffold_split + drugood 3-splits on disk: leakage_utils=leakage_utils.py, drugood_all_results=all_results.json | wit=PASS, spec=PASS |
| 3 | L1 | `p3_protocol_alignment_metric_match` | PASS | 0 | eval_all_evaluators.py on disk (7 Tier-1 LEAD metric definitions bit-for-bit identical to baseline) | wit=PASS, spec=PASS |
| 4 | L1 | `p3_protocol_alignment_seed_cohort_per_seed_per_split` | PASS | 0 | per-seed + per-split stdout format emitters registered: metal_cytotox / platinai / nci60 | wit=PASS, **spec=FAIL** |
| 5 | L1 | `p3_protocol_alignment_honest_limitation_minor_section` | PASS | 0 | HONEST_NEGATIVE_CATALOG.md on disk with §6 catalog section (1902463 bytes) | wit=PASS, spec=PASS |
| 6 | L1 | `p3_protocol_alignment_paper_section_6_template` | PASS | 0 | §6 templates on disk: section_6_template, paper_section_6_v4 | wit=PASS, spec=PASS |
| 7 | L1 | `p3_protocol_alignment_user_ok_gate` | PASS | 0 | **all 7 retrain wrappers have USER_OK gate**: launch_wf124_p3_g5.sh, launch_wf124_p3_g6.sh, launch_wf299_cfm_main.sh, launch_wf388_a1_metal_coord_retrain.sh, launch_wf394a1_lambda_cfm_patch.sh, launch_wf125_p1_controlled_pair.sh, launch_wf390a1_m2_r2_path_a.sh | wit=PASS, spec=PASS |
| 8 | L1 | `p3_protocol_alignment_baseline_paper_refs_bib` | **FAIL** | 0 | **refs.bib missing 3 keys: `['krasnov2026metalcytotoxdb', 'rusanov2025platinai', 'ji2022drugood']`** | wit=FAIL, spec=FAIL |
| 9 | L2 | `p3_protocol_recommended_seed_count_3` | PASS | 0 | 3 trainer wrappers emit per-seed JSON | wit=PASS, spec=WARN |
| 10 | L2 | `p3_protocol_recommended_per_cell_breakdown` | PASS | 0 | per-cell breakdown CLI flag registered | wit=PASS, spec=WARN |
| 11 | L2 | `p3_protocol_recommended_5_fold_cv_compatibility_50x` | PASS | 0 | 5-fold CV CLI flag registered | wit=PASS, spec=PASS |
| 12 | L2 | `p3_protocol_recommended_krasnov_paper_table_4_replication` | WARN | 0 | missing: `molmetal/reports/krasnov_2026_paper_table4_replication` | wit=WARN, spec=WARN |
| 13 | L2 | `p3_protocol_recommended_drugood_3_splits_full_alignment` | WARN | 0 | only 0/3 DrugOOD splits in all_results.json: `[]` | wit=WARN, spec=WARN |
| 14 | L2 | `p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup` | WARN | 0 | missing: `molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline` | wit=WARN, spec=WARN |
| 15 | L2 | `p3_protocol_recommended_section_6_honest_minor_section_auto_emit` | WARN | 1 | missing: `molmetal/scripts/auto_emit_section_6_minor.py` | wit=WARN, spec=WARN |
| 16 | L5 | `krasnov_m1_per_seed_per_split_5fold_5metal` | PASS | 1 | krasnov hook emitted with source_present=True | wit=PASS, spec=PASS |
| 17 | L5 | `krasnov_m21_per_cell_line_per_seed` | PASS | 1 | krasnov hook emitted with source_present=True | wit=PASS, spec=PASS |
| 18 | L5 | `krasnov_protocol_alignment_diffs_documented` | PASS | 2 | krasnov hook emitted with source_present=True | wit=PASS, spec=PASS |
| 19 | L5 | `rusanov_m2_per_cell_line_per_cohort` | PASS | 1 | rusanov hook emitted with source_present=True | wit=PASS, spec=PASS |
| 20 | L5 | `rusanov_m2_4cell_baseline_comparison` | PASS | 2 | rusanov hook emitted with source_present=True | wit=PASS, spec=PASS |
| 21 | L5 | `rusanov_protocol_alignment_diffs_documented` | PASS | 2 | rusanov hook emitted with source_present=True | wit=PASS, spec=PASS |
| 22 | L5 | `drugood_m17_assay_split_auc` | PASS | 1 | drugood hook emitted with source_present=True | wit=PASS, spec=PASS |
| 23 | L5 | `drugood_m17_scaffold_split_auc` | PASS | 1 | drugood hook emitted with source_present=True | wit=PASS, spec=PASS |
| 24 | L5 | `drugood_m17_size_split_auc` | PASS | 1 | drugood hook emitted with source_present=True | wit=PASS, spec=PASS |

`args missing on disk` for the L5 rows is the canonical
`molmetal/reports/protocol_validation` dir (absent) plus, for rows 18/20/21,
the second missing arg. Those still PASS because `_hook_passthrough_payload`
returns PASS whenever **one** source is present (`pre_flight_check.py:12404-12423`, `source_present = (...)` at 12404 and `if not source_present:` at 12423).

---

## 3. THE DECISIVE ANSWER

> **If the 24 were registered as they were originally registered, how many would
> be FAIL, and which ones?**
>
> **Exactly ONE: `p3_protocol_alignment_baseline_paper_refs_bib`** (L1, Tier-1
> BLOCKING). 19 PASS, 4 WARN, 0 exceptions.

> **Does wiring them turn the pre-flight RED?**
>
> **YES — by exactly one gate.** `pre_flight_check.py:17595-17596` is
> `if n_fail > 0: return 1`. `WARN` has **no** branch in that tail, so the 4
> WARNs do **not** change the exit code. One FAIL ⇒ `all` pipeline exits 1.
>
> The single FAIL is `p3_protocol_alignment_baseline_paper_refs_bib` — the gate
> that checks `paper/refs.bib` for the 3 baseline-paper BibTeX keys.

**Routing caveat (spec §1.3, line 98):** the 9 L5 hooks are specified with
"**no main-Layer-0 routing** — these are post-launch hooks". Registering only
the 15 L1+L2 gates into `all` still yields the same 1 FAIL (L1-8 is an L1
gate) ⇒ still RED. Gate-count arithmetic: `all` 129 → **144** (L1+L2 only) or
**153** (all 24) — see branch 3 for the `EXPECTED_GATE_COUNTS` pin.

---

## 4. Cause classification

### 4.1 The one FAIL

| Gate | Cause | Detail |
|---|---|---|
| `p3_protocol_alignment_baseline_paper_refs_bib` | **(iii) blocked on USER AUTHORIZATION** — with an embedded **(i) real DATA gap** | The gate reads `paper/refs.bib` (100 `@`-entries, 46,684 bytes) and finds **0 of 3** required keys. The repair is a write under `paper/**`, which AGENTS.md §5.3 forbids without explicit user OK. Sub-gap: `krasnov2026metalcytotoxdb` and `rusanov2025platinai` **already exist verbatim** in `.tools/sci-paper/refs/references.bib` (full `@article` entries with journal/volume/pages) and are copyable; `ji2022drugood` exists **nowhere** in the repo as a BibTeX entry (only as a name in `TODO/03_plans/PRE_TRAINING_GATE_v2.md`, `TODO/completed/432a1_pre_training_gate_v2.md`, and the gate's own default tuple) and must be authored. |

**Minimal repair (NOT applied — `paper/**` is forbidden for this wave):**
`paper/refs.bib` — append the two copyable `@article` blocks from
`.tools/sci-paper/refs/references.bib` (keys `krasnov2026metalcytotoxdb`,
`rusanov2025platinai`) and author a third `@inproceedings{ji2022drugood, …}`
for Ji et al., *DrugOOD* (ICLR 2022). Requires explicit user OK per AGENTS.md
§5.3. `TODO/02_metrics/METRICS_TARGETS.md:640-641` still shows the two
`\citep{…}` checklist items unticked, so this is a known-open item, not a surprise.

### 4.2 The four WARNs (non-blocking; `main()` ignores WARN)

| Gate | Cause | Detail | Minimal repair |
|---|---|---|---|
| `p3_protocol_recommended_krasnov_paper_table_4_replication` | **(i) real DATA gap** | `molmetal/reports/krasnov_2026_paper_table4_replication/` does not exist (never created; `git ls-files` has no such path). The gate only checks the **directory**'s existence (`pre_flight_check.py:11655`), not the template file the spec asks for. | create the dir + a real Table-4 replication template (not an empty stub — git does not track empty dirs, so a file is required). **Not applied** (WARN is non-blocking; a stub created only to satisfy a gate would be a fake artifact). |
| `p3_protocol_recommended_drugood_3_splits_full_alignment` | **(ii) real CODE defect — wrong source file** | The gate reads `molmetal/reports/wf265_collect/all_results.json` and finds 0/3 split keys. **The 3 splits ARE on disk**: `metrics/by_metric/drugood_ood_auc.json` → `per_axis_auc = {"assay": 0.7304477112563803, "scaffold": 0.6902449048522429, "size": 0.6763469128123165}`. The collector file's 15 keys are G1..G6 / wf263a / wf264 progress keys, not split AUCs — the source the gate names no longer carries the data it wants. | repoint the gate's `drugood_all_results_json` argument at `metrics/by_metric/drugood_ood_auc.json` (and read `per_axis_auc`), or restore split keys into the collector file. **Not applied** — the fix lives in `molmetal/scripts/pre_flight_check.py`, which is READ-ONLY for this wave. |
| `p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup` | **(i) real DATA gap** | `molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline/` does not exist. The spec's Table S4-S5 per-cell AUCs exist only as a prose reference (`TODO/03_baselines/COMPARISON_BASELINES.md:74`), never as a lookup table. | materialise the Table S4-S5 baseline table under that path. **Not applied** (WARN is non-blocking). |
| `p3_protocol_recommended_section_6_honest_minor_section_auto_emit` | **(i) real DATA gap** | `molmetal/scripts/auto_emit_section_6_minor.py` does not exist and never has (`git log --all -- <path>` is empty). | write the auto-emit script. **Not applied** (WARN is non-blocking). |

---

## 5. Are L1 blocking and L2 warn? — verbatim from the spec

Yes, by design. `TODO/03_plans/PRE_TRAINING_GATE_v2.md`:

- **L1 = BLOCKING** — line 108: *"These 8 gates are BLOCKING — must PASS before any
  retrain launch. They enforce the protocol + metric + dataset alignment contract
  per `D-Wf-431`."*; line 189: *"All 8 BLOCKING; fail any one → retrain BLOCKED per
  `AGENTS.md §5.1`."*; the per-gate "Wave" field on each of the 8 reads
  *"pre-retrain (BLOCKING)"* (L1-8 additionally: *"pre-retrain (BLOCKING; 0
  paper/* touches per AGENTS.md §5.3)"*).
- **L2 = WARN-only** — line 195: *"These 7 gates are WARN-only (strong
  recommendation but non-blocking). They enforce best-practice conventions."*;
  line 257: *"All 7 WARN-only; retrain proceeds even with WARN but logs the
  recommendation."*

The implementation matches that intent: every L2 gate returns `"WARN"` (never
`"FAIL"`) on a missing input — see `pre_flight_check.py:11563, 11598, 11631,
11661, 11689, 11749, 11777` — and every L1 gate returns `"FAIL"`.

**But the pre-flight runner does not implement the L1/L2 split**: `main()` treats
any FAIL as exit 1 regardless of layer, and treats WARN as exit-code-neutral.
So the L1/L2 tiering lives only in the gate functions' return status, not in the
harness.

---

## 6. Two gates the brief asked about specifically

### 6.1 `p3_protocol_alignment_baseline_paper_refs_bib` — CONFIRMED FAIL
`paper/refs.bib` (46,684 bytes, 100 `@`-entries) contains **0** of
`krasnov2026metalcytotoxdb`, `rusanov2025platinai`, `ji2022drugood` — and a
case-insensitive grep for `krasnov|rusanov|drugood|ji2022|metalcytotox|platinai|cytotox`
returns **no hits at all**: none of the three baseline papers is cited in
`paper/refs.bib` under *any* key. Verdict: **FAIL** (evidence: `refs.bib
missing 3 keys: [...]`). Cause **(iii)**, see §4.1.

### 6.2 `p3_protocol_alignment_user_ok_gate` — PASS, and it needs no human permission
This is a **code audit**: `pre_flight_check.py:11005-11049` (`def gate_p3_protocol_alignment_user_ok_gate`) reads each wrapper
file and checks for the literal string `"USER_OK"`. The intended
`wrapper_names` list was recovered from the original registration block
(backup blob line 6393-6401) and is the same 7 wrappers the spec §2 L1-7 names:

`launch_wf124_p3_g5.sh`, `launch_wf124_p3_g6.sh`, `launch_wf299_cfm_main.sh`,
`launch_wf388_a1_metal_coord_retrain.sh`, `launch_wf394a1_lambda_cfm_patch.sh`,
`launch_wf125_p1_controlled_pair.sh`, `launch_wf390a1_m2_r2_path_a.sh`

**All 7 exist in `scripts/` and all 7 contain the string `USER_OK`**
(independent grep counts: 13 / 17 / 11 / 23 / 19 / 23 / 21 occurrences).
Verdict: **PASS**. **No wrapper is missing the string.** Note the gate checks only
that the string is *present* — it does **not** verify the spec's second clause
("correctly positioned BEFORE the `--dry-run` exit block"), so a PASS here is
weaker than the spec's contract.

---

## 7. Weak-gate findings (the PASSes are weaker than the spec)

Several implemented gates verify far less than the spec's "Verdict gate" line.
Measured on this tree — the spec'd condition is **false** while the gate returns
**PASS**:

| Gate | Spec'd check | What the implementation actually does | Spec'd condition on this tree |
|---|---|---|---|
| L1-4 `…_seed_cohort_per_seed_per_split` | detect `eval_per_seed_per_split` in `parse_args()` + per-seed/per-split stdout format | only checks that 4 **files** exist | **FALSE** — `eval_per_seed_per_split` appears in **none** of `train_metal_cytotox.py`, `train_platinai_4cell.py`, `train_nci60_10cell.py` |
| L2-2 `…_per_cell_breakdown` | `--per-cell-breakdown` flag registered + emits per-cell JSON | only checks that 3 **files** exist | **FALSE** — no `per_cell_breakdown` flag in `train_platinai_4cell.py` or `train_nci60_10cell.py` |
| L2-3 `…_5_fold_cv_compatibility_50x` | `--eval-5fold-cv` flag registered | only checks that 2 **files** exist | **FALSE** — no 5-fold-CV flag in `train_metal_cytotox.py` or `drugood_loader.py` |
| L1-1 `…_dataset_match` | dataset **name** matches baseline per PROTOCOL_ALIGNMENT_MATRIX §1 | only checks 6 **paths** exist | not verified — no name comparison in code |
| L1-2 `…_split_match` | `scaffold_split` exists **AND** assay/scaffold/size splits present for m17 | only checks 2 **files** exist | **FALSE** — `wf265_collect/all_results.json` has 0/3 split keys (same defect as L2-5) |
| L1-3 `…_metric_match` | evaluator output matches baseline metric definitions, bit-for-bit | only checks that `eval_all_evaluators.py` exists | not verified — the evidence string asserts "bit-for-bit identical to baseline" from **file existence alone** |
| L1-5 `…_honest_limitation_minor_section` | all **7** Tier-1 LEAD metrics have a §6 catalog entry + minor-section template | only checks the catalog file exists and contains `"## §6"` | not verified — no per-metric check |
| L5 × 9 | e.g. "125+ records", "45+ records", "12 records", "delta > +0.03", "delta_imold > -0.02" | only checks that **one** source file/dir exists (`source_present`) | **FALSE** — no record-count or delta threshold is computed anywhere in `_hook_passthrough_payload` (`pre_flight_check.py:12404-12423`, `source_present = (...)` at 12404 and `if not source_present:` at 12423) |

L2-1 (`seed_count_3`) is the one whose spec'd condition actually holds:
`--seeds` is `nargs="+"`, `default=[0, 1, 2]` in all three trainers (verified in
`train_metal_cytotox.py:142-147`, `train_platinai_4cell.py:891-895`).

**Also measured — and already repaired by another branch mid-wave:** the L5
hooks' "auto-emit" was a **silent no-op** when the output dir was absent —
`_emit_protocol_validation_hook_json` returned immediately if `hook_dir.exists()`
was false, and **nothing in the repo created**
`molmetal/reports/protocol_validation/`. So the hooks reported *"hook emitted
with source_present=True"* in their **evidence string** while writing **nothing**.
This was MEASURED against the file as of 2026-11-01 11:53 (before the concurrent
edit). At **11:54:50 the main thread landed a Wf-A169 repair** to
`_emit_protocol_validation_hook_json` (`pre_flight_check.py:12322-12383`, docstring
"WF-A169 REPAIR (2026-11-01) — TWO defects … (D1) THE HOOK WAS NEVER EMITTED …
(2) SIBLING hook_idS WERE SILENTLY DROPPED"): the emitter now creates the
directory and merges payloads. **Re-running this branch's runner after that
repair produced the same tally (19 PASS / 4 WARN / 1 FAIL / 0 RAISED)** and now
emits all 9 hook_ids into the canonical location:

```
molmetal/reports/protocol_validation/krasnov_hook.json  -> 3 hook_ids
molmetal/reports/protocol_validation/rusanov_hook.json  -> 3 hook_ids
molmetal/reports/protocol_validation/drugood_hook.json  -> 3 hook_ids
```

Those three files are gitignored (`.gitignore:158 molmetal/reports/**/*.json`).
A second copy of the payload, captured with `hook_dir` redirected into this
report's own directory, is at
`molmetal/reports/wfa169_dead_gate_repair/hook_payload_capture/`.

> **Concurrency note:** `molmetal/scripts/pre_flight_check.py` is being edited
> concurrently by another branch of this wave (734,797 bytes, mtime
> 2026-11-01 11:54:50, during this branch's run). Every verdict in this report
> was **re-run and re-confirmed** against the post-edit file; the tally is
> unchanged. All `file:line` citations above are as of that mtime — quote the
> **function names** if the file moves again.

---

## 8. Reproduction

```
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
  molmetal/reports/wfa169_dead_gate_repair/wfa169b2_exec_verdict.py
# -> molmetal/reports/wfa169_dead_gate_repair/verdict_table.json

# recover the original registration block:
git show aee5f8ab^:TODO/archive/wf437_simplify/pre_flight_check.py.bak.wf437 \
  | grep -n '"p3_protocol_alignment_dataset_match",'
```

All numbers in this report are MEASURED on this tree on 2026-11-01 and trace to
`verdict_table.json` (produced above) or to the `file:line` / `git show` commands
quoted inline. Nothing here is PROJECTED.
