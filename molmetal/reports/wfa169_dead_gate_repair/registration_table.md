# Wf-A169 — registration table for the 24 dead Wf-432-A1 gates

**Tag: MEASURED (2026-11-01, Wf-A169).** Every path below was checked on disk;
every line reference is a `file:line` in the tree as it stood when the patch was
generated (snapshot sha256 `ae878bcd52491903e42a37faef8cabfd0b1b59cff5ea05d4520389993d2c9ce9`,
734,797 bytes, `molmetal/scripts/pre_flight_check.py`).

**0 UNDETERMINED.** All 24 gates' arguments were recoverable from the spec +
the gate signatures + the already-defined (and previously **unused**) `_p2_*`
locals + the witness test's `arg_sets`. Nothing was invented.

---

## 1. The finding, re-derived

| Claim | How re-derived | Result |
|---|---|---|
| The 24 gate **functions** exist | `grep -n "^def gate_p3_protocol\|^def gate_krasnov\|^def gate_rusanov\|^def gate_drugood" molmetal/scripts/pre_flight_check.py` | 24/24 present (L10768..L12580 region) |
| They were **never registered** | `git log --all -S'<name>' --oneline -- molmetal/scripts/pre_flight_check.py` for each of the 24 names | **exactly one commit each — `bc29aa90` (2026-10-02)** — the commit that added the *definitions*. The string count never exceeded the definition count, so no registration call was ever written (not "written, then removed") |
| The Wf-437 "29 SCAFFOLD-only gate tuples REMOVED" comment does **not** refer to these 24 | `git show 140e1f82 -- molmetal/scripts/pre_flight_check.py \| grep '^-' \| grep p3_protocol_` → **no match**; the only `p3_protocol_*` gates registered today are the 4 `p3_protocol_alignment_yaml_*` (Wf-350-C1 family, registered at `pre_flight_check.py:14800-14840`) | comment is about a different family; the 24 were **never** in any pipeline |
| The `_p2_*` locals were written for exactly these gates and never used | `grep -oE "_p2_[a-z0-9_]+" molmetal/scripts/pre_flight_check.py \| sort \| uniq -c` | 26 `_p2_*` locals at count 1 (definition only). The 8 that ARE used (count ≥ 2) belong to the registered L3/L4 gates. **Every `_p2_*` needed by the 24 was at count 1 = unused** |
| Live baseline counts | live import of the working-tree module | `all`=129, `launch`=131, `frontier`=25, `chemprop_v2_v3`=18, `cfm`=8; `EXPECTED_GATE_COUNTS` = same |

**Correction to the task brief.** The brief cited the Wf-432-A1 close-out's byte
gap (348,994 committed vs 359,637 claimed) as proof of a missing registration.
That arithmetic no longer holds on this tree: `molmetal/scripts/pre_flight_check.py`
is now 734,797 bytes (other waves added ~386 KB). The **git-history proof is
what survives**, and it is stronger than the byte gap — see above.

---

## 2. Registration conventions copied from the siblings

| Element | Value | Justification |
|---|---|---|
| Tuple shape | `(<name>, lambda: gate_fn(<name>, <priority>, <pipeline>, *args))` | sibling L3/L4 block, `pre_flight_check.py:14392-14478`; sibling yaml family, `:14800-14840` |
| Priority string | `"P3"` | sibling L3/L4 block (`:14395`, `:14406`, `:14419`, `:14436`, …); witness test calls every gate with `"P3"` (`test_wf433a1_gate_v2_wire.py:325`) |
| Pipeline argument passed to the gate fn | literal `"all"` | sibling L3/L4 block passes literal `"all"` (`:14397`, `:14438`, …); witness test passes `"all"` (`:325`) |
| Target pipeline | `all` only (so `launch` inherits it — `build_gates` normalises `"launch"` → `"all"` at `:13912-13915`) | spec §1.3 routing, `PRE_TRAINING_GATE_v2.md:92-99`; witness test `:340-353` asserts each of the 24 is in `all` and **not** in `frontier`/`cfm` |
| Path construction | module-level constants `REPO_ROOT` / `MOLMETAL_ROOT` / `SCRIPT_DIR` / `SCRIPTS_DIR` / `DATA_DIR` / `DOT_DATA_DIR` (`pre_flight_check.py:259-266`, `:308-310`) + the `_p2_*` locals | sibling blocks |

**L5 routing note.** Spec §1.3 (`PRE_TRAINING_GATE_v2.md:99`) says the L5 hooks
have "no main-Layer-0 routing — these are post-launch hooks", while the
`build_gates` header comment says all 32 route to `'all'` only
(`pre_flight_check.py:14369`). The witness test resolves the ambiguity:
it asserts each of the 9 L5 names **is** in `build_gates("all")`
(`test_wf433a1_gate_v2_wire.py:340-344`). "No main-Layer-0 routing" therefore
means *not* frontier / chemprop_v2_v3 / cfm. Routed to `all`.

---

## 3. L1 — Tier-1 must-pass (8 gates), pipeline `all`, priority `P3`

Spec: `TODO/03_plans/PRE_TRAINING_GATE_v2.md` §2.

| # | Gate name | Arg (position) | Value passed | Justification |
|---|---|---|---|---|
| 1 | `p3_protocol_alignment_dataset_match` | `metalcytotox_csv` | `cytox_csv_v4` | `pre_flight_check.py:14002` = `DOT_DATA_DIR/"metalcytotox_5metal_multitask_v4.csv"`. **`registry/datasets.yaml:66-86` marks v4 `lifecycle_status: ACTIVE`, `locked_for: [m1_ru_auc, m19_lomo, m21_nci60_9cell]`; v1 (`:19-31`) is `SUPERSEDED`**. Spec L1-1 Source line `PRE_TRAINING_GATE_v2.md:114`; witness test `:147,195` |
| 1 | " | `platinai_csv` | `_p2_platinai_csv` | `:14373` = `DATA_DIR/"platinai_4cell.csv"` (exists). Spec Source `:114`; test `:148,196` |
| 1 | " | `crossdocked_csv` | `_p2_crossdocked_csv` | `:14374` = `DATA_DIR/"crossdocked100_manifest.csv"` (exists). Spec Source `:114`; test `:149,197` |
| 1 | " | `nci60_csv` | `_p2_nci60_csv` | `:14375` = `DATA_DIR/"nci60_precious_9cell.csv"` (exists). Spec Source `:114`; test `:150,198` |
| 1 | " | `balanced_rh_json` | `_p2_balanced_rh_json` | `:14376-14381` = `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` (exists). Spec Source `:114`; test `:151-156,199` |
| 1 | " | `drugood_loader` | `_p2_drugood_loader` | `:14382` = `DATA_DIR/"drugood_loader.py"` (exists). Spec Source `:114`; test `:157,200` |
| 2 | `p3_protocol_alignment_split_match` | `leakage_utils_py` | `_p2_leakage_utils` | `:14383` = `DATA_DIR/"leakage_utils.py"` (exists). Spec Source `:122`; test `:158,203` |
| 2 | " | `drugood_all_results_json` | `_p2_drugood_all_results` | `:14384-14386` = `molmetal/reports/wf265_collect/all_results.json` (exists). Spec Source `:122`; test `:159,203` |
| 3 | `p3_protocol_alignment_metric_match` | `eval_all_evaluators_py` | `_p2_eval_all_evaluators` | `:14387` = `SCRIPT_DIR/"eval_all_evaluators.py"` (exists). Spec Source `:130`; test `:160,204` |
| 4 | `p3_protocol_alignment_seed_cohort_per_seed_per_split` | `train_metal_cytotox_py` | `_p2_train_metal_cytotox` | `:14388` (exists). Spec Source `:138`; test `:161,206` |
| 4 | " | `train_platinai_py` | `_p2_train_platinai` | `:14389` = `SCRIPT_DIR/"train_platinai_4cell.py"` (exists). Spec Source `:138`; test `:162,207` |
| 4 | " | `train_nci60_10cell_py` | `_p2_train_nci60_10cell` | `:14390` (exists). Spec Source `:138`; test `:163,208` |
| 4 | " | `per_seed_template_dir` | `_p2_per_seed_template_dir` | `:14391` = `molmetal/reports/wf131_p3_deferred_runs` (exists). Spec Source `:138` ("template"); test `:164,209` |
| 5 | `p3_protocol_alignment_honest_limitation_minor_section` | `honest_negative_catalog_md` | `_p2_honest_negative_catalog` | `:14392` = `docs/HONEST_NEGATIVE_CATALOG.md` (exists; contains `## §6` at 158 places). Spec Source `:146`; test `:165,211` |
| 6 | `p3_protocol_alignment_paper_section_6_template` | `section_6_template_md` | `_p2_section_6_template` | `:14393-14395` = `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (exists). Spec Source `:154`; test `:166,212` |
| 6 | " | `paper_section_6_v4_md` | `_p2_paper_section_6_v4` | `:14396-14398` = `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (exists). Spec Source `:154`; test `:167,214` |
| 7 | `p3_protocol_alignment_user_ok_gate` | `scripts_dir` | `SCRIPTS_DIR` | `:262-264` — the constant's own comment reads "Wf-432-A1: project root scripts dir (USER_OK wrapper audit)". Test `:192,216` |
| 7 | " | `wrapper_names` | `_p2_wrapper_names` | `:14432-14440` — the 7-wrapper tuple, verbatim the spec's list at `PRE_TRAINING_GATE_v2.md:161` and the test's `:183-191`. All 7 files exist and each contains `USER_OK` (measured: 13/17/11/23/19/23/21 occurrences) |
| 8 | `p3_protocol_alignment_baseline_paper_refs_bib` | `paper_refs_bib` | `_p2_paper_refs_bib` | `:14406` = `paper/refs.bib` (exists, 100 entries). Spec Spec `:169-172` / Source `:173`; test `:171,217`. `required_keys` left at the function default `("krasnov2026metalcytotoxdb","rusanov2025platinai","ji2022drugood")` — see §5 honest-negative |

## 4. L2 — Tier-2 recommended (7 gates), pipeline `all`, priority `P3`

Spec: `PRE_TRAINING_GATE_v2.md` §3. These are WARN-only by function contract.

| # | Gate name | Arg (position) | Value passed | Justification |
|---|---|---|---|---|
| 1 | `p3_protocol_recommended_seed_count_3` | `train_metal_cytotox_py` | `_p2_train_metal_cytotox` | `:14388`. Spec Source `:201`; test `:161,219` |
| 1 | " | `train_platinai_py` | `_p2_train_platinai` | `:14389`. Spec Source `:201`; test `:162,220` |
| 1 | " | `train_nci60_10cell_py` | `_p2_train_nci60_10cell` | `:14390`. Spec Source `:201`; test `:163,221` |
| 2 | `p3_protocol_recommended_per_cell_breakdown` | `train_platinai_py` | `_p2_train_platinai` | `:14389`. Spec Source `:209`; test `:162,224` |
| 2 | " | `train_nci60_10cell_py` | `_p2_train_nci60_10cell` | `:14390`. Spec Source `:209`; test `:163,225` |
| 2 | " | `per_cohort_template_json` | `_p2_m2_r2_per_cohort` | `:14419-14421` = `metrics/by_metric/m2_r2_per_cohort.json` (exists). **Spec Source `:209` names `molmetal/reports/wf388a1_m2_r2_real/result.json`, which does NOT exist on disk**; the witness test `:226` uses the metrics JSON, which does. Documented in §5 |
| 3 | `p3_protocol_recommended_5_fold_cv_compatibility_50x` | `train_metal_cytotox_py` | `_p2_train_metal_cytotox` | `:14388`. Spec Source `:217`; test `:161,229` |
| 3 | " | `drugood_loader_py` | `_p2_drugood_loader` | `:14382`. Spec Source `:217` ("`molmetal/scripts/drugood_loader.py`" — the spec puts it under `scripts/`, the tree has it under `molmetal/data/`, and `_p2_drugood_loader` is the tree's real location). Test `:157,230` |
| 4 | `p3_protocol_recommended_krasnov_paper_table_4_replication` | `reports_dir` | `MOLMETAL_ROOT / "reports"` | the gate appends `krasnov_2026_paper_table4_replication` itself (`pre_flight_check.py:11655`), and spec `:225` Source = `molmetal/reports/krasnov_2026_paper_table4_replication/` → parent is `molmetal/reports`. Test `:232-234` |
| 5 | `p3_protocol_recommended_drugood_3_splits_full_alignment` | `drugood_all_results_json` | `_p2_drugood_all_results` | `:14384-14386`. Spec Source `:233`; test `:159,235` |
| 6 | `p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup` | `reports_dir` | `MOLMETAL_ROOT / "reports"` | the gate appends `rusanov_2025_paper_table_s4_s5_baseline` (`pre_flight_check.py:11739`); spec Source `:241` = `molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline/`. Test `:236-239` |
| 6 | " | `train_platinai_py` | `_p2_train_platinai` | `:14389`. Spec Source `:241`; test `:162,238` |
| 7 | `p3_protocol_recommended_section_6_honest_minor_section_auto_emit` | `auto_emit_section_6_minor_py` | `_p2_auto_emit_section_6_minor` | `:14407` = `SCRIPT_DIR/"auto_emit_section_6_minor.py"`. Spec Source `:249`; test `:172,240` |

## 5. L5 — per-baseline protocol validation hooks (9 gates), pipeline `all`, priority `P3`

Spec: `PRE_TRAINING_GATE_v2.md` §4. `hook_dir` is
`molmetal/reports/protocol_validation/` for all 9 — the hook emitter writes
`{baseline}_hook.json` into it (`pre_flight_check.py:12322-12337`), which is
exactly the spec's §4 summary destination
(`PRE_TRAINING_GATE_v2.md:345`).

| # | Gate name | Arg (position) | Value passed | Justification |
|---|---|---|---|---|
| 1 | `krasnov_m1_per_seed_per_split_5fold_5metal` | `hook_dir` | `_p2_protocol_validation_dir` | `:14430` |
| 1 | " | `metal_lomo_auc_json` | `_p2_metal_lomo_auc` | `:14423-14425` = `metrics/by_metric/metal_lomo_auc_wasserstein.json` (exists). Spec Source `:271`; test `:179,267` |
| 2 | `krasnov_m21_per_cell_line_per_seed` | `hook_dir` | `_p2_protocol_validation_dir` | `:14430` |
| 2 | " | `nci60_precious_9cell_dir` | `_p2_nci60_9cell_dir` | `:14426` = `molmetal/reports/wf_nci60_precious_9cell` (exists). Spec Source `:279`; test `:180,268` |
| 3 | `krasnov_protocol_alignment_diffs_documented` | `hook_dir` | `_p2_protocol_validation_dir` | `:14430` |
| 3 | " | `honest_negative_catalog_md` | `_p2_honest_negative_catalog` | `:14392`. Spec Source `:287`; test `:165,270` |
| 3 | " | `protocol_alignment_matrix_md` | `_p2_protocol_alignment_matrix` | `:14403-14405`, **path corrected by this patch** from `TODO/03_plans/…` to `TODO/03_baselines/…` (see §6). Spec Source `:287` + spec cross-ref `:10` and `:461` |
| 4 | `rusanov_m2_per_cell_line_per_cohort` | `hook_dir` | `_p2_protocol_validation_dir` | `:14430` |
| 4 | " | `m2_r2_per_cohort_json` | `_p2_m2_r2_per_cohort` | `:14419-14421` (exists). Spec Source `:297`; test `:178,274` |
| 5 | `rusanov_m2_4cell_baseline_comparison` | `hook_dir` | `_p2_protocol_validation_dir` | `:14430` |
| 5 | " | `rusanov_baseline_dir` | `_p2_rusanov_baseline_dir` | `:14427-14429` = `molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline` (does **not** exist — §7). Spec Source `:305`; test `:181,275` |
| 5 | " | `m2_r2_per_cohort_json` | `_p2_m2_r2_per_cohort` | `:14419-14421`. Spec Source `:305`; test `:178,275` |
| 6 | `rusanov_protocol_alignment_diffs_documented` | `hook_dir` | `_p2_protocol_validation_dir` | `:14430` |
| 6 | " | `honest_negative_catalog_md` | `_p2_honest_negative_catalog` | `:14392`. Spec Source `:313`; test `:165,277` |
| 6 | " | `protocol_alignment_matrix_md` | `_p2_protocol_alignment_matrix` | `:14403-14405` (path corrected — §6). Spec Source `:313`; test `:170,278` |
| 7 | `drugood_m17_assay_split_auc` | `hook_dir` | `_p2_protocol_validation_dir` | `:14430` |
| 7 | " | `drugood_all_results_json` | `_p2_drugood_all_results` | `:14384-14386`. Spec Source `:323`; test `:159,281` |
| 8 | `drugood_m17_scaffold_split_auc` | `hook_dir` | `_p2_protocol_validation_dir` | `:14430` |
| 8 | " | `drugood_all_results_json` | `_p2_drugood_all_results` | `:14384-14386`. Spec Source `:331`; test `:159,282` |
| 9 | `drugood_m17_size_split_auc` | `hook_dir` | `_p2_protocol_validation_dir` | `:14430` |
| 9 | " | `drugood_all_results_json` | `_p2_drugood_all_results` | `:14384-14386`. Spec Source `:339`; test `:159,283` |

---

## 6. The one non-additive line in the patch (and why it is safe)

The patch changes **exactly one** existing line:

```
-            REPO_ROOT / "TODO" / "03_plans" / "PROTOCOL_ALIGNMENT_MATRIX.md"
+            REPO_ROOT / "TODO" / "03_baselines" / "PROTOCOL_ALIGNMENT_MATRIX.md"
```

* `TODO/03_plans/PROTOCOL_ALIGNMENT_MATRIX.md` — **does not exist** (measured).
* `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` — exists, 6,043 bytes
  (measured). It is the canonical location per the spec's own cross-references
  (`PRE_TRAINING_GATE_v2.md:10` and `:461`).
* The local `_p2_protocol_alignment_matrix` was **unused** before this patch
  (usage count 1 = the definition), so nothing else in the file is affected.

Without this correction the two L5 protocol-diff gates would still pass (their
`source_present` test reads `honest_negative_catalog_md`, which exists), but
they would carry a dead path. The witness test also uses the wrong
`03_plans` path (`test_wf433a1_gate_v2_wire.py:170`) — the tree, not the test,
is the authority here.

---

## 7. Honest negatives surfaced BY the registration (not wiring defects)

Measured by invoking all 24 runners on the patched module:
**19 PASS, 4 WARN, 1 FAIL.**

| Gate | Verdict | Evidence (measured) |
|---|---|---|
| `p3_protocol_alignment_baseline_paper_refs_bib` | **FAIL** | `refs.bib missing 3 keys: ['krasnov2026metalcytotoxdb', 'rusanov2025platinai', 'ji2022drugood']` — the 3 keys are genuinely absent from `paper/refs.bib` (100 `@`-entries, none matching). This is a **BLOCKING** gate per spec `:174` |
| `p3_protocol_recommended_krasnov_paper_table_4_replication` | WARN | `missing: molmetal/reports/krasnov_2026_paper_table4_replication` (dir does not exist) |
| `p3_protocol_recommended_drugood_3_splits_full_alignment` | WARN | `only 0/3 DrugOOD splits in all_results.json` — `molmetal/reports/wf265_collect/all_results.json` has keys `G1_todo_180_g4_complete` … `g4_overall_aggregate_full_15pair`, no `assay`/`scaffold`/`size` split keys |
| `p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup` | WARN | `missing: molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline` (dir does not exist) |
| `p3_protocol_recommended_section_6_honest_minor_section_auto_emit` | WARN | `missing: molmetal/scripts/auto_emit_section_6_minor.py` (file does not exist) |

These are the pre-existing honest-negatives the 24 dead gates were designed to
surface. They are why Wf-437 classified the family as "SCAFFOLD-only" — but
Wf-437's *removal* never touched these 24 (see §1). Registering them makes the
`all` pipeline FAIL on the missing BibTeX keys.

---

## 8. Known side effects the main thread must know

1. **`EXPECTED_GATE_COUNTS` is NOT bumped by this patch** (it is another
   branch's file). The explicit `expected_total` drift-check gate inside
   `build_gates("all")` will therefore FAIL with
   `actual=153 vs expected=129 (+DRIFT (silent gate bloat))` until the count
   pin lands. The two edits must land together.
2. `EXPECTED_GATE_COUNTS["launch"]` must go 131 → 155 for the same reason.
3. `frontier` (25), `chemprop_v2_v3` (18) and `cfm` (8) are **unchanged** —
   measured after applying the patch.
4. The witness test `molmetal/tests/test_wf433a1_gate_v2_wire.py` pins
   `EXPECTED_GATE_COUNTS["all"] == 127` and `len(set(names)) == 127`; both pins
   are already stale (today: 129) and will need the count-pin branch's update.
   Its 24 "missing from build_gates('all')" failures are what this patch fixes.
5. The patch was generated against a **snapshot** of a file that the main thread
   is concurrently editing. Re-verify with `git apply --check` before applying;
   `apply_probe.json` records the snapshot sha256. If it no longer applies, run
   `CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/reports/wfa169_dead_gate_repair/regen_patch.py`
   — it re-bases the patch onto the current file and re-runs the whole probe
   (scratch `git apply --check` → `git apply` → byte-compare → import →
   per-pipeline counts → the 24 verdicts) with no change to this table's
   conclusions.

### Line-reference caveat

Every `file:line` above is relative to the **patch snapshot**
(`molmetal/scripts/pre_flight_check.py`, sha256
`ae878bcd52491903e42a37faef8cabfd0b1b59cff5ea05d4520389993d2c9ce9`, 734,797
bytes) — the exact bytes the diff's `a/` side is. The file grew by ~386 KB
during this wave, so absolute line numbers will drift if the main thread keeps
editing it; the *function* names are stable.
