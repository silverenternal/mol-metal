# Wf-A175 — DEAD-GATE WIRING DECISION TABLE

**Branch:** WIRING · **Date:** 2026-11-01 · **CPU-only** (no GPU, no network, no `paper/**` write)
**Deliverable:** a decision table, NOT a patch. Nothing was wired. Nothing outside
`molmetal/reports/wfa175_dead_gate_wiring/` was written.

Machine-readable twin: `decision_table.json` (one record per gate).
Generator: `_build_table.py` / `_build_report.py` (both re-runnable, both read-only elsewhere).

---

## §0 Headline (all numbers MEASURED 2026-11-01)

- **0 of 24** dead gates are registered in **any** pipeline (`all`=129, `launch`=131, `frontier`=25, `chemprop_v2_v3`=18, `cfm`=8; introspection via `pre_flight_check.build_gates(<pipeline>)`).
- **Invoking** all 24 against the real tree gives **19 PASS / 4 WARN / 1 FAIL / 0 RAISED** (denominator 24). This independently reproduces the Wf-A169 number (`docs/HONEST_NEGATIVE_CATALOG.md` §6.266.5).
- **Spec-threshold status:** 11 MET / 7 SHORT / 5 MISSING_SOURCE / 1 UNSPEC.
- **Wiring verdict:** 8 WIREABLE-BY-READONLY-REPAIR / 7 NEEDS-MEASUREMENT / 2 NEEDS-GPU / 1 NEEDS-PAPER-AUTH / 6 DEFER-BY-DESIGN.
- **The PASS column is not a quality column.** By implementation, the 24 break down as **20 pure path-existence checks / 3 single-substring checks / 1 JSON-parsing check** (classification verified against each gate's own source with an assertion, not hand-typed). Of the 19 live PASSes, 9 are the L5 stubs and 6 more check nothing but a path's existence while the spec's own sub-condition is unmet. See §3.

---

## §1 The decision table

`spec_threshold` is quoted VERBATIM from the `**Verdict gate**:` line of
`TODO/03_plans/PRE_TRAINING_GATE_v2.md` at the recorded `spec_line`.
`measured_num / measured_den` are measured against that threshold — NOT against the
gate's own implementation (the two disagree for most of the 24; see §3).
`implementation_kind` says what the gate body actually does: `path` = only `.exists()` on
its arguments, `substring` = reads a file and looks for a literal string, `parse` = parses
JSON and inspects keys. **Not one of the 24 evaluates the spec's threshold.**

### L1 — Tier-1 must-pass (BLOCKING per PRE_TRAINING_GATE_v2.md §2)

| gate | spec line | spec threshold (verbatim, abridged) | measured | status | impl | verdict |
|---|---|---|---|---|---|---|
| `p3_protocol_alignment_dataset_match` | 115 | `PASS` if all 7 Tier-1 LEAD metrics have a matching dataset on disk; `FAIL` if any dataset is missing … | 6 / 6 | **MET** | path | WIREABLE-BY-READONLY-REPAIR |
| `p3_protocol_alignment_split_match` | 123 | `PASS` if all splits documented in `PROTOCOL_ALIGNMENT_MATRIX.md` §1 (DATASET + METRIC ALIGN is OK; pr… | 0 / 2 | **SHORT** | path | WIREABLE-BY-READONLY-REPAIR |
| `p3_protocol_alignment_metric_match` | 131 | `PASS` if all 7 metric definitions are bit-for-bit identical to baseline (Cohen 1988 effect size bands… | 0 / 7 | **UNSPEC** | path | DEFER-BY-DESIGN |
| `p3_protocol_alignment_seed_cohort_per_seed_per_split` | 139 | `PASS` if per-seed + per-split stdout format is detected in `--help` output + `eval_per_seed_per_split… | 0 / 3 | **SHORT** | path | NEEDS-MEASUREMENT |
| `p3_protocol_alignment_honest_limitation_minor_section` | 147 | `PASS` if all 7 Tier-1 LEAD metrics have §6 catalog entry + minor-section template present | 7 / 7 | **MET** | substring | WIREABLE-BY-READONLY-REPAIR |
| `p3_protocol_alignment_paper_section_6_template` | 155 | `PASS` if both templates present + per-metric drop-down compiles (CPU-only verification) | 2 / 2 | **MET** | path | WIREABLE-BY-READONLY-REPAIR |
| `p3_protocol_alignment_user_ok_gate` | 163 | `PASS` if all 7 wrappers have USER_OK gate present + correctly positioned BEFORE `--dry-run` exit block | 7 / 7 | **MET** | substring | WIREABLE-BY-READONLY-REPAIR |
| `p3_protocol_alignment_baseline_paper_refs_bib` | 174 | `PASS` if all 3 BibTeX entries present + DOIs parseable | 0 / 3 | **MISSING_SOURCE** | substring | NEEDS-PAPER-AUTH |

### L2 — Tier-2 recommended (WARN-only per §3)

| gate | spec line | spec threshold (verbatim, abridged) | measured | status | impl | verdict |
|---|---|---|---|---|---|---|
| `p3_protocol_recommended_seed_count_3` | 202 | `PASS` if all wrappers default to 3+ seeds; otherwise `WARN` if 1-seed (acceptable for first pass per … | 3 / 3 | **MET** | path | WIREABLE-BY-READONLY-REPAIR |
| `p3_protocol_recommended_per_cell_breakdown` | 210 | `PASS` if per-cell breakdown CLI flag present + emits valid JSON; otherwise `WARN` | 0 / 2 | **SHORT** | path | NEEDS-MEASUREMENT |
| `p3_protocol_recommended_5_fold_cv_compatibility_50x` | 218 | `PASS` if 5-fold CV CLI flag present; otherwise `WARN` (current 1-fold scaffold-split acceptable for §… | 0 / 2 | **SHORT** | path | NEEDS-MEASUREMENT |
| `p3_protocol_recommended_krasnov_paper_table_4_replication` | 226 | `PASS` if template file present; otherwise `WARN` (paper-Table-4 replication is RECOMMENDED per D-Wf-4… | 0 / 1 | **MISSING_SOURCE** | path | NEEDS-MEASUREMENT |
| `p3_protocol_recommended_drugood_3_splits_full_alignment` | 234 | `PASS` if all 3 splits present + iMoLD-gate PASS for each split | 3 / 3 | **MET** | parse | WIREABLE-BY-READONLY-REPAIR |
| `p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup` | 242 | `PASS` if baseline table present + lookup emits comparison vs baseline per cell-line | 0 / 1 | **MISSING_SOURCE** | path | NEEDS-MEASUREMENT |
| `p3_protocol_recommended_section_6_honest_minor_section_auto_emit` | 250 | `PASS` if auto-emit script present + emits valid §6 minor-section per Tier-1 LEAD metric; otherwise `W… | 0 / 1 | **MISSING_SOURCE** | path | WIREABLE-BY-READONLY-REPAIR |

### L5 — per-baseline protocol-validation hooks (post-launch, §4)

| gate | spec line | spec threshold (verbatim, abridged) | measured | status | impl | verdict |
|---|---|---|---|---|---|---|
| `krasnov_m1_per_seed_per_split_5fold_5metal` | 272 | `PASS` if 5+ metals × 5 seeds × 5 folds (125+ records) present; else `WARN` | 25 / 125 | **SHORT** | path | NEEDS-GPU |
| `krasnov_m21_per_cell_line_per_seed` | 280 | `PASS` if 9 cell-lines × 5+ seeds (45+ records) present; else `WARN` | 9 / 45 | **SHORT** | path | NEEDS-GPU |
| `krasnov_protocol_alignment_diffs_documented` | 288 | `PASS` if §6 paragraph exists for each Tier-1 LEAD metric × Krasnov | 2 / 2 | **MET** | path | DEFER-BY-DESIGN |
| `rusanov_m2_per_cell_line_per_cohort` | 298 | `PASS` if 4 cell-lines × 3 cohorts (12 records) present; else `WARN` | 3 / 12 | **SHORT** | path | NEEDS-MEASUREMENT |
| `rusanov_m2_4cell_baseline_comparison` | 306 | `PASS` if at least 1 cell-line `delta > +0.03` (Tier 1 match) | — | **MISSING_SOURCE** | path | NEEDS-MEASUREMENT |
| `rusanov_protocol_alignment_diffs_documented` | 314 | `PASS` if §6 paragraph exists for each Tier-1 LEAD metric × Rusanov | 1 / 1 | **MET** | path | DEFER-BY-DESIGN |
| `drugood_m17_assay_split_auc` | 324 | `PASS` if `delta_imold > -0.02` (Tier 1 match) | +0.009348 vs bar -0.02 | **MET** | path | DEFER-BY-DESIGN |
| `drugood_m17_scaffold_split_auc` | 332 | `PASS` if `delta_imold > -0.02` | +0.001845 vs bar -0.02 | **MET** | path | DEFER-BY-DESIGN |
| `drugood_m17_size_split_auc` | 340 | `PASS` if `delta_imold > -0.02` | -0.002853 vs bar -0.02 | **MET** | path | DEFER-BY-DESIGN |

---

## §2 Per-gate blockers (why)

**L1 — Tier-1 must-pass (BLOCKING per PRE_TRAINING_GATE_v2.md §2)**

- **`p3_protocol_alignment_dataset_match`** — Gate enumerates 6 dataset paths but its evidence string claims 'all 7 Tier-1 LEAD metric datasets'; PROTOCOL_ALIGNMENT_MATRIX §1 lists m3/m7/m15 as 'NO FORMAL PAPER COMPARISON — Tier-2 supplementary', so no dataset is expected for them. The 6 paths exist. Fixing the evidence string is a read-only edit.
  - evidence: `molmetal/scripts/pre_flight_check.py:10768 (evidence string); TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md:15-27`
- **`p3_protocol_alignment_split_match`** — Both named sub-conditions fail. (a) 'leakage_utils.py::scaffold_split' does not exist — the module has no scaffold symbol at all; the scaffold splitter lives in molmetal/data/splits.py as the class ScaffoldSplitter. (b) all_results.json carries 0 of 3 DrugOOD split keys (the 3 splits live in metrics/by_metric/drugood_ood_auc.json). Repair = re-point both paths; the underlying data DO exist, so the repaired gate stays GREEN. Wiring it UNREPAIRED adds a vacuous PASS that contradicts L2-5's WARN on the same file.
  - evidence: `molmetal/data/leakage_utils.py:24-187 (no scaffold symbol); molmetal/data/splits.py:610 (ScaffoldSplitter); molmetal/reports/wf265_collect/all_results.json (0/3 splits); metrics/by_metric/drugood_ood_auc.json:6-9 (splits)`
- **`p3_protocol_alignment_metric_match`** — 'Bit-for-bit identical to baseline' is not a machine-checkable criterion: no comparison oracle exists in the repo, and the named source reports (molmetal/reports/wf331_r1{1,2,3}_*.json) are not on disk (only wf331r3/r4/r5 scripts + tests). The gate checks one file's existence. There is nothing to repair without first DEFINING the comparison.
  - evidence: `molmetal/scripts/pre_flight_check.py:10853; TODO/03_plans/PRE_TRAINING_GATE_v2.md:131`
- **`p3_protocol_alignment_seed_cohort_per_seed_per_split`** — The spec names an 'eval_per_seed_per_split' flag registered in parse_args(); it occurs 0 times in all three trainers. The trainers DO emit per-seed JSON, but through the --seeds default [0,1,2] path, not through a flag of that name. Either the trainer needs the flag (a §5.7 50-step CPU smoke is then mandatory) or the spec text must be corrected to name the mechanism that exists.
  - evidence: `molmetal/scripts/train_metal_cytotox.py:142-146; molmetal/scripts/train_platinai_4cell.py:891-895; molmetal/scripts/train_nci60_10cell.py:860-864; TODO/03_plans/PRE_TRAINING_GATE_v2.md:139`
- **`p3_protocol_alignment_honest_limitation_minor_section`** — All 7 Tier-1 LEAD metrics have at least one §6 catalog entry (loose reading, which is what the spec states). Under the repo's own stricter rule (metric token AND a protocol/split token) only 5/7 qualify — m15 and m17 have no §6 entry naming a protocol difference. The gate itself checks only the '## §6' header.
  - evidence: `docs/HONEST_NEGATIVE_CATALOG.md (138 '### §6.' entries); molmetal/scripts/pre_flight_check.py:10923`
- **`p3_protocol_alignment_paper_section_6_template`** — Both template files exist. The second half of the verdict ('per-metric drop-down compiles') was NOT verified — no LaTeX run was performed (CPU-only branch, and the templates live under docs/, not paper/, so the compile itself is permitted but was not run). The template's per-metric subsections cover m1/m2/m3/m6_kappa/m7 and bundle m14/m17/m19; m21 has no dedicated subsection.
  - evidence: `docs/SECTION_6_LIMITATIONS_TEMPLATE.md:112-180; docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md:159,360-408; molmetal/scripts/pre_flight_check.py:10972`
- **`p3_protocol_alignment_user_ok_gate`** — All 7 wrappers contain the USER_OK gate AND, measured against the real DRY_RUN exit block, place it BEFORE that block. The gate itself only greps for the string 'USER_OK' anywhere in the file — it does not check the ordering the spec requires, so it under-checks a condition that happens to hold today.
  - evidence: `molmetal/scripts/pre_flight_check.py:11005; scripts/launch_wf124_p3_g5.sh:65,198; scripts/launch_wf390a1_m2_r2_path_a.sh:117,234`
- **`p3_protocol_alignment_baseline_paper_refs_bib`** — paper/refs.bib is missing all 3 baseline keys. Closing this is a write under paper/** — AGENTS.md §5.3 / TODO-84 forbids it without the user's explicit OK. Wiring the gate before the keys exist turns the 'all' pipeline RED today, which would block every launch over a citation-key defect that is not a launch blocker.
  - evidence: `paper/refs.bib (0/3 keys); molmetal/scripts/pre_flight_check.py:11052; AGENTS.md §5.3`

**L2 — Tier-2 recommended (WARN-only per §3)**

- **`p3_protocol_recommended_seed_count_3`** — All 3 trainers default --seeds to [0,1,2] and all 3 emit an 'n_seeds' field in the result JSON. The gate only checks that the three files exist; strengthening it to parse the default is a read-only edit and would still be GREEN.
  - evidence: `molmetal/scripts/train_metal_cytotox.py:142-146,336; molmetal/scripts/train_platinai_4cell.py:891-895,1125; molmetal/scripts/train_nci60_10cell.py:860-864,1018`
- **`p3_protocol_recommended_per_cell_breakdown`** — The spec's 'per-cell breakdown CLI flag' is registered 0 times in train_platinai_4cell.py and train_nci60_10cell.py. Closing it is a TRAINER wire-in (a new argparse flag + a per-cell JSON emitter), not a pre-flight edit — and per AGENTS.md §5.7 it would need a 50-step CPU smoke before being called shipped. The gate as written PASSes on file existence, so it is a vacuous PASS today.
  - evidence: `molmetal/scripts/train_platinai_4cell.py; molmetal/scripts/train_nci60_10cell.py; TODO/03_plans/PRE_TRAINING_GATE_v2.md:210`
- **`p3_protocol_recommended_5_fold_cv_compatibility_50x`** — No 5-fold CV flag exists: 0 matches for 5fold/5_fold/five_fold in train_metal_cytotox.py and 0 in drugood_loader.py. Closing it is a trainer/loader feature, i.e. the same class as the gate above. The gate as written PASSes on file existence.
  - evidence: `molmetal/scripts/train_metal_cytotox.py; molmetal/data/drugood_loader.py; TODO/03_plans/PRE_TRAINING_GATE_v2.md:218`
- **`p3_protocol_recommended_krasnov_paper_table_4_replication`** — molmetal/reports/krasnov_2026_paper_table4_replication/ does not exist. The spec's own verdict is 'PASS if template file present', so creating an EMPTY directory would turn this into a vacuous PASS over nothing. The underlying content is Krasnov 2026 Table 4 (5-metal x 5-fold x per-fold AUC), which needs a primary-record fetch and, for per-fold AUCs, a GPU retrain.
  - evidence: `molmetal/scripts/pre_flight_check.py:11647; TODO/03_plans/PRE_TRAINING_GATE_v2.md:266-272`
- **`p3_protocol_recommended_drugood_3_splits_full_alignment`** — All 3 DrugOOD splits and a per-split iMoLD-gate verdict exist — but NOT in the file the gate reads. all_results.json carries 0 of 3 split keys; metrics/by_metric/drugood_ood_auc.json carries all 3 (assay/scaffold/size, gate_pass true for each). The gate's detector ('split in data or any(split in k for k in data.keys())') would still miss them after a path fix, because the splits sit in a nested key — so the detector needs fixing too. Both edits are read-only and the repaired gate is GREEN.
  - evidence: `metrics/by_metric/drugood_ood_auc.json:6-9,52-64; molmetal/reports/wf265_collect/all_results.json (0/3 split keys); molmetal/scripts/pre_flight_check.py:11676`
- **`p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup`** — molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline/ does not exist. Its content is Rusanov 2025 Table S4-S5 — an EXTERNAL primary record that must be fetched (a search summary will not do). The verdict's second half ('lookup emits comparison vs baseline per cell-line') additionally needs train_platinai.py --per-cell-breakdown, which does not exist (see p3_protocol_recommended_per_cell_breakdown).
  - evidence: `molmetal/scripts/pre_flight_check.py:11729; TODO/03_plans/PRE_TRAINING_GATE_v2.md:236-242`
- **`p3_protocol_recommended_section_6_honest_minor_section_auto_emit`** — molmetal/scripts/auto_emit_section_6_minor.py does not exist. The verdict needs a script that auto-emits a §6 minor-section per Tier-1 LEAD metric — a NEW file under molmetal/scripts/ (allowed; it must not write under paper/, per AGENTS.md §5.3). Until it exists the gate WARNs; it cannot be wired as a PASS.
  - evidence: `molmetal/scripts/pre_flight_check.py:11764; TODO/03_plans/PRE_TRAINING_GATE_v2.md:250`

**L5 — per-baseline protocol-validation hooks (post-launch, §4)**

- **`krasnov_m1_per_seed_per_split_5fold_5metal`** — The source metrics/by_metric/metal_lomo_auc_wasserstein.json holds 25 per_run_auc scalars; the per-fold values are not stored per run (per_fold_auc_mean is a 5-key summary over held-out metals). The spec's own template source (molmetal/reports/wf131_p3_deferred_runs/per_seed_metrics_seed*.json) holds 0 matching files. Producing 125 (metal x seed x fold) records means a 5-metal x 5-seed x 5-fold retrain = a GPU run.
  - evidence: `metrics/by_metric/metal_lomo_auc_wasserstein.json (per_run_auc=25); molmetal/reports/wf131_p3_deferred_runs/ (0 per_seed_metrics_seed*.json); TODO/03_plans/PRE_TRAINING_GATE_v2.md:266-272`
- **`krasnov_m21_per_cell_line_per_seed`** — metrics/by_metric/nci60_precious_9cell_auc.json holds 9 per_cell_line records, and EVERY one of them carries per_seed_available=false — the 9x5=45 per-(cell x seed) records the spec asks for do not exist anywhere on disk. The metric header says n_seeds=5, so the run did happen; the per-seed values were not persisted. Recovering them needs a retrain.
  - evidence: `metrics/by_metric/nci60_precious_9cell_auc.json (per_cell_line=9, per_seed_available=false); TODO/03_plans/PRE_TRAINING_GATE_v2.md:275-280`
- **`krasnov_protocol_alignment_diffs_documented`** — Doc-presence threshold is met (m1 and m21 both have a qualifying §6 entry). BUT this gate WRITES on invocation: calling it created molmetal/reports/protocol_validation/{krasnov,rusanov,drugood}_hook.json (measured — the directory did not exist before the probe and was removed afterwards to restore the tree). A post-launch hook is not a pre-flight gate, and the spec's own §1.3 routing rule excludes the 9 L5 hooks from the 'all' pipeline.
  - evidence: `molmetal/scripts/pre_flight_check.py:12350-12454 (_hook_passthrough_payload + emitter); TODO/03_plans/PRE_TRAINING_GATE_v2.md:§1.3 (routing); docs/HONEST_NEGATIVE_CATALOG.md §6.266.3`
- **`rusanov_m2_per_cell_line_per_cohort`** — The source metrics/by_metric/m2_r2_per_cohort.json holds 3 cohorts and no per-cell-line axis. The plan doc names a different file (molmetal/reports/wf388a1_m2_r2_real/result.json) which does not exist — the real directory is wf388a1_m2_per_cohort_real, whose result.json also has no per-cell axis, and whose producing script has 0 per-cell references. The ridge model is CPU-cheap (elapsed_sec 0.08), so the gap is COMPUTABLE — but platinai_4cell's labels are SIMULATED and .data/molmetal/platinai_4cell.csv is classified DO-NOT-TRAIN ('Its ~0.50 AUC is a null test, not a measurement'). Manufacturing the 12 records from that source would create a number that is not a measurement.
  - evidence: `metrics/by_metric/m2_r2_per_cohort.json (per_cohort=3, label_provenance.labels=SIMULATED); molmetal/reports/wf388a1_m2_per_cohort_real/result.json (exists); molmetal/scripts/wf388a1_m2_per_cohort_real.py; TODO/03_plans/PRE_TRAINING_GATE_v2.md:291-298`
- **`rusanov_m2_4cell_baseline_comparison`** — Both named sources are absent: wf388a1_m2_r2_real/result.json (the spec's primary) and molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline/ (the spec's alternate). The threshold is 'at least 1 cell-line with delta > +0.03', which needs our per-cell-line m2 values (blocked above) AND the Rusanov 2025 Table S4-S5 baseline (an external primary record). No number can be measured today.
  - evidence: `molmetal/scripts/pre_flight_check.py:12542; TODO/03_plans/PRE_TRAINING_GATE_v2.md:301-306`
- **`rusanov_protocol_alignment_diffs_documented`** — Doc-presence threshold is met (m2 has a qualifying §6 entry). Same two caveats as the Krasnov sibling: the gate WRITES a hook JSON on invocation, and the spec's §1.3 routing rule excludes the 9 L5 hooks from 'all'.
  - evidence: `molmetal/scripts/pre_flight_check.py:12564; TODO/03_plans/PRE_TRAINING_GATE_v2.md:§1.3`
- **`drugood_m17_assay_split_auc`** — delta_imold = +0.009348 vs the -0.02 bar. The value lives in metrics/by_metric/drugood_ood_auc.json (2026-10-07, official DrugOOD IC50 axis partitions, 3 seeds, assay OOD AUC 0.730448) — NOT in the all_results.json the gate reads. The plan doc's own 'current state' prose ('metacytotox_fallback assay 0.7479') is STALE against that metric JSON, and PROTOCOL_ALIGNMENT_MATRIX §1 still calls m17 'NOT ALIGN — fallback cohort'. The JSON's own honest_framing: 'protocol-comparable but NOT backbone-comparable'. Also DEFER: L5 routing + write side effect.
  - evidence: `metrics/by_metric/drugood_ood_auc.json (per_axis_auc_delta_vs_imold, honest_framing); TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md:24; TODO/03_plans/PRE_TRAINING_GATE_v2.md:319-324`
- **`drugood_m17_scaffold_split_auc`** — delta_imold = +0.001845 vs the -0.02 bar (scaffold OOD AUC 0.690245). Same source / staleness / backbone caveats as the assay sibling; the margin here is only 0.0218 above the bar.
  - evidence: `metrics/by_metric/drugood_ood_auc.json (per_axis_auc_delta_vs_imold.scaffold); TODO/03_plans/PRE_TRAINING_GATE_v2.md:327-332`
- **`drugood_m17_size_split_auc`** — delta_imold = -0.002853 vs the -0.02 bar (size OOD AUC 0.676347) — the closest of the three to the bar, 0.0171 of margin. Same source / staleness / backbone caveats as the assay sibling.
  - evidence: `metrics/by_metric/drugood_ood_auc.json (per_axis_auc_delta_vs_imold.size); TODO/03_plans/PRE_TRAINING_GATE_v2.md:335-340`

---

## §3 The gap between what the gates check and what the spec asks

This is the finding that decides the wiring: **the gate bodies are almost entirely
file-existence checks, while the spec's verdict gates are content checks.**
Six independent measurements, all taken from disk:

| spec asks for | gate actually checks | measured |
|---|---|---|
| L1-2 `leakage_utils.py::scaffold_split` exists | `leakage_utils.py` exists | symbol present = **False**; the scaffold splitter is the class `ScaffoldSplitter` in `molmetal/data/splits.py` (present = True) |
| L1-2 `all_results.json` has assay/scaffold/size | file exists | splits found = **0 / 3** (the 3 live in `metrics/by_metric/drugood_ood_auc.json`: **3 / 3**) |
| L1-4 `eval_per_seed_per_split` registered in `parse_args()` | 3 trainer files exist | flag occurrences = **0 / 3** |
| L2-2 `--per-cell-breakdown` flag registered | 2 trainer files exist | flag occurrences = **0 / 2** |
| L2-3 `--eval-5fold-cv` flag registered | 2 files exist | 5-fold flag occurrences = **0 / 2** |
| L1-7 USER_OK gate **positioned before** the `--dry-run` exit block | the string `USER_OK` appears anywhere | positioning holds in **7 / 7** wrappers |

Two more, in the other direction — the gates are **stricter** than their spec:

- **L2-1** (`seed_count_3`): the spec wants `--seeds 0 1 2` defaults; measured **3 / 3** trainers default to `[0,1,2]` and **3 / 3** emit `n_seeds`. The spec condition is MET; the gate only checks file existence.
- **L2-5** (`drugood_3_splits`): the spec wants all 3 splits + an iMoLD-gate PASS per split. `metrics/by_metric/drugood_ood_auc.json` carries all 3 splits with `gate_pass` = {'assay': True, 'scaffold': True, 'size': True} — but the gate reads `all_results.json`, which has **0 / 3**. A path fix alone is not enough: the gate's detector (`split in data or any(split in k for k in data.keys())`) would still miss splits nested under a list-valued key.

**L1-5** is the one place where the spec's wording and the repo's stricter rule diverge: the spec asks for '§6 catalog entry exists for that metric' — measured **7 / 7** Tier-1 LEAD metrics have one (m1, m2, m3, m7, m15, m17, m21). Under the repo's own rule in `molmetal/scripts/wfa171_l5_threshold.py` (metric token **AND** a protocol/split token) only **5 / 7** qualify — m15 and m17 have no §6 entry naming a protocol difference. The gate itself checks only that the string `## §6` appears.

**L1-3** (`metric_match`) is the one verdict that cannot be repaired at all: 'all 7 metric
definitions are bit-for-bit identical to baseline' has no comparison oracle in the repo, and
the source reports the spec names (`molmetal/reports/wf331_r1{1,2,3}_*.json`) are not on disk.
Its status is therefore **UNSPEC**, not SHORT: there is no measurable definition to miss.

---

## §4 The 9 L5 hooks: three separate reasons not to wire them

**(a) The spec's own routing rule excludes them.** `PRE_TRAINING_GATE_v2.md` §1.3:
*'L5 per-baseline protocol validation hooks hook to … (no main-Layer-0 routing — these are
post-launch hooks)'*. §0/§1.2 nevertheless count them in the `95 + 32 = 127` arithmetic.
The spec contradicts itself (§6.266.4). Wiring all 24 into `all` would take it to
**153** (and `launch` to **155**) — 
violating §1.3. This needs a user-level ruling, not a patch.

**(b) They are not read-only.** MEASURED: calling any of the 9 writes
`molmetal/reports/protocol_validation/{krasnov,rusanov,drugood}_hook.json`
(the directory did not exist before the probe — `docs/HONEST_NEGATIVE_CATALOG.md` §6.266.3
records it as absent — and was removed again afterwards to restore the tree). The emitter
now creates `hook_dir` and merges sibling `hook_id`s, so the side effect is *by design*
for a post-launch hook, and *wrong* for a pre-flight gate.

**(c) Three are SHORT and one is MISSING_SOURCE on their own thresholds** — measured with the
repo's own read-only module `molmetal/scripts/wfa171_l5_threshold.py` (thresholds parsed from
the plan doc at runtime):

| L5 hook | spec threshold | measured | status |
|---|---|---|---|
| `krasnov_m1_per_seed_per_split_5fold_5metal` | 272 | 25 / 125 | SHORT |
| `krasnov_m21_per_cell_line_per_seed` | 280 | 9 / 45 | SHORT |
| `rusanov_m2_per_cell_line_per_cohort` | 298 | 3 / 12 | SHORT |
| `rusanov_m2_4cell_baseline_comparison` | 306 | None / None | MISSING_SOURCE |

The remaining 6 L5 hooks (2 doc-presence + 3 DrugOOD value + the L1/L2 sibling) are
numerically MET — but 'MET' here means *the threshold is met*, not *the gate checks it*.
The gate checks file existence only; the threshold comparison lives in `wfa171`, which is
**not wired into the gate**. Wiring the 9 as-is would add 9 vacuous PASSes to a 129-gate
pre-flight.

---

## §5 The MINIMAL, READ-ONLY-ONLY wiring set

### §5.1 Wire TODAY as-is (4 gates) — GREEN, read-only, no GPU, no paper write

- `p3_protocol_alignment_honest_limitation_minor_section` — spec threshold **MET**; live verdict **PASS**.
- `p3_protocol_alignment_paper_section_6_template` — spec threshold **MET**; live verdict **PASS**.
- `p3_protocol_alignment_user_ok_gate` — spec threshold **MET**; live verdict **PASS**.
- `p3_protocol_recommended_seed_count_3` — spec threshold **MET**; live verdict **PASS**.

All four are GREEN on the real tree today. Wiring them changes `all` 129 → 133 and `launch` 131 → 135; `frontier` / `chemprop_v2_v3` / `cfm` unchanged.

### §5.2 Wire TODAY after a READ-ONLY repair (4 gates) — still GREEN, still no GPU, still no paper write

- `p3_protocol_alignment_dataset_match` — spec threshold **MET**. Repair: correct the evidence string — the gate enumerates 6 dataset paths and then asserts 'all 7 Tier-1 LEAD metric datasets'. One string edit in `pre_flight_check.py`; the 6 paths all exist.
- `p3_protocol_alignment_split_match` — spec threshold **SHORT**. Repair: re-point both arguments: `leakage_utils.py` → `molmetal/data/splits.py` (where `ScaffoldSplitter` actually lives) and `all_results.json` → `metrics/by_metric/drugood_ood_auc.json` (which carries all 3 splits). Both targets exist, so the repaired gate is GREEN. **Until then it must NOT be wired** (see §5.3 Group B).
- `p3_protocol_recommended_drugood_3_splits_full_alignment` — spec threshold **MET**. Repair: re-point the source AND fix the detector: the splits sit under a nested key, so the current `split in data or any(split in k for k in data.keys())` scan would still miss them after a path fix. Two edits, both in `pre_flight_check.py`; the repaired gate is GREEN (3/3 splits, `gate_pass` true for each).
- `p3_protocol_recommended_section_6_honest_minor_section_auto_emit` — spec threshold **MISSING_SOURCE**. Repair: author the missing emitter `molmetal/scripts/auto_emit_section_6_minor.py`. That is a NEW file under `molmetal/scripts/` — NOT a `paper/` write, so AGENTS.md §5.3 is not engaged as long as the emitted section is not injected into `paper/`.

These four are the only ones where a read-only edit makes the gate *mean* something it
currently does not: the first three are re-pointed at the file that actually holds the
data, the fourth needs its emitter script authored (a `molmetal/scripts/` file — NOT a
`paper/` write, so §5.3 is not engaged as long as the emitted section is not injected into
`paper/`).

### §5.3 MUST NOT be wired today — and why

**Group A — wiring it would turn the pre-flight RED over a non-launch defect (1 gate).**

- `p3_protocol_alignment_baseline_paper_refs_bib` — today **FAIL** (0 / 3 keys present in `paper/refs.bib`). It is a Tier-1 BLOCKING gate; wiring it as-is turns `all` RED today and blocks every launch. Closing it requires a write
  under `paper/**` — AGENTS.md §5.3 / TODO-84 forbid that without the user's explicit OK.
  **Verdict: NEEDS-PAPER-AUTH.**

**Group B — the spec's own sub-condition is unmet, so the gate would be a vacuous PASS (3 gates).**

(`p3_protocol_alignment_split_match` also belongs here until its repair lands — it is
listed in §5.2 because the repair is read-only, but wiring it unrepaired would add a PASS
that directly contradicts L2-5's WARN on the same `all_results.json`.)

- `p3_protocol_alignment_seed_cohort_per_seed_per_split` — `eval_per_seed_per_split` occurs
  0 / 3 times. Closing it is a **trainer** wire-in, not a pre-flight edit;
  AGENTS.md §5.7 would then require a 50-step CPU smoke before it can be called shipped.
- `p3_protocol_recommended_per_cell_breakdown` and
  `p3_protocol_recommended_5_fold_cv_compatibility_50x` — same class: the named CLI flags
  do not exist (0 / 2 and 0 / 2).

**Group C — the threshold needs a measurement that does not exist yet (4 gates).**

- `p3_protocol_recommended_krasnov_paper_table_4_replication` — the directory is absent, but
  the spec's verdict is 'PASS if template file present', so an EMPTY directory would satisfy
  it. Creating one turns a missing measurement into a PASS over nothing.
- `p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup` — needs the Rusanov 2025
  Table S4-S5 values (an EXTERNAL primary record — a search summary will not do) plus a
  `--per-cell-breakdown` lookup that does not exist.
- `krasnov_m1_per_seed_per_split_5fold_5metal` (25 / 125) and
  `krasnov_m21_per_cell_line_per_seed` (9 / 45) — the per-(seed × fold) and per-(cell × seed)
  records are not on disk. `per_cell_line` records carry `per_seed_available=false`; the spec's
  own template source holds 0 matching files. Both need a **GPU retrain**.

**Group D — manufacturing the number would be dishonest (1 gate).**

- `rusanov_m2_per_cell_line_per_cohort` — the ridge model is CPU-cheap
  (`elapsed_sec` 0.08) and the per-cell-line axis could be computed on CPU, so the 12 records
  are *computable*. But `metrics/by_metric/m2_r2_per_cohort.json` records
  `label_provenance.labels = SIMULATED`, and `.data/molmetal/platinai_4cell.csv` is classified
  **DO-NOT-TRAIN** (*'Its ~0.50 AUC is a null test, not a measurement'*). Filling the 12-record
  gap from that source would produce a number that is not a measurement. **Verdict:
  NEEDS-MEASUREMENT — on a real-label cohort.**

**Group E — the 9 L5 hooks (9 gates).** DEFER-BY-DESIGN — the three reasons in §4.

---

## §6 Two doc-staleness findings that the wiring decision depends on

**(a) The plan doc's DrugOOD 'current state' is stale.** `PRE_TRAINING_GATE_v2.md`
§4 (lines 323/331/339) says the source is `wf265_collect/all_results.json` with
*'metacytotox_fallback assay 0.7479'*. The actual measurement lives in
`metrics/by_metric/drugood_ood_auc.json` (dated 2026-10-07): **official** DrugOOD IC50 axis
partitions, 3 seeds, per-axis OOD AUC 0.730448 / 0.690245 / 0.676347, deltas vs iMoLD
+0.009348 / +0.001845 / -0.002853. The plan doc's 0.7479 is not in that file.
`TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` §1 still describes m17 as
*'NOT ALIGN — fallback cohort (5,207-row metacytotox_fallback)'*, which the 2026-10-07 metric
JSON contradicts. The metric JSON's own `honest_framing` adds the residual caveat:
*'Backbone is a Morgan-FP MLP, NOT the graph NN iMoLD uses, so the AUCs are
protocol-comparable but NOT backbone-comparable.'*

The official-data claim is cross-checked against disk, not taken on the metric's word:
`.data/molmetal/drugood/ic50/csv/lbap_refined_ic50_{assay,scaffold,size}.csv` exist with
**{'lbap_refined_ic50_assay.csv': 266522, 'lbap_refined_ic50_scaffold.csv': 266522, 'lbap_refined_ic50_size.csv': 266522}** data rows each, and their header carries the
DrugOOD-published partition columns
(`config,split,smiles,y,assay_id,cls_label,domain_id,scaffold_split_id,size_split_id,protein_split_id,protein_family_split_id`). The metric JSON's `data_source` is
`drugood_official_lbap_refined_ic50` (dated 2026-10-07), so the
'fallback cohort' description in the matrix is stale, not the metric.

**(b) `PROTOCOL_ALIGNMENT_MATRIX.md` has no §1.4.** The path the 2 L5 doc-presence gates
read is `TODO/03_plans/PROTOCOL_ALIGNMENT_MATRIX.md`, which **does not exist**; the canonical
file is `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md`, and its headings are
§0–§4 — **§1.4 is absent** (§1.4 present = False). The spec cites
'§1.4' three times (lines 287, 313, and the L5 blocks). Note that `wfa171_l5_threshold.py`
uses the CORRECT path; the canonical arg set in
`molmetal/tests/test_wf433a1_gate_v2_wire.py` uses the wrong one — so the staged wiring
patch must be corrected before it is applied.

---

## §7 Reproduction

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/reports/wfa175_dead_gate_wiring/_build_table.py
CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/reports/wfa175_dead_gate_wiring/_build_report.py
```

Both scripts are CPU-only, write only into this directory, and read the repo read-only.
`_build_table.py` calls all 24 gate functions (which is how the live-verdict column is
measured) and then **removes the `molmetal/reports/protocol_validation/` directory it
created**, so the tree is left exactly as found — the write side effect of §4(b) is
reproduced and undone on every run.

