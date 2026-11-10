#!/usr/bin/env python3
"""Wf-A175 — render report.md from decision_table.json (same directory)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
OUT = REPO / "molmetal/reports/wfa175_dead_gate_wiring"
d = json.loads((OUT / "decision_table.json").read_text(encoding="utf-8"))
G = {g["gate"]: g for g in d["gates"]}
S = d["supporting_measurements"]

ORDER = [
    ("L1 — Tier-1 must-pass (BLOCKING per PRE_TRAINING_GATE_v2.md §2)", [
        "p3_protocol_alignment_dataset_match",
        "p3_protocol_alignment_split_match",
        "p3_protocol_alignment_metric_match",
        "p3_protocol_alignment_seed_cohort_per_seed_per_split",
        "p3_protocol_alignment_honest_limitation_minor_section",
        "p3_protocol_alignment_paper_section_6_template",
        "p3_protocol_alignment_user_ok_gate",
        "p3_protocol_alignment_baseline_paper_refs_bib",
    ]),
    ("L2 — Tier-2 recommended (WARN-only per §3)", [
        "p3_protocol_recommended_seed_count_3",
        "p3_protocol_recommended_per_cell_breakdown",
        "p3_protocol_recommended_5_fold_cv_compatibility_50x",
        "p3_protocol_recommended_krasnov_paper_table_4_replication",
        "p3_protocol_recommended_drugood_3_splits_full_alignment",
        "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup",
        "p3_protocol_recommended_section_6_honest_minor_section_auto_emit",
    ]),
    ("L5 — per-baseline protocol-validation hooks (post-launch, §4)", [
        "krasnov_m1_per_seed_per_split_5fold_5metal",
        "krasnov_m21_per_cell_line_per_seed",
        "krasnov_protocol_alignment_diffs_documented",
        "rusanov_m2_per_cell_line_per_cohort",
        "rusanov_m2_4cell_baseline_comparison",
        "rusanov_protocol_alignment_diffs_documented",
        "drugood_m17_assay_split_auc",
        "drugood_m17_scaffold_split_auc",
        "drugood_m17_size_split_auc",
    ]),
]

L: list[str] = []
A = L.append

A("# Wf-A175 — DEAD-GATE WIRING DECISION TABLE")
A("")
A("**Branch:** WIRING · **Date:** 2026-11-01 · **CPU-only** (no GPU, no network, no `paper/**` write)")
A("**Deliverable:** a decision table, NOT a patch. Nothing was wired. Nothing outside")
A("`molmetal/reports/wfa175_dead_gate_wiring/` was written.")
A("")
A("Machine-readable twin: `decision_table.json` (one record per gate).")
A("Generator: `_build_table.py` / `_build_report.py` (both re-runnable, both read-only elsewhere).")
A("")
A("---")
A("")
A("## §0 Headline (all numbers MEASURED 2026-11-01)")
A("")
A(f"- **0 of {d['target_gate_count']}** dead gates are registered in **any** pipeline "
  f"(`all`={d['live_gate_counts']['all']}, `launch`={d['live_gate_counts']['launch']}, "
  f"`frontier`={d['live_gate_counts']['frontier']}, "
  f"`chemprop_v2_v3`={d['live_gate_counts']['chemprop_v2_v3']}, "
  f"`cfm`={d['live_gate_counts']['cfm']}; introspection via "
  f"`pre_flight_check.build_gates(<pipeline>)`).")
A(f"- **Invoking** all 24 against the real tree gives "
  f"**{d['live_verdict_histogram']['PASS']} PASS / {d['live_verdict_histogram']['WARN']} WARN / "
  f"{d['live_verdict_histogram']['FAIL']} FAIL / {d['live_verdict_histogram']['RAISED']} RAISED** "
  f"(denominator 24). This independently reproduces the Wf-A169 number "
  f"(`docs/HONEST_NEGATIVE_CATALOG.md` §6.266.5).")
A(f"- **Spec-threshold status:** {d['status_histogram']['MET']} MET / "
  f"{d['status_histogram']['SHORT']} SHORT / {d['status_histogram']['MISSING_SOURCE']} MISSING_SOURCE / "
  f"{d['status_histogram']['UNSPEC']} UNSPEC.")
A(f"- **Wiring verdict:** {d['verdict_histogram']['WIREABLE-BY-READONLY-REPAIR']} WIREABLE-BY-READONLY-REPAIR / "
  f"{d['verdict_histogram']['NEEDS-MEASUREMENT']} NEEDS-MEASUREMENT / "
  f"{d['verdict_histogram']['NEEDS-GPU']} NEEDS-GPU / "
  f"{d['verdict_histogram']['NEEDS-PAPER-AUTH']} NEEDS-PAPER-AUTH / "
  f"{d['verdict_histogram']['DEFER-BY-DESIGN']} DEFER-BY-DESIGN.")
A(f"- **The PASS column is not a quality column.** By implementation, the 24 break down as "
  f"**{d['implementation_kind_histogram']['path']} pure path-existence checks / "
  f"{d['implementation_kind_histogram']['substring']} single-substring checks / "
  f"{d['implementation_kind_histogram']['parse']} JSON-parsing check** "
  f"(classification verified against each gate's own source with an assertion, not hand-typed). "
  f"Of the 19 live PASSes, 9 are the L5 stubs and 6 more check nothing but a path's existence "
  f"while the spec's own sub-condition is unmet. See §3.")
A("")
A("---")
A("")
A("## §1 The decision table")
A("")
A("`spec_threshold` is quoted VERBATIM from the `**Verdict gate**:` line of")
A("`TODO/03_plans/PRE_TRAINING_GATE_v2.md` at the recorded `spec_line`.")
A("`measured_num / measured_den` are measured against that threshold — NOT against the")
A("gate's own implementation (the two disagree for most of the 24; see §3).")
A("`implementation_kind` says what the gate body actually does: `path` = only `.exists()` on")
A("its arguments, `substring` = reads a file and looks for a literal string, `parse` = parses")
A("JSON and inspects keys. **Not one of the 24 evaluates the spec's threshold.**")
A("")

for title, names in ORDER:
    A(f"### {title}")
    A("")
    A("| gate | spec line | spec threshold (verbatim, abridged) | measured | status | impl | verdict |")
    A("|---|---|---|---|---|---|---|")
    for n in names:
        g = G[n]
        thr = g["spec_threshold"].replace("**", "")
        thr = thr if len(thr) <= 105 else thr[:102] + "…"
        num = g["measured_num"]
        den = g["measured_den"]
        if num is None:
            meas = "—"
        elif den in (None, -0.02) and isinstance(num, float) and num < 0:
            meas = f"{num:+.6f} vs bar {den}"
        elif isinstance(num, float):
            meas = f"{num:+.6f} vs bar {den}"
        else:
            meas = f"{num} / {den}"
        A(f"| `{n}` | {g['spec_line']} | {thr} | {meas} | **{g['status']}** | "
          f"{g['implementation_kind']} | {g['verdict']} |")
    A("")

A("---")
A("")
A("## §2 Per-gate blockers (why)")
A("")
for title, names in ORDER:
    A(f"**{title}**")
    A("")
    for n in names:
        g = G[n]
        A(f"- **`{n}`** — {g['blocker']}")
        A(f"  - evidence: `{g['evidence_path']}`")
    A("")

A("---")
A("")
A("## §3 The gap between what the gates check and what the spec asks")
A("")
A("This is the finding that decides the wiring: **the gate bodies are almost entirely")
A("file-existence checks, while the spec's verdict gates are content checks.**")
A("Six independent measurements, all taken from disk:")
A("")
A("| spec asks for | gate actually checks | measured |")
A("|---|---|---|")
A(f"| L1-2 `leakage_utils.py::scaffold_split` exists | `leakage_utils.py` exists | "
  f"symbol present = **{S['leakage_utils_has_scaffold_split']}**; the scaffold splitter is the class "
  f"`ScaffoldSplitter` in `molmetal/data/splits.py` (present = {S['scaffold_splitter_impl_in_molmetal_data_splits']}) |")
A(f"| L1-2 `all_results.json` has assay/scaffold/size | file exists | splits found = "
  f"**{S['drugood_splits_in_all_results_json']} / 3** (the 3 live in `metrics/by_metric/drugood_ood_auc.json`: "
  f"**{S['drugood_splits_in_metric_json']} / 3**) |")
A(f"| L1-4 `eval_per_seed_per_split` registered in `parse_args()` | 3 trainer files exist | "
  f"flag occurrences = **{S['trainers_with_eval_per_seed_per_split_flag']} / 3** |")
A(f"| L2-2 `--per-cell-breakdown` flag registered | 2 trainer files exist | "
  f"flag occurrences = **{S['trainers_with_per_cell_breakdown_flag']} / 2** |")
A(f"| L2-3 `--eval-5fold-cv` flag registered | 2 files exist | "
  f"5-fold flag occurrences = **{S['sources_with_5fold_flag']} / 2** |")
A(f"| L1-7 USER_OK gate **positioned before** the `--dry-run` exit block | the string `USER_OK` appears anywhere | "
  f"positioning holds in **{S['wrappers_with_user_ok_before_dry_run_exit']} / {S['wrappers_total']}** wrappers |")
A("")
A(f"Two more, in the other direction — the gates are **stricter** than their spec:")
A("")
A(f"- **L2-1** (`seed_count_3`): the spec wants `--seeds 0 1 2` defaults; measured "
  f"**{S['trainers_with_seeds_default_0_1_2']} / 3** trainers default to `[0,1,2]` and "
  f"**{S['trainers_emitting_n_seeds']} / 3** emit `n_seeds`. The spec condition is MET; the gate only checks file existence.")
A(f"- **L2-5** (`drugood_3_splits`): the spec wants all 3 splits + an iMoLD-gate PASS per split. "
  f"`metrics/by_metric/drugood_ood_auc.json` carries all 3 splits with `gate_pass` = "
  f"{S['drugood_per_split_gate_pass']} — but the gate reads `all_results.json`, which has **0 / 3**. "
  f"A path fix alone is not enough: the gate's detector (`split in data or any(split in k for k in data.keys())`) "
  f"would still miss splits nested under a list-valued key.")
A("")
A(f"**L1-5** is the one place where the spec's wording and the repo's stricter rule diverge: "
  f"the spec asks for '§6 catalog entry exists for that metric' — measured "
  f"**{S['catalog_section6_lead_metric_coverage_loose']} / 7** Tier-1 LEAD metrics have one (m1, m2, m3, m7, m15, m17, m21). "
  f"Under the repo's own rule in `molmetal/scripts/wfa171_l5_threshold.py` (metric token **AND** a "
  f"protocol/split token) only **{S['catalog_section6_lead_metric_coverage_strict']} / 7** qualify — m15 and m17 have "
  f"no §6 entry naming a protocol difference. The gate itself checks only that the string `## §6` appears.")
A("")
A("**L1-3** (`metric_match`) is the one verdict that cannot be repaired at all: 'all 7 metric")
A("definitions are bit-for-bit identical to baseline' has no comparison oracle in the repo, and")
A("the source reports the spec names (`molmetal/reports/wf331_r1{1,2,3}_*.json`) are not on disk.")
A("Its status is therefore **UNSPEC**, not SHORT: there is no measurable definition to miss.")
A("")
A("---")
A("")
A("## §4 The 9 L5 hooks: three separate reasons not to wire them")
A("")
A("**(a) The spec's own routing rule excludes them.** `PRE_TRAINING_GATE_v2.md` §1.3:")
A("*'L5 per-baseline protocol validation hooks hook to … (no main-Layer-0 routing — these are")
A("post-launch hooks)'*. §0/§1.2 nevertheless count them in the `95 + 32 = 127` arithmetic.")
A("The spec contradicts itself (§6.266.4). Wiring all 24 into `all` would take it to")
A(f"**{d['live_gate_counts']['all'] + 24}** (and `launch` to **{d['live_gate_counts']['launch'] + 24}**) — ")
A("violating §1.3. This needs a user-level ruling, not a patch.")
A("")
A("**(b) They are not read-only.** MEASURED: calling any of the 9 writes")
A("`molmetal/reports/protocol_validation/{krasnov,rusanov,drugood}_hook.json`")
A("(the directory did not exist before the probe — `docs/HONEST_NEGATIVE_CATALOG.md` §6.266.3")
A("records it as absent — and was removed again afterwards to restore the tree). The emitter")
A("now creates `hook_dir` and merges sibling `hook_id`s, so the side effect is *by design*")
A("for a post-launch hook, and *wrong* for a pre-flight gate.")
A("")
A("**(c) Three are SHORT and one is MISSING_SOURCE on their own thresholds** — measured with the")
A("repo's own read-only module `molmetal/scripts/wfa171_l5_threshold.py` (thresholds parsed from")
A("the plan doc at runtime):")
A("")
A("| L5 hook | spec threshold | measured | status |")
A("|---|---|---|---|")
for n in ("krasnov_m1_per_seed_per_split_5fold_5metal",
          "krasnov_m21_per_cell_line_per_seed",
          "rusanov_m2_per_cell_line_per_cohort",
          "rusanov_m2_4cell_baseline_comparison"):
    g = G[n]
    A(f"| `{n}` | {g['spec_line']} | {g['measured_num']} / {g['measured_den']} | {g['status']} |")
A("")
A("The remaining 6 L5 hooks (2 doc-presence + 3 DrugOOD value + the L1/L2 sibling) are")
A("numerically MET — but 'MET' here means *the threshold is met*, not *the gate checks it*.")
A("The gate checks file existence only; the threshold comparison lives in `wfa171`, which is")
A("**not wired into the gate**. Wiring the 9 as-is would add 9 vacuous PASSes to a 129-gate")
A("pre-flight.")
A("")
A("---")
A("")
A("## §5 The MINIMAL, READ-ONLY-ONLY wiring set")
A("")
A("### §5.1 Wire TODAY as-is (4 gates) — GREEN, read-only, no GPU, no paper write")
A("")
for n in d["minimal_wiring_set"]["wire_today"]:
    A(f"- `{n}` — spec threshold **{G[n]['status']}**; live verdict "
      f"**{G[n]['live_verdict_if_invoked']}**.")
A("")
A("All four are GREEN on the real tree today. Wiring them changes `all` "
  f"{d['live_gate_counts']['all']} → {d['live_gate_counts']['all'] + 4} and `launch` "
  f"{d['live_gate_counts']['launch']} → {d['live_gate_counts']['launch'] + 4}; `frontier` / "
  "`chemprop_v2_v3` / `cfm` unchanged.")
A("")
A("### §5.2 Wire TODAY after a READ-ONLY repair (4 gates) — still GREEN, still no GPU, still no paper write")
A("")
REPAIR = {
    "p3_protocol_alignment_dataset_match":
        "correct the evidence string — the gate enumerates 6 dataset paths and then asserts "
        "'all 7 Tier-1 LEAD metric datasets'. One string edit in `pre_flight_check.py`; the "
        "6 paths all exist.",
    "p3_protocol_alignment_split_match":
        "re-point both arguments: `leakage_utils.py` → `molmetal/data/splits.py` (where "
        "`ScaffoldSplitter` actually lives) and `all_results.json` → "
        "`metrics/by_metric/drugood_ood_auc.json` (which carries all 3 splits). Both targets "
        "exist, so the repaired gate is GREEN. **Until then it must NOT be wired** (see §5.3 "
        "Group B).",
    "p3_protocol_recommended_drugood_3_splits_full_alignment":
        "re-point the source AND fix the detector: the splits sit under a nested key, so the "
        "current `split in data or any(split in k for k in data.keys())` scan would still miss "
        "them after a path fix. Two edits, both in `pre_flight_check.py`; the repaired gate is "
        "GREEN (3/3 splits, `gate_pass` true for each).",
    "p3_protocol_recommended_section_6_honest_minor_section_auto_emit":
        "author the missing emitter `molmetal/scripts/auto_emit_section_6_minor.py`. That is a "
        "NEW file under `molmetal/scripts/` — NOT a `paper/` write, so AGENTS.md §5.3 is not "
        "engaged as long as the emitted section is not injected into `paper/`.",
}
for n in d["minimal_wiring_set"]["wire_today_after_readonly_repair"]:
    A(f"- `{n}` — spec threshold **{G[n]['status']}**. Repair: {REPAIR[n]}")
A("")
A("These four are the only ones where a read-only edit makes the gate *mean* something it")
A("currently does not: the first three are re-pointed at the file that actually holds the")
A("data, the fourth needs its emitter script authored (a `molmetal/scripts/` file — NOT a")
A("`paper/` write, so §5.3 is not engaged as long as the emitted section is not injected into")
A("`paper/`).")
A("")
A("### §5.3 MUST NOT be wired today — and why")
A("")
A("**Group A — wiring it would turn the pre-flight RED over a non-launch defect (1 gate).**")
A("")
A(f"- `p3_protocol_alignment_baseline_paper_refs_bib` — today **{G['p3_protocol_alignment_baseline_paper_refs_bib']['live_verdict_if_invoked']}** "
  f"({S['refs_bib_keys_present']} / 3 keys present in `paper/refs.bib`). It is a Tier-1 BLOCKING gate; "
  "wiring it as-is turns `all` RED today and blocks every launch. Closing it requires a write")
A("  under `paper/**` — AGENTS.md §5.3 / TODO-84 forbid that without the user's explicit OK.")
A("  **Verdict: NEEDS-PAPER-AUTH.**")
A("")
A("**Group B — the spec's own sub-condition is unmet, so the gate would be a vacuous PASS (3 gates).**")
A("")
A("(`p3_protocol_alignment_split_match` also belongs here until its repair lands — it is")
A("listed in §5.2 because the repair is read-only, but wiring it unrepaired would add a PASS")
A("that directly contradicts L2-5's WARN on the same `all_results.json`.)")
A("")
A("- `p3_protocol_alignment_seed_cohort_per_seed_per_split` — `eval_per_seed_per_split` occurs")
A(f"  {S['trainers_with_eval_per_seed_per_split_flag']} / 3 times. Closing it is a **trainer** wire-in, not a pre-flight edit;")
A("  AGENTS.md §5.7 would then require a 50-step CPU smoke before it can be called shipped.")
A("- `p3_protocol_recommended_per_cell_breakdown` and")
A("  `p3_protocol_recommended_5_fold_cv_compatibility_50x` — same class: the named CLI flags")
A(f"  do not exist ({S['trainers_with_per_cell_breakdown_flag']} / 2 and {S['sources_with_5fold_flag']} / 2).")
A("")
A("**Group C — the threshold needs a measurement that does not exist yet (4 gates).**")
A("")
A("- `p3_protocol_recommended_krasnov_paper_table_4_replication` — the directory is absent, but")
A("  the spec's verdict is 'PASS if template file present', so an EMPTY directory would satisfy")
A("  it. Creating one turns a missing measurement into a PASS over nothing.")
A("- `p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup` — needs the Rusanov 2025")
A("  Table S4-S5 values (an EXTERNAL primary record — a search summary will not do) plus a")
A("  `--per-cell-breakdown` lookup that does not exist.")
A("- `krasnov_m1_per_seed_per_split_5fold_5metal` (25 / 125) and")
A("  `krasnov_m21_per_cell_line_per_seed` (9 / 45) — the per-(seed × fold) and per-(cell × seed)")
A("  records are not on disk. `per_cell_line` records carry `per_seed_available=false`; the spec's")
A(f"  own template source holds {S['wf131_per_seed_template_files']} matching files. Both need a **GPU retrain**.")
A("")
A("**Group D — manufacturing the number would be dishonest (1 gate).**")
A("")
A("- `rusanov_m2_per_cell_line_per_cohort` — the ridge model is CPU-cheap")
A("  (`elapsed_sec` 0.08) and the per-cell-line axis could be computed on CPU, so the 12 records")
A("  are *computable*. But `metrics/by_metric/m2_r2_per_cohort.json` records")
A("  `label_provenance.labels = SIMULATED`, and `.data/molmetal/platinai_4cell.csv` is classified")
A("  **DO-NOT-TRAIN** (*'Its ~0.50 AUC is a null test, not a measurement'*). Filling the 12-record")
A("  gap from that source would produce a number that is not a measurement. **Verdict:")
A("  NEEDS-MEASUREMENT — on a real-label cohort.**")
A("")
A("**Group E — the 9 L5 hooks (9 gates).** DEFER-BY-DESIGN — the three reasons in §4.")
A("")
A("---")
A("")
A("## §6 Two doc-staleness findings that the wiring decision depends on")
A("")
A("**(a) The plan doc's DrugOOD 'current state' is stale.** `PRE_TRAINING_GATE_v2.md`")
A("§4 (lines 323/331/339) says the source is `wf265_collect/all_results.json` with")
A("*'metacytotox_fallback assay 0.7479'*. The actual measurement lives in")
A("`metrics/by_metric/drugood_ood_auc.json` (dated 2026-10-07): **official** DrugOOD IC50 axis")
A("partitions, 3 seeds, per-axis OOD AUC 0.730448 / 0.690245 / 0.676347, deltas vs iMoLD")
A("+0.009348 / +0.001845 / -0.002853. The plan doc's 0.7479 is not in that file.")
A("`TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` §1 still describes m17 as")
A("*'NOT ALIGN — fallback cohort (5,207-row metacytotox_fallback)'*, which the 2026-10-07 metric")
A("JSON contradicts. The metric JSON's own `honest_framing` adds the residual caveat:")
A("*'Backbone is a Morgan-FP MLP, NOT the graph NN iMoLD uses, so the AUCs are")
A("protocol-comparable but NOT backbone-comparable.'*")
A("")
A("The official-data claim is cross-checked against disk, not taken on the metric's word:")
A(f"`.data/molmetal/drugood/ic50/csv/lbap_refined_ic50_{{assay,scaffold,size}}.csv` exist with")
A(f"**{S['drugood_official_csv_data_rows']}** data rows each, and their header carries the")
A("DrugOOD-published partition columns")
A(f"(`{S['drugood_official_csv_columns']}`). The metric JSON's `data_source` is")
A(f"`{S['drugood_metric_json_source']}` (dated {S['drugood_metric_json_date']}), so the")
A("'fallback cohort' description in the matrix is stale, not the metric.")
A("")
A("**(b) `PROTOCOL_ALIGNMENT_MATRIX.md` has no §1.4.** The path the 2 L5 doc-presence gates")
A("read is `TODO/03_plans/PROTOCOL_ALIGNMENT_MATRIX.md`, which **does not exist**; the canonical")
A("file is `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md`, and its headings are")
A(f"§0–§4 — **§1.4 is absent** (§1.4 present = {S['protocol_alignment_matrix_has_section_1_4']}). The spec cites")
A("'§1.4' three times (lines 287, 313, and the L5 blocks). Note that `wfa171_l5_threshold.py`")
A("uses the CORRECT path; the canonical arg set in")
A("`molmetal/tests/test_wf433a1_gate_v2_wire.py` uses the wrong one — so the staged wiring")
A("patch must be corrected before it is applied.")
A("")
A("---")
A("")
A("## §7 Reproduction")
A("")
A("```")
A("CUDA_VISIBLE_DEVICES=\"\" .venv/bin/python molmetal/reports/wfa175_dead_gate_wiring/_build_table.py")
A("CUDA_VISIBLE_DEVICES=\"\" .venv/bin/python molmetal/reports/wfa175_dead_gate_wiring/_build_report.py")
A("```")
A("")
A("Both scripts are CPU-only, write only into this directory, and read the repo read-only.")
A("`_build_table.py` calls all 24 gate functions (which is how the live-verdict column is")
A("measured) and then **removes the `molmetal/reports/protocol_validation/` directory it")
A("created**, so the tree is left exactly as found — the write side effect of §4(b) is")
A("reproduced and undone on every run.")
A("")

sys.path.insert(0, str(REPO))
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

atomic_write_file(str(OUT / "report.md"), "\n".join(L) + "\n")
print("wrote", OUT / "report.md", len("\n".join(L)), "chars")
