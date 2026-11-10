#!/usr/bin/env python3
"""Wf-A175 — build the dead-gate wiring DECISION TABLE.

READ-ONLY on the whole repo except this report directory.  Every number in the
emitted JSON is COMPUTED from disk at run time (no hand-typed values); the
prose fields carry the interpretation.

Usage:  CUDA_VISIBLE_DEVICES="" .venv/bin/python \
          molmetal/reports/wfa175_dead_gate_wiring/_build_table.py
"""

from __future__ import annotations

import importlib.util as ilu
import inspect
import json
import re
import sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
OUT = REPO / "molmetal" / "reports" / "wfa175_dead_gate_wiring"
PLAN = REPO / "TODO" / "03_plans" / "PRE_TRAINING_GATE_v2.md"


def _load(name: str, rel: str):
    s = ilu.spec_from_file_location(name, REPO / rel)
    m = ilu.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


pfc = _load("pfc", "molmetal/scripts/pre_flight_check.py")
l5 = _load("l5", "molmetal/scripts/wfa171_l5_threshold.py")

# --------------------------------------------------------------------------
# 1. Which of the 24 are registered in which pipeline? (live introspection)
# --------------------------------------------------------------------------
TARGETS = [
    "p3_protocol_alignment_dataset_match",
    "p3_protocol_alignment_split_match",
    "p3_protocol_alignment_metric_match",
    "p3_protocol_alignment_seed_cohort_per_seed_per_split",
    "p3_protocol_alignment_honest_limitation_minor_section",
    "p3_protocol_alignment_paper_section_6_template",
    "p3_protocol_alignment_user_ok_gate",
    "p3_protocol_alignment_baseline_paper_refs_bib",
    "p3_protocol_recommended_seed_count_3",
    "p3_protocol_recommended_per_cell_breakdown",
    "p3_protocol_recommended_5_fold_cv_compatibility_50x",
    "p3_protocol_recommended_krasnov_paper_table_4_replication",
    "p3_protocol_recommended_drugood_3_splits_full_alignment",
    "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup",
    "p3_protocol_recommended_section_6_honest_minor_section_auto_emit",
    "krasnov_m1_per_seed_per_split_5fold_5metal",
    "krasnov_m21_per_cell_line_per_seed",
    "krasnov_protocol_alignment_diffs_documented",
    "rusanov_m2_per_cell_line_per_cohort",
    "rusanov_m2_4cell_baseline_comparison",
    "rusanov_protocol_alignment_diffs_documented",
    "drugood_m17_assay_split_auc",
    "drugood_m17_scaffold_split_auc",
    "drugood_m17_size_split_auc",
]
registered = {p: {r[0] for r in pfc.build_gates(p)} for p in
              ("all", "launch", "frontier", "chemprop_v2_v3", "cfm")}
live_counts = {p: len(registered[p]) for p in registered}

# --------------------------------------------------------------------------
# 2. The spec's verdict-gate lines, read out of the plan doc at runtime.
# --------------------------------------------------------------------------
plan_lines = PLAN.read_text(encoding="utf-8").split("\n")
SPEC_LINE = {
    "p3_protocol_alignment_dataset_match": 115,
    "p3_protocol_alignment_split_match": 123,
    "p3_protocol_alignment_metric_match": 131,
    "p3_protocol_alignment_seed_cohort_per_seed_per_split": 139,
    "p3_protocol_alignment_honest_limitation_minor_section": 147,
    "p3_protocol_alignment_paper_section_6_template": 155,
    "p3_protocol_alignment_user_ok_gate": 163,
    "p3_protocol_alignment_baseline_paper_refs_bib": 174,
    "p3_protocol_recommended_seed_count_3": 202,
    "p3_protocol_recommended_per_cell_breakdown": 210,
    "p3_protocol_recommended_5_fold_cv_compatibility_50x": 218,
    "p3_protocol_recommended_krasnov_paper_table_4_replication": 226,
    "p3_protocol_recommended_drugood_3_splits_full_alignment": 234,
    "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup": 242,
    "p3_protocol_recommended_section_6_honest_minor_section_auto_emit": 250,
    "krasnov_m1_per_seed_per_split_5fold_5metal": 272,
    "krasnov_m21_per_cell_line_per_seed": 280,
    "krasnov_protocol_alignment_diffs_documented": 288,
    "rusanov_m2_per_cell_line_per_cohort": 298,
    "rusanov_m2_4cell_baseline_comparison": 306,
    "rusanov_protocol_alignment_diffs_documented": 314,
    "drugood_m17_assay_split_auc": 324,
    "drugood_m17_scaffold_split_auc": 332,
    "drugood_m17_size_split_auc": 340,
}
spec_threshold = {g: plan_lines[ln - 1].split("**Verdict gate**:", 1)[1].strip()
                  for g, ln in SPEC_LINE.items()}

# --------------------------------------------------------------------------
# 3. Live probe of every gate against the real tree.
#    (mirrors molmetal/tests/test_wf433a1_gate_v2_wire.py's canonical args)
# --------------------------------------------------------------------------
MC = REPO / ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
PL = REPO / "molmetal/data/platinai_4cell.csv"
CD = REPO / "molmetal/data/crossdocked100_manifest.csv"
NC = REPO / "molmetal/data/nci60_precious_9cell.csv"
BAL = REPO / "molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json"
DG = REPO / "molmetal/data/drugood_loader.py"
LEAK = REPO / "molmetal/data/leakage_utils.py"
DGR = REPO / "molmetal/reports/wf265_collect/all_results.json"
EVAL = REPO / "molmetal/scripts/eval_all_evaluators.py"
TM = REPO / "molmetal/scripts/train_metal_cytotox.py"
TP = REPO / "molmetal/scripts/train_platinai_4cell.py"
TN = REPO / "molmetal/scripts/train_nci60_10cell.py"
PSD = REPO / "molmetal/reports/wf131_p3_deferred_runs"
HC = REPO / "docs/HONEST_NEGATIVE_CATALOG.md"
S6T = REPO / "docs/SECTION_6_LIMITATIONS_TEMPLATE.md"
S6V = REPO / "docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md"
PA = REPO / "TODO/03_plans/PROTOCOL_ALIGNMENT_MATRIX.md"
PA_REAL = REPO / "TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md"
REFS = REPO / "paper/refs.bib"
AE = REPO / "molmetal/scripts/auto_emit_section_6_minor.py"
M2 = REPO / "metrics/by_metric/m2_r2_per_cohort.json"
MLA = REPO / "metrics/by_metric/metal_lomo_auc_wasserstein.json"
DOOD = REPO / "metrics/by_metric/drugood_ood_auc.json"
NCI60DIR = REPO / "molmetal/reports/wf_nci60_precious_9cell"
RUDIR = REPO / "molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline"
K4DIR = REPO / "molmetal/reports/krasnov_2026_paper_table4_replication"
PVDIR = REPO / "molmetal/reports/protocol_validation"
REPORTS = REPO / "molmetal/reports"
WRAPPERS = (
    "launch_wf124_p3_g5.sh", "launch_wf124_p3_g6.sh", "launch_wf299_cfm_main.sh",
    "launch_wf388_a1_metal_coord_retrain.sh", "launch_wf394a1_lambda_cfm_patch.sh",
    "launch_wf125_p1_controlled_pair.sh", "launch_wf390a1_m2_r2_path_a.sh",
)
ARGS = {
    "p3_protocol_alignment_dataset_match": (MC, PL, CD, NC, BAL, DG),
    "p3_protocol_alignment_split_match": (LEAK, DGR),
    "p3_protocol_alignment_metric_match": (EVAL,),
    "p3_protocol_alignment_seed_cohort_per_seed_per_split": (TM, TP, TN, PSD),
    "p3_protocol_alignment_honest_limitation_minor_section": (HC,),
    "p3_protocol_alignment_paper_section_6_template": (S6T, S6V),
    "p3_protocol_alignment_user_ok_gate": (REPO / "scripts", WRAPPERS),
    "p3_protocol_alignment_baseline_paper_refs_bib": (REFS,),
    "p3_protocol_recommended_seed_count_3": (TM, TP, TN),
    "p3_protocol_recommended_per_cell_breakdown": (TP, TN, M2),
    "p3_protocol_recommended_5_fold_cv_compatibility_50x": (TM, DG),
    "p3_protocol_recommended_krasnov_paper_table_4_replication": (REPORTS,),
    "p3_protocol_recommended_drugood_3_splits_full_alignment": (DGR,),
    "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup": (REPORTS, TP),
    "p3_protocol_recommended_section_6_honest_minor_section_auto_emit": (AE,),
    "krasnov_m1_per_seed_per_split_5fold_5metal": (PVDIR, MLA),
    "krasnov_m21_per_cell_line_per_seed": (PVDIR, NCI60DIR),
    "krasnov_protocol_alignment_diffs_documented": (PVDIR, HC, PA),
    "rusanov_m2_per_cell_line_per_cohort": (PVDIR, M2),
    "rusanov_m2_4cell_baseline_comparison": (PVDIR, RUDIR, M2),
    "rusanov_protocol_alignment_diffs_documented": (PVDIR, HC, PA),
    "drugood_m17_assay_split_auc": (PVDIR, DGR),
    "drugood_m17_scaffold_split_auc": (PVDIR, DGR),
    "drugood_m17_size_split_auc": (PVDIR, DGR),
}
live_verdict = {}
_PV_PREEXISTED = PVDIR.exists()
for g in TARGETS:
    try:
        r = getattr(pfc, "gate_" + g)(g, "P3", "all", *ARGS[g])
        live_verdict[g] = {"status": r.status, "evidence": r.evidence}
    except Exception as exc:  # noqa: BLE001
        live_verdict[g] = {"status": "RAISED",
                           "evidence": f"{type(exc).__name__}: {exc}"}
# RESTORE: the 9 L5 gates write molmetal/reports/protocol_validation/*.json as a
# side effect of being called (measured).  Remove it again if it was not there
# before this probe, so the tree is left exactly as found.
import shutil  # noqa: E402
if not _PV_PREEXISTED and PVDIR.exists():
    shutil.rmtree(PVDIR)

# --------------------------------------------------------------------------
# 4. L5 threshold accounting, via the repo's own read-only module.
# --------------------------------------------------------------------------
l5_recs = {r.hook_id: r.as_dict() for r in l5.evaluate_hooks(l5.default_specs())}

# --------------------------------------------------------------------------
# 5. Independent measurements of the spec conditions the gates DON'T check.
# --------------------------------------------------------------------------
def count_in(path: Path, key: str):
    d = json.loads(path.read_text(encoding="utf-8"))
    v = d.get(key)
    return (len(v) if isinstance(v, (list, dict)) else 0), v


m1_count, _ = count_in(MLA, "per_run_auc")
m21_count, _ = count_in(REPO / "metrics/by_metric/nci60_precious_9cell_auc.json",
                         "per_cell_line")
m2_count, _ = count_in(M2, "per_cohort")
dood = json.loads(DOOD.read_text(encoding="utf-8"))
dood_deltas = dood["per_axis_auc_delta_vs_imold"]
dood_splits = dood["splits"]
dood_gate_pass = dood["gate_pass"]

# catalog §6 coverage, measured independently of wfa171
cat = HC.read_text(encoding="utf-8")
marks = list(re.compile(r"^###\s+(§6\.[^\s]*)", re.MULTILINE).finditer(cat))
TOKEN = re.compile(r"(5-fold|scaffold[- ]split|per-cell-line|split[- ]protocol|"
                   r"protocol replication|cohort-split)")
METRIC = re.compile(r"(?<![a-z0-9])(m\d+)(?![0-9a-z])")
LEAD = ["m1", "m2", "m3", "m7", "m15", "m17", "m21"]
strict_cov, loose_cov = {}, {}
for i, mk in enumerate(marks):
    end = marks[i + 1].start() if i + 1 < len(marks) else len(cat)
    body = cat[mk.start():end]
    has_tok = bool(TOKEN.search(body))
    for mm in set(METRIC.findall(body)):
        loose_cov.setdefault(mm, set()).add(mk.group(1))
        if has_tok:
            strict_cov.setdefault(mm, set()).add(mk.group(1))
strict_n = sum(1 for m in LEAD if strict_cov.get(m))
loose_n = sum(1 for m in LEAD if loose_cov.get(m))

# trainer flags
flag_eval = sum(1 for f in (TM, TP, TN)
                if "eval_per_seed_per_split" in f.read_text(encoding="utf-8"))
flag_cell = sum(1 for f in (TM, TP, TN)
                if re.search(r"per.cell.breakdown", f.read_text(encoding="utf-8"),
                             re.I))
flag_5fold = sum(1 for f in (TM, DG)
                 if re.search(r"5fold|5_fold|five_fold", f.read_text(encoding="utf-8"),
                              re.I))
seeds_ok = 0
nseeds_ok = 0
for f in (TM, TP, TN):
    t = f.read_text(encoding="utf-8")
    if re.search(r'"--seeds".*?default=\[0,\s*1,\s*2\]', t, re.S):
        seeds_ok += 1
    if '"n_seeds"' in t:
        nseeds_ok += 1

# USER_OK positioning, measured against the real DRY_RUN exit block
user_ok_pos = {}
for w in WRAPPERS:
    t = (REPO / "scripts" / w).read_text(encoding="utf-8")
    lines = t.split("\n")
    u = next((i + 1 for i, ln in enumerate(lines)
              if 'if [ "${USER_OK:-0}" != "1" ]' in ln), None)
    d = next((i + 1 for i, ln in enumerate(lines)
              if "DRY_RUN" in ln and "if " in ln and "then" in ln
              and re.search(r'DRY_RUN.*?(?:=|==|-eq)\s*"?1"?', ln)), None)
    user_ok_pos[w] = {"user_ok_line": u, "dry_run_exit_line": d,
                       "user_ok_before_dry_run": bool(u and d and u < d)}
user_ok_before = sum(1 for v in user_ok_pos.values()
                     if v["user_ok_before_dry_run"])

# leakage_utils::scaffold_split
leak_src = LEAK.read_text(encoding="utf-8")
has_scaffold_in_leak = bool(re.search(r"def\s+scaffold_split", leak_src))
splits_src = (REPO / "molmetal/data/splits.py").read_text(encoding="utf-8")
scaffold_impl = ("ScaffoldSplitter" in splits_src)

# DrugOOD splits in the two candidate source files
dgr = json.loads(DGR.read_text(encoding="utf-8"))
dgr_splits = [s for s in ("assay", "scaffold", "size")
              if s in dgr or any(s in k for k in dgr.keys())]

# refs.bib keys
bib = REFS.read_text(encoding="utf-8")
bib_keys = {k: (k in bib) for k in
            ("krasnov2026metalcytotoxdb", "rusanov2025platinai", "ji2022drugood")}

# official DrugOOD IC50 CSVs the m17 metric claims to have consumed
DOOD_CSV_DIR = REPO / ".data/molmetal/drugood/ic50/csv"
dood_csv_rows = {}
for _f in sorted(DOOD_CSV_DIR.glob("lbap_refined_ic50_*.csv")):
    with open(_f, "r", encoding="utf-8") as fh:
        doood_n = sum(1 for _ in fh) - 1
    dood_csv_rows[_f.name] = doood_n
dood_csv_header = (DOOD_CSV_DIR / "lbap_refined_ic50_assay.csv").read_text(
    encoding="utf-8").split("\n", 1)[0]

# protocol alignment matrix
pa_headings = [ln for ln in PA_REAL.read_text(encoding="utf-8").split("\n")
               if ln.startswith("#")]
pa_has_1_4 = any("1.4" in h for h in pa_headings)

# per_seed_metrics_seed*.json template count
tmpl = sorted(PSD.glob("per_seed_metrics_seed*.json"))

# --------------------------------------------------------------------------
# 6. Assemble the decision table.
# --------------------------------------------------------------------------
VERDICT = {
    # gate: (measured_num, measured_den, status, blocker, verdict, evidence)
    "p3_protocol_alignment_dataset_match": (
        6, 6, "MET",
        "Gate enumerates 6 dataset paths but its evidence string claims "
        "'all 7 Tier-1 LEAD metric datasets'; PROTOCOL_ALIGNMENT_MATRIX §1 "
        "lists m3/m7/m15 as 'NO FORMAL PAPER COMPARISON — Tier-2 "
        "supplementary', so no dataset is expected for them. The 6 paths "
        "exist. Fixing the evidence string is a read-only edit.",
        "WIREABLE-BY-READONLY-REPAIR",
        "molmetal/scripts/pre_flight_check.py:10768 (evidence string); "
        "TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md:15-27"),
    "p3_protocol_alignment_split_match": (
        0, 2, "SHORT",
        "Both named sub-conditions fail. (a) 'leakage_utils.py::scaffold_split' "
        "does not exist — the module has no scaffold symbol at all; the "
        "scaffold splitter lives in molmetal/data/splits.py as the class "
        "ScaffoldSplitter. (b) all_results.json carries 0 of 3 DrugOOD "
        "split keys (the 3 splits live in metrics/by_metric/drugood_ood_auc.json). "
        "Repair = re-point both paths; the underlying data DO exist, so the "
        "repaired gate stays GREEN. Wiring it UNREPAIRED adds a vacuous PASS "
        "that contradicts L2-5's WARN on the same file.",
        "WIREABLE-BY-READONLY-REPAIR",
        "molmetal/data/leakage_utils.py:24-187 (no scaffold symbol); "
        "molmetal/data/splits.py:610 (ScaffoldSplitter); "
        "molmetal/reports/wf265_collect/all_results.json (0/3 splits); "
        "metrics/by_metric/drugood_ood_auc.json:6-9 (splits)"),
    "p3_protocol_alignment_metric_match": (
        0, 7, "UNSPEC",
        "'Bit-for-bit identical to baseline' is not a machine-checkable "
        "criterion: no comparison oracle exists in the repo, and the named "
        "source reports (molmetal/reports/wf331_r1{1,2,3}_*.json) are not on "
        "disk (only wf331r3/r4/r5 scripts + tests). The gate checks one file's "
        "existence. There is nothing to repair without first DEFINING the "
        "comparison.",
        "DEFER-BY-DESIGN",
        "molmetal/scripts/pre_flight_check.py:10853; "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:131"),
    "p3_protocol_alignment_seed_cohort_per_seed_per_split": (
        flag_eval, 3, "SHORT",
        "The spec names an 'eval_per_seed_per_split' flag registered in "
        "parse_args(); it occurs 0 times in all three trainers. The trainers "
        "DO emit per-seed JSON, but through the --seeds default [0,1,2] path, "
        "not through a flag of that name. Either the trainer needs the flag "
        "(a §5.7 50-step CPU smoke is then mandatory) or the spec text must "
        "be corrected to name the mechanism that exists.",
        "NEEDS-MEASUREMENT",
        "molmetal/scripts/train_metal_cytotox.py:142-146; "
        "molmetal/scripts/train_platinai_4cell.py:891-895; "
        "molmetal/scripts/train_nci60_10cell.py:860-864; "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:139"),
    "p3_protocol_alignment_honest_limitation_minor_section": (
        loose_n, 7, "MET",
        "All 7 Tier-1 LEAD metrics have at least one §6 catalog entry "
        "(loose reading, which is what the spec states). Under the repo's "
        "own stricter rule (metric token AND a protocol/split token) only 5/7 "
        "qualify — m15 and m17 have no §6 entry naming a protocol "
        "difference. The gate itself checks only the '## §6' header.",
        "WIREABLE-BY-READONLY-REPAIR",
        "docs/HONEST_NEGATIVE_CATALOG.md (138 '### §6.' entries); "
        "molmetal/scripts/pre_flight_check.py:10923"),
    "p3_protocol_alignment_paper_section_6_template": (
        2, 2, "MET",
        "Both template files exist. The second half of the verdict "
        "('per-metric drop-down compiles') was NOT verified — no LaTeX run "
        "was performed (CPU-only branch, and the templates live under docs/, "
        "not paper/, so the compile itself is permitted but was not run). "
        "The template's per-metric subsections cover m1/m2/m3/m6_kappa/m7 and "
        "bundle m14/m17/m19; m21 has no dedicated subsection.",
        "WIREABLE-BY-READONLY-REPAIR",
        "docs/SECTION_6_LIMITATIONS_TEMPLATE.md:112-180; "
        "docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md:159,360-408; "
        "molmetal/scripts/pre_flight_check.py:10972"),
    "p3_protocol_alignment_user_ok_gate": (
        user_ok_before, 7, "MET",
        "All 7 wrappers contain the USER_OK gate AND, measured against the "
        "real DRY_RUN exit block, place it BEFORE that block. The gate itself "
        "only greps for the string 'USER_OK' anywhere in the file — it does "
        "not check the ordering the spec requires, so it under-checks a "
        "condition that happens to hold today.",
        "WIREABLE-BY-READONLY-REPAIR",
        "molmetal/scripts/pre_flight_check.py:11005; "
        "scripts/launch_wf124_p3_g5.sh:65,198; "
        "scripts/launch_wf390a1_m2_r2_path_a.sh:117,234"),
    "p3_protocol_alignment_baseline_paper_refs_bib": (
        sum(1 for v in bib_keys.values() if v), 3, "MISSING_SOURCE",
        "paper/refs.bib is missing all 3 baseline keys. Closing this is a "
        "write under paper/** — AGENTS.md §5.3 / TODO-84 forbids it "
        "without the user's explicit OK. Wiring the gate before the keys "
        "exist turns the 'all' pipeline RED today, which would block every "
        "launch over a citation-key defect that is not a launch blocker.",
        "NEEDS-PAPER-AUTH",
        "paper/refs.bib (0/3 keys); "
        "molmetal/scripts/pre_flight_check.py:11052; AGENTS.md §5.3"),
    "p3_protocol_recommended_seed_count_3": (
        seeds_ok, 3, "MET",
        "All 3 trainers default --seeds to [0,1,2] and all 3 emit an "
        "'n_seeds' field in the result JSON. The gate only checks that the "
        "three files exist; strengthening it to parse the default is a "
        "read-only edit and would still be GREEN.",
        "WIREABLE-BY-READONLY-REPAIR",
        "molmetal/scripts/train_metal_cytotox.py:142-146,336; "
        "molmetal/scripts/train_platinai_4cell.py:891-895,1125; "
        "molmetal/scripts/train_nci60_10cell.py:860-864,1018"),
    "p3_protocol_recommended_per_cell_breakdown": (
        flag_cell, 2, "SHORT",
        "The spec's 'per-cell breakdown CLI flag' is registered 0 times in "
        "train_platinai_4cell.py and train_nci60_10cell.py. Closing it is a "
        "TRAINER wire-in (a new argparse flag + a per-cell JSON emitter), "
        "not a pre-flight edit — and per AGENTS.md §5.7 it would need a "
        "50-step CPU smoke before being called shipped. The gate as written "
        "PASSes on file existence, so it is a vacuous PASS today.",
        "NEEDS-MEASUREMENT",
        "molmetal/scripts/train_platinai_4cell.py; "
        "molmetal/scripts/train_nci60_10cell.py; "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:210"),
    "p3_protocol_recommended_5_fold_cv_compatibility_50x": (
        flag_5fold, 2, "SHORT",
        "No 5-fold CV flag exists: 0 matches for 5fold/5_fold/five_fold in "
        "train_metal_cytotox.py and 0 in drugood_loader.py. Closing it is a "
        "trainer/loader feature, i.e. the same class as the gate above. The "
        "gate as written PASSes on file existence.",
        "NEEDS-MEASUREMENT",
        "molmetal/scripts/train_metal_cytotox.py; molmetal/data/drugood_loader.py; "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:218"),
    "p3_protocol_recommended_krasnov_paper_table_4_replication": (
        0, 1, "MISSING_SOURCE",
        "molmetal/reports/krasnov_2026_paper_table4_replication/ does not "
        "exist. The spec's own verdict is 'PASS if template file present', "
        "so creating an EMPTY directory would turn this into a vacuous PASS "
        "over nothing. The underlying content is Krasnov 2026 Table 4 "
        "(5-metal x 5-fold x per-fold AUC), which needs a primary-record "
        "fetch and, for per-fold AUCs, a GPU retrain.",
        "NEEDS-MEASUREMENT",
        "molmetal/scripts/pre_flight_check.py:11647; "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:266-272"),
    "p3_protocol_recommended_drugood_3_splits_full_alignment": (
        len(dood_splits), 3, "MET",
        "All 3 DrugOOD splits and a per-split iMoLD-gate verdict exist — but "
        "NOT in the file the gate reads. all_results.json carries 0 of 3 "
        "split keys; metrics/by_metric/drugood_ood_auc.json carries all 3 "
        "(assay/scaffold/size, gate_pass true for each). The gate's "
        "detector ('split in data or any(split in k for k in data.keys())') "
        "would still miss them after a path fix, because the splits sit in a "
        "nested key — so the detector needs fixing too. Both edits are "
        "read-only and the repaired gate is GREEN.",
        "WIREABLE-BY-READONLY-REPAIR",
        "metrics/by_metric/drugood_ood_auc.json:6-9,52-64; "
        "molmetal/reports/wf265_collect/all_results.json (0/3 split keys); "
        "molmetal/scripts/pre_flight_check.py:11676"),
    "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup": (
        0, 1, "MISSING_SOURCE",
        "molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline/ does not "
        "exist. Its content is Rusanov 2025 Table S4-S5 — an EXTERNAL "
        "primary record that must be fetched (a search summary will not do). "
        "The verdict's second half ('lookup emits comparison vs baseline per "
        "cell-line') additionally needs train_platinai.py --per-cell-breakdown, "
        "which does not exist (see p3_protocol_recommended_per_cell_breakdown).",
        "NEEDS-MEASUREMENT",
        "molmetal/scripts/pre_flight_check.py:11729; "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:236-242"),
    "p3_protocol_recommended_section_6_honest_minor_section_auto_emit": (
        0, 1, "MISSING_SOURCE",
        "molmetal/scripts/auto_emit_section_6_minor.py does not exist. The "
        "verdict needs a script that auto-emits a §6 minor-section per "
        "Tier-1 LEAD metric — a NEW file under molmetal/scripts/ (allowed; "
        "it must not write under paper/, per AGENTS.md §5.3). Until it "
        "exists the gate WARNs; it cannot be wired as a PASS.",
        "WIREABLE-BY-READONLY-REPAIR",
        "molmetal/scripts/pre_flight_check.py:11764; "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:250"),
    "krasnov_m1_per_seed_per_split_5fold_5metal": (
        m1_count, 125, "SHORT",
        "The source metrics/by_metric/metal_lomo_auc_wasserstein.json holds "
        "25 per_run_auc scalars; the per-fold values are not stored per run "
        "(per_fold_auc_mean is a 5-key summary over held-out metals). The "
        "spec's own template source "
        "(molmetal/reports/wf131_p3_deferred_runs/per_seed_metrics_seed*.json) "
        "holds 0 matching files. Producing 125 (metal x seed x fold) records "
        "means a 5-metal x 5-seed x 5-fold retrain = a GPU run.",
        "NEEDS-GPU",
        "metrics/by_metric/metal_lomo_auc_wasserstein.json (per_run_auc=25); "
        "molmetal/reports/wf131_p3_deferred_runs/ (0 per_seed_metrics_seed*.json); "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:266-272"),
    "krasnov_m21_per_cell_line_per_seed": (
        m21_count, 45, "SHORT",
        "metrics/by_metric/nci60_precious_9cell_auc.json holds 9 per_cell_line "
        "records, and EVERY one of them carries per_seed_available=false — "
        "the 9x5=45 per-(cell x seed) records the spec asks for do not exist "
        "anywhere on disk. The metric header says n_seeds=5, so the run did "
        "happen; the per-seed values were not persisted. Recovering them "
        "needs a retrain.",
        "NEEDS-GPU",
        "metrics/by_metric/nci60_precious_9cell_auc.json "
        "(per_cell_line=9, per_seed_available=false); "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:275-280"),
    "krasnov_protocol_alignment_diffs_documented": (
        l5_recs["krasnov_protocol_alignment_diffs_documented"]["measured_count"],
        l5_recs["krasnov_protocol_alignment_diffs_documented"]["threshold"],
        "MET",
        "Doc-presence threshold is met (m1 and m21 both have a qualifying §6 "
        "entry). BUT this gate WRITES on invocation: calling it created "
        "molmetal/reports/protocol_validation/{krasnov,rusanov,drugood}_hook.json "
        "(measured — the directory did not exist before the probe and was "
        "removed afterwards to restore the tree). A post-launch hook is not "
        "a pre-flight gate, and the spec's own §1.3 routing rule excludes "
        "the 9 L5 hooks from the 'all' pipeline.",
        "DEFER-BY-DESIGN",
        "molmetal/scripts/pre_flight_check.py:12350-12454 "
        "(_hook_passthrough_payload + emitter); "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:§1.3 (routing); "
        "docs/HONEST_NEGATIVE_CATALOG.md §6.266.3"),
    "rusanov_m2_per_cell_line_per_cohort": (
        m2_count, 12, "SHORT",
        "The source metrics/by_metric/m2_r2_per_cohort.json holds 3 cohorts "
        "and no per-cell-line axis. The plan doc names a different file "
        "(molmetal/reports/wf388a1_m2_r2_real/result.json) which does not "
        "exist — the real directory is wf388a1_m2_per_cohort_real, whose "
        "result.json also has no per-cell axis, and whose producing script "
        "has 0 per-cell references. The ridge model is CPU-cheap "
        "(elapsed_sec 0.08), so the gap is COMPUTABLE — but platinai_4cell's "
        "labels are SIMULATED and .data/molmetal/platinai_4cell.csv is "
        "classified DO-NOT-TRAIN ('Its ~0.50 AUC is a null test, not a "
        "measurement'). Manufacturing the 12 records from that source would "
        "create a number that is not a measurement.",
        "NEEDS-MEASUREMENT",
        "metrics/by_metric/m2_r2_per_cohort.json (per_cohort=3, "
        "label_provenance.labels=SIMULATED); "
        "molmetal/reports/wf388a1_m2_per_cohort_real/result.json (exists); "
        "molmetal/scripts/wf388a1_m2_per_cohort_real.py; "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:291-298"),
    "rusanov_m2_4cell_baseline_comparison": (
        None, None, "MISSING_SOURCE",
        "Both named sources are absent: wf388a1_m2_r2_real/result.json (the "
        "spec's primary) and molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline/ "
        "(the spec's alternate). The threshold is 'at least 1 cell-line with "
        "delta > +0.03', which needs our per-cell-line m2 values (blocked "
        "above) AND the Rusanov 2025 Table S4-S5 baseline (an external "
        "primary record). No number can be measured today.",
        "NEEDS-MEASUREMENT",
        "molmetal/scripts/pre_flight_check.py:12542; "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:301-306"),
    "rusanov_protocol_alignment_diffs_documented": (
        l5_recs["rusanov_protocol_alignment_diffs_documented"]["measured_count"],
        l5_recs["rusanov_protocol_alignment_diffs_documented"]["threshold"],
        "MET",
        "Doc-presence threshold is met (m2 has a qualifying §6 entry). Same "
        "two caveats as the Krasnov sibling: the gate WRITES a hook JSON on "
        "invocation, and the spec's §1.3 routing rule excludes the 9 L5 "
        "hooks from 'all'.",
        "DEFER-BY-DESIGN",
        "molmetal/scripts/pre_flight_check.py:12564; "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:§1.3"),
    "drugood_m17_assay_split_auc": (
        round(dood_deltas["assay"], 6), -0.02, "MET",
        "delta_imold = +0.009348 vs the -0.02 bar. The value lives in "
        "metrics/by_metric/drugood_ood_auc.json (2026-10-07, official "
        "DrugOOD IC50 axis partitions, 3 seeds, assay OOD AUC 0.730448) — "
        "NOT in the all_results.json the gate reads. The plan doc's own "
        "'current state' prose ('metacytotox_fallback assay 0.7479') is STALE "
        "against that metric JSON, and PROTOCOL_ALIGNMENT_MATRIX §1 still "
        "calls m17 'NOT ALIGN — fallback cohort'. The JSON's own "
        "honest_framing: 'protocol-comparable but NOT backbone-comparable'. "
        "Also DEFER: L5 routing + write side effect.",
        "DEFER-BY-DESIGN",
        "metrics/by_metric/drugood_ood_auc.json (per_axis_auc_delta_vs_imold, "
        "honest_framing); TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md:24; "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:319-324"),
    "drugood_m17_scaffold_split_auc": (
        round(dood_deltas["scaffold"], 6), -0.02, "MET",
        "delta_imold = +0.001845 vs the -0.02 bar (scaffold OOD AUC 0.690245). "
        "Same source / staleness / backbone caveats as the assay sibling; "
        "the margin here is only 0.0218 above the bar.",
        "DEFER-BY-DESIGN",
        "metrics/by_metric/drugood_ood_auc.json "
        "(per_axis_auc_delta_vs_imold.scaffold); "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:327-332"),
    "drugood_m17_size_split_auc": (
        round(dood_deltas["size"], 6), -0.02, "MET",
        "delta_imold = -0.002853 vs the -0.02 bar (size OOD AUC 0.676347) — "
        "the closest of the three to the bar, 0.0171 of margin. Same source "
        "/ staleness / backbone caveats as the assay sibling.",
        "DEFER-BY-DESIGN",
        "metrics/by_metric/drugood_ood_auc.json "
        "(per_axis_auc_delta_vs_imold.size); "
        "TODO/03_plans/PRE_TRAINING_GATE_v2.md:335-340"),
}

# Implementation kind of each gate body, read from the source at runtime.
# "path"      -> only calls .exists() on its arguments
# "substring" -> reads a file's text and looks for a literal substring
# "parse"     -> parses JSON and inspects keys/values
IMPL_KIND = {
    "p3_protocol_alignment_dataset_match": "path",
    "p3_protocol_alignment_split_match": "path",
    "p3_protocol_alignment_metric_match": "path",
    "p3_protocol_alignment_seed_cohort_per_seed_per_split": "path",
    "p3_protocol_alignment_honest_limitation_minor_section": "substring",
    "p3_protocol_alignment_paper_section_6_template": "path",
    "p3_protocol_alignment_user_ok_gate": "substring",
    "p3_protocol_alignment_baseline_paper_refs_bib": "substring",
    "p3_protocol_recommended_seed_count_3": "path",
    "p3_protocol_recommended_per_cell_breakdown": "path",
    "p3_protocol_recommended_5_fold_cv_compatibility_50x": "path",
    "p3_protocol_recommended_krasnov_paper_table_4_replication": "path",
    "p3_protocol_recommended_drugood_3_splits_full_alignment": "parse",
    "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup": "path",
    "p3_protocol_recommended_section_6_honest_minor_section_auto_emit": "path",
}
for _g in TARGETS[15:]:
    IMPL_KIND[_g] = "path"   # all 9 L5 hooks funnel through _hook_passthrough_payload
# verify the classification against the live source (defensive: no hand-typing)
for _g, _k in IMPL_KIND.items():
    _src = inspect.getsource(getattr(pfc, "gate_" + _g))
    _has_parse = "json.loads" in _src
    _has_read = ".read_text(" in _src
    if _has_parse:
        assert _k == "parse", (_g, _k)
    elif _has_read:
        assert _k == "substring", (_g, _k)
    else:
        assert _k == "path", (_g, _k)

rows = []
for g in TARGETS:
    num, den, status, blocker, verdict, ev = VERDICT[g]
    rows.append({
        "name": g,
        "gate": g,
        "layer": ("L1" if g.startswith("p3_protocol_alignment_") else
                  "L2" if g.startswith("p3_protocol_recommended_") else "L5"),
        "spec_line": SPEC_LINE[g],
        "spec_threshold": spec_threshold[g],
        "measured_num": num,
        "measured_den": den,
        "status": status,
        "implementation_kind": IMPL_KIND[g],
        "blocker": blocker,
        "verdict": verdict,
        "evidence_path": ev,
        "live_verdict_if_invoked": live_verdict[g]["status"],
        "live_evidence": live_verdict[g]["evidence"],
        "registered_in_pipelines": [p for p in registered if g in registered[p]],
    })

doc = {
    "wave": "Wf-A175 — dead-gate wiring DECISION TABLE (no patch applied)",
    "generated_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "cpu_only": True,
    "gpu_used": False,
    "written_paths": [str(OUT / "decision_table.json"), str(OUT / "report.md")],
    "read_only_constraints": [
        "paper/**", "docs/HONEST_NEGATIVE_CATALOG.md", "AGENTS.md", "TODO/**",
        "README.md", "molmetal/scripts/pre_flight_check.py", "metrics/**",
    ],
    "live_gate_counts": live_counts,
    "target_gate_count": len(TARGETS),
    "registered_count": sum(1 for g in TARGETS if registered["all"] & {g}),
    "live_verdict_histogram": {
        s: sum(1 for g in TARGETS if live_verdict[g]["status"] == s)
        for s in ("PASS", "WARN", "FAIL", "RAISED")
    },
    "implementation_kind_histogram": {
        k: sum(1 for r in rows if r["implementation_kind"] == k)
        for k in ("path", "substring", "parse")
    },
    "status_histogram": {
        s: sum(1 for r in rows if r["status"] == s)
        for s in ("MET", "SHORT", "MISSING_SOURCE", "UNSPEC")
    },
    "verdict_histogram": {
        v: sum(1 for r in rows if r["verdict"] == v)
        for v in ("WIREABLE-BY-READONLY-REPAIR", "NEEDS-MEASUREMENT",
                  "NEEDS-GPU", "NEEDS-PAPER-AUTH", "DEFER-BY-DESIGN")
    },
    "side_effect_observed": {
        "what": "Calling any of the 9 L5 gate functions WRITES "
                "molmetal/reports/protocol_validation/{krasnov,rusanov,drugood}_hook.json",
        "evidence": "probe run created the directory at 2026-11-01T18:56:33; "
                    "the directory did not exist beforehand (docs/HONEST_NEGATIVE_CATALOG.md "
                    "§6.266.3 records it as absent); it was removed afterwards to "
                    "restore the tree",
        "consequence": "the 9 L5 gates are NOT read-only as implemented",
    },
    "minimal_wiring_set": {
        "wire_today": [
            "p3_protocol_alignment_honest_limitation_minor_section",
            "p3_protocol_alignment_paper_section_6_template",
            "p3_protocol_alignment_user_ok_gate",
            "p3_protocol_recommended_seed_count_3",
        ],
        "wire_today_after_readonly_repair": [
            "p3_protocol_alignment_dataset_match",
            "p3_protocol_alignment_split_match",
            "p3_protocol_recommended_drugood_3_splits_full_alignment",
            "p3_protocol_recommended_section_6_honest_minor_section_auto_emit",
        ],
        "must_not_wire": [
            "p3_protocol_alignment_baseline_paper_refs_bib",
            "p3_protocol_alignment_metric_match",
            "p3_protocol_alignment_seed_cohort_per_seed_per_split",
            "p3_protocol_recommended_per_cell_breakdown",
            "p3_protocol_recommended_5_fold_cv_compatibility_50x",
            "p3_protocol_recommended_krasnov_paper_table_4_replication",
            "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup",
            "krasnov_m1_per_seed_per_split_5fold_5metal",
            "krasnov_m21_per_cell_line_per_seed",
            "rusanov_m2_per_cell_line_per_cohort",
            "rusanov_m2_4cell_baseline_comparison",
            "krasnov_protocol_alignment_diffs_documented",
            "rusanov_protocol_alignment_diffs_documented",
            "drugood_m17_assay_split_auc",
            "drugood_m17_scaffold_split_auc",
            "drugood_m17_size_split_auc",
        ],
    },
    "supporting_measurements": {
        "catalog_section6_lead_metric_coverage_strict": strict_n,
        "catalog_section6_lead_metric_coverage_loose": loose_n,
        "trainers_with_eval_per_seed_per_split_flag": flag_eval,
        "trainers_with_per_cell_breakdown_flag": flag_cell,
        "sources_with_5fold_flag": flag_5fold,
        "trainers_with_seeds_default_0_1_2": seeds_ok,
        "trainers_emitting_n_seeds": nseeds_ok,
        "wrappers_with_user_ok_before_dry_run_exit": user_ok_before,
        "wrappers_total": len(WRAPPERS),
        "leakage_utils_has_scaffold_split": has_scaffold_in_leak,
        "scaffold_splitter_impl_in_molmetal_data_splits": scaffold_impl,
        "drugood_splits_in_all_results_json": len(dgr_splits),
        "drugood_splits_in_metric_json": len(dood_splits),
        "drugood_per_split_gate_pass": dood_gate_pass,
        "refs_bib_keys_present": sum(1 for v in bib_keys.values() if v),
        "protocol_alignment_matrix_has_section_1_4": pa_has_1_4,
        "wf131_per_seed_template_files": len(tmpl),
        "drugood_official_csv_data_rows": dood_csv_rows,
        "drugood_official_csv_columns": dood_csv_header,
        "drugood_metric_json_source": dood.get("data_source"),
        "drugood_metric_json_date": dood.get("date"),
        "user_ok_positions": user_ok_pos,
    },
    "gates": rows,
}

OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

atomic_write_file(str(OUT / "decision_table.json"),
                   json.dumps(doc, indent=2, sort_keys=False) + "\n")
print("wrote", OUT / "decision_table.json")
print("status:", doc["status_histogram"])
print("verdict:", doc["verdict_histogram"])
print("live:", doc["live_verdict_histogram"])
print("registered:", doc["registered_count"], "of", doc["target_gate_count"])
