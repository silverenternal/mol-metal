#!/usr/bin/env python3
"""Wf-A175 refutation branch — INDEPENDENT live probe of the 24 dead gates.

CPU-only.  Writes ONLY inside this directory.

DEVIATION FROM THE CANONICAL ARGS (documented in refutation.md):
  the canonical arg set (molmetal/tests/test_wf433a1_gate_v2_wire.py:247) passes
  `pv_dir = MOLMETAL_ROOT/"reports"/"protocol_validation"` to the 9 L5 gates,
  and the L5 emitter CREATES that directory.  This probe redirects `pv_dir` to
  `<this dir>/_pv_hook_dir` so that no file is written outside the refutation
  directory.  The L5 verdict depends only on (a) source_present (unchanged) and
  (b) whether mkdir+atomic_write succeeded (same outcome in either directory),
  so the verdict vector is unaffected by the redirect.
"""

from __future__ import annotations

import importlib.util as ilu
import json
import sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
HERE = Path(__file__).resolve().parent
PV = HERE / "_pv_hook_dir"

MOLMETAL_ROOT = REPO / "molmetal"
SCRIPT_DIR = MOLMETAL_ROOT / "scripts"


def _load(name: str, rel: str):
    s = ilu.spec_from_file_location(name, REPO / rel)
    m = ilu.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


pfc = _load("pfc", "molmetal/scripts/pre_flight_check.py")

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

mc_csv = REPO / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
pl_csv = MOLMETAL_ROOT / "data" / "platinai_4cell.csv"
cd_csv = MOLMETAL_ROOT / "data" / "crossdocked100_manifest.csv"
nc_csv = MOLMETAL_ROOT / "data" / "nci60_precious_9cell.csv"
bal_json = MOLMETAL_ROOT / "reports" / "wf387a2_balanced_rh_cohort" / "balanced_rh_cohort.json"
dg_loader = MOLMETAL_ROOT / "data" / "drugood_loader.py"
leak = MOLMETAL_ROOT / "data" / "leakage_utils.py"
dg_results = MOLMETAL_ROOT / "reports" / "wf265_collect" / "all_results.json"
eval_all = SCRIPT_DIR / "eval_all_evaluators.py"
t_metal = SCRIPT_DIR / "train_metal_cytotox.py"
t_plat = SCRIPT_DIR / "train_platinai_4cell.py"
t_nci = SCRIPT_DIR / "train_nci60_10cell.py"
per_seed_dir = MOLMETAL_ROOT / "reports" / "wf131_p3_deferred_runs"
honest_cat = REPO / "docs" / "HONEST_NEGATIVE_CATALOG.md"
s6_tpl = REPO / "docs" / "SECTION_6_LIMITATIONS_TEMPLATE.md"
s6_v4 = REPO / "docs" / "PAPER_SECTION_6_v4_LIMITATIONS_READY.md"
s3_v4 = REPO / "docs" / "PAPER_SECTION_3_METRIC_TABLE_FINAL.md"
m6_drop = REPO / "docs" / "M6_KAPPA_DROP_DECISION.md"
pa_matrix = REPO / "TODO" / "03_plans" / "PROTOCOL_ALIGNMENT_MATRIX.md"
refs = REPO / "paper" / "refs.bib"
auto_emit = SCRIPT_DIR / "auto_emit_section_6_minor.py"
m2_json = REPO / "metrics" / "by_metric" / "m2_r2_per_cohort.json"
ml_auc = REPO / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json"
nci60_dir = MOLMETAL_ROOT / "reports" / "wf_nci60_precious_9cell"
ru_dir = MOLMETAL_ROOT / "reports" / "rusanov_2025_paper_table_s4_s5_baseline"
wrapper_names = (
    "launch_wf124_p3_g5.sh",
    "launch_wf124_p3_g6.sh",
    "launch_wf299_cfm_main.sh",
    "launch_wf388_a1_metal_coord_retrain.sh",
    "launch_wf394a1_lambda_cfm_patch.sh",
    "launch_wf125_p1_controlled_pair.sh",
    "launch_wf390a1_m2_r2_path_a.sh",
)
scripts_dir = REPO / "scripts"

arg_sets = {
    "p3_protocol_alignment_dataset_match": (mc_csv, pl_csv, cd_csv, nc_csv, bal_json, dg_loader),
    "p3_protocol_alignment_split_match": (leak, dg_results),
    "p3_protocol_alignment_metric_match": (eval_all,),
    "p3_protocol_alignment_seed_cohort_per_seed_per_split": (t_metal, t_plat, t_nci, per_seed_dir),
    "p3_protocol_alignment_honest_limitation_minor_section": (honest_cat,),
    "p3_protocol_alignment_paper_section_6_template": (s6_tpl, s6_v4),
    "p3_protocol_alignment_user_ok_gate": (scripts_dir, wrapper_names),
    "p3_protocol_alignment_baseline_paper_refs_bib": (refs,),
    "p3_protocol_recommended_seed_count_3": (t_metal, t_plat, t_nci),
    "p3_protocol_recommended_per_cell_breakdown": (t_plat, t_nci, m2_json),
    "p3_protocol_recommended_5_fold_cv_compatibility_50x": (t_metal, dg_loader),
    "p3_protocol_recommended_krasnov_paper_table_4_replication": (MOLMETAL_ROOT / "reports",),
    "p3_protocol_recommended_drugood_3_splits_full_alignment": (dg_results,),
    "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup": (MOLMETAL_ROOT / "reports", t_plat),
    "p3_protocol_recommended_section_6_honest_minor_section_auto_emit": (auto_emit,),
    "krasnov_m1_per_seed_per_split_5fold_5metal": (PV, ml_auc),
    "krasnov_m21_per_cell_line_per_seed": (PV, nci60_dir),
    "krasnov_protocol_alignment_diffs_documented": (PV, honest_cat, pa_matrix),
    "rusanov_m2_per_cell_line_per_cohort": (PV, m2_json),
    "rusanov_m2_4cell_baseline_comparison": (PV, ru_dir, m2_json),
    "rusanov_protocol_alignment_diffs_documented": (PV, honest_cat, pa_matrix),
    "drugood_m17_assay_split_auc": (PV, dg_results),
    "drugood_m17_scaffold_split_auc": (PV, dg_results),
    "drugood_m17_size_split_auc": (PV, dg_results),
}

records = []
hist = {"PASS": 0, "WARN": 0, "FAIL": 0, "RAISED": 0}
for name in TARGETS:
    fn = getattr(pfc, "gate_" + name)
    try:
        res = fn(name, "P3", "all", *arg_sets[name])
        status = res.status
        ev = res.evidence
    except Exception as exc:  # noqa: BLE001
        status = "RAISED"
        ev = f"{type(exc).__name__}: {exc}"
    hist[status] = hist.get(status, 0) + 1
    records.append({"name": name, "status": status, "evidence": ev})

out = {
    "probe": "wfa175 refutation — independent live probe of the 24 dead gates",
    "cpu_only": True,
    "gpu_used": False,
    "pv_dir_redirected_to": str(PV),
    "histogram": hist,
    "denominator": len(records),
    "records": records,
}
(HERE / "probe24.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(hist, sort_keys=True))
for r in records:
    print(f"{r['status']:7s} {r['name']:60s} {r['evidence'][:110]}")
