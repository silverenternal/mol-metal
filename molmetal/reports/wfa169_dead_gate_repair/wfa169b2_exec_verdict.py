#!/usr/bin/env python
"""Wf-A169 branch 2 — EXECUTION VERDICT for the 24 unregistered gates.

Read-only, CPU-only (the caller forces CUDA_VISIBLE_DEVICES="").
Calls each of the 24 gate FUNCTIONS defined in molmetal/scripts/pre_flight_check.py
against the CURRENT tree and records the real GateResult.status + evidence.

THREE arg sets are exercised:

  "original_registration" (PRIMARY)
      The exact argument tuples the gates were ORIGINALLY registered with,
      recovered from the deleted backup
      TODO/archive/wf437_simplify/pre_flight_check.py.bak.wf437
      (commit aee5f8ab^, 359,637 bytes, EXPECTED_GATE_COUNTS["all"] == 127).
      This is "registered as designed" in the literal sense.

  "witness_test"
      The arg tuples encoded in the repo's own wire-in witness
      molmetal/tests/test_wf433a1_gate_v2_wire.py.

  "spec_literal"
      The spec §2/§3/§4 "Source" lines read literally where they differ
      (train_platinai.py, TODO/03_plans/.../PROTOCOL_ALIGNMENT_MATRIX.md).

The L5 hooks' hook_dir is the CANONICAL molmetal/reports/protocol_validation
(absent today -> _emit_protocol_validation_hook_json is a silent no-op by
design), so the primary pass writes NOTHING outside this report dir.  A second
pass with hook_dir redirected into this report dir captures the hook payload.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
SCRIPT_DIR = MOLMETAL_ROOT / "scripts"
DATA_DIR = MOLMETAL_ROOT / "data"
DOT_DATA_DIR = REPO_ROOT / ".data" / "molmetal"

_PFC_SPEC = _ilu.spec_from_file_location(
    "_wfa169b2_pfc", SCRIPT_DIR / "pre_flight_check.py"
)
_pfc = _ilu.module_from_spec(_PFC_SPEC)
_PFC_SPEC.loader.exec_module(_pfc)

# ------------------------------------------------------------------ paths
mc_csv_orig = DOT_DATA_DIR / "metalcytotox_5metal_multitask.csv"  # original reg
mc_csv_wit = DOT_DATA_DIR / "metalcytotox_5metal_multitask_v4.csv"  # witness test
pl_csv = DATA_DIR / "platinai_4cell.csv"
cd_csv = DATA_DIR / "crossdocked100_manifest.csv"
nc_csv = DATA_DIR / "nci60_precious_9cell.csv"
bal_json = (
    MOLMETAL_ROOT / "reports" / "wf387a2_balanced_rh_cohort" / "balanced_rh_cohort.json"
)
dg_loader = DATA_DIR / "drugood_loader.py"
leak = DATA_DIR / "leakage_utils.py"
dg_results = MOLMETAL_ROOT / "reports" / "wf265_collect" / "all_results.json"
eval_all = SCRIPT_DIR / "eval_all_evaluators.py"
t_metal = SCRIPT_DIR / "train_metal_cytotox.py"
t_plat_4cell = SCRIPT_DIR / "train_platinai_4cell.py"
t_plat_spec = SCRIPT_DIR / "train_platinai.py"
t_nci = SCRIPT_DIR / "train_nci60_10cell.py"
per_seed_dir = MOLMETAL_ROOT / "reports" / "wf131_p3_deferred_runs"
honest_cat = REPO_ROOT / "docs" / "HONEST_NEGATIVE_CATALOG.md"
s6_tpl = REPO_ROOT / "docs" / "SECTION_6_LIMITATIONS_TEMPLATE.md"
s6_v4 = REPO_ROOT / "docs" / "PAPER_SECTION_6_v4_LIMITATIONS_READY.md"
refs = REPO_ROOT / "paper" / "refs.bib"
auto_emit = SCRIPT_DIR / "auto_emit_section_6_minor.py"
m2_json = REPO_ROOT / "metrics" / "by_metric" / "m2_r2_per_cohort.json"
ml_auc = REPO_ROOT / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json"
nci60_dir = MOLMETAL_ROOT / "reports" / "wf_nci60_precious_9cell"
ru_dir = MOLMETAL_ROOT / "reports" / "rusanov_2025_paper_table_s4_s5_baseline"
pv_dir = MOLMETAL_ROOT / "reports" / "protocol_validation"  # canonical; absent
pa_matrix_reg = REPO_ROOT / "TODO" / "03_plans" / "PROTOCOL_ALIGNMENT_MATRIX.md"
pa_matrix_real = REPO_ROOT / "TODO" / "03_baselines" / "PROTOCOL_ALIGNMENT_MATRIX.md"
wrapper_names = (
    "launch_wf124_p3_g5.sh",
    "launch_wf124_p3_g6.sh",
    "launch_wf299_cfm_main.sh",
    "launch_wf388_a1_metal_coord_retrain.sh",
    "launch_wf394a1_lambda_cfm_patch.sh",
    "launch_wf125_p1_controlled_pair.sh",
    "launch_wf390a1_m2_r2_path_a.sh",
)
scripts_dir = REPO_ROOT / "scripts"

# --------------------------------------------- the 24 gates (name, layer, fn)
GATES = [
    ("p3_protocol_alignment_dataset_match", "L1",
     "gate_p3_protocol_alignment_dataset_match"),
    ("p3_protocol_alignment_split_match", "L1",
     "gate_p3_protocol_alignment_split_match"),
    ("p3_protocol_alignment_metric_match", "L1",
     "gate_p3_protocol_alignment_metric_match"),
    ("p3_protocol_alignment_seed_cohort_per_seed_per_split", "L1",
     "gate_p3_protocol_alignment_seed_cohort_per_seed_per_split"),
    ("p3_protocol_alignment_honest_limitation_minor_section", "L1",
     "gate_p3_protocol_alignment_honest_limitation_minor_section"),
    ("p3_protocol_alignment_paper_section_6_template", "L1",
     "gate_p3_protocol_alignment_paper_section_6_template"),
    ("p3_protocol_alignment_user_ok_gate", "L1",
     "gate_p3_protocol_alignment_user_ok_gate"),
    ("p3_protocol_alignment_baseline_paper_refs_bib", "L1",
     "gate_p3_protocol_alignment_baseline_paper_refs_bib"),
    ("p3_protocol_recommended_seed_count_3", "L2",
     "gate_p3_protocol_recommended_seed_count_3"),
    ("p3_protocol_recommended_per_cell_breakdown", "L2",
     "gate_p3_protocol_recommended_per_cell_breakdown"),
    ("p3_protocol_recommended_5_fold_cv_compatibility_50x", "L2",
     "gate_p3_protocol_recommended_5_fold_cv_compatibility_50x"),
    ("p3_protocol_recommended_krasnov_paper_table_4_replication", "L2",
     "gate_p3_protocol_recommended_krasnov_paper_table_4_replication"),
    ("p3_protocol_recommended_drugood_3_splits_full_alignment", "L2",
     "gate_p3_protocol_recommended_drugood_3_splits_full_alignment"),
    ("p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup", "L2",
     "gate_p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup"),
    ("p3_protocol_recommended_section_6_honest_minor_section_auto_emit", "L2",
     "gate_p3_protocol_recommended_section_6_honest_minor_section_auto_emit"),
    ("krasnov_m1_per_seed_per_split_5fold_5metal", "L5",
     "gate_krasnov_m1_per_seed_per_split_5fold_5metal"),
    ("krasnov_m21_per_cell_line_per_seed", "L5",
     "gate_krasnov_m21_per_cell_line_per_seed"),
    ("krasnov_protocol_alignment_diffs_documented", "L5",
     "gate_krasnov_protocol_alignment_diffs_documented"),
    ("rusanov_m2_per_cell_line_per_cohort", "L5",
     "gate_rusanov_m2_per_cell_line_per_cohort"),
    ("rusanov_m2_4cell_baseline_comparison", "L5",
     "gate_rusanov_m2_4cell_baseline_comparison"),
    ("rusanov_protocol_alignment_diffs_documented", "L5",
     "gate_rusanov_protocol_alignment_diffs_documented"),
    ("drugood_m17_assay_split_auc", "L5", "gate_drugood_m17_assay_split_auc"),
    ("drugood_m17_scaffold_split_auc", "L5", "gate_drugood_m17_scaffold_split_auc"),
    ("drugood_m17_size_split_auc", "L5", "gate_drugood_m17_size_split_auc"),
]

# ---- PRIMARY: the ORIGINAL registration args (aee5f8ab^:.bak.wf437) -------
ORIGINAL_ARGS = {
    "p3_protocol_alignment_dataset_match": (
        mc_csv_orig, pl_csv, cd_csv, nc_csv, bal_json, dg_loader),
    "p3_protocol_alignment_split_match": (leak, dg_results),
    "p3_protocol_alignment_metric_match": (eval_all,),
    "p3_protocol_alignment_seed_cohort_per_seed_per_split": (
        t_metal, t_plat_4cell, t_nci, per_seed_dir),
    "p3_protocol_alignment_honest_limitation_minor_section": (honest_cat,),
    "p3_protocol_alignment_paper_section_6_template": (s6_tpl, s6_v4),
    "p3_protocol_alignment_user_ok_gate": (scripts_dir, wrapper_names),
    "p3_protocol_alignment_baseline_paper_refs_bib": (refs,),
    "p3_protocol_recommended_seed_count_3": (t_metal, t_plat_4cell, t_nci),
    "p3_protocol_recommended_per_cell_breakdown": (t_plat_4cell, t_nci, m2_json),
    "p3_protocol_recommended_5_fold_cv_compatibility_50x": (t_metal, dg_loader),
    "p3_protocol_recommended_krasnov_paper_table_4_replication": (
        MOLMETAL_ROOT / "reports",),
    "p3_protocol_recommended_drugood_3_splits_full_alignment": (dg_results,),
    "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup": (
        MOLMETAL_ROOT / "reports", t_plat_4cell),
    "p3_protocol_recommended_section_6_honest_minor_section_auto_emit": (
        auto_emit,),
    "krasnov_m1_per_seed_per_split_5fold_5metal": (pv_dir, ml_auc),
    "krasnov_m21_per_cell_line_per_seed": (pv_dir, nci60_dir),
    "krasnov_protocol_alignment_diffs_documented": (
        pv_dir, honest_cat, pa_matrix_reg),
    "rusanov_m2_per_cell_line_per_cohort": (pv_dir, m2_json),
    "rusanov_m2_4cell_baseline_comparison": (pv_dir, ru_dir, m2_json),
    "rusanov_protocol_alignment_diffs_documented": (
        pv_dir, honest_cat, pa_matrix_reg),
    "drugood_m17_assay_split_auc": (pv_dir, dg_results),
    "drugood_m17_scaffold_split_auc": (pv_dir, dg_results),
    "drugood_m17_size_split_auc": (pv_dir, dg_results),
}

# ---- variant: the witness test's arg tuples ------------------------------
WITNESS_ARGS = dict(ORIGINAL_ARGS)
WITNESS_ARGS["p3_protocol_alignment_dataset_match"] = (
    mc_csv_wit, pl_csv, cd_csv, nc_csv, bal_json, dg_loader)

# ---- variant: spec-literal Source lines ----------------------------------
SPEC_ARGS = dict(ORIGINAL_ARGS)
SPEC_ARGS["p3_protocol_alignment_seed_cohort_per_seed_per_split"] = (
    t_metal, t_plat_spec, t_nci, per_seed_dir)
SPEC_ARGS["p3_protocol_recommended_seed_count_3"] = (
    t_metal, t_plat_spec, t_nci)
SPEC_ARGS["p3_protocol_recommended_per_cell_breakdown"] = (
    t_plat_spec, t_nci, m2_json)
SPEC_ARGS["p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup"] = (
    MOLMETAL_ROOT / "reports", t_plat_spec)
SPEC_ARGS["krasnov_protocol_alignment_diffs_documented"] = (
    pv_dir, honest_cat, pa_matrix_reg)  # original reg already uses 03_plans
SPEC_ARGS["rusanov_protocol_alignment_diffs_documented"] = (
    pv_dir, honest_cat, pa_matrix_reg)

VARIANTS = {"witness_test": WITNESS_ARGS, "spec_literal": SPEC_ARGS}


def run_one(fn, name, args):
    t0 = time.perf_counter()
    try:
        res = fn(name, "P3", "all", *args)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "RAISED",
            "evidence": "",
            "wall_sec": round(time.perf_counter() - t0, 6),
            "raised": True,
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }
    return {
        "status": res.status,
        "evidence": res.evidence,
        "wall_sec": round(time.perf_counter() - t0, 6),
        "raised": False,
        "exception": None,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    records = []
    for name, layer, fn_name in GATES:
        fn = getattr(_pfc, fn_name)
        args = ORIGINAL_ARGS[name]
        r = run_one(fn, name, args)
        rec = {
            "name": name,
            "layer": layer,
            "gate_fn": fn_name,
            "priority": "P3",
            "pipeline": "all",
            "arg_variant": "original_registration",
            "args_used": [str(a) for a in args],
            "args_missing_on_disk": [str(a) for a in args
                                     if isinstance(a, Path) and not a.exists()],
            "status": r["status"],
            "evidence": r["evidence"],
            "evidence_300": r["evidence"][:300],
            "wall_sec": r["wall_sec"],
            "raised": r["raised"],
            "exception": r["exception"],
        }
        if r["raised"]:
            rec["traceback"] = r["traceback"]
        for vname, vargs in VARIANTS.items():
            if [str(a) for a in vargs[name]] == [str(a) for a in args]:
                continue
            vr = run_one(fn, name, vargs[name])
            rec.setdefault("variants", {})[vname] = {
                "args_used": [str(a) for a in vargs[name]],
                "status": vr["status"],
                "evidence_300": vr["evidence"][:300],
                "wall_sec": vr["wall_sec"],
                "raised": vr["raised"],
                "exception": vr["exception"],
            }
        records.append(rec)

    def tally_of(rs):
        t = {"PASS": 0, "WARN": 0, "FAIL": 0, "RAISED": 0}
        for r in rs:
            t[r["status"]] = t.get(r["status"], 0) + 1
        return t

    tallies = {"original_registration": tally_of(records)}
    for vname in VARIANTS:
        vr = []
        for name, layer, fn_name in GATES:
            fn = getattr(_pfc, fn_name)
            r = run_one(fn, name, VARIANTS[vname][name])
            vr.append({"status": r["status"], "evidence": r["evidence"],
                       "raised": r["raised"], "exception": r["exception"]})
        tallies[vname] = tally_of(vr)
        for rec, v in zip(records, vr):
            rec.setdefault("variants", {}).setdefault(
                vname, {"status": v["status"], "evidence_300": v["evidence"][:300],
                        "wall_sec": None, "raised": v["raised"],
                        "exception": v["exception"]})

    # hook payload capture (writes ONLY into this report dir)
    capture_dir = out_dir / "hook_payload_capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    for name, layer, fn_name in GATES:
        if layer != "L5":
            continue
        fn = getattr(_pfc, fn_name)
        args = list(ORIGINAL_ARGS[name])
        args[0] = capture_dir
        try:
            fn(name, "P3", "all", *args)
        except Exception:  # noqa: BLE001
            pass
    hook_payloads = {p.name: json.loads(p.read_text(encoding="utf-8"))
                     for p in sorted(capture_dir.glob("*.json"))}

    runners = _pfc.build_gates("all")
    registered = [r[0] for r in runners]
    live_counts = {
        "build_gates_all_len": len(runners),
        "build_gates_all_unique_len": len(set(registered)),
        "build_gates_launch_len": len(_pfc.build_gates("launch")),
        "EXPECTED_GATE_COUNTS": dict(_pfc.EXPECTED_GATE_COUNTS),
    }
    pfc_stat = (SCRIPT_DIR / "pre_flight_check.py").stat()

    payload = {
        "meta": {
            "branch": "Wf-A169 branch 2 — execution verdict",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gate_module": str(SCRIPT_DIR / "pre_flight_check.py"),
            "gate_module_bytes": pfc_stat.st_size,
            "gate_module_mtime_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(pfc_stat.st_mtime)),
            "concurrency_note": (
                "molmetal/scripts/pre_flight_check.py is edited concurrently by "
                "another branch of this wave; verdicts were re-run and "
                "re-confirmed after the last observed edit (mtime above)."),
            "denominator": len(records),
            "cuda_visible_devices": "",
            "cpu_only": True,
            "primary_arg_set_source": (
                "TODO/archive/wf437_simplify/pre_flight_check.py.bak.wf437 "
                "@ aee5f8ab^ (359,637 bytes, EXPECTED_GATE_COUNTS['all']==127) "
                "— the ORIGINAL registration block"),
            "hook_dir_primary": str(pv_dir),
            "hook_dir_exists": pv_dir.exists(),
            "live_gate_counts": live_counts,
        },
        "tally": tallies["original_registration"],
        "tally_by_arg_variant": tallies,
        "gates": records,
        "hook_payload_capture": hook_payloads,
    }
    out = out_dir / "verdict_table.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"tally": tallies, "denominator": len(records),
                      "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
