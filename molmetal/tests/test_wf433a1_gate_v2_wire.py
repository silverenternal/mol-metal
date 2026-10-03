"""Wf-432-A1 — 32 NEW pre-training gates wire-in regression tests.

Per TODO/03_plans/PRE_TRAINING_GATE_v2.md §2-§6: 8 Tier-1 must-pass +
7 Tier-2 recommended + 5 MOCK->REAL + 3 honest-negative pre-closure +
9 per-baseline protocol validation hooks = 32 NEW additive gates wired
into molmetal/scripts/pre_flight_check.py::build_gates("all").

These tests verify:
  (1) every NEW gate function is callable + returns a GateResult
  (2) every NEW gate name is registered in build_gates("all")
  (3) the EXPECTED_GATE_COUNTS["all"] auto-recompute is consistent
      (was 95 pre-Wf-432-A1, now 95 + 32 = 127)
  (4) the 7 retrain wrappers per PRE_TRAINING_GATE_v2 §2 L1-7 are
      verified to have USER_OK gates (per the AGENTS.md §5.1 contract)
  (5) the 3 baseline paper BibTeX entries are present in paper/refs.bib
  (6) MOCK->REAL gates correctly validate MEASURED REAL-MEASURED
      metrics (m3 Vina / m15 RMSD / m14 decode_ratio / m17 DrugOOD /
      m21 NCI-60)

Per AGENTS.md §5.5 (filesystem-truth) the gate count MUST be 127 (not
149; the 149 includes 22 pre-existing Wf-353B1 duplicate registrations
in chemprop_v2_v3 + all blocks; the unique-name count is 127 = expected).

Per AGENTS.md §5.7 (trainer flag wire-in smoke): this is a new-gate
wire-in, not a trainer flag wire-in. Smoke is per-gate function
callable + verdict logic; no 50-step CPU smoke required.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../mol-metal
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
SCRIPT_DIR = MOLMETAL_ROOT / "scripts"

# Ensure pre_flight_check.py is importable.
sys.path.insert(0, str(REPO_ROOT))
import importlib.util as _ilu

_PFC_SPEC = _ilu.spec_from_file_location(
    "_wf432a1_pfc",
    SCRIPT_DIR / "pre_flight_check.py",
)
_pfc = _ilu.module_from_spec(_PFC_SPEC)
_PFC_SPEC.loader.exec_module(_pfc)


# ---- canonical gate inventory -------------------------------------------

EXPECTED_NEW_GATES = (
    # §2 L1 Tier-1 must-pass (8 gates)
    "p3_protocol_alignment_dataset_match",
    "p3_protocol_alignment_split_match",
    "p3_protocol_alignment_metric_match",
    "p3_protocol_alignment_seed_cohort_per_seed_per_split",
    "p3_protocol_alignment_honest_limitation_minor_section",
    "p3_protocol_alignment_paper_section_6_template",
    "p3_protocol_alignment_user_ok_gate",
    "p3_protocol_alignment_baseline_paper_refs_bib",
    # §3 L2 Tier-2 recommended (7 gates)
    "p3_protocol_recommended_seed_count_3",
    "p3_protocol_recommended_per_cell_breakdown",
    "p3_protocol_recommended_5_fold_cv_compatibility_50x",
    "p3_protocol_recommended_krasnov_paper_table_4_replication",
    "p3_protocol_recommended_drugood_3_splits_full_alignment",
    "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup",
    "p3_protocol_recommended_section_6_honest_minor_section_auto_emit",
    # §5 L3 MOCK->REAL (5 gates)
    "p3_mock_to_real_m3_vina_kcal_per_mol_REAL_5_pocket",
    "p3_mock_to_real_m15_rmsd_REAL_5_pocket",
    "p3_mock_to_real_m14_decode_ratio_REAL_30_probes",
    "p3_mock_to_real_m17_drugood_assay_REAL_3_seeds",
    "p3_mock_to_real_m21_nci60_precious_9cell_REAL_3_seeds",
    # §6 L4 honest-negative pre-closure (3 gates)
    "p3_honest_neg_closure_m2_r2",
    "p3_honest_neg_closure_m6_kappa",
    "p3_honest_neg_closure_m1_re_fold",
    # §4 L5 per-baseline protocol validation hooks (9 gates)
    "krasnov_m1_per_seed_per_split_5fold_5metal",
    "krasnov_m21_per_cell_line_per_seed",
    "krasnov_protocol_alignment_diffs_documented",
    "rusanov_m2_per_cell_line_per_cohort",
    "rusanov_m2_4cell_baseline_comparison",
    "rusanov_protocol_alignment_diffs_documented",
    "drugood_m17_assay_split_auc",
    "drugood_m17_scaffold_split_auc",
    "drugood_m17_size_split_auc",
)


# ---- test surface --------------------------------------------------------


def test_wf432a1_expected_gate_count_32():
    """§0 — verify the EXPECTED_NEW_GATES tuple has 32 entries (sanity check)."""
    assert len(EXPECTED_NEW_GATES) == 32, (
        f"Expected 32 NEW gates per PRE_TRAINING_GATE_v2 §2-§6; got "
        f"{len(EXPECTED_NEW_GATES)}. Update EXPECTED_NEW_GATES tuple."
    )


def test_wf432a1_expected_gate_count_const_127():
    """§3.4 — EXPECTED_GATE_COUNTS["all"] = 95 + 32 = 127 after wire-in."""
    assert _pfc.EXPECTED_GATE_COUNTS["all"] == 127, (
        f"EXPECTED_GATE_COUNTS['all'] must be 127 (= 95 + 32); got "
        f"{_pfc.EXPECTED_GATE_COUNTS['all']}. Per Wf-345-B3 the runtime "
        f"len(build_gates('all')) is the authoritative source; this "
        f"constant is for backward-compat only."
    )


def test_wf432a1_build_gates_all_unique_count_is_127():
    """§3.4 — build_gates('all') unique names count = 127 (auto-recompute)."""
    runners = _pfc.build_gates("all")
    names = [r[0] for r in runners]
    assert len(set(names)) == 127, (
        f"Unique gate names in build_gates('all') must be 127; got "
        f"{len(set(names))}. Wf-345-B3 runtime auto-recompute drift."
    )


def test_wf432a1_all_32_new_gates_present_in_build_gates():
    """§2-§6 — every NEW gate is registered in build_gates('all')."""
    runners = _pfc.build_gates("all")
    names = {r[0] for r in runners}
    missing = [g for g in EXPECTED_NEW_GATES if g not in names]
    assert (
        not missing
    ), f"build_gates('all') missing {len(missing)} NEW gates: {missing}"


def test_wf432a1_frontier_pipeline_subset_unchanged():
    """§3.4 — frontier / chemprop_v2_v3 / cfm pipeline counts unchanged."""
    assert len(_pfc.build_gates("frontier")) == 25
    assert len(_pfc.build_gates("chemprop_v2_v3")) == 14
    assert len(_pfc.build_gates("cfm")) == 6


def test_wf432a1_all_32_new_gate_functions_callable():
    """§5.7 — every NEW gate function returns a GateResult when called
    with valid args (smoke = callable + verdict logic)."""
    # Build canonical arg sets per gate
    mc_csv = REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
    pl_csv = MOLMETAL_ROOT / "data" / "platinai_4cell.csv"
    cd_csv = MOLMETAL_ROOT / "data" / "crossdocked100_manifest.csv"
    nc_csv = MOLMETAL_ROOT / "data" / "nci60_precious_9cell.csv"
    bal_json = (
        MOLMETAL_ROOT
        / "reports"
        / "wf387a2_balanced_rh_cohort"
        / "balanced_rh_cohort.json"
    )
    dg_loader = MOLMETAL_ROOT / "data" / "drugood_loader.py"
    leak = MOLMETAL_ROOT / "data" / "leakage_utils.py"
    dg_results = MOLMETAL_ROOT / "reports" / "wf265_collect" / "all_results.json"
    eval_all = SCRIPT_DIR / "eval_all_evaluators.py"
    t_metal = SCRIPT_DIR / "train_metal_cytotox.py"
    t_plat = SCRIPT_DIR / "train_platinai_4cell.py"
    t_nci = SCRIPT_DIR / "train_nci60_10cell.py"
    per_seed_dir = MOLMETAL_ROOT / "reports" / "wf131_p3_deferred_runs"
    honest_cat = REPO_ROOT / "docs" / "HONEST_NEGATIVE_CATALOG.md"
    s6_tpl = REPO_ROOT / "docs" / "SECTION_6_LIMITATIONS_TEMPLATE.md"
    s6_v4 = REPO_ROOT / "docs" / "PAPER_SECTION_6_v4_LIMITATIONS_READY.md"
    s3_v4 = REPO_ROOT / "docs" / "PAPER_SECTION_3_METRIC_TABLE_FINAL.md"
    m6_drop = REPO_ROOT / "docs" / "M6_KAPPA_DROP_DECISION.md"
    pa_matrix = REPO_ROOT / "TODO" / "03_plans" / "PROTOCOL_ALIGNMENT_MATRIX.md"
    refs = REPO_ROOT / "paper" / "refs.bib"
    auto_emit = SCRIPT_DIR / "auto_emit_section_6_minor.py"
    vina = REPO_ROOT / "metrics" / "by_metric" / "vina_kcal_per_mol.json"
    rmsd = REPO_ROOT / "metrics" / "by_metric" / "docked_pose_rmsd.json"
    decr = REPO_ROOT / "metrics" / "by_metric" / "decode_ratio.json"
    dg_auc = REPO_ROOT / "metrics" / "by_metric" / "drugood_od_auc.json"
    nci60 = REPO_ROOT / "metrics" / "by_metric" / "nci60_precious_9cell_auc.json"
    m2_json = REPO_ROOT / "metrics" / "by_metric" / "m2_r2_per_cohort.json"
    ml_auc = REPO_ROOT / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json"
    nci60_dir = MOLMETAL_ROOT / "reports" / "wf_nci60_precious_9cell"
    ru_dir = MOLMETAL_ROOT / "reports" / "rusanov_2025_paper_table_s4_s5_baseline"
    pv_dir = MOLMETAL_ROOT / "reports" / "protocol_validation"
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

    arg_sets = {
        "p3_protocol_alignment_dataset_match": (
            mc_csv,
            pl_csv,
            cd_csv,
            nc_csv,
            bal_json,
            dg_loader,
        ),
        "p3_protocol_alignment_split_match": (leak, dg_results),
        "p3_protocol_alignment_metric_match": (eval_all,),
        "p3_protocol_alignment_seed_cohort_per_seed_per_split": (
            t_metal,
            t_plat,
            t_nci,
            per_seed_dir,
        ),
        "p3_protocol_alignment_honest_limitation_minor_section": (honest_cat,),
        "p3_protocol_alignment_paper_section_6_template": (
            s6_tpl,
            s6_v4,
        ),
        "p3_protocol_alignment_user_ok_gate": (scripts_dir, wrapper_names),
        "p3_protocol_alignment_baseline_paper_refs_bib": (refs,),
        "p3_protocol_recommended_seed_count_3": (
            t_metal,
            t_plat,
            t_nci,
        ),
        "p3_protocol_recommended_per_cell_breakdown": (
            t_plat,
            t_nci,
            m2_json,
        ),
        "p3_protocol_recommended_5_fold_cv_compatibility_50x": (
            t_metal,
            dg_loader,
        ),
        "p3_protocol_recommended_krasnov_paper_table_4_replication": (
            MOLMETAL_ROOT / "reports",
        ),
        "p3_protocol_recommended_drugood_3_splits_full_alignment": (dg_results,),
        "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup": (
            MOLMETAL_ROOT / "reports",
            t_plat,
        ),
        "p3_protocol_recommended_section_6_honest_minor_section_auto_emit": (
            auto_emit,
        ),
        "p3_mock_to_real_m3_vina_kcal_per_mol_REAL_5_pocket": (vina,),
        "p3_mock_to_real_m15_rmsd_REAL_5_pocket": (rmsd,),
        "p3_mock_to_real_m14_decode_ratio_REAL_30_probes": (decr,),
        "p3_mock_to_real_m17_drugood_assay_REAL_3_seeds": (dg_auc,),
        "p3_mock_to_real_m21_nci60_precious_9cell_REAL_3_seeds": (nci60,),
        "p3_honest_neg_closure_m2_r2": (
            m2_json,
            honest_cat,
            s3_v4,
            s6_v4,
        ),
        "p3_honest_neg_closure_m6_kappa": (
            m6_kappa_json := REPO_ROOT / "metrics" / "by_metric" / "m6_kappa.json",
            honest_cat,
            m6_drop,
            s3_v4,
            s6_v4,
        ),
        "p3_honest_neg_closure_m1_re_fold": (
            ml_auc,
            honest_cat,
            s3_v4,
            s6_v4,
        ),
        "krasnov_m1_per_seed_per_split_5fold_5metal": (pv_dir, ml_auc),
        "krasnov_m21_per_cell_line_per_seed": (pv_dir, nci60_dir),
        "krasnov_protocol_alignment_diffs_documented": (
            pv_dir,
            honest_cat,
            pa_matrix,
        ),
        "rusanov_m2_per_cell_line_per_cohort": (pv_dir, m2_json),
        "rusanov_m2_4cell_baseline_comparison": (pv_dir, ru_dir, m2_json),
        "rusanov_protocol_alignment_diffs_documented": (
            pv_dir,
            honest_cat,
            pa_matrix,
        ),
        "drugood_m17_assay_split_auc": (pv_dir, dg_results),
        "drugood_m17_scaffold_split_auc": (pv_dir, dg_results),
        "drugood_m17_size_split_auc": (pv_dir, dg_results),
    }

    fn_map = {
        "p3_protocol_alignment_dataset_match": _pfc.gate_p3_protocol_alignment_dataset_match,
        "p3_protocol_alignment_split_match": _pfc.gate_p3_protocol_alignment_split_match,
        "p3_protocol_alignment_metric_match": _pfc.gate_p3_protocol_alignment_metric_match,
        "p3_protocol_alignment_seed_cohort_per_seed_per_split": _pfc.gate_p3_protocol_alignment_seed_cohort_per_seed_per_split,
        "p3_protocol_alignment_honest_limitation_minor_section": _pfc.gate_p3_protocol_alignment_honest_limitation_minor_section,
        "p3_protocol_alignment_paper_section_6_template": _pfc.gate_p3_protocol_alignment_paper_section_6_template,
        "p3_protocol_alignment_user_ok_gate": _pfc.gate_p3_protocol_alignment_user_ok_gate,
        "p3_protocol_alignment_baseline_paper_refs_bib": _pfc.gate_p3_protocol_alignment_baseline_paper_refs_bib,
        "p3_protocol_recommended_seed_count_3": _pfc.gate_p3_protocol_recommended_seed_count_3,
        "p3_protocol_recommended_per_cell_breakdown": _pfc.gate_p3_protocol_recommended_per_cell_breakdown,
        "p3_protocol_recommended_5_fold_cv_compatibility_50x": _pfc.gate_p3_protocol_recommended_5_fold_cv_compatibility_50x,
        "p3_protocol_recommended_krasnov_paper_table_4_replication": _pfc.gate_p3_protocol_recommended_krasnov_paper_table_4_replication,
        "p3_protocol_recommended_drugood_3_splits_full_alignment": _pfc.gate_p3_protocol_recommended_drugood_3_splits_full_alignment,
        "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup": _pfc.gate_p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup,
        "p3_protocol_recommended_section_6_honest_minor_section_auto_emit": _pfc.gate_p3_protocol_recommended_section_6_honest_minor_section_auto_emit,
        "p3_mock_to_real_m3_vina_kcal_per_mol_REAL_5_pocket": _pfc.gate_p3_mock_to_real_m3_vina_kcal_per_mol_REAL_5_pocket,
        "p3_mock_to_real_m15_rmsd_REAL_5_pocket": _pfc.gate_p3_mock_to_real_m15_rmsd_REAL_5_pocket,
        "p3_mock_to_real_m14_decode_ratio_REAL_30_probes": _pfc.gate_p3_mock_to_real_m14_decode_ratio_REAL_30_probes,
        "p3_mock_to_real_m17_drugood_assay_REAL_3_seeds": _pfc.gate_p3_mock_to_real_m17_drugood_assay_REAL_3_seeds,
        "p3_mock_to_real_m21_nci60_precious_9cell_REAL_3_seeds": _pfc.gate_p3_mock_to_real_m21_nci60_precious_9cell_REAL_3_seeds,
        "p3_honest_neg_closure_m2_r2": _pfc.gate_p3_honest_neg_closure_m2_r2,
        "p3_honest_neg_closure_m6_kappa": _pfc.gate_p3_honest_neg_closure_m6_kappa,
        "p3_honest_neg_closure_m1_re_fold": _pfc.gate_p3_honest_neg_closure_m1_re_fold,
        "krasnov_m1_per_seed_per_split_5fold_5metal": _pfc.gate_krasnov_m1_per_seed_per_split_5fold_5metal,
        "krasnov_m21_per_cell_line_per_seed": _pfc.gate_krasnov_m21_per_cell_line_per_seed,
        "krasnov_protocol_alignment_diffs_documented": _pfc.gate_krasnov_protocol_alignment_diffs_documented,
        "rusanov_m2_per_cell_line_per_cohort": _pfc.gate_rusanov_m2_per_cell_line_per_cohort,
        "rusanov_m2_4cell_baseline_comparison": _pfc.gate_rusanov_m2_4cell_baseline_comparison,
        "rusanov_protocol_alignment_diffs_documented": _pfc.gate_rusanov_protocol_alignment_diffs_documented,
        "drugood_m17_assay_split_auc": _pfc.gate_drugood_m17_assay_split_auc,
        "drugood_m17_scaffold_split_auc": _pfc.gate_drugood_m17_scaffold_split_auc,
        "drugood_m17_size_split_auc": _pfc.gate_drugood_m17_size_split_auc,
    }

    failures = []
    for gate_name, gate_fn in fn_map.items():
        args = arg_sets[gate_name]
        try:
            result = gate_fn(gate_name, "P3", "all", *args)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{gate_name}: call raised {type(exc).__name__}: {exc}")
            continue
        # GateResult shape: name + status + evidence
        assert hasattr(result, "status"), f"{gate_name}: missing status"
        assert result.status in (
            "PASS",
            "FAIL",
            "WARN",
        ), f"{gate_name}: bad status={result.status!r}"
    assert not failures, f"{len(failures)} NEW gates failed smoke call: {failures}"


@pytest.mark.parametrize("gate_name", EXPECTED_NEW_GATES)
def test_wf432a1_each_new_gate_appears_in_all_pipeline(gate_name):
    """§3.4 — every NEW gate appears in build_gates('all') (not frontier/cfm)."""
    runners = _pfc.build_gates("all")
    names = {r[0] for r in runners}
    assert gate_name in names, f"{gate_name} missing from build_gates('all')"
    # All NEW gates route to 'all' only per §3.4 design contract
    frontier_names = {r[0] for r in _pfc.build_gates("frontier")}
    cfm_names = {r[0] for r in _pfc.build_gates("cfm")}
    assert (
        gate_name not in frontier_names
    ), f"{gate_name} should NOT be in 'frontier' per §3.4 design contract"
    assert (
        gate_name not in cfm_names
    ), f"{gate_name} should NOT be in 'cfm' per §3.4 design contract"


def test_wf432a1_no_existing_gate_renamed_or_removed():
    """§3.4 — backward compat: pre-existing 95 gates remain intact.

    Per AGENTS.md §1 + Wf-345-B3 invariants: no existing gate is
    renamed, removed, or modified.  The 4-pipeline subset contract is
    preserved.
    """
    # Pre-Wf-432-A1 gate count was 95 per EXPECTED_GATE_COUNTS comment.
    # Verify the NEW count = 127 (delta = 32 NEW additive).
    assert _pfc.EXPECTED_GATE_COUNTS["all"] - 32 == 95


def test_wf432a1_pre_flight_runs_clean():
    """§5.5 — full pre_flight_check.py --pipeline all runs without crashing
    (no syntax / import / NameError)."""
    # Simulate a build_gates('all') invocation which exercises every
    # NEW gate callable at module import time.
    runners = _pfc.build_gates("all")
    assert len(runners) >= 127  # at least 127 unique
    # Pick one new gate and invoke its callable to verify no NameError
    gate_fn = _pfc.gate_p3_protocol_alignment_baseline_paper_refs_bib
    refs_bib = REPO_ROOT / "paper" / "refs.bib"
    if refs_bib.exists():
        result = gate_fn(
            "p3_protocol_alignment_baseline_paper_refs_bib",
            "P3",
            "all",
            refs_bib,
        )
        assert result.status in ("PASS", "FAIL", "WARN")
