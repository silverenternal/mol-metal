"""Text of the registration block inserted into pre_flight_check.py (Wf-A169).

Kept in a separate module so the diff generator can splice it without
shell-quoting hazards.  Every argument below is justified in
molmetal/reports/wfa169_dead_gate_repair/registration_table.md.
"""

INSERT_BLOCK = '''
        # ---- Wf-A169 (2026-11-01): the 24 Wf-432-A1 gates that were
        # DEFINED but NEVER REGISTERED in any pipeline.  RE-REGISTERED here.
        #
        # MEASURED provenance (re-derived 2026-11-01, not taken on faith):
        #   * `git log --all -S'<gate name>' -- molmetal/scripts/pre_flight_check.py`
        #     returns EXACTLY ONE commit for every one of the 24 names:
        #     bc29aa90 (2026-10-02) — the commit that ADDED the function
        #     DEFINITIONS.  The string count never rose above 1, so the
        #     registration call was never written, not "written then removed".
        #   * The Wf-437 "29 SCAFFOLD-only gate tuples REMOVED" comment
        #     (build_gates, "Wf-437 simplify") does NOT refer to these 24:
        #     the only `p3_protocol_*` gates actually registered today are
        #     the 4 `p3_protocol_alignment_yaml_*` (Wf-350-C1 family), and
        #     `git show 140e1f82 -- pre_flight_check.py` removes no
        #     `p3_protocol_alignment_*` / `*_recommended_*` /
        #     `krasnov_*` / `rusanov_*` / `drugood_*` line at all.
        #   * The 24 gate FUNCTIONS are intact at L10768..L12580.
        #
        # Routing: PRE_TRAINING_GATE_v2.md §1.3 — L1 + L2 route to the
        # `all` pipeline ONLY; L5 hooks also to `all` only (the §4 summary
        # puts their output at molmetal/reports/protocol_validation/*.json,
        # "no main-Layer-0 routing" = not frontier / chemprop_v2_v3 / cfm).
        # `frontier` / `chemprop_v2_v3` / `cfm` counts are UNCHANGED.
        # Priority "P3" + literal pipeline "all" mirror the sibling L3/L4
        # registrations directly above (same block, same conventions).
        #
        # Argument sources:
        #   (a) the `_p2_*` locals defined at the top of this `all` block
        #       (Wf-432-A1's own "Spec source paths (all verified on disk)"
        #       set) — every one of them was UNUSED until now;
        #   (b) TODO/03_plans/PRE_TRAINING_GATE_v2.md §2 / §3 / §4;
        #   (c) molmetal/tests/test_wf433a1_gate_v2_wire.py `arg_sets`.
        runners += [
            # ---- §2 L1 Tier-1 must-pass (8 gates; BLOCKING) ----
            (
                "p3_protocol_alignment_dataset_match",
                lambda: gate_p3_protocol_alignment_dataset_match(
                    "p3_protocol_alignment_dataset_match",
                    "P3",
                    "all",
                    # metalcytotox: cytox_csv_v4 (registry/datasets.yaml
                    # lifecycle_status=ACTIVE, locked_for m1_ru_auc /
                    # m19_lomo / m21_nci60_9cell).  NOT `_p2_metalcytotox_csv`
                    # (v1), which registry/datasets.yaml marks SUPERSEDED.
                    cytox_csv_v4,
                    _p2_platinai_csv,
                    _p2_crossdocked_csv,
                    _p2_nci60_csv,
                    _p2_balanced_rh_json,
                    _p2_drugood_loader,
                ),
            ),
            (
                "p3_protocol_alignment_split_match",
                lambda: gate_p3_protocol_alignment_split_match(
                    "p3_protocol_alignment_split_match",
                    "P3",
                    "all",
                    _p2_leakage_utils,
                    _p2_drugood_all_results,
                ),
            ),
            (
                "p3_protocol_alignment_metric_match",
                lambda: gate_p3_protocol_alignment_metric_match(
                    "p3_protocol_alignment_metric_match",
                    "P3",
                    "all",
                    _p2_eval_all_evaluators,
                ),
            ),
            (
                "p3_protocol_alignment_seed_cohort_per_seed_per_split",
                lambda: gate_p3_protocol_alignment_seed_cohort_per_seed_per_split(
                    "p3_protocol_alignment_seed_cohort_per_seed_per_split",
                    "P3",
                    "all",
                    _p2_train_metal_cytotox,
                    _p2_train_platinai,
                    _p2_train_nci60_10cell,
                    _p2_per_seed_template_dir,
                ),
            ),
            (
                "p3_protocol_alignment_honest_limitation_minor_section",
                lambda: gate_p3_protocol_alignment_honest_limitation_minor_section(
                    "p3_protocol_alignment_honest_limitation_minor_section",
                    "P3",
                    "all",
                    _p2_honest_negative_catalog,
                ),
            ),
            (
                "p3_protocol_alignment_paper_section_6_template",
                lambda: gate_p3_protocol_alignment_paper_section_6_template(
                    "p3_protocol_alignment_paper_section_6_template",
                    "P3",
                    "all",
                    _p2_section_6_template,
                    _p2_paper_section_6_v4,
                ),
            ),
            (
                "p3_protocol_alignment_user_ok_gate",
                lambda: gate_p3_protocol_alignment_user_ok_gate(
                    "p3_protocol_alignment_user_ok_gate",
                    "P3",
                    "all",
                    SCRIPTS_DIR,
                    _p2_wrapper_names,
                ),
            ),
            (
                "p3_protocol_alignment_baseline_paper_refs_bib",
                lambda: gate_p3_protocol_alignment_baseline_paper_refs_bib(
                    "p3_protocol_alignment_baseline_paper_refs_bib",
                    "P3",
                    "all",
                    _p2_paper_refs_bib,
                ),
            ),
            # ---- §3 L2 Tier-2 recommended (7 gates; WARN-only) ----
            (
                "p3_protocol_recommended_seed_count_3",
                lambda: gate_p3_protocol_recommended_seed_count_3(
                    "p3_protocol_recommended_seed_count_3",
                    "P3",
                    "all",
                    _p2_train_metal_cytotox,
                    _p2_train_platinai,
                    _p2_train_nci60_10cell,
                ),
            ),
            (
                "p3_protocol_recommended_per_cell_breakdown",
                lambda: gate_p3_protocol_recommended_per_cell_breakdown(
                    "p3_protocol_recommended_per_cell_breakdown",
                    "P3",
                    "all",
                    _p2_train_platinai,
                    _p2_train_nci60_10cell,
                    _p2_m2_r2_per_cohort,
                ),
            ),
            (
                "p3_protocol_recommended_5_fold_cv_compatibility_50x",
                lambda: gate_p3_protocol_recommended_5_fold_cv_compatibility_50x(
                    "p3_protocol_recommended_5_fold_cv_compatibility_50x",
                    "P3",
                    "all",
                    _p2_train_metal_cytotox,
                    _p2_drugood_loader,
                ),
            ),
            (
                "p3_protocol_recommended_krasnov_paper_table_4_replication",
                lambda: gate_p3_protocol_recommended_krasnov_paper_table_4_replication(
                    "p3_protocol_recommended_krasnov_paper_table_4_replication",
                    "P3",
                    "all",
                    MOLMETAL_ROOT / "reports",
                ),
            ),
            (
                "p3_protocol_recommended_drugood_3_splits_full_alignment",
                lambda: gate_p3_protocol_recommended_drugood_3_splits_full_alignment(
                    "p3_protocol_recommended_drugood_3_splits_full_alignment",
                    "P3",
                    "all",
                    _p2_drugood_all_results,
                ),
            ),
            (
                "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup",
                lambda: gate_p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup(
                    "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup",
                    "P3",
                    "all",
                    MOLMETAL_ROOT / "reports",
                    _p2_train_platinai,
                ),
            ),
            (
                "p3_protocol_recommended_section_6_honest_minor_section_auto_emit",
                lambda: gate_p3_protocol_recommended_section_6_honest_minor_section_auto_emit(
                    "p3_protocol_recommended_section_6_honest_minor_section_auto_emit",
                    "P3",
                    "all",
                    _p2_auto_emit_section_6_minor,
                ),
            ),
            # ---- §4 L5 per-baseline protocol validation hooks (9 gates) ----
            # hook_dir = molmetal/reports/protocol_validation/ ; the hook
            # emitter writes {krasnov,rusanov,drugood}_hook.json into it.
            (
                "krasnov_m1_per_seed_per_split_5fold_5metal",
                lambda: gate_krasnov_m1_per_seed_per_split_5fold_5metal(
                    "krasnov_m1_per_seed_per_split_5fold_5metal",
                    "P3",
                    "all",
                    _p2_protocol_validation_dir,
                    _p2_metal_lomo_auc,
                ),
            ),
            (
                "krasnov_m21_per_cell_line_per_seed",
                lambda: gate_krasnov_m21_per_cell_line_per_seed(
                    "krasnov_m21_per_cell_line_per_seed",
                    "P3",
                    "all",
                    _p2_protocol_validation_dir,
                    _p2_nci60_9cell_dir,
                ),
            ),
            (
                "krasnov_protocol_alignment_diffs_documented",
                lambda: gate_krasnov_protocol_alignment_diffs_documented(
                    "krasnov_protocol_alignment_diffs_documented",
                    "P3",
                    "all",
                    _p2_protocol_validation_dir,
                    _p2_honest_negative_catalog,
                    _p2_protocol_alignment_matrix,
                ),
            ),
            (
                "rusanov_m2_per_cell_line_per_cohort",
                lambda: gate_rusanov_m2_per_cell_line_per_cohort(
                    "rusanov_m2_per_cell_line_per_cohort",
                    "P3",
                    "all",
                    _p2_protocol_validation_dir,
                    _p2_m2_r2_per_cohort,
                ),
            ),
            (
                "rusanov_m2_4cell_baseline_comparison",
                lambda: gate_rusanov_m2_4cell_baseline_comparison(
                    "rusanov_m2_4cell_baseline_comparison",
                    "P3",
                    "all",
                    _p2_protocol_validation_dir,
                    _p2_rusanov_baseline_dir,
                    _p2_m2_r2_per_cohort,
                ),
            ),
            (
                "rusanov_protocol_alignment_diffs_documented",
                lambda: gate_rusanov_protocol_alignment_diffs_documented(
                    "rusanov_protocol_alignment_diffs_documented",
                    "P3",
                    "all",
                    _p2_protocol_validation_dir,
                    _p2_honest_negative_catalog,
                    _p2_protocol_alignment_matrix,
                ),
            ),
            (
                "drugood_m17_assay_split_auc",
                lambda: gate_drugood_m17_assay_split_auc(
                    "drugood_m17_assay_split_auc",
                    "P3",
                    "all",
                    _p2_protocol_validation_dir,
                    _p2_drugood_all_results,
                ),
            ),
            (
                "drugood_m17_scaffold_split_auc",
                lambda: gate_drugood_m17_scaffold_split_auc(
                    "drugood_m17_scaffold_split_auc",
                    "P3",
                    "all",
                    _p2_protocol_validation_dir,
                    _p2_drugood_all_results,
                ),
            ),
            (
                "drugood_m17_size_split_auc",
                lambda: gate_drugood_m17_size_split_auc(
                    "drugood_m17_size_split_auc",
                    "P3",
                    "all",
                    _p2_protocol_validation_dir,
                    _p2_drugood_all_results,
                ),
            ),
        ]
'''

# The `_p2_protocol_alignment_matrix` local points at
# TODO/03_plans/PROTOCOL_ALIGNMENT_MATRIX.md, which does not exist; the
# canonical file is TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md (per
# PRE_TRAINING_GATE_v2.md §1.3 cross-references line 10 and line 461).
# This local was UNUSED before this patch, so correcting it changes nothing
# else in the file.
OLD_MATRIX = '''        _p2_protocol_alignment_matrix = (
            REPO_ROOT / "TODO" / "03_plans" / "PROTOCOL_ALIGNMENT_MATRIX.md"
        )
'''

NEW_MATRIX = '''        # Wf-A169 (2026-11-01): the canonical file is under TODO/03_baselines/
        # (PRE_TRAINING_GATE_v2.md §1.3 cross-references, line 10 + line 461).
        # The 03_plans/ path never existed on disk.  This local was UNUSED
        # until Wf-A169 registered the two L5 protocol-diff hooks.
        _p2_protocol_alignment_matrix = (
            REPO_ROOT / "TODO" / "03_baselines" / "PROTOCOL_ALIGNMENT_MATRIX.md"
        )
'''

# Anchor: the closing `]` of the L3/L4 `runners += [...]` block inside the
# `if pipeline == "all":` branch.  The block ends with the
# `p3_mock_to_real_m21_nci60_precious_9cell_REAL_3_seeds` tuple.
ANCHOR = '''                    METRICS_DIR / "nci60_precious_9cell_auc.json",
                ),
            ),
        ]
    # ---- §3.1 P1 chemprop v2 v3 (12) ----
'''

ANCHOR_REPLACEMENT = '''                    METRICS_DIR / "nci60_precious_9cell_auc.json",
                ),
            ),
        ]
''' + INSERT_BLOCK + '''    # ---- §3.1 P1 chemprop v2 v3 (12) ----
'''
