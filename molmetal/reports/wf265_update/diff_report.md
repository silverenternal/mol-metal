# Wf-265 paper §4 diff_report

- generated_utc: `2026-09-28T16:24:41+00:00`
- collect_json:  `molmetal/reports/wf265_collect/all_results.json`
- paper_doc:     `TODO/completed/188_paper_4_krasnov_protocol_aligned.md`
- n_claims:      11
- dry_run:       True

## Claims

| claim_id | label | old_value | new_value | delta | json_path |
|---|---|---:|---:|---:|---|
| `overall_mean_test_auc` | 5-metal mean AUC (TODO-154 in-domain) | 0.8860 | **0.7209** | -0.1651 | `g4_overall_aggregate.mean_test_auc` |
| `per_metal_Ir_mean_test_auc` | Ir per-metal AUC (TODO-154) | 0.8850 | **0.6702** | -0.2148 | `g4_per_metal_aggregate.Ir.mean_test_auc` |
| `per_metal_Os_mean_test_auc` | Os per-metal AUC (TODO-154) | 0.8890 | **0.7025** | -0.1865 | `g4_per_metal_aggregate.Os.mean_test_auc` |
| `per_metal_Re_mean_test_auc` | Re per-metal AUC (TODO-154) | 0.8820 | **0.7700** | -0.1120 | `g4_per_metal_aggregate.Re.mean_test_auc` |
| `per_metal_Rh_mean_test_auc` | Rh per-metal AUC (TODO-154) | 0.8630 | **0.7227** | -0.1403 | `g4_per_metal_aggregate.Rh.mean_test_auc` |
| `per_metal_Ru_mean_test_auc` | Ru per-metal AUC (TODO-154) | 0.9090 | **0.7483** | -0.1607 | `g4_per_metal_aggregate.Ru.mean_test_auc` |
| `wf269c_yuelbond_full_decode_ratio` | Wf-269C full (YuelBond) decode_ratio (real cohort, MEASURED) | 1.0000 | **1.0000** | +0.0000 | `metrics/by_metric/cfm_decode_real.json::value_vs_wf188` |
| `wf269c_no_yuelbond_decode_ratio` | Wf-269C no_yuelbond decode_ratio (real cohort, MEASURED) | — | **0.9645** | — | `metrics/by_metric/cfm_decode_real.json::regression_disclosure.current_value` |
| `wf271b_m17_reeval_below_random` | Wf-271B m17_drugood_v3_reeval BELOW_RANDOM verdict | — | **0.4544** | — | `metrics/by_metric/drugood_v3_scaffold.json::per_metal.*.test_auc_mean` |
| `wf275b_drugood_adapter_rows` | Wf-275B DrugOOD ECFP4+scaffold adapter n_rows (MEASURED) | — | **95236.0000** | — | `molmetal/reports/wf275b_drugood_adapter/adapter_report.md::n_source_rows` |
| `wf275c_m14_lead_delta_yuelbond_off` | Wf-275C m14 LEAD precondition delta (YuelBond off) | — | **-1.0000** | — | `TODO/completed/m14_lead_deployment_precondition.md::Δ_yuelbond_off` |

## Provenance

- **NEW values** come from the Wf-265-Collect JSON. Every row carries its dotted JSON path so the number can be back-traced verbatim.
- **OLD values** are parsed out of the paper doc by the pattern catalogue in `paper_auto_update.py::PATTERNS`. If a pattern fails to match, the row is reported as `—` (missing); this is a **catastrophic failure** signal, not a soft skip — the user must extend the pattern catalogue.
