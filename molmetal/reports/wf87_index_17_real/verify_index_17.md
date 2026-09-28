# Wf-87 Phase 3 — INDEX §17 verification report

**Generated**: 2026-09-26T12:58:32.161424+00:00
**Section**: §17 TODO-89 5-Category Audit
**Verifier**: `molmetal/scripts/verify_index_17_real.py`

## §17.1 counts (claimed in INDEX)

| Category | Claimed |
|---|---:|
| STRONG_REAL_PAPER_READY | **10** |
| WEAK_REAL_NEEDS_MORE_DATA | **11** |
| MOCK_NEEDS_REMEASURE | **5** |
| DEFERRED_HOST_GAP | **3** |

## Per-metric on-disk audit

- **Expected**: 25
- **Existing on disk**: 25
- **With measured value**: 24
- **Missing file**: 0

| Metric | Source | Exists | Value | Verdict |
|---|---|:---:|:---:|---|
| m1 | `metrics/by_metric/metalcytotox_ru_auc.json` | YES | YES | MEASURED |
| m2 | `metrics/by_metric/platinai_4cell_auc.json` | YES | YES | MEASURED |
| m3 | `metrics/by_metric/vina_kcal_per_mol.json` | YES | YES | MEASURED |
| m4 | `metrics/by_metric/vina_triple_success.json` | YES | YES | MEASURED |
| m5 | `metrics/by_metric/sa_mean.json` | YES | YES | MEASURED |
| m6 | `metrics/by_metric/qed.json` | YES | YES | MEASURED |
| m7 | `metrics/by_metric/diversity_tanimoto.json` | YES | YES | MEASURED |
| m8 | `metrics/by_metric/pb_pass_rate.json` | YES | YES | MEASURED |
| m9 | `metrics/by_metric/aqueous_solubility_logS.json` | YES | YES | MEASURED |
| m10 | `metrics/by_metric/vina_min_score.json` | YES | YES | MEASURED |
| m11 | `metrics/by_metric/n_distinct.json` | YES | YES | MEASURED |
| m13 | `metrics/by_metric/rdkit_validity.json` | YES | YES | MEASURED |
| m14 | `metrics/by_metric/decode_ratio.json` | YES | YES | MEASURED |
| m15 | `metrics/by_metric/rigid_fragment_rmsd.json` | YES | YES | MEASURED |
| m16 | `metrics/by_metric/pearson_r_pic50.json` | YES | YES | MEASURED |
| m17 | `metrics/by_metric/drugood_ood_auc.json` | YES | YES | MEASURED |
| m19 | `metrics/by_metric/metal_lomo_auc.json` | YES | YES | MEASURED |
| m21 | `metrics/by_metric/nci60_precious_9cell_auc.json` | YES | YES | MEASURED |
| ring_size | `metrics/by_metric/ring_size_distribution.json` | YES | YES | MEASURED |
| metallodrug_vc | `metrics/by_metric/metallodrug_vertical_coverage.json` | YES | YES | MEASURED |
| external_scorers | `metrics/by_metric/external_scorers_summary.json` | YES | NO | NO_VALUE |
| per_pocket_vina | `metrics/by_metric/vina_high_affinity.json` | YES | YES | MEASURED |
| m22 | `metrics/by_metric/metric18_protein_metal_v2.json` | YES | YES | MEASURED |
| m23 | `metrics/by_metric/platinai_3cell_resistance.json` | YES | YES | MEASURED |
| m24 | `molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json` | YES | YES | MEASURED |

## STRONG_REAL cross-check (claimed metrics vs on-disk evidence)

- **Claimed STRONG_REAL**: ['m11', 'm13', 'm14', 'm17', 'm21', 'm22', 'm23', 'm24', 'ring_size', 'metallodrug_vc']
- **Actually MEASURED on disk**: ['m11', 'm13', 'm14', 'm17', 'm21', 'ring_size', 'metallodrug_vc', 'm22', 'm23', 'm24']
- **Missing measured value**: []
- **Verdict**: **PASS**

## §17.4 cross-references (file existence)

- **Expected**: 25
- **Missing**: 0

| Path | Exists | Size |
|---|:---:|---:|
| `TODO/03_plans/LAMBDA_ROADMAP.md` | YES | 42,290 |
| `TODO/completed/83j_lambda_closeout.md` | YES | 15,613 |
| `TODO/completed/82_metric_gap_filling.md` | YES | 23,723 |
| `TODO/completed/89_wf43_routing_policy.md` | YES | 5,944 |
| `TODO/completed/85_mock_audit.md` | YES | 14,657 |
| `TODO/completed/87_r2_debug.md` | YES | 16,243 |
| `TODO/completed/87l_undefined_metrics_definition.md` | YES | 13,912 |
| `TODO/completed/88_r18_verification.md` | YES | 5,416 |
| `TODO/completed/89_wf47_real_runs_execution.md` | YES | 4,826 |
| `TODO/completed/90_wf48_score_any_router.md` | YES | 5,173 |
| `TODO/completed/91_wf49_r2_root_fix.md` | YES | 7,178 |
| `TODO/completed/r20_status_ledger_2026_09_26.md` | YES | 27,384 |
| `TODO/completed/100_r20_r21_consolidation.md` | YES | 15,711 |
| `TODO/completed/100_wf71_p3_wf49_cross_round2.md` | YES | 5,813 |
| `TODO/pending/r22_l13_l16_gap_analysis.md` | YES | 23,883 |
| `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` | YES | 31,589 |
| `molmetal/scripts/audit_5cat_extend.py` | YES | 23,523 |
| `molmetal/reports/wf72_5cat_extend/audit_22_to_25.json` | YES | 18,880 |
| `molmetal/reports/wf31_m1_m2_m7_lift_analysis.json` | YES | 18,363 |
| `molmetal/reports/wf31_m1_m2_m7_lift_summary.md` | YES | 17,044 |
| `molmetal/reports/wf32_m16_m19_m8_lift_analysis.json` | YES | 12,892 |
| `molmetal/reports/wf32_m16_m19_m8_lift_summary.md` | YES | 11,040 |
| `molmetal/reports/wf84_dmpnn_ckpt/training_summary.json` | YES | 3,639 |
| `molmetal/reports/wf84_dmpnn_ckpt/diagnostic_synthesis.md` | YES | 1,233 |
| `molmetal/reports/wf84_dmpnn_ckpt/ckpt_inventory.json` | YES | 2,865 |

## Honest framing provenance

- **Policy**: party-twenty-spirit §3.5 (0 容忍 + 不掩盖)
- **Applied**: Counts claimed in §17.1 are cross-checked against on-disk JSON/JSON-or-MD existence + numeric value presence. Cross-reference files in §17.4 are checked for on-disk existence. No fabricated MEASURED numbers; every discrepancy is surfaced explicitly.

## One-line verdict

**PASS** on STRONG_REAL cross-check; 0 cross-ref file(s) missing of 25; 0 metric source(s) missing of 25.