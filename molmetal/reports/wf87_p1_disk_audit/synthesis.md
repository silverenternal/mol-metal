# Wf-87 Phase 1 — Per-Metric Disk Audit

**Date**: 2026-10-11  
**Workflow**: Wf-87 Phase 1 — per-metric disk audit  
**Constraint**: CPU-only; no GPU retrain; paper-gate (TODO-84) honored; party-twenty-spirit §3.5 honest framing  

## 1. Aggregate

- `metrics/by_metric/` files scanned: **174** JSON files.
- Conceptual metrics audited (per `TODO/INDEX.md` §17.1): **26**.
- Conceptual category distribution:
  - **DEFERRED**: 3
  - **MOCK**: 2
  - **STRONG_REAL**: 10
  - **WEAK_REAL**: 11
- On-disk state distribution (post-scan):
  - **DEFERRED**: 2
  - **MEASURED**: 20
  - **REVIEW_NEEDED**: 4

## 2. Per-metric table (22 conceptual metrics)

| # | Metric | Bucket | On-disk state | M | P | D | Latest value | mtime_utc | Source |
|---:|---|---|---|---:|---:|---:|---:|---|---|

| 1 | QED (`m6`) | STRONG_REAL | MEASURED | 1 | 0 | 0 | — | `2026-10-08T12:15:26.248688+00:00` | `metrics/by_metric/qed.json` (+`TODO/completed/87l_undefined_metrics_definition.md`) |
| 2 | Water solubility logS (`m9`) | STRONG_REAL | MEASURED | 0 | 0 | 0 | -7.6122 | `2026-10-01T09:00:09.669039+00:00` | `metrics/by_metric/aqueous_solubility_logS.json` (+`TODO/completed/87l_undefined_metrics_definition.md`) |
| 3 | n_distinct (`m11`) | STRONG_REAL | MEASURED | 4 | 0 | 0 | 20.0000 | `2026-09-25T04:54:08.754357+00:00` | `metrics/by_metric/n_distinct.json` (+`—`) |
| 4 | RDKit validity (`m13`) | STRONG_REAL | MEASURED | 1 | 0 | 0 | 1.0000 | `2026-10-03T09:58:25.508375+00:00` | `metrics/by_metric/rdkit_validity.json` (+`TODO/completed/87l_undefined_metrics_definition.md`) |
| 5 | CFM decode (`m14`) | STRONG_REAL | MEASURED | 13 | 0 | 0 | 0.9980 | `2026-10-04T06:11:58.211846+00:00` | `metrics/by_metric/decode_ratio.json` (+`—`) |
| 6 | DrugOOD OOD AUC (`m17`) | STRONG_REAL | MEASURED | 1 | 0 | 0 | — | `2026-10-07T19:18:59.767023+00:00` | `metrics/by_metric/drugood_ood_auc.json` (+`—`) |
| 7 | NCI-60 9-cell AUC (`m21`) | STRONG_REAL | MEASURED | 1 | 0 | 0 | 0.4200 | `2026-10-07T19:18:59.767803+00:00` | `metrics/by_metric/nci60_precious_9cell_auc.json` (+`—`) |
| 8 | m18 Protein-Metal binding AUC (`m22`) | STRONG_REAL | MEASURED | 1 | 0 | 0 | — | `2026-10-03T09:58:25.503375+00:00` | `metrics/by_metric/metric18_protein_metal_v2.json` (+`—`) |
| 9 | PlatinAI 3-cell resistance AUC (`m23`) | STRONG_REAL | MEASURED | 1 | 0 | 0 | — | `2026-10-03T09:58:25.516375+00:00` | `metrics/by_metric/platinai_3cell_resistance.json` (+`—`) |
| 10 | cross-dataset macro R² (`m24`) | STRONG_REAL | MEASURED | 1 | 0 | 0 | — | `2026-09-26T09:31:13.769665+00:00` | `molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json` (+`—`) |
| 11 | Ru AUC (`m1`) | WEAK_REAL | MEASURED | 2 | 0 | 0 | 0.7613 | `2026-10-03T22:40:09.120225+00:00` | `metrics/by_metric/metalcytotox_ru_auc.json` (+`—`) |
| 12 | Pt 4-cell AUC (`m2`) | WEAK_REAL | MEASURED | 1 | 0 | 0 | — | `2026-10-10T20:42:21.886252+00:00` | `metrics/by_metric/platinai_4cell_auc.json` (+`—`) |
| 13 | Vina triple success (`m4`) | WEAK_REAL | MEASURED | 1 | 0 | 0 | — | `2026-10-03T09:58:25.516375+00:00` | `metrics/by_metric/vina_triple_success.json` (+`—`) |
| 14 | SA mean (`m5`) | WEAK_REAL | MEASURED | 12 | 0 | 0 | 2.4583 | `2026-09-28T08:28:09.329217+00:00` | `metrics/by_metric/sa_mean.json` (+`—`) |
| 15 | Diversity Tanimoto (`m7`) | WEAK_REAL | MEASURED | 6 | 0 | 0 | 0.0000 | `2026-10-03T09:58:25.514375+00:00` | `metrics/by_metric/diversity_tanimoto.json` (+`—`) |
| 16 | PoseBusters pass-rate (`m8`) | WEAK_REAL | MEASURED | 2 | 0 | 0 | 0.8460 | `2026-10-03T09:58:25.514375+00:00` | `metrics/by_metric/pb_pass_rate.json` (+`—`) |
| 17 | Vina min score (`m10`) | WEAK_REAL | MEASURED | 1 | 0 | 0 | — | `2026-10-03T09:58:25.515375+00:00` | `metrics/by_metric/vina_min_score.json` (+`—`) |
| 18 | RMSD (`m15`) | WEAK_REAL | REVIEW_NEEDED | 0 | 0 | 0 | — | `2026-09-26T07:09:12.613798+00:00` | `TODO/completed/85_mock_audit.md` (+`—`) |
| 19 | Affinity Pearson r (`m16`) | WEAK_REAL | MEASURED | 8 | 0 | 0 | 0.4074 | `2026-10-08T14:01:12.925273+00:00` | `metrics/by_metric/pearson_r_pic50.json` (+`—`) |
| 20 | Multi-metal LOMO (`m19`) | WEAK_REAL | MEASURED | 1 | 0 | 0 | 0.5346 | `2026-10-03T09:58:25.516375+00:00` | `metrics/by_metric/metal_lomo_auc.json` (+`—`) |
| 21 | per-pocket Vina Phase-1B (`per_pocket_vina`) | WEAK_REAL | REVIEW_NEEDED | 0 | 0 | 0 | — | `2026-09-27T02:51:20.159948+00:00` | `TODO/completed/82b_vina_dock_retrain.md` (+`—`) |
| 22 | Vina Dock stub (`m3`) | MOCK | REVIEW_NEEDED | 0 | 0 | 0 | — | `2026-09-27T02:51:20.159948+00:00` | `TODO/completed/82b_vina_dock_retrain.md` (+`—`) |
| 23 | 4 external scorers (coord_geometry / phototherapy / ptiv / metalloprotein) (`external_scorers`) | MOCK | REVIEW_NEEDED | 0 | 0 | 0 | — | `2026-09-26T07:09:12.613798+00:00` | `TODO/completed/85_mock_audit.md` (+`—`) |
| 24 | diversity_subpocket (`diversity_subpocket`) | DEFERRED | MEASURED | 3 | 0 | 1 | — | `2026-10-07T21:57:07.674739+00:00` | `metrics/by_metric/diversity_subpocket.json` (+`—`) |
| 25 | steric_clash (PB dock-mode) (`steric_clash`) | DEFERRED | DEFERRED | 0 | 0 | 1 | — | `2026-10-08T05:46:19.157671+00:00` | `metrics/by_metric/steric_clash_pb_dock.json` (+`—`) |
| 26 | vina_*_test_001 (×4 sub-variants) (`vina_test_001`) | DEFERRED | DEFERRED | 0 | 0 | 1 | — | `2026-10-03T09:58:25.515375+00:00` | `metrics/by_metric/vina_score_test_001.json` (+`—`) |

## 3. Honest-framing notes

- All values above are taken directly from on-disk evidence — no fabricated MEASURED numbers and no GPU retrain executed.
- `MEASURED_BELOW_GATE` (e.g. `metalcytotox_ru_auc.json`) is treated as MEASURED on disk but flagged WEAK_REAL because the value is below the Krasnov-2026 gate (0.81).
- `diversity_tanimoto.json` carries the R17 baseline (0.1065) as canonical for *generated* molecules; the Wf-31 entry (`+0.9036`) is on a *different cohort* (PlatinAI 5K random subset) and is not an apples-to-apples lift. See file-level `note`.
- The three `DEFERRED_HOST_GAP` rows (`diversity_subpocket`, `steric_clash_pb_dock`, `vina_*_test_001`) are kept on disk with an explicit `deferred_reason` per Wf-61 §1.4–1.6 close-out.
- The m22/m23/m24 STRONG_REAL rows reference `metric18_protein_metal_v2.json`, `platinai_3cell_resistance.json`, and the Wf-71 P3 cross-dataset summary JSON respectively.

## 4. Cross-references

- `TODO/INDEX.md` §17 (22 metrics × 5 categories table)
- `TODO/INDEX.md` §17.5 (R22 plan rows TODO 100-107)
- `TODO/completed/r20_status_ledger_2026_09_26.md` (Wf-70)
- `molmetal/scripts/audit_5cat_extend.py` (Wf-72 Phase 3)
- `molmetal/reports/wf72_5cat_extend/audit_22_to_25.json`

## 5. Outputs

- `molmetal/reports/wf87_p1_disk_audit/per_metric_status.json`
- `molmetal/reports/wf87_p1_disk_audit/synthesis.md` (this file)
