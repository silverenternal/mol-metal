# Wf-261-Regenerate — Strict Bemis-Murcko Scaffold 70/10/20 Split

**Date**: 2026-09-28T20:24:32+0800  
**Duration**: 11.53 s

## §1 Run summary

| Field | Value |
|---|---|
| `started_at_iso` | `2026-09-28T20:24:32+0800` |
| `duration_seconds` | `11.53` |
| `v1_path` | `/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask.csv` |
| `v1_backup` | `/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v1.csv` |
| `v2_path` | `/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v2.csv` |
| `manifest_path` | `/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v2_split_manifest.json` |
| `n_total_rows` | `26801` |
| `n_dropped_null` | `169` |
| `n_unique_scaffolds` | `3069` |
| `n_rows_v2` | `26632` |
| `split_counts` | `{'train': 18137, 'test': 5736, 'val': 2759}` |
| `split_pct` | `{'train': 68.1, 'test': 21.54, 'val': 10.36}` |
| `scaffold_count_per_split` | `{'train': 2148, 'val': 307, 'test': 614}` |
| `sha256_v2` | `cc3ea24755f1b14a69c8c3acd827db0bc1dd123301479c9ca490920725adf063` |
| `overlap_pairs_count` | `0` |
| `g1_v1_backed_up` | `True` |
| `g2_v2_csv_written` | `True` |
| `g3_zero_scaffold_leak` | `True` |
| `g4_row_distribution_ok` | `True` |
| `g5_split_manifest_written` | `True` |
| `honest_framing_note` | `169 of 26801 rows dropped due to null/empty Bemis-Murcko scaffold (computed fresh via RDKit MurckoScaffold.GetScaffoldForMol); 3069 unique scaffolds across 26632 kept rows. These dropouts are reported honestly in regen_report.md (G_dropped_rows).` |

## §2 Acceptance gates (FLAT)

| Gate | Value |
|---|---|
| G1_v1_backed_up | `True` |
| G2_v2_csv_written | `True` |
| G3_zero_scaffold_leak | `True` |
| G4_row_distribution_ok | `True` |
| G5_split_manifest_written | `True` |

## §3 Honest framing

169 of 26801 rows dropped due to null/empty Bemis-Murcko scaffold (computed fresh via RDKit MurckoScaffold.GetScaffoldForMol); 3069 unique scaffolds across 26632 kept rows. These dropouts are reported honestly in regen_report.md (G_dropped_rows).
