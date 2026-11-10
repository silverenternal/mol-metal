# Per-Wave Checklist — wave_a

_Generated: 2026-11-02T13:51:08+00:00_  _Schema: 262.v2_

## Summary

- preflight pass count: **3**
- preflight fail count: **1**
- wave budget: **48.0 h** (47.5 h remaining after abort penalties)

## P0 BLOCKER gates

| Status | Gate | Evidence |
|---|---|---|
| `PASS` | `p1_ssot_path_reference_check` | ok |
| `FAIL` | `p3_blocker_probe` | synthetic FAIL |

## P1 VERIFY gates

| Status | Gate | Evidence |
|---|---|---|
| `PASS` | `p3_anchorguard_version_pin` | ok |

## P2 DOCUMENTED gates

| Status | Gate | Evidence |
|---|---|---|
| `PASS` | `p3_yaml_only_launch_check` | ok |

## P3 PAPER-GRADE gates

_no gates in this priority_

## Config summary

- yaml_path: `/home/hugo/codes/mol-metal/configs/cfm_full_retrain.yaml`
- parse_ok: **False**
- error: ``
- n_required_keys_present: **7** of 9
- missing_keys: `train, eval`

## Abort criteria active (§7.2)

| Criterion | Description |
|---|---|
| `7.2.8_partial_pilot` | n_completed < n_total (5/12 partial pilot) |

## Per-pair sub-checklist

| Metal | Seed | Expected ckpt | Expected AUC | Status | Wall (s) |
|---|---|---|---|---|---|
| Ru | 0 | `dmpnn_mtw_ru_seed0.pt` | 0.7613 | `IN-FLIGHT` | 463.2 |
| Ru | 1 | `dmpnn_mtw_ru_seed1.pt` | — | `IN-FLIGHT` | — |
| Ru | 2 | `dmpnn_mtw_ru_seed2.pt` | — | `IN-FLIGHT` | — |
| Ru | 3 | `dmpnn_mtw_ru_seed3.pt` | — | `IN-FLIGHT` | — |
| Ru | 4 | `dmpnn_mtw_ru_seed4.pt` | — | `IN-FLIGHT` | — |
| Ir | 0 | `dmpnn_mtw_ir_seed0.pt` | — | `IN-FLIGHT` | — |
| Ir | 1 | `dmpnn_mtw_ir_seed1.pt` | — | `IN-FLIGHT` | — |
| Ir | 2 | `dmpnn_mtw_ir_seed2.pt` | — | `IN-FLIGHT` | — |
| Ir | 3 | `dmpnn_mtw_ir_seed3.pt` | — | `PENDING` | — |
| Ir | 4 | `dmpnn_mtw_ir_seed4.pt` | — | `PENDING` | — |
| Os | 0 | `dmpnn_mtw_os_seed0.pt` | — | `IN-FLIGHT` | — |
| Os | 1 | `dmpnn_mtw_os_seed1.pt` | — | `IN-FLIGHT` | — |
| Os | 2 | `dmpnn_mtw_os_seed2.pt` | — | `IN-FLIGHT` | — |
| Os | 3 | `dmpnn_mtw_os_seed3.pt` | — | `PENDING` | — |
| Os | 4 | `dmpnn_mtw_os_seed4.pt` | — | `PENDING` | — |
| Rh | 0 | `dmpnn_mtw_rh_seed0.pt` | — | `IN-FLIGHT` | — |
| Rh | 1 | `dmpnn_mtw_rh_seed1.pt` | — | `IN-FLIGHT` | — |
| Rh | 2 | `dmpnn_mtw_rh_seed2.pt` | — | `IN-FLIGHT` | — |
| Rh | 3 | `dmpnn_mtw_rh_seed3.pt` | — | `PENDING` | — |
| Rh | 4 | `dmpnn_mtw_rh_seed4.pt` | — | `PENDING` | — |
| Re | 0 | `dmpnn_mtw_re_seed0.pt` | — | `IN-FLIGHT` | — |
| Re | 1 | `dmpnn_mtw_re_seed1.pt` | — | `IN-FLIGHT` | — |
| Re | 2 | `dmpnn_mtw_re_seed2.pt` | — | `IN-FLIGHT` | — |
| Re | 3 | `dmpnn_mtw_re_seed3.pt` | — | `PENDING` | — |
| Re | 4 | `dmpnn_mtw_re_seed4.pt` | — | `PENDING` | — |

---
_Honest framing (party-twenty-spirit §3.5): every entry above is POPULATED from JSON / on-disk evidence, not synthesised. Counts and status values trace to the referenced files; MEASURED vs PROJECTED is preserved per row. Re-run this script before each wave re-launch._
