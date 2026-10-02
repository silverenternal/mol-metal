# Per-Wave Checklist — Wave-A Wf-300 P1.4 smoke

_Generated: 2026-09-29T01:59:32+00:00_  _Schema: 262.v1_

## Summary

- preflight pass count: **20**
- preflight fail count: **2**
- wave budget: **10.0 h** (9.5 h remaining after abort penalties)

## P0 BLOCKER gates

_no gates in this priority_

## P1 VERIFY gates

| Status | Gate | Evidence |
|---|---|---|
| `PASS` | `p1_chemprop_indices_fix` | grep 'indices = train_idx.copy()' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_kendall_optimizer` | grep 'kendall.parameters()' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_dann_per_metal` | grep 'per_metal' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_nan_guard` | grep 'nan_to_num' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_grad_clip` | grep 'clip_grad_norm' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_epoch_avg` | grep 'w2_epoch_avg_history' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_ablation_flags` | help exposes all 4 token(s) of '--no-multi-task --no-wass --no-kendall --no-dann' (rc=0) |
| `PASS` | `p1_chemprop_seeds_min` | help exposes all 1 token(s) of '--n-seeds-min' (rc=0) |
| `PASS` | `p1_chemprop_master_wass_flag` | help exposes all 1 token(s) of '--use-master-wass' (rc=0) |
| `PASS` | `p1_chemprop_master_cath_flag` | help exposes all 1 token(s) of '--use-master-cath' (rc=0) |
| `PASS` | `p1_chemprop_click_aug_flag` | help exposes all 1 token(s) of '--use-click-aug' (rc=0) |
| `PASS` | `p1_chemprop_smoke` | --quick exit=0 in 43.5s (rc=0) | tail: [2026-09-28 20:03:39,959] INFO wf200 :: [metal=Ru seed=0] epoch=1/2 loss=-1.2693  |
| `PASS` | `p1_cfm_script_exists` | file exists: /home/hugo/codes/mol-metal/molmetal/scripts/metallo_drug_smoke_retrain.py |
| `PASS` | `p1_cfm_config_exists` | file exists: /home/hugo/codes/mol-metal/configs/cfm_full_retrain.yaml |
| `PASS` | `p1_cfm_decoder_exists` | file exists: /home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py |
| `FAIL` | `p1_cfm_wasserstein_loss_exists` | file MISSING: /home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/cfm_wasserstein_loss.py |
| `PASS` | `p1_cfm_smoke_decoded` | /home/hugo/codes/mol-metal/molmetal/reports/wf188_m14_full_metalcytotox/result.json: value=0.995821 >= 0.5 |
| `PASS` | `p1_cfm_catherine_30metals` | /home/hugo/codes/mol-metal/molmetal/reports/wf256_apply/catherine_30/per_metal_pass.json: value=1.000000 >= 0.95 |

## P2 DOCUMENTED gates

| Status | Gate | Evidence |
|---|---|---|
| `PASS` | `p2_data_metalcytotox_exists` | wc -l /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask.csv = 26802 >= 26800 |
| `PASS` | `p2_data_metallo_drugs_exists` | wc -l /home/hugo/codes/mol-metal/molmetal/data/metallo_drugs_500_train.csv = 450 >= 449 |
| `PASS` | `p2_tests_pass` | pytest PASS: 24 passed, 1 warning in 3.42s |
| `FAIL` | `p2_vina_installed` | which vina -> not found; apt-list did not show match |

## Config summary

- yaml_path: `configs/cfm_per_innovation_ablation_p1.4.yaml`
- parse_ok: **False**
- error: ``
- n_required_keys_present: **8** of 9
- missing_keys: `train`

## Abort criteria active (§7.2)

| Criterion | Description |
|---|---|
| `7.2.8_partial_pilot` | n_completed < n_total (5/12 partial pilot) |

## Per-pair sub-checklist

| Metal | Seed | Expected ckpt | Expected AUC | Status | Wall (s) |
|---|---|---|---|---|---|
| Ru | 0 | `dmpnn_mtw_ru_seed0.pt` | 0.7613 | `DONE` | 463.2 |
| Ru | 1 | `dmpnn_mtw_ru_seed1.pt` | — | `IN-FLIGHT` | — |
| Ru | 2 | `dmpnn_mtw_ru_seed2.pt` | — | `IN-FLIGHT` | — |
| Ru | 3 | `dmpnn_mtw_ru_seed3.pt` | — | `PENDING` | — |
| Ru | 4 | `dmpnn_mtw_ru_seed4.pt` | — | `PENDING` | — |
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
