# Phase 3 Smoke Retrain — Stacked P0+P1 Fixes on 500-mol Metallodrug Pool

**Date**: 2026-09-16
**Script**: `molmetal/scripts/metallo_drug_smoke_retrain.py`
**Inputs**: `molmetal/data/metallo_drugs_500_train.csv` (449 SMILES, 358 3D-embedded, 91 skipped)

## Configuration

| Knob | Value |
|---|---|
| steps (planned) | 10000 |
| steps (executed) | 10000 |
| batch_size | 4 |
| lr | 0.0001 |
| hidden_dim | 128 |
| n_layers | 3 |
| joint_train | True |
| bond_head | learned |
| atom vocab | 20 (metallodrug = 14) |
| bond_head in_dim | 265 (expected 265) |
| device | cuda |
| cuda_available | True |
| torch | 2.14.0+cu130 |
| wall-clock | 555.5s (0.06s/step) |

## P0 fixes active
all ON

| Fix | Status |
|---|---|
| F1 BondAwareDecoder.decode wired | True |
| F2 BondOrderHead in_dim correct (9 + 2*hidden_dim) | True |
| F3 vocab_mask in training CE loss | True |
| F4 hidden_dim >= 64 | True |
| F5 no bonds=zeros placeholder | True |

## P1 fixes active
| Fix | Status |
|---|---|
| P1.1 hidden_dim=128 | True |
| P1.2 learnable vel_scale | True |
| P1.3 PCGrad multi-task | False (off — bit-compat) |
| P1.4 ConnectivityAwareDecoder | True |

## TODO-21 Lambda-as-reward (Strategy 1)
| Knob | Value |
|---|---|
| enabled | False |
| samples requested | 64 |
| dataset size (kept) | 0 |
| after zero-weight filter | 0 |
| zero-weight dropped | 0 |
| mix ratio | 0.25 |
| reweight alpha | 1.0 |
| pocket_id | metallo_drug_smoke |
| loss is weighted | False |

## Loss curves (summary)
| Loss | first | last | min | delta |
|---|---|---|---|---|
| cfm   | 19.450136184692383 | 13.878986358642578 | 3.1310064792633057 | -5.571149826049805 |
| atom  | 2.995732307434082 | 1.1945253610610962 | 0.8304498791694641 | -1.8012069463729858 |
| bond  | 27.81342124938965 | 1.0720338821411133 | 0.9184232354164124 | -26.741387367248535 |
| total | 47.563133239746094 | 15.070472717285156 | 4.368435859680176 | -32.49266052246094 |

## Decode smoke (n_samples=16 per checkpoint)
| step | decode_ratio | n_decoded | mean_atoms | disconnected | wall_s |
|---|---|---|---|---|---|
| 1000 | 1.000 | 16 | 8.0 | 0 | 1.7 |
| 2000 | 1.000 | 16 | 8.0 | 0 | 0.5 |
| 3000 | 1.000 | 16 | 8.0 | 0 | 0.3 |
| 4000 | 1.000 | 16 | 8.0 | 0 | 0.5 |
| 5000 | 1.000 | 16 | 8.0 | 0 | 0.4 |
| 6000 | 1.000 | 16 | 8.0 | 0 | 0.3 |
| 7000 | 1.000 | 16 | 8.0 | 0 | 0.2 |
| 8000 | 1.000 | 16 | 8.0 | 0 | 0.5 |
| 9000 | 1.000 | 16 | 8.0 | 0 | 0.6 |
| 10000 | 1.000 | 16 | 8.0 | 0 | 0.6 |

**decode_ratio at step 200**: None
**decode_ratio at step 500**: None
**decode_ratio at step 1000**: 1.0
**max decode_ratio observed**: 1.000
**first non-zero step**: 1000

## Verdict
**MEASURED decode lift from 0/192 floor**

References:
* Prior baseline (5000-step, h=32, single-fix): `wf_gpu_recovery_now/final.md` decode_ratio=0/192
* P0 fix inventory: `molmetal/reports/wf_cfm_p0_fixes/`
* P1 fix inventory: `molmetal/reports/wf_cfm_p1_fixes/`
* TODO/pending/24_cfm_architecture_redo_plan.md

JSON: `final.json`
Checkpoint: `logs/r17_retrain_10k/seed_2/ckpt.pt`
