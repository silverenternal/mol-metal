# Phase 3 Smoke Retrain — Stacked P0+P1 Fixes on 500-mol Metallodrug Pool

**Date**: 2026-09-16
**Script**: `molmetal/scripts/metallo_drug_smoke_retrain.py`
**Inputs**: `molmetal/data/metallo_drugs_500_train.csv` (449 SMILES, 358 3D-embedded, 91 skipped)

## Configuration

| Knob | Value |
|---|---|
| steps (planned) | 1000 |
| steps (executed) | 1000 |
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
| wall-clock | 88.5s (0.09s/step) |

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
| cfm   | 21.406187057495117 | 7.449220657348633 | 3.772672414779663 | -13.956966400146484 |
| atom  | 2.995732307434082 | 1.4297178983688354 | 1.036828875541687 | -1.5660144090652466 |
| bond  | 25.434694290161133 | 1.2415069341659546 | 1.0007952451705933 | -24.193187355995178 |
| total | 47.14045715332031 | 8.833699226379395 | 5.242047309875488 | -38.30675792694092 |

## Decode smoke (n_samples=16 per checkpoint)
| step | decode_ratio | n_decoded | mean_atoms | disconnected | wall_s |
|---|---|---|---|---|---|
| 250 | 1.000 | 16 | 8.0 | 0 | 1.6 |
| 500 | 1.000 | 16 | 8.0 | 0 | 0.3 |
| 750 | 1.000 | 16 | 8.0 | 0 | 0.3 |
| 1000 | 1.000 | 16 | 8.0 | 0 | 0.5 |

**decode_ratio at step 200**: None
**decode_ratio at step 500**: 1.0
**decode_ratio at step 1000**: 1.0
**max decode_ratio observed**: 1.000
**first non-zero step**: 250

## Verdict
**MEASURED decode lift from 0/192 floor**

References:
* Prior baseline (5000-step, h=32, single-fix): `wf_gpu_recovery_now/final.md` decode_ratio=0/192
* P0 fix inventory: `molmetal/reports/wf_cfm_p0_fixes/`
* P1 fix inventory: `molmetal/reports/wf_cfm_p1_fixes/`
* TODO/pending/24_cfm_architecture_redo_plan.md

JSON: `final.json`
Checkpoint: `logs/yuelbond_p1_shim_fix/seed_1/ckpt.pt`
