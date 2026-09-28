# Phase 3 Smoke Retrain — Stacked P0+P1 Fixes on 500-mol Metallodrug Pool

**Date**: 2026-09-16
**Script**: `molmetal/scripts/metallo_drug_smoke_retrain.py`
**Inputs**: `molmetal/data/metallo_drugs_500_train.csv` (449 SMILES, 358 3D-embedded, 91 skipped)

## Configuration

| Knob | Value |
|---|---|
| steps (planned) | 2000 |
| steps (executed) | 2000 |
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
| wall-clock | 172.6s (0.09s/step) |

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
| cfm   | 19.450136184692383 | 9.992439270019531 | 3.7503819465637207 | -9.457696914672852 |
| atom  | 2.995732307434082 | 1.4222095012664795 | 0.979856550693512 | -1.5735228061676025 |
| bond  | 27.81342124938965 | 1.3415189981460571 | 1.0894842147827148 | -26.47190225124359 |
| total | 47.563133239746094 | 11.476179122924805 | 5.197903633117676 | -36.08695411682129 |

## Decode smoke (n_samples=16 per checkpoint)
| step | decode_ratio | n_decoded | mean_atoms | disconnected | wall_s |
|---|---|---|---|---|---|
| 500 | 0.000 | 0 | 0.0 | 16 | 0.0 |
| 1000 | 0.000 | 0 | 0.0 | 16 | 0.0 |
| 1500 | 0.000 | 0 | 0.0 | 16 | 0.0 |
| 2000 | 0.000 | 0 | 0.0 | 16 | 0.0 |

**decode_ratio at step 200**: None
**decode_ratio at step 500**: 0.0
**decode_ratio at step 1000**: 0.0
**max decode_ratio observed**: 0.000
**first non-zero step**: None

## Verdict
**STILL 0 — coordinate-quality wall persists (bond head trains but decoder fails)**

References:
* Prior baseline (5000-step, h=32, single-fix): `wf_gpu_recovery_now/final.md` decode_ratio=0/192
* P0 fix inventory: `molmetal/reports/wf_cfm_p0_fixes/`
* P1 fix inventory: `molmetal/reports/wf_cfm_p1_fixes/`
* TODO/pending/24_cfm_architecture_redo_plan.md

JSON: `final.json`
Checkpoint: `logs/yuelbond_p1_full/seed_2/ckpt.pt`
