# Phase 3 Smoke Retrain — Stacked P0+P1 Fixes on 500-mol Metallodrug Pool

**Date**: 2026-09-16
**Script**: `molmetal/scripts/metallo_drug_smoke_retrain.py`
**Inputs**: `molmetal/data/metallo_drugs_500_train.csv` (449 SMILES, 358 3D-embedded, 91 skipped)

## Configuration

| Knob | Value |
|---|---|
| steps (planned) | 200 |
| steps (executed) | 200 |
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
| wall-clock | 98.1s (0.49s/step) |

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
| cfm   | 18.538455963134766 | 17.548206329345703 | 5.849001407623291 | -0.9902496337890625 |
| atom  | 2.995732307434082 | 2.8457460403442383 | 1.1223002672195435 | -0.14998626708984375 |
| bond  | 27.684694290161133 | 1.3583728075027466 | 1.177765130996704 | -26.326321482658386 |
| total | 46.52272415161133 | 19.19115447998047 | 7.515613555908203 | -27.33156967163086 |

## Decode smoke (n_samples=16 per checkpoint)
| step | decode_ratio | n_decoded | mean_atoms | disconnected | wall_s |
|---|---|---|---|---|---|
| 100 | 0.000 | 0 | 0.0 | 16 | 0.0 |
| 200 | 0.000 | 0 | 0.0 | 16 | 0.0 |

**decode_ratio at step 200**: 0.0
**decode_ratio at step 500**: None
**decode_ratio at step 1000**: None
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
Checkpoint: `logs/wf_todo33_p1_vocab_fix/seed_0/ckpt.pt`
