# TODO-33 P1 Full — F1 (FP32) + F2 (warmup) + F4 (clamp) 3-seed verification

**Date**: 2026-09-25
**Script**: `molmetal/scripts/metallo_drug_smoke_retrain.py`
**Inputs**: `molmetal/data/metallo_drugs_500_train.csv` (449 SMILES, 358 3D-embedded)
**Per-run config**: steps=2000, batch=4, h=128, n_layers=3, bond_head=learned, joint_train=True, lr_warmup_steps=200, MOLMETAL_CFM_AMP_DTYPE=float32

## Per-seed results

| seed | cfm_loss_min | decode@500 | decode@1000 | decode@1500 | decode@2000 | max_decode_ratio | first_nonzero_step | wall_s |
|---|---|---|---|---|---|---|---|---|
| 0 | 4.179 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | None | 94.4 |
| 1 | 4.084 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | None | 172.5 |
| 2 | 3.750 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | None | 172.6 |

## Seed-mean ± std

| metric | mean | std |
|---|---|---|
| cfm_loss_min | 4.005 | 0.184 |
| decode_ratio_500 | 0.000 | 0.000 |
| decode_ratio_1000 | 0.000 | 0.000 |
| decode_ratio_1500 | 0.000 | 0.000 |
| decode_ratio_2000 | 0.000 | 0.000 |

## BF16 baseline comparison

- BF16 baseline (`logs/yuelbond_multiseed/`, 3-seed avg): decode_ratio = 0.000 at all probes
- F1+F2+F4 seeds with **any** decode lift (>0): **0/3**
- BF16 floor of 0/192 lifted: **NO** (≥2/3 seeds required)

## Verdict

**FAIL** — at least one probe with decode_ratio > 0.5 in ≥2/3 seeds: **0/3**

## Honest caveats

1. **Decode probe error**: every probe in every seed failed with the same error:
   `TypeError: `model` must be a VelocityNet; got ModelWrapper.`
   This is a NEW regression compared to the BF16 baseline (which also failed decode_ratio=0 but with a different failure mode). The decode smoke probe never actually evaluates the trained model — it crashes before evaluation. So the 0/0/0 decode_ratio in this run is the **probe crash result**, not a true measure of F1+F2+F4 decoder quality.

2. **Loss curves are FINITE** (no Inf spikes) — the F4 clamp fix correctly stabilises the training loss. cfm_loss min is ~4.0 ± 0.18 across all 3 seeds, vs the BF16 baseline where min was 7.37 with frequent Inf spikes. The training is mathematically healthy.

3. **Wall-clock per seed**: 94s for seed=0 (sole), but seeds 1+2 ran in parallel and each finished in ~3 min due to GPU contention (vs ~94s when alone). Total wall for the 3-seed sweep: ~3 min.

4. **Atom and bond losses are also finite** (atom 0.99-1.39, bond 1.13-1.25). F4 has demonstrated the expected effect: it bounds the logit space so atom_loss = -log(softmax) cannot diverge to Inf.

5. **decode_ratio floor remains 0** but for a DIFFERENT reason than BF16: BF16 fails because the decoder produces disconnected coordinates (`disconnected=16/16`); the F1+F2+F4 probe fails BEFORE the decoder runs because the `isinstance(model, VelocityNet)` check rejects the `ModelWrapper`. The next workflow round should fix this isinstance regression (likely introduced by R17 path-b decoder rework wiring changes).

## Files

- Logs: `logs/yuelbond_p1_full/seed_{0,1,2}/{final.json,final.md,ckpt.pt}`
- Per-seed runner logs: `logs/yuelbond_p1_full/seed_{0,1,2}.log`
- BF16 baseline: `logs/yuelbond_multiseed/seed_{0,1,2}/final.json` (all decode_ratio=0)
