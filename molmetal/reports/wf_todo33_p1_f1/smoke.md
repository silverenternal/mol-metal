# TODO-33 P1 F1 — FP32 vs BF16 smoke (atom_loss=Inf, decode_ratio=0)

**Date**: 2026-09-25
**Hypothesis tested**: Blackwell BF16 autocast is the cause of `atom_loss=Inf` from step 1
and the downstream `decode_ratio=0.000` wall observed on the YuelBond smoke.

## Command

```bash
export MOLMETAL_CFM_AMP_DTYPE=float32
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs/yuelbond_p1_f1
rm -rf logs/yuelbond_p1_f1/seed_0
uv run --no-sync python molmetal/scripts/metallo_drug_smoke_retrain.py \
  --steps 2000 --n-train 500 --batch-size 4 \
  --hidden-dim 128 --n-layers 3 --bond-head learned --joint-train \
  --seed 0 \
  --out-dir logs/yuelbond_p1_f1/seed_0 \
  --decode-smoke-every 500
```

`MOLMETAL_CFM_AMP_DTYPE` is honoured at
`molmetal/adapters/flow_matching_lipman/__init__.py:88-95`
(`maybe_enable_amp()`); `"float32"` maps to `torch.float32`, bypassing BF16 autocast entirely.

## Result: FP32 does NOT fix it

| metric            | BF16 baseline (b=8)        | FP32 (b=4)                  |
|-------------------|----------------------------|-----------------------------|
| atom_loss first   | Inf                        | Inf                         |
| atom_loss last    | Inf                        | Inf                         |
| atom_loss min     | 1.219                      | 0.984                       |
| atom_loss max     | Inf                        | Inf                         |
| cfm_loss last     | 252330                     | 894                         |
| decode @ 500      | 0.000  (TypeError)         | 0.000  (TypeError)          |
| decode @ 1000     | 0.000  (TypeError)         | 0.000  (TypeError)          |
| decode @ 1500     | 0.000  (TypeError)         | 0.000  (TypeError)          |
| decode @ 2000     | 0.000  (TypeError)         | 0.000  (TypeError)          |
| max_decode_ratio  | 0.000                      | 0.000                       |
| wall-clock        | 104.3 s                    | 96.2 s                      |
| verdict           | "STILL 0 — coordinate-quality wall" | "STILL 0 — coordinate-quality wall" |

Every probe (BF16 baseline AND FP32) carries the same pre-existing decode error:
`TypeError: \`model\` must be a VelocityNet; got ModelWrapper.`

## Verdict

1. **FP32 is NOT the fix.** atom_loss is still Inf from step 1 under FP32; the spike
   pattern (mostly Inf with occasional finite bursts like 26.9 at step 600, 11.8 at
   step 900, 3.7 at step 1150, 10.2 at step 1400) is identical in shape to the BF16
   run. The Inf is structural, not a BF16 autocast artefact.
2. **decode_ratio=0.000 is a SEPARATE pre-existing bug.** All 4 BF16 probes AND all 4
   FP32 probes raise `TypeError: model must be a VelocityNet; got ModelWrapper` before
   the ODE solver ever integrates. The decode wall is independent of the AMP dtype and
   pre-dates this work.
3. **Root cause of the decode TypeError:** at
   `molmetal/adapters/flow_matching_lipman/__init__.py:2486-2488` the adapter constructs
   `wrapper = self._ModelWrapper(model=...)` and passes that wrapper to
   `self._ODESolver(velocity_model=wrapper)`. The shim at
   `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py:28-33`
   forwards `model=velocity_model` straight into
   `flow_matching.sampler.FlowMatchingSampler(model=...)`, which enforces
   `isinstance(model, VelocityNet)` at `flow_matching/sampler.py:79-82` and rejects the
   `ModelWrapper`. Fix is to unwrap (or have the ODESolver shim unwrap) `wrapper.model`
   before handing it to the upstream sampler.

## Next steps (do NOT execute in this WF — diagnosis only)

- F1 FP32: **fail** — no code change needed; rule out FP32 as a candidate fix.
- The atom_loss=Inf wall needs separate investigation (loss value explosion, not
  numerics). Suggest opening a follow-up TODO.
- The decode TypeError is a 1-line shim fix in
  `_reference/solver/__init__.py` (unwrap `velocity_model.model` when it is a
  `ModelWrapper`) and should be unblocked immediately because it gates every decode
  smoke probe from running.

## Artifacts

- Log: `logs/yuelbond_p1_f1/seed_0.log`
- Final: `logs/yuelbond_p1_f1/seed_0/final.{md,json}`
- Ckpt: `logs/yuelbond_p1_f1/seed_0/ckpt.pt`
- Baseline BF16: `logs/yuelbond_multiseed/seed_0.{log,final.{md,json},ckpt.pt}`